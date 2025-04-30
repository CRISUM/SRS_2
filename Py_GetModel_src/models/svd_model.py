#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Py_GetModel_src/models/svd_model.py - Matrix factorization model based on SVD
Author: YourName
Date: 2025-04-27
Description: Implements collaborative filtering using Singular Value Decomposition
"""

import numpy as np
import pandas as pd
from scipy.sparse.linalg import svds
import logging
import pickle
import os

from .base_model import BaseRecommenderModel

logger = logging.getLogger(__name__)


class SVDModel(BaseRecommenderModel):
    """Matrix factorization model using SVD"""

    def __init__(self, n_components=50, random_state=42):
        """Initialize SVD model

        Args:
            n_components (int): Number of latent factors
            random_state (int): Random seed for reproducibility
        """
        self.n_components = n_components
        self.random_state = random_state
        self.user_factors = None
        self.item_factors = None
        self.global_mean = None
        self.user_map = {}
        self.item_map = {}
        self.reverse_user_map = {}
        self.reverse_item_map = {}
        self.user_item_matrix = None

    def fit(self, data):
        """Train the model with provided data

        Args:
            data (DataFrame or dict): If DataFrame, must contain user_id, item_id, and rating columns
                                     If dict, must contain 'user_item_matrix' key
        Returns:
            self: Trained model
        """
        logger.info("Training SVD model...")

        if isinstance(data, pd.DataFrame):
            # Convert DataFrame to user-item matrix
            df_subset = data[['user_id', 'app_id']].copy()

            # 优先使用 rating_new 列
            if 'rating_new' in data.columns and pd.api.types.is_numeric_dtype(data['rating_new']):
                df_subset['rating_value'] = data['rating_new']
                rating_col = 'rating_value'
            elif 'rating' in data.columns and pd.api.types.is_numeric_dtype(data['rating']):
                df_subset['rating_value'] = data['rating']
                rating_col = 'rating_value'
            elif 'is_recommended' in data.columns:
                # Convert boolean to numeric
                df_subset['rating_value'] = data['is_recommended'].astype(int) * 10
                rating_col = 'rating_value'
            else:
                # Use hours as rating
                rating_col = 'hours'
                df_subset['rating_value'] = data['hours'].fillna(0)
                rating_col = 'rating_value'

            # Create user-item matrix
            self.user_item_matrix = pd.pivot_table(
                df_subset,
                values=rating_col,
                index='user_id',
                columns='app_id',
                aggfunc='mean',
                fill_value=0
            )
        elif isinstance(data, dict) and 'user_item_matrix' in data:
            self.user_item_matrix = data['user_item_matrix']
        else:
            raise ValueError(
                "Data must be a DataFrame with user_id, app_id, and rating columns or a dict with user_item_matrix")

        # Create user and item mappings
        self.user_map = {user: i for i, user in enumerate(self.user_item_matrix.index)}
        self.item_map = {item: i for i, item in enumerate(self.user_item_matrix.columns)}
        self.reverse_user_map = {i: user for user, i in self.user_map.items()}
        self.reverse_item_map = {i: item for item, i in self.item_map.items()}

        # Perform matrix factorization using SVD
        matrix = self.user_item_matrix.values
        self.global_mean = np.mean(matrix[matrix > 0]) if np.count_nonzero(matrix) > 0 else 0

        # Center the matrix (subtract mean from non-zero entries)
        centered_matrix = matrix.copy()
        centered_matrix[centered_matrix > 0] -= self.global_mean

        # Perform truncated SVD
        u, sigma, vt = svds(centered_matrix, k=min(self.n_components, min(matrix.shape) - 1))

        # Sort the singular values in descending order
        idx = np.argsort(sigma)[::-1]
        sigma = sigma[idx]
        u = u[:, idx]
        vt = vt[idx, :]

        # Store the decomposition results
        # User factors with sigma incorporated
        self.user_factors = u @ np.diag(np.sqrt(sigma))
        # Item factors with sigma incorporated
        self.item_factors = np.diag(np.sqrt(sigma)) @ vt

        logger.info(f"SVD model trained successfully with {self.n_components} factors")
        return self

    def predict(self, user_id, item_id):
        """Predict rating for a user-item pair

        Args:
            user_id: User ID
            item_id: Item ID

        Returns:
            float: Predicted rating (0-1 scale)
        """
        if self.user_factors is None or self.item_factors is None:
            logger.warning("Model not trained yet")
            return 0.5

        # Check if user and item exist in the model
        if user_id not in self.user_map or item_id not in self.item_map:
            # Return global mean for cold-start cases
            return 0.5

        user_idx = self.user_map[user_id]
        item_idx = self.item_map[item_id]

        # Compute predicted rating
        user_factor = self.user_factors[user_idx]
        item_factor = self.item_factors[:, item_idx]

        # Prediction = global mean + user-item interaction
        prediction = self.global_mean + np.dot(user_factor, item_factor)

        # Clip and normalize to 0-1 range
        prediction = max(0, min(10, prediction)) / 10

        return prediction

    def recommend(self, user_id, n=10):
        """Generate top-N recommendations for a user

        Args:
            user_id: User ID
            n (int): Number of recommendations

        Returns:
            list: List of (item_id, score) tuples
        """
        if self.user_factors is None or self.item_factors is None:
            logger.warning("Model not trained yet")
            return []

        # Check if user exists in the model
        if user_id not in self.user_map:
            logger.warning(f"User {user_id} not found in the model")
            return []

        user_idx = self.user_map[user_id]
        user_factor = self.user_factors[user_idx]

        # Get user's existing items to filter them out
        if self.user_item_matrix is not None:
            user_items = self.user_item_matrix.loc[user_id]
            existing_items = set(user_items[user_items > 0].index)
        else:
            existing_items = set()

        # Calculate scores for all items
        scores = []
        for item_id, item_idx in self.item_map.items():
            if item_id in existing_items:
                continue

            item_factor = self.item_factors[:, item_idx]
            score = self.global_mean + np.dot(user_factor, item_factor)
            scores.append((item_id, max(0, min(10, score)) / 10))

        # Sort by score in descending order
        scores.sort(key=lambda x: x[1], reverse=True)

        # Return top N
        return scores[:n]

    def update(self, new_data):
        """Update model with new data (incremental learning)

        Args:
            new_data (DataFrame): New interactions data

        Returns:
            self: Updated model
        """
        logger.info("Updating SVD model with new data...")

        if self.user_item_matrix is None:
            # If model isn't trained yet, just do a full training
            return self.fit(new_data)

        # Determine rating column
        if 'rating' in new_data.columns:
            rating_col = 'rating'
        elif 'is_recommended' in new_data.columns:
            # Convert boolean to numeric
            new_data['rating_value'] = new_data['is_recommended'].astype(int) * 10
            rating_col = 'rating_value'
        else:
            # Use hours as rating
            rating_col = 'hours'
            # Normalize hours to 0-10 scale for consistency
            max_hours = new_data['hours'].max()
            if max_hours > 0:
                new_data['rating_value'] = new_data['hours'] * 10 / max_hours
                rating_col = 'rating_value'

        # Process new users and items
        new_users = set(new_data['user_id']) - set(self.user_map.keys())
        new_items = set(new_data['app_id']) - set(self.item_map.keys())

        # If there are new users or items, we need to expand the matrix
        if new_users or new_items:
            # Create updated index and columns
            updated_index = list(self.user_item_matrix.index) + list(new_users)
            updated_columns = list(self.user_item_matrix.columns) + list(new_items)

            # Create expanded matrix
            expanded_matrix = pd.DataFrame(
                0,
                index=updated_index,
                columns=updated_columns
            )

            # Fill in existing values
            expanded_matrix.loc[
                self.user_item_matrix.index, self.user_item_matrix.columns] = self.user_item_matrix.values

            # Update user-item matrix
            self.user_item_matrix = expanded_matrix

            # Update mappings
            self.user_map = {user: i for i, user in enumerate(self.user_item_matrix.index)}
            self.item_map = {item: i for i, item in enumerate(self.user_item_matrix.columns)}
            self.reverse_user_map = {i: user for user, i in self.user_map.items()}
            self.reverse_item_map = {i: item for item, i in self.item_map.items()}

        # Update matrix with new ratings
        for _, row in new_data.iterrows():
            user_id = row['user_id']
            item_id = row['app_id']
            rating = row[rating_col]

            # Update rating in user-item matrix
            self.user_item_matrix.loc[user_id, item_id] = rating

        # Retrain the model
        matrix = self.user_item_matrix.values
        self.global_mean = np.mean(matrix[matrix > 0]) if np.count_nonzero(matrix) > 0 else 0

        # Center the matrix
        centered_matrix = matrix.copy()
        centered_matrix[centered_matrix > 0] -= self.global_mean

        # Perform truncated SVD
        u, sigma, vt = svds(centered_matrix, k=min(self.n_components, min(matrix.shape) - 1))

        # Sort the singular values in descending order
        idx = np.argsort(sigma)[::-1]
        sigma = sigma[idx]
        u = u[:, idx]
        vt = vt[idx, :]

        # Store the decomposition results
        self.user_factors = u @ np.diag(np.sqrt(sigma))
        self.item_factors = np.diag(np.sqrt(sigma)) @ vt

        logger.info(f"SVD model updated successfully")
        return self

    def save(self, path):
        """Save model to disk

        Args:
            path (str): Directory path

        Returns:
            bool: Success
        """
        logger.info(f"Saving SVD model to {path}")

        os.makedirs(path, exist_ok=True)

        model_data = {
            'n_components': self.n_components,
            'random_state': self.random_state,
            'user_factors': self.user_factors,
            'item_factors': self.item_factors,
            'global_mean': self.global_mean,
            'user_map': self.user_map,
            'item_map': self.item_map,
            'reverse_user_map': self.reverse_user_map,
            'reverse_item_map': self.reverse_item_map
        }

        try:
            with open(os.path.join(path, 'svd_model.pkl'), 'wb') as f:
                pickle.dump(model_data, f)

            # Save user-item matrix separately
            if self.user_item_matrix is not None:
                self.user_item_matrix.to_pickle(os.path.join(path, 'user_item_matrix.pkl'))

            logger.info("SVD model saved successfully")
            return True
        except Exception as e:
            logger.error(f"Error saving SVD model: {str(e)}")
            return False

    def load(self, path):
        """Load model from disk

        Args:
            path (str): Directory path

        Returns:
            self: Loaded model
        """
        logger.info(f"Loading SVD model from {path}")

        try:
            # Load model data
            with open(os.path.join(path, 'svd_model.pkl'), 'rb') as f:
                model_data = pickle.load(f)

            self.n_components = model_data['n_components']
            self.random_state = model_data['random_state']
            self.user_factors = model_data['user_factors']
            self.item_factors = model_data['item_factors']
            self.global_mean = model_data['global_mean']
            self.user_map = model_data['user_map']
            self.item_map = model_data['item_map']
            self.reverse_user_map = model_data['reverse_user_map']
            self.reverse_item_map = model_data['reverse_item_map']

            # Load user-item matrix if available
            matrix_path = os.path.join(path, 'user_item_matrix.pkl')
            if os.path.exists(matrix_path):
                self.user_item_matrix = pd.read_pickle(matrix_path)

            logger.info("SVD model loaded successfully")
            return self
        except Exception as e:
            logger.error(f"Error loading SVD model: {str(e)}")
            return None