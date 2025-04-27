#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
src/models/knn_model.py - KNN-based collaborative filtering models
Author: YourName
Date: 2025-04-27
Description: Implements user-based and item-based KNN collaborative filtering
"""

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
import logging
import pickle
import os

from .base_model import (BaseRecommenderModel)

logger = logging.getLogger(__name__)


class KNNModel(BaseRecommenderModel):
    """KNN-based collaborative filtering model"""

    def __init__(self, type='user', n_neighbors=20, metric='cosine', algorithm='brute'):
        """Initialize KNN model

        Args:
            type (str): 'user' for user-based CF, 'item' for item-based CF
            n_neighbors (int): Number of neighbors to consider
            metric (str): Distance metric
            algorithm (str): KNN algorithm
        """
        self.type = type
        self.n_neighbors = n_neighbors
        self.metric = metric
        self.algorithm = algorithm
        self.model = None
        self.user_indices = {}
        self.item_indices = {}
        self.reversed_user_indices = {}
        self.reversed_item_indices = {}
        self.user_item_matrix = None
        self.sparse_matrix = None

    def fit(self, data):
        """Train the model with provided data

        Args:
            data (DataFrame or dict): If DataFrame, must contain user_id, item_id, and rating columns
                                     If dict, must contain 'user_item_matrix' key
        Returns:
            self: Trained model
        """
        logger.info(f"Training {self.type}-based KNN model...")

        # Determine the input type and process accordingly
        if isinstance(data, pd.DataFrame):
            # Extract the necessary columns
            df_subset = data[['user_id', 'app_id']].copy()

            # 优先使用 rating_new 列
            if 'rating_new' in data.columns and pd.api.types.is_numeric_dtype(data['rating_new']):
                df_subset['rating_value'] = data['rating_new']
                rating_col = 'rating_value'
            elif 'rating' in data.columns and pd.api.types.is_numeric_dtype(data['rating']):
                df_subset['rating_value'] = data['rating']
                rating_col = 'rating_value'
            elif 'is_recommended' in data.columns:
                # Convert boolean to numeric value
                df_subset['rating_value'] = data['is_recommended'].astype(int) * 10
                rating_col = 'rating_value'
            else:
                # Use hours as interaction value
                df_subset['rating_value'] = data['hours'].fillna(0)
                rating_col = 'rating_value'

            # Create pivot table
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

        # Create index mappings
        self.user_indices = {user: i for i, user in enumerate(self.user_item_matrix.index)}
        self.item_indices = {item: i for i, item in enumerate(self.user_item_matrix.columns)}
        self.reversed_user_indices = {i: user for user, i in self.user_indices.items()}
        self.reversed_item_indices = {i: item for item, i in self.item_indices.items()}

        # Convert to sparse matrix for efficiency
        self.sparse_matrix = csr_matrix(self.user_item_matrix.values)

        # Initialize and train the model
        if self.type == 'user':
            # User-based KNN
            n_neighbors = min(self.n_neighbors, len(self.user_indices))
            self.model = NearestNeighbors(
                n_neighbors=n_neighbors,
                metric=self.metric,
                algorithm=self.algorithm,
                n_jobs=-1
            )
            self.model.fit(self.sparse_matrix)
        else:
            # Item-based KNN
            n_neighbors = min(self.n_neighbors, len(self.item_indices))
            self.model = NearestNeighbors(
                n_neighbors=n_neighbors,
                metric=self.metric,
                algorithm=self.algorithm,
                n_jobs=-1
            )
            self.model.fit(self.sparse_matrix.T)  # Transpose for item similarity

        logger.info(f"{self.type}-based KNN model trained successfully")
        return self

    def predict(self, user_id, item_id):
        """Predict rating for a user-item pair

        Args:
            user_id: User ID
            item_id: Item ID

        Returns:
            float: Predicted rating (0-1 scale)
        """
        if self.model is None:
            logger.warning("Model not trained yet")
            return 0.5

        # Check if user and item are in the model
        if user_id not in self.user_indices or item_id not in self.item_indices:
            return 0.5  # Default score for cold-start cases

        try:
            user_idx = self.user_indices[user_id]
            item_idx = self.item_indices[item_id]

            if self.type == 'user':
                # User-based prediction
                # Get user vector
                user_vector = self.sparse_matrix[user_idx].toarray().reshape(1, -1)

                # Find most similar users
                distances, indices = self.model.kneighbors(user_vector, n_neighbors=min(10, self.model.n_neighbors))

                # Calculate weighted score from similar users
                similar_users = indices[0]
                similarities = 1 - distances[0]  # Convert distance to similarity

                # Filter out the user itself
                if user_idx in similar_users:
                    user_idx_pos = np.where(similar_users == user_idx)[0][0]
                    similar_users = np.delete(similar_users, user_idx_pos)
                    similarities = np.delete(similarities, user_idx_pos)

                if len(similar_users) == 0:
                    return 0.5

                # Get similar users' ratings for the target item
                ratings = []
                for i, similar_user_idx in enumerate(similar_users):
                    rating = self.sparse_matrix[similar_user_idx, item_idx]
                    if rating > 0:  # Only consider non-zero ratings
                        ratings.append((rating, similarities[i]))

                if not ratings:
                    return 0.5

                # Weighted average of ratings
                weighted_sum = sum(rating * similarity for rating, similarity in ratings)
                similarity_sum = sum(similarity for _, similarity in ratings)

                if similarity_sum == 0:
                    return 0.5

                predicted_rating = weighted_sum / similarity_sum

                # Normalize to 0-1 range
                max_rating = 10  # Assuming ratings are on a 0-10 scale
                normalized_rating = predicted_rating / max_rating

                return normalized_rating

            else:
                # Item-based prediction
                # Get item vector
                item_vector = self.sparse_matrix[:, item_idx].T.toarray().reshape(1, -1)

                # Find most similar items
                distances, indices = self.model.kneighbors(item_vector, n_neighbors=min(10, self.model.n_neighbors))

                # Calculate weighted score from similar items
                similar_items = indices[0]
                similarities = 1 - distances[0]  # Convert distance to similarity

                # Filter out the item itself
                if item_idx in similar_items:
                    item_idx_pos = np.where(similar_items == item_idx)[0][0]
                    similar_items = np.delete(similar_items, item_idx_pos)
                    similarities = np.delete(similarities, item_idx_pos)

                if len(similar_items) == 0:
                    return 0.5

                # Get user's ratings for similar items
                ratings = []
                for i, similar_item_idx in enumerate(similar_items):
                    rating = self.sparse_matrix[user_idx, similar_item_idx]
                    if rating > 0:  # Only consider non-zero ratings
                        ratings.append((rating, similarities[i]))

                if not ratings:
                    return 0.5

                # Weighted average of ratings
                weighted_sum = sum(rating * similarity for rating, similarity in ratings)
                similarity_sum = sum(similarity for _, similarity in ratings)

                if similarity_sum == 0:
                    return 0.5

                predicted_rating = weighted_sum / similarity_sum

                # Normalize to 0-1 range
                max_rating = 10  # Assuming ratings are on a 0-10 scale
                normalized_rating = predicted_rating / max_rating

                return normalized_rating

        except Exception as e:
            logger.error(f"Error in KNN prediction: {str(e)}")
            return 0.5

    def recommend(self, user_id, n=10):
        """Generate top-N recommendations for a user

        Args:
            user_id: User ID
            n (int): Number of recommendations

        Returns:
            list: List of (item_id, score) tuples
        """
        if self.model is None:
            logger.warning("Model not trained yet")
            return []

        # Check if user exists in the model
        if user_id not in self.user_indices:
            logger.warning(f"User {user_id} not found in the model")
            return []

        try:
            user_idx = self.user_indices[user_id]

            # Get user's already seen items
            user_vector = self.sparse_matrix[user_idx].toarray().ravel()
            seen_indices = np.where(user_vector > 0)[0]
            seen_items = {self.reversed_item_indices[idx] for idx in seen_indices}

            if self.type == 'user':
                # User-based recommendation
                # Get user vector
                user_vector = self.sparse_matrix[user_idx].toarray().reshape(1, -1)

                # Find most similar users
                distances, indices = self.model.kneighbors(
                    user_vector,
                    n_neighbors=min(self.n_neighbors, self.sparse_matrix.shape[0] - 1)
                )

                # Calculate weighted score from similar users
                similar_users = indices[0]
                similarities = 1 - distances[0]  # Convert distance to similarity

                # Filter out the user itself
                if user_idx in similar_users:
                    user_idx_pos = np.where(similar_users == user_idx)[0][0]
                    similar_users = np.delete(similar_users, user_idx_pos)
                    similarities = np.delete(similarities, user_idx_pos)

                if len(similar_users) == 0:
                    return []

                # Calculate predicted scores for all items
                item_scores = {}

                for item_id, item_idx in self.item_indices.items():
                    # Skip already seen items
                    if item_id in seen_items:
                        continue

                    # Get similar users' ratings for this item
                    ratings = []
                    for i, similar_user_idx in enumerate(similar_users):
                        rating = self.sparse_matrix[similar_user_idx, item_idx]
                        if rating > 0:  # Only consider non-zero ratings
                            ratings.append((rating, similarities[i]))

                    if not ratings:
                        continue

                    # Weighted average of ratings
                    weighted_sum = sum(rating * similarity for rating, similarity in ratings)
                    similarity_sum = sum(similarity for _, similarity in ratings)

                    if similarity_sum == 0:
                        continue

                    predicted_rating = weighted_sum / similarity_sum

                    # Normalize and store
                    max_rating = 10  # Assuming ratings are on a 0-10 scale
                    item_scores[item_id] = predicted_rating / max_rating

            else:
                # Item-based recommendation
                # Calculate predicted scores for all items
                item_scores = {}

                for item_id, item_idx in self.item_indices.items():
                    # Skip already seen items
                    if item_id in seen_items:
                        continue

                    # Get item vector
                    item_vector = self.sparse_matrix[:, item_idx].T.toarray().reshape(1, -1)

                    # Find most similar items that the user has rated
                    distances, indices = self.model.kneighbors(
                        item_vector,
                        n_neighbors=min(self.n_neighbors, self.sparse_matrix.shape[1] - 1)
                    )

                    similar_items = indices[0]
                    similarities = 1 - distances[0]  # Convert distance to similarity

                    # Filter out the item itself
                    if item_idx in similar_items:
                        item_idx_pos = np.where(similar_items == item_idx)[0][0]
                        similar_items = np.delete(similar_items, item_idx_pos)
                        similarities = np.delete(similarities, item_idx_pos)

                    if len(similar_items) == 0:
                        continue

                    # Get user's ratings for similar items
                    ratings = []
                    for i, similar_item_idx in enumerate(similar_items):
                        rating = self.sparse_matrix[user_idx, similar_item_idx]
                        if rating > 0:  # Only consider non-zero ratings
                            ratings.append((rating, similarities[i]))

                    if not ratings:
                        continue

                    # Weighted average of ratings
                    weighted_sum = sum(rating * similarity for rating, similarity in ratings)
                    similarity_sum = sum(similarity for _, similarity in ratings)

                    if similarity_sum == 0:
                        continue

                    predicted_rating = weighted_sum / similarity_sum

                    # Normalize and store
                    max_rating = 10  # Assuming ratings are on a 0-10 scale
                    item_scores[item_id] = predicted_rating / max_rating

            # Sort by score in descending order
            sorted_items = sorted(item_scores.items(), key=lambda x: x[1], reverse=True)

            # Return top N
            return sorted_items[:n]

        except Exception as e:
            logger.error(f"Error in KNN recommendation: {str(e)}")
            return []

    def update(self, new_data):
        """Update model with new data (incremental learning)

        Args:
            new_data (DataFrame): New interactions data

        Returns:
            self: Updated model
        """
        logger.info(f"Updating {self.type}-based KNN model...")

        if self.user_item_matrix is None:
            # If model isn't trained yet, just do a full training
            return self.fit(new_data)

        # Determine rating column
        if 'rating' in new_data.columns:
            rating_col = 'rating'
        elif 'is_recommended' in new_data.columns:
            # Convert boolean to numeric value
            new_data['rating_value'] = new_data['is_recommended'].astype(int) * 10
            rating_col = 'rating_value'
        else:
            # Use hours as interaction value
            rating_col = 'hours'

        # Process new users and items
        new_users = set(new_data['user_id']) - set(self.user_indices.keys())
        new_items = set(new_data['app_id']) - set(self.item_indices.keys())

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
            self.user_indices = {user: i for i, user in enumerate(self.user_item_matrix.index)}
            self.item_indices = {item: i for i, item in enumerate(self.user_item_matrix.columns)}
            self.reversed_user_indices = {i: user for user, i in self.user_indices.items()}
            self.reversed_item_indices = {i: item for item, i in self.item_indices.items()}

        # Update matrix with new ratings
        for _, row in new_data.iterrows():
            user_id = row['user_id']
            item_id = row['app_id']
            rating = row[rating_col]

            # Update rating in user-item matrix
            self.user_item_matrix.loc[user_id, item_id] = rating

        # Convert to sparse matrix
        self.sparse_matrix = csr_matrix(self.user_item_matrix.values)

        # Retrain the model
        if self.type == 'user':
            # User-based KNN
            n_neighbors = min(self.n_neighbors, len(self.user_indices))
            self.model = NearestNeighbors(
                n_neighbors=n_neighbors,
                metric=self.metric,
                algorithm=self.algorithm,
                n_jobs=-1
            )
            self.model.fit(self.sparse_matrix)
        else:
            # Item-based KNN
            n_neighbors = min(self.n_neighbors, len(self.item_indices))
            self.model = NearestNeighbors(
                n_neighbors=n_neighbors,
                metric=self.metric,
                algorithm=self.algorithm,
                n_jobs=-1
            )
            self.model.fit(self.sparse_matrix.T)  # Transpose for item similarity

        logger.info(f"{self.type}-based KNN model updated successfully")
        return self

    def save(self, path):
        """Save model to disk

        Args:
            path (str): Directory path

        Returns:
            bool: Success
        """
        logger.info(f"Saving {self.type}-based KNN model to {path}")

        os.makedirs(path, exist_ok=True)

        try:
            # Save model data
            model_data = {
                'type': self.type,
                'n_neighbors': self.n_neighbors,
                'metric': self.metric,
                'algorithm': self.algorithm,
                'user_indices': self.user_indices,
                'item_indices': self.item_indices,
                'reversed_user_indices': self.reversed_user_indices,
                'reversed_item_indices': self.reversed_item_indices
            }

            with open(os.path.join(path, 'knn_model_metadata.pkl'), 'wb') as f:
                pickle.dump(model_data, f)

            # Save user-item matrix
            if self.user_item_matrix is not None:
                self.user_item_matrix.to_pickle(os.path.join(path, 'user_item_matrix.pkl'))

            # Save sparse matrix
            if self.sparse_matrix is not None:
                with open(os.path.join(path, 'sparse_matrix.pkl'), 'wb') as f:
                    pickle.dump(self.sparse_matrix, f)

            logger.info(f"{self.type}-based KNN model saved successfully")
            return True
        except Exception as e:
            logger.error(f"Error saving KNN model: {str(e)}")
            return False

    def load(self, path):
        """Load model from disk

        Args:
            path (str): Directory path

        Returns:
            self: Loaded model
        """
        logger.info(f"Loading KNN model from {path}")

        try:
            # Load model metadata
            with open(os.path.join(path, 'knn_model_metadata.pkl'), 'rb') as f:
                model_data = pickle.load(f)

            self.type = model_data['type']
            self.n_neighbors = model_data['n_neighbors']
            self.metric = model_data['metric']
            self.algorithm = model_data['algorithm']
            self.user_indices = model_data['user_indices']
            self.item_indices = model_data['item_indices']
            self.reversed_user_indices = model_data['reversed_user_indices']
            self.reversed_item_indices = model_data['reversed_item_indices']

            # Load user-item matrix
            user_item_matrix_path = os.path.join(path, 'user_item_matrix.pkl')
            if os.path.exists(user_item_matrix_path):
                self.user_item_matrix = pd.read_pickle(user_item_matrix_path)

            # Load sparse matrix
            sparse_matrix_path = os.path.join(path, 'sparse_matrix.pkl')
            if os.path.exists(sparse_matrix_path):
                with open(sparse_matrix_path, 'rb') as f:
                    self.sparse_matrix = pickle.load(f)

            # Recreate the model
            if self.sparse_matrix is not None:
                if self.type == 'user':
                    n_neighbors = min(self.n_neighbors, self.sparse_matrix.shape[0])
                    self.model = NearestNeighbors(
                        n_neighbors=n_neighbors,
                        metric=self.metric,
                        algorithm=self.algorithm,
                        n_jobs=-1
                    )
                    self.model.fit(self.sparse_matrix)
                else:
                    n_neighbors = min(self.n_neighbors, self.sparse_matrix.shape[1])
                    self.model = NearestNeighbors(
                        n_neighbors=n_neighbors,
                        metric=self.metric,
                        algorithm=self.algorithm,
                        n_jobs=-1
                    )
                    self.model.fit(self.sparse_matrix.T)

            logger.info("KNN model loaded successfully")
            return self
        except Exception as e:
            logger.error(f"Error loading KNN model: {str(e)}")
            return None