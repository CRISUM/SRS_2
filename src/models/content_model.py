#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
src/models/content_model.py - Content-based recommendation model
Author: YourName
Date: 2025-04-27
Description: Implements content-based recommendation using item similarities
"""

import numpy as np
import logging
import pickle
import os

from .base_model import BaseRecommenderModel

logger = logging.getLogger(__name__)

class ContentBasedModel(BaseRecommenderModel):
    """Content-based recommendation model using item similarities"""

    def __init__(self, similarity_matrix=None):
        """Initialize content-based model

        Args:
            similarity_matrix (dict): Dictionary mapping item_id to list of (similar_item_id, score) tuples
        """
        self.similarity_matrix = similarity_matrix or {}
        self.user_preferences = {}
        self.popular_items = []

    def fit(self, data):
        """Train the model with provided data

        Args:
            data (dict): Dictionary containing:
                - similarity_matrix: Item similarity matrix
                - user_data: Optional user interaction data
                - item_data: Optional item metadata

        Returns:
            self: Trained model
        """
        logger.info("Training content-based model...")

        # Handle different input formats
        if isinstance(data, dict):
            if 'similarity_matrix' in data:
                self.similarity_matrix = data['similarity_matrix']
            elif 'embeddings' in data:
                # Create similarity matrix from embeddings
                from sklearn.metrics.pairwise import cosine_similarity
                import numpy as np

                embeddings = data['embeddings']
                items = list(embeddings.keys())

                # Create embedding matrix
                matrix = np.array([embeddings[item_id] for item_id in items])

                # Calculate similarity
                sim_matrix = cosine_similarity(matrix)

                # Convert to similarity dictionary
                self.similarity_matrix = {}
                for i, item_id in enumerate(items):
                    sims = [(items[j], sim_matrix[i, j]) for j in range(len(items)) if i != j]
                    sims.sort(key=lambda x: x[1], reverse=True)
                    self.similarity_matrix[item_id] = sims
        else:
            # Try to extract user preferences from DataFrame
            try:
                interactions_df = data
                # Group by user_id and calculate preferences
                for user_id in interactions_df['user_id'].unique():
                    user_data = interactions_df[interactions_df['user_id'] == user_id]

                    # Get user's positively rated items
                    if 'is_recommended' in user_data.columns:
                        positive_items = user_data[user_data['is_recommended'] == True]['app_id'].tolist()
                    elif 'rating' in user_data.columns:
                        # Assuming ratings above 7 are positive (on a 1-10 scale)
                        positive_items = user_data[user_data['rating'] >= 7]['app_id'].tolist()
                    else:
                        # Use top 25% by hours as positive
                        hours_threshold = user_data['hours'].quantile(0.75)
                        positive_items = user_data[user_data['hours'] >= hours_threshold]['app_id'].tolist()

                    self.user_preferences[user_id] = positive_items
            except Exception as e:
                logger.warning(f"Error extracting user preferences: {str(e)}")

        logger.info(f"Content-based model trained with {len(self.similarity_matrix)} items")
        return self

    def predict(self, user_id, item_id):
        """Predict rating for a user-item pair

        Args:
            user_id: User ID
            item_id: Item ID

        Returns:
            float: Predicted rating (0-1 scale)
        """
        # Check if item exists in similarity matrix
        if item_id not in self.similarity_matrix:
            return 0.5

        # Check if we have user preferences
        if user_id not in self.user_preferences or not self.user_preferences[user_id]:
            return 0.5

        # Get user's liked items
        liked_items = self.user_preferences[user_id]

        # Get similarity between target item and user's liked items
        similarities = []
        for liked_item in liked_items:
            # Find similarity between liked_item and item_id
            for sim_item, sim_score in self.similarity_matrix.get(liked_item, []):
                if sim_item == item_id:
                    similarities.append(sim_score)
                    break

        # If no similarities found, return default score
        if not similarities:
            return 0.5

        # Return average similarity
        return np.mean(similarities)

    def recommend(self, user_id, n=10):
        """Generate top-N recommendations for a user

        Args:
            user_id: User ID
            n (int): Number of recommendations

        Returns:
            list: List of (item_id, score) tuples
        """
        # Check if we have user preferences
        if user_id not in self.user_preferences or not self.user_preferences[user_id]:
            logger.warning(f"No preferences found for user {user_id}, returning popular items")
            return self.popular_items[:n]

        # Get user's liked items
        liked_items = self.user_preferences[user_id]

        # Get similar items to ones the user likes
        candidate_items = {}

        for liked_item in liked_items:
            if liked_item in self.similarity_matrix:
                for sim_item, sim_score in self.similarity_matrix[liked_item]:
                    # Skip items the user already likes
                    if sim_item in liked_items:
                        continue

                    # Update candidate score (take maximum similarity)
                    if sim_item not in candidate_items or sim_score > candidate_items[sim_item]:
                        candidate_items[sim_item] = sim_score

        # Sort candidates by score
        sorted_candidates = sorted(candidate_items.items(), key=lambda x: x[1], reverse=True)

        # Return top N
        return sorted_candidates[:n]

    def update(self, new_data):
        """Update model with new data (incremental learning)

        Args:
            new_data (dict or DataFrame): New data to update the model with

        Returns:
            self: Updated model
        """
        logger.info("Updating content-based model...")

        # Handle different input formats
        if isinstance(new_data, dict):
            # Update similarity matrix
            if 'similarity_matrix' in new_data:
                self.similarity_matrix.update(new_data['similarity_matrix'])

            # Update user preferences
            if 'user_preferences' in new_data:
                for user_id, prefs in new_data['user_preferences'].items():
                    if user_id in self.user_preferences:
                        # Add new preferences while avoiding duplicates
                        self.user_preferences[user_id] = list(set(self.user_preferences[user_id] + prefs))
                    else:
                        self.user_preferences[user_id] = prefs

            # Update popular items
            if 'popular_items' in new_data:
                self.popular_items = new_data['popular_items']
        else:
            # Try to extract user preferences from DataFrame
            try:
                interactions_df = new_data
                # Group by user_id and update preferences
                for user_id in interactions_df['user_id'].unique():
                    user_data = interactions_df[interactions_df['user_id'] == user_id]

                    # Get user's positively rated items
                    if 'is_recommended' in user_data.columns:
                        positive_items = user_data[user_data['is_recommended'] == True]['app_id'].tolist()
                    elif 'rating' in user_data.columns:
                        # Assuming ratings above 7 are positive (on a 1-10 scale)
                        positive_items = user_data[user_data['rating'] >= 7]['app_id'].tolist()
                    else:
                        # Use top 25% by hours as positive
                        hours_threshold = user_data['hours'].quantile(0.75)
                        positive_items = user_data[user_data['hours'] >= hours_threshold]['app_id'].tolist()

                    # Update existing preferences
                    if user_id in self.user_preferences:
                        self.user_preferences[user_id] = list(set(self.user_preferences[user_id] + positive_items))
                    else:
                        self.user_preferences[user_id] = positive_items
            except Exception as e:
                logger.warning(f"Error updating user preferences: {str(e)}")

        logger.info("Content-based model updated successfully")
        return self

    def save(self, path):
        """Save model to disk

        Args:
            path (str): Directory path

        Returns:
            bool: Success
        """
        logger.info(f"Saving content-based model to {path}")

        try:
            os.makedirs(path, exist_ok=True)

            # Save similarity matrix
            with open(os.path.join(path, 'content_similarity.pkl'), 'wb') as f:
                pickle.dump(self.similarity_matrix, f)

            # Save user preferences
            with open(os.path.join(path, 'user_preferences.pkl'), 'wb') as f:
                pickle.dump(self.user_preferences, f)

            # Save popular items
            with open(os.path.join(path, 'popular_items.pkl'), 'wb') as f:
                pickle.dump(self.popular_items, f)

            logger.info("Content-based model saved successfully")
            return True
        except Exception as e:
            logger.error(f"Error saving content-based model: {str(e)}")
            return False

    def load(self, path):
        """Load model from disk

        Args:
            path (str): Directory path

        Returns:
            self: Loaded model
        """
        logger.info(f"Loading content-based model from {path}")

        try:
            # Load similarity matrix
            with open(os.path.join(path, 'content_similarity.pkl'), 'rb') as f:
                self.similarity_matrix = pickle.load(f)

            # Load user preferences if available
            user_prefs_path = os.path.join(path, 'user_preferences.pkl')
            if os.path.exists(user_prefs_path):
                with open(user_prefs_path, 'rb') as f:
                    self.user_preferences = pickle.load(f)

            # Load popular items if available
            popular_items_path = os.path.join(path, 'popular_items.pkl')
            if os.path.exists(popular_items_path):
                with open(popular_items_path, 'rb') as f:
                    self.popular_items = pickle.load(f)

            logger.info("Content-based model loaded successfully")
            return self
        except Exception as e:
            logger.error(f"Error loading content-based model: {str(e)}")
            return None