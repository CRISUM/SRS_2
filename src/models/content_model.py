# !/usr/bin/env python
# -*- coding: utf-8 -*-

"""
src/models/content_model.py - Enhanced content-based recommendation model
Author: YourName
Date: 2025-04-29
Description: Implements content-based recommendation using item similarities
             and TF-IDF for tag processing to improve sparse data performance
"""

import numpy as np
import logging
import pickle
import os
import traceback
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

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
        self.item_metadata = {}  # Store game metadata for better recommendations

    def fit(self, data):
        """Train the model with provided data

        Args:
            data (dict or DataFrame): Training data

        Returns:
            self: Trained model
        """
        logger.info("Training content-based model...")

        try:
            # Handle dict input format with pre-computed similarities
            if isinstance(data, dict):
                if 'similarity_matrix' in data:
                    self.similarity_matrix = data['similarity_matrix']
                elif 'embeddings' in data:
                    # Create similarity matrix from embeddings
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

                # Store user preferences if provided
                if 'user_preferences' in data:
                    self.user_preferences = data['user_preferences']

                # Store popular items if provided
                if 'popular_items' in data:
                    self.popular_items = data['popular_items']

                # Store item metadata if provided
                if 'item_metadata' in data:
                    self.item_metadata = data['item_metadata']

            # Process DataFrame input
            elif isinstance(data, pd.DataFrame):
                df = data

                # Extract game metadata
                game_metadata = {}
                for _, row in df.drop_duplicates('app_id').iterrows():
                    game_id = row['app_id']

                    # Build feature vector with available columns
                    features = {
                        'tags': row.get('tags', '').split(',') if isinstance(row.get('tags', ''), str) else [],
                        'title': row.get('title', f"Game {game_id}")
                    }

                    # Add optional features if available
                    for col in ['price_final', 'win', 'mac', 'linux', 'rating', 'positive_ratio', 'date_release']:
                        if col in row and not pd.isna(row[col]):
                            features[col] = row[col]

                    game_metadata[game_id] = features

                self.item_metadata = game_metadata

                # Compute game similarities using TF-IDF on tags
                game_tags = {}
                for game_id, metadata in game_metadata.items():
                    tags = metadata['tags']
                    game_tags[game_id] = ' '.join([tag.strip() for tag in tags]) if tags else ''

                # Use TF-IDF vectorizer
                vectorizer = TfidfVectorizer(min_df=1)
                all_game_ids = list(game_tags.keys())
                tag_texts = [game_tags[gid] for gid in all_game_ids]

                # Check if we have enough data
                if len(tag_texts) > 1 and any(tag_texts):
                    tag_matrix = vectorizer.fit_transform(tag_texts)

                    # Compute game similarities
                    tag_similarity = cosine_similarity(tag_matrix)

                    # Create game similarity dictionary
                    game_similarities = {}
                    for i, game_id in enumerate(all_game_ids):
                        similar_games = [(all_game_ids[j], tag_similarity[i, j])
                                         for j in range(len(all_game_ids)) if i != j]
                        similar_games.sort(key=lambda x: x[1], reverse=True)
                        game_similarities[game_id] = similar_games

                    self.similarity_matrix = game_similarities
                    logger.info(f"Created similarity matrix with {len(game_similarities)} games")
                else:
                    logger.warning("Not enough tag data to create similarity matrix")

                # Extract user preferences from interactions
                for user_id in df['user_id'].unique():
                    user_data = df[df['user_id'] == user_id]

                    # Get user's positively rated items
                    if 'is_recommended' in user_data.columns:
                        positive_items = user_data[user_data['is_recommended'] == True]['app_id'].tolist()
                    elif 'rating' in user_data.columns:
                        # Assuming ratings above 7 are positive (on a 1-10 scale)
                        positive_items = user_data[user_data['rating'] >= 7]['app_id'].tolist()
                    else:
                        try:
                            # Use top 25% by hours as positive
                            hours_threshold = user_data['hours'].quantile(0.75)
                            positive_items = user_data[user_data['hours'] >= hours_threshold]['app_id'].tolist()
                        except:
                            # Fallback to all items if hours calculation fails
                            positive_items = user_data['app_id'].tolist()

                    self.user_preferences[user_id] = positive_items

                # Calculate popular items for fallback
                if len(df) > 0:
                    # Count games by frequency
                    game_counts = df['app_id'].value_counts()

                    # Normalize by total users
                    total_users = df['user_id'].nunique()

                    # Convert to (game_id, score) format
                    popular_items = [(game_id, count / total_users) for game_id, count in game_counts.items()]

                    # Sort by score
                    popular_items.sort(key=lambda x: x[1], reverse=True)

                    # Keep top 100
                    self.popular_items = popular_items[:100]

            logger.info(f"Content-based model trained with {len(self.similarity_matrix)} items")
            return self

        except Exception as e:
            logger.error(f"Error training content-based model: {str(e)}")
            logger.error(traceback.format_exc())
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
            return 0.5  # Default score

        # Check if we have user preferences
        if user_id not in self.user_preferences or not self.user_preferences[user_id]:
            return 0.5  # Default score

        # Get user's liked items
        liked_items = self.user_preferences[user_id]

        # Get similarity between target item and user's liked items
        similarities = []
        for liked_item in liked_items:
            # Find similarity between liked_item and item_id
            if liked_item in self.similarity_matrix:
                for sim_item, sim_score in self.similarity_matrix[liked_item]:
                    if sim_item == item_id:
                        similarities.append(sim_score)
                        break

        # If no similarities found, return default score
        if not similarities:
            return 0.5

        # Return average similarity
        return min(0.95, np.mean(similarities))  # Cap at 0.95 to avoid over-confidence

    def update(self, new_data):
        """Update model with new data (incremental learning)

        Args:
            new_data (dict or DataFrame): New data to update the model with

        Returns:
            self: Updated model
        """
        logger.info("Updating content-based model...")

        try:
            # Handle different input formats
            if isinstance(new_data, dict):
                # Update similarity matrix
                if 'similarity_matrix' in new_data:
                    for item_id, sims in new_data['similarity_matrix'].items():
                        self.similarity_matrix[item_id] = sims

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

                # Update item metadata
                if 'item_metadata' in new_data:
                    self.item_metadata.update(new_data['item_metadata'])

            # Process DataFrame input for incremental update
            elif isinstance(new_data, pd.DataFrame):
                df = new_data

                # Extract new game metadata
                new_game_metadata = {}
                for _, row in df.drop_duplicates('app_id').iterrows():
                    game_id = row['app_id']

                    # Check if we already have this game
                    if game_id in self.item_metadata:
                        continue

                    # Build feature vector with available columns
                    features = {
                        'tags': row.get('tags', '').split(',') if isinstance(row.get('tags', ''), str) else [],
                        'title': row.get('title', f"Game {game_id}")
                    }

                    # Add optional features if available
                    for col in ['price_final', 'win', 'mac', 'linux', 'rating', 'positive_ratio', 'date_release']:
                        if col in row and not pd.isna(row[col]):
                            features[col] = row[col]

                    new_game_metadata[game_id] = features

                # If we have new games, update similarity matrix
                if new_game_metadata:
                    # Merge with existing metadata
                    all_metadata = {**self.item_metadata, **new_game_metadata}
                    self.item_metadata = all_metadata

                    # Recalculate similarities for all games
                    game_tags = {}
                    for game_id, metadata in all_metadata.items():
                        tags = metadata['tags']
                        game_tags[game_id] = ' '.join([tag.strip() for tag in tags]) if tags else ''

                    # Use TF-IDF vectorizer
                    vectorizer = TfidfVectorizer(min_df=1)
                    all_game_ids = list(game_tags.keys())
                    tag_texts = [game_tags[gid] for gid in all_game_ids]

                    # Check if we have enough data
                    if len(tag_texts) > 1 and any(tag_texts):
                        tag_matrix = vectorizer.fit_transform(tag_texts)

                        # Compute game similarities
                        tag_similarity = cosine_similarity(tag_matrix)

                        # Create game similarity dictionary
                        game_similarities = {}
                        for i, game_id in enumerate(all_game_ids):
                            similar_games = [(all_game_ids[j], tag_similarity[i, j])
                                             for j in range(len(all_game_ids)) if i != j]
                            similar_games.sort(key=lambda x: x[1], reverse=True)
                            game_similarities[game_id] = similar_games

                        self.similarity_matrix = game_similarities

                # Update user preferences from new interactions
                for user_id in df['user_id'].unique():
                    user_data = df[df['user_id'] == user_id]

                    # Get new positive interactions
                    if 'is_recommended' in user_data.columns:
                        positive_items = user_data[user_data['is_recommended'] == True]['app_id'].tolist()
                    elif 'rating' in user_data.columns:
                        positive_items = user_data[user_data['rating'] >= 7]['app_id'].tolist()
                    else:
                        try:
                            hours_threshold = user_data['hours'].quantile(0.75)
                            positive_items = user_data[user_data['hours'] >= hours_threshold]['app_id'].tolist()
                        except:
                            positive_items = user_data['app_id'].tolist()

                    # Update user preferences
                    if user_id in self.user_preferences:
                        self.user_preferences[user_id] = list(set(self.user_preferences[user_id] + positive_items))
                    else:
                        self.user_preferences[user_id] = positive_items

                # Update popular items
                if len(df) > 0:
                    # Get new game counts
                    new_game_counts = df['app_id'].value_counts()

                    # Create dictionary of existing popular items
                    popular_dict = dict(self.popular_items)

                    # Update with new data
                    total_users = df['user_id'].nunique()
                    for game_id, count in new_game_counts.items():
                        score = count / total_users
                        if game_id in popular_dict:
                            # Weighted update (70% new, 30% old)
                            popular_dict[game_id] = 0.3 * popular_dict[game_id] + 0.7 * score
                        else:
                            popular_dict[game_id] = score

                    # Convert back to sorted list
                    self.popular_items = sorted(popular_dict.items(), key=lambda x: x[1], reverse=True)[:100]

            logger.info("Content-based model updated successfully")
            return self

        except Exception as e:
            logger.error(f"Error updating content-based model: {str(e)}")
            logger.error(traceback.format_exc())
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

            # Save item metadata
            with open(os.path.join(path, 'item_metadata.pkl'), 'wb') as f:
                pickle.dump(self.item_metadata, f)

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

            # Load item metadata if available
            metadata_path = os.path.join(path, 'item_metadata.pkl')
            if os.path.exists(metadata_path):
                with open(metadata_path, 'rb') as f:
                    self.item_metadata = pickle.load(f)

            logger.info("Content-based model loaded successfully")
            return self
        except Exception as e:
            logger.error(f"Error loading content-based model: {str(e)}")
            return None

    def recommend(self, user_id, n=10):
        """Generate top-N recommendations for a user based on content similarity

        Args:
            user_id: User ID to generate recommendations for
            n (int): Number of recommendations to return

        Returns:
            list: List of (item_id, score) tuples ordered by recommendation score
        """
        # Get user's liked items
        liked_items = self.user_preferences.get(user_id, [])

        # Fallback to popular items if no preferences
        if not liked_items:
            return self.popular_items[:n]

        # Collect candidate items with aggregated similarity scores
        candidate_scores = {}

        # Aggregate similarity scores from all liked items
        for liked_item in liked_items:
            if liked_item not in self.similarity_matrix:
                continue

            # Get similar items and their scores
            for similar_item, score in self.similarity_matrix[liked_item]:
                # Skip items the user already liked
                if similar_item in liked_items:
                    continue

                # Sum similarity scores from all liked items
                candidate_scores[similar_item] = candidate_scores.get(similar_item, 0) + score

        # Sort candidates by score in descending order
        sorted_candidates = sorted(candidate_scores.items(), key=lambda x: x[1], reverse=True)

        # Take top-N items
        recommendations = sorted_candidates[:n]

        # Fill with popular items if needed (exclude duplicates)
        if len(recommendations) < n:
            recommended_ids = {item[0] for item in recommendations}
            popular_fallback = [
                                   (item[0], item[1]) for item in self.popular_items
                                   if item[0] not in recommended_ids
                               ][:n - len(recommendations)]
            recommendations += popular_fallback

        return recommendations
