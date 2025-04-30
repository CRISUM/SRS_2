#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Py_GetModel_src/models/popularity_model.py - Popularity-based recommendation model
Author: YourName
Date: 2025-05-01
Description: Implements a simple popularity-based recommendation model for sparse data scenarios
"""

import numpy as np
import pandas as pd
import logging
import pickle
import os
import traceback
from collections import defaultdict

from .base_model import BaseRecommenderModel

logger = logging.getLogger(__name__)


class PopularityModel(BaseRecommenderModel):
    """Popularity-based recommendation model for sparse data"""

    def __init__(self, time_decay_factor=0.9, diversity_factor=0.3):
        """Initialize popularity model

        Args:
            time_decay_factor (float): Factor for time decay in popularity calculation
            diversity_factor (float): Factor for diversity boost in recommendations
        """
        self.time_decay_factor = time_decay_factor
        self.diversity_factor = diversity_factor
        self.popular_items = []
        self.item_data = {}
        self.item_tags = defaultdict(set)
        self.item_category_scores = defaultdict(dict)
        self.category_items = defaultdict(list)

    def fit(self, data):
        """Train the model with provided data

        Args:
            data (DataFrame or dict): Training data

        Returns:
            self: Trained model
        """
        logger.info("Training popularity-based model...")

        try:
            # Handle different input formats
            if isinstance(data, dict) and 'df' in data:
                df = data['df']
            elif isinstance(data, pd.DataFrame):
                df = data
            else:
                logger.error("Invalid input data format")
                return self

            # Calculate popularity scores with advanced weighting
            item_scores = self._calculate_popularity_scores(df)

            # Extract item tags and metadata
            self._extract_item_metadata(df)

            # Group items by categories for diversity
            self._group_items_by_category()

            # Sort by score in descending order
            self.popular_items = sorted(item_scores.items(), key=lambda x: x[1], reverse=True)

            logger.info(f"Popularity model trained with {len(self.popular_items)} items")
            return self

        except Exception as e:
            logger.error(f"Error training popularity model: {str(e)}")
            logger.error(traceback.format_exc())
            return self

    def _calculate_popularity_scores(self, df):
        """Calculate weighted popularity scores for items

        Args:
            df (DataFrame): Data containing user-item interactions

        Returns:
            dict: Item ID to popularity score mapping
        """
        item_scores = {}

        # Get basic interaction counts
        item_counts = df['app_id'].value_counts()

        # Calculate positive ratings ratio if available
        positive_ratio = {}
        if 'is_recommended' in df.columns:
            for item_id in item_counts.index:
                item_data = df[df['app_id'] == item_id]
                if len(item_data) > 0:
                    positive_ratio[item_id] = item_data['is_recommended'].mean()

        # Calculate average playtime if available
        avg_playtime = {}
        if 'hours' in df.columns:
            for item_id in item_counts.index:
                item_data = df[df['app_id'] == item_id]
                if len(item_data) > 0:
                    avg_playtime[item_id] = item_data['hours'].mean()

        # Calculate recency if date column available
        recency_score = {}
        if 'date' in df.columns and pd.api.types.is_datetime64_dtype(df['date']):
            latest_date = df['date'].max()
            for item_id in item_counts.index:
                item_data = df[df['app_id'] == item_id]
                if len(item_data) > 0:
                    item_latest = item_data['date'].max()
                    days_old = (latest_date - item_latest).days
                    recency_score[item_id] = self.time_decay_factor ** (days_old / 30)  # 30-day decay

        # Compute final scores with weighted components
        total_users = df['user_id'].nunique()
        for item_id, count in item_counts.items():
            # Base popularity (normalized by user count)
            base_score = count / total_users

            # Positive rating boost
            rating_boost = positive_ratio.get(item_id, 0.5) * 0.3

            # Playtime boost (normalized)
            playtime = avg_playtime.get(item_id, 0)
            max_playtime = max(avg_playtime.values()) if avg_playtime else 1
            playtime_boost = (playtime / max_playtime) * 0.2 if max_playtime > 0 else 0

            # Recency boost
            recency_boost = recency_score.get(item_id, 0.8) * 0.1

            # Calculate final score
            final_score = (base_score * 0.4) + rating_boost + playtime_boost + recency_boost

            # Store score
            item_scores[item_id] = final_score

        return item_scores

    def _extract_item_metadata(self, df):
        """Extract item metadata for diversity calculations

        Args:
            df (DataFrame): Data containing item metadata
        """
        # Extract tags and categories
        for _, row in df.drop_duplicates('app_id').iterrows():
            item_id = row['app_id']

            # Store basic item data
            self.item_data[item_id] = {'id': item_id}

            # Extract title if available
            if 'title' in row and pd.notna(row['title']):
                self.item_data[item_id]['title'] = row['title']

            # Extract tags if available
            if 'tags' in row and pd.notna(row['tags']):
                tags = [tag.strip() for tag in str(row['tags']).split(',')]
                self.item_tags[item_id] = set(tags)

                # Also store in item data
                self.item_data[item_id]['tags'] = tags

    def _group_items_by_category(self):
        """Group items by category/tag for diversity"""
        # Identify common tags
        all_tags = []
        for tags in self.item_tags.values():
            all_tags.extend(tags)

        # Count tag frequencies
        tag_counts = defaultdict(int)
        for tag in all_tags:
            tag_counts[tag] += 1

        # Get top tags as categories (those appearing in at least 5 items)
        common_tags = [tag for tag, count in tag_counts.items() if count >= 5]

        # Group items by category and calculate within-category scores
        for tag in common_tags:
            # Find items with this tag
            category_items = []

            for item_id, tags in self.item_tags.items():
                if tag in tags:
                    # Find item score in popular items
                    score = 0
                    for pid, pscore in self.popular_items:
                        if pid == item_id:
                            score = pscore
                            break

                    category_items.append((item_id, score))

            # Sort by score and store
            category_items.sort(key=lambda x: x[1], reverse=True)
            self.category_items[tag] = category_items

            # Calculate within-category scores (normalized rank)
            for i, (item_id, _) in enumerate(category_items):
                normalized_rank = 1.0 - (i / len(category_items)) if len(category_items) > 1 else 1.0
                self.item_category_scores[item_id][tag] = normalized_rank

    def predict(self, user_id, item_id):
        """Predict rating for a user-item pair

        Args:
            user_id: User ID
            item_id: Item ID

        Returns:
            float: Predicted rating (0-1 scale)
        """
        # For popularity model, user_id doesn't matter
        # Find item in popular items
        for pid, score in self.popular_items:
            if pid == item_id:
                return min(0.95, score)  # Cap at 0.95

        return 0.1  # Default low score for unknown items

    def recommend(self, user_id, n=10):
        """Generate top-N recommendations for a user

        Args:
            user_id: User ID
            n (int): Number of recommendations

        Returns:
            list: List of (item_id, score) tuples
        """
        try:
            # For pure popularity model, we ignore user_id
            # But for better UX, we'll add diversity by category

            # If we have categories, create a diverse set
            if self.category_items:
                # Get top categories
                top_categories = list(self.category_items.keys())[:min(10, len(self.category_items))]

                # Select some items from each category
                selected_items = {}
                items_per_category = max(1, n // len(top_categories))

                for category in top_categories:
                    # Get top items in this category
                    category_top = self.category_items[category][:items_per_category]

                    for item_id, base_score in category_top:
                        if item_id not in selected_items:
                            # Calculate category diversity bonus
                            category_rank = self.item_category_scores[item_id].get(category, 0)
                            diversity_bonus = self.diversity_factor * category_rank

                            # Calculate adjusted score
                            adjusted_score = base_score * (1.0 + diversity_bonus)
                            selected_items[item_id] = adjusted_score

                # Add additional popular items if needed
                if len(selected_items) < n:
                    for item_id, score in self.popular_items:
                        if item_id not in selected_items:
                            selected_items[item_id] = score

                            if len(selected_items) >= n:
                                break

                # Convert to sorted list
                recommendations = sorted(selected_items.items(), key=lambda x: x[1], reverse=True)
                return recommendations[:n]
            else:
                # Fallback to pure popularity
                return self.popular_items[:n]

        except Exception as e:
            logger.error(f"Error generating popularity recommendations: {str(e)}")
            logger.error(traceback.format_exc())
            return self.popular_items[:n]

    def update(self, new_data):
        """Update model with new data (incremental learning)

        Args:
            new_data (DataFrame or dict): New data for updating

        Returns:
            self: Updated model
        """
        logger.info("Updating popularity model...")

        try:
            # Similar to fit, but with some weight to previous scores
            if isinstance(new_data, dict) and 'df' in new_data:
                df = new_data['df']
            elif isinstance(new_data, pd.DataFrame):
                df = new_data
            else:
                logger.error("Invalid input data format for update")
                return self

            # Calculate popularity scores for new data
            new_scores = self._calculate_popularity_scores(df)

            # Extract item metadata from new data
            self._extract_item_metadata(df)

            # Update existing scores (weighted average)
            existing_scores = dict(self.popular_items)

            for item_id, new_score in new_scores.items():
                if item_id in existing_scores:
                    # 30% old score, 70% new score
                    existing_scores[item_id] = (0.3 * existing_scores[item_id]) + (0.7 * new_score)
                else:
                    existing_scores[item_id] = new_score

            # Update popular items
            self.popular_items = sorted(existing_scores.items(), key=lambda x: x[1], reverse=True)

            # Regroup items by category
            self._group_items_by_category()

            logger.info(f"Popularity model updated with {len(new_scores)} items")
            return self

        except Exception as e:
            logger.error(f"Error updating popularity model: {str(e)}")
            logger.error(traceback.format_exc())
            return self

    def save(self, path):
        """Save model to disk

        Args:
            path (str): Directory path

        Returns:
            bool: Success
        """
        logger.info(f"Saving popularity model to {path}")

        try:
            os.makedirs(path, exist_ok=True)

            # Save model data
            model_data = {
                'popular_items': self.popular_items,
                'item_data': self.item_data,
                'item_tags': dict(self.item_tags),  # Convert defaultdict to dict for serialization
                'item_category_scores': dict(self.item_category_scores),
                'category_items': dict(self.category_items),
                'time_decay_factor': self.time_decay_factor,
                'diversity_factor': self.diversity_factor
            }

            with open(os.path.join(path, 'popularity_model.pkl'), 'wb') as f:
                pickle.dump(model_data, f)

            logger.info("Popularity model saved successfully")
            return True
        except Exception as e:
            logger.error(f"Error saving popularity model: {str(e)}")
            logger.error(traceback.format_exc())
            return False

    def load(self, path):
        """Load model from disk

        Args:
            path (str): Directory path

        Returns:
            self: Loaded model
        """
        logger.info(f"Loading popularity model from {path}")

        try:
            # Load model data
            with open(os.path.join(path, 'popularity_model.pkl'), 'rb') as f:
                model_data = pickle.load(f)

            self.popular_items = model_data['popular_items']
            self.item_data = model_data['item_data']
            self.item_tags = defaultdict(set, model_data['item_tags'])
            self.item_category_scores = defaultdict(dict, model_data['item_category_scores'])
            self.category_items = defaultdict(list, model_data['category_items'])
            self.time_decay_factor = model_data['time_decay_factor']
            self.diversity_factor = model_data['diversity_factor']

            logger.info("Popularity model loaded successfully")
            return self
        except Exception as e:
            logger.error(f"Error loading popularity model: {str(e)}")
            logger.error(traceback.format_exc())
            return None