#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
recommender_system.py - Main class for Steam Game Recommender System
Date: 2025-04-26
Description: Hybrid recommendation system for Steam games using KNN, SVD, and sequence models
"""

import pandas as pd
import numpy as np
import torch
import logging
import os
import pickle
import json
import gc
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD
from collections import defaultdict
import traceback

# Import local modules
from models import SVDModel, SimpleRecommenderModel, GameSequenceModel

logger = logging.getLogger(__name__)


class SteamRecommender:
    """Steam game recommendation system main class"""

    def __init__(self, data_path, config=None):
        """
        Initialize recommendation system

        Parameters:
            data_path (str): Data file path
            config (dict, optional): Configuration parameters
        """
        self.data_path = data_path

        # Default configuration
        self.config = {
            'sequence_params': {
                'embedding_dim': 64,
                'hidden_dim': 128,
                'num_layers': 2,
                'dropout': 0.2,
                'batch_size': 64,
                'learning_rate': 0.001,
                'epochs': 10
            },
            'knn_params': {
                'user_neighbors': 20,
                'item_neighbors': 20,
                'metric': 'cosine',
                'algorithm': 'brute',
            },
            'svd_params': {
                'n_components': 50,
                'random_state': 42,
            },
            'tag_embedding_dim': 50,
            'text_embedding_dim': 100,
            'max_seq_length': 20,
            'time_decay_factor': 0.9,
            'n_recommendations': 10,
            'user_knn_weight': 0.25,
            'item_knn_weight': 0.25,
            'svd_weight': 0.2,
            'content_weight': 0.15,
            'sequence_weight': 0.15,
            'use_gpu': torch.cuda.is_available()
        }

        # Update configuration if provided
        if config:
            self.config.update(config)

        # Initialize models and encoders
        self.user_knn_model = None
        self.item_knn_model = None
        self.simple_model = None
        self.sequence_model = None
        self.svd_model = None
        self.label_encoders = {}
        self.tfidf_model = None
        self.tfidf_svd = None
        self.tag_vectorizer = None
        self.game_embeddings = None
        self.user_embeddings = None
        self.device = torch.device('cuda' if self.config['use_gpu'] and torch.cuda.is_available() else 'cpu')

        # Initialize caches
        self.feature_cache = {}
        self.recommendation_cache = {}
        self.score_cache = {}

        # Initialize training tracking
        self.training_history = {}
        self.evaluation_results = None

        logger.info(f"Recommendation system initialized, using device: {self.device}")

    def load_data(self):
        """Load data in chunks"""
        logger.info(f"Loading data in chunks: {self.data_path}")

        # First read header to get column names
        try:
            header_df = pd.read_csv(self.data_path, nrows=1)
            column_names = header_df.columns.tolist()
            logger.info(f"Data columns: {column_names}")

            # Check if required columns exist
            required_cols = ['user_id', 'app_id', 'is_recommended', 'hours', 'title', 'tags']
            missing_cols = [col for col in required_cols if col not in column_names]
            if missing_cols:
                logger.error(f"CSV missing required columns: {missing_cols}")

                # Look for possible matching columns (case-insensitive)
                column_lower = [col.lower() for col in column_names]
                for missing in missing_cols:
                    possible_matches = [column_names[i] for i, col in enumerate(column_lower) if col == missing.lower()]
                    if possible_matches:
                        logger.info(f"'{missing}' possible matches: {possible_matches}")

                return False
        except Exception as e:
            logger.error(f"Error reading CSV header: {str(e)}")
            return False

        # Estimate data size
        chunk_size = 500000  # About 250-300MB per chunk

        # Initialize statistics tracking
        unique_users = set()
        unique_games = set()
        user_stats = {}
        game_stats = {}
        chunk_count = 0

        try:
            for chunk in pd.read_csv(self.data_path, chunksize=chunk_size):
                chunk_count += 1
                logger.info(f"Processing data chunk {chunk_count}")

                # Ensure necessary columns exist
                for col in required_cols:
                    if col not in chunk.columns:
                        logger.warning(f"Chunk {chunk_count} missing column '{col}', skipping")
                        continue

                # Update statistics
                unique_users.update(chunk['user_id'].unique())
                unique_games.update(chunk['app_id'].unique())

                # Process user statistics
                for _, row in chunk.iterrows():
                    user_id = row['user_id']
                    app_id = row['app_id']

                    # Handle possible missing values
                    try:
                        is_recommended = row['is_recommended']
                        hours = float(row['hours']) if not pd.isna(row['hours']) else 0.0
                    except (KeyError, ValueError):
                        is_recommended = False
                        hours = 0.0

                    # Update user stats
                    if user_id not in user_stats:
                        user_stats[user_id] = {'game_count': 0, 'total_hours': 0, 'recommended_count': 0}
                    user_stats[user_id]['game_count'] += 1
                    user_stats[user_id]['total_hours'] += hours
                    if is_recommended:
                        user_stats[user_id]['recommended_count'] += 1

                    # Update game stats
                    if app_id not in game_stats:
                        title = row.get('title', f"Unknown Game {app_id}")
                        tags = row.get('tags', "")

                        game_stats[app_id] = {
                            'user_count': 0,
                            'total_hours': 0,
                            'recommended_count': 0,
                            'title': title,
                            'tags': tags
                        }
                    game_stats[app_id]['user_count'] += 1
                    game_stats[app_id]['total_hours'] += hours
                    if is_recommended:
                        game_stats[app_id]['recommended_count'] += 1

                # Free memory
                del chunk
                gc.collect()

            logger.info(f"Processed {chunk_count} data chunks")

            # Convert statistics to DataFrames
            self.user_df = pd.DataFrame.from_dict(user_stats, orient='index')
            self.user_df.reset_index(inplace=True)
            self.user_df.rename(columns={'index': 'user_id'}, inplace=True)

            self.game_df = pd.DataFrame.from_dict(game_stats, orient='index')
            self.game_df.reset_index(inplace=True)
            self.game_df.rename(columns={'index': 'app_id'}, inplace=True)

            # Calculate additional statistics
            self.user_df['recommendation_ratio'] = self.user_df['recommended_count'] / self.user_df['game_count']
            self.game_df['recommendation_ratio'] = self.game_df['recommended_count'] / self.game_df['user_count']
            self.game_df['avg_hours'] = self.game_df['total_hours'] / self.game_df['user_count']

            # Sample data for training
            self._create_training_sample()

            logger.info(f"Data loading complete, found {len(unique_users)} unique users and {len(unique_games)} games")
            return True

        except Exception as e:
            logger.error(f"Error processing data: {str(e)}")
            logger.error(traceback.format_exc())
            return False

    def _create_training_sample(self):
        """Create training sample from large dataset"""
        logger.info("Creating training sample...")

        try:
            # Select active users and popular games
            if len(self.user_df) > 10000:
                active_users = self.user_df.sort_values('game_count', ascending=False).head(10000)['user_id'].values
            else:
                active_users = self.user_df['user_id'].values

            if len(self.game_df) > 5000:
                popular_games = self.game_df.sort_values('user_count', ascending=False).head(5000)['app_id'].values
            else:
                popular_games = self.game_df['app_id'].values

            # Read portion of original data for training
            sample_rows = []
            sample_size = min(1000000, len(self.user_df) * 10)  # Limit sample size

            rows_collected = 0
            chunk_size = 500000

            for chunk in pd.read_csv(self.data_path, chunksize=chunk_size):
                # Ensure required columns exist
                if not all(col in chunk.columns for col in ['user_id', 'app_id', 'is_recommended']):
                    continue

                # Filter interactions of active users and popular games
                filtered_chunk = chunk[
                    (chunk['user_id'].isin(active_users)) &
                    (chunk['app_id'].isin(popular_games))
                    ]

                if len(filtered_chunk) > 0:
                    sample_rows.append(filtered_chunk)
                    rows_collected += len(filtered_chunk)

                if rows_collected >= sample_size:
                    break

            if not sample_rows:
                logger.warning("No sample data collected!")
                return False

            # Merge samples
            self.df = pd.concat(sample_rows)

            # Split into training and testing sets
            from sklearn.model_selection import train_test_split
            self.train_df, self.test_df = train_test_split(
                self.df, test_size=0.2, random_state=42
            )

            logger.info(
                f"Created training sample with {len(self.df)} rows, train set: {len(self.train_df)}, test set: {len(self.test_df)}"
            )
            return True

        except Exception as e:
            logger.error(f"Error creating training sample: {str(e)}")
            logger.error(traceback.format_exc())
            return False

    def engineer_features(self):
        """Engineer features for recommendation models"""
        logger.info("Starting feature engineering...")

        if not hasattr(self, 'train_df') or self.train_df is None:
            logger.error("No training data available for feature engineering")
            return False

        try:
            # 1. Create user features
            user_features = self.train_df.groupby('user_id').agg({
                'app_id': 'count',
                'hours': ['sum', 'mean', 'max'],
                'is_recommended': ['mean', 'sum']
            })

            # Flatten column names
            user_features.columns = ['_'.join(col).strip() for col in user_features.columns.values]
            user_features.reset_index(inplace=True)

            # Rename columns for clarity
            user_features.rename(columns={
                'app_id_count': 'user_game_count',
                'is_recommended_mean': 'user_recommendation_ratio',
                'is_recommended_sum': 'user_recommended_count',
                'hours_sum': 'user_total_hours',
                'hours_mean': 'user_avg_hours',
                'hours_max': 'user_max_hours'
            }, inplace=True)

            # 2. Create game features
            game_features = self.train_df.groupby('app_id').agg({
                'user_id': 'count',
                'hours': ['sum', 'mean', 'max'],
                'is_recommended': ['mean', 'sum']
            })

            # Flatten column names
            game_features.columns = ['_'.join(col).strip() for col in game_features.columns.values]
            game_features.reset_index(inplace=True)

            # Rename columns for clarity
            game_features.rename(columns={
                'user_id_count': 'game_user_count',
                'is_recommended_mean': 'game_recommendation_ratio',
                'is_recommended_sum': 'game_recommended_count',
                'hours_sum': 'game_total_hours',
                'hours_mean': 'game_avg_hours',
                'hours_max': 'game_max_hours'
            }, inplace=True)

            # 3. Process game tags
            if 'tags' in self.train_df.columns:
                # Extract top tags per game
                top_tags = {}
                tag_counts = {}

                for app_id, tags_str in zip(self.train_df['app_id'], self.train_df['tags']):
                    if pd.notna(tags_str) and tags_str:
                        tags = [tag.strip() for tag in tags_str.split(',')]
                        top_tags[app_id] = tags[:3]  # Keep top 3 tags
                        for tag in tags:
                            tag_counts[tag] = tag_counts.get(tag, 0) + 1

                # Get most common tags
                common_tags = sorted(tag_counts.items(), key=lambda x: x[1], reverse=True)[:50]
                common_tags = [tag for tag, _ in common_tags]

                # Create tag features for games
                for i, tag in enumerate(common_tags[:10]):
                    tag_col = f'top_tag_{i + 1}'
                    game_features[tag_col] = game_features['app_id'].apply(
                        lambda app_id: 1 if app_id in top_tags and tag in top_tags[app_id] else 0
                    )

            # 4. Create interaction features
            # Merge user and game features into training data
            self.train_df = self.train_df.merge(user_features, on='user_id', how='left')
            self.train_df = self.train_df.merge(game_features, on='app_id', how='left')

            # Do the same for test data
            self.test_df = self.test_df.merge(user_features, on='user_id', how='left')
            self.test_df = self.test_df.merge(game_features, on='app_id', how='left')

            # 5. Create user preference match features
            # Calculate preference match between user history and game tags
            if 'tags' in self.train_df.columns:
                # Function to extract user's preferred tags from history
                def get_user_preferred_tags(user_id):
                    user_games = self.train_df[self.train_df['user_id'] == user_id]
                    user_tags = []
                    for tags_str in user_games['tags'].dropna():
                        if tags_str:
                            user_tags.extend([tag.strip() for tag in tags_str.split(',')])

                    # Return most frequent tags
                    tag_counter = {}
                    for tag in user_tags:
                        tag_counter[tag] = tag_counter.get(tag, 0) + 1

                    # Sort by frequency
                    sorted_tags = sorted(tag_counter.items(), key=lambda x: x[1], reverse=True)
                    return [tag for tag, _ in sorted_tags[:10]]

                # Create user tag preferences
                user_tag_prefs = {}
                for user_id in self.train_df['user_id'].unique():
                    user_tag_prefs[user_id] = get_user_preferred_tags(user_id)

                # Calculate tag match ratio for each user-game pair
                def calc_tag_match(row):
                    user_id, tags_str = row['user_id'], row['tags']
                    if user_id not in user_tag_prefs or pd.isna(tags_str) or not tags_str:
                        return 0.0

                    game_tags = [tag.strip() for tag in tags_str.split(',')]
                    user_tags = user_tag_prefs[user_id]

                    matches = len(set(game_tags) & set(user_tags))
                    total = len(set(game_tags) | set(user_tags)) if len(set(game_tags) | set(user_tags)) > 0 else 1

                    return matches / total

                # Add tag match feature
                self.train_df['tag_match'] = self.train_df.apply(calc_tag_match, axis=1)
                self.test_df['tag_match'] = self.test_df.apply(calc_tag_match, axis=1)

            # 6. Create sequence features
            self.create_sequence_features()

            logger.info("Feature engineering completed")
            return True

        except Exception as e:
            logger.error(f"Error in feature engineering: {str(e)}")
            logger.error(traceback.format_exc())
            return False

    def create_sequence_features(self):
        """Create sequential features based on user history"""
        if not hasattr(self, 'train_df') or self.train_df is None:
            logger.error("No training data available for sequence features")
            return

        try:
            # Sort data by user and timestamp (if available)
            logger.info("Creating sequence features from user history...")

            if 'date' in self.train_df.columns:
                self.train_df = self.train_df.sort_values(['user_id', 'date'])
            else:
                self.train_df = self.train_df.sort_values('user_id')

            # Create user history sequences
            user_sequences = {}

            for user_id in self.train_df['user_id'].unique():
                user_data = self.train_df[self.train_df['user_id'] == user_id]

                # Create sequences of previous games, ratings, and hours
                prev_apps = []
                prev_ratings = []
                prev_hours = []

                for idx, row in user_data.iterrows():
                    # Store current interaction's sequence data
                    self.train_df.at[idx, 'prev_apps'] = prev_apps[-self.config['max_seq_length']:] if prev_apps else []
                    self.train_df.at[idx, 'prev_ratings'] = prev_ratings[
                                                            -self.config['max_seq_length']:] if prev_ratings else []
                    self.train_df.at[idx, 'prev_hours'] = prev_hours[
                                                          -self.config['max_seq_length']:] if prev_hours else []

                    # Update sequences for next interaction
                    prev_apps.append(row['app_id'])
                    prev_ratings.append(1 if row['is_recommended'] else 0)
                    prev_hours.append(row['hours'])

            # Create similar sequence features for test data
            if 'date' in self.test_df.columns:
                self.test_df = self.test_df.sort_values(['user_id', 'date'])
            else:
                self.test_df = self.test_df.sort_values('user_id')

            for user_id in self.test_df['user_id'].unique():
                # Get user history from training data
                user_train_data = self.train_df[self.train_df['user_id'] == user_id]

                if len(user_train_data) > 0:
                    # Extract sequences from training data
                    prev_apps = user_train_data['app_id'].tolist()
                    prev_ratings = [1 if r else 0 for r in user_train_data['is_recommended'].tolist()]
                    prev_hours = user_train_data['hours'].tolist()
                else:
                    prev_apps = []
                    prev_ratings = []
                    prev_hours = []

                # Add sequence data to each test interaction
                user_test_data = self.test_df[self.test_df['user_id'] == user_id]

                for idx, row in user_test_data.iterrows():
                    self.test_df.at[idx, 'prev_apps'] = prev_apps[-self.config['max_seq_length']:] if prev_apps else []
                    self.test_df.at[idx, 'prev_ratings'] = prev_ratings[
                                                           -self.config['max_seq_length']:] if prev_ratings else []
                    self.test_df.at[idx, 'prev_hours'] = prev_hours[
                                                         -self.config['max_seq_length']:] if prev_hours else []

                    # Update sequences
                    prev_apps.append(row['app_id'])
                    prev_ratings.append(1 if row['is_recommended'] else 0)
                    prev_hours.append(row['hours'])

            # Calculate sequence statistics features
            def calc_sequence_stats(row):
                prev_ratings = row['prev_ratings'] if isinstance(row['prev_ratings'], list) else []
                prev_hours = row['prev_hours'] if isinstance(row['prev_hours'], list) else []

                # Calculate decayed average rating and hours
                rating_sum, rating_weight_sum = 0, 0
                hours_sum, hours_weight_sum = 0, 0

                for i, (rating, hours) in enumerate(zip(prev_ratings, prev_hours)):
                    # Apply time decay - more recent interactions have higher weight
                    weight = self.config['time_decay_factor'] ** (len(prev_ratings) - i - 1)
                    rating_sum += rating * weight
                    rating_weight_sum += weight
                    hours_sum += hours * weight
                    hours_weight_sum += weight

                avg_prev_rating = rating_sum / rating_weight_sum if rating_weight_sum > 0 else 0
                avg_prev_hours = hours_sum / hours_weight_sum if hours_weight_sum > 0 else 0

                return pd.Series({
                    'prev_game_count': len(row['prev_apps']) if isinstance(row['prev_apps'], list) else 0,
                    'avg_prev_rating': avg_prev_rating,
                    'avg_prev_hours': avg_prev_hours
                })

            # Add sequence statistics to dataframes
            seq_stats_train = self.train_df.apply(calc_sequence_stats, axis=1)
            self.train_df = pd.concat([self.train_df, seq_stats_train], axis=1)

            seq_stats_test = self.test_df.apply(calc_sequence_stats, axis=1)
            self.test_df = pd.concat([self.test_df, seq_stats_test], axis=1)

            # Store sequence feature columns for model training
            self.sequence_feature_columns = [
                'prev_game_count', 'avg_prev_rating', 'avg_prev_hours',
                'user_game_count', 'user_recommendation_ratio', 'user_total_hours'
            ]

            logger.info("Sequence features created successfully")

        except Exception as e:
            logger.error(f"Error creating sequence features: {str(e)}")
            logger.error(traceback.format_exc())

    def train_knn_model(self):
        """Train KNN models for collaborative filtering"""
        logger.info("Training KNN models...")

        # Check if we have sufficient data
        if not hasattr(self, 'train_df') or self.train_df is None or len(self.train_df) == 0:
            logger.error("Insufficient training data for KNN models")
            return None

        # Create user-game interaction matrix
        # Use rating if available, otherwise use is_recommended or hours
        if 'rating' in self.train_df.columns:
            rating_col = 'rating'
        elif 'is_recommended' in self.train_df.columns:
            # Convert boolean to numeric value
            self.train_df['rating_value'] = self.train_df['is_recommended'].astype(int) * 10
            rating_col = 'rating_value'
        else:
            # Use hours as interaction value
            rating_col = 'hours'

        # Create pivot table
        user_game_matrix = pd.pivot_table(
            self.train_df,
            values=rating_col,
            index='user_id',
            columns='app_id',
            aggfunc='mean',
            fill_value=0
        )

        logger.info(f"Created user-game matrix of shape: {user_game_matrix.shape}")

        # Store user and game ID mappings for recommendations
        self.user_indices = {user: i for i, user in enumerate(user_game_matrix.index)}
        self.app_indices = {app: i for i, app in enumerate(user_game_matrix.columns)}
        self.reversed_user_indices = {i: user for user, i in self.user_indices.items()}
        self.reversed_app_indices = {i: app for app, i in self.app_indices.items()}

        # Convert to sparse matrix for efficiency
        user_game_sparse = csr_matrix(user_game_matrix.values)

        # Get KNN parameters from config
        n_neighbors = min(self.config['knn_params'].get('user_neighbors', 20), len(self.user_indices))
        metric = self.config['knn_params'].get('metric', 'cosine')
        algorithm = self.config['knn_params'].get('algorithm', 'brute')

        # Train user-based KNN model
        self.user_knn_model = NearestNeighbors(
            n_neighbors=n_neighbors,
            metric=metric,
            algorithm=algorithm,
            n_jobs=-1
        )
        self.user_knn_model.fit(user_game_sparse)

        # Train item-based KNN model
        item_neighbors = min(self.config['knn_params'].get('item_neighbors', 20), len(self.app_indices))
        self.item_knn_model = NearestNeighbors(
            n_neighbors=item_neighbors,
            metric=metric,
            algorithm=algorithm,
            n_jobs=-1
        )
        self.item_knn_model.fit(user_game_sparse.T)  # Transpose for item similarity

        # Store original matrix for predictions
        self.user_game_matrix = user_game_matrix
        self.user_game_sparse_matrix = user_game_sparse

        # Track training
        self.training_history['knn_trained'] = True
        logger.info("KNN models training completed")
        return (self.user_knn_model, self.item_knn_model)

    def train_svd_model(self):
        """Train SVD model for collaborative filtering"""
        logger.info("Training SVD model for collaborative filtering...")

        # Check if training data is available
        if not hasattr(self, 'train_df') or self.train_df is None:
            logger.error("No training data available")
            return None

        # Determine rating column
        if 'rating' in self.train_df.columns:
            rating_col = 'rating'
        elif 'is_recommended' in self.train_df.columns:
            # Convert boolean to numeric
            self.train_df['rating_value'] = self.train_df['is_recommended'].astype(int) * 10
            rating_col = 'rating_value'
        else:
            # Use normalized hours as rating
            rating_col = 'hours'
            # Normalize hours to 0-10 scale for consistency
            max_hours = self.train_df['hours'].max()
            if max_hours > 0:
                self.train_df['rating_value'] = self.train_df['hours'] * 10 / max_hours
                rating_col = 'rating_value'

        # Create user-item matrix
        user_item_matrix = pd.pivot_table(
            self.train_df,
            values=rating_col,
            index='user_id',
            columns='app_id',
            aggfunc='mean',
            fill_value=0
        )

        # Get number of components from config
        n_components = min(
            self.config['svd_params'].get('n_components', 50),
            min(user_item_matrix.shape) - 1  # Must be smaller than matrix dimensions
        )

        # Train SVD model
        self.svd_model = SVDModel(n_components=n_components)
        self.svd_model.fit(user_item_matrix)

        # Track training
        self.training_history['svd_trained'] = True
        logger.info("SVD model training completed")
        return self.svd_model

    def train_simple_model(self):
        """Train a simple classification model for recommendations"""
        logger.info("Training simple recommendation model...")

        # Check if training data is available
        if not hasattr(self, 'train_df') or self.train_df is None:
            logger.error("No training data available")
            return None

        # Prepare features and target
        target_col = 'is_recommended'
        id_cols = ['user_id', 'app_id', 'date', 'review_id', 'prev_apps', 'prev_ratings', 'prev_hours']
        categorical_cols = [col for col in self.train_df.columns if col.endswith('_encoded')]

        # Remove non-feature columns
        exclude_cols = id_cols + [target_col, 'tags', 'description']

        # Select numerical and boolean features
        feature_cols = []
        for col in self.train_df.columns:
            if col in exclude_cols or (hasattr(self.train_df[col].iloc[0], '__iter__') and
                                       not isinstance(self.train_df[col].iloc[0], (str, bytes))):
                continue
            if self.train_df[col].dtype in [np.int64, np.float64, np.bool_]:
                feature_cols.append(col)

        # Filter out potential leakage features
        leakage_features = [
            'rating_new',
            'recommended_count_x', 'recommended_count_y',
            'recommendation_ratio_x', 'recommendation_ratio_y',
            'is_recommended_value', 'is_recommended_sum'
        ]

        safe_feature_cols = [col for col in feature_cols if col not in leakage_features]
        logger.info(f"Using {len(safe_feature_cols)} features for simple model")

        # Prepare data
        X_train = self.train_df[safe_feature_cols]
        y_train = self.train_df[target_col].astype(int)

        # Train the model
        self.simple_model = SimpleRecommenderModel(classifier='logistic')
        self.simple_model.fit(X_train, y_train)

        # Store feature importance
        self.feature_importance = pd.DataFrame({
            'Feature': safe_feature_cols,
            'Importance': self.simple_model.feature_importance()
        }).sort_values(by='Importance', ascending=False)

        # Track training
        self.training_history['simple_model_trained'] = True
        logger.info("Simple recommendation model training completed")
        return self.simple_model

    def train_sequence_model(self):
        """Train sequence model using user history features"""
        logger.info("Training sequence model...")

        if not hasattr(self, 'train_df') or self.train_df is None or not hasattr(self, 'sequence_feature_columns'):
            logger.error("Missing training data or sequence feature columns")
            return None

        try:
            # Prepare sequence features for training
            X = self.train_df[self.sequence_feature_columns].values
            y = self.train_df['is_recommended'].astype(float).values

            # Initialize model
            self.sequence_model = GameSequenceModel(
                num_features=len(self.sequence_feature_columns),
                hidden_dim=self.config['sequence_params'].get('hidden_dim', 128),
                num_layers=self.config['sequence_params'].get('num_layers', 2),
                dropout=self.config['sequence_params'].get('dropout', 0.2)
            ).to(self.device)

            # Training parameters
            batch_size = min(self.config['sequence_params'].get('batch_size', 64), len(X))
            learning_rate = self.config['sequence_params'].get('learning_rate', 0.001)
            epochs = self.config['sequence_params'].get('epochs', 10)

            # Setup optimizer and loss function
            optimizer = torch.optim.Adam(self.sequence_model.parameters(), lr=learning_rate)
            criterion = torch.nn.BCELoss()

            # Training history
            history = {
                'loss': []
            }

            # Training loop
            self.sequence_model.train()
            for epoch in range(epochs):
                # Shuffle data
                indices = np.random.permutation(len(X))
                X_shuffled = X[indices]
                y_shuffled = y[indices]

                # Mini-batch training
                total_loss = 0
                num_batches = 0

                for i in range(0, len(X), batch_size):
                    # Get batch
                    X_batch = X_shuffled[i:i + batch_size]
                    y_batch = y_shuffled[i:i + batch_size]

                    # Convert to tensors
                    X_tensor = torch.FloatTensor(X_batch).to(self.device)
                    y_tensor = torch.FloatTensor(y_batch).to(self.device)

                    # Forward pass
                    optimizer.zero_grad()
                    outputs = self.sequence_model(X_tensor)

                    # Calculate loss
                    loss = criterion(outputs, y_tensor)
                    total_loss += loss.item()

                    # Backward pass and optimize
                    loss.backward()
                    optimizer.step()

                    num_batches += 1

                # Calculate average loss for epoch
                avg_loss = total_loss / num_batches
                history['loss'].append(avg_loss)
                logger.info(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}")

            # Save training history
            self.training_history['sequence_loss'] = history['loss']
            self.training_history['sequence_trained'] = True

            # Set model to evaluation mode
            self.sequence_model.eval()
            logger.info("Sequence model training completed")
            return self.sequence_model

        except Exception as e:
            logger.error(f"Error training sequence model: {str(e)}")
            logger.error(traceback.format_exc())
            return None

    def create_game_embeddings(self):
        """Create game embeddings from content and interaction data"""
        logger.info("Creating game embeddings...")

        if not hasattr(self, 'train_df') or self.train_df is None:
            logger.error("No training data available")
            return None

        try:
            # Extract tag features if available
            if 'tags' in self.train_df.columns:
                # Create tag-based embeddings
                from sklearn.feature_extraction.text import TfidfVectorizer

                # Group by app_id to get unique games
                games_df = self.train_df.drop_duplicates('app_id')[['app_id', 'tags']]

                # Prepare tag text
                games_df['tag_text'] = games_df['tags'].fillna('').apply(
                    lambda x: ' '.join([tag.strip() for tag in x.split(',')]) if isinstance(x, str) else ''
                )

                # Apply TF-IDF to tags
                self.tag_vectorizer = TfidfVectorizer(max_features=100)
                tag_features = self.tag_vectorizer.fit_transform(games_df['tag_text'])

                # Create tag embeddings
                tag_embedding_dim = min(self.config['tag_embedding_dim'], tag_features.shape[1])
                svd = TruncatedSVD(n_components=tag_embedding_dim, random_state=42)
                tag_embeddings = svd.fit_transform(tag_features)

                # Create embedding dictionary
                self.game_embeddings = {}
                for i, app_id in enumerate(games_df['app_id']):
                    self.game_embeddings[app_id] = tag_embeddings[i]

                logger.info(f"Created tag-based embeddings for {len(self.game_embeddings)} games")

            # Enhance embeddings with collaborative information if SVD model is available
            if hasattr(self, 'svd_model') and self.svd_model is not None:
                # Get game factors from SVD model
                item_factors = self.svd_model.item_factors

                # Create or enhance game embeddings
                if not hasattr(self, 'game_embeddings') or self.game_embeddings is None:
                    self.game_embeddings = {}

                for app_id, idx in self.svd_model.item_map.items():
                    # Get SVD factors for this game
                    svd_embedding = item_factors[idx]

                    # If game already has tag-based embedding, concatenate with SVD factors
                    if app_id in self.game_embeddings:
                        tag_embedding = self.game_embeddings[app_id]
                        # Normalize both embeddings
                        tag_norm = np.linalg.norm(tag_embedding)
                        svd_norm = np.linalg.norm(svd_embedding)

                        if tag_norm > 0 and svd_norm > 0:
                            tag_embedding = tag_embedding / tag_norm
                            svd_embedding = svd_embedding / svd_norm

                        # Concatenate and store
                        self.game_embeddings[app_id] = np.concatenate([tag_embedding, svd_embedding])
                    else:
                        # Store SVD factors directly
                        self.game_embeddings[app_id] = svd_embedding

                logger.info(f"Enhanced embeddings with collaborative information")

            # Store number of games with embeddings
            self.training_history['games_with_embeddings'] = len(self.game_embeddings) if hasattr(self,
                                                                                                  'game_embeddings') else 0
            return self.game_embeddings

        except Exception as e:
            logger.error(f"Error creating game embeddings: {str(e)}")
            logger.error(traceback.format_exc())
            return None

    def train_content_model(self):
        """Train content-based model for cold-start recommendations"""
        logger.info("Training content-based model...")

        # Check if we have tag data
        if not any(col.startswith('top_tag_') for col in self.train_df.columns):
            logger.warning("No tag data found, skipping content-based model training")
            return None

        # Extract numerical tag features
        tag_cols = []
        for col in self.train_df.columns:
            if col in self.test_df.columns and not col.startswith(('user_id', 'app_id', 'prev_')):
                # Only include numerical features
                if self.train_df[col].dtype in [np.int64, np.float64, np.bool_]:
                    tag_cols.append(col)

        logger.info(f"Using {len(tag_cols)} numerical features for content model")

        if len(tag_cols) == 0:
            logger.warning("No suitable numerical features found, skipping content model")
            return None

        # Aggregate features by game
        game_features = self.train_df.groupby('app_id')[tag_cols].mean().reset_index()

        # Calculate cosine similarity
        game_matrix = csr_matrix(game_features[tag_cols].values)
        similarity_matrix = cosine_similarity(game_matrix)

        # Create game index mapping
        game_idx = {game_id: idx for idx, game_id in enumerate(game_features['app_id'])}

        # Store similarity matrix
        self.content_similarity = {
            'matrix': similarity_matrix,
            'game_idx': game_idx
        }

        # Track training
        self.training_history['content_model_trained'] = True
        logger.info("Content-based model training completed")
        return self.content_similarity

    def predict_user_knn_score(self, user_id, game_id):
        """Predict score using user-based KNN"""
        if not hasattr(self, 'user_knn_model') or self.user_knn_model is None:
            return 0.5

        try:
            # Check if user and game are in training data
            if user_id not in self.user_indices or game_id not in self.app_indices:
                return 0.5

            # Get user and game indices
            user_idx = self.user_indices[user_id]
            app_idx = self.app_indices[game_id]

            # Get user vector
            user_vector = self.user_game_sparse_matrix[user_idx].toarray().reshape(1, -1)

            # Find most similar users
            distances, indices = self.user_knn_model.kneighbors(user_vector, n_neighbors=10)

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

            # Calculate similar users' ratings for target game
            ratings = []
            weights = []

            for i, similar_user_idx in enumerate(similar_users):
                similar_user_id = self.reversed_user_indices[similar_user_idx]
                # Find this user's rating for the game
                similar_user_data = self.train_df[
                    (self.train_df['user_id'] == similar_user_id) &
                    (self.train_df['app_id'] == game_id)
                    ]

                if len(similar_user_data) > 0:
                    # Get rating (prioritize rating column, then is_recommended, lastly hours)
                    if 'rating' in similar_user_data.columns:
                        rating = similar_user_data['rating'].iloc[0]
                    elif 'is_recommended' in similar_user_data.columns:
                        rating = 10 if similar_user_data['is_recommended'].iloc[0] else 0
                    else:
                        # Use normalized hours
                        hours = similar_user_data['hours'].iloc[0]
                        rating = min(10, hours / 10)  # Map hours to 0-10 range

                    ratings.append(rating)
                    weights.append(similarities[i])

            # If no similar users rated this game, return default score
            if len(ratings) == 0:
                return 0.5

            # Calculate weighted average rating
            weighted_rating = np.average(ratings, weights=weights)
            # Normalize to 0-1 range
            normalized_rating = weighted_rating / 10

            return normalized_rating

        except Exception as e:
            logger.error(f"Error in user KNN prediction: {str(e)}")
            return 0.5

    def predict_item_knn_score(self, user_id, game_id):
        """Predict score using item-based KNN"""
        if not hasattr(self, 'item_knn_model') or self.item_knn_model is None:
            return 0.5

        try:
            # Check if game is in training data
            if game_id not in self.app_indices:
                return 0.5

            # Get game index
            app_idx = self.app_indices[game_id]

            # Get game vector
            item_vector = self.user_game_sparse_matrix.T[app_idx].toarray().reshape(1, -1)

            # Find most similar games
            distances, indices = self.item_knn_model.kneighbors(item_vector, n_neighbors=10)

            # Calculate weighted score from similar games
            similar_items = indices[0]
            similarities = 1 - distances[0]  # Convert distance to similarity

            # Filter out the game itself
            if app_idx in similar_items:
                app_idx_pos = np.where(similar_items == app_idx)[0][0]
                similar_items = np.delete(similar_items, app_idx_pos)
                similarities = np.delete(similarities, app_idx_pos)

            if len(similar_items) == 0:
                return 0.5

            # Calculate user's ratings for similar games
            ratings = []
            weights = []

            for i, similar_item_idx in enumerate(similar_items):
                similar_app_id = self.reversed_app_indices[similar_item_idx]
                # Find user's rating for this game
                user_rating_data = self.train_df[
                    (self.train_df['user_id'] == user_id) &
                    (self.train_df['app_id'] == similar_app_id)
                    ]

                if len(user_rating_data) > 0:
                    # Get rating
                    if 'rating' in user_rating_data.columns:
                        rating = user_rating_data['rating'].iloc[0]
                    elif 'is_recommended' in user_rating_data.columns:
                        rating = 10 if user_rating_data['is_recommended'].iloc[0] else 0
                    else:
                        # Use normalized hours
                        hours = user_rating_data['hours'].iloc[0]
                        rating = min(10, hours / 10)  # Map hours to 0-10 range

                    ratings.append(rating)
                    weights.append(similarities[i])

            # If user hasn't rated any similar games, return default score
            if len(ratings) == 0:
                return 0.5

            # Calculate weighted average rating
            weighted_rating = np.average(ratings, weights=weights)
            # Normalize to 0-1 range
            normalized_rating = weighted_rating / 10

            return normalized_rating

        except Exception as e:
            logger.error(f"Error in item KNN prediction: {str(e)}")
            return 0.5

    def predict_svd_score(self, user_id, game_id):
        """Predict rating using SVD model"""
        if not hasattr(self, 'svd_model') or self.svd_model is None:
            return 0.5

        try:
            return self.svd_model.predict(user_id, game_id)
        except Exception as e:
            logger.error(f"Error in SVD prediction: {str(e)}")
            return 0.5

    def predict_simple_model_score(self, user_id, game_id):
        """Predict using simple model"""
        if not hasattr(self, 'simple_model') or self.simple_model is None:
            return 0.5

        try:
            # Extract features for prediction
            features = self.extract_prediction_features(user_id, game_id)

            # Make prediction
            return self.simple_model.predict(features)[0]
        except Exception as e:
            logger.error(f"Error in simple model prediction: {str(e)}")
            return 0.5

    def predict_sequence_score(self, user_id, game_id):
        """Predict using sequence model"""
        if not hasattr(self, 'sequence_model') or self.sequence_model is None:
            return 0.5

        try:
            # Ensure we have feature columns
            if not hasattr(self, 'sequence_feature_columns'):
                logger.error("Missing sequence feature columns, cannot predict")
                return 0.5

            # Create feature dictionary matching training features
            features = {}
            for col in self.sequence_feature_columns:
                features[col] = 0.0  # Default value

            # Try to fill with actual values
            user_data = None
            if hasattr(self, 'user_df') and user_id in self.user_df['user_id'].values:
                user_data = self.user_df[self.user_df['user_id'] == user_id].iloc[0]

                # Fill user features
                if 'game_count' in self.sequence_feature_columns and 'game_count' in user_data:
                    features['game_count'] = user_data['game_count']

                if 'prev_game_count' in self.sequence_feature_columns and 'game_count' in user_data:
                    features['prev_game_count'] = user_data['game_count']

                if 'avg_prev_rating' in self.sequence_feature_columns and 'recommendation_ratio' in user_data:
                    features['avg_prev_rating'] = user_data['recommendation_ratio']

                if 'total_hours' in self.sequence_feature_columns and 'total_hours' in user_data:
                    features['total_hours'] = user_data['total_hours']

            # Use consistent feature order
            feature_vector = [features[col] for col in self.sequence_feature_columns]

            # Convert to tensor and predict
            input_tensor = torch.FloatTensor([feature_vector]).to(self.device)

            self.sequence_model.eval()
            with torch.no_grad():
                score = self.sequence_model(input_tensor).item()
                return score

        except Exception as e:
            logger.error(f"Error in sequence model prediction: {str(e)}")
            return 0.5

    def predict_content_score(self, user_id, game_id):
        """Predict using content-based model"""
        if not hasattr(self, 'content_similarity') or self.content_similarity is None:
            return 0.5

        # Find games the user likes
        if not hasattr(self, 'df'):
            # Try using train_df if full df is not available
            if hasattr(self, 'train_df'):
                df_to_use = self.train_df
            else:
                return 0.5
        else:
            df_to_use = self.df

        # Get user's liked games
        user_liked_games = df_to_use[
            (df_to_use['user_id'] == user_id) &
            (df_to_use['is_recommended'] == True)
            ]['app_id'].tolist()

        # If user has no liked games, return default score
        if not user_liked_games:
            return 0.5

        # Calculate similarity between target game and user's liked games
        similarity_scores = []

        for liked_game in user_liked_games:
            # Check if both games are in similarity matrix
            if (liked_game in self.content_similarity['game_idx'] and
                    game_id in self.content_similarity['game_idx']):
                idx1 = self.content_similarity['game_idx'][liked_game]
                idx2 = self.content_similarity['game_idx'][game_id]
                similarity = self.content_similarity['matrix'][idx1, idx2]
                similarity_scores.append(similarity)

        # If no scores could be calculated, return default
        if not similarity_scores:
            return 0.5

        # Return average similarity
        return sum(similarity_scores) / len(similarity_scores)

    def extract_prediction_features(self, user_id, game_id):
        """Extract features for prediction models"""
        # Use cache if available
        cache_key = f"{user_id}_{game_id}"
        if cache_key in self.feature_cache:
            return self.feature_cache[cache_key]

        # Get feature names from simple model if available
        if hasattr(self, 'simple_model') and self.simple_model is not None and hasattr(self.simple_model,
                                                                                       'feature_names'):
            feature_names = self.simple_model.feature_names

            # Create feature array
            features = np.zeros(len(feature_names))

            # Get user and game data
            user_data = None
            if hasattr(self, 'train_df'):
                if user_id in self.train_df['user_id'].values:
                    user_data = self.train_df[self.train_df['user_id'] == user_id].iloc[0]
            elif hasattr(self, 'test_df'):
                if user_id in self.test_df['user_id'].values:
                    user_data = self.test_df[self.test_df['user_id'] == user_id].iloc[0]

            game_data = None
            if hasattr(self, 'train_df'):
                if game_id in self.train_df['app_id'].values:
                    game_data = self.train_df[self.train_df['app_id'] == game_id].iloc[0]
            elif hasattr(self, 'test_df'):
                if game_id in self.test_df['app_id'].values:
                    game_data = self.test_df[self.test_df['app_id'] == game_id].iloc[0]

            # Extract feature values
            if user_data is not None and game_data is not None:
                feature_dict = {}

                # Collect valid features from user and game data
                for col in self.train_df.columns:
                    if (col in feature_names and col not in
                            ['user_id', 'app_id', 'is_recommended', 'date', 'review_id',
                             'prev_apps', 'prev_ratings', 'prev_hours', 'description', 'tags']):
                        if col in user_data:
                            feature_dict[col] = user_data[col]
                        elif col in game_data:
                            feature_dict[col] = game_data[col]

                # Fill feature array
                for i, name in enumerate(feature_names):
                    if name in feature_dict:
                        features[i] = feature_dict[name]

            # Cache and return
            self.feature_cache[cache_key] = features
            return features

        # If simple model doesn't exist, return default features
        return self.get_default_features()

    def get_default_features(self):
        """Get default feature vector (for cold-start)"""
        # If simple model exists, return zeros matching its features
        if hasattr(self, 'simple_model') and self.simple_model is not None and hasattr(self.simple_model,
                                                                                       'feature_names'):
            return np.zeros(len(self.simple_model.feature_names))

        # Otherwise use average features from training data
        elif hasattr(self, 'train_df') and self.train_df is not None:
            # Find all usable numerical features
            num_features = []
            for col in self.train_df.columns:
                if col not in ['user_id', 'app_id', 'date', 'review_id', 'is_recommended',
                               'prev_apps', 'prev_ratings', 'prev_hours', 'description', 'tags']:
                    if self.train_df[col].dtype in [np.int64, np.float64, np.bool_]:
                        num_features.append(col)

            # Return mean values as default features
            if num_features:
                return self.train_df[num_features].mean().values

        # If all else fails, return a single zero
        return np.array([0.0])

    def predict_score(self, user_id, game_id):
        """Predict user's preference score for a game using the hybrid approach"""
        # Use cache if available
        cache_key = f"{user_id}_{game_id}"
        if cache_key in self.score_cache:
            return self.score_cache[cache_key]

        try:
            # Get scores from different models
            user_knn_score = self.predict_user_knn_score(user_id, game_id)
            item_knn_score = self.predict_item_knn_score(user_id, game_id)
            svd_score = self.predict_svd_score(user_id, game_id)
            content_score = self.predict_content_score(user_id, game_id)
            sequence_score = self.predict_sequence_score(user_id, game_id)

            # Get weights from config
            weights = {
                'user_knn': self.config.get('user_knn_weight', 0.25),
                'item_knn': self.config.get('item_knn_weight', 0.25),
                'svd': self.config.get('svd_weight', 0.2),
                'content': self.config.get('content_weight', 0.15),
                'sequence': self.config.get('sequence_weight', 0.15)
            }

            # Calculate weighted average
            final_score = (
                    weights['user_knn'] * user_knn_score +
                    weights['item_knn'] * item_knn_score +
                    weights['svd'] * svd_score +
                    weights['content'] * content_score +
                    weights['sequence'] * sequence_score
            )

            # Cache and return result
            self.score_cache[cache_key] = final_score
            return final_score
        except Exception as e:
            logger.error(f"Error predicting score: {str(e)}")
            return 0.5

    def generate_recommendations(self, user_id, n=10):
        """Generate recommendations for a user"""
        logger.info(f"Generating recommendations for user {user_id}, top {n}")

        # Use cache if available
        cache_key = f"rec_{user_id}_{n}"
        if cache_key in self.recommendation_cache:
            return self.recommendation_cache[cache_key]

        # For new users or unknown users, use hybrid cold-start approach
        if (not hasattr(self, 'user_indices') or user_id not in self.user_indices) and (
                not hasattr(self, 'df') or user_id not in self.df['user_id'].unique()):
            logger.info(f"User {user_id} not found in training data, using cold-start recommendations")
            return self.handle_cold_start_user(n)

        # Get user's previous interactions to avoid recommending them
        user_games = set()

        # Check in main dataframe
        if hasattr(self, 'df'):
            user_games.update(self.df[self.df['user_id'] == user_id]['app_id'])

        # Check in training data
        elif hasattr(self, 'train_df'):
            user_games.update(self.train_df[self.train_df['user_id'] == user_id]['app_id'])

            # Also check test data
            if hasattr(self, 'test_df'):
                user_games.update(self.test_df[self.test_df['user_id'] == user_id]['app_id'])

        # Get all possible game IDs
        all_games = set()

        # From game dataframe if available
        if hasattr(self, 'game_df'):
            all_games.update(self.game_df['app_id'])

        # From app indices if available
        elif hasattr(self, 'app_indices'):
            all_games.update(self.app_indices.keys())

        # From training data if needed
        elif hasattr(self, 'train_df'):
            all_games.update(self.train_df['app_id'].unique())

        # Calculate candidate games (not yet interacted with)
        candidate_games = all_games - user_games

        # If no candidate games, use popular games
        if not candidate_games:
            logger.warning(f"No candidate games for user {user_id}, using popular games")
            return self.get_popular_games(n)

        # Calculate scores for all candidate games
        predictions = []
        for game_id in candidate_games:
            # Get predicted score
            score = self.predict_score(user_id, game_id)
            predictions.append((game_id, score))

        # Sort predictions by score in descending order
        predictions.sort(key=lambda x: x[1], reverse=True)

        # Return top N recommendations
        recommendations = predictions[:n]

        # Cache results
        self.recommendation_cache[cache_key] = recommendations

        logger.info(f"Generated {len(recommendations)} recommendations for user {user_id}")
        return recommendations

    def handle_cold_start_user(self, n=10):
        """Handle cold-start user (new user)"""
        logger.info("Handling cold-start user...")
        return self.get_popular_games(n)

    def get_popular_games(self, n=10):
        """Get popular games - optimized version"""
        # Use game_df if available
        if hasattr(self, 'game_df') and self.game_df is not None:
            game_popularity = self.game_df.copy()

            # Add popularity score
            if 'recommendation_ratio' in game_popularity.columns and 'user_count' in game_popularity.columns:
                game_popularity['popularity_score'] = (
                    game_popularity['user_count'] * 0.7 +
                    game_popularity['recommendation_ratio'] * 0.3
                )
            else:
                # Use available metrics
                if 'user_count' in game_popularity.columns:
                    game_popularity['popularity_score'] = game_popularity['user_count']
                else:
                    # No metrics available, use random scores
                    game_popularity['popularity_score'] = np.random.random(len(game_popularity))

            # Sort and get top N
            popular_games = game_popularity.sort_values('popularity_score', ascending=False).head(n)

            # Return (game_id, score) tuples
            return [(game_id, score) for game_id, score in zip(
                popular_games['app_id'], popular_games['popularity_score']
            )]

        # Use training data if game_df not available
        elif hasattr(self, 'train_df') and self.train_df is not None:
            # Count occurrences of each game
            game_counts = self.train_df['app_id'].value_counts().reset_index()
            game_counts.columns = ['app_id', 'count']

            # Calculate recommendation ratio if possible
            if 'is_recommended' in self.train_df.columns:
                game_recs = self.train_df.groupby('app_id')['is_recommended'].mean().reset_index()
                game_counts = game_counts.merge(game_recs, on='app_id')
                game_counts['popularity_score'] = game_counts['count'] * 0.7 + game_counts['is_recommended'] * 0.3
            else:
                game_counts['popularity_score'] = game_counts['count']

            # Get top N
            popular_games = game_counts.sort_values('popularity_score', ascending=False).head(n)

            # Return (game_id, score) tuples
            return [(game_id, score) for game_id, score in zip(
                popular_games['app_id'], popular_games['popularity_score']
            )]

        # Return empty list if no data available
        else:
            logger.warning("No data available for popular game recommendations")
            return []

    def get_content_recommendations(self, app_id, top_n=10):
        """Generate content-based recommendations for a game"""
        logger.info(f"Generating content recommendations for game {app_id}")

        # Check content similarity matrix
        if not hasattr(self, 'content_similarity') or self.content_similarity is None:
            logger.warning("Content similarity matrix not available, returning popular games")
            return self.get_popular_games(top_n)

        # Check if game is in similarity matrix
        if app_id not in self.content_similarity['game_idx']:
            logger.warning(f"Game {app_id} not in similarity matrix, returning popular games")
            return self.get_popular_games(top_n)

        # Get game index
        idx = self.content_similarity['game_idx'][app_id]

        # Get similarity scores
        sim_scores = self.content_similarity['matrix'][idx]

        # Create reverse index mapping
        reverse_idx = {idx: game_id for game_id, idx in self.content_similarity['game_idx'].items()}

        # Find similar games (exclude self)
        sim_items = [(reverse_idx[i], sim_scores[i])
                     for i in range(len(sim_scores))
                     if i != idx and i in reverse_idx]

        # Sort by similarity and get top N
        recommendations = sorted(sim_items, key=lambda x: x[1], reverse=True)[:top_n]

        return recommendations

    def update_knn_model(self, new_data_df):
        """Incrementally update KNN models with new data"""
        logger.info("Incrementally updating KNN models...")

        if not hasattr(self, 'user_knn_model') or self.user_knn_model is None:
            logger.warning("KNN models don't exist, will train from scratch")
            self.train_knn_model()
            return

        try:
            # Ensure new data has required columns
            required_cols = ['user_id', 'app_id']
            if not all(col in new_data_df.columns for col in required_cols):
                logger.error("New data missing required columns, cannot update")
                return

            # Determine rating column
            if 'rating' in new_data_df.columns:
                rating_col = 'rating'
            elif 'is_recommended' in new_data_df.columns:
                new_data_df['rating_value'] = new_data_df['is_recommended'].astype(int) * 10
                rating_col = 'rating_value'
            elif 'hours' in new_data_df.columns:
                rating_col = 'hours'
            else:
                logger.error("New data missing rating columns, cannot update")
                return

            # Check if user-game matrix exists
            if not hasattr(self, 'user_game_matrix') or self.user_game_matrix is None:
                logger.error("Original user-game matrix missing, cannot update")
                return

            # Update matrix with new data
            for _, row in new_data_df.iterrows():
                user_id = row['user_id']
                app_id = row['app_id']
                rating = row[rating_col]

                # Handle new users
                if user_id not in self.user_indices:
                    # Add to user index mapping
                    new_user_idx = len(self.user_indices)
                    self.user_indices[user_id] = new_user_idx
                    self.reversed_user_indices[new_user_idx] = user_id

                    # Add row to matrix
                    new_row = pd.DataFrame(
                        [0] * len(self.user_game_matrix.columns),
                        index=[user_id],
                        columns=self.user_game_matrix.columns
                    )
                    self.user_game_matrix = pd.concat([self.user_game_matrix, new_row])

                # Handle new games
                if app_id not in self.app_indices:
                    # Add to game index mapping
                    new_app_idx = len(self.app_indices)
                    self.app_indices[app_id] = new_app_idx
                    self.reversed_app_indices[new_app_idx] = app_id

                    # Add column to matrix
                    self.user_game_matrix[app_id] = 0

                # Update rating
                self.user_game_matrix.at[user_id, app_id] = rating

            # Update sparse matrix
            self.user_game_sparse_matrix = csr_matrix(self.user_game_matrix.values)

            # Retrain KNN models with updated data
            # User-based model
            self.user_knn_model = NearestNeighbors(
                n_neighbors=min(self.config['knn_params'].get('user_neighbors', 20), len(self.user_indices)),
                metric=self.config['knn_params'].get('metric', 'cosine'),
                algorithm=self.config['knn_params'].get('algorithm', 'brute'),
                n_jobs=-1
            )
            self.user_knn_model.fit(self.user_game_sparse_matrix)

            # Item-based model
            self.item_knn_model = NearestNeighbors(
                n_neighbors=min(self.config['knn_params'].get('item_neighbors', 20), len(self.app_indices)),
                metric=self.config['knn_params'].get('metric', 'cosine'),
                algorithm=self.config['knn_params'].get('algorithm', 'brute'),
                n_jobs=-1
            )
            self.item_knn_model.fit(self.user_game_sparse_matrix.T)

            # Clear caches
            self.recommendation_cache = {}
            self.score_cache = {}

            logger.info("KNN models successfully updated")

        except Exception as e:
            logger.error(f"Error updating KNN models: {str(e)}")
            logger.error(traceback.format_exc())

    def update_svd_model(self, new_data_df):
        """Incrementally update SVD model with new data"""
        logger.info("Incrementally updating SVD model...")

        if not hasattr(self, 'svd_model') or self.svd_model is None:
            logger.warning("SVD model doesn't exist, will train from scratch")
            self.train_svd_model()
            return

        try:
            # For SVD, it's simpler to update the data and retrain
            # Merge new data into existing data
            if hasattr(self, 'df') and self.df is not None:
                current_df = self.df.copy()
            elif hasattr(self, 'train_df') and self.train_df is not None:
                current_df = self.train_df.copy()
            else:
                logger.error("No existing data to merge with, cannot update SVD")
                return

            # Merge new interactions
            for _, row in new_data_df.iterrows():
                user_id = row['user_id']
                app_id = row['app_id']

                # Check if interaction exists
                mask = (current_df['user_id'] == user_id) & (current_df['app_id'] == app_id)
                if sum(mask) > 0:
                    # Update existing interaction
                    for col in new_data_df.columns:
                        if col in current_df.columns:
                            current_df.loc[mask, col] = row[col]
                else:
                    # Add new interaction
                    current_df = pd.concat([current_df, pd.DataFrame([row])])

            # Update df or train_df
            if hasattr(self, 'df'):
                self.df = current_df
            if hasattr(self, 'train_df'):
                self.train_df = current_df

            # Retrain SVD model
            self.train_svd_model()

            # Clear caches
            self.recommendation_cache = {}
            self.score_cache = {}

            logger.info("SVD model successfully updated")

        except Exception as e:
            logger.error(f"Error updating SVD model: {str(e)}")
            logger.error(traceback.format_exc())

    def update_simple_model(self, new_data_df):
        """Incrementally update simple classification model with new data"""
        logger.info("Incrementally updating simple model...")

        if not hasattr(self, 'simple_model') or self.simple_model is None:
            logger.warning("Simple model doesn't exist, will train from scratch")
            self.train_simple_model()
            return

        try:
            # Merge new data into existing data
            if hasattr(self, 'df') and self.df is not None:
                current_df = self.df.copy()
            elif hasattr(self, 'train_df') and self.train_df is not None:
                current_df = self.train_df.copy()
            else:
                logger.error("No existing data to merge with, cannot update simple model")
                return

            # Merge new interactions
            for _, row in new_data_df.iterrows():
                user_id = row['user_id']
                app_id = row['app_id']

                # Check if interaction exists
                mask = (current_df['user_id'] == user_id) & (current_df['app_id'] == app_id)
                if sum(mask) > 0:
                    # Update existing interaction
                    for col in new_data_df.columns:
                        if col in current_df.columns:
                            current_df.loc[mask, col] = row[col]
                else:
                    # Add new interaction
                    current_df = pd.concat([current_df, pd.DataFrame([row])])

            # Update df or train_df
            if hasattr(self, 'df'):
                self.df = current_df
            if hasattr(self, 'train_df'):
                self.train_df = current_df

            # Re-engineer features
            self.engineer_features()

            # Retrain simple model
            self.train_simple_model()

            # Clear caches
            self.recommendation_cache = {}
            self.score_cache = {}
            self.feature_cache = {}

            logger.info("Simple model successfully updated")

        except Exception as e:
            logger.error(f"Error updating simple model: {str(e)}")
            logger.error(traceback.format_exc())

    def update_sequence_model(self, new_data_df):
        """Incrementally update sequence model with new data"""
        logger.info("Incrementally updating sequence model...")

        if not hasattr(self, 'sequence_model') or self.sequence_model is None:
            logger.warning("Sequence model doesn't exist, will train from scratch")
            self.train_sequence_model()
            return

        try:
            # For sequence model, we'll update the data and fine-tune the model
            if hasattr(self, 'df') and self.df is not None:
                current_df = self.df.copy()
            elif hasattr(self, 'train_df') and self.train_df is not None:
                current_df = self.train_df.copy()
            else:
                logger.error("No existing data to merge with, cannot update sequence model")
                return

            # Merge new interactions
            for _, row in new_data_df.iterrows():
                user_id = row['user_id']
                app_id = row['app_id']

                # Check if interaction exists
                mask = (current_df['user_id'] == user_id) & (current_df['app_id'] == app_id)
                if sum(mask) > 0:
                    # Update existing interaction
                    for col in new_data_df.columns:
                        if col in current_df.columns:
                            current_df.loc[mask, col] = row[col]
                else:
                    # Add new interaction
                    current_df = pd.concat([current_df, pd.DataFrame([row])])

            # Update df or train_df
            if hasattr(self, 'df'):
                self.df = current_df
            if hasattr(self, 'train_df'):
                self.train_df = current_df

            # Update sequence features
            self.create_sequence_features()

            if not hasattr(self, 'sequence_feature_columns') or not self.sequence_feature_columns:
                logger.error("Missing sequence feature columns, cannot update model")
                return

            # Prepare data for fine-tuning
            X = self.train_df[self.sequence_feature_columns].values
            y = self.train_df['is_recommended'].astype(float).values

            # Setup fine-tuning parameters
            batch_size = min(self.config['sequence_params'].get('batch_size', 64), len(X))
            learning_rate = self.config['sequence_params'].get('learning_rate', 0.001) * 0.5  # Lower rate for fine-tuning
            epochs = max(2, self.config['sequence_params'].get('epochs', 10) // 3)  # Fewer epochs for fine-tuning

            # Setup optimizer and loss
            optimizer = torch.optim.Adam(self.sequence_model.parameters(), lr=learning_rate)
            criterion = torch.nn.BCELoss()

            # Fine-tuning
            self.sequence_model.train()
            total_loss = 0

            for epoch in range(epochs):
                # Shuffle data
                indices = np.random.permutation(len(X))
                X_shuffled = X[indices]
                y_shuffled = y[indices]

                epoch_loss = 0
                num_batches = 0

                for i in range(0, len(X), batch_size):
                    # Get batch
                    X_batch = X_shuffled[i:i + batch_size]
                    y_batch = y_shuffled[i:i + batch_size]

                    # Convert to tensors
                    X_tensor = torch.FloatTensor(X_batch).to(self.device)
                    y_tensor = torch.FloatTensor(y_batch).to(self.device)

                    # Forward pass
                    optimizer.zero_grad()
                    outputs = self.sequence_model(X_tensor)

                    # Calculate loss
                    loss = criterion(outputs, y_tensor)
                    epoch_loss += loss.item()

                    # Backward pass and optimize
                    loss.backward()
                    optimizer.step()

                    num_batches += 1

                # Calculate average loss
                avg_loss = epoch_loss / num_batches
                total_loss += avg_loss
                logger.info(f"Fine-tuning epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}")

            # Set model to evaluation mode
            self.sequence_model.eval()

            # Clear caches
            self.recommendation_cache = {}
            self.score_cache = {}

            logger.info(f"Sequence model successfully fine-tuned, avg loss: {total_loss / epochs:.4f}")

        except Exception as e:
            logger.error(f"Error updating sequence model: {str(e)}")
            logger.error(traceback.format_exc())

    def update_content_model(self, games_df):
        """Update content-based model with new game data"""
        logger.info("Updating content-based model...")

        try:
            # Merge new game data with existing
            if hasattr(self, 'df') and self.df is not None:
                current_df = self.df.copy()
            elif hasattr(self, 'train_df') and self.train_df is not None:
                current_df = self.train_df.copy()
            else:
                logger.error("No existing data to merge with, cannot update content model")
                return

            # Update game information
            for _, row in games_df.iterrows():
                app_id = row['app_id']

                # Check if game exists
                game_exists = app_id in current_df['app_id'].values

                if game_exists:
                    # Update game information
                    mask = current_df['app_id'] == app_id
                    for col in games_df.columns:
                        if col in current_df.columns:
                            current_df.loc[mask, col] = row[col]
                else:
                    # Add new game (placeholder data)
                    placeholder = {
                        'app_id': app_id,
                        'title': row.get('title', f"Game {app_id}"),
                        'user_id': -1,  # Placeholder user
                        'is_recommended': False,
                        'hours': 0
                    }
                    # Add additional columns from game data
                    for col in games_df.columns:
                        if col not in placeholder and col in current_df.columns:
                            placeholder[col] = row[col]

                    # Add to dataframe
                    current_df = pd.concat([current_df, pd.DataFrame([placeholder])])

            # Update df or train_df
            if hasattr(self, 'df'):
                self.df = current_df
            if hasattr(self, 'train_df'):
                self.train_df = current_df

            # Retrain content model
            self.train_content_model()

            # Update game embeddings if applicable
            if hasattr(self, 'create_game_embeddings'):
                self.create_game_embeddings()

            # Clear caches
            self.recommendation_cache = {}
            self.score_cache = {}

            logger.info("Content model successfully updated")

        except Exception as e:
            logger.error(f"Error updating content model: {str(e)}")
            logger.error(traceback.format_exc())

    def incremental_update(self, interactions_df, games_df=None, users_df=None):
        """Perform incremental update of all models"""
        logger.info("Starting comprehensive incremental update...")

        try:
            # Update KNN model
            if len(interactions_df) > 0:
                self.update_knn_model(interactions_df)

            # Update SVD model
            if len(interactions_df) > 0:
                self.update_svd_model(interactions_df)

            # Update simple model
            if len(interactions_df) > 0:
                self.update_simple_model(interactions_df)

            # Update sequence model
            if len(interactions_df) > 0:
                self.update_sequence_model(interactions_df)

            # Update content model
            if games_df is not None and len(games_df) > 0:
                self.update_content_model(games_df)

            # Update game embeddings
            self.create_game_embeddings()

            # Clear caches
            self.recommendation_cache = {}
            self.score_cache = {}
            self.feature_cache = {}

            logger.info("Incremental update completed")

        except Exception as e:
            logger.error(f"Error in incremental update: {str(e)}")
            logger.error(traceback.format_exc())

    def evaluate_recommendations(self, k_values=[5, 10, 20], test_users=None):
        """Evaluate recommendation system performance"""
        logger.info("Evaluating recommendation system...")

        # Ensure test data is available
        if not hasattr(self, 'test_df') or self.test_df is None:
            logger.error("No test data available for evaluation")
            return None

        # Limit evaluation users to improve efficiency
        if test_users is None:
            max_test_users = min(100, len(self.test_df['user_id'].unique()))
            test_users = np.random.choice(self.test_df['user_id'].unique(), max_test_users, replace=False)

        logger.info(f"Evaluating with {len(test_users)} test users")

        # Initialize metrics
        metrics = {
            'precision': {k: [] for k in k_values},
            'recall': {k: [] for k in k_values},
            'ndcg': {k: [] for k in k_values},
            'diversity': {k: [] for k in k_values},
            'coverage': []
        }

        # Track all recommended games for coverage calculation
        all_recommended_games = set()
        all_games = set(self.test_df['app_id'].unique())

        # Evaluate each test user
        for user_id in test_users:
            # Get user's liked games from test set
            user_liked_games = set(self.test_df[
                (self.test_df['user_id'] == user_id) &
                (self.test_df['is_recommended'] == True)
            ]['app_id'].values)

            # Skip users with no liked games
            if not user_liked_games:
                continue

            # Generate recommendations for this user
            max_k = max(k_values)
            recommendations = self.generate_recommendations(user_id, max_k)
            recommended_games = [game_id for game_id, _ in recommendations]

            # Update coverage tracking
            all_recommended_games.update(recommended_games)

            # Calculate metrics for each k value
            for k in k_values:
                top_k_games = recommended_games[:k]

                # Precision = hits / k
                hits = len(set(top_k_games) & user_liked_games)
                precision = hits / k if k > 0 else 0
                metrics['precision'][k].append(precision)

                # Recall = hits / total_liked
                recall = hits / len(user_liked_games) if user_liked_games else 0
                metrics['recall'][k].append(recall)

                # NDCG
                if len(self.test_df) > 0 and k > 0:
                    # Create relevance array (1 for liked, 0 for not)
                    relevance = [1 if game in user_liked_games else 0 for game in top_k_games]
                    # Ideal ordering
                    ideal_relevance = sorted(relevance, reverse=True)

                    # Calculate DCG
                    dcg = sum(rel / np.log2(i + 2) for i, rel in enumerate(relevance))
                    # Calculate ideal DCG
                    idcg = sum(rel / np.log2(i + 2) for i, rel in enumerate(ideal_relevance))

                    # Calculate NDCG
                    ndcg = dcg / idcg if idcg > 0 else 0
                    metrics['ndcg'][k].append(ndcg)

                # Calculate diversity if tags available
                if 'tags' in self.test_df.columns:
                    # Get tags for recommended games
                    game_tags = {}
                    for game_id in top_k_games:
                        # Find game in test set
                        game_data = self.test_df[self.test_df['app_id'] == game_id]
                        if not game_data.empty and pd.notna(game_data['tags'].iloc[0]):
                            tags = set(tag.strip() for tag in game_data['tags'].iloc[0].split(','))
                            game_tags[game_id] = tags
                        else:
                            game_tags[game_id] = set()

                    # Calculate average pairwise Jaccard distance
                    if len(game_tags) >= 2:
                        diversity_scores = []
                        for i, (game1, tags1) in enumerate(game_tags.items()):
                            for game2, tags2 in list(game_tags.items())[i + 1:]:
                                if tags1 and tags2:  # Only if both have tags
                                    jaccard_similarity = len(tags1 & tags2) / len(tags1 | tags2)
                                    jaccard_distance = 1 - jaccard_similarity
                                    diversity_scores.append(jaccard_distance)

                        if diversity_scores:
                            diversity = sum(diversity_scores) / len(diversity_scores)
                            metrics['diversity'][k].append(diversity)

        # Calculate coverage
        coverage = len(all_recommended_games) / len(all_games) if all_games else 0
        metrics['coverage'] = coverage

        # Calculate average metrics
        results = {}
        for metric in ['precision', 'recall', 'ndcg', 'diversity']:
            results[metric] = {k: np.mean(metrics[metric][k]) if metrics[metric][k] else 0 for k in k_values}
        results['coverage'] = metrics['coverage']

        # Print results
        logger.info("Evaluation results:")
        for metric in ['precision', 'recall', 'ndcg', 'diversity']:
            logger.info(f"{metric.capitalize()}:")
            for k in k_values:
                logger.info(f"  @{k}: {results[metric][k]:.4f}")
        logger.info(f"Coverage: {results['coverage']:.4f}")

        # Store results
        self.evaluation_results = results
        return results

        def save_model(self, path='steam_recommender_model'):
            """Save model and related data"""
            logger.info(f"Saving model to {path}...")

            # Create save directory
            os.makedirs(path, exist_ok=True)

            # Save KNN model related data
            if hasattr(self, 'user_knn_model') and self.user_knn_model is not None:
                # Save index mappings
                with open(os.path.join(path, 'user_indices.pkl'), 'wb') as f:
                    pickle.dump(self.user_indices, f)
                with open(os.path.join(path, 'app_indices.pkl'), 'wb') as f:
                    pickle.dump(self.app_indices, f)
                with open(os.path.join(path, 'reversed_user_indices.pkl'), 'wb') as f:
                    pickle.dump(self.reversed_user_indices, f)
                with open(os.path.join(path, 'reversed_app_indices.pkl'), 'wb') as f:
                    pickle.dump(self.reversed_app_indices, f)

                # Save user-game matrix
                if hasattr(self, 'user_game_matrix') and self.user_game_matrix is not None:
                    self.user_game_matrix.to_pickle(os.path.join(path, 'user_game_matrix.pkl'))

                # Save KNN models
                with open(os.path.join(path, 'user_knn_model.pkl'), 'wb') as f:
                    pickle.dump(self.user_knn_model, f)
                with open(os.path.join(path, 'item_knn_model.pkl'), 'wb') as f:
                    pickle.dump(self.item_knn_model, f)

            # Save SVD model
            if hasattr(self, 'svd_model') and self.svd_model is not None:
                with open(os.path.join(path, 'svd_model.pkl'), 'wb') as f:
                    pickle.dump(self.svd_model, f)

            # Save simple model
            if hasattr(self, 'simple_model') and self.simple_model is not None:
                with open(os.path.join(path, 'simple_model.pkl'), 'wb') as f:
                    pickle.dump(self.simple_model, f)

            # Save sequence model
            if hasattr(self, 'sequence_model') and self.sequence_model is not None:
                torch.save(self.sequence_model.state_dict(), os.path.join(path, 'sequence_model.pt'))
                # Save sequence feature columns
                if hasattr(self, 'sequence_feature_columns'):
                    with open(os.path.join(path, 'sequence_features.pkl'), 'wb') as f:
                        pickle.dump(self.sequence_feature_columns, f)

            # Save encoders and other components
            if hasattr(self, 'label_encoders') and self.label_encoders:
                with open(os.path.join(path, 'label_encoders.pkl'), 'wb') as f:
                    pickle.dump(self.label_encoders, f)

            if hasattr(self, 'scaler') and self.scaler is not None:
                with open(os.path.join(path, 'scaler.pkl'), 'wb') as f:
                    pickle.dump(self.scaler, f)

            if hasattr(self, 'tfidf_model') and self.tfidf_model is not None:
                with open(os.path.join(path, 'tfidf_model.pkl'), 'wb') as f:
                    pickle.dump(self.tfidf_model, f)

            if hasattr(self, 'tfidf_svd') and self.tfidf_svd is not None:
                with open(os.path.join(path, 'tfidf_svd.pkl'), 'wb') as f:
                    pickle.dump(self.tfidf_svd, f)

            if hasattr(self, 'game_embeddings') and self.game_embeddings is not None:
                with open(os.path.join(path, 'game_embeddings.pkl'), 'wb') as f:
                    pickle.dump(self.game_embeddings, f)

            if hasattr(self, 'content_similarity') and self.content_similarity is not None:
                with open(os.path.join(path, 'content_similarity.pkl'), 'wb') as f:
                    pickle.dump(self.content_similarity, f)

            # Save configuration
            with open(os.path.join(path, 'config.json'), 'w') as f:
                # Create a copy to avoid modifying original
                save_config = self.config.copy()
                # Convert any numpy values to native Python types for JSON serialization
                for k, v in save_config.items():
                    if isinstance(v, dict):
                        for k2, v2 in v.items():
                            if isinstance(v2, np.ndarray):
                                save_config[k][k2] = v2.tolist()
                            elif isinstance(v2, np.integer):
                                save_config[k][k2] = int(v2)
                            elif isinstance(v2, np.floating):
                                save_config[k][k2] = float(v2)
                    elif isinstance(v, np.ndarray):
                        save_config[k] = v.tolist()
                    elif isinstance(v, np.integer):
                        save_config[k] = int(v)
                    elif isinstance(v, np.floating):
                        save_config[k] = float(v)

                json.dump(save_config, f, indent=4)

            # Save training history and evaluation results
            if hasattr(self, 'training_history') and self.training_history:
                with open(os.path.join(path, 'training_history.json'), 'w') as f:
                    # Convert dict values to JSON serializable
                    history = {}
                    for k, v in self.training_history.items():
                        if isinstance(v, np.ndarray):
                            history[k] = v.tolist()
                        elif isinstance(v, list) and len(v) > 0 and isinstance(v[0], np.ndarray):
                            history[k] = [arr.tolist() for arr in v]
                        else:
                            history[k] = v

                    json.dump(history, f, indent=4)

            if hasattr(self, 'evaluation_results') and self.evaluation_results:
                with open(os.path.join(path, 'evaluation_results.json'), 'w') as f:
                    json.dump(self.evaluation_results, f, indent=4)

            # Save feature importance if available
            if hasattr(self, 'feature_importance') and self.feature_importance is not None:
                self.feature_importance.to_csv(os.path.join(path, 'feature_importance.csv'), index=False)

            logger.info(f"Model save completed")
            return True

        def load_model(self, path='steam_recommender_model'):
            """Load model and related data"""
            logger.info(f"Loading model from {path}...")

            # Check if directory exists
            if not os.path.exists(path):
                logger.error(f"Model directory {path} does not exist")
                return False

            try:
                # Load configuration
                config_path = os.path.join(path, 'config.json')
                if os.path.exists(config_path):
                    with open(config_path, 'r') as f:
                        self.config = json.load(f)

                # Load KNN model related data
                user_indices_path = os.path.join(path, 'user_indices.pkl')
                app_indices_path = os.path.join(path, 'app_indices.pkl')
                reversed_user_indices_path = os.path.join(path, 'reversed_user_indices.pkl')
                reversed_app_indices_path = os.path.join(path, 'reversed_app_indices.pkl')
                user_game_matrix_path = os.path.join(path, 'user_game_matrix.pkl')
                user_knn_model_path = os.path.join(path, 'user_knn_model.pkl')
                item_knn_model_path = os.path.join(path, 'item_knn_model.pkl')

                # Load index mappings
                if os.path.exists(user_indices_path):
                    with open(user_indices_path, 'rb') as f:
                        self.user_indices = pickle.load(f)

                if os.path.exists(app_indices_path):
                    with open(app_indices_path, 'rb') as f:
                        self.app_indices = pickle.load(f)

                if os.path.exists(reversed_user_indices_path):
                    with open(reversed_user_indices_path, 'rb') as f:
                        self.reversed_user_indices = pickle.load(f)

                if os.path.exists(reversed_app_indices_path):
                    with open(reversed_app_indices_path, 'rb') as f:
                        self.reversed_app_indices = pickle.load(f)

                # Load user-game matrix
                if os.path.exists(user_game_matrix_path):
                    self.user_game_matrix = pd.read_pickle(user_game_matrix_path)
                    # Rebuild sparse matrix
                    self.user_game_sparse_matrix = csr_matrix(self.user_game_matrix.values)

                # Load KNN models
                if os.path.exists(user_knn_model_path):
                    with open(user_knn_model_path, 'rb') as f:
                        self.user_knn_model = pickle.load(f)

                if os.path.exists(item_knn_model_path):
                    with open(item_knn_model_path, 'rb') as f:
                        self.item_knn_model = pickle.load(f)

                # Load SVD model
                svd_model_path = os.path.join(path, 'svd_model.pkl')
                if os.path.exists(svd_model_path):
                    with open(svd_model_path, 'rb') as f:
                        self.svd_model = pickle.load(f)

                # Load simple model
                simple_model_path = os.path.join(path, 'simple_model.pkl')
                if os.path.exists(simple_model_path):
                    with open(simple_model_path, 'rb') as f:
                        self.simple_model = pickle.load(f)

                # Load encoders and other components
                encoders_path = os.path.join(path, 'label_encoders.pkl')
                if os.path.exists(encoders_path):
                    with open(encoders_path, 'rb') as f:
                        self.label_encoders = pickle.load(f)

                scaler_path = os.path.join(path, 'scaler.pkl')
                if os.path.exists(scaler_path):
                    with open(scaler_path, 'rb') as f:
                        self.scaler = pickle.load(f)

                tfidf_path = os.path.join(path, 'tfidf_model.pkl')
                if os.path.exists(tfidf_path):
                    with open(tfidf_path, 'rb') as f:
                        self.tfidf_model = pickle.load(f)

                svd_path = os.path.join(path, 'tfidf_svd.pkl')
                if os.path.exists(svd_path):
                    with open(svd_path, 'rb') as f:
                        self.tfidf_svd = pickle.load(f)

                embeddings_path = os.path.join(path, 'game_embeddings.pkl')
                if os.path.exists(embeddings_path):
                    with open(embeddings_path, 'rb') as f:
                        self.game_embeddings = pickle.load(f)

                similarity_path = os.path.join(path, 'content_similarity.pkl')
                if os.path.exists(similarity_path):
                    with open(similarity_path, 'rb') as f:
                        self.content_similarity = pickle.load(f)

                # Load sequence model features
                sequence_features_path = os.path.join(path, 'sequence_features.pkl')
                sequence_model_path = os.path.join(path, 'sequence_model.pt')

                if os.path.exists(sequence_features_path):
                    with open(sequence_features_path, 'rb') as f:
                        self.sequence_feature_columns = pickle.load(f)

                # Load sequence model if features are available
                if os.path.exists(sequence_model_path) and hasattr(self, 'sequence_feature_columns'):
                    # Initialize model architecture before loading weights
                    self.sequence_model = GameSequenceModel(
                        num_features=len(self.sequence_feature_columns),
                        hidden_dim=self.config['sequence_params'].get('hidden_dim', 128),
                        num_layers=self.config['sequence_params'].get('num_layers', 2),
                        dropout=self.config['sequence_params'].get('dropout', 0.2)
                    ).to(self.device)

                    self.sequence_model.load_state_dict(
                        torch.load(sequence_model_path, map_location=self.device)
                    )
                    self.sequence_model.eval()

                # Load training history and evaluation results
                history_path = os.path.join(path, 'training_history.json')
                if os.path.exists(history_path):
                    with open(history_path, 'r') as f:
                        self.training_history = json.load(f)

                eval_path = os.path.join(path, 'evaluation_results.json')
                if os.path.exists(eval_path):
                    with open(eval_path, 'r') as f:
                        self.evaluation_results = json.load(f)

                # Load feature importance
                importance_path = os.path.join(path, 'feature_importance.csv')
                if os.path.exists(importance_path):
                    self.feature_importance = pd.read_csv(importance_path)

                logger.info(f"Model loading completed successfully")
                return True

            except Exception as e:
                logger.error(f"Error loading model: {str(e)}")
                logger.error(traceback.format_exc())
                return False

        def visualize_results(self):
            """Visualize evaluation results"""
            if not hasattr(self, 'evaluation_results') or self.evaluation_results is None:
                logger.warning("No evaluation results to visualize")
                return

            try:
                # Set up nice plotting style
                plt.style.use('ggplot')

                # Create figure for metrics
                plt.figure(figsize=(15, 10))

                # Get k values
                if 'precision' in self.evaluation_results:
                    k_values = sorted(self.evaluation_results['precision'].keys())
                else:
                    logger.warning("No precision metrics found")
                    return

                # Plot precision, recall, NDCG
                metrics = ['precision', 'recall', 'ndcg', 'diversity']
                for i, metric in enumerate(metrics, 1):
                    if metric in self.evaluation_results:
                        plt.subplot(2, 2, i)

                        values = [self.evaluation_results[metric][str(k)]
                                  if isinstance(k, int) and str(k) in self.evaluation_results[metric]
                                  else self.evaluation_results[metric].get(k, 0)
                                  for k in k_values]

                        plt.plot(k_values, values, 'o-', linewidth=2)
                        plt.title(f'{metric.capitalize()} @ k')
                        plt.xlabel('k')
                        plt.ylabel(metric)
                        plt.grid(True)

                        # Add value labels
                        for x, y in zip(k_values, values):
                            plt.annotate(
                                f"{y:.3f}",
                                (x, y),
                                textcoords="offset points",
                                xytext=(0, 10),
                                ha='center'
                            )

                plt.tight_layout()
                plt.savefig('evaluation_metrics.png', dpi=300, bbox_inches='tight')
                plt.close()

                # Create coverage visualization
                plt.figure(figsize=(8, 6))

                coverage = self.evaluation_results.get('coverage', 0)
                plt.bar(['Coverage'], [coverage], color='skyblue')
                plt.ylim(0, 1.1)
                plt.ylabel('Score')
                plt.title('Recommendation Coverage')
                plt.grid(axis='y')

                # Add value label
                plt.annotate(
                    f"{coverage:.3f}",
                    (0, coverage),
                    textcoords="offset points",
                    xytext=(0, 10),
                    ha='center'
                )

                plt.tight_layout()
                plt.savefig('coverage_metric.png', dpi=300, bbox_inches='tight')
                plt.close()

                # If feature importance is available
                if hasattr(self, 'feature_importance') and self.feature_importance is not None:
                    plt.figure(figsize=(12, 8))

                    # Get top 15 features
                    n_features = min(15, len(self.feature_importance))
                    top_features = self.feature_importance.sort_values('Importance', ascending=True).tail(n_features)

                    # Create horizontal bar chart
                    plt.barh(top_features['Feature'], top_features['Importance'], color='skyblue')
                    plt.xlabel('Importance')
                    plt.title('Feature Importance')
                    plt.tight_layout()
                    plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
                    plt.close()

                logger.info("Evaluation visualizations created successfully")

            except Exception as e:
                logger.error(f"Error visualizing results: {str(e)}")
                logger.error(traceback.format_exc())

        # Visualization methods
        def visualize_user_game_matrix(self, save_path=None):
            """Visualize the user-game matrix sparsity"""
            if not hasattr(self, 'user_game_matrix') or self.user_game_matrix is None:
                logger.warning("User-game matrix not available for visualization")
                return

            try:
                # Calculate sparsity
                matrix = self.user_game_matrix
                non_zeros = np.count_nonzero(matrix.values)
                total_cells = matrix.shape[0] * matrix.shape[1]
                sparsity = 1 - (non_zeros / total_cells)

                plt.figure(figsize=(10, 8))
                plt.spy(matrix.values, markersize=0.1, aspect='auto')
                plt.title(f'User-Game Matrix Sparsity: {sparsity:.2%}')
                plt.xlabel('Games')
                plt.ylabel('Users')

                if save_path:
                    plt.savefig(save_path, dpi=300, bbox_inches='tight')
                    logger.info(f"Matrix sparsity plot saved to {save_path}")
                else:
                    plt.show()

                plt.close()

                # Print statistics
                print(f"\nUser-Game Matrix Analysis:")
                print(f"Matrix shape: {matrix.shape[0]} users × {matrix.shape[1]} games")
                print(f"Non-zero cells: {non_zeros:,} ({non_zeros / total_cells:.2%})")
                print(f"Sparsity: {sparsity:.2%}")

                return sparsity

            except Exception as e:
                logger.error(f"Error visualizing user-game matrix: {str(e)}")
                logger.error(traceback.format_exc())
                return None

        def visualize_game_similarity(self, game_id, n=10, save_path=None):
            """Visualize similarity between a game and its most similar games"""
            if not hasattr(self, 'content_similarity') or self.content_similarity is None:
                logger.warning("Content similarity not available for visualization")
                return

            try:
                # Check if game exists in similarity matrix
                if game_id not in self.content_similarity['game_idx']:
                    logger.warning(f"Game ID {game_id} not found in similarity matrix")
                    return

                # Get similar games
                similar_games = self.get_content_recommendations(game_id, n)

                # Get game titles
                if hasattr(self, 'df') and 'app_id' in self.df.columns and 'title' in self.df.columns:
                    # Get title of target game
                    target_title = self.df[self.df['app_id'] == game_id]['title'].iloc[0]

                    # Get titles of similar games
                    similar_titles = []
                    similarity_scores = []
                    for sim_id, score in similar_games:
                        title = self.df[self.df['app_id'] == sim_id]['title'].iloc[0]
                        similar_titles.append(title)
                        similarity_scores.append(score)

                    # Create visualization
                    plt.figure(figsize=(12, 8))
                    y_pos = np.arange(len(similar_titles))

                    plt.barh(y_pos, similarity_scores, align='center')
                    plt.yticks(y_pos, similar_titles)
                    plt.xlabel('Similarity Score')
                    plt.title(f'Games Similar to "{target_title}"')

                    if save_path:
                        plt.savefig(save_path, dpi=300, bbox_inches='tight')
                        logger.info(f"Game similarity visualization saved to {save_path}")
                    else:
                        plt.show()

                    plt.close()

                    return similar_games
                else:
                    logger.warning("Game metadata not available for visualization")
                    return similar_games

            except Exception as e:
                logger.error(f"Error visualizing game similarity: {str(e)}")
                logger.error(traceback.format_exc())
                return None

        def visualize_user_recommendations(self, user_id, n=10, save_path=None):
            """Visualize recommendations for a specific user"""
            try:
                # Generate recommendations
                recommendations = self.generate_recommendations(user_id, n)

                if not recommendations:
                    logger.warning(f"No recommendations generated for user {user_id}")
                    return None

                # Get game titles and scores
                titles = []
                scores = []

                if hasattr(self, 'df') and 'app_id' in self.df.columns and 'title' in self.df.columns:
                    for game_id, score in recommendations:
                        game_data = self.df[self.df['app_id'] == game_id]
                        if not game_data.empty:
                            titles.append(game_data['title'].iloc[0])
                            scores.append(score)
                        else:
                            titles.append(f"Game {game_id}")
                            scores.append(score)

                    # Create visualization
                    plt.figure(figsize=(12, 8))
                    y_pos = np.arange(len(titles))

                    plt.barh(y_pos, scores, align='center')
                    plt.yticks(y_pos, titles)
                    plt.xlabel('Recommendation Score')
                    plt.title(f'Top {n} Recommendations for User {user_id}')

                    if save_path:
                        plt.savefig(save_path, dpi=300, bbox_inches='tight')
                        logger.info(f"User recommendations visualization saved to {save_path}")
                    else:
                        plt.show()

                    plt.close()

                return recommendations

            except Exception as e:
                logger.error(f"Error visualizing user recommendations: {str(e)}")
                logger.error(traceback.format_exc())
                return None

    def main():
        """Main function to demonstrate usage"""
        # Set up logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )

        # Initialize recommender
        recommender = SteamRecommender('steam_top_100000.csv')

        # Load data
        recommender.load_data()

        # Engineer features
        recommender.engineer_features()

        # Train models
        recommender.train_knn_model()
        recommender.train_svd_model()
        recommender.train_simple_model()
        recommender.train_sequence_model()
        recommender.create_game_embeddings()
        recommender.train_content_model()

        # Evaluate recommender
        evaluation_results = recommender.evaluate_recommendations()
        recommender.evaluation_results = evaluation_results

        # Visualize results
        recommender.visualize_results()

        # Save model
        recommender.save_model('steam_recommender_model')

        # Generate recommendations for a sample user
        if len(recommender.df['user_id'].unique()) > 0:
            sample_user = recommender.df['user_id'].iloc[0]
            recommendations = recommender.generate_recommendations(sample_user, 10)

            # Print recommendations
            print(f"\nRecommendations for user {sample_user}:")
            for i, (game_id, score) in enumerate(recommendations, 1):
                game_title = recommender.df[recommender.df['app_id'] == game_id]['title'].iloc[0]
                print(f"{i}. {game_title} (ID: {game_id}, Score: {score:.4f})")

        print("\nRecommendation system demonstration completed!")

    if __name__ == "__main__":
        main()