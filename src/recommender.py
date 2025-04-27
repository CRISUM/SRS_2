#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
src/recommender.py - Main Steam Game Recommender System
Author: YourName
Date: 2025-04-27
Description: Main class that integrates all components for Steam game recommendations
"""

import logging
import os
import json
import pickle
import numpy as np
import pandas as pd
import torch
import traceback

from data.data_processor import SteamDataProcessor
from data.feature_extractor import GameFeatureExtractor
from models.knn_model import KNNModel
from models.svd_model import SVDModel
from models.sequence_model import SequenceRecommender
from models.hybrid_model import HybridRecommender
from evaluation.evaluator import RecommenderEvaluator
from visualization.visualizer import RecommenderVisualizer

logger = logging.getLogger(__name__)


class SteamRecommender:
    """Steam game recommendation system main class"""

    def __init__(self, data_path=None, config=None):
        """Initialize recommendation system

        Args:
            data_path (str): Data file path
            config (dict): Configuration parameters
        """
        # Set up default configuration
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

        # Update with provided config
        if config:
            self.config.update(config)

        # Initialize components
        self.data_processor = SteamDataProcessor(self.config)
        self.feature_extractor = GameFeatureExtractor(self.config)
        self.evaluator = RecommenderEvaluator(k_values=[5, 10, 20])
        self.visualizer = RecommenderVisualizer(output_dir='visualizations')

        # Initialize models
        self.models = {}
        self.hybrid_model = None

        # Data path
        self.data_path = data_path

        # Device for PyTorch models
        self.device = torch.device('cuda' if self.config['use_gpu'] and torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")

        # Caches
        self.recommendation_cache = {}

        # For storing evaluation results
        self.evaluation_results = None

    def load_data(self):
        """Load and process data"""
        logger.info(f"Loading data from {self.data_path}")
        try:
            # Call data processor to load data
            success = self.data_processor.load_data(self.data_path)

            if not success:
                logger.error("Failed to load data")
                return False

            logger.info("Data loaded successfully")
            return True
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            logger.error(traceback.format_exc())
            return False

    def engineer_features(self):
        """Perform feature engineering"""
        logger.info("Engineering features for recommendation models")
        try:
            # Call data processor to engineer features
            success = self.data_processor.engineer_features()

            if not success:
                logger.error("Failed to engineer features")
                return False

            # Extract game content features using the feature extractor
            self.feature_extractor.config = self.config

            # Prepare tag features if available
            if hasattr(self.data_processor, 'train_df') and 'tags' in self.data_processor.train_df.columns:
                logger.info("Extracting tag features...")
                self.feature_extractor.extract_tag_features(self.data_processor.train_df)

            # Prepare text features if available
            if hasattr(self.data_processor, 'train_df') and 'description' in self.data_processor.train_df.columns:
                logger.info("Extracting text features...")
                self.feature_extractor.extract_text_features(self.data_processor.train_df)

            logger.info("Feature engineering completed successfully")
            return True
        except Exception as e:
            logger.error(f"Error engineering features: {str(e)}")
            logger.error(traceback.format_exc())
            return False

    def train_models(self):
        """Train all recommendation models"""
        logger.info("Training recommendation models...")

        try:
            # Make sure data is loaded and processed
            if not hasattr(self.data_processor, 'train_df') or self.data_processor.train_df is None:
                logger.error("No training data available, please load data first")
                return False

            train_df = self.data_processor.train_df
            test_df = self.data_processor.test_df

            # 1. Train KNN models
            logger.info("Training User-based KNN model...")
            user_knn = KNNModel(type='user',
                                n_neighbors=self.config['knn_params']['user_neighbors'],
                                metric=self.config['knn_params']['metric'],
                                algorithm=self.config['knn_params']['algorithm'])
            user_knn.fit(train_df)
            self.models['user_knn'] = user_knn

            logger.info("Training Item-based KNN model...")
            item_knn = KNNModel(type='item',
                                n_neighbors=self.config['knn_params']['item_neighbors'],
                                metric=self.config['knn_params']['metric'],
                                algorithm=self.config['knn_params']['algorithm'])
            item_knn.fit(train_df)
            self.models['item_knn'] = item_knn

            # 2. Train SVD model
            logger.info("Training SVD model...")
            svd_model = SVDModel(n_components=self.config['svd_params']['n_components'],
                                 random_state=self.config['svd_params']['random_state'])
            svd_model.fit(train_df)
            self.models['svd'] = svd_model

            # 3. Train sequence model
            logger.info("Training sequence model...")
            # Prepare sequence features
            if hasattr(self.data_processor, 'sequence_feature_columns'):
                sequence_features = self.data_processor.sequence_feature_columns

                # Extract features and targets
                X = train_df[sequence_features].values
                y = train_df['is_recommended'].astype(float).values

                # Prepare user data dictionary
                user_data = {}
                for user_id in train_df['user_id'].unique():
                    user_items = train_df[train_df['user_id'] == user_id]['app_id'].tolist()
                    user_data[user_id] = {'items': user_items}

                # Prepare item data dictionary
                item_data = {}
                for app_id in train_df['app_id'].unique():
                    item_data[app_id] = {'id': app_id}

                # Prepare training data for sequence model
                sequence_data = {
                    'X': X,
                    'y': y,
                    'feature_columns': sequence_features,
                    'user_data': user_data,
                    'item_data': item_data
                }

                # Initialize and train sequence model
                sequence_model = SequenceRecommender(
                    hidden_dim=self.config['sequence_params']['hidden_dim'],
                    num_layers=self.config['sequence_params']['num_layers'],
                    dropout=self.config['sequence_params']['dropout'],
                    learning_rate=self.config['sequence_params']['learning_rate'],
                    batch_size=self.config['sequence_params']['batch_size'],
                    epochs=self.config['sequence_params']['epochs'],
                    device=self.device
                )
                sequence_model.fit(sequence_data)
                self.models['sequence'] = sequence_model
            else:
                logger.warning("Sequence feature columns not found, skipping sequence model training")

            # 4. Create content-based similarity matrix
            logger.info("Creating content-based similarity matrix...")
            self.feature_extractor.create_game_embeddings(train_df)
            content_sim = self.feature_extractor.get_similarity_matrix(self.feature_extractor.game_embeddings)
            self.models['content'] = content_sim

            # 5. Create hybrid recommender
            logger.info("Creating hybrid recommender...")
            model_weights = {
                'user_knn': self.config['user_knn_weight'],
                'item_knn': self.config['item_knn_weight'],
                'svd': self.config['svd_weight'],
                'sequence': self.config['sequence_weight'],
                'content': self.config['content_weight']
            }

            self.hybrid_model = HybridRecommender(self.models, model_weights)

            # 6. Prepare popular items for cold start
            popular_games = self.get_popular_games(20)
            self.hybrid_model.popular_items = popular_games

            logger.info("All models trained successfully")
            return True

        except Exception as e:
            logger.error(f"Error training models: {str(e)}")
            logger.error(traceback.format_exc())
            return False

    def generate_recommendations(self, user_id, n=10):
        """Generate recommendations for a user

        Args:
            user_id: User ID
            n (int): Number of recommendations

        Returns:
            list: List of (game_id, score) tuples
        """
        logger.info(f"Generating {n} recommendations for user {user_id}")

        # Check cache first
        cache_key = f"rec_{user_id}_{n}"
        if cache_key in self.recommendation_cache:
            return self.recommendation_cache[cache_key]

        try:
            # Use hybrid model if available
            if self.hybrid_model is not None:
                recommendations = self.hybrid_model.recommend(user_id, n)
            else:
                # Fall back to popular games
                recommendations = self.get_popular_games(n)

            # Cache results
            self.recommendation_cache[cache_key] = recommendations

            return recommendations
        except Exception as e:
            logger.error(f"Error generating recommendations: {str(e)}")
            logger.error(traceback.format_exc())
            return self.get_popular_games(n)

    def evaluate_recommendations(self, k_values=None):
        """Evaluate recommendation performance

        Args:
            k_values (list): List of k values to evaluate

        Returns:
            dict: Evaluation metrics
        """
        logger.info("Evaluating recommendation performance...")

        try:
            # Set default k values if not provided
            if k_values is None:
                k_values = [5, 10, 20]

            # Make sure test data is available
            if not hasattr(self.data_processor, 'test_df') or self.data_processor.test_df is None:
                logger.error("No test data available for evaluation")
                return None

            # Use the evaluator to evaluate the hybrid model
            metrics = self.evaluator.evaluate(
                model=self.hybrid_model,
                test_df=self.data_processor.test_df,
                k_values=k_values
            )

            # Store evaluation results
            self.evaluation_results = metrics

            return metrics
        except Exception as e:
            logger.error(f"Error evaluating recommendations: {str(e)}")
            logger.error(traceback.format_exc())
            return None

    def save_model(self, path=None):
        """Save all models and data

        Args:
            path (str): Directory path

        Returns:
            bool: Success
        """
        # Set default path if not provided
        if path is None:
            path = 'steam_recommender_model'

        logger.info(f"Saving model to {path}...")

        try:
            # Create directory
            os.makedirs(path, exist_ok=True)

            # Save configuration
            with open(os.path.join(path, 'config.json'), 'w') as f:
                json.dump(self.config, f, indent=2)

            # Save individual models
            models_dir = os.path.join(path, 'models')
            os.makedirs(models_dir, exist_ok=True)

            for name, model in self.models.items():
                model_dir = os.path.join(models_dir, name)
                os.makedirs(model_dir, exist_ok=True)
                try:
                    model.save(model_dir)
                except Exception as e:
                    logger.error(f"Error saving {name} model: {str(e)}")

            # Save hybrid model
            if self.hybrid_model is not None:
                hybrid_dir = os.path.join(path, 'hybrid')
                os.makedirs(hybrid_dir, exist_ok=True)
                self.hybrid_model.save(hybrid_dir)

            # Save feature extractor
            if hasattr(self.feature_extractor, 'game_embeddings') and self.feature_extractor.game_embeddings:
                with open(os.path.join(path, 'game_embeddings.pkl'), 'wb') as f:
                    pickle.dump(self.feature_extractor.game_embeddings, f)

            # Save popular games list for cold start
            popular_games = self.get_popular_games(100)
            with open(os.path.join(path, 'popular_games.pkl'), 'wb') as f:
                pickle.dump(popular_games, f)

            # Save evaluation results if available
            if self.evaluation_results:
                with open(os.path.join(path, 'evaluation_results.json'), 'w') as f:
                    # Convert numpy types to native Python types
                    evaluation_results = {}
                    for metric, values in self.evaluation_results.items():
                        if isinstance(values, dict):
                            evaluation_results[metric] = {
                                str(k): float(v) for k, v in values.items()
                            }
                        else:
                            evaluation_results[metric] = float(values)

                    json.dump(evaluation_results, f, indent=2)

            logger.info(f"Model saved successfully to {path}")
            return True
        except Exception as e:
            logger.error(f"Error saving model: {str(e)}")
            logger.error(traceback.format_exc())
            return False

    def load_model(self, path=None):
        """Load all models and data

        Args:
            path (str): Directory path

        Returns:
            bool: Success
        """
        # Set default path if not provided
        if path is None:
            path = 'steam_recommender_model'

        logger.info(f"Loading model from {path}...")

        try:
            # Load configuration
            with open(os.path.join(path, 'config.json'), 'r') as f:
                self.config = json.load(f)

            # Initialize models dictionary
            self.models = {}

            # Load models
            models_dir = os.path.join(path, 'models')

            # User KNN
            user_knn_dir = os.path.join(models_dir, 'user_knn')
            if os.path.exists(user_knn_dir):
                user_knn = KNNModel(type='user')
                user_knn.load(user_knn_dir)
                self.models['user_knn'] = user_knn

            # Item KNN
            item_knn_dir = os.path.join(models_dir, 'item_knn')
            if os.path.exists(item_knn_dir):
                item_knn = KNNModel(type='item')
                item_knn.load(item_knn_dir)
                self.models['item_knn'] = item_knn

            # SVD model
            svd_dir = os.path.join(models_dir, 'svd')
            if os.path.exists(svd_dir):
                svd_model = SVDModel()
                svd_model.load(svd_dir)
                self.models['svd'] = svd_model

            # Sequence model
            sequence_dir = os.path.join(models_dir, 'sequence')
            if os.path.exists(sequence_dir):
                sequence_model = SequenceRecommender(device=self.device)
                sequence_model.load(sequence_dir)
                self.models['sequence'] = sequence_model

            # Content model
            content_dir = os.path.join(models_dir, 'content')
            if os.path.exists(content_dir):
                # For content similarity matrix, load directly
                with open(os.path.join(content_dir, 'content_similarity.pkl'), 'rb') as f:
                    content_sim = pickle.load(f)
                self.models['content'] = content_sim

            # Load game embeddings if available
            embeddings_path = os.path.join(path, 'game_embeddings.pkl')
            if os.path.exists(embeddings_path):
                with open(embeddings_path, 'rb') as f:
                    self.feature_extractor.game_embeddings = pickle.load(f)

            # Load popular games
            popular_games_path = os.path.join(path, 'popular_games.pkl')
            if os.path.exists(popular_games_path):
                with open(popular_games_path, 'rb') as f:
                    popular_games = pickle.load(f)
            else:
                popular_games = []

            # Load hybrid model
            hybrid_dir = os.path.join(path, 'hybrid')
            if os.path.exists(hybrid_dir):
                self.hybrid_model = HybridRecommender()
                hybrid_result = self.hybrid_model.load(hybrid_dir)

                # If load returns a tuple (model, model_list), update models
                if isinstance(hybrid_result, tuple):
                    self.hybrid_model, model_list = hybrid_result

                    # Add missing models to hybrid model
                    for name in model_list:
                        if name in self.models and name not in self.hybrid_model.models:
                            self.hybrid_model.add_model(name, self.models[name])

                # Set popular items for cold start
                self.hybrid_model.popular_items = popular_games
            else:
                # Create hybrid model
                model_weights = {
                    'user_knn': self.config.get('user_knn_weight', 0.25),
                    'item_knn': self.config.get('item_knn_weight', 0.25),
                    'svd': self.config.get('svd_weight', 0.2),
                    'sequence': self.config.get('sequence_weight', 0.15),
                    'content': self.config.get('content_weight', 0.15)
                }
                self.hybrid_model = HybridRecommender(self.models, model_weights)
                self.hybrid_model.popular_items = popular_games
                # Load evaluation results if available
                eval_results_path = os.path.join(path, 'evaluation_results.json')
                if os.path.exists(eval_results_path):
                    with open(eval_results_path, 'r') as f:
                        self.evaluation_results = json.load(f)

                logger.info(f"Model loaded successfully from {path}")
                return True
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            logger.error(traceback.format_exc())
            return False

    def visualize_results(self):
        """Visualize evaluation results"""
        logger.info("Visualizing evaluation results...")

        try:
            # Check if evaluation results are available
            if not self.evaluation_results:
                logger.warning("No evaluation results available for visualization")
                return False

            # Use visualizer to create visualizations
            self.visualizer.visualize_metrics(self.evaluation_results)

            # Visualize training history if available
            for name, model in self.models.items():
                if hasattr(model, 'training_history') and model.training_history:
                    self.visualizer.visualize_training_history(model.training_history, name)

            # Visualize feature importance if available
            for name, model in self.models.items():
                if hasattr(model, 'feature_importance') and model.feature_importance is not None:
                    self.visualizer.visualize_feature_importance(model.feature_importance, name)

            logger.info("Visualizations created successfully")
            return True
        except Exception as e:
            logger.error(f"Error visualizing results: {str(e)}")
            logger.error(traceback.format_exc())
            return False

    def handle_cold_start_user(self, n=10):
        """Handle cold-start user (new user)

        Args:
            n (int): Number of recommendations

        Returns:
            list: List of (game_id, score) tuples for popular games
        """
        logger.info(f"Handling cold-start user with {n} popular recommendations")
        return self.get_popular_games(n)

    def get_popular_games(self, n=10):
        """Get popular games

        Args:
            n (int): Number of games to return

        Returns:
            list: List of (game_id, score) tuples
        """
        logger.info(f"Getting {n} popular games")

        try:
            # Check if data is loaded
            if not hasattr(self.data_processor, 'game_df') or self.data_processor.game_df is None:
                logger.warning("No game data available")
                return []

            # Get game popularity data
            game_df = self.data_processor.game_df.copy()

            # Calculate popularity score
            if 'recommendation_ratio' in game_df.columns and 'user_count' in game_df.columns:
                game_df['popularity_score'] = (
                        0.6 * game_df['user_count'] / game_df['user_count'].max() +
                        0.4 * game_df['recommendation_ratio']
                )
            elif 'user_count' in game_df.columns:
                game_df['popularity_score'] = game_df['user_count'] / game_df['user_count'].max()
            else:
                logger.warning("Insufficient data for popularity calculation")
                return []

            # Sort by popularity score
            popular_games = game_df.sort_values('popularity_score', ascending=False).head(n)

            # Return as (game_id, score) tuples
            return list(zip(popular_games['app_id'], popular_games['popularity_score']))
        except Exception as e:
            logger.error(f"Error getting popular games: {str(e)}")
            logger.error(traceback.format_exc())
            return []

    def incremental_update(self, interactions_df, games_df=None, users_df=None):
        """Perform incremental update of models

        Args:
            interactions_df (DataFrame): New user-item interactions
            games_df (DataFrame): New game data (optional)
            users_df (DataFrame): New user data (optional)

        Returns:
            bool: Success
        """
        logger.info(f"Performing incremental update with {len(interactions_df)} new interactions")

        try:
            # Prepare update data for each model
            update_data = {}

            # For KNN models
            if 'user_knn' in self.models:
                update_data['user_knn'] = interactions_df

            if 'item_knn' in self.models:
                update_data['item_knn'] = interactions_df

            # For SVD model
            if 'svd' in self.models:
                update_data['svd'] = interactions_df

            # For sequence model
            if 'sequence' in self.models and hasattr(self.data_processor, 'sequence_feature_columns'):
                # Engineer sequence features for new data
                sequence_df = interactions_df.copy()

                # Try to apply the same feature engineering as in training
                try:
                    # Merge with existing user/game data if available
                    if hasattr(self.data_processor, 'train_df'):
                        # Get user features
                        user_features = self.data_processor.train_df.groupby('user_id').agg({
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

                        # Merge with sequence_df
                        sequence_df = sequence_df.merge(user_features, on='user_id', how='left')

                        # Get game features
                        game_features = self.data_processor.train_df.groupby('app_id').agg({
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

                        # Merge with sequence_df
                        sequence_df = sequence_df.merge(game_features, on='app_id', how='left')

                    # Create sequence statistics
                    sequence_df['prev_game_count'] = sequence_df['user_id'].map(
                        lambda user_id: len(
                            self.data_processor.train_df[self.data_processor.train_df['user_id'] == user_id])
                    )

                    sequence_df['avg_prev_rating'] = sequence_df['user_id'].map(
                        lambda user_id:
                        self.data_processor.train_df[self.data_processor.train_df['user_id'] == user_id][
                            'is_recommended'].mean()
                    )

                    sequence_df['avg_prev_hours'] = sequence_df['user_id'].map(
                        lambda user_id:
                        self.data_processor.train_df[self.data_processor.train_df['user_id'] == user_id]['hours'].mean()
                    )

                    # Fill missing values
                    for col in self.data_processor.sequence_feature_columns:
                        if col in sequence_df.columns:
                            sequence_df[col] = sequence_df[col].fillna(0)
                        else:
                            sequence_df[col] = 0

                    # Extract features and targets
                    X = sequence_df[self.data_processor.sequence_feature_columns].values
                    y = sequence_df['is_recommended'].astype(float).values

                    # Prepare user data dictionary
                    user_data = {}
                    for user_id in sequence_df['user_id'].unique():
                        user_items = sequence_df[sequence_df['user_id'] == user_id]['app_id'].tolist()
                        user_data[user_id] = {'items': user_items}

                    # Prepare item data dictionary
                    item_data = {}
                    for app_id in sequence_df['app_id'].unique():
                        item_data[app_id] = {'id': app_id}

                    # Prepare sequence model update data
                    update_data['sequence'] = {
                        'X': X,
                        'y': y,
                        'feature_columns': self.data_processor.sequence_feature_columns,
                        'user_data': user_data,
                        'item_data': item_data
                    }
                except Exception as e:
                    logger.error(f"Error preparing sequence features: {str(e)}")
                    logger.error(traceback.format_exc())

            # For content model
            if 'content' in self.models and games_df is not None and len(games_df) > 0:
                update_data['content'] = games_df

            # Update popular games if needed
            if games_df is not None or len(interactions_df) > 0:
                popular_games = self.get_popular_games(100)
                update_data['popular_items'] = popular_games

            # Use hybrid model to update all components
            if self.hybrid_model is not None:
                self.hybrid_model.update(update_data)

                # Clear recommendation cache
                self.recommendation_cache = {}

                logger.info("Incremental update completed successfully")
                return True
            else:
                logger.error("Hybrid model not available for update")
                return False
        except Exception as e:
            logger.error(f"Error in incremental update: {str(e)}")
            logger.error(traceback.format_exc())
            return False