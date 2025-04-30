#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Py_GetModel_src/recommender.py - Main Steam Game Recommender System
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
from models.content_model import ContentBasedModel
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
                'user_neighbors': 30,
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
            'svd_weight': 0.25,
            'content_weight': 0.125,
            'sequence_weight': 0.125,
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

            # 3. Train game sequence model (replacing the user sequence model)
            logger.info("Training game sequence model...")
            # Create game sequence model optimized for sparse data
            from models.sequence_model import SequenceRecommender
            game_seq_model = SequenceRecommender(
                hidden_dim=self.config['sequence_params']['hidden_dim'],
                num_layers=self.config['sequence_params']['num_layers'],
                dropout=self.config['sequence_params']['dropout'],
                learning_rate=self.config['sequence_params']['learning_rate'],
                batch_size=self.config['sequence_params']['batch_size'],
                epochs=self.config['sequence_params']['epochs'],
                device=self.device
            )
            # Pass the DataFrame directly to utilize game-to-game transitions
            game_seq_model.fit(train_df)
            self.models['sequence'] = game_seq_model

            # 4. Create enhanced content-based model
            logger.info("Creating enhanced content-based model...")
            self.feature_extractor.create_game_embeddings(train_df)
            content_sim = self.feature_extractor.get_similarity_matrix(self.feature_extractor.game_embeddings)
            content_model = ContentBasedModel(content_sim)

            # Use better dataset for content-based model
            content_model.fit(train_df)

            # Set popular items for cold start
            popular_games = self.get_popular_games(20)
            content_model.popular_items = popular_games

            self.models['content'] = content_model

            # 5. Create hybrid recommender with adjusted weights for sparse data
            if all(model is not None for model in self.models.values()):
                logger.info("Creating hybrid recommender with optimized weights...")

                # Adjust weights for sparse data - increase weight for content and item-based models
                # Reduce weight for user-based models that need rich user history
                model_weights = {
                    'user_knn': self.config.get('user_knn_weight', 0.15),  # Reduced from 0.25
                    'item_knn': self.config.get('item_knn_weight', 0.30),  # Increased from 0.25
                    'svd': self.config.get('svd_weight', 0.15),  # Reduced from 0.25
                    'sequence': self.config.get('sequence_weight', 0.15),  # Increased from 0.125
                    'content': self.config.get('content_weight', 0.25)  # Increased from 0.125
                }

                self.hybrid_model = HybridRecommender(self.models, model_weights)

                # 6. Prepare enhanced popular items for cold start
                popular_games = self.get_popular_games(50)  # Get more for better diversity
                self.hybrid_model.popular_items = popular_games

                # 7. Configure hybrid model for better diversity and coverage
                if hasattr(self.hybrid_model, 'config'):
                    self.hybrid_model.config = self.config
                else:
                    setattr(self.hybrid_model, 'config', self.config)

                # Set diversity settings if not already present
                if not hasattr(self.hybrid_model, 'diversity_factor'):
                    self.hybrid_model.diversity_factor = 0.3

                # Enable caching for performance
                if not hasattr(self.hybrid_model, 'enable_cache'):
                    self.hybrid_model.enable_cache = True

                logger.info("All models trained successfully")
                return True
            else:
                logger.error("Some models failed to train, hybrid model creation skipped")
                return False

        except Exception as e:
            logger.error(f"Error training models: {str(e)}")
            logger.error(traceback.format_exc())
            # 确保出错时hybrid_model被设为None
            self.hybrid_model = None
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
        """Evaluate recommendation performance"""
        logger.info("Evaluating recommendation performance...")

        try:
            # 检查hybrid_model是否可用
            if self.hybrid_model is None:
                logger.error("Hybrid model is not available for evaluation")
                return None

            # 检查测试数据是否可用
            if not hasattr(self.data_processor, 'test_df') or self.data_processor.test_df is None:
                logger.error("No test data available for evaluation")
                return None

            # 添加这行记录测试数据大小
            logger.info(
                f"Test data contains {len(self.data_processor.test_df)} rows and {self.data_processor.test_df['user_id'].nunique()} unique users")

            # 设置默认k值
            if k_values is None:
                k_values = [5, 10, 20]

            # 使用evaluator评估hybrid模型
            metrics = self.evaluator.evaluate(
                model=self.hybrid_model,
                test_df=self.data_processor.test_df,
                k_values=k_values
            )

            # 检查结果
            if metrics is None:
                logger.warning("Evaluation returned no results")
                return None

            # 存储评估结果
            self.evaluation_results = metrics
            logger.info("Evaluation completed successfully")

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
                content_model = ContentBasedModel()
                content_model.load(content_dir)
                self.models['content'] = content_model

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
                    'user_knn': self.config.get('user_knn_weight', 0.15),
                    'item_knn': self.config.get('item_knn_weight', 0.30),
                    'svd': self.config.get('svd_weight', 0.2),
                    'sequence': self.config.get('sequence_weight', 0.15),
                    'content': self.config.get('content_weight', 0.20)
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

    def enhance_content_model(self):
        """增强基于内容的推荐模型"""
        logger.info("Enhancing content-based recommendation model...")

        try:
            # 检查是否有训练数据
            if not hasattr(self.data_processor, 'train_df') or self.data_processor.train_df is None:
                logger.error("No training data available for enhancing content model")
                return False

            # 1. 提取更丰富的游戏特征
            game_metadata = {}
            for _, row in self.data_processor.train_df.drop_duplicates('app_id').iterrows():
                game_id = row['app_id']

                # 构建特征向量
                features = {
                    'tags': row.get('tags', '').split(',') if isinstance(row.get('tags', ''), str) else [],
                    'title': row.get('title', f"Game {game_id}")
                }

                # 添加可选特征（如果存在）
                for col in ['price_final', 'win', 'mac', 'linux', 'rating', 'positive_ratio', 'date_release']:
                    if col in row and not pd.isna(row[col]):
                        features[col] = row[col]

                game_metadata[game_id] = features

            # 2. 计算游戏相似度
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.metrics.pairwise import cosine_similarity

            # 准备标签文本
            game_tags = {}
            for game_id, metadata in game_metadata.items():
                tags = metadata['tags']
                game_tags[game_id] = ' '.join([tag.strip() for tag in tags]) if tags else ''

            # 使用TF-IDF向量化游戏标签
            vectorizer = TfidfVectorizer(min_df=1)
            all_game_ids = list(game_tags.keys())
            tag_texts = [game_tags[gid] for gid in all_game_ids]

            # 检查是否有足够的数据
            if len(tag_texts) > 1 and any(tag_texts):
                tag_matrix = vectorizer.fit_transform(tag_texts)

                # 计算游戏相似度
                tag_similarity = cosine_similarity(tag_matrix)

                # 创建游戏相似度字典
                game_similarities = {}
                for i, game_id in enumerate(all_game_ids):
                    similar_games = [(all_game_ids[j], tag_similarity[i, j])
                                     for j in range(len(all_game_ids)) if i != j]
                    similar_games.sort(key=lambda x: x[1], reverse=True)
                    game_similarities[game_id] = similar_games

                # 增强内容模型
                if 'content' in self.models:
                    content_model = self.models['content']
                    content_model.similarity_matrix = game_similarities
                    content_model.item_metadata = game_metadata

                    logger.info(f"Enhanced content model with {len(game_similarities)} game similarities")
                    return True
                else:
                    logger.warning("Content model not found in models dictionary")
                    return False
            else:
                logger.warning("Not enough tag data to create enhanced content model")
                return False

        except Exception as e:
            logger.error(f"Error enhancing content model: {str(e)}")
            logger.error(traceback.format_exc())
            return False

    def optimize_model_weights(self):
        """优化模型权重以适应稀疏数据"""
        logger.info("Optimizing model weights for sparse data...")

        try:
            if self.hybrid_model is None:
                logger.error("No hybrid model available for weight optimization")
                return False

            # 检查是否有用户数据
            has_user_data = self.data_processor.train_df['user_id'].nunique() > 0
            if not has_user_data:
                logger.warning("No user data found, cannot analyze sparsity")
                return False

            # 1. 分析数据稀疏度
            user_counts = self.data_processor.train_df.groupby('user_id').size()

            # 计算用户交互数据的统计指标
            avg_user_interactions = user_counts.mean()
            median_user_interactions = user_counts.median()
            sparse_user_ratio = (user_counts <= 2).mean()  # 交互<=2的用户比例

            logger.info(f"Data sparsity analysis: Avg={avg_user_interactions:.2f}, "
                        f"Median={median_user_interactions:.2f}, "
                        f"Sparse user ratio={sparse_user_ratio:.2f}")

            # 2. 根据数据稀疏性调整权重
            # 数据越稀疏，越偏向基于内容和物品的方法
            # 数据越丰富，越偏向基于用户协同过滤的方法

            # 基础权重配置
            base_weights = {
                'user_knn': 0.25,
                'item_knn': 0.25,
                'svd': 0.25,
                'sequence': 0.125,
                'content': 0.125
            }

            # 根据稀疏度调整权重
            adjusted_weights = {}

            # 非常稀疏（大多数用户只有1-2个交互）
            if sparse_user_ratio > 0.7:
                adjusted_weights = {
                    'user_knn': 0.1,  # 降低基于用户的权重
                    'item_knn': 0.35,  # 增加基于物品的权重
                    'svd': 0.15,  # 降低SVD权重
                    'sequence': 0.15,  # 保持序列模型权重
                    'content': 0.25  # 大幅提高内容模型权重
                }
                logger.info("Applying weights for very sparse data")

            # 中等稀疏（大约一半用户有少量交互）
            elif sparse_user_ratio > 0.4:
                adjusted_weights = {
                    'user_knn': 0.15,  # 降低基于用户的权重
                    'item_knn': 0.30,  # 增加基于物品的权重
                    'svd': 0.20,  # 略微降低SVD权重
                    'sequence': 0.15,  # 保持序列模型权重
                    'content': 0.20  # 增加内容模型权重
                }
                logger.info("Applying weights for moderately sparse data")

            # 数据较丰富
            else:
                adjusted_weights = {
                    'user_knn': 0.25,  # 保持基于用户的权重
                    'item_knn': 0.25,  # 保持基于物品的权重
                    'svd': 0.25,  # 保持SVD权重
                    'sequence': 0.15,  # 略微增加序列模型权重
                    'content': 0.10  # 略微降低内容模型权重
                }
                logger.info("Applying weights for relatively rich data")

            # 3. 更新混合模型权重
            self.hybrid_model.weights = adjusted_weights

            # 同时更新配置中的权重
            for model_name, weight in adjusted_weights.items():
                weight_key = f"{model_name}_weight"
                self.config[weight_key] = weight

            logger.info(f"Updated model weights: {adjusted_weights}")
            return True

        except Exception as e:
            logger.error(f"Error optimizing model weights: {str(e)}")
            logger.error(traceback.format_exc())
            return False

    def improve_diversity_and_coverage(self):
        """提高推荐的多样性和覆盖率"""
        logger.info("Improving recommendation diversity and coverage...")

        try:
            if self.hybrid_model is None:
                logger.error("No hybrid model available for diversity improvement")
                return False

            # 1. 设置多样性因子
            self.hybrid_model.diversity_factor = 0.3  # 0.3 的多样性权重在多样性和相关性之间取得平衡

            # 2. 启用缓存以提高性能
            self.hybrid_model.enable_cache = True

            # 3. 增加冷启动策略中的多样性
            # 已经在 get_cold_start_recommendations 中实现

            # 4. 将配置传递给混合模型以便其能访问全局设置
            if hasattr(self.hybrid_model, 'config'):
                self.hybrid_model.config.update(self.config)
            else:
                setattr(self.hybrid_model, 'config', self.config)

            # 5. 为物品多样性添加相关信息
            if hasattr(self.data_processor, 'train_df') and 'tags' in self.data_processor.train_df.columns:
                # 创建游戏标签字典
                game_tags = {}
                for _, row in self.data_processor.train_df.drop_duplicates('app_id').iterrows():
                    if pd.notna(row['tags']) and row['tags']:
                        tags = [tag.strip() for tag in str(row['tags']).split(',')]
                        game_tags[row['app_id']] = set(tags)

                # 传递给混合模型
                if hasattr(self.hybrid_model, 'item_tags'):
                    self.hybrid_model.item_tags.update(game_tags)
                else:
                    setattr(self.hybrid_model, 'item_tags', game_tags)

            logger.info("Successfully configured diversity and coverage improvements")
            return True

        except Exception as e:
            logger.error(f"Error improving diversity and coverage: {str(e)}")
            logger.error(traceback.format_exc())
            return False

    def train_and_optimize(self):
        """训练并优化推荐系统，包括所有增强策略"""
        logger.info("Training and optimizing the recommendation system...")

        try:
            # 1. 首先加载和处理数据
            if not hasattr(self.data_processor, 'train_df') or self.data_processor.train_df is None:
                success = self.load_data()
                if not success:
                    logger.error("Failed to load data")
                    return False

            # 2. 进行特征工程
            success = self.engineer_features()
            if not success:
                logger.error("Failed to engineer features")
                return False

            # 3. 检查数据情况 - 记录详细统计信息
            train_df = self.data_processor.train_df

            # 记录数据统计情况
            user_count = train_df['user_id'].nunique()
            item_count = train_df['app_id'].nunique()
            interactions = len(train_df)

            avg_interactions_per_user = interactions / user_count if user_count > 0 else 0
            density = interactions / (user_count * item_count) if user_count > 0 and item_count > 0 else 0

            logger.info(f"Data statistics: {user_count} users, {item_count} items, "
                        f"{interactions} interactions")
            logger.info(f"Average interactions per user: {avg_interactions_per_user:.2f}, "
                        f"Matrix density: {density:.6f}")

            # 3.1 记录用户交互分布
            user_interaction_counts = train_df.groupby('user_id').size()
            logger.info(f"User interaction distribution: Min={user_interaction_counts.min()}, "
                        f"Max={user_interaction_counts.max()}, "
                        f"Median={user_interaction_counts.median()}, "
                        f"Mean={user_interaction_counts.mean():.2f}")

            # 用户交互分布直方图
            if hasattr(self, 'visualizer'):
                try:
                    plt.figure(figsize=(10, 6))
                    plt.hist(user_interaction_counts, bins=30, alpha=0.7)
                    plt.title('User Interaction Distribution', fontsize=16)
                    plt.xlabel('Number of Interactions', fontsize=14)
                    plt.ylabel('Number of Users', fontsize=14)
                    plt.grid(True, alpha=0.3)
                    plt.savefig(os.path.join(self.visualizer.output_dir, 'user_interaction_distribution.png'),
                                dpi=300, bbox_inches='tight')
                    plt.close()
                except Exception as e:
                    logger.warning(f"Could not create user interaction histogram: {str(e)}")

            # 4. 训练基础模型
            logger.info("Training base recommendation models...")
            success = self.train_models()
            if not success:
                logger.error("Failed to train models")
                return False

            # 5. 增强内容模型 - 明确记录日志
            logger.info("Enhancing content-based model...")
            if self.enhance_content_model():
                logger.info("Content-based model enhanced successfully")
            else:
                logger.warning("Content-based model enhancement failed or skipped")

            # 6. 优化模型权重 - 确保执行并记录结果
            logger.info("Optimizing model weights for data characteristics...")
            if self.optimize_model_weights():
                logger.info("Model weights optimized successfully")

                # 检查权重是否实际更新
                if hasattr(self.hybrid_model, 'weights'):
                    logger.info(f"New model weights: {self.hybrid_model.weights}")
                else:
                    logger.warning("Hybrid model has no weights attribute after optimization")
            else:
                logger.warning("Model weight optimization failed or skipped")

            # 7. 增强多样性和覆盖率
            logger.info("Improving recommendation diversity and coverage...")
            if self.improve_diversity_and_coverage():
                logger.info("Diversity and coverage improvements applied successfully")
            else:
                logger.warning("Diversity and coverage improvements failed or skipped")

            # 8. 评估推荐质量
            logger.info("Evaluating recommendation quality...")
            evaluation_results = self.evaluate_recommendations()
            if evaluation_results:
                logger.info("Recommendation system evaluation results:")
                for metric in ['precision', 'recall', 'ndcg', 'diversity']:
                    if metric in evaluation_results:
                        for k, value in evaluation_results[metric].items():
                            logger.info(f"{metric}@{k}: {value:.4f}")

                if 'coverage' in evaluation_results:
                    logger.info(f"Coverage: {evaluation_results['coverage']:.4f}")
            else:
                logger.warning("Recommendation evaluation failed or returned no results")

            # 9. 测试不同的KNN参数
            logger.info("Testing different KNN parameters...")
            if hasattr(self, 'test_knn_clustering'):
                knn_results = self.test_knn_clustering(
                    user_neighbors_range=[5, 10, 15, 20, 25, 30, 40],
                    item_neighbors_range=[5, 10, 15, 20, 25, 30]
                )

                # 可视化KNN测试结果
                if knn_results and hasattr(self.visualizer, 'visualize_knn_optimization'):
                    self.visualizer.visualize_knn_optimization(knn_results)
                elif knn_results:
                    logger.info("KNN test completed but no visualization method available")
                else:
                    logger.warning("KNN parameter testing failed or returned no results")
            else:
                logger.warning("KNN testing method not available")

            # 10. 创建可视化结果
            logger.info("Creating visualizations...")
            if self.visualize_results():
                logger.info("Visualizations created successfully")
            else:
                logger.warning("Visualization creation failed or skipped")

            logger.info("Recommendation system training and optimization completed successfully")
            return True

        except Exception as e:
            logger.error(f"Error in training and optimization process: {str(e)}")
            logger.error(traceback.format_exc())
            return False

    def test_knn_clustering(self, user_neighbors_range=None, item_neighbors_range=None):
        """测试不同KNN邻居数量对模型性能的影响

        Args:
            user_neighbors_range (list): 要测试的user_knn邻居数量列表
            item_neighbors_range (list): 要测试的item_knn邻居数量列表

        Returns:
            dict: 不同参数组合的评估结果
        """
        logger.info("测试不同KNN邻居数量的影响...")

        # 设置默认测试范围 - 更多的测试值
        if user_neighbors_range is None:
            user_neighbors_range = [5, 10, 15, 20, 25, 30, 40, 50, 60]
        if item_neighbors_range is None:
            item_neighbors_range = [5, 10, 15, 20, 25, 30, 35, 40]

        # 存储原始配置以便测试后恢复
        original_config = self.config.copy()

        results = {}
        best_ndcg = 0
        best_params = None

        try:
            # 确保数据已加载
            if not hasattr(self.data_processor, 'train_df') or self.data_processor.train_df is None:
                logger.error("无法测试KNN参数：数据未加载")
                return None

            train_df = self.data_processor.train_df
            test_df = self.data_processor.test_df

            # 对每个参数组合进行测试
            for u_neigh in user_neighbors_range:
                for i_neigh in item_neighbors_range:
                    config_key = f"user_{u_neigh}_item_{i_neigh}"
                    logger.info(f"测试参数组合: user_neighbors={u_neigh}, item_neighbors={i_neigh}")

                    # 更新配置
                    self.config['knn_params']['user_neighbors'] = u_neigh
                    self.config['knn_params']['item_neighbors'] = i_neigh

                    # 训练KNN模型
                    user_knn = KNNModel(type='user',
                                        n_neighbors=u_neigh,
                                        metric=self.config['knn_params']['metric'],
                                        algorithm=self.config['knn_params']['algorithm'])
                    user_knn.fit(train_df)

                    item_knn = KNNModel(type='item',
                                        n_neighbors=i_neigh,
                                        metric=self.config['knn_params']['metric'],
                                        algorithm=self.config['knn_params']['algorithm'])
                    item_knn.fit(train_df)

                    # 创建临时混合模型仅包含KNN模型
                    temp_models = {'user_knn': user_knn, 'item_knn': item_knn}
                    model_weights = {'user_knn': 0.5, 'item_knn': 0.5}
                    temp_hybrid = HybridRecommender(temp_models, model_weights)

                    # 评估结果
                    metrics = self.evaluator.evaluate(
                        model=temp_hybrid,
                        test_df=test_df,
                        k_values=[5, 10, 20]  # 使用多个k值评估
                    )

                    if metrics is not None:
                        results[config_key] = metrics

                        # 跟踪最佳参数 (使用NDCG@10作为主要指标)
                        ndcg_10 = metrics['ndcg'][10]
                        if ndcg_10 > best_ndcg:
                            best_ndcg = ndcg_10
                            best_params = {
                                'user_neighbors': u_neigh,
                                'item_neighbors': i_neigh,
                                'ndcg@10': ndcg_10
                            }

                        logger.info(f"参数组合 {config_key} 的NDCG@10: {ndcg_10:.4f}")

            # 恢复原始配置
            self.config = original_config

            # 打印最佳参数
            if best_params:
                logger.info(f"最佳KNN参数: user_neighbors={best_params['user_neighbors']}, "
                            f"item_neighbors={best_params['item_neighbors']}, "
                            f"NDCG@10={best_params['ndcg@10']:.4f}")

            # 可视化KNN测试结果
            self._visualize_knn_test_results(results)

            return results

        except Exception as e:
            # 恢复原始配置
            self.config = original_config

            logger.error(f"测试KNN参数时出错: {str(e)}")
            logger.error(traceback.format_exc())
            return None

    # 在recommender.py中添加以下函数
    def cross_validate_models(self, k_folds=3):
        """使用交叉验证评估模型性能"""
        logger.info(f"Performing {k_folds}-fold cross validation...")

        # 确保数据已加载
        if not hasattr(self.data_processor, 'train_df') or self.data_processor.train_df is None:
            logger.error("No training data available for cross validation")
            return False

        # 获取所有用户ID
        all_users = self.data_processor.train_df['user_id'].unique()
        np.random.shuffle(all_users)

        # 划分用户为k个分组
        user_folds = np.array_split(all_users, k_folds)

        results = []
        for i in range(k_folds):
            logger.info(f"Cross validation fold {i + 1}/{k_folds}")

            # 选择当前fold的测试用户
            test_users = user_folds[i]

            # 训练模型
            self.train_models()

            # 评估模型
            metrics = self.evaluator.evaluate(
                model=self.hybrid_model,
                test_df=self.data_processor.test_df,
                test_users=test_users
            )

            if metrics:
                results.append(metrics)

        # 计算平均性能
        if results:
            avg_results = {}
            for metric in ['precision', 'recall', 'ndcg', 'diversity']:
                if all(metric in result for result in results):
                    avg_results[metric] = {}
                    for k in results[0][metric]:
                        avg_results[metric][k] = np.mean([result[metric][k] for result in results])

            if 'coverage' in results[0]:
                avg_results['coverage'] = np.mean([result['coverage'] for result in results])

            logger.info("Cross validation average results:")
            for metric in ['precision', 'recall', 'ndcg', 'diversity']:
                logger.info(f"{metric.capitalize()}:")
                for k, value in avg_results[metric].items():
                    logger.info(f"  @{k}: {value:.4f}")

            logger.info(f"Coverage: {avg_results['coverage']:.4f}")

            return avg_results

        return None

    def _visualize_knn_test_results(self, results):
        """可视化KNN测试结果

        Args:
            results (dict): 测试结果字典，键为参数组合，值为评估指标
        """
        if not results:
            logger.warning("没有KNN测试结果可视化")
            return

        try:
            # 确保可视化目录存在
            if not hasattr(self, 'visualizer'):
                from visualization.visualizer import RecommenderVisualizer
                self.visualizer = RecommenderVisualizer(output_dir='visualizations')

            # 提取参数组合和对应的NDCG@10值
            user_neighbors = []
            item_neighbors = []
            ndcg_values = []

            for config_key, metrics in results.items():
                # 解析配置键 "user_X_item_Y"
                parts = config_key.split('_')
                if len(parts) >= 4:
                    user_n = int(parts[1])
                    item_n = int(parts[3])
                    ndcg = metrics['ndcg'][10]  # 使用NDCG@10

                    user_neighbors.append(user_n)
                    item_neighbors.append(item_n)
                    ndcg_values.append(ndcg)

            # 创建热图数据
            import numpy as np
            import pandas as pd
            import matplotlib.pyplot as plt
            import seaborn as sns

            # 获取唯一的邻居值
            unique_user_n = sorted(set(user_neighbors))
            unique_item_n = sorted(set(item_neighbors))

            # 创建热图矩阵
            heatmap_data = np.zeros((len(unique_user_n), len(unique_item_n)))

            # 填充热图数据
            for u, i, ndcg in zip(user_neighbors, item_neighbors, ndcg_values):
                u_idx = unique_user_n.index(u)
                i_idx = unique_item_n.index(i)
                heatmap_data[u_idx, i_idx] = ndcg

            # 创建热图
            plt.figure(figsize=(12, 10))
            ax = sns.heatmap(
                heatmap_data,
                annot=True,
                fmt=".4f",
                cmap="viridis",
                xticklabels=unique_item_n,
                yticklabels=unique_user_n,
                cbar_kws={'label': 'NDCG@10'}
            )

            plt.title('KNN Parameter Optimization - NDCG@10', fontsize=16)
            plt.xlabel('Item Neighbors', fontsize=14)
            plt.ylabel('User Neighbors', fontsize=14)

            # 保存图表
            output_dir = self.visualizer.output_dir
            os.makedirs(output_dir, exist_ok=True)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'knn_parameter_optimization.png'), dpi=300, bbox_inches='tight')
            plt.close()

            # 创建折线图 - 用户邻居数量对性能的影响
            plt.figure(figsize=(12, 6))

            # 对每个item_n值绘制一条线
            for item_n in unique_item_n:
                item_ndcg = []
                for user_n in unique_user_n:
                    try:
                        config_key = f"user_{user_n}_item_{item_n}"
                        if config_key in results:
                            item_ndcg.append(results[config_key]['ndcg'][10])
                        else:
                            item_ndcg.append(None)  # 缺失数据
                    except:
                        item_ndcg.append(None)

                # 绘制线条，忽略缺失值
                valid_indices = [i for i, v in enumerate(item_ndcg) if v is not None]
                valid_user_n = [unique_user_n[i] for i in valid_indices]
                valid_ndcg = [item_ndcg[i] for i in valid_indices]

                if valid_ndcg:
                    plt.plot(valid_user_n, valid_ndcg, marker='o', label=f'Item Neighbors={item_n}')

            plt.title('Impact of User Neighbors on NDCG@10', fontsize=16)
            plt.xlabel('User Neighbors', fontsize=14)
            plt.ylabel('NDCG@10', fontsize=14)
            plt.grid(True, alpha=0.3)
            plt.legend()

            plt.savefig(os.path.join(output_dir, 'user_neighbors_impact.png'), dpi=300, bbox_inches='tight')
            plt.close()

            # 创建折线图 - 物品邻居数量对性能的影响
            plt.figure(figsize=(12, 6))

            # 对每个user_n值绘制一条线
            for user_n in unique_user_n:
                user_ndcg = []
                for item_n in unique_item_n:
                    try:
                        config_key = f"user_{user_n}_item_{item_n}"
                        if config_key in results:
                            user_ndcg.append(results[config_key]['ndcg'][10])
                        else:
                            user_ndcg.append(None)  # 缺失数据
                    except:
                        user_ndcg.append(None)

                # 绘制线条，忽略缺失值
                valid_indices = [i for i, v in enumerate(user_ndcg) if v is not None]
                valid_item_n = [unique_item_n[i] for i in valid_indices]
                valid_ndcg = [user_ndcg[i] for i in valid_indices]

                if valid_ndcg:
                    plt.plot(valid_item_n, valid_ndcg, marker='o', label=f'User Neighbors={user_n}')

            plt.title('Impact of Item Neighbors on NDCG@10', fontsize=16)
            plt.xlabel('Item Neighbors', fontsize=14)
            plt.ylabel('NDCG@10', fontsize=14)
            plt.grid(True, alpha=0.3)
            plt.legend()

            plt.savefig(os.path.join(output_dir, 'item_neighbors_impact.png'), dpi=300, bbox_inches='tight')
            plt.close()

            logger.info("KNN测试结果可视化已完成")

        except Exception as e:
            logger.error(f"KNN测试结果可视化错误: {str(e)}")
            logger.error(traceback.format_exc())