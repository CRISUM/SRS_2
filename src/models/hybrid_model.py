#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
src/models/hybrid_model.py - Hybrid recommendation model
Author: YourName
Date: 2025-04-27
Description: Implements hybrid recommendation approach combining multiple models
"""

import numpy as np
import pandas as pd
import logging
import os
import pickle
import json

from .base_model import BaseRecommenderModel

logger = logging.getLogger(__name__)


class HybridRecommender(BaseRecommenderModel):
    """Hybrid recommender combining multiple recommendation models"""

    def __init__(self, models=None, weights=None):
        """Initialize hybrid recommender

        Args:
            models (dict): Dictionary of model name -> model instance
            weights (dict): Dictionary of model name -> weight
        """
        self.models = models or {}
        self.weights = weights or {}
        self.normalize_weights()

        # Cache for recommendations
        self.recommendation_cache = {}
        self.item_data = {}
        self.user_data = {}

        # Cold start handling
        self.popular_items = []

    def normalize_weights(self):
        """Normalize weights to sum to 1.0"""
        if not self.weights:
            return

        weight_sum = sum(self.weights.values())
        if weight_sum > 0:
            for key in self.weights:
                self.weights[key] = self.weights[key] / weight_sum

    def add_model(self, name, model, weight=1.0):
        """Add a model to the hybrid recommender

        Args:
            name (str): Model name
            model: Model instance
            weight (float): Weight for this model

        Returns:
            self: Updated hybrid recommender
        """
        self.models[name] = model
        self.weights[name] = weight
        self.normalize_weights()

        # Clear cache when adding a new model
        self.recommendation_cache = {}

        return self

    def fit(self, data):
        """Train all models with provided data

        Args:
            data (dict): Dictionary mapping model names to their training data

        Returns:
            self: Trained hybrid recommender
        """
        logger.info("Training hybrid recommender models...")

        if not isinstance(data, dict):
            raise ValueError("Data must be a dictionary mapping model names to their training data")

        # Train each model with its specific data
        for name, model in self.models.items():
            if name in data:
                logger.info(f"Training {name} model...")
                model.fit(data[name])
            else:
                logger.warning(f"No training data provided for {name} model")

        # Extract item and user data if available
        for name, model_data in data.items():
            if isinstance(model_data, dict):
                if 'item_data' in model_data:
                    self.item_data.update(model_data['item_data'])
                if 'user_data' in model_data:
                    self.user_data.update(model_data['user_data'])

        # Prepare popular items for cold start
        if 'popular_items' in data:
            self.popular_items = data['popular_items']

        logger.info("Hybrid recommender training completed")
        return self

    def predict(self, user_id, item_id):
        """Predict rating for a user-item pair

        Args:
            user_id: User ID
            item_id: Item ID

        Returns:
            float: Predicted rating (0-1 scale)
        """
        # Check if any models are available
        if not self.models:
            logger.warning("No models available for prediction")
            return 0.5

        # Get predictions from all models
        predictions = {}
        for name, model in self.models.items():
            try:
                prediction = model.predict(user_id, item_id)
                predictions[name] = prediction
            except Exception as e:
                logger.warning(f"Error getting prediction from {name} model: {str(e)}")
                predictions[name] = 0.5  # Default score

        # Calculate weighted average
        weighted_sum = 0.0
        weight_sum = 0.0

        for name, prediction in predictions.items():
            if name in self.weights:
                weight = self.weights[name]
                weighted_sum += prediction * weight
                weight_sum += weight

        # Return weighted average or default score
        if weight_sum > 0:
            return weighted_sum / weight_sum
        else:
            return 0.5

    def recommend(self, user_id, n=10):
        """Generate top-N recommendations for a user

        Args:
            user_id: User ID
            n (int): Number of recommendations

        Returns:
            list: List of (item_id, score) tuples
        """
        # 检查缓存
        cache_key = f"{user_id}_{n}"
        if cache_key in self.recommendation_cache:
            return self.recommendation_cache[cache_key]

        # 检查是否有可用模型
        if not self.models:
            logger.warning("No models available for recommendations")
            return self.popular_items[:n] if self.popular_items else []

        # 初始化标志，判断用户是否是冷启动用户
        is_cold_start = True

        # 从各个模型获取推荐
        all_recommendations = {}
        model_recommendations = {}
        model_success_count = 0

        # 增加这段: 获取用户的交互历史和时间权重
        user_time_weights = {}
        if user_id in self.user_data and 'interactions' in self.user_data[user_id]:
            interactions = self.user_data[user_id]['interactions']
            # 如果交互数据有时间信息，计算时间权重
            if any('date' in interaction for interaction in interactions):
                latest_date = max(pd.to_datetime(interaction['date'])
                                  for interaction in interactions if 'date' in interaction)

                for interaction in interactions:
                    if 'date' in interaction and 'app_id' in interaction:
                        days_ago = (latest_date - pd.to_datetime(interaction['date'])).days
                        # 应用时间衰减因子 (0.9^(days/30)) - 每30天衰减10%
                        decay_factor = self.config.get('time_decay_factor', 0.9)
                        time_weight = decay_factor ** (days_ago / 30)
                        user_time_weights[interaction['app_id']] = time_weight

        for name, model in self.models.items():
            try:
                recommendations = model.recommend(user_id, n * 3)  # 获取更多推荐以允许重叠

                # 如果成功获得推荐，用户不是冷启动用户
                if recommendations and len(recommendations) > 0:
                    is_cold_start = False
                    model_success_count += 1

                model_recommendations[name] = recommendations

                # 使用模型权重添加到所有推荐中
                weight = self.weights.get(name, 0.0)
                for item_id, score in recommendations:
                    if item_id not in all_recommendations:
                        all_recommendations[item_id] = 0.0

                    # 如果有时间权重信息，应用到分数上
                    time_weight_multiplier = user_time_weights.get(item_id, 1.0) if user_time_weights else 1.0
                    all_recommendations[item_id] += score * weight * time_weight_multiplier
            except Exception as e:
                logger.warning(f"Error getting recommendations from {name} model: {str(e)}")

        # 如果没有获得足够的推荐，使用冷启动策略
        if is_cold_start or model_success_count < 2 or not all_recommendations:
            logger.info(f"Using cold start strategy for user {user_id}")
            recommendations = self.get_cold_start_recommendations(user_id, n)
            self.recommendation_cache[cache_key] = recommendations
            return recommendations

        # 按得分降序排序
        sorted_items = sorted(all_recommendations.items(), key=lambda x: x[1], reverse=True)

        # 存入缓存
        recommendations = sorted_items[:n]
        self.recommendation_cache[cache_key] = recommendations

        return recommendations

    def update(self, new_data):
        """Update models with new data (incremental learning)

        Args:
            new_data (dict): Dictionary mapping model names to their update data

        Returns:
            self: Updated hybrid recommender
        """
        logger.info("Updating hybrid recommender models...")

        if not isinstance(new_data, dict):
            raise ValueError("New data must be a dictionary mapping model names to their update data")

        # Update each model with its specific data
        for name, model in self.models.items():
            if name in new_data:
                logger.info(f"Updating {name} model...")
                try:
                    model.update(new_data[name])
                except Exception as e:
                    logger.error(f"Error updating {name} model: {str(e)}")
            else:
                logger.debug(f"No update data provided for {name} model")

        # Update item and user data if available
        for name, model_data in new_data.items():
            if isinstance(model_data, dict):
                if 'item_data' in model_data:
                    self.item_data.update(model_data['item_data'])
                if 'user_data' in model_data:
                    self.user_data.update(model_data['user_data'])

        # Update popular items if provided
        if 'popular_items' in new_data:
            self.popular_items = new_data['popular_items']

        # Clear cache after update
        self.recommendation_cache = {}

        logger.info("Hybrid recommender update completed")
        return self

    def save(self, path):
        """Save hybrid recommender to disk

        Args:
            path (str): Directory path

        Returns:
            bool: Success
        """
        logger.info(f"Saving hybrid recommender to {path}")

        os.makedirs(path, exist_ok=True)

        try:
            # Save weights
            with open(os.path.join(path, 'hybrid_weights.json'), 'w') as f:
                json.dump(self.weights, f, indent=2)

            # Save model list
            model_list = list(self.models.keys())
            with open(os.path.join(path, 'hybrid_model_list.json'), 'w') as f:
                json.dump(model_list, f, indent=2)

            # Save popular items
            with open(os.path.join(path, 'popular_items.pkl'), 'wb') as f:
                pickle.dump(self.popular_items, f)

            # Save model data
            with open(os.path.join(path, 'hybrid_data.pkl'), 'wb') as f:
                pickle.dump({
                    'item_data': self.item_data,
                    'user_data': self.user_data
                }, f)

            # Create directory for each model
            for name, model in self.models.items():
                model_dir = os.path.join(path, name)
                os.makedirs(model_dir, exist_ok=True)
                try:
                    model.save(model_dir)
                except Exception as e:
                    logger.error(f"Error saving {name} model: {str(e)}")

            logger.info("Hybrid recommender saved successfully")
            return True
        except Exception as e:
            logger.error(f"Error saving hybrid recommender: {str(e)}")
            return False

    def load(self, path):
        """Load hybrid recommender from disk

        Args:
            path (str): Directory path

        Returns:
            self: Loaded hybrid recommender
        """
        logger.info(f"Loading hybrid recommender from {path}")

        try:
            # Load weights
            with open(os.path.join(path, 'hybrid_weights.json'), 'r') as f:
                self.weights = json.load(f)

            # Load model list
            with open(os.path.join(path, 'hybrid_model_list.json'), 'r') as f:
                model_list = json.load(f)

            # Load popular items if available
            try:
                with open(os.path.join(path, 'popular_items.pkl'), 'rb') as f:
                    self.popular_items = pickle.load(f)
            except FileNotFoundError:
                logger.warning("Popular items file not found")

            # Load model data if available
            try:
                with open(os.path.join(path, 'hybrid_data.pkl'), 'rb') as f:
                    data = pickle.load(f)
                    self.item_data = data.get('item_data', {})
                    self.user_data = data.get('user_data', {})
            except FileNotFoundError:
                logger.warning("Hybrid data file not found")

            # Models need to be loaded by the main system
            # We don't instantiate models here, just return the model list
            logger.info(f"Hybrid recommender loaded with weights for models: {list(self.weights.keys())}")
            return self, model_list
        except Exception as e:
            logger.error(f"Error loading hybrid recommender: {str(e)}")
            return None


    def get_cold_start_recommendations(self, user_id, n=10):
        """Generate recommendations for cold start users

        Args:
            user_id: User ID
            n (int): Number of recommendations

        Returns:
            list: List of (item_id, score) tuples
        """
        logger.info(f"Generating cold start recommendations for user {user_id}")

        # 检查是否有任何用户信息
        if user_id in self.user_data:
            user_info = self.user_data[user_id]

            # 如果有标签偏好，基于标签推荐
            if 'top_tags' in user_info and user_info['top_tags']:
                # 查找含有用户偏好标签的热门游戏
                tag_based_recs = {}

                # 对于每个热门游戏，检查标签匹配
                for game_id, score in self.popular_items:
                    if game_id in self.item_data and 'tags' in self.item_data[game_id]:
                        game_tags = self.item_data[game_id]['tags']

                        # 计算标签匹配度
                        matching_tags = set(game_tags) & set(user_info['top_tags'])
                        if matching_tags:
                            # 计算匹配分数
                            match_score = len(matching_tags) / len(user_info['top_tags'])
                            tag_based_recs[game_id] = score * (0.5 + 0.5 * match_score)

                # 排序并返回
                if tag_based_recs:
                    sorted_recs = sorted(tag_based_recs.items(), key=lambda x: x[1], reverse=True)
                    return sorted_recs[:n]

        # 如果没有可用的用户信息或标签推荐，返回热门物品
        return self.popular_items[:n]