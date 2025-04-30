#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Py_GetModel_src/models/hybrid_model.py - Hybrid recommendation model
Author: YourName
Date: 2025-04-27
Description: Implements hybrid recommendation approach combining multiple models
"""
import traceback

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
        if hasattr(self, 'enable_cache') and self.enable_cache and cache_key in self.recommendation_cache:
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

        # 为多样性追踪各个模型提供的独特物品
        unique_items_by_model = {}

        # 跟踪被推荐物品的来源模型
        item_source_models = {}

        # 增加这段: 获取用户的交互历史和时间权重
        user_time_weights = {}
        if hasattr(self, 'user_data') and user_id in self.user_data and 'interactions' in self.user_data[user_id]:
            interactions = self.user_data[user_id]['interactions']
            # 如果交互数据有时间信息，计算时间权重
            if any('date' in interaction for interaction in interactions):
                latest_date = max(pd.to_datetime(interaction['date'])
                                  for interaction in interactions if 'date' in interaction)

                for interaction in interactions:
                    if 'date' in interaction and 'app_id' in interaction:
                        days_ago = (latest_date - pd.to_datetime(interaction['date'])).days
                        # 应用时间衰减因子 (0.9^(days/30)) - 每30天衰减10%
                        decay_factor = self.config.get('time_decay_factor', 0.9) if hasattr(self, 'config') else 0.9
                        time_weight = decay_factor ** (days_ago / 30)
                        user_time_weights[interaction['app_id']] = time_weight

        # 用于多样性控制的物品标签集合
        item_tags = {}

        # 获取推荐数量上限，用于多样性
        n_extended = n * 3  # 获取更多候选项用于多样性

        for name, model in self.models.items():
            try:
                # 获取每个模型的推荐
                recommendations = model.recommend(user_id, n_extended)

                # 如果成功获得推荐，用户不是冷启动用户
                if recommendations and len(recommendations) > 0:
                    is_cold_start = False
                    model_success_count += 1

                model_recommendations[name] = recommendations

                # 跟踪此模型提供的独特物品
                unique_items_by_model[name] = {item_id for item_id, _ in recommendations}

                # 记录物品的来源模型
                for item_id, _ in recommendations:
                    if item_id not in item_source_models:
                        item_source_models[item_id] = []
                    item_source_models[item_id].append(name)

                # 使用模型权重添加到所有推荐中
                weight = self.weights.get(name, 0.0)
                for item_id, score in recommendations:
                    if item_id not in all_recommendations:
                        all_recommendations[item_id] = 0.0

                    # 如果有时间权重信息，应用到分数上
                    time_weight_multiplier = user_time_weights.get(item_id, 1.0) if user_time_weights else 1.0
                    all_recommendations[item_id] += score * weight * time_weight_multiplier

                    # 如果这是来自content模型的推荐，收集标签信息
                    if name == 'content' and hasattr(model, 'item_metadata'):
                        if item_id in model.item_metadata and 'tags' in model.item_metadata[item_id]:
                            item_tags[item_id] = set(model.item_metadata[item_id]['tags'])
            except Exception as e:
                logger.warning(f"Error getting recommendations from {name} model: {str(e)}")

        # 如果没有获得足够的推荐，使用冷启动策略
        if is_cold_start or model_success_count < 2 or not all_recommendations:
            logger.info(f"Using cold start strategy for user {user_id}")
            recommendations = self.get_cold_start_recommendations(user_id, n)

            # 存入缓存（如果启用）
            if hasattr(self, 'enable_cache') and self.enable_cache:
                self.recommendation_cache[cache_key] = recommendations

            return recommendations

        # 按得分降序排序
        sorted_items = sorted(all_recommendations.items(), key=lambda x: x[1], reverse=True)

        # 增强多样性和覆盖率的策略
        diversity_factor = getattr(self, 'diversity_factor', 0.3)  # 默认多样性因子

        # 1. 先选择一些顶级推荐
        selected_items = []
        selected_ids = set()
        selected_tags = set()

        # 取前1/4作为高置信度推荐
        top_k = max(1, int(n * 0.25))
        for i in range(min(top_k, len(sorted_items))):
            item_id, score = sorted_items[i]
            selected_items.append((item_id, score))
            selected_ids.add(item_id)

            # 更新已选标签集合
            if item_id in item_tags:
                selected_tags.update(item_tags[item_id])

        # 2. 然后选择来自不同模型的多样化推荐
        # 每个模型轮流贡献推荐，偏好那些标签集与已选项不同的
        remaining_slots = n - len(selected_items)

        # 如果有足够的剩余槽位，从每个模型中获取推荐
        if remaining_slots > 0 and model_recommendations:
            # 计算每个模型应该贡献的项目数（至少1个）
            models_with_recs = [name for name, recs in model_recommendations.items() if recs]
            items_per_model = max(1, remaining_slots // len(models_with_recs))

            # 轮流从每个模型中选择
            for name in models_with_recs:
                model_items = 0

                for item_id, score in model_recommendations[name]:
                    # 跳过已选物品
                    if item_id in selected_ids:
                        continue

                    # 计算多样性分数
                    diversity_score = 1.0
                    if item_id in item_tags and selected_tags:
                        # 计算与已选标签的重叠度
                        overlap = len(item_tags[item_id].intersection(selected_tags))
                        total = len(item_tags[item_id].union(selected_tags))
                        if total > 0:
                            similarity = overlap / total
                            diversity_score = 1.0 - similarity

                    # 调整分数以反映多样性
                    adjusted_score = score * (1.0 + diversity_factor * diversity_score)

                    # 添加到已选项
                    selected_items.append((item_id, adjusted_score))
                    selected_ids.add(item_id)

                    # 更新已选标签
                    if item_id in item_tags:
                        selected_tags.update(item_tags[item_id])

                    # 限制每个模型的贡献
                    model_items += 1
                    if model_items >= items_per_model or len(selected_items) >= n:
                        break

                # 检查是否已经选择了足够多的项目
                if len(selected_items) >= n:
                    break

        # 3. 如果仍然没有足够的推荐，从排序列表中添加更多
        while len(selected_items) < n and len(sorted_items) > len(selected_ids):
            for item_id, score in sorted_items:
                if item_id not in selected_ids:
                    selected_items.append((item_id, score))
                    selected_ids.add(item_id)

                    if len(selected_items) >= n:
                        break

        # 重新排序最终结果
        final_recommendations = sorted(selected_items, key=lambda x: x[1], reverse=True)

        # 存入缓存（如果启用）
        if hasattr(self, 'enable_cache') and self.enable_cache:
            self.recommendation_cache[cache_key] = final_recommendations[:n]

        return final_recommendations[:n]

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
        logger.info(f"Generating enhanced cold start recommendations for user {user_id}")

        try:
            # 1. 尝试基于有限用户信息构建推荐
            limited_info_available = False
            tag_based_recs = {}

            # 检查是否有基本用户信息
            if hasattr(self, 'user_data') and user_id in self.user_data:
                user_info = self.user_data[user_id]
                limited_info_available = True

                # 基于用户标签偏好推荐
                if 'top_tags' in user_info and user_info['top_tags']:
                    user_tags = set(user_info['top_tags'])

                    # 从content模型获取游戏标签信息
                    if 'content' in self.models:
                        content_model = self.models['content']

                        # 有两种可能的标签数据结构
                        game_tags = {}
                        if hasattr(content_model, 'game_tags'):
                            game_tags = content_model.game_tags
                        elif hasattr(content_model, 'item_metadata'):
                            # 从metadata中提取标签
                            for game_id, metadata in content_model.item_metadata.items():
                                if 'tags' in metadata:
                                    game_tags[game_id] = set(metadata['tags']) if isinstance(metadata['tags'],
                                                                                             list) else set()

                        # 基于标签匹配计算推荐
                        if game_tags:
                            for game_id, pop_score in self.popular_items:
                                if game_id in game_tags:
                                    game_tag_set = game_tags[game_id]

                                    # 计算标签交集
                                    matching_tags = user_tags.intersection(game_tag_set)

                                    if matching_tags:
                                        # 计算标签匹配比例
                                        match_ratio = len(matching_tags) / len(user_tags)
                                        # 结合流行度和标签匹配度
                                        tag_based_recs[game_id] = pop_score * (0.7 + 0.3 * match_ratio)

            # 2. 增加多样性推荐
            diverse_recs = {}

            # 从流行游戏中选择多样化的游戏
            if self.popular_items:
                # 按类别标签分组
                tag_to_games = {}

                # 收集标签信息
                for game_id, _ in self.popular_items:
                    game_tags = []

                    # 尝试从各种可能的来源获取标签
                    if 'content' in self.models:
                        content_model = self.models['content']
                        if hasattr(content_model, 'game_tags') and game_id in content_model.game_tags:
                            game_tags = content_model.game_tags[game_id]
                        elif hasattr(content_model, 'item_metadata') and game_id in content_model.item_metadata:
                            metadata = content_model.item_metadata[game_id]
                            if 'tags' in metadata:
                                game_tags = metadata['tags']

                    # 添加到标签映射
                    for tag in game_tags:
                        if tag not in tag_to_games:
                            tag_to_games[tag] = []
                        tag_to_games[tag].append(game_id)

                # 从每个主要标签中选择顶级游戏
                if tag_to_games:
                    # 按包含游戏数量排序的主要标签
                    sorted_tags = sorted(tag_to_games.items(), key=lambda x: len(x[1]), reverse=True)
                    top_tags = [tag for tag, _ in sorted_tags[:min(10, len(sorted_tags))]]

                    # 为每个顶级标签选择最流行的游戏
                    for tag in top_tags:
                        games = tag_to_games[tag]
                        # 找到该标签下最流行的游戏
                        best_game = None
                        best_score = 0

                        for game_id in games:
                            # 在流行物品列表中查找
                            for pop_game, pop_score in self.popular_items:
                                if pop_game == game_id and pop_score > best_score:
                                    best_game = game_id
                                    best_score = pop_score

                        if best_game and best_game not in diverse_recs:
                            diverse_recs[best_game] = best_score * 0.7  # 降低权重以平衡各种游戏

            # 3. 合并基于标签和多样性的推荐
            final_recs = {}

            # 先添加基于标签的推荐（如果有）
            if tag_based_recs:
                # 按分数排序
                sorted_tag_recs = sorted(tag_based_recs.items(), key=lambda x: x[1], reverse=True)
                # 添加前n/2个
                for i in range(min(n // 2, len(sorted_tag_recs))):
                    game_id, score = sorted_tag_recs[i]
                    final_recs[game_id] = score

            # 添加多样性推荐
            if diverse_recs:
                sorted_diverse = sorted(diverse_recs.items(), key=lambda x: x[1], reverse=True)
                for game_id, score in sorted_diverse:
                    if game_id not in final_recs:
                        final_recs[game_id] = score

                        # 当我们有足够多的推荐时停止
                        if len(final_recs) >= n:
                            break

            # 4. 如果仍然不够，添加流行游戏
            if len(final_recs) < n:
                for game_id, score in self.popular_items:
                    if game_id not in final_recs:
                        final_recs[game_id] = score

                        if len(final_recs) >= n:
                            break

            # 转换为列表格式并排序
            final_recommendations = sorted(final_recs.items(), key=lambda x: x[1], reverse=True)

            return final_recommendations[:n]

        except Exception as e:
            logger.error(f"Error generating cold start recommendations: {str(e)}")
            logger.error(traceback.format_exc())
            # 简单回退到流行游戏
            return self.popular_items[:n]
