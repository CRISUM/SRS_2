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

from pandas._libs.parsers import defaultdict

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

    # In Py_GetModel_src/models/hybrid_model.py
    # Modify the recommend method to add a stronger diversity component:

    def recommend(self, user_id, n=10):
        """Generate top-N recommendations for a user with enhanced diversity

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
        model_success = {}

        # 从各个模型获取推荐
        all_recommendations = {}
        model_recommendations = {}
        model_success_count = 0

        # 用于多样性控制的物品标签集合
        item_tags = {}

        # 为多样性追踪各个模型提供的独特物品
        unique_items_by_model = {}

        # 获取推荐数量上限，用于多样性
        n_extended = n * 3  # 获取更多候选项用于多样性

        # 定义多样性因子 - 更高的值会增加多样性
        diversity_factor = 0.4  # 增加到0.4，比之前的0.3更高

        for name, model in self.models.items():
            try:
                # 获取每个模型的推荐
                recommendations = model.recommend(user_id, n_extended)

                # 记录是否成功
                model_success[name] = len(recommendations) > 0

                # 如果成功获得推荐，用户不是冷启动用户
                if recommendations and len(recommendations) > 0:
                    is_cold_start = False
                    model_success_count += 1

                model_recommendations[name] = recommendations

                # 跟踪此模型提供的独特物品
                unique_items_by_model[name] = {item_id for item_id, _ in recommendations}

                # 使用模型权重添加到所有推荐中
                weight = self.weights.get(name, 0.0)
                for item_id, score in recommendations:
                    if item_id not in all_recommendations:
                        all_recommendations[item_id] = 0.0

                    all_recommendations[item_id] += score * weight

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

        # 从每个模型中选择一些多样化推荐
        # 优先选择不同模型提供的物品，以增加多样性
        available_models = list(unique_items_by_model.keys())

        # 按权重排序模型，确保从高权重模型开始
        available_models.sort(key=lambda m: self.weights.get(m, 0), reverse=True)

        # 每个模型轮流贡献物品
        for name in available_models:
            # 跳过已经处理的模型
            if not unique_items_by_model[name]:
                continue

            # 从当前模型的物品中选择最具多样性的物品
            model_items = [(item_id, score) for item_id, score in model_recommendations[name]
                           if item_id not in selected_ids]

            if not model_items:
                continue

            # 计算每个物品的多样性分数
            diverse_items = []

            for item_id, base_score in model_items:
                # 计算与已选物品的多样性
                diversity_score = 1.0  # 默认完全多样

                if item_id in item_tags and selected_tags:
                    # 计算标签重叠
                    item_tag_set = item_tags[item_id]
                    overlap = len(item_tag_set.intersection(selected_tags))
                    if overlap > 0 and len(item_tag_set) > 0:
                        similarity = overlap / len(item_tag_set)
                        diversity_score = 1.0 - similarity

                # 计算调整后的分数
                adjusted_score = base_score * (1.0 + diversity_factor * diversity_score)
                diverse_items.append((item_id, adjusted_score))

            # 按调整后的分数排序
            diverse_items.sort(key=lambda x: x[1], reverse=True)

            # 添加这个模型的前2个多样性物品
            items_to_add = min(2, len(diverse_items))
            for i in range(items_to_add):
                item_id, score = diverse_items[i]
                if item_id not in selected_ids:
                    selected_items.append((item_id, score))
                    selected_ids.add(item_id)

                    # 更新标签集合
                    if item_id in item_tags:
                        selected_tags.update(item_tags[item_id])

                    # 如果已经有足够的推荐，停止添加
                    if len(selected_items) >= n:
                        break

            # 如果已经有足够的推荐，停止添加
            if len(selected_items) >= n:
                break

        # 如果仍然没有足够的推荐，从排序列表中添加更多
        remaining = n - len(selected_items)
        if remaining > 0:
            for item_id, score in sorted_items:
                if item_id not in selected_ids:
                    selected_items.append((item_id, score))
                    selected_ids.add(item_id)

                    remaining -= 1
                    if remaining <= 0:
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

    # In Py_GetModel_src/models/hybrid_model.py
    # Update the get_cold_start_recommendations method:

    # 修改Py_GetModel_src/models/hybrid_model.py中的get_cold_start_recommendations方法

    def get_cold_start_recommendations(self, user_id, n=10):
        """Enhanced cold start recommendations for extremely sparse data

        Args:
            user_id: User ID
            n (int): Number of recommendations

        Returns:
            list: List of (item_id, score) tuples
        """
        logger.info(f"Generating enhanced cold start recommendations for user {user_id}")

        recommendations = []

        try:
            # 1. 首先尝试使用流行度模型
            if 'popularity' in self.models:
                pop_model = self.models['popularity']
                pop_recs = pop_model.recommend(user_id, n)

                if pop_recs and len(pop_recs) > 0:
                    # 存储流行度推荐
                    recommendations.extend(pop_recs[:int(n / 2)])

            # 2. 从内容模型获取多样化推荐
            if 'content' in self.models and len(recommendations) < n:
                content_model = self.models['content']

                # 如果内容模型有item_metadata
                if hasattr(content_model, 'item_metadata') and content_model.item_metadata:
                    # 收集并分组所有标签
                    tag_items = defaultdict(list)

                    for item_id, metadata in content_model.item_metadata.items():
                        if 'tags' in metadata and metadata['tags']:
                            for tag in metadata['tags']:
                                tag_items[tag].append(item_id)

                    # 选择多样化的标签和游戏
                    selected_items = {item_id for item_id, _ in recommendations}

                    # 按标签流行度排序
                    popular_tags = sorted(tag_items.items(), key=lambda x: len(x[1]), reverse=True)

                    # 从每个流行标签中选择一个游戏
                    for tag, items in popular_tags[:min(10, len(popular_tags))]:
                        if len(recommendations) >= n:
                            break

                        # 尝试在这个标签中找到一个还未选择的游戏
                        for item_id in items:
                            if item_id not in selected_items:
                                # 寻找此游戏在流行游戏列表中的分数
                                score = 0.5  # 默认分数
                                for pop_id, pop_score in self.popular_items:
                                    if pop_id == item_id:
                                        score = pop_score
                                        break

                                recommendations.append((item_id, score))
                                selected_items.add(item_id)
                                break

            # 3. 如果仍需要更多推荐，添加物品KNN模型的推荐
            if 'item_knn' in self.models and len(recommendations) < n:
                item_knn = self.models['item_knn']

                # 从物品KNN模型获取推荐
                knn_recs = item_knn.recommend(user_id, n)

                if knn_recs and len(knn_recs) > 0:
                    # 添加KNN推荐，确保不重复
                    selected_ids = {item_id for item_id, _ in recommendations}

                    for item_id, score in knn_recs:
                        if item_id not in selected_ids and len(recommendations) < n:
                            recommendations.append((item_id, score))
                            selected_ids.add(item_id)

            # 4. 最后，确保我们有足够的推荐，如果需要，添加流行游戏
            if len(recommendations) < n:
                selected_ids = {item_id for item_id, _ in recommendations}

                for item_id, score in self.popular_items:
                    if item_id not in selected_ids:
                        recommendations.append((item_id, score))
                        selected_ids.add(item_id)

                        if len(recommendations) >= n:
                            break

            # 排序并返回
            recommendations.sort(key=lambda x: x[1], reverse=True)
            return recommendations[:n]

        except Exception as e:
            logger.error(f"Error in enhanced cold start: {str(e)}")
            logger.error(traceback.format_exc())
            # 始终返回自己的推荐或流行游戏
            if len(recommendations) >= n / 2:
                return recommendations[:n]
            else:
                return self.popular_items[:n]