#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
src/evaluation/evaluator.py - Model evaluation module
Author: YourName
Date: 2025-04-27
Description: Implements metrics and evaluation functions for recommendation models
"""

import numpy as np
import pandas as pd
import logging
import traceback
from tqdm import tqdm
from sklearn.metrics import ndcg_score

logger = logging.getLogger(__name__)


class RecommenderEvaluator:
    """Evaluator for recommendation models"""

    def __init__(self, k_values=None):
        """Initialize evaluator

        Args:
            k_values (list): List of k values for evaluation metrics
        """
        self.k_values = k_values or [5, 10, 20]
        self.results = None

    def evaluate(self, model, test_df, test_users=None, k_values=None):
        """Evaluate model performance"""
        logger.info("Starting model evaluation...")

        try:
            # 使用提供的k值或默认值
            k_values = k_values or self.k_values

            # 检查测试数据
            if test_df is None or len(test_df) == 0:
                logger.error("Empty test DataFrame provided")
                return None

            # 记录测试数据信息
            logger.info(f"Test DataFrame has {len(test_df)} rows and {test_df['user_id'].nunique()} unique users")

            # 简单直接的测试用户选择方法
            if test_users is None:
                # 获取所有有推荐项目的用户
                users_with_recommendations = test_df[test_df['is_recommended'] == True]['user_id'].unique()
                logger.info(f"Found {len(users_with_recommendations)} users with recommended items")

                # 如果用户数量足够，随机选择100个用户
                if len(users_with_recommendations) > 0:
                    sample_size = min(100, len(users_with_recommendations))
                    test_users = np.random.choice(users_with_recommendations, sample_size, replace=False)
                    logger.info(f"Selected {len(test_users)} test users randomly from users with recommendations")
                else:
                    logger.warning("No users with recommendations found in test data")
                    return None

            logger.info(f"Evaluating model with {len(test_users)} test users")

            # 初始化指标
            metrics = {
                'precision': {k: [] for k in k_values},
                'recall': {k: [] for k in k_values},
                'ndcg': {k: [] for k in k_values},
                'diversity': {k: [] for k in k_values},
                'coverage': []
            }

            # 跟踪所有推荐的物品，用于计算覆盖率
            all_recommended_items = set()
            all_items = set(test_df['app_id'].unique())

            # 记录成功评估的用户数
            successful_users = 0

            # 评估每个测试用户
            for user_id in tqdm(test_users, desc="Evaluating users"):
                # 获取用户的相关物品(正向交互)
                user_relevant_items = set(test_df[
                                              (test_df['user_id'] == user_id) &
                                              (test_df['is_recommended'] == True)
                                              ]['app_id'].values)

                # 跳过没有相关物品的用户
                if not user_relevant_items:
                    logger.debug(f"User {user_id} has no relevant items, skipping")
                    continue

                # 获取该用户的推荐
                max_k = max(k_values)
                try:
                    recommendations = model.recommend(user_id, max_k)
                    if not recommendations:
                        logger.warning(f"No recommendations generated for user {user_id}")
                        continue

                    recommended_items = [item_id for item_id, _ in recommendations]
                    successful_users += 1
                except Exception as e:
                    logger.error(f"Error getting recommendations for user {user_id}: {str(e)}")
                    continue

                # 更新覆盖率跟踪
                all_recommended_items.update(recommended_items)

                # 计算每个k值的指标
                for k in k_values:
                    # 获取top-k推荐
                    top_k_items = recommended_items[:k]

                    # 计算精确度和召回率
                    precision, recall = self.calculate_precision_recall(user_relevant_items, top_k_items, k)
                    metrics['precision'][k].append(precision)
                    metrics['recall'][k].append(recall)

                    # 计算NDCG
                    ndcg = self.calculate_ndcg(user_relevant_items, top_k_items, k)
                    metrics['ndcg'][k].append(ndcg)

                    # 如果有标签数据，计算多样性
                    diversity = self.calculate_diversity(top_k_items, test_df, k)
                    if diversity is not None:
                        metrics['diversity'][k].append(diversity)

            # 如果没有成功评估用户，返回None
            if successful_users == 0:
                logger.warning("No users could be successfully evaluated")
                return None

            # 计算覆盖率
            coverage = len(all_recommended_items) / len(all_items) if len(all_items) > 0 else 0
            metrics['coverage'] = coverage

            # 计算平均指标
            results = {}
            for metric in ['precision', 'recall', 'ndcg', 'diversity']:
                results[metric] = {k: np.mean(metrics[metric][k]) if metrics[metric][k] else 0 for k in k_values}

            results['coverage'] = metrics['coverage']
            results['successful_users'] = successful_users

            # 记录结果
            logger.info("Evaluation results:")
            for metric in ['precision', 'recall', 'ndcg', 'diversity']:
                logger.info(f"{metric.capitalize()}:")
                for k in k_values:
                    logger.info(f"  @{k}: {results[metric][k]:.4f}")

            logger.info(f"Coverage: {results['coverage']:.4f}")
            logger.info(f"Successfully evaluated {successful_users} out of {len(test_users)} users")

            # 存储结果
            self.results = results
            return results

        except Exception as e:
            logger.error(f"Error in model evaluation: {str(e)}")
            logger.error(traceback.format_exc())
            return None

    def calculate_precision_recall(self, true_items, pred_items, k):
        """Calculate precision and recall at k

        Args:
            true_items (set): Set of relevant items
            pred_items (list): List of recommended items
            k (int): Recommendation list length

        Returns:
            tuple: (precision, recall)
        """
        # Count hits (items that are both recommended and relevant)
        n_hits = len(set(pred_items) & true_items)

        # Calculate precision
        precision = n_hits / k if k > 0 else 0

        # Calculate recall
        recall = n_hits / len(true_items) if len(true_items) > 0 else 0

        return precision, recall

    def calculate_ndcg(self, true_items, pred_items, k):
        """Calculate NDCG at k

        Args:
            true_items (set): Set of relevant items
            pred_items (list): List of recommended items
            k (int): Recommendation list length

        Returns:
            float: NDCG score
        """
        # Create relevance array (1 if item is relevant, 0 otherwise)
        relevance = np.array([1 if item in true_items else 0 for item in pred_items])

        # If no relevant items in recommendations, return 0
        if sum(relevance) == 0:
            return 0

        # Create ideal relevance array
        ideal_relevance = np.zeros_like(relevance)
        ideal_relevance[:min(len(true_items), k)] = 1

        # Calculate NDCG
        try:
            return ndcg_score([ideal_relevance], [relevance])
        except:
            # Manually calculate NDCG if sklearn version fails
            # DCG = sum(rel_i / log2(i+1)) for i from 1 to k
            dcg = sum(rel / np.log2(i + 2) for i, rel in enumerate(relevance))

            # IDCG = sum(rel_i / log2(i+1)) for i from 1 to k (sorted)
            idcg = sum(rel / np.log2(i + 2) for i, rel in enumerate(sorted(relevance, reverse=True)))

            return dcg / idcg if idcg > 0 else 0

    def calculate_diversity(self, pred_items, data_df, k):
        """Calculate diversity at k

        Args:
            pred_items (list): List of recommended items
            data_df (DataFrame): Data containing item metadata
            k (int): Recommendation list length

        Returns:
            float: Diversity score (or None if tag data not available)
        """
        # Check if tags column exists
        if 'tags' not in data_df.columns:
            return None

        # Get tag data for recommended items
        item_tags = {}
        for item_id in pred_items:
            item_data = data_df[data_df['app_id'] == item_id]
            if len(item_data) > 0 and 'tags' in item_data.columns and pd.notna(item_data['tags'].iloc[0]):
                tags_str = item_data['tags'].iloc[0]
                item_tags[item_id] = set(tag.strip() for tag in tags_str.split(','))
            else:
                item_tags[item_id] = set()

        # Calculate pairwise Jaccard distance
        if len(item_tags) < 2:
            return 0  # Not enough items with tags

        n_pairs = 0
        sum_distances = 0

        for i, (item1, tags1) in enumerate(item_tags.items()):
            for item2, tags2 in list(item_tags.items())[i + 1:]:
                # Skip if either set is empty
                if not tags1 or not tags2:
                    continue

                # Calculate Jaccard similarity and distance
                intersection = len(tags1 & tags2)
                union = len(tags1 | tags2)

                if union > 0:
                    similarity = intersection / union
                    distance = 1 - similarity
                    sum_distances += distance
                    n_pairs += 1

        # Return average distance (higher means more diverse)
        if n_pairs > 0:
            return sum_distances / n_pairs
        else:
            return 0

    def calculate_coverage(self, all_pred_items, all_items):
        """Calculate catalog coverage

        Args:
            all_pred_items (set): Set of all recommended items
            all_items (set): Set of all available items

        Returns:
            float: Coverage ratio
        """
        # Coverage = |recommended items| / |all items|
        return len(all_pred_items) / len(all_items) if len(all_items) > 0 else 0