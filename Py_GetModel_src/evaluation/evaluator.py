#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Py_GetModel_src/evaluation/evaluator.py - Model evaluation module
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

    # 在Py_GetModel_src/evaluation/evaluator.py中

    def evaluate(self, model, test_df, test_users=None, k_values=None):
        """Evaluate model performance with improved user selection for sparse data"""
        logger.info("Starting model evaluation with sparse data optimization...")

        try:
            # 使用提供的k值或默认值
            k_values = k_values or self.k_values

            # 检查测试数据
            if test_df is None or len(test_df) == 0:
                logger.error("Empty test DataFrame provided")
                return None

            # 记录测试数据信息
            logger.info(f"Test DataFrame has {len(test_df)} rows and {test_df['user_id'].nunique()} unique users")

            # 改进的测试用户选择方法
            if test_users is None:
                # 选择可能的测试用户：有任何交互记录的用户
                all_possible_users = test_df['user_id'].unique()
                logger.info(f"Found {len(all_possible_users)} potential test users")

                # 首先尝试选择有正向交互的用户
                if 'is_recommended' in test_df.columns:
                    users_with_recommendations = test_df[test_df['is_recommended'] == True]['user_id'].unique()
                    logger.info(f"Found {len(users_with_recommendations)} users with recommended items")

                    # 如果有足够的用户，优先选择有推荐的用户
                    if len(users_with_recommendations) > 10:
                        sample_size = min(100, len(users_with_recommendations))
                        test_users = np.random.choice(users_with_recommendations, sample_size, replace=False)
                        logger.info(f"Selected {len(test_users)} test users with recommended items")
                    else:
                        # 加入部分没有推荐的用户以增加多样性
                        users_without_recommendations = np.setdiff1d(all_possible_users, users_with_recommendations)

                        # 确定用户选择比例
                        with_rec_count = min(len(users_with_recommendations), 50)
                        without_rec_count = min(100 - with_rec_count, len(users_without_recommendations))

                        # 合并两组用户
                        with_rec_users = np.random.choice(users_with_recommendations, with_rec_count,
                                                          replace=False) if with_rec_count > 0 else []
                        without_rec_users = np.random.choice(users_without_recommendations, without_rec_count,
                                                             replace=False) if without_rec_count > 0 else []

                        test_users = np.concatenate([with_rec_users, without_rec_users])
                        logger.info(
                            f"Selected mixed user set: {with_rec_count} with recommendations, {without_rec_count} without")
                else:
                    # 如果没有推荐字段，随机选择用户
                    sample_size = min(100, len(all_possible_users))
                    test_users = np.random.choice(all_possible_users, sample_size, replace=False)
                    logger.info(f"Selected {len(test_users)} random test users")

            # 确保我们有用户可以评估
            if len(test_users) == 0:
                logger.error("No users available for testing")
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

                    # 增加详细日志帮助调试
                    if not recommendations:
                        logger.debug(f"No recommendations generated for user {user_id}")
                        continue
                    elif len(recommendations) < max_k:
                        logger.debug(
                            f"Only {len(recommendations)} recommendations generated for user {user_id}, requested {max_k}")

                    recommended_items = [item_id for item_id, _ in recommendations]
                    successful_users += 1

                    # 记录第一个成功用户的推荐详情，帮助调试
                    if successful_users == 1:
                        logger.debug(f"First successful recommendation for user {user_id}: {recommendations[:3]}...")

                except Exception as e:
                    logger.error(f"Error getting recommendations for user {user_id}: {str(e)}")
                    logger.error(traceback.format_exc())
                    continue

                # 更新覆盖率跟踪
                all_recommended_items.update(recommended_items)

                # 计算每个k值的指标
                for k in k_values:
                    # 确保我们有足够的推荐
                    if len(recommendations) < k:
                        logger.debug(f"Only {len(recommendations)} recommendations available for k={k}")
                        continue

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
                    if k in results[metric]:
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

    # 在Py_GetModel_src/evaluation/evaluator.py中
    # 修改calculate_precision_recall方法

    def calculate_precision_recall(self, true_items, pred_items, k):
        """Calculate precision and recall at k with better sparse data handling

        Args:
            true_items (set): Set of relevant items
            pred_items (list): List of recommended items
            k (int): Recommendation list length

        Returns:
            tuple: (precision, recall)
        """
        # 确保输入是集合或可以转换为集合的对象
        if not true_items:
            true_items = set()  # 确保是空集而非None
        else:
            true_items = set(true_items)  # 转换为集合

        if not pred_items:
            pred_items = []  # 确保是空列表而非None

        # 实际使用的k值是max(len(pred_items), k)
        actual_k = min(len(pred_items), k)

        # 如果没有推荐，返回0
        if actual_k == 0:
            return 0, 0

        # 只考虑前k个推荐
        top_k_items = pred_items[:actual_k]

        # 计算命中数量
        n_hits = len(set(top_k_items) & true_items)

        # 计算precision
        precision = n_hits / actual_k if actual_k > 0 else 0

        # 计算recall
        recall = n_hits / len(true_items) if len(true_items) > 0 else 0

        return precision, recall


    # In Py_GetModel_src/evaluation/evaluator.py
    # Modify the calculate_ndcg method to better handle sparse data:

    def calculate_ndcg(self, true_items, pred_items, k):
        """Calculate NDCG at k with better handling of sparse data

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

        # Calculate DCG and IDCG manually to handle sparse data better
        dcg = 0
        idcg = 0

        for i, rel in enumerate(relevance):
            # Use log2(i+2) to avoid division by zero when i=0
            dcg += rel / np.log2(i + 2)

        for i, rel in enumerate(sorted(relevance, reverse=True)):
            idcg += rel / np.log2(i + 2)

        # Return NDCG, handling division by zero
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