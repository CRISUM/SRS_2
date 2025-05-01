# !/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Py_GetModel_src/models/content_model.py - Enhanced content-based recommendation model
Author: YourName
Date: 2025-04-29
Description: Implements content-based recommendation using item similarities
             and TF-IDF for tag processing to improve sparse data performance
"""
import gc

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
        """Train the model with provided data using memory-efficient optimization

        Args:
            data (DataFrame or dict): Training data

        Returns:
            self: Trained model
        """
        logger.info("Training optimized content-based model...")

        try:
            # Handle different input formats
            if isinstance(data, dict) and 'df' in data:
                df = data['df']
            elif isinstance(data, pd.DataFrame):
                df = data
            else:
                logger.error("Invalid input data format")
                return self

            # 1. 对大数据集进行采样
            original_len = len(df)
            if len(df) > 300000:
                logger.info(f"Large dataset detected ({original_len} rows), sampling to 300,000")
                df = df.sample(n=300000, random_state=42)

            # 2. 限制唯一物品数量
            unique_items = df['app_id'].nunique()
            logger.info(f"Dataset contains {unique_items} unique items")

            if unique_items > 10000:
                # 只保留最受欢迎的10000个物品
                popular_items = df['app_id'].value_counts().nlargest(10000).index
                df = df[df['app_id'].isin(popular_items)]
                logger.info(f"Limited to 10,000 most popular items, {len(df)} rows remaining")

            # 3. 优化处理游戏元数据
            logger.info("Processing game metadata with optimized method...")
            game_metadata = {}
            game_tags = {}

            # 使用更高效的批处理
            batch_size = 1000
            unique_items = df['app_id'].unique()

            for i in range(0, len(unique_items), batch_size):
                batch_items = unique_items[i:i + batch_size]
                batch_df = df[df['app_id'].isin(batch_items)].drop_duplicates('app_id')

                for _, row in batch_df.iterrows():
                    game_id = row['app_id']

                    # 创建简化的元数据
                    features = {'id': game_id}

                    # 添加标题
                    if 'title' in row and pd.notna(row['title']):
                        features['title'] = row['title']
                    else:
                        features['title'] = f"Game {game_id}"

                    # 处理标签 - 限制数量
                    if 'tags' in row and pd.notna(row['tags']) and row['tags']:
                        tags = [tag.strip() for tag in str(row['tags']).split(',')]
                        # 最多保留前15个标签
                        tags = tags[:15]
                        features['tags'] = tags
                        game_tags[game_id] = tags
                    else:
                        features['tags'] = []
                        game_tags[game_id] = []

                    # 添加其他可选特征
                    for col in ['price_final', 'win', 'mac', 'linux', 'rating', 'positive_ratio']:
                        if col in row and pd.notna(row[col]):
                            features[col] = row[col]

                    game_metadata[game_id] = features

                # 手动释放内存
                del batch_df
                gc.collect()

            self.item_metadata = game_metadata

            # 4. 高效构建相似度矩阵
            logger.info("Building similarity matrix with optimized method...")

            # 只处理有标签的游戏
            game_ids_with_tags = [gid for gid, tags in game_tags.items() if tags]

            # 如果标签过多，限制处理的游戏数量
            max_items_to_process = 8000
            if len(game_ids_with_tags) > max_items_to_process:
                logger.info(f"Too many games with tags ({len(game_ids_with_tags)}), limiting to {max_items_to_process}")
                # 保留标签数量最多的游戏
                games_by_tag_count = [(gid, len(game_tags[gid])) for gid in game_ids_with_tags]
                games_by_tag_count.sort(key=lambda x: x[1], reverse=True)
                game_ids_with_tags = [gid for gid, _ in games_by_tag_count[:max_items_to_process]]

            # 5. 使用更高效的相似度计算
            from joblib import Parallel, delayed

            # 定义用于并行处理的函数
            def compute_similarities_batch(game_id, other_game_ids, game_tags):
                similarities = []
                game1_tags = set(game_tags[game_id])

                for other_id in other_game_ids:
                    if game_id == other_id:
                        continue

                    game2_tags = set(game_tags[other_id])

                    # 快速计算Jaccard相似度
                    intersection = len(game1_tags.intersection(game2_tags))
                    union = len(game1_tags.union(game2_tags))

                    similarity = intersection / union if union > 0 else 0

                    if similarity > 0.1:  # 只保存相似度大于阈值的项
                        similarities.append((other_id, similarity))

                # 排序并只保留前K个最相似的项
                similarities.sort(key=lambda x: x[1], reverse=True)
                return game_id, similarities[:50]  # 最多保留前50个相似游戏

            # 并行计算相似度
            logger.info(f"Computing similarities for {len(game_ids_with_tags)} games...")

            # 限制核心数
            n_jobs = min(8, os.cpu_count() or 1)

            # 分批处理
            batch_size = 50
            similarity_results = {}

            for i in range(0, len(game_ids_with_tags), batch_size):
                batch = game_ids_with_tags[i:i + batch_size]
                logger.info(
                    f"Processing batch {i // batch_size + 1}/{(len(game_ids_with_tags) + batch_size - 1) // batch_size}")

                batch_results = Parallel(n_jobs=n_jobs, verbose=1)(
                    delayed(compute_similarities_batch)(game_id, game_ids_with_tags, game_tags)
                    for game_id in batch
                )

                # 将结果添加到字典中
                for game_id, similarities in batch_results:
                    similarity_results[game_id] = similarities

                # 手动释放内存
                gc.collect()

            self.similarity_matrix = similarity_results

            # 6. 计算流行度得分以用于冷启动，更轻量级计算
            if len(df) > 0:
                # 计算游戏人气
                game_counts = df['app_id'].value_counts()
                total_users = df['user_id'].nunique()

                # 计算带评分的热度
                popular_games = []
                for game_id, count in game_counts.items()[:100]:  # 只取前100个
                    # 计算流行度分数
                    popularity = count / total_users

                    # 如果有推荐信息，考虑正面评价
                    if 'is_recommended' in df.columns:
                        game_df = df[df['app_id'] == game_id]
                        if len(game_df) > 0:
                            positive_ratio = game_df['is_recommended'].mean()
                            score = (popularity * 0.7) + (positive_ratio * 0.3)
                        else:
                            score = popularity
                    else:
                        score = popularity

                    popular_games.append((game_id, score))

                # 按分数排序
                popular_games.sort(key=lambda x: x[1], reverse=True)
                self.popular_items = popular_games[:100]  # 只保留前100个

            # 7. 处理用户偏好，但限制处理的用户数量
            max_users_to_process = 50000
            user_count = df['user_id'].nunique()

            if user_count > max_users_to_process:
                logger.info(f"Too many users ({user_count}), limiting to {max_users_to_process} most active")
                active_users = df['user_id'].value_counts().nlargest(max_users_to_process).index
                user_df = df[df['user_id'].isin(active_users)]
            else:
                user_df = df

            # 处理用户偏好
            user_preferences = {}
            for user_id in user_df['user_id'].unique():
                user_data = user_df[user_df['user_id'] == user_id]

                # 简化处理: 只保留用户喜欢的游戏ID
                if 'is_recommended' in user_data.columns:
                    liked_items = user_data[user_data['is_recommended'] == True]['app_id'].tolist()
                elif 'hours' in user_data.columns:
                    # 使用游戏时长作为喜好指标
                    avg_hours = user_data['hours'].mean()
                    liked_items = user_data[user_data['hours'] >= avg_hours]['app_id'].tolist()
                else:
                    liked_items = user_data['app_id'].tolist()

                # 只保存有喜好的用户
                if liked_items:
                    user_preferences[user_id] = liked_items

            self.user_preferences = user_preferences

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
        """Save model to disk with optimization for large models

        Args:
            path (str): Directory path

        Returns:
            bool: Success
        """
        logger.info(f"Saving content-based model to {path}")

        try:
            os.makedirs(path, exist_ok=True)

            # 1. 优化相似度矩阵保存 - 分块保存
            logger.info("Saving similarity matrix in chunks...")
            similarity_chunks = {}
            chunk_size = 1000
            matrix_keys = list(self.similarity_matrix.keys())

            for i in range(0, len(matrix_keys), chunk_size):
                chunk_keys = matrix_keys[i:i + chunk_size]
                chunk_data = {k: self.similarity_matrix[k] for k in chunk_keys}

                # 保存这个分块
                chunk_path = os.path.join(path, f'content_similarity_chunk_{i // chunk_size}.pkl')
                with open(chunk_path, 'wb') as f:
                    pickle.dump(chunk_data, f, protocol=pickle.HIGHEST_PROTOCOL)

                # 记录这个分块包含的键
                similarity_chunks[f'chunk_{i // chunk_size}'] = chunk_keys

            # 保存分块信息
            with open(os.path.join(path, 'similarity_chunks.pkl'), 'wb') as f:
                pickle.dump(similarity_chunks, f, protocol=pickle.HIGHEST_PROTOCOL)

            # 2. 优化用户偏好保存 - 如果数据量大
            if len(self.user_preferences) > 10000:
                logger.info("Saving user preferences in chunks...")
                user_chunks = {}
                user_keys = list(self.user_preferences.keys())

                for i in range(0, len(user_keys), chunk_size):
                    chunk_keys = user_keys[i:i + chunk_size]
                    chunk_data = {k: self.user_preferences[k] for k in chunk_keys}

                    # 保存这个分块
                    chunk_path = os.path.join(path, f'user_preferences_chunk_{i // chunk_size}.pkl')
                    with open(chunk_path, 'wb') as f:
                        pickle.dump(chunk_data, f, protocol=pickle.HIGHEST_PROTOCOL)

                    # 记录这个分块包含的键
                    user_chunks[f'chunk_{i // chunk_size}'] = chunk_keys

                # 保存分块信息
                with open(os.path.join(path, 'user_preference_chunks.pkl'), 'wb') as f:
                    pickle.dump(user_chunks, f, protocol=pickle.HIGHEST_PROTOCOL)
            else:
                # 数据量小，直接保存整个字典
                with open(os.path.join(path, 'user_preferences.pkl'), 'wb') as f:
                    pickle.dump(self.user_preferences, f, protocol=pickle.HIGHEST_PROTOCOL)

            # 3. 优化物品元数据保存 - 压缩或分块
            if len(self.item_metadata) > 10000:
                logger.info("Saving item metadata in chunks...")
                metadata_chunks = {}
                metadata_keys = list(self.item_metadata.keys())

                for i in range(0, len(metadata_keys), chunk_size):
                    chunk_keys = metadata_keys[i:i + chunk_size]
                    chunk_data = {k: self.item_metadata[k] for k in chunk_keys}

                    # 保存这个分块
                    chunk_path = os.path.join(path, f'item_metadata_chunk_{i // chunk_size}.pkl')
                    with open(chunk_path, 'wb') as f:
                        pickle.dump(chunk_data, f, protocol=pickle.HIGHEST_PROTOCOL)

                    # 记录这个分块包含的键
                    metadata_chunks[f'chunk_{i // chunk_size}'] = chunk_keys

                # 保存分块信息
                with open(os.path.join(path, 'metadata_chunks.pkl'), 'wb') as f:
                    pickle.dump(metadata_chunks, f, protocol=pickle.HIGHEST_PROTOCOL)
            else:
                # 数据量小，直接保存
                with open(os.path.join(path, 'item_metadata.pkl'), 'wb') as f:
                    pickle.dump(self.item_metadata, f, protocol=pickle.HIGHEST_PROTOCOL)

            # 4. 保存热门物品列表 - 体积小，直接保存
            with open(os.path.join(path, 'popular_items.pkl'), 'wb') as f:
                pickle.dump(self.popular_items, f, protocol=pickle.HIGHEST_PROTOCOL)

            logger.info("Content-based model saved successfully with optimized method")
            return True
        except Exception as e:
            logger.error(f"Error saving content-based model: {str(e)}")
            logger.error(traceback.format_exc())
            return False

    def load(self, path):
        """Load model from disk with optimization for chunked files

        Args:
            path (str): Directory path

        Returns:
            self: Loaded model
        """
        logger.info(f"Loading content-based model from {path}")

        try:
            # 1. 加载相似度矩阵 - 处理分块
            similarity_chunks_path = os.path.join(path, 'similarity_chunks.pkl')
            if os.path.exists(similarity_chunks_path):
                logger.info("Loading chunked similarity matrix...")
                with open(similarity_chunks_path, 'rb') as f:
                    similarity_chunks = pickle.load(f)

                # 初始化空字典
                self.similarity_matrix = {}

                # 逐个加载分块
                for chunk_id, chunk_keys in similarity_chunks.items():
                    chunk_num = int(chunk_id.split('_')[1])
                    chunk_path = os.path.join(path, f'content_similarity_chunk_{chunk_num}.pkl')

                    if os.path.exists(chunk_path):
                        with open(chunk_path, 'rb') as f:
                            chunk_data = pickle.load(f)

                        # 合并到主字典
                        self.similarity_matrix.update(chunk_data)
                    else:
                        logger.warning(f"Missing chunk file: {chunk_path}")
            else:
                # 尝试加载单一文件
                single_path = os.path.join(path, 'content_similarity.pkl')
                if os.path.exists(single_path):
                    with open(single_path, 'rb') as f:
                        self.similarity_matrix = pickle.load(f)
                else:
                    logger.warning("No similarity matrix files found")
                    self.similarity_matrix = {}

            # 2. 加载用户偏好 - 处理分块
            user_chunks_path = os.path.join(path, 'user_preference_chunks.pkl')
            if os.path.exists(user_chunks_path):
                logger.info("Loading chunked user preferences...")
                with open(user_chunks_path, 'rb') as f:
                    user_chunks = pickle.load(f)

                # 初始化空字典
                self.user_preferences = {}

                # 逐个加载分块
                for chunk_id, chunk_keys in user_chunks.items():
                    chunk_num = int(chunk_id.split('_')[1])
                    chunk_path = os.path.join(path, f'user_preferences_chunk_{chunk_num}.pkl')

                    if os.path.exists(chunk_path):
                        with open(chunk_path, 'rb') as f:
                            chunk_data = pickle.load(f)

                        # 合并到主字典
                        self.user_preferences.update(chunk_data)
                    else:
                        logger.warning(f"Missing chunk file: {chunk_path}")
            else:
                # 尝试加载单一文件
                single_path = os.path.join(path, 'user_preferences.pkl')
                if os.path.exists(single_path):
                    with open(single_path, 'rb') as f:
                        self.user_preferences = pickle.load(f)
                else:
                    logger.warning("No user preferences files found")
                    self.user_preferences = {}

            # 3. 加载物品元数据 - 处理分块
            metadata_chunks_path = os.path.join(path, 'metadata_chunks.pkl')
            if os.path.exists(metadata_chunks_path):
                logger.info("Loading chunked item metadata...")
                with open(metadata_chunks_path, 'rb') as f:
                    metadata_chunks = pickle.load(f)

                # 初始化空字典
                self.item_metadata = {}

                # 逐个加载分块
                for chunk_id, chunk_keys in metadata_chunks.items():
                    chunk_num = int(chunk_id.split('_')[1])
                    chunk_path = os.path.join(path, f'item_metadata_chunk_{chunk_num}.pkl')

                    if os.path.exists(chunk_path):
                        with open(chunk_path, 'rb') as f:
                            chunk_data = pickle.load(f)

                        # 合并到主字典
                        self.item_metadata.update(chunk_data)
                    else:
                        logger.warning(f"Missing chunk file: {chunk_path}")
            else:
                # 尝试加载单一文件
                single_path = os.path.join(path, 'item_metadata.pkl')
                if os.path.exists(single_path):
                    with open(single_path, 'rb') as f:
                        self.item_metadata = pickle.load(f)
                else:
                    logger.warning("No item metadata files found")
                    self.item_metadata = {}

            # 4. 加载热门物品
            popular_items_path = os.path.join(path, 'popular_items.pkl')
            if os.path.exists(popular_items_path):
                with open(popular_items_path, 'rb') as f:
                    self.popular_items = pickle.load(f)
            else:
                logger.warning("No popular items file found")
                self.popular_items = []

            logger.info(
                f"Content-based model loaded successfully: {len(self.similarity_matrix)} items with similarities")
            return self
        except Exception as e:
            logger.error(f"Error loading content-based model: {str(e)}")
            logger.error(traceback.format_exc())
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
