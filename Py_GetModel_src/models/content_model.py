# !/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Py_GetModel_src/models/content_model.py - Enhanced content-based recommendation model
Author: YourName
Date: 2025-04-29
Description: Implements content-based recommendation using item similarities
             and TF-IDF for tag processing to improve sparse data performance
"""

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
        """Train the model with provided data

        Args:
            data (dict or DataFrame): Training data

        Returns:
            self: Trained model
        """
        logger.info("Training content-based model...")

        try:
            # Handle dict input format with pre-computed similarities
            if isinstance(data, dict):
                if 'similarity_matrix' in data:
                    self.similarity_matrix = data['similarity_matrix']
                elif 'embeddings' in data:
                    # Create similarity matrix from embeddings
                    embeddings = data['embeddings']
                    items = list(embeddings.keys())

                    # Create embedding matrix
                    matrix = np.array([embeddings[item_id] for item_id in items])

                    # Calculate similarity
                    sim_matrix = cosine_similarity(matrix)

                    # Convert to similarity dictionary
                    self.similarity_matrix = {}
                    for i, item_id in enumerate(items):
                        sims = [(items[j], sim_matrix[i, j]) for j in range(len(items)) if i != j]
                        sims.sort(key=lambda x: x[1], reverse=True)
                        self.similarity_matrix[item_id] = sims

                # Store user preferences if provided
                if 'user_preferences' in data:
                    self.user_preferences = data['user_preferences']

                # Store popular items if provided
                if 'popular_items' in data:
                    self.popular_items = data['popular_items']

                # Store item metadata if provided
                if 'item_metadata' in data:
                    self.item_metadata = data['item_metadata']

            # Process DataFrame input
            elif isinstance(data, pd.DataFrame):
                df = data

                # Extract game metadata
                game_metadata = {}
                for _, row in df.drop_duplicates('app_id').iterrows():
                    game_id = row['app_id']

                    # Build feature vector with available columns
                    features = {
                        'tags': row.get('tags', '').split(',') if isinstance(row.get('tags', ''), str) else [],
                        'title': row.get('title', f"Game {game_id}")
                    }

                    # Add optional features if available
                    for col in ['price_final', 'win', 'mac', 'linux', 'rating', 'positive_ratio', 'date_release']:
                        if col in row and not pd.isna(row[col]):
                            features[col] = row[col]

                    # 增强1: 添加描述特征（如果存在）
                    if 'description' in row and isinstance(row['description'], str) and len(row['description']) > 0:
                        # 提取描述中的关键词 (简单实现，仅提取长词)
                        words = [w.lower() for w in row['description'].split() if len(w) > 5]
                        features['description_keywords'] = words[:20]  # 限制关键词数量

                    # 增强2: 添加加权标签属性
                    if features['tags']:
                        weighted_tags = {}
                        for i, tag in enumerate(features['tags']):
                            tag = tag.strip()
                            if tag:
                                # 根据标签位置给予权重，前面的标签更重要
                                weight = 1.0 / (i + 1)
                                weighted_tags[tag] = weight
                        features['weighted_tags'] = weighted_tags

                    game_metadata[game_id] = features

                self.item_metadata = game_metadata

                # 增强3: 改进游戏相似度计算
                # Compute game similarities using TF-IDF on tags
                game_tags = {}
                for game_id, metadata in game_metadata.items():
                    # 使用标签和标题共同构建特征文本
                    tag_text = ' '.join([tag.strip() for tag in metadata['tags']]) if metadata['tags'] else ''
                    title_text = metadata.get('title', '')

                    # 增加标题中的词权重
                    if title_text:
                        title_words = ' '.join([word.lower() for word in title_text.split()])
                        # 标题重复两次增加权重
                        game_tags[game_id] = f"{tag_text} {title_words} {title_words}"
                    else:
                        game_tags[game_id] = tag_text

                # 增强4: 调整TF-IDF向量化参数
                # Use TF-IDF vectorizer with improved parameters
                vectorizer = TfidfVectorizer(
                    min_df=1,
                    max_df=0.8,  # 忽略在80%以上游戏中出现的常见词
                    ngram_range=(1, 2),  # 使用1-gram和2-gram
                    stop_words='english'  # 去除英文停用词
                )
                all_game_ids = list(game_tags.keys())
                tag_texts = [game_tags[gid] for gid in all_game_ids]

                # Check if we have enough data
                if len(tag_texts) > 1 and any(tag_texts):
                    tag_matrix = vectorizer.fit_transform(tag_texts)

                    # 增强5: 加入游戏属性进行相似度计算
                    # 如果有足够多的游戏有额外属性，可以合并到相似度计算中
                    has_extra_features = False
                    price_features = np.zeros((len(all_game_ids), 1))

                    for i, game_id in enumerate(all_game_ids):
                        if 'price_final' in game_metadata[game_id]:
                            price = float(game_metadata[game_id]['price_final'])
                            # 价格规范化到0-1范围
                            price_features[i, 0] = min(1.0, price / 60.0)  # 假设60是最高价格
                            has_extra_features = True

                    # 如果有额外特征，合并到标签矩阵中
                    if has_extra_features:
                        # 将价格特征与标签特征结合
                        # 这里需要将sparse矩阵转换为dense后才能合并
                        tag_dense = tag_matrix.toarray()

                        # 对两种特征进行加权合并 (90% 标签, 10% 价格)
                        combined_features = np.hstack([
                            tag_dense * 0.9,
                            price_features * 0.1
                        ])

                        # 计算相似度
                        tag_similarity = cosine_similarity(combined_features)
                    else:
                        # 使用原始标签矩阵计算相似度
                        tag_similarity = cosine_similarity(tag_matrix)

                    # Create game similarity dictionary
                    game_similarities = {}
                    for i, game_id in enumerate(all_game_ids):
                        similar_games = [(all_game_ids[j], tag_similarity[i, j])
                                         for j in range(len(all_game_ids)) if i != j]

                        # 增强6: 提高相似游戏的多样性
                        # 根据标签种类进行简单聚类
                        if len(similar_games) > 10 and 'tags' in game_metadata[game_id]:
                            diverse_similar = []
                            seen_tags = set()
                            source_tags = set(t.strip().lower() for t in game_metadata[game_id]['tags'])

                            # 首先基于分数排序
                            similar_games.sort(key=lambda x: x[1], reverse=True)

                            # 然后选择具有不同标签的游戏
                            for sim_game, sim_score in similar_games:
                                if sim_game in game_metadata and 'tags' in game_metadata[sim_game]:
                                    sim_tags = set(t.strip().lower() for t in game_metadata[sim_game]['tags'])

                                    # 找出此游戏的主要标签（与源游戏不同的标签）
                                    diff_tags = sim_tags - source_tags
                                    if not diff_tags:
                                        # 如果没有不同的标签，仍添加此游戏
                                        diverse_similar.append((sim_game, sim_score))
                                    else:
                                        # 检查是否有新的主要标签
                                        main_tag = list(diff_tags)[0] if diff_tags else None
                                        if main_tag and main_tag not in seen_tags:
                                            seen_tags.add(main_tag)
                                            # 增加分数以提高多样性游戏的排名
                                            diverse_similar.append((sim_game, sim_score * 1.1))
                                        else:
                                            diverse_similar.append((sim_game, sim_score))
                                else:
                                    diverse_similar.append((sim_game, sim_score))

                            # 重新排序
                            diverse_similar.sort(key=lambda x: x[1], reverse=True)
                            similar_games = diverse_similar
                        else:
                            similar_games.sort(key=lambda x: x[1], reverse=True)

                        game_similarities[game_id] = similar_games

                    self.similarity_matrix = game_similarities
                    logger.info(f"Created enhanced similarity matrix with {len(game_similarities)} games")
                else:
                    logger.warning("Not enough tag data to create similarity matrix")

                # Extract user preferences from interactions with improved weighting
                for user_id in df['user_id'].unique():
                    user_data = df[df['user_id'] == user_id]

                    # 增强7: 改进用户偏好提取
                    # 创建加权物品列表，根据不同信号给予物品不同权重
                    weighted_items = {}

                    # 处理显式推荐
                    if 'is_recommended' in user_data.columns:
                        for _, row in user_data.iterrows():
                            game_id = row['app_id']
                            if row['is_recommended'] == True:
                                # 基础权重 1.0
                                base_weight = 1.0

                                # 如果有游戏时间，增加权重
                                if 'hours' in row and pd.notna(row['hours']) and row['hours'] > 0:
                                    hours = float(row['hours'])
                                    # 时间权重: 最高到1.5 (30小时以上)
                                    time_factor = min(1.5, 1.0 + hours / 60.0)
                                    base_weight *= time_factor

                                weighted_items[game_id] = weighted_items.get(game_id, 0) + base_weight

                    # 处理评分
                    elif 'rating' in user_data.columns:
                        for _, row in user_data.iterrows():
                            game_id = row['app_id']
                            if pd.notna(row['rating']) and row['rating'] >= 7:
                                # 评分7-10的权重从1.0到2.0
                                rating_weight = 0.5 + (row['rating'] / 10)
                                weighted_items[game_id] = weighted_items.get(game_id, 0) + rating_weight

                    # 基于游戏时间
                    else:
                        try:
                            avg_hours = user_data['hours'].mean() if 'hours' in user_data else 0
                            for _, row in user_data.iterrows():
                                game_id = row['app_id']
                                if 'hours' in row and pd.notna(row['hours']):
                                    hours = row['hours']
                                    if hours > avg_hours:
                                        # 高于平均时间的游戏获得更高权重
                                        weighted_items[game_id] = hours / max(1.0, avg_hours)
                        except:
                            # 回退到简单列表
                            positive_items = user_data['app_id'].tolist()
                            self.user_preferences[user_id] = positive_items
                            continue  # 跳过后续处理

                    # 根据权重排序物品
                    sorted_items = sorted(weighted_items.items(), key=lambda x: x[1], reverse=True)

                    # 只保留权重最高的物品（去除异常值）
                    if len(sorted_items) > 2:
                        # 保留前75%的物品
                        cutoff = int(len(sorted_items) * 0.75)
                        positive_items = [item[0] for item in sorted_items[:max(cutoff, 2)]]
                    else:
                        positive_items = [item[0] for item in sorted_items]

                    self.user_preferences[user_id] = positive_items

                # 增强8: 改进热门物品计算
                # Calculate popular items for fallback with diversity
                if len(df) > 0:
                    # 除了流行度，还考虑好评率
                    game_ratings = {}

                    if 'is_recommended' in df.columns:
                        for game_id in df['app_id'].unique():
                            game_data = df[df['app_id'] == game_id]
                            if len(game_data) >= 3:  # 至少3个评价才计算
                                pos_ratio = game_data['is_recommended'].mean()
                                game_ratings[game_id] = pos_ratio

                    # Count games by frequency
                    game_counts = df['app_id'].value_counts()

                    # Normalize by total users
                    total_users = df['user_id'].nunique()

                    # 结合流行度和评分计算最终分数
                    popular_items = []
                    for game_id, count in game_counts.items():
                        pop_score = count / total_users

                        # 如果有评分，结合评分和流行度
                        if game_id in game_ratings:
                            rating_score = game_ratings[game_id]
                            final_score = (pop_score * 0.7) + (rating_score * 0.3)
                        else:
                            final_score = pop_score

                        popular_items.append((game_id, final_score))

                    # Sort by score
                    popular_items.sort(key=lambda x: x[1], reverse=True)

                    # 增强9: 添加多样性到热门物品中
                    if len(popular_items) > 100 and game_metadata:
                        # 按标签分组
                        tag_groups = {}

                        # 收集前200个流行游戏
                        top_games = popular_items[:200]

                        for game_id, score in top_games:
                            if game_id in game_metadata and 'tags' in game_metadata[game_id]:
                                for tag in game_metadata[game_id]['tags'][:3]:  # 使用前3个标签
                                    tag = tag.strip()
                                    if tag:
                                        if tag not in tag_groups:
                                            tag_groups[tag] = []
                                        tag_groups[tag].append((game_id, score))

                        # 从不同标签组中选择游戏
                        diverse_popular = []

                        # 按游戏数量排序标签
                        sorted_tags = sorted(tag_groups.items(), key=lambda x: len(x[1]), reverse=True)

                        # 从每个主要标签中选择顶级游戏
                        for tag, games in sorted_tags:
                            if games:
                                # 排序并选择该标签下最流行的游戏
                                games.sort(key=lambda x: x[1], reverse=True)
                                best_game = games[0]

                                # 如果游戏尚未添加，则添加
                                if best_game[0] not in [g[0] for g in diverse_popular]:
                                    diverse_popular.append(best_game)

                        # 如果多样性列表足够长，使用它
                        if len(diverse_popular) >= 50:
                            # 保留前100个游戏
                            popular_items = diverse_popular[:100]
                        else:
                            # 保留前100个原始流行游戏
                            popular_items = popular_items[:100]
                    else:
                        # Keep top 100
                        popular_items = popular_items[:100]

                    self.popular_items = popular_items

                logger.info(
                    f"Content-based model trained with {len(self.similarity_matrix)} items and enhanced features")
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
        """Save model to disk

        Args:
            path (str): Directory path

        Returns:
            bool: Success
        """
        logger.info(f"Saving content-based model to {path}")

        try:
            os.makedirs(path, exist_ok=True)

            # Save similarity matrix
            with open(os.path.join(path, 'content_similarity.pkl'), 'wb') as f:
                pickle.dump(self.similarity_matrix, f)

            # Save user preferences
            with open(os.path.join(path, 'user_preferences.pkl'), 'wb') as f:
                pickle.dump(self.user_preferences, f)

            # Save popular items
            with open(os.path.join(path, 'popular_items.pkl'), 'wb') as f:
                pickle.dump(self.popular_items, f)

            # Save item metadata
            with open(os.path.join(path, 'item_metadata.pkl'), 'wb') as f:
                pickle.dump(self.item_metadata, f)

            logger.info("Content-based model saved successfully")
            return True
        except Exception as e:
            logger.error(f"Error saving content-based model: {str(e)}")
            return False

    def load(self, path):
        """Load model from disk

        Args:
            path (str): Directory path

        Returns:
            self: Loaded model
        """
        logger.info(f"Loading content-based model from {path}")

        try:
            # Load similarity matrix
            with open(os.path.join(path, 'content_similarity.pkl'), 'rb') as f:
                self.similarity_matrix = pickle.load(f)

            # Load user preferences if available
            user_prefs_path = os.path.join(path, 'user_preferences.pkl')
            if os.path.exists(user_prefs_path):
                with open(user_prefs_path, 'rb') as f:
                    self.user_preferences = pickle.load(f)

            # Load popular items if available
            popular_items_path = os.path.join(path, 'popular_items.pkl')
            if os.path.exists(popular_items_path):
                with open(popular_items_path, 'rb') as f:
                    self.popular_items = pickle.load(f)

            # Load item metadata if available
            metadata_path = os.path.join(path, 'item_metadata.pkl')
            if os.path.exists(metadata_path):
                with open(metadata_path, 'rb') as f:
                    self.item_metadata = pickle.load(f)

            logger.info("Content-based model loaded successfully")
            return self
        except Exception as e:
            logger.error(f"Error loading content-based model: {str(e)}")
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
