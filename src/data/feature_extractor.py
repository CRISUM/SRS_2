#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
src/data/feature_extractor.py - Feature extraction utilities
Author: YourName
Date: 2025-04-27
Description: Specialized functions for extracting features from game and user data
"""
import traceback

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
import logging

logger = logging.getLogger(__name__)


class GameFeatureExtractor:
    """Extract features from game data"""

    def __init__(self, config=None):
        """Initialize feature extractor"""
        self.config = config or {}
        self.tfidf_vectorizer = None
        self.svd_model = None

    def extract_tag_features(self, df):
        """Extract features from game tags"""
        # Implement tag feature extraction

    def extract_text_features(self, df):
        """Extract features from game descriptions"""
        # Implement text feature extraction

    def create_game_embeddings(self, df, collaborative_factors=None):
        """Create game embeddings from content and optionally collaborative data"""
        logger.info("Creating game embeddings...")

        if df is None:
            logger.error("No training data available")
            return None

        try:
            # 初始化 game_embeddings 属性
            self.game_embeddings = {}

            # Extract tag features if available
            if 'tags' in df.columns:
                # Create tag-based embeddings
                from sklearn.feature_extraction.text import TfidfVectorizer

                # Group by app_id to get unique games
                games_df = df.drop_duplicates('app_id')[['app_id', 'tags']]

                # Prepare tag text
                games_df['tag_text'] = games_df['tags'].fillna('').apply(
                    lambda x: ' '.join([tag.strip() for tag in x.split(',')]) if isinstance(x, str) else ''
                )

                # Apply TF-IDF to tags
                self.tag_vectorizer = TfidfVectorizer(max_features=100)
                tag_features = self.tag_vectorizer.fit_transform(games_df['tag_text'])

                # Create tag embeddings
                tag_embedding_dim = min(self.config.get('tag_embedding_dim', 50), tag_features.shape[1])
                svd = TruncatedSVD(n_components=tag_embedding_dim, random_state=42)
                tag_embeddings = svd.fit_transform(tag_features)

                # Create embedding dictionary
                for i, app_id in enumerate(games_df['app_id']):
                    self.game_embeddings[app_id] = tag_embeddings[i]

                logger.info(f"Created tag-based embeddings for {len(self.game_embeddings)} games")

            # Enhance embeddings with collaborative information if SVD model is available
            if collaborative_factors is not None:
                # Logic for collaborative factors...
                logger.info(f"Enhanced embeddings with collaborative information")

            # Store number of games with embeddings
            if not hasattr(self, 'training_history'):
                self.training_history = {}
            self.training_history['games_with_embeddings'] = len(self.game_embeddings) if hasattr(self,
                                                                                                  'game_embeddings') else 0

            return self.game_embeddings

        except Exception as e:
            logger.error(f"Error creating game embeddings: {str(e)}")
            logger.error(traceback.format_exc())
            # 确保即使出错也返回一个空字典而不是None
            self.game_embeddings = {}
            return self.game_embeddings

    def get_similarity_matrix(self, embeddings):
        """Create item-item similarity matrix from embeddings"""
        if not embeddings or len(embeddings) == 0:
            logger.warning("No embeddings available to create similarity matrix")
            # 返回一个空对象而不是None，避免后续错误
            return {}

        try:
            # 创建物品相似度矩阵
            game_ids = list(embeddings.keys())
            n_games = len(game_ids)

            # 如果只有一个游戏，返回一个简单的字典
            if n_games <= 1:
                return {game_ids[0]: [(game_ids[0], 1.0)]} if n_games == 1 else {}

            # 创建嵌入向量矩阵
            embedding_matrix = np.zeros((n_games, len(next(iter(embeddings.values())))))
            for i, game_id in enumerate(game_ids):
                embedding_matrix[i] = embeddings[game_id]

            # 计算余弦相似度
            from sklearn.metrics.pairwise import cosine_similarity
            similarity_matrix = cosine_similarity(embedding_matrix)

            # 创建游戏ID到相似游戏的映射
            game_similarities = {}
            for i, game_id in enumerate(game_ids):
                # 获取相似度分数
                sims = [(game_ids[j], similarity_matrix[i, j]) for j in range(n_games) if i != j]
                # 按相似度排序
                sims.sort(key=lambda x: x[1], reverse=True)
                game_similarities[game_id] = sims

            return game_similarities
        except Exception as e:
            logger.error(f"Error creating similarity matrix: {str(e)}")
            logger.error(traceback.format_exc())
            return {}