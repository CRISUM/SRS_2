#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Steam游戏推荐系统核心模块
日期: 2025-04-24
描述: 基于LightGBM和序列行为的Steam游戏推荐系统
"""
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from collections import defaultdict, Counter
import pickle
import warnings
import logging
import os
import random
import gc
import json

# 设置随机种子，确保结果可复现
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_SEED)

# 设置警告和日志
warnings.filterwarnings('ignore')
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("steam_recommender.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# File: src/steam_recommender.py
# Core modifications for Steam Recommender

# New simple recommendation model class to replace LightGBM
class SimpleRecommenderModel:
    """A simple model to replace LightGBM for binary recommendation tasks"""

    def __init__(self, classifier='logistic'):
        """
        Initialize the simple recommender model

        Parameters:
            classifier (str): Type of classifier to use ('logistic', 'svm', etc.)
        """
        self.classifier_type = classifier
        self.model = None
        self.feature_names = None
        self.scaler = StandardScaler()

    def fit(self, X, y, categorical_features=None):
        """Train the model"""
        # Store feature names
        if isinstance(X, pd.DataFrame):
            self.feature_names = X.columns.tolist()
            X = X.values

        # Fit the scaler
        X_scaled = self.scaler.fit_transform(X)

        # Create and train the model
        if self.classifier_type == 'logistic':
            self.model = LogisticRegression(max_iter=1000, random_state=42, n_jobs=-1)
        else:
            # Default to logistic regression
            self.model = LogisticRegression(max_iter=1000, random_state=42, n_jobs=-1)

        self.model.fit(X_scaled, y)
        return self

    def predict(self, X):
        """Predict probabilities"""
        if isinstance(X, pd.DataFrame):
            if self.feature_names:
                # Ensure columns match
                X = X[self.feature_names]
            X = X.values

        # Handle 2D array shape for a single sample
        if len(X.shape) == 1:
            X = X.reshape(1, -1)

        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)[:, 1]

    def feature_importance(self, importance_type='weight'):
        """Get feature importance, mimicking LightGBM API"""
        if self.model is None or self.feature_names is None:
            return []

        if hasattr(self.model, 'coef_'):
            # For linear models, use coefficients as feature importance
            importances = np.abs(self.model.coef_[0])
            return importances

        # Default to equal importance if model doesn't expose coefficients
        return np.ones(len(self.feature_names))


# SVD model for collaborative filtering
class SVDModel:
    """SVD-based collaborative filtering model"""

    def __init__(self, n_components=50):
        """
        Initialize the SVD model

        Parameters:
            n_components (int): Number of latent factors
        """
        self.n_components = n_components
        self.svd = TruncatedSVD(n_components=n_components, random_state=42)
        self.user_factors = None
        self.item_factors = None
        self.user_map = {}
        self.item_map = {}
        self.global_mean = 0

    def fit(self, user_item_matrix):
        """Fit the SVD model to a user-item matrix"""
        # Store mapping from user/item IDs to matrix indices
        self.user_map = {user: i for i, user in enumerate(user_item_matrix.index)}
        self.item_map = {item: i for i, item in enumerate(user_item_matrix.columns)}

        # Calculate global mean rating
        self.global_mean = user_item_matrix.values.mean()

        # Center the data
        centered_matrix = user_item_matrix.values - self.global_mean

        # Fit SVD
        self.svd.fit(centered_matrix)

        # Get latent factors
        self.item_factors = self.svd.components_.T
        self.user_factors = centered_matrix @ self.item_factors / np.sqrt(self.svd.singular_values_)

        return self

    def predict(self, user_id, item_id):
        """Predict rating for a user-item pair"""
        if user_id not in self.user_map or item_id not in self.item_map:
            return self.global_mean

        user_idx = self.user_map[user_id]
        item_idx = self.item_map[item_id]

        user_vector = self.user_factors[user_idx]
        item_vector = self.item_factors[item_idx]

        # Prediction = global mean + user-item interaction
        prediction = self.global_mean + np.dot(user_vector, item_vector)

        # Normalize to 0-1 range
        return min(max(prediction / 10, 0), 1)  # Assuming ratings are 0-10


# Methods to add to the SteamRecommender class

def train_svd_model(self):
    """Train SVD model for collaborative filtering"""
    logger.info("Training SVD model for collaborative filtering...")

    # Create user-item matrix
    if not hasattr(self, 'train_df') or self.train_df is None:
        logger.error("No training data available")
        return None

    # Determine rating column
    if 'rating' in self.train_df.columns:
        rating_col = 'rating'
    elif 'is_recommended' in self.train_df.columns:
        # Convert boolean to numeric
        self.train_df['rating_value'] = self.train_df['is_recommended'].astype(int) * 10
        rating_col = 'rating_value'
    else:
        # Use hours as rating
        rating_col = 'hours'
        # Normalize hours to 0-10 scale to be consistent with ratings
        max_hours = self.train_df['hours'].max()
        if max_hours > 0:
            self.train_df['rating_value'] = self.train_df['hours'] * 10 / max_hours
            rating_col = 'rating_value'

    # Create user-item matrix
    user_item_matrix = pd.pivot_table(
        self.train_df,
        values=rating_col,
        index='user_id',
        columns='app_id',
        aggfunc='mean',
        fill_value=0
    )

    # Train SVD model
    self.svd_model = SVDModel(n_components=min(50, min(user_item_matrix.shape) - 1))
    self.svd_model.fit(user_item_matrix)

    logger.info("SVD model training completed")
    return self.svd_model


def predict_svd_score(self, user_id, game_id):
    """Predict rating using SVD model"""
    if not hasattr(self, 'svd_model') or self.svd_model is None:
        return 0.5

    try:
        return self.svd_model.predict(user_id, game_id)
    except Exception as e:
        logger.error(f"Error in SVD prediction: {str(e)}")
        return 0.5


def train_simple_model(self):
    """Train a simple classification model instead of LightGBM"""
    logger.info("Training simple recommendation model...")

    # Check if we have training data
    if not hasattr(self, 'train_df') or self.train_df is None:
        logger.error("No training data available")
        return None

    # Prepare features and target
    target_col = 'is_recommended'
    id_cols = ['user_id', 'app_id', 'date', 'review_id', 'prev_apps', 'prev_ratings', 'prev_hours']
    categorical_cols = [col for col in self.train_df.columns if col.endswith('_encoded')]

    # Remove non-feature columns
    exclude_cols = id_cols + [target_col, 'tags', 'description']

    # Select numerical and boolean features
    feature_cols = []
    for col in self.train_df.columns:
        if col in exclude_cols or (hasattr(self.train_df[col].iloc[0], '__iter__') and
                                   not isinstance(self.train_df[col].iloc[0], (str, bytes))):
            continue
        if self.train_df[col].dtype in [np.int64, np.float64, np.bool_]:
            feature_cols.append(col)

    # Filter out potential leakage features
    leakage_features = [
        'rating_new',
        'recommended_count_x', 'recommended_count_y',
        'recommendation_ratio_x', 'recommendation_ratio_y',
        'is_recommended_value', 'is_recommended_sum', 'pref_match'
    ]

    safe_feature_cols = [col for col in feature_cols if col not in leakage_features]
    logger.info(f"Using {len(safe_feature_cols)} features for simple model")

    # Prepare data
    X_train = self.train_df[safe_feature_cols]
    y_train = self.train_df[target_col].astype(int)

    # Train the model
    self.simple_model = SimpleRecommenderModel(classifier='logistic')
    self.simple_model.fit(X_train, y_train)

    # Get feature importance
    self.feature_importance = pd.DataFrame({
        'Feature': safe_feature_cols,
        'Importance': self.simple_model.feature_importance()
    }).sort_values(by='Importance', ascending=False)

    logger.info("Simple recommendation model training completed")
    return self.simple_model


def predict_simple_model_score(self, user_id, game_id):
    """Predict using simple model"""
    if not hasattr(self, 'simple_model') or self.simple_model is None:
        return 0.5

    try:
        # Extract features for prediction
        features = self.extract_prediction_features(user_id, game_id)

        # Make prediction
        return self.simple_model.predict(features)[0]
    except Exception as e:
        logger.error(f"Error in simple model prediction: {str(e)}")
        return 0.5


def predict_score(self, user_id, game_id):
    """Predict user's preference score for a game (hybrid approach)"""
    # Use cache if available
    cache_key = f"{user_id}_{game_id}"
    if cache_key in self.score_cache:
        return self.score_cache[cache_key]

    try:
        # Get scores from different models
        user_knn_score = self.predict_user_knn_score(user_id, game_id)
        item_knn_score = self.predict_item_knn_score(user_id, game_id)
        svd_score = self.predict_svd_score(user_id, game_id)
        content_score = self.predict_content_score(user_id, game_id)
        sequence_score = self.predict_sequence_score(user_id, game_id)

        # New weights for the hybrid model
        weights = {
            'user_knn': 0.25,
            'item_knn': 0.25,
            'svd': 0.2,
            'content': 0.15,
            'sequence': 0.15
        }

        # Calculate weighted average
        final_score = (
                weights['user_knn'] * user_knn_score +
                weights['item_knn'] * item_knn_score +
                weights['svd'] * svd_score +
                weights['content'] * content_score +
                weights['sequence'] * sequence_score
        )

        # Cache and return result
        self.score_cache[cache_key] = final_score
        return final_score
    except Exception as e:
        logger.error(f"Error predicting score: {str(e)}")
        return 0.5


def update_simple_model(self, new_data_df):
    """Incrementally update the simple model with new data"""
    logger.info("Incrementally updating simple model...")

    if not hasattr(self, 'simple_model') or self.simple_model is None:
        logger.warning("Simple model doesn't exist, will train from scratch")
        self.train_simple_model()
        return

    try:
        # Update with new data
        # Merge new data into existing data
        current_df = self.df.copy() if hasattr(self, 'df') and self.df is not None else pd.DataFrame()

        # Add new data
        for idx, row in new_data_df.iterrows():
            user_id = row['user_id']
            app_id = row['app_id']

            # Check if interaction exists
            mask = (current_df['user_id'] == user_id) & (current_df['app_id'] == app_id)
            if sum(mask) > 0:
                # Update existing row
                for col in new_data_df.columns:
                    if col in current_df.columns:
                        current_df.loc[mask, col] = row[col]
            else:
                # Add new row
                current_df = pd.concat([current_df, pd.DataFrame([row])])

        # Update df
        self.df = current_df

        # Re-engineer features
        self.engineer_features()

        # Retrain the model
        self.train_simple_model()

    except Exception as e:
        logger.error(f"Error updating simple model: {str(e)}")


def update_svd_model(self, new_data_df):
    """Incrementally update the SVD model with new data"""
    logger.info("Incrementally updating SVD model...")

    if not hasattr(self, 'svd_model') or self.svd_model is None:
        logger.warning("SVD model doesn't exist, will train from scratch")
        self.train_svd_model()
        return

    try:
        # For SVD, it's usually more effective to retrain from scratch
        # Merge new data
        current_df = self.df.copy() if hasattr(self, 'df') and self.df is not None else pd.DataFrame()

        # Add new data
        for idx, row in new_data_df.iterrows():
            user_id = row['user_id']
            app_id = row['app_id']

            # Check if interaction exists
            mask = (current_df['user_id'] == user_id) & (current_df['app_id'] == app_id)
            if sum(mask) > 0:
                # Update existing row
                for col in new_data_df.columns:
                    if col in current_df.columns:
                        current_df.loc[mask, col] = row[col]
            else:
                # Add new row
                current_df = pd.concat([current_df, pd.DataFrame([row])])

        # Update df
        self.df = current_df

        # Re-train SVD model
        self.train_svd_model()

    except Exception as e:
        logger.error(f"Error updating SVD model: {str(e)}")


def incremental_update(self, interactions_df, games_df=None, users_df=None):
    """Perform incremental update of all models"""
    logger.info("Starting comprehensive incremental update...")

    try:
        # Update KNN model
        if len(interactions_df) > 0:
            self.update_knn_model(interactions_df)

        # Update SVD model
        if len(interactions_df) > 0:
            self.update_svd_model(interactions_df)

        # Update simple model
        if len(interactions_df) > 0:
            self.update_simple_model(interactions_df)

        # Update sequence model
        if len(interactions_df) > 0 and hasattr(self, 'update_sequence_model'):
            self.update_sequence_model(interactions_df)

        # Update content model
        if games_df is not None and len(games_df) > 0 and hasattr(self, 'update_content_model'):
            self.update_content_model(games_df)

        # Update game embeddings
        if hasattr(self, 'create_game_embeddings'):
            self.create_game_embeddings()

        # Clear caches
        self.recommendation_cache = {}
        self.score_cache = {}
        self.feature_cache = {}

        logger.info("Incremental update completed")

    except Exception as e:
        logger.error(f"Error in incremental update: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())

class SteamRecommender:
    """Steam游戏推荐系统的主类，整合了数据处理、特征工程、模型训练和评估等功能"""

    def __init__(self, data_path, config=None):
        """
        初始化推荐系统

        参数:
            data_path (str): 数据文件路径
            config (dict, optional): 配置参数字典
        """
        self.data_path = data_path
        # 默认配置
        self.config = {
            'lgbm_params': {
                'objective': 'binary',
                'metric': 'auc',
                'boosting_type': 'gbdt',
                'learning_rate': 0.05,
                'num_leaves': 31,
                'max_depth': -1,
                'min_child_samples': 20,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'random_state': RANDOM_SEED,
                'n_estimators': 1000,
                'early_stopping_rounds': 50,
                'verbose': -1
            },
            'sequence_params': {
                'embedding_dim': 64,
                'hidden_dim': 128,
                'num_layers': 2,
                'dropout': 0.2,
                'batch_size': 64,
                'learning_rate': 0.001,
                'epochs': 10
            },
            'tag_embedding_dim': 50,
            'text_embedding_dim': 100,
            'max_seq_length': 20,
            'time_decay_factor': 0.9,
            'n_recommendations': 10,
            'content_weight': 0.3,
            'sequence_weight': 0.3,
            'lgbm_weight': 0.4,
            'use_gpu': torch.cuda.is_available()
        }

        # 更新配置（如果提供了自定义配置）
        if config:
            self.config.update(config)

        # 初始化模型和编码器
        self.lgbm_model = None
        self.sequence_model = None
        self.label_encoders = {}
        self.tfidf_model = None
        self.tfidf_svd = None
        self.tag_vectorizer = None
        self.game_embeddings = None
        self.user_embeddings = None
        self.device = torch.device('cuda' if self.config['use_gpu'] and torch.cuda.is_available() else 'cpu')

        # 在这里添加缓存字典
        self.feature_cache = {}  # 用于缓存特征
        self.recommendation_cache = {}  # 用于缓存推荐结果
        self.score_cache = {}  # 用于缓存预测分数

        logger.info(f"初始化推荐系统完成，使用设备: {self.device}")

    def load_data(self):
        """使用分块加载数据"""
        logger.info(f"开始分块加载数据: {self.data_path}")

        # 首先读取头部来获取列名
        try:
            header_df = pd.read_csv(self.data_path, nrows=1)
            column_names = header_df.columns.tolist()
            logger.info(f"数据列名: {column_names}")

            # 检查所需的列是否存在
            required_cols = ['user_id', 'app_id', 'is_recommended', 'hours', 'title', 'tags']
            missing_cols = [col for col in required_cols if col not in column_names]
            if missing_cols:
                logger.error(f"CSV文件缺少必要的列: {missing_cols}")

                # 查看可能的替代列名（处理大小写不敏感的情况）
                column_lower = [col.lower() for col in column_names]
                for missing in missing_cols:
                    possible_matches = [column_names[i] for i, col in enumerate(column_lower) if col == missing.lower()]
                    if possible_matches:
                        logger.info(f"'{missing}'可能的匹配列: {possible_matches}")

                return False
        except Exception as e:
            logger.error(f"读取CSV文件头部时出错: {str(e)}")
            return False

        # 估算数据大小
        chunk_size = 500000  # 每块约250-300MB

        # 初始化统计变量
        unique_users = set()
        unique_games = set()

        # 创建用户和游戏的基本统计信息字典
        user_stats = {}
        game_stats = {}

        # 分块处理
        chunk_count = 0

        try:
            for chunk in pd.read_csv(self.data_path, chunksize=chunk_size):
                chunk_count += 1
                logger.info(f"处理数据块 {chunk_count}")

                # 确保必要的列存在
                for col in required_cols:
                    if col not in chunk.columns:
                        logger.warning(f"数据块 {chunk_count} 中缺少列 '{col}'，跳过此块")
                        continue

                # 更新基本统计信息
                unique_users.update(chunk['user_id'].unique())
                unique_games.update(chunk['app_id'].unique())

                # 处理用户统计信息
                for _, row in chunk.iterrows():
                    user_id = row['user_id']
                    app_id = row['app_id']

                    # 处理可能的缺失值
                    try:
                        is_recommended = row['is_recommended']
                        hours = float(row['hours']) if not pd.isna(row['hours']) else 0.0
                    except (KeyError, ValueError):
                        is_recommended = False
                        hours = 0.0

                    # 更新用户统计
                    if user_id not in user_stats:
                        user_stats[user_id] = {'game_count': 0, 'total_hours': 0, 'recommended_count': 0}
                    user_stats[user_id]['game_count'] += 1
                    user_stats[user_id]['total_hours'] += hours
                    if is_recommended:
                        user_stats[user_id]['recommended_count'] += 1

                    # 更新游戏统计
                    if app_id not in game_stats:
                        title = row.get('title', f"Unknown Game {app_id}")
                        tags = row.get('tags', "")

                        game_stats[app_id] = {
                            'user_count': 0,
                            'total_hours': 0,
                            'recommended_count': 0,
                            'title': title,
                            'tags': tags
                        }
                    game_stats[app_id]['user_count'] += 1
                    game_stats[app_id]['total_hours'] += hours
                    if is_recommended:
                        game_stats[app_id]['recommended_count'] += 1

                # 释放内存
                del chunk
                import gc
                gc.collect()

            logger.info(f"处理了 {chunk_count} 个数据块")

            # 将统计信息转换为DataFrame
            self.user_df = pd.DataFrame.from_dict(user_stats, orient='index')
            self.user_df.reset_index(inplace=True)
            self.user_df.rename(columns={'index': 'user_id'}, inplace=True)

            self.game_df = pd.DataFrame.from_dict(game_stats, orient='index')
            self.game_df.reset_index(inplace=True)
            self.game_df.rename(columns={'index': 'app_id'}, inplace=True)

            # 计算额外的统计信息
            self.user_df['recommendation_ratio'] = self.user_df['recommended_count'] / self.user_df['game_count']
            self.game_df['recommendation_ratio'] = self.game_df['recommended_count'] / self.game_df['user_count']
            self.game_df['avg_hours'] = self.game_df['total_hours'] / self.game_df['user_count']

            # 采样数据用于训练
            self._create_training_sample()

            logger.info(f"数据加载完成，发现 {len(unique_users)} 个独立用户和 {len(unique_games)} 个游戏")
            return True

        except Exception as e:
            logger.error(f"处理数据时出错: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return False

    def train_content_model(self):
        """训练基于内容的模型（用于处理冷启动问题）"""
        logger.info("开始训练基于内容的模型...")

        # 为测试集中的新游戏创建基于内容的相似度矩阵
        # 这主要基于游戏标签和描述

        # 检查是否有标签数据
        if not any(col.startswith('top_tag_') for col in self.train_df.columns):
            logger.warning("没有标签数据，跳过基于内容的模型训练")
            return None

        # 提取数值型的标签特征
        tag_cols = []
        for col in self.train_df.columns:
            if col in self.test_df.columns and not col.startswith(('user_id', 'app_id', 'prev_')):
                # 只选择数值型特征
                if self.train_df[col].dtype in [np.int64, np.float64, np.bool_]:
                    tag_cols.append(col)

        logger.info(f"使用 {len(tag_cols)} 个数值型特征进行内容模型训练")

        if len(tag_cols) == 0:
            logger.warning("没有找到合适的数值型特征，跳过内容模型训练")
            return None

        # 按游戏聚合标签特征
        game_features = self.train_df.groupby('app_id')[tag_cols].mean().reset_index()

        # 计算游戏间的余弦相似度
        from sklearn.metrics.pairwise import cosine_similarity
        from scipy.sparse import csr_matrix

        # 创建特征矩阵
        game_matrix = csr_matrix(game_features[tag_cols].values)

        # 计算相似度矩阵
        similarity_matrix = cosine_similarity(game_matrix)

        # 创建游戏ID到索引的映射
        game_idx = {game_id: idx for idx, game_id in enumerate(game_features['app_id'])}

        # 存储相似度矩阵
        self.content_similarity = {
            'matrix': similarity_matrix,
            'game_idx': game_idx
        }

        logger.info("基于内容的模型训练完成")
        return self.content_similarity


    def predict_score(self, user_id, game_id):
        """预测用户对游戏的评分（KNN混合方式）"""
        # 使用缓存
        cache_key = f"{user_id}_{game_id}"
        if cache_key in self.score_cache:
            return self.score_cache[cache_key]

        try:
            # 计算用户KNN得分
            user_knn_score = self.predict_user_knn_score(user_id, game_id)

            # 计算物品KNN得分
            item_knn_score = self.predict_item_knn_score(user_id, game_id)

            # 计算序列模型得分
            seq_score = self.predict_sequence_score(user_id, game_id)

            # 计算内容模型得分
            content_score = self.predict_content_score(user_id, game_id)

            # 分配权重 (调整为适合KNN的新权重)
            user_knn_weight = 0.3
            item_knn_weight = 0.3
            sequence_weight = 0.2
            content_weight = 0.2

            # 加权平均
            final_score = (
                    user_knn_weight * user_knn_score +
                    item_knn_weight * item_knn_score +
                    sequence_weight * seq_score +
                    content_weight * content_score
            )

            # 缓存结果
            self.score_cache[cache_key] = final_score

            return final_score
        except Exception as e:
            logger.error(f"预测分数时出错: {str(e)}")
            # 出错时返回默认分数
            return 0.5

    def extract_prediction_features(self, user_id, game_id):
        """提取用于预测的特征"""
        # 使用缓存
        cache_key = f"{user_id}_{game_id}"
        if cache_key in self.feature_cache:
            return self.feature_cache[cache_key]

        # 如果模型存在，获取模型所需的特征名称
        if self.lgbm_model is not None:
            feature_names = self.lgbm_model.feature_name()

            # 创建一个全零的特征数组（与模型特征数量相同）
            features = np.zeros(len(feature_names))

            # 获取用户特征和游戏特征
            user_data = None
            if user_id in self.train_df['user_id'].values:
                user_data = self.train_df[self.train_df['user_id'] == user_id]
            elif user_id in self.test_df['user_id'].values:
                user_data = self.test_df[self.test_df['user_id'] == user_id]

            game_data = None
            if game_id in self.train_df['app_id'].values:
                game_data = self.train_df[self.train_df['app_id'] == game_id]
            elif game_id in self.test_df['app_id'].values:
                game_data = self.test_df[self.test_df['app_id'] == game_id]

            # 如果找到了用户和游戏数据，尝试提取特征
            if user_data is not None and game_data is not None and len(user_data) > 0 and len(game_data) > 0:
                # 为了简单起见，我们取第一条用户和游戏数据
                user_features = user_data.iloc[0]
                game_features = game_data.iloc[0]

                # 设法提取出所有有效的特征列
                valid_cols = []
                for col in self.train_df.columns:
                    try:
                        # 尝试获取一个样本值来检查它是不是列表类型
                        val = self.train_df[col].iloc[0]
                        if not isinstance(val, (list, dict, set)) and col not in ['user_id', 'app_id', 'date',
                                                                                  'review_id', 'is_recommended',
                                                                                  'prev_apps', 'prev_ratings',
                                                                                  'prev_hours', 'description', 'tags']:
                            valid_cols.append(col)
                    except:
                        continue

                # 尝试构建一个特征字典
                feature_dict = {}
                for col in valid_cols:
                    # 尝试获取特征值
                    if col in user_features:
                        feature_dict[col] = user_features[col]
                    elif col in game_features:
                        feature_dict[col] = game_features[col]

                # 现在基于模型的特征名构建特征数组
                for i, name in enumerate(feature_names):
                    if name in feature_dict:
                        features[i] = feature_dict[name]

            # 缓存并返回特征数组
            self.feature_cache[cache_key] = features
            return features

    def get_default_features(self):
        """获取默认特征（用于冷启动场景）"""
        # 如果模型存在，返回一个与模型特征数量相同的零数组
        if self.lgbm_model is not None:
            feature_names = self.lgbm_model.feature_name()
            return np.zeros(len(feature_names))

        # 如果模型不存在但有训练数据，基于训练数据创建默认特征
        elif hasattr(self, 'train_df') and self.train_df is not None:
            # 找出所有可用的数值特征
            num_features = []
            for col in self.train_df.columns:
                try:
                    val = self.train_df[col].iloc[0]
                    if not isinstance(val, (list, dict, set)) and col not in ['user_id', 'app_id', 'date', 'review_id',
                                                                              'is_recommended', 'prev_apps',
                                                                              'prev_ratings', 'prev_hours',
                                                                              'description', 'tags']:
                        if np.issubdtype(self.train_df[col].dtype, np.number) or self.train_df[col].dtype == bool:
                            num_features.append(col)
                except:
                    continue

            # 返回平均值作为默认特征
            if num_features:
                return self.train_df[num_features].mean().values

        # 如果什么都没有，返回一个单一的零
        return np.array([0.0])

    def predict_sequence_score(self, user_id, game_id):
        """使用序列模型预测得分"""
        if not hasattr(self, 'sequence_model') or self.sequence_model is None:
            return 0.5

        try:
            # 确保有保存的特征列表
            if not hasattr(self, 'sequence_feature_columns'):
                logger.error("缺少序列特征列表，无法进行预测")
                return 0.5

            # 创建与训练特征完全匹配的特征字典
            features = {}
            for col in self.sequence_feature_columns:
                features[col] = 0.0  # 默认值

            # 尝试填充实际值
            user_data = None
            if hasattr(self, 'user_df') and user_id in self.user_df['user_id'].values:
                user_data = self.user_df[self.user_df['user_id'] == user_id].iloc[0]

                # 填充用户特征
                if 'game_count' in self.sequence_feature_columns and 'game_count' in user_data:
                    features['game_count'] = user_data['game_count']

                if 'prev_game_count' in self.sequence_feature_columns and 'game_count' in user_data:
                    features['prev_game_count'] = user_data['game_count']

                if 'avg_prev_rating' in self.sequence_feature_columns and 'recommendation_ratio' in user_data:
                    features['avg_prev_rating'] = user_data['recommendation_ratio']

                if 'total_hours' in self.sequence_feature_columns and 'total_hours' in user_data:
                    features['total_hours'] = user_data['total_hours']

            # 使用完全一致的特征顺序
            feature_vector = [features[col] for col in self.sequence_feature_columns]

            # 打印特征向量（调试用）
            logger.debug(f"序列模型输入特征（{len(feature_vector)}维）: {feature_vector}")

            # 转换为张量并预测
            input_tensor = torch.FloatTensor([feature_vector]).to(self.device)

            self.sequence_model.eval()
            with torch.no_grad():
                score = self.sequence_model(input_tensor).item()
                return score

        except Exception as e:
            logger.error(f"序列模型预测出错: {str(e)}")
            return 0.5

    def predict_content_score(self, user_id, game_id):
        """使用基于内容的模型预测得分"""
        if not hasattr(self, 'content_similarity') or self.content_similarity is None:
            return 0.5

        # 获取用户喜欢的游戏
        user_liked_games = self.df[
            (self.df['user_id'] == user_id) &
            (self.df['is_recommended'] == True)
            ]['app_id'].tolist()

        # 如果用户没有喜欢的游戏，返回默认得分
        if not user_liked_games:
            return 0.5

        # 计算目标游戏与用户喜欢的游戏的平均相似度
        similarity_scores = []

        for liked_game in user_liked_games:
            # 检查游戏是否在相似度矩阵中
            if liked_game in self.content_similarity['game_idx'] and game_id in self.content_similarity['game_idx']:
                idx1 = self.content_similarity['game_idx'][liked_game]
                idx2 = self.content_similarity['game_idx'][game_id]
                similarity = self.content_similarity['matrix'][idx1, idx2]
                similarity_scores.append(similarity)

        # 如果没有可计算的相似度分数，返回默认得分
        if not similarity_scores:
            return 0.5

        # 返回平均相似度
        return sum(similarity_scores) / len(similarity_scores)

    def handle_cold_start_user(self, n=10):
        """处理冷启动用户（新用户）"""
        logger.info("处理冷启动用户...")

        # 为新用户推荐热门且评价高的游戏
        return self.get_popular_games(n)

    def get_popular_games(self, n=10):
        """获取热门游戏 - 优化版本"""
        # 使用game_df而不是df
        game_popularity = self.game_df.copy()

        # 添加流行度得分
        if 'recommendation_ratio' in game_popularity.columns and 'user_count' in game_popularity.columns:
            game_popularity['popularity_score'] = (
                    game_popularity['user_count'] * 0.7 +
                    game_popularity['recommendation_ratio'] * 0.3
            )
        else:
            # 如果没有这些列，使用可用的列
            if 'user_count' in game_popularity.columns:
                game_popularity['popularity_score'] = game_popularity['user_count']
            else:
                # 没有可用的指标，使用随机分数
                game_popularity['popularity_score'] = np.random.random(len(game_popularity))

        # 排序并获取前N个游戏
        popular_games = game_popularity.sort_values('popularity_score', ascending=False).head(n)

        # 返回游戏ID和得分
        return [(game_id, score) for game_id, score in zip(
            popular_games['app_id'], popular_games['popularity_score']
        )]

    def save_model(self, path='steam_recommender_model'):
        """Save model and related data"""
        logger.info(f"Saving model to {path}...")

        # Create save directory
        os.makedirs(path, exist_ok=True)

        # Save KNN model related data
        if hasattr(self, 'user_knn_model') and self.user_knn_model is not None:
            # Save index mappings
            with open(os.path.join(path, 'user_indices.pkl'), 'wb') as f:
                pickle.dump(self.user_indices, f)
            with open(os.path.join(path, 'app_indices.pkl'), 'wb') as f:
                pickle.dump(self.app_indices, f)
            with open(os.path.join(path, 'reversed_user_indices.pkl'), 'wb') as f:
                pickle.dump(self.reversed_user_indices, f)
            with open(os.path.join(path, 'reversed_app_indices.pkl'), 'wb') as f:
                pickle.dump(self.reversed_app_indices, f)

            # Save user-game matrix
            if hasattr(self, 'user_game_matrix') and self.user_game_matrix is not None:
                self.user_game_matrix.to_pickle(os.path.join(path, 'user_game_matrix.pkl'))

            # Save KNN models
            with open(os.path.join(path, 'user_knn_model.pkl'), 'wb') as f:
                pickle.dump(self.user_knn_model, f)
            with open(os.path.join(path, 'item_knn_model.pkl'), 'wb') as f:
                pickle.dump(self.item_knn_model, f)

        # Save SVD model
        if hasattr(self, 'svd_model') and self.svd_model is not None:
            with open(os.path.join(path, 'svd_model.pkl'), 'wb') as f:
                pickle.dump(self.svd_model, f)

        # Save simple model (replacing LightGBM)
        if hasattr(self, 'simple_model') and self.simple_model is not None:
            with open(os.path.join(path, 'simple_model.pkl'), 'wb') as f:
                pickle.dump(self.simple_model, f)

        # Save sequence model
        if hasattr(self, 'sequence_model') and self.sequence_model is not None:
            torch.save(self.sequence_model.state_dict(), os.path.join(path, 'sequence_model.pt'))
            # Save sequence feature columns
            if hasattr(self, 'sequence_feature_columns'):
                with open(os.path.join(path, 'sequence_features.pkl'), 'wb') as f:
                    pickle.dump(self.sequence_feature_columns, f)

        # Save encoders and other components
        with open(os.path.join(path, 'label_encoders.pkl'), 'wb') as f:
            pickle.dump(self.label_encoders, f)

        if hasattr(self, 'scaler') and self.scaler is not None:
            with open(os.path.join(path, 'scaler.pkl'), 'wb') as f:
                pickle.dump(self.scaler, f)

        if hasattr(self, 'tfidf_model') and self.tfidf_model is not None:
            with open(os.path.join(path, 'tfidf_model.pkl'), 'wb') as f:
                pickle.dump(self.tfidf_model, f)

        if hasattr(self, 'tfidf_svd') and self.tfidf_svd is not None:
            with open(os.path.join(path, 'tfidf_svd.pkl'), 'wb') as f:
                pickle.dump(self.tfidf_svd, f)

        if hasattr(self, 'game_embeddings') and self.game_embeddings is not None:
            with open(os.path.join(path, 'game_embeddings.pkl'), 'wb') as f:
                pickle.dump(self.game_embeddings, f)

        if hasattr(self, 'content_similarity') and self.content_similarity is not None:
            with open(os.path.join(path, 'content_similarity.pkl'), 'wb') as f:
                pickle.dump(self.content_similarity, f)

        # Save configuration
        with open(os.path.join(path, 'config.json'), 'w') as f:
            json.dump(self.config, f, indent=4)

        logger.info(f"Model save completed")

    def load_model(self, path='steam_recommender_model'):
        """Load model and related data"""
        logger.info(f"Loading model from {path}...")

        # Check if save directory exists
        if not os.path.exists(path):
            logger.error(f"Model directory {path} does not exist")
            return False

        # Load configuration
        config_path = os.path.join(path, 'config.json')
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                self.config = json.load(f)

        # Load KNN model related data
        user_indices_path = os.path.join(path, 'user_indices.pkl')
        app_indices_path = os.path.join(path, 'app_indices.pkl')
        reversed_user_indices_path = os.path.join(path, 'reversed_user_indices.pkl')
        reversed_app_indices_path = os.path.join(path, 'reversed_app_indices.pkl')
        user_game_matrix_path = os.path.join(path, 'user_game_matrix.pkl')
        user_knn_model_path = os.path.join(path, 'user_knn_model.pkl')
        item_knn_model_path = os.path.join(path, 'item_knn_model.pkl')

        # Load index mappings
        if os.path.exists(user_indices_path):
            with open(user_indices_path, 'rb') as f:
                self.user_indices = pickle.load(f)

        if os.path.exists(app_indices_path):
            with open(app_indices_path, 'rb') as f:
                self.app_indices = pickle.load(f)

        if os.path.exists(reversed_user_indices_path):
            with open(reversed_user_indices_path, 'rb') as f:
                self.reversed_user_indices = pickle.load(f)

        if os.path.exists(reversed_app_indices_path):
            with open(reversed_app_indices_path, 'rb') as f:
                self.reversed_app_indices = pickle.load(f)

        # Load user-game matrix
        if os.path.exists(user_game_matrix_path):
            self.user_game_matrix = pd.read_pickle(user_game_matrix_path)
            # Rebuild sparse matrix
            self.user_game_sparse_matrix = csr_matrix(self.user_game_matrix.values)

        # Load KNN models
        if os.path.exists(user_knn_model_path):
            with open(user_knn_model_path, 'rb') as f:
                self.user_knn_model = pickle.load(f)

        if os.path.exists(item_knn_model_path):
            with open(item_knn_model_path, 'rb') as f:
                self.item_knn_model = pickle.load(f)

        # Load SVD model
        svd_model_path = os.path.join(path, 'svd_model.pkl')
        if os.path.exists(svd_model_path):
            with open(svd_model_path, 'rb') as f:
                self.svd_model = pickle.load(f)

        # Load simple model (replacing LightGBM)
        simple_model_path = os.path.join(path, 'simple_model.pkl')
        if os.path.exists(simple_model_path):
            with open(simple_model_path, 'rb') as f:
                self.simple_model = pickle.load(f)

        # Load encoders and other components
        encoders_path = os.path.join(path, 'label_encoders.pkl')
        if os.path.exists(encoders_path):
            with open(encoders_path, 'rb') as f:
                self.label_encoders = pickle.load(f)

        scaler_path = os.path.join(path, 'scaler.pkl')
        if os.path.exists(scaler_path):
            with open(scaler_path, 'rb') as f:
                self.scaler = pickle.load(f)

        tfidf_path = os.path.join(path, 'tfidf_model.pkl')
        if os.path.exists(tfidf_path):
            with open(tfidf_path, 'rb') as f:
                self.tfidf_model = pickle.load(f)

        svd_path = os.path.join(path, 'tfidf_svd.pkl')
        if os.path.exists(svd_path):
            with open(svd_path, 'rb') as f:
                self.tfidf_svd = pickle.load(f)

        embeddings_path = os.path.join(path, 'game_embeddings.pkl')
        if os.path.exists(embeddings_path):
            with open(embeddings_path, 'rb') as f:
                self.game_embeddings = pickle.load(f)

        similarity_path = os.path.join(path, 'content_similarity.pkl')
        if os.path.exists(similarity_path):
            with open(similarity_path, 'rb') as f:
                self.content_similarity = pickle.load(f)

        # Load sequence model
        sequence_path = os.path.join(path, 'sequence_model.pt')
        sequence_features_path = os.path.join(path, 'sequence_features.pkl')

        if os.path.exists(sequence_features_path):
            with open(sequence_features_path, 'rb') as f:
                self.sequence_feature_columns = pickle.load(f)

        if os.path.exists(sequence_path):
            # Need to initialize model architecture before loading weights
            logger.info("Sequence model weights exist but need to manually rebuild model architecture before loading")
            # Rebuild logic can be added here if needed

        logger.info(f"Model loading completed")
        return True

    def get_content_recommendations(self, app_id, top_n=10):
        """
        基于内容的推荐，给定一个游戏ID，推荐相似的游戏

        参数:
            app_id: 游戏ID
            top_n: 推荐数量

        返回:
            list: [(game_id, score), ...] 推荐游戏列表
        """
        logger.info(f"为游戏 {app_id} 生成内容推荐")

        # 检查内容相似度矩阵是否存在
        if not hasattr(self, 'content_similarity') or self.content_similarity is None:
            logger.warning("内容相似度矩阵不存在，返回热门游戏")
            return self.get_popular_games(top_n)

        # 检查游戏是否在相似度矩阵中
        if app_id not in self.content_similarity['game_idx']:
            logger.warning(f"游戏 {app_id} 不在相似度矩阵中，返回热门游戏")
            return self.get_popular_games(top_n)

        # 获取游戏索引
        idx = self.content_similarity['game_idx'][app_id]

        # 获取相似度分数
        sim_scores = self.content_similarity['matrix'][idx]

        # 创建游戏ID到索引的反向映射
        reverse_idx = {idx: game_id for game_id, idx in self.content_similarity['game_idx'].items()}

        # 找出最相似的游戏（排除自己）
        sim_items = [(reverse_idx[i], sim_scores[i])
                     for i in range(len(sim_scores))
                     if i != idx and i in reverse_idx]

        # 排序并获取前N个
        recommendations = sorted(sim_items, key=lambda x: x[1], reverse=True)[:top_n]

        return recommendations

    def _create_training_sample(self):
        """从大型数据集创建训练样本"""
        logger.info("创建训练样本...")

        try:
            # 选择活跃用户和热门游戏
            if len(self.user_df) > 10000:
                active_users = self.user_df.sort_values('game_count', ascending=False).head(10000)['user_id'].values
            else:
                active_users = self.user_df['user_id'].values

            if len(self.game_df) > 5000:
                popular_games = self.game_df.sort_values('user_count', ascending=False).head(5000)['app_id'].values
            else:
                popular_games = self.game_df['app_id'].values

            # 读取部分原始数据用于训练
            sample_rows = []
            sample_size = min(1000000, len(self.user_df) * 10)  # 限制样本大小

            rows_collected = 0
            chunk_size = 500000

            for chunk in pd.read_csv(self.data_path, chunksize=chunk_size):
                # 确保必要的列存在
                if not all(col in chunk.columns for col in ['user_id', 'app_id', 'is_recommended']):
                    continue

                # 过滤出活跃用户和热门游戏的交互
                filtered_chunk = chunk[
                    (chunk['user_id'].isin(active_users)) &
                    (chunk['app_id'].isin(popular_games))
                    ]

                if len(filtered_chunk) > 0:
                    sample_rows.append(filtered_chunk)
                    rows_collected += len(filtered_chunk)

                if rows_collected >= sample_size:
                    break

            if not sample_rows:
                logger.warning("没有收集到样本数据！")
                return False

            # 合并样本
            self.sample_df = pd.concat(sample_rows)

            # 分割训练和测试
            from sklearn.model_selection import train_test_split
            self.train_df, self.test_df = train_test_split(
                self.sample_df, test_size=0.2, random_state=42
            )

            logger.info(
                f"创建了 {len(self.sample_df)} 行的训练样本，训练集 {len(self.train_df)} 行，测试集 {len(self.test_df)} 行")
            return True

        except Exception as e:
            logger.error(f"创建训练样本时出错: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return False

    def train_knn_model(self):
        """训练KNN模型，替代LightGBM模型"""
        logger.info("开始训练KNN模型...")

        # 检查是否有足够的数据
        if not hasattr(self, 'train_df') or self.train_df is None or len(self.train_df) == 0:
            logger.error("训练数据不足，无法训练KNN模型")
            return None

        # 创建用户-游戏交互矩阵
        # 使用用户-游戏评分（如果有）或者是否推荐作为交互值
        if 'rating' in self.train_df.columns:
            rating_col = 'rating'
        elif 'is_recommended' in self.train_df.columns:
            # 将布尔值转换为0和1
            self.train_df['rating_value'] = self.train_df['is_recommended'].astype(int) * 10
            rating_col = 'rating_value'
        else:
            # 如果没有评分或推荐，使用游戏时间作为交互值
            rating_col = 'hours'

        # 创建透视表
        user_game_matrix = pd.pivot_table(
            self.train_df,
            values=rating_col,
            index='user_id',
            columns='app_id',
            aggfunc='mean',
            fill_value=0
        )

        logger.info(f"创建的用户-游戏矩阵大小: {user_game_matrix.shape}")

        # 存储用户和游戏ID映射，用于后续推荐
        self.user_indices = {user: i for i, user in enumerate(user_game_matrix.index)}
        self.app_indices = {app: i for i, app in enumerate(user_game_matrix.columns)}
        self.reversed_user_indices = {i: user for user, i in self.user_indices.items()}
        self.reversed_app_indices = {i: app for app, i in self.app_indices.items()}

        # 转换为稀疏矩阵提高效率
        matrix = csr_matrix(user_game_matrix.values)

        # 创建基于用户的KNN模型
        self.user_knn_model = NearestNeighbors(
            metric='cosine',
            algorithm='brute',
            n_neighbors=min(20, len(self.user_indices)),
            n_jobs=-1
        )
        self.user_knn_model.fit(matrix)

        # 创建基于物品的KNN模型
        self.item_knn_model = NearestNeighbors(
            metric='cosine',
            algorithm='brute',
            n_neighbors=min(20, len(self.app_indices)),
            n_jobs=-1
        )
        self.item_knn_model.fit(matrix.T)  # 转置矩阵用于物品相似度

        # 存储原始矩阵以便后续计算
        self.user_game_matrix = user_game_matrix
        self.user_game_sparse_matrix = matrix

        logger.info("KNN模型训练完成")
        return (self.user_knn_model, self.item_knn_model)

    def predict_user_knn_score(self, user_id, app_id):
        """使用用户KNN预测评分"""
        if not hasattr(self, 'user_knn_model') or self.user_knn_model is None:
            return 0.5

        try:
            # 检查用户和游戏是否在训练数据中
            if user_id not in self.user_indices or app_id not in self.app_indices:
                return 0.5

            # 获取用户索引
            user_idx = self.user_indices[user_id]
            app_idx = self.app_indices[app_id]

            # 获取用户的向量
            user_vector = self.user_game_sparse_matrix[user_idx].toarray().reshape(1, -1)

            # 找到最相似的用户
            distances, indices = self.user_knn_model.kneighbors(user_vector, n_neighbors=10)

            # 计算相似用户的加权评分
            similar_users = indices[0]
            similarities = 1 - distances[0]  # 将距离转换为相似度

            # 过滤掉用户自己
            if user_idx in similar_users:
                user_idx_pos = np.where(similar_users == user_idx)[0][0]
                similar_users = np.delete(similar_users, user_idx_pos)
                similarities = np.delete(similarities, user_idx_pos)

            if len(similar_users) == 0:
                return 0.5

            # 计算相似用户对目标游戏的评分
            ratings = []
            weights = []

            for i, similar_user_idx in enumerate(similar_users):
                similar_user_id = self.reversed_user_indices[similar_user_idx]
                # 查找该用户对该游戏的评分
                similar_user_data = self.train_df[
                    (self.train_df['user_id'] == similar_user_id) &
                    (self.train_df['app_id'] == app_id)
                    ]

                if len(similar_user_data) > 0:
                    # 获取评分（优先使用rating，其次是is_recommended）
                    if 'rating' in similar_user_data.columns:
                        rating = similar_user_data['rating'].iloc[0]
                    elif 'is_recommended' in similar_user_data.columns:
                        rating = 10 if similar_user_data['is_recommended'].iloc[0] else 0
                    else:
                        # 如果没有评分或推荐，使用归一化的游戏时间
                        hours = similar_user_data['hours'].iloc[0]
                        rating = min(10, hours / 10)  # 将时间映射到0-10的范围

                    ratings.append(rating)
                    weights.append(similarities[i])

            # 如果没有相似用户评价过该游戏，返回默认分数
            if len(ratings) == 0:
                return 0.5

            # 计算加权平均评分
            weighted_rating = np.average(ratings, weights=weights)
            # 将评分归一化到0-1范围
            normalized_rating = weighted_rating / 10

            return normalized_rating

        except Exception as e:
            logger.error(f"用户KNN预测出错: {str(e)}")
            return 0.5

    def predict_item_knn_score(self, user_id, app_id):
        """使用物品KNN预测评分"""
        if not hasattr(self, 'item_knn_model') or self.item_knn_model is None:
            return 0.5

        try:
            # 检查用户和游戏是否在训练数据中
            if app_id not in self.app_indices:
                return 0.5

            # 获取游戏索引
            app_idx = self.app_indices[app_id]

            # 获取游戏的向量
            item_vector = self.user_game_sparse_matrix.T[app_idx].toarray().reshape(1, -1)

            # 找到最相似的游戏
            distances, indices = self.item_knn_model.kneighbors(item_vector, n_neighbors=10)

            # 计算相似游戏的加权评分
            similar_items = indices[0]
            similarities = 1 - distances[0]  # 将距离转换为相似度

            # 过滤掉游戏自己
            if app_idx in similar_items:
                app_idx_pos = np.where(similar_items == app_idx)[0][0]
                similar_items = np.delete(similar_items, app_idx_pos)
                similarities = np.delete(similarities, app_idx_pos)

            if len(similar_items) == 0:
                return 0.5

            # 计算用户对相似游戏的评分
            ratings = []
            weights = []

            for i, similar_item_idx in enumerate(similar_items):
                similar_app_id = self.reversed_app_indices[similar_item_idx]
                # 查找用户对该游戏的评分
                user_rating_data = self.train_df[
                    (self.train_df['user_id'] == user_id) &
                    (self.train_df['app_id'] == similar_app_id)
                    ]

                if len(user_rating_data) > 0:
                    # 获取评分
                    if 'rating' in user_rating_data.columns:
                        rating = user_rating_data['rating'].iloc[0]
                    elif 'is_recommended' in user_rating_data.columns:
                        rating = 10 if user_rating_data['is_recommended'].iloc[0] else 0
                    else:
                        # 如果没有评分或推荐，使用归一化的游戏时间
                        hours = user_rating_data['hours'].iloc[0]
                        rating = min(10, hours / 10)  # 将时间映射到0-10的范围

                    ratings.append(rating)
                    weights.append(similarities[i])

            # 如果用户没有评价过任何相似游戏，返回默认分数
            if len(ratings) == 0:
                return 0.5

            # 计算加权平均评分
            weighted_rating = np.average(ratings, weights=weights)
            # 将评分归一化到0-1范围
            normalized_rating = weighted_rating / 10

            return normalized_rating

        except Exception as e:
            logger.error(f"物品KNN预测出错: {str(e)}")
            return 0.5


# 封装所有增量更新为一个方法
def incremental_update(self, interactions_df, games_df=None, users_df=None):
    """
    执行所有模型的增量更新

    参数:
        interactions_df (DataFrame): 用户-游戏交互数据
        games_df (DataFrame): 游戏数据，可选
        users_df (DataFrame): 用户数据，可选
    """
    logger.info("开始执行全面增量更新...")

    try:
        # 更新KNN模型
        if len(interactions_df) > 0:
            self.update_knn_model(interactions_df)

        # 更新序列模型
        if len(interactions_df) > 0 and hasattr(self, 'update_sequence_model'):
            self.update_sequence_model(interactions_df)

        # 更新内容模型
        if games_df is not None and len(games_df) > 0 and hasattr(self, 'update_content_model'):
            self.update_content_model(games_df)

        # 更新游戏嵌入向量
        if hasattr(self, 'create_game_embeddings'):
            self.create_game_embeddings()

        # 清除推荐缓存
        self.recommendation_cache = {}
        self.score_cache = {}
        self.feature_cache = {}

        logger.info("全面增量更新完成")

    except Exception as e:
        logger.error(f"执行增量更新时出错: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())

def update_knn_model(self, new_data_df):
    """
    增量更新KNN模型

    参数:
        new_data_df (DataFrame): 新的交互数据
    """
    logger.info("开始增量更新KNN模型...")

    if not hasattr(self, 'user_knn_model') or self.user_knn_model is None:
        logger.warning("KNN模型不存在，将进行完整训练")
        self.train_knn_model()
        return

    try:
        # 确保新数据包含必要的列
        required_cols = ['user_id', 'app_id']
        if not all(col in new_data_df.columns for col in required_cols):
            logger.error("新数据缺少必要的列，无法执行增量训练")
            return

        # 检查评分列
        if 'rating' in new_data_df.columns:
            rating_col = 'rating'
        elif 'is_recommended' in new_data_df.columns:
            new_data_df['rating_value'] = new_data_df['is_recommended'].astype(int) * 10
            rating_col = 'rating_value'
        elif 'hours' in new_data_df.columns:
            rating_col = 'hours'
        else:
            logger.error("新数据缺少评分相关列，无法执行增量训练")
            return

        # 将新数据合并到原始数据
        if not hasattr(self, 'user_game_matrix') or self.user_game_matrix is None:
            logger.error("原始用户-游戏矩阵不存在，无法执行增量训练")
            return

        # 遍历新数据，更新用户-游戏矩阵
        for _, row in new_data_df.iterrows():
            user_id = row['user_id']
            app_id = row['app_id']
            rating = row[rating_col]

            # 如果是新用户或新游戏，需要扩展矩阵
            if user_id not in self.user_indices:
                # 为新用户添加一行
                new_user_idx = len(self.user_indices)
                self.user_indices[user_id] = new_user_idx
                self.reversed_user_indices[new_user_idx] = user_id

                # 在矩阵中添加一行
                new_row = pd.DataFrame(
                    [0] * len(self.user_game_matrix.columns),
                    index=[user_id],
                    columns=self.user_game_matrix.columns
                )
                self.user_game_matrix = pd.concat([self.user_game_matrix, new_row])

            if app_id not in self.app_indices:
                # 为新游戏添加一列
                new_app_idx = len(self.app_indices)
                self.app_indices[app_id] = new_app_idx
                self.reversed_app_indices[new_app_idx] = app_id

                # 在矩阵中添加一列
                self.user_game_matrix[app_id] = 0

            # 更新评分
            self.user_game_matrix.at[user_id, app_id] = rating

        # 更新稀疏矩阵
        self.user_game_sparse_matrix = csr_matrix(self.user_game_matrix.values)

        # 重新训练KNN模型
        self.user_knn_model = NearestNeighbors(
            metric='cosine',
            algorithm='brute',
            n_neighbors=min(20, len(self.user_indices)),
            n_jobs=-1
        )
        self.user_knn_model.fit(self.user_game_sparse_matrix)

        # 更新物品KNN模型
        self.item_knn_model = NearestNeighbors(
            metric='cosine',
            algorithm='brute',
            n_neighbors=min(20, len(self.app_indices)),
            n_jobs=-1
        )
        self.item_knn_model.fit(self.user_game_sparse_matrix.T)

        logger.info("KNN模型增量更新完成")

    except Exception as e:
        logger.error(f"增量更新KNN模型时出错: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())


def generate_hybrid_recommendations(self, user_id, n=10):
    """
    生成多样化的混合推荐

    参数:
        user_id (int): 用户ID
        n (int): 推荐数量

    返回:
        list: [(game_id, score), ...] 推荐游戏列表
    """
    logger.info(f"为用户 {user_id} 生成混合推荐...")

    # 使用缓存
    cache_key = f"hybrid_{user_id}_{n}"
    if cache_key in self.recommendation_cache:
        return self.recommendation_cache[cache_key]

    # 对于新用户或找不到的用户，使用热门推荐
    if not hasattr(self, 'user_indices') or user_id not in self.user_indices:
        logger.warning(f"用户 {user_id} 未在训练数据中找到")
        return self.get_popular_games(n)

    # 获取用户已经评论过的游戏
    user_games = set()

    # 从训练集获取
    if hasattr(self, 'train_df'):
        user_train_games = self.train_df[self.train_df['user_id'] == user_id]['app_id'].values
        user_games.update(user_train_games)

    # 从测试集获取
    if hasattr(self, 'test_df'):
        user_test_games = self.test_df[self.test_df['user_id'] == user_id]['app_id'].values
        user_games.update(user_test_games)

    # 准备三种不同的推荐方法
    knn_recs = self.generate_knn_recommendations(user_id, n)
    content_recs = self.generate_content_based_recommendations(user_id, n)
    popular_recs = self.get_popular_games(n)

    # 每种方法的权重
    knn_weight = 0.5
    content_weight = 0.3
    popular_weight = 0.2

    # 合并推荐结果，避免重复
    merged_scores = {}

    # 处理KNN推荐
    for game_id, score in knn_recs:
        if game_id not in user_games:  # 过滤已交互的游戏
            merged_scores[game_id] = score * knn_weight

    # 处理基于内容的推荐
    for game_id, score in content_recs:
        if game_id not in user_games:  # 过滤已交互的游戏
            if game_id in merged_scores:
                merged_scores[game_id] += score * content_weight
            else:
                merged_scores[game_id] = score * content_weight

    # 处理热门推荐（确保新用户也有推荐结果）
    for game_id, score in popular_recs:
        if game_id not in user_games:  # 过滤已交互的游戏
            if game_id in merged_scores:
                merged_scores[game_id] += score * popular_weight
            else:
                merged_scores[game_id] = score * popular_weight

    # 按分数排序
    sorted_games = sorted(
        [(game_id, score) for game_id, score in merged_scores.items()],
        key=lambda x: x[1],
        reverse=True
    )

    # 取前N个结果
    recommendations = sorted_games[:n]

    # 缓存结果
    self.recommendation_cache[cache_key] = recommendations

    logger.info(f"为用户 {user_id} 生成了 {len(recommendations)} 条混合推荐")
    return recommendations


def generate_knn_recommendations(self, user_id, n=10):
    """
    使用KNN生成推荐

    参数:
        user_id (int): 用户ID
        n (int): 推荐数量

    返回:
        list: [(game_id, score), ...] 推荐游戏列表
    """
    # 检查用户是否在KNN模型中
    if not hasattr(self, 'user_indices') or user_id not in self.user_indices:
        return self.get_popular_games(n)

    try:
        # 获取用户已评价的游戏
        user_games = set()
        if hasattr(self, 'train_df'):
            user_train_games = self.train_df[self.train_df['user_id'] == user_id]['app_id'].values
            user_games.update(user_train_games)

        if hasattr(self, 'test_df'):
            user_test_games = self.test_df[self.test_df['user_id'] == user_id]['app_id'].values
            user_games.update(user_test_games)

        # 获取可能的游戏列表
        candidate_games = set(self.app_indices.keys()) - user_games

        # 如果没有候选游戏，返回热门游戏
        if not candidate_games:
            return self.get_popular_games(n)

        # 为每个候选游戏计算预测得分
        predictions = []
        for game_id in candidate_games:
            # 计算用户对游戏的预测得分
            user_score = self.predict_user_knn_score(user_id, game_id)
            item_score = self.predict_item_knn_score(user_id, game_id)

            # 结合两种KNN得分
            combined_score = (user_score + item_score) / 2
            predictions.append((game_id, combined_score))

        # 按得分排序
        predictions.sort(key=lambda x: x[1], reverse=True)

        # 返回前N个结果
        return predictions[:n]

    except Exception as e:
        logger.error(f"生成KNN推荐时出错: {str(e)}")
        return self.get_popular_games(n)


def generate_content_based_recommendations(self, user_id, n=10):
    """
    基于用户偏好生成内容推荐

    参数:
        user_id (int): 用户ID
        n (int): 推荐数量

    返回:
        list: [(game_id, score), ...] 推荐游戏列表
    """
    if not hasattr(self, 'content_similarity') or self.content_similarity is None:
        return self.get_popular_games(n)

    try:
        # 获取用户喜欢的游戏
        liked_games = []

        if hasattr(self, 'train_df'):
            liked_train = self.train_df[
                (self.train_df['user_id'] == user_id) &
                (self.train_df['is_recommended'] == True)
                ]['app_id'].values
            liked_games.extend(liked_train)

        if hasattr(self, 'test_df'):
            liked_test = self.test_df[
                (self.test_df['user_id'] == user_id) &
                (self.test_df['is_recommended'] == True)
                ]['app_id'].values
            liked_games.extend(liked_test)

        # 如果用户没有喜欢的游戏，使用游戏时间较长的游戏
        if not liked_games and hasattr(self, 'train_df'):
            user_games = self.train_df[self.train_df['user_id'] == user_id]
            if len(user_games) > 0:
                # 按游戏时间排序
                top_games = user_games.sort_values('hours', ascending=False).head(3)
                liked_games = top_games['app_id'].values

        # 如果还是没有游戏，返回热门游戏
        if not liked_games:
            return self.get_popular_games(n)

        # 对每个喜欢的游戏找出相似游戏
        all_similar_games = []

        for game_id in liked_games:
            if game_id in self.content_similarity['game_idx']:
                similar_games = self.get_content_recommendations(game_id, n=10)
                all_similar_games.extend(similar_games)

        # 如果没有找到相似游戏，返回热门游戏
        if not all_similar_games:
            return self.get_popular_games(n)

        # 合并相似游戏，按相似度排序
        game_scores = {}
        for game_id, score in all_similar_games:
            if game_id not in liked_games:  # 排除用户已经喜欢的游戏
                if game_id in game_scores:
                    game_scores[game_id] = max(game_scores[game_id], score)
                else:
                    game_scores[game_id] = score

        # 按分数排序
        recommendations = sorted(
            [(game_id, score) for game_id, score in game_scores.items()],
            key=lambda x: x[1],
            reverse=True
        )

        return recommendations[:n]

    except Exception as e:
        logger.error(f"生成内容推荐时出错: {str(e)}")
        return self.get_popular_games(n)

def main():
    """主函数，演示推荐系统的使用"""
    # 设置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # 初始化推荐系统
    recommender = SteamRecommender('steam_top_100000.csv')

    # 加载数据
    recommender.load_data()

    # 特征工程
    recommender.engineer_features()

    # 训练LightGBM模型
    recommender.train_lgbm_model()

    # 训练序列模型
    recommender.train_sequence_model()

    # 创建游戏嵌入
    recommender.create_game_embeddings()

    # 训练内容模型
    recommender.train_content_model()

    # 评估推荐系统
    evaluation_results = recommender.evaluate_recommendations()
    recommender.evaluation_results = evaluation_results

    # 可视化结果
    recommender.visualize_results()

    # 保存模型
    recommender.save_model()

    # 为特定用户生成推荐
    if len(recommender.df['user_id'].unique()) > 0:
        example_user = recommender.df['user_id'].iloc[0]
        recommendations = recommender.generate_recommendations(example_user, 10)

        # 打印推荐结果
        print(f"\n为用户 {example_user} 的推荐:")
        for i, (game_id, score) in enumerate(recommendations, 1):
            game_title = recommender.df[recommender.df['app_id'] == game_id]['title'].iloc[0]
            print(f"{i}. {game_title} (ID: {game_id}, Score: {score:.4f})")

    print("\n推荐系统运行完成！")


if __name__ == "__main__":
    main()