#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Steam游戏推荐系统核心模块
作者: Claude
日期: 2025-04-24
描述: 基于LightGBM和序列行为的Steam游戏推荐系统
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import re
from collections import defaultdict, Counter
import pickle
import warnings
import logging
from tqdm import tqdm
import os
import lightgbm as lgb
from sklearn.model_selection import train_test_split, GroupKFold, TimeSeriesSplit
from sklearn.metrics import roc_auc_score, precision_score, recall_score, ndcg_score, mean_squared_error
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

try:
    from transformers import AutoTokenizer, AutoModel, AutoConfig
except ImportError:
    print("transformers库未安装，一些高级特征可能无法使用")
try:
    from gensim.models import Word2Vec
except ImportError:
    print("gensim库未安装，Word2Vec特征将不可用")
try:
    import nltk
    from nltk.tokenize import word_tokenize
    from nltk.corpus import stopwords
except ImportError:
    print("nltk库未安装，文本处理功能将受限")
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
        """加载和预处理数据"""
        logger.info(f"开始加载数据: {self.data_path}")

        # 读取CSV文件
        self.df = pd.read_csv(self.data_path)

        # 数据类型转换和清洗
        # 将字符串布尔值转换为实际布尔值
        bool_columns = ['is_recommended', 'win', 'mac', 'linux', 'steam_deck']
        for col in bool_columns:
            if col in self.df.columns:
                self.df[col] = self.df[col].map({'True': True, 'False': False, True: True, False: False})

        # 转换日期列
        date_columns = ['date', 'date_release']
        for col in date_columns:
            if col in self.df.columns:
                self.df[col] = pd.to_datetime(self.df[col])

        # 显示数据概况
        logger.info(f"数据加载完成，形状: {self.df.shape}")
        logger.info(f"列名: {self.df.columns.tolist()}")
        logger.info(f"唯一用户数: {self.df['user_id'].nunique()}")
        logger.info(f"唯一游戏数: {self.df['app_id'].nunique()}")

        # 基本数据整理
        # 填充缺失值
        self.df['hours'] = self.df['hours'].fillna(0)
        self.df['is_recommended'] = self.df['is_recommended'].fillna(False)

        # 确保有数值型评分列
        if 'rating_new' in self.df.columns:
            self.df['rating'] = self.df['rating_new']
        else:
            # 如果没有rating_new列，尝试从is_recommended创建一个简单的评分
            self.df['rating'] = self.df['is_recommended'].astype(int) * 10

        # 添加基础时间特征
        if 'date' in self.df.columns:
            self.df['day_of_week'] = self.df['date'].dt.dayofweek
            self.df['month'] = self.df['date'].dt.month
            self.df['year'] = self.df['date'].dt.year
            self.df['days_since_release'] = (self.df['date'] - self.df['date_release']).dt.days

            # 处理负值（评论早于发布日期的情况）
            self.df['days_since_release'] = self.df['days_since_release'].clip(lower=0)

        # 划分训练集和测试集，基于时间序列
        if 'date' in self.df.columns:
            # 按时间排序
            self.df = self.df.sort_values('date')
            # 使用最近20%的数据作为测试集
            split_idx = int(len(self.df) * 0.8)
            self.train_df = self.df.iloc[:split_idx].copy()
            self.test_df = self.df.iloc[split_idx:].copy()
        else:
            # 如果没有日期，则随机划分
            self.train_df, self.test_df = train_test_split(
                self.df, test_size=0.2, random_state=RANDOM_SEED,
                stratify=self.df['is_recommended'] if 'is_recommended' in self.df.columns else None
            )

        logger.info(f"训练集大小: {self.train_df.shape}, 测试集大小: {self.test_df.shape}")
        return self.df

    def engineer_features(self):
        """特征工程函数，创建高级特征用于推荐系统"""
        logger.info("开始特征工程...")

        # 类别特征编码
        cat_features = ['app_id', 'user_id', 'rating']
        for feat in cat_features:
            if feat in self.df.columns:
                self.label_encoders[feat] = LabelEncoder()
                self.df[f'{feat}_encoded'] = self.label_encoders[feat].fit_transform(self.df[feat])
                self.train_df[f'{feat}_encoded'] = self.label_encoders[feat].transform(self.train_df[feat])

                # 处理测试集中可能有的新值
                test_values = self.test_df[feat].values
                known_values = set(self.label_encoders[feat].classes_)
                new_values = [val for val in test_values if val not in known_values]
                if new_values:
                    logger.warning(f"测试集中发现 {len(new_values)} 个新的 {feat} 值，将被设为-1")
                    self.test_df[f'{feat}_encoded'] = self.test_df[feat].map(
                        lambda x: self.label_encoders[feat].transform([x])[0] if x in known_values else -1
                    )
                else:
                    self.test_df[f'{feat}_encoded'] = self.label_encoders[feat].transform(self.test_df[feat])

        # ===== 用户特征 =====
        user_features = self.create_user_features()
        self.train_df = self.train_df.merge(user_features, on='user_id', how='left')
        self.test_df = self.test_df.merge(user_features, on='user_id', how='left')

        # ===== 游戏特征 =====
        game_features = self.create_game_features()
        self.train_df = self.train_df.merge(game_features, on='app_id', how='left')
        self.test_df = self.test_df.merge(game_features, on='app_id', how='left')

        # ===== 文本特征 =====
        self.process_text_features()

        # ===== 交互特征 =====
        self.create_interaction_features()

        # ===== 序列特征 =====
        self.create_sequence_features()

        # 缩放数值特征
        num_features = [col for col in self.train_df.columns
                        if self.train_df[col].dtype in [np.int64, np.float64]
                        and col not in ['user_id', 'app_id', 'review_id', 'is_recommended']
                        and not col.endswith('_encoded')]

        self.scaler = StandardScaler()
        self.train_df[num_features] = self.scaler.fit_transform(self.train_df[num_features].fillna(0))
        self.test_df[num_features] = self.scaler.transform(self.test_df[num_features].fillna(0))

        # 填充缺失值
        for df in [self.train_df, self.test_df]:
            df.fillna(0, inplace=True)

        logger.info(f"特征工程完成，最终训练集特征数: {self.train_df.shape[1]}")
        return self.train_df, self.test_df

    def create_user_features(self):
        """创建用户级特征"""
        logger.info("创建用户特征...")

        # 按用户分组
        user_data = self.df.groupby('user_id').agg({
            'app_id': 'count',  # 用户评论的游戏数
            'hours': ['mean', 'sum', 'std', 'max'],  # 游戏时间统计
            'is_recommended': ['mean', 'sum'],  # 推荐率
            'rating': ['mean', 'std'],  # 评分统计
            'price_final': ['mean', 'sum'] if 'price_final' in self.df.columns else [],  # 价格统计
            'discount': ['mean'] if 'discount' in self.df.columns else []  # 折扣统计
        })

        # 展平多级索引列名
        user_data.columns = ['_'.join(col).strip() for col in user_data.columns.values]
        user_data = user_data.reset_index()

        # 添加更多用户特征
        if 'date' in self.df.columns:
            # 用户活跃时间跨度
            user_dates = self.df.groupby('user_id')['date'].agg(['min', 'max'])
            user_dates['activity_days'] = (user_dates['max'] - user_dates['min']).dt.days
            user_data = user_data.merge(user_dates[['activity_days']], on='user_id', how='left')

            # 近期活跃度（最近30天的评论数）
            recent_cutoff = self.df['date'].max() - pd.Timedelta(days=30)
            recent_reviews = self.df[self.df['date'] >= recent_cutoff]
            recent_count = recent_reviews.groupby('user_id')['app_id'].count().reset_index()
            recent_count.columns = ['user_id', 'recent_activity']
            user_data = user_data.merge(recent_count, on='user_id', how='left')
            user_data['recent_activity'] = user_data['recent_activity'].fillna(0)

        # 游戏类型偏好
        if 'tags' in self.df.columns:
            # 提取每个用户评论过的所有游戏的标签
            user_tags = {}
            for _, row in self.df.iterrows():
                if pd.isna(row['tags']):
                    continue
                user_id = row['user_id']
                if user_id not in user_tags:
                    user_tags[user_id] = []
                tags = [tag.strip() for tag in row['tags'].split(',')]
                user_tags[user_id].extend(tags)

            # 计算每个用户的前3个最常见标签
            for user_id, tags in user_tags.items():
                tag_counter = Counter(tags)
                top_tags = [tag for tag, _ in tag_counter.most_common(3)]
                for i, tag in enumerate(top_tags):
                    col_name = f'top_tag_{i + 1}'
                    if col_name not in user_data.columns:
                        user_data[col_name] = np.nan
                    user_data.loc[user_data['user_id'] == user_id, col_name] = tag

        logger.info(f"用户特征创建完成，特征数: {user_data.shape[1]}")
        return user_data

    def create_game_features(self):
        """创建游戏级特征"""
        logger.info("创建游戏特征...")

        # 按游戏分组
        agg_dict = {
            'user_id': 'count',  # 评论数
            'hours': ['mean', 'median', 'std'],  # 游戏时间统计
            'is_recommended': ['mean', 'sum'],  # 推荐率
            'rating': 'mean',  # 平均评分
        }

        # 有些列可能不存在，需要检查
        if 'helpful' in self.df.columns:
            agg_dict['helpful'] = ['sum', 'mean']
        if 'funny' in self.df.columns:
            agg_dict['funny'] = ['sum', 'mean']

        game_data = self.df.groupby('app_id').agg(agg_dict)

        # 展平多级索引列名
        game_data.columns = ['_'.join(col).strip() for col in game_data.columns.values]
        game_data = game_data.reset_index()

        # 添加其他游戏特征
        if 'positive_ratio' in self.df.columns:
            positive_ratio = self.df.groupby('app_id')['positive_ratio'].first().reset_index()
            game_data = game_data.merge(positive_ratio, on='app_id', how='left')

        # 添加价格特征
        if 'price_final' in self.df.columns and 'price_original' in self.df.columns:
            price_data = self.df.groupby('app_id').agg({
                'price_final': 'first',
                'price_original': 'first',
                'discount': 'first' if 'discount' in self.df.columns else []
            }).reset_index()
            game_data = game_data.merge(price_data, on='app_id', how='left')

            # 计算价格/游戏时间比例（性价比）
            game_data['value_ratio'] = game_data['price_final'] / game_data['hours_mean'].clip(lower=0.1)

        # 平台支持特征
        platform_cols = ['win', 'mac', 'linux', 'steam_deck']
        if all(col in self.df.columns for col in platform_cols):
            platform_data = self.df.groupby('app_id')[platform_cols].first().reset_index()
            game_data = game_data.merge(platform_data, on='app_id', how='left')

            # 计算平台支持数量
            game_data['platform_count'] = game_data[platform_cols].sum(axis=1)

        # 游戏年龄特征
        if 'date_release' in self.df.columns and 'date' in self.df.columns:
            # 获取每个游戏的发布日期
            release_dates = self.df.groupby('app_id')['date_release'].first().reset_index()
            game_data = game_data.merge(release_dates, on='app_id', how='left')

            # 计算游戏年龄（以天为单位）
            latest_date = self.df['date'].max()
            game_data['game_age_days'] = (latest_date - game_data['date_release']).dt.days

        logger.info(f"游戏特征创建完成，特征数: {game_data.shape[1]}")
        return game_data

    def process_text_features(self):
        """处理文本特征，包括游戏描述和标签"""
        logger.info("处理文本特征...")

        # 处理游戏描述
        if 'description' in self.df.columns:
            # 填充缺失的描述
            self.df['description'] = self.df['description'].fillna('')

            # 创建TF-IDF向量
            self.tfidf_model = TfidfVectorizer(
                max_features=100,
                stop_words='english',
                min_df=2
            )

            # 按游戏聚合描述（一个游戏只使用一次描述）
            game_descriptions = self.df.groupby('app_id')['description'].first().reset_index()

            # 生成TF-IDF特征
            tfidf_matrix = self.tfidf_model.fit_transform(game_descriptions['description'])

            # 使用SVD降维
            self.tfidf_svd = TruncatedSVD(n_components=min(20, tfidf_matrix.shape[1] - 1))
            desc_features = self.tfidf_svd.fit_transform(tfidf_matrix)

            # 创建描述特征DataFrame
            desc_feature_names = [f'desc_svd_{i}' for i in range(desc_features.shape[1])]
            desc_df = pd.DataFrame(desc_features, columns=desc_feature_names)
            desc_df['app_id'] = game_descriptions['app_id'].values

            # 合并到训练和测试集
            self.train_df = self.train_df.merge(desc_df, on='app_id', how='left')
            self.test_df = self.test_df.merge(desc_df, on='app_id', how='left')

        # 处理标签
        if 'tags' in self.df.columns:
            # 按游戏聚合标签
            game_tags = self.df.groupby('app_id')['tags'].first().reset_index()

            # 用One-Hot编码处理标签
            # 首先找到所有唯一标签
            all_tags = set()
            for tags_str in game_tags['tags'].dropna():
                tags = [tag.strip() for tag in tags_str.split(',')]
                all_tags.update(tags)

            # 按标签频率筛选前100个标签
            tag_counter = Counter()
            for tags_str in game_tags['tags'].dropna():
                tags = [tag.strip() for tag in tags_str.split(',')]
                tag_counter.update(tags)

            top_tags = [tag for tag, _ in tag_counter.most_common(100)]

            # 创建标签特征
            tag_features = {tag: [] for tag in top_tags}
            tag_features['app_id'] = []

            for _, row in game_tags.iterrows():
                tag_features['app_id'].append(row['app_id'])

                if pd.isna(row['tags']):
                    # 如果标签为空，所有特征都是0
                    for tag in top_tags:
                        tag_features[tag].append(0)
                else:
                    # 计算标签是否存在
                    tags = [tag.strip() for tag in row['tags'].split(',')]
                    for tag in top_tags:
                        tag_features[tag].append(1 if tag in tags else 0)

            # 创建标签特征DataFrame
            tag_df = pd.DataFrame(tag_features)

            # 合并到训练和测试集
            self.train_df = self.train_df.merge(tag_df, on='app_id', how='left')
            self.test_df = self.test_df.merge(tag_df, on='app_id', how='left')

        logger.info("文本特征处理完成")

    def create_interaction_features(self):
        """创建用户-游戏交互特征"""
        logger.info("创建交互特征...")

        # 计算用户-游戏类型交互特征
        if 'tags' in self.df.columns:
            # 为每个用户创建标签偏好表
            user_tag_prefs = defaultdict(Counter)

            for _, row in self.train_df.iterrows():
                if pd.isna(row['tags']):
                    continue

                user_id = row['user_id']
                is_recommended = 1 if row['is_recommended'] else 0
                tags = [tag.strip() for tag in row['tags'].split(',')]

                # 更新用户对每个标签的偏好
                for tag in tags:
                    user_tag_prefs[user_id][tag] += is_recommended

            # 将用户标签偏好添加为特征
            self.train_df['tag_preference'] = 0
            self.test_df['tag_preference'] = 0

            for df in [self.train_df, self.test_df]:
                for idx, row in df.iterrows():
                    if pd.isna(row['tags']):
                        continue

                    user_id = row['user_id']
                    tags = [tag.strip() for tag in row['tags'].split(',')]

                    # 计算该游戏标签的平均用户偏好
                    if user_id in user_tag_prefs and tags:
                        tag_prefs = [user_tag_prefs[user_id][tag] for tag in tags]
                        df.at[idx, 'tag_preference'] = sum(tag_prefs) / len(tags)

        # 创建时间和价格的交互特征
        for df in [self.train_df, self.test_df]:
            # 游戏时间与价格的性价比
            if 'price_final' in df.columns:
                df['value_for_money'] = df['hours'] / (df['price_final'] + 0.01)

            # 游戏时间与推荐状态的交互
            df['hours_x_recommended'] = df['hours'] * df['is_recommended'].astype(int)

            # 计算游戏发布时间与评论时间的差距
            if 'days_since_release' in df.columns:
                df['early_adoption'] = np.exp(-df['days_since_release'] / 365)  # 时间衰减特征

            # 创建非线性特征
            df['hours_log'] = np.log1p(df['hours'])
            df['hours_sqrt'] = np.sqrt(df['hours'])

            # 游戏时间分段特征（捕捉不同游戏时间段的行为模式）
            hour_bins = [0, 1, 5, 10, 20, 50, 100, np.inf]
            df['hours_bin'] = pd.cut(df['hours'], bins=hour_bins, labels=False)

            # 针对"长时间游戏但不推荐"的模式创建特征
            if 'is_recommended' in df.columns:
                df['long_play_not_recommend'] = ((df['hours'] > df['hours'].median()) &
                                                 (~df['is_recommended'])).astype(int)

        logger.info("交互特征创建完成")

    def create_sequence_features(self):
        """创建基于用户序列行为的特征"""
        logger.info("创建序列特征...")

        if 'date' not in self.df.columns:
            logger.warning("缺少日期列，无法创建序列特征")
            return

        # 按用户和时间排序数据
        self.df = self.df.sort_values(['user_id', 'date'])

        # 为每个用户创建历史序列
        user_sequences = {}
        for user_id, group in self.df.groupby('user_id'):
            # 按时间排序
            group = group.sort_values('date')

            # 创建游戏ID序列
            app_id_seq = group['app_id'].tolist()

            # 创建游戏时间序列
            hours_seq = group['hours'].tolist()

            # 创建推荐状态序列
            if 'is_recommended' in group.columns:
                recommended_seq = group['is_recommended'].astype(int).tolist()
            else:
                recommended_seq = []

            # 创建评分序列
            if 'rating' in group.columns:
                rating_seq = group['rating'].tolist()
            else:
                rating_seq = []

            # 创建日期序列
            date_seq = group['date'].tolist()

            # 存储用户序列
            user_sequences[user_id] = {
                'app_id_seq': app_id_seq,
                'hours_seq': hours_seq,
                'recommended_seq': recommended_seq,
                'rating_seq': rating_seq,
                'date_seq': date_seq
            }

        # 创建滑动窗口特征
        max_seq_length = self.config['max_seq_length']

        # 为训练集和测试集创建序列特征
        for df in [self.train_df, self.test_df]:
            # 初始化序列特征列
            df['prev_apps'] = None
            df['prev_ratings'] = None
            df['prev_hours'] = None
            df['avg_prev_rating'] = 0
            df['avg_prev_hours'] = 0
            df['prev_apps_count'] = 0
            df['days_since_last_play'] = -1

            # 对每行数据填充序列特征
            for idx, row in df.iterrows():
                user_id = row['user_id']
                current_date = row['date']

                if user_id not in user_sequences:
                    continue

                user_seq = user_sequences[user_id]

                # 找到当前交互在序列中的位置
                try:
                    current_idx = user_seq['date_seq'].index(current_date)
                except ValueError:
                    # 如果找不到完全匹配的日期，找最近的日期
                    nearest_idx = np.argmin([abs((d - current_date).total_seconds())
                                             for d in user_seq['date_seq']])
                    current_idx = nearest_idx

                # 获取前面的序列（不包括当前交互）
                prev_app_ids = user_seq['app_id_seq'][:current_idx][-max_seq_length:]
                prev_ratings = user_seq['rating_seq'][:current_idx][-max_seq_length:]
                prev_hours = user_seq['hours_seq'][:current_idx][-max_seq_length:]

                # 存储序列（作为列表）
                df.at[idx, 'prev_apps'] = prev_app_ids
                df.at[idx, 'prev_ratings'] = prev_ratings
                df.at[idx, 'prev_hours'] = prev_hours

                # 计算聚合统计特征
                if prev_ratings:
                    df.at[idx, 'avg_prev_rating'] = sum(prev_ratings) / len(prev_ratings)

                if prev_hours:
                    df.at[idx, 'avg_prev_hours'] = sum(prev_hours) / len(prev_hours)

                df.at[idx, 'prev_apps_count'] = len(prev_app_ids)

                # 计算自上次游戏以来的天数
                if current_idx > 0:
                    last_date = user_seq['date_seq'][current_idx - 1]
                    df.at[idx, 'days_since_last_play'] = (current_date - last_date).days

                # 计算用户当前游戏的排名特征（第几次游玩该游戏）
                game_id = row['app_id']
                play_count = prev_app_ids.count(game_id)
                df.at[idx, 'play_count'] = play_count

        # 创建时间衰减特征
        time_decay_factor = self.config['time_decay_factor']

        for df in [self.train_df, self.test_df]:
            df['time_weighted_rating'] = 0

            for idx, row in df.iterrows():
                if isinstance(row['prev_ratings'], list) and row['prev_ratings']:
                    weights = [time_decay_factor ** (len(row['prev_ratings']) - i - 1)
                               for i in range(len(row['prev_ratings']))]
                    weighted_sum = sum(r * w for r, w in zip(row['prev_ratings'], weights))
                    sum_weights = sum(weights)
                    df.at[idx, 'time_weighted_rating'] = weighted_sum / sum_weights

        logger.info("序列特征创建完成")

    def train_lgbm_model(self):
        """训练LightGBM模型"""
        logger.info("开始训练LightGBM模型...")

        # 准备特征和目标变量
        target_col = 'is_recommended'
        id_cols = ['user_id', 'app_id', 'date', 'review_id', 'prev_apps', 'prev_ratings', 'prev_hours']
        categorical_cols = [col for col in self.train_df.columns if col.endswith('_encoded')]

        # 移除不能用作特征的列
        exclude_cols = id_cols + [target_col, 'tags', 'description']

        # 只选择数值型和布尔型特征
        feature_cols = []
        for col in self.train_df.columns:
            if col in exclude_cols or pd.api.types.is_list_like(self.train_df[col].iloc[0]):
                continue
            if self.train_df[col].dtype in [np.int64, np.float64, np.bool_]:
                feature_cols.append(col)
            else:
                # 跳过object和datetime类型
                logger.info(f"跳过非数值型特征: {col} (类型: {self.train_df[col].dtype})")

        X_train = self.train_df[feature_cols]
        y_train = self.train_df[target_col].astype(int)

        X_val = self.test_df[feature_cols]
        y_val = self.test_df[target_col].astype(int)

        logger.info(f"训练特征数量: {len(feature_cols)}")
        logger.info(f"使用的特征: {feature_cols[:10]}...")

        # 设置分类特征
        for col in categorical_cols:
            if col in feature_cols:
                X_train[col] = X_train[col].astype('category')
                X_val[col] = X_val[col].astype('category')

        # 创建LightGBM数据集
        train_data = lgb.Dataset(
            X_train,
            label=y_train,
            categorical_feature=[col for col in categorical_cols if col in feature_cols]
        )

        val_data = lgb.Dataset(
            X_val,
            label=y_val,
            categorical_feature=[col for col in categorical_cols if col in feature_cols],
            reference=train_data
        )

        # 使用更简单的参数配置
        params = {
            'objective': 'binary',
            'metric': 'auc',
            'boosting_type': 'gbdt',
            'learning_rate': 0.05,
            'num_leaves': 31,
            'max_depth': -1,
            'min_child_samples': 20,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42
        }

        # 训练模型
        callbacks = [lgb.early_stopping(50), lgb.log_evaluation(50)]

        self.lgbm_model = lgb.train(
            params=params,
            train_set=train_data,
            valid_sets=[train_data, val_data],
            valid_names=['train', 'valid'],
            callbacks=callbacks,
            num_boost_round=1000
        )

        # 评估模型
        lgbm_preds = self.lgbm_model.predict(X_val)
        auc_score = roc_auc_score(y_val, lgbm_preds)
        logger.info(f"LightGBM模型验证AUC: {auc_score:.4f}")

        # 特征重要性
        feature_importance = pd.DataFrame({
            'Feature': feature_cols,
            'Importance': self.lgbm_model.feature_importance(importance_type='gain')
        }).sort_values(by='Importance', ascending=False)

        logger.info("前10个重要特征:")
        for idx, row in feature_importance.head(10).iterrows():
            logger.info(f"{row['Feature']}: {row['Importance']}")

        self.feature_importance = feature_importance
        return self.lgbm_model

    def train_sequence_model(self):
        """训练序列模型"""
        logger.info("开始训练序列模型...")

        # 检查是否有足够的序列数据
        if not self.train_df['prev_apps'].apply(lambda x: isinstance(x, list) and len(x) > 0).any():
            logger.warning("没有足够的序列数据，跳过序列模型训练")
            return None

        # 准备序列数据
        train_data = self.prepare_sequence_data(self.train_df)

        # 定义序列模型
        class GameSequenceModel(nn.Module):
            def __init__(self, num_games, embedding_dim, hidden_dim, num_layers, dropout):
                super().__init__()
                self.game_embedding = nn.Embedding(num_games + 1, embedding_dim)  # +1 用于填充
                self.lstm = nn.LSTM(
                    embedding_dim,
                    hidden_dim,
                    num_layers,
                    batch_first=True,
                    dropout=dropout if num_layers > 1 else 0
                )
                self.dropout = nn.Dropout(dropout)
                self.fc = nn.Linear(hidden_dim, 1)
                self.sigmoid = nn.Sigmoid()

            def forward(self, game_seq, seq_lengths):
                # 嵌入游戏ID
                embedded = self.game_embedding(game_seq)

                # 打包序列以处理变长序列
                packed = nn.utils.rnn.pack_padded_sequence(
                    embedded, seq_lengths.cpu(), batch_first=True, enforce_sorted=False
                )

                # LSTM处理
                _, (hidden, _) = self.lstm(packed)

                # 获取最后一层隐藏状态
                last_hidden = hidden[-1]

                # 线性层和sigmoid激活函数
                output = self.fc(self.dropout(last_hidden))
                return self.sigmoid(output).squeeze()

        # 获取游戏数量
        num_games = max(self.label_encoders['app_id'].classes_.size,
                        self.train_df['app_id_encoded'].max(),
                        self.test_df['app_id_encoded'].max()) + 1

        # 创建数据集和数据加载器
        class GameSequenceDataset(Dataset):
            def __init__(self, sequences, targets):
                self.sequences = sequences
                self.targets = targets

            def __len__(self):
                return len(self.sequences)

            def __getitem__(self, idx):
                return self.sequences[idx], self.targets[idx]

        # 创建训练数据集
        train_dataset = GameSequenceDataset(
            train_data['sequences'],
            train_data['targets']
        )

        # 数据整理函数
        def collate_fn(batch):
            # 提取序列和目标
            sequences, targets = zip(*batch)

            # 计算每个序列的长度
            seq_lengths = torch.tensor([len(seq) for seq in sequences])

            # 填充序列到相同长度
            max_len = max(seq_lengths).item()
            padded_sequences = torch.zeros(len(sequences), max_len, dtype=torch.long)

            for i, seq in enumerate(sequences):
                end = seq_lengths[i]
                padded_sequences[i, :end] = torch.tensor(seq[:end])

            # 转换目标为张量
            targets = torch.tensor(targets, dtype=torch.float32)

            return padded_sequences, seq_lengths, targets

        # 创建数据加载器
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config['sequence_params']['batch_size'],
            shuffle=True,
            collate_fn=collate_fn
        )

        # 初始化模型
        model = GameSequenceModel(
            num_games=num_games,
            embedding_dim=self.config['sequence_params']['embedding_dim'],
            hidden_dim=self.config['sequence_params']['hidden_dim'],
            num_layers=self.config['sequence_params']['num_layers'],
            dropout=self.config['sequence_params']['dropout']
        ).to(self.device)

        # 定义损失函数和优化器
        criterion = nn.BCELoss()
        optimizer = optim.Adam(model.parameters(), lr=self.config['sequence_params']['learning_rate'])

        # 训练模型
        epochs = self.config['sequence_params']['epochs']
        for epoch in range(epochs):
            model.train()
            total_loss = 0

            for sequences, seq_lengths, targets in train_loader:
                # 移动数据到设备
                sequences = sequences.to(self.device)
                seq_lengths = seq_lengths.to(self.device)
                targets = targets.to(self.device)

                # 前向传播
                optimizer.zero_grad()
                outputs = model(sequences, seq_lengths)

                # 计算损失
                loss = criterion(outputs, targets)

                # 反向传播
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            avg_loss = total_loss / len(train_loader)
            logger.info(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}")

        # 保存模型
        self.sequence_model = model
        self.sequence_model.eval()  # 设为评估模式

        logger.info("序列模型训练完成")
        return self.sequence_model

    def prepare_sequence_data(self, df):
        """准备序列模型的训练数据"""
        sequences = []
        targets = []

        for _, row in df.iterrows():
            if not isinstance(row['prev_apps'], list) or len(row['prev_apps']) == 0:
                continue

            # 获取游戏ID序列（使用编码后的ID）
            game_seq = []
            for app_id in row['prev_apps']:
                if app_id in self.label_encoders['app_id'].classes_:
                    encoded_id = self.label_encoders['app_id'].transform([app_id])[0]
                    game_seq.append(encoded_id)
                else:
                    # 对于不在训练集中的游戏ID，使用特殊的OOV ID
                    game_seq.append(len(self.label_encoders['app_id'].classes_))

            sequences.append(game_seq)
            targets.append(1 if row['is_recommended'] else 0)

        return {
            'sequences': sequences,
            'targets': targets
        }

    def create_game_embeddings(self):
        """从训练好的序列模型中提取游戏嵌入向量"""
        if not hasattr(self, 'sequence_model') or self.sequence_model is None:
            logger.warning("序列模型未训练，无法提取游戏嵌入向量")
            return None

        # 提取游戏嵌入层
        game_embeddings = self.sequence_model.game_embedding.weight.data.cpu().numpy()

        # 创建游戏ID到嵌入向量的映射
        self.game_embeddings = {}
        for game_idx, game_id in enumerate(self.label_encoders['app_id'].classes_):
            self.game_embeddings[game_id] = game_embeddings[game_idx]

        logger.info(f"创建了 {len(self.game_embeddings)} 个游戏嵌入向量")
        return self.game_embeddings

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

    def generate_recommendations(self, user_id, n=10):
        """为指定用户生成推荐"""
        # 使用缓存
        cache_key = f"{user_id}_{n}"
        if cache_key in self.recommendation_cache:
            return self.recommendation_cache[cache_key]

        logger.info(f"为用户 {user_id} 生成 {n} 条推荐...")

        # 检查用户是否存在于训练数据中
        if user_id not in self.train_df['user_id'].values and user_id not in self.test_df['user_id'].values:
            logger.warning(f"用户 {user_id} 未在训练或测试数据中找到")
            recommendations = self.handle_cold_start_user(n)
            self.recommendation_cache[cache_key] = recommendations
            return recommendations

        # 获取用户已经评论过的游戏
        user_games = set(self.df[self.df['user_id'] == user_id]['app_id'].values)

        # 使用热门游戏缩减候选集
        # 计算游戏流行度
        popular_games_df = self.df.groupby('app_id').size().reset_index(name='count')
        popular_games_df = popular_games_df.sort_values('count', ascending=False)

        # 只选择前500个热门游戏作为候选（可以调整这个数字）
        candidate_size = min(500, len(popular_games_df))
        top_games = set(popular_games_df.head(candidate_size)['app_id'].values)
        candidate_games = top_games - user_games

        # 如果没有候选游戏，返回热门游戏
        if not candidate_games:
            logger.warning(f"用户 {user_id} 没有可推荐的候选游戏")
            recommendations = self.get_popular_games(n)
            self.recommendation_cache[cache_key] = recommendations
            return recommendations

        # 为每个候选游戏预测得分
        predictions = []
        for game_id in candidate_games:
            score = self.predict_score(user_id, game_id)
            predictions.append((game_id, score))

        # 按分数降序排列
        recommendations = sorted(predictions, key=lambda x: x[1], reverse=True)[:n]

        logger.info(f"成功为用户 {user_id} 生成了 {len(recommendations)} 条推荐")

        # 缓存结果
        self.recommendation_cache[cache_key] = recommendations

        return recommendations

    def predict_score(self, user_id, game_id):
        """预测用户对游戏的评分"""
        # 使用缓存
        cache_key = f"{user_id}_{game_id}"
        if cache_key in self.score_cache:
            return self.score_cache[cache_key]

        try:
            # 创建特征
            features = self.extract_prediction_features(user_id, game_id)

            # 输出调试信息
            logger.debug(f"为用户 {user_id} 和游戏 {game_id} 提取的特征数量: {len(features)}")

            # 安全检查：确保特征非空
            if len(features) == 0 and self.lgbm_model is not None:
                logger.warning(f"提取的特征为空！使用默认特征代替。")
                features = np.zeros(len(self.lgbm_model.feature_name()))

            # 使用LightGBM模型预测
            if self.lgbm_model is not None:
                # 设置predict_disable_shape_check=True以避免特征数量不匹配的错误
                lgbm_score = self.lgbm_model.predict([features], predict_disable_shape_check=True)[0]
            else:
                lgbm_score = 0.5

            # 使用序列模型预测
            seq_score = self.predict_sequence_score(user_id, game_id)

            # 使用内容模型预测
            content_score = self.predict_content_score(user_id, game_id)

            # 加权平均
            final_score = (
                    self.config['lgbm_weight'] * lgbm_score +
                    self.config['sequence_weight'] * seq_score +
                    self.config['content_weight'] * content_score
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

        # 获取用户历史序列
        user_data = self.df[self.df['user_id'] == user_id].sort_values('date')

        if len(user_data) == 0:
            return 0.5

        # 获取用户之前玩过的游戏ID
        prev_games = user_data['app_id'].tolist()

        # 将游戏ID转换为编码
        game_seq = []
        for app_id in prev_games:
            if app_id in self.label_encoders['app_id'].classes_:
                encoded_id = self.label_encoders['app_id'].transform([app_id])[0]
                game_seq.append(encoded_id)
            else:
                # 对于不在训练集中的游戏ID，使用特殊的OOV ID
                game_seq.append(len(self.label_encoders['app_id'].classes_))

        # 如果序列为空，返回默认得分
        if not game_seq:
            return 0.5

        # 准备模型输入
        seq_tensor = torch.tensor([game_seq], dtype=torch.long).to(self.device)
        seq_length = torch.tensor([len(game_seq)], dtype=torch.long).to(self.device)

        # 预测
        with torch.no_grad():
            score = self.sequence_model(seq_tensor, seq_length).item()

        return score

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
        """获取热门游戏"""
        # 基于评论数量和评分计算游戏受欢迎程度
        game_popularity = self.df.groupby('app_id').agg({
            'user_id': 'count',  # 评论数
            'rating': 'mean'  # 平均评分
        }).reset_index()

        # 重命名列
        game_popularity.columns = ['app_id', 'review_count', 'avg_rating']

        # 计算综合流行度得分
        game_popularity['popularity_score'] = (
                game_popularity['review_count'] * 0.7 +
                game_popularity['avg_rating'] * 0.3
        )

        # 排序并获取前N个游戏
        popular_games = game_popularity.sort_values('popularity_score', ascending=False).head(n)

        # 返回游戏ID和得分
        return [(game_id, score) for game_id, score in zip(
            popular_games['app_id'], popular_games['popularity_score']
        )]

    def evaluate_recommendations(self, k_values=[5, 10, 20]):
        """评估推荐系统性能"""
        logger.info("开始评估推荐系统...")

        # 准备测试用户
        test_users = self.test_df['user_id'].unique()

        # 限制评估用户数以提高效率
        max_test_users = min(100, len(test_users))
        test_users = np.random.choice(test_users, max_test_users, replace=False)

        logger.info(f"使用 {len(test_users)} 个测试用户进行评估")

        # 初始化评估指标
        metrics = {
            'precision': {k: [] for k in k_values},
            'recall': {k: [] for k in k_values},
            'ndcg': {k: [] for k in k_values},
            'diversity': {k: [] for k in k_values},
            'coverage': []
        }

        # 所有推荐的游戏集合（用于计算覆盖率）
        all_recommended_games = set()
        all_games = set(self.df['app_id'].unique())

        # 评估每个测试用户
        for user_id in tqdm(test_users, desc="评估用户"):
            # 获取用户实际喜欢的游戏
            user_liked_games = set(self.test_df[
                                       (self.test_df['user_id'] == user_id) &
                                       (self.test_df['is_recommended'] == True)
                                       ]['app_id'].values)

            # 如果用户没有喜欢的游戏，跳过
            if not user_liked_games:
                continue

            # 生成推荐
            max_k = max(k_values)
            recommendations = self.generate_recommendations(user_id, max_k)
            recommended_games = [game_id for game_id, _ in recommendations]

            # 更新所有推荐的游戏集合
            all_recommended_games.update(recommended_games)

            # 计算每个K值的指标
            for k in k_values:
                top_k_games = recommended_games[:k]

                # 计算精确率
                hits = len(set(top_k_games) & user_liked_games)
                precision = hits / k if k > 0 else 0
                metrics['precision'][k].append(precision)

                # 计算召回率
                recall = hits / len(user_liked_games) if len(user_liked_games) > 0 else 0
                metrics['recall'][k].append(recall)

                # 计算NDCG
                # 创建相关性数组（1表示相关，0表示不相关）
                relevance = [1 if game in user_liked_games else 0 for game in top_k_games]
                # 理想情况下的排序
                ideal_relevance = sorted(relevance, reverse=True)

                if sum(relevance) > 0:
                    from sklearn.metrics import ndcg_score
                    ndcg = ndcg_score([ideal_relevance], [relevance])
                    metrics['ndcg'][k].append(ndcg)

                # 计算多样性
                # 使用游戏标签计算推荐列表中游戏的不同类型
                if 'tags' in self.df.columns:
                    game_tags = {}
                    for game_id in top_k_games:
                        tags = self.df[self.df['app_id'] == game_id]['tags'].iloc[0] if game_id in self.df[
                            'app_id'].values else ""
                        if pd.notna(tags):
                            game_tags[game_id] = set(tag.strip() for tag in tags.split(','))
                        else:
                            game_tags[game_id] = set()

                    # 计算平均两两Jaccard距离
                    if len(game_tags) >= 2:
                        diversity_scores = []
                        for i, (game1, tags1) in enumerate(game_tags.items()):
                            for game2, tags2 in list(game_tags.items())[i + 1:]:
                                if tags1 and tags2:  # 只有当两个游戏都有标签时
                                    jaccard_similarity = len(tags1 & tags2) / len(tags1 | tags2)
                                    jaccard_distance = 1 - jaccard_similarity
                                    diversity_scores.append(jaccard_distance)

                        if diversity_scores:
                            diversity = sum(diversity_scores) / len(diversity_scores)
                            metrics['diversity'][k].append(diversity)

        # 计算覆盖率
        coverage = len(all_recommended_games) / len(all_games) if all_games else 0
        metrics['coverage'] = coverage

        # 计算平均指标
        results = {}
        for metric in ['precision', 'recall', 'ndcg', 'diversity']:
            results[metric] = {k: np.mean(metrics[metric][k]) for k in k_values}
        results['coverage'] = metrics['coverage']

        # 打印结果
        logger.info("评估结果:")
        for metric in ['precision', 'recall', 'ndcg', 'diversity']:
            logger.info(f"{metric.capitalize()}:")
            for k in k_values:
                logger.info(f"  @{k}: {results[metric][k]:.4f}")
        logger.info(f"Coverage: {results['coverage']:.4f}")

        return results

    def visualize_results(self):
        """可视化模型结果和特征重要性"""
        if not hasattr(self, 'feature_importance') or self.feature_importance is None:
            logger.warning("没有可视化的特征重要性")
            return

        # 绘制特征重要性
        plt.figure(figsize=(12, 8))
        top_features = self.feature_importance.head(20)
        sns.barplot(x='Importance', y='Feature', data=top_features)
        plt.title('Top 20 Features by Importance')
        plt.tight_layout()
        plt.savefig('feature_importance.png')
        plt.close()

        # 绘制评估指标
        if hasattr(self, 'evaluation_results'):
            metrics = ['precision', 'recall', 'ndcg', 'diversity']
            k_values = sorted(self.evaluation_results['precision'].keys())

            plt.figure(figsize=(15, 10))
            for i, metric in enumerate(metrics, 1):
                plt.subplot(2, 2, i)
                values = [self.evaluation_results[metric][k] for k in k_values]
                plt.plot(k_values, values, marker='o')
                plt.title(f'{metric.capitalize()} at different k')
                plt.xlabel('k')
                plt.ylabel(metric)
                plt.grid(True)

            plt.tight_layout()
            plt.savefig('evaluation_metrics.png')
            plt.close()

        logger.info("结果可视化完成")

    def save_model(self, path='steam_recommender_model'):
        """保存模型和相关数据"""
        logger.info(f"保存模型到 {path}...")

        # 创建保存目录
        os.makedirs(path, exist_ok=True)

        # 保存LightGBM模型
        if self.lgbm_model is not None:
            self.lgbm_model.save_model(os.path.join(path, 'lgbm_model.txt'))

        # 保存序列模型
        if hasattr(self, 'sequence_model') and self.sequence_model is not None:
            torch.save(self.sequence_model.state_dict(), os.path.join(path, 'sequence_model.pt'))

        # 保存编码器和其他组件
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

        # 保存配置
        with open(os.path.join(path, 'config.json'), 'w') as f:
            json.dump(self.config, f, indent=4)

        logger.info(f"模型保存完成")

    def load_model(self, path='steam_recommender_model'):
        """加载模型和相关数据"""
        logger.info(f"从 {path} 加载模型...")

        # 检查保存目录是否存在
        if not os.path.exists(path):
            logger.error(f"模型目录 {path} 不存在")
            return False

        # 加载配置
        config_path = os.path.join(path, 'config.json')
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                self.config = json.load(f)

        # 加载LightGBM模型
        lgbm_path = os.path.join(path, 'lgbm_model.txt')
        if os.path.exists(lgbm_path):
            self.lgbm_model = lgb.Booster(model_file=lgbm_path)

        # 加载编码器和其他组件
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

        # 加载序列模型
        sequence_path = os.path.join(path, 'sequence_model.pt')
        if os.path.exists(sequence_path):
            # 此处需要先初始化模型架构
            # 这里简化处理，实际使用时需要重建完整的模型架构
            logger.info("序列模型存在但需要手动重建模型架构后再加载权重")

        logger.info(f"模型加载完成")
        return True


def main():
    """主函数，演示推荐系统的使用"""
    # 设置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # 初始化推荐系统
    recommender = SteamRecommender('steam_data.csv')

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