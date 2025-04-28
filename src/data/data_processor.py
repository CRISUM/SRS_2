#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
src/data/data_processor.py - Data processing module
Author: YourName
Date: 2025-04-27
Description: Handles data loading, preprocessing, and feature engineering
"""
import traceback

import pandas as pd
import numpy as np
import logging
import gc
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)


class SteamDataProcessor:
    """Data processor for Steam recommendation system"""

    def __init__(self, config=None):
        """Initialize data processor"""
        self.config = config or {}
        self.df = None
        self.train_df = None
        self.test_df = None

    def optimize_datatypes(self):
        """Optimize DataFrame data types to reduce memory usage"""
        for col in self.df.columns:
            if self.df[col].dtype == 'float64':
                self.df[col] = self.df[col].astype('float32')
            if self.df[col].dtype == 'int64':
                self.df[col] = self.df[col].astype('int32')

        logger.info("Optimized data types to reduce memory usage")

    def load_data(self, data_path, chunk_size=500000):
        """Load data in chunks"""
        logger.info(f"Loading data in chunks: {data_path}")

        # First read header to get column names
        try:
            header_df = pd.read_csv(data_path, nrows=1)
            column_names = header_df.columns.tolist()
            logger.info(f"Data columns: {column_names}")

            # 检查必要的列
            required_cols = ['user_id', 'app_id', 'is_recommended', 'hours', 'title', 'tags']
            missing_cols = [col for col in required_cols if col not in column_names]
            if missing_cols:
                logger.error(f"CSV missing required columns: {missing_cols}")
                # 尝试匹配列名
                column_lower = [col.lower() for col in column_names]
                for missing in missing_cols:
                    possible_matches = [column_names[i] for i, col in enumerate(column_lower) if
                                        col == missing.lower()]
                    if possible_matches:
                        logger.info(f"'{missing}' possible matches: {possible_matches}")
                return False
        except Exception as e:
            logger.error(f"Error reading CSV header: {str(e)}")
            return False

        # 初始化统计信息跟踪
        unique_users = set()
        unique_games = set()
        user_stats = {}
        game_stats = {}
        chunk_count = 0
        all_data = []  # 用于存储所有数据

        try:
            for chunk in pd.read_csv(data_path, chunksize=chunk_size):
                chunk_count += 1
                logger.info(f"Processing data chunk {chunk_count}")

                # 确保必要的列存在
                for col in required_cols:
                    if col not in chunk.columns:
                        logger.warning(f"Chunk {chunk_count} missing column '{col}', skipping")
                        continue

                # 更新统计信息
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

                # 存储当前数据块
                all_data.append(chunk)

                # 释放内存
                del chunk
                gc.collect()

            logger.info(f"Processed {chunk_count} data chunks")

            # 将统计信息转换为 DataFrames
            self.user_df = pd.DataFrame.from_dict(user_stats, orient='index')
            self.user_df.reset_index(inplace=True)
            self.user_df.rename(columns={'index': 'user_id'}, inplace=True)

            self.game_df = pd.DataFrame.from_dict(game_stats, orient='index')
            self.game_df.reset_index(inplace=True)
            self.game_df.rename(columns={'index': 'app_id'}, inplace=True)

            # 计算额外统计信息
            self.user_df['recommendation_ratio'] = self.user_df['recommended_count'] / self.user_df['game_count']
            self.game_df['recommendation_ratio'] = self.game_df['recommended_count'] / self.game_df['user_count']
            self.game_df['avg_hours'] = self.game_df['total_hours'] / self.game_df['user_count']

            # 合并所有数据块创建主数据框
            if all_data:
                self.df = pd.concat(all_data, ignore_index=True)
            else:
                logger.warning("No data chunks were processed")
                return False

            # 创建训练和测试样本
            self.split_data_into_train_test()

            logger.info(f"Data loading complete, found {len(unique_users)} unique users and {len(unique_games)} games")
            return True

        except Exception as e:
            logger.error(f"Error processing data: {str(e)}")
            logger.error(traceback.format_exc())
            return False

    def engineer_features(self):
        """Generate features for recommendation models"""
        logger.info("Starting feature engineering...")

        if not hasattr(self, 'train_df') or self.train_df is None:
            logger.error("No training data available for feature engineering")
            return False

        try:
            # 1. Create user features
            user_features = self.train_df.groupby('user_id').agg({
                'app_id': 'count',
                'hours': ['sum', 'mean', 'max'],
                'is_recommended': ['mean', 'sum']
            })

            # Flatten column names
            user_features.columns = ['_'.join(col).strip() for col in user_features.columns.values]
            user_features.reset_index(inplace=True)

            # Rename columns for clarity
            user_features.rename(columns={
                'app_id_count': 'user_game_count',
                'is_recommended_mean': 'user_recommendation_ratio',
                'is_recommended_sum': 'user_recommended_count',
                'hours_sum': 'user_total_hours',
                'hours_mean': 'user_avg_hours',
                'hours_max': 'user_max_hours'
            }, inplace=True)

            # 2. Create game features
            game_features = self.train_df.groupby('app_id').agg({
                'user_id': 'count',
                'hours': ['sum', 'mean', 'max'],
                'is_recommended': ['mean', 'sum']
            })

            # Flatten column names
            game_features.columns = ['_'.join(col).strip() for col in game_features.columns.values]
            game_features.reset_index(inplace=True)

            # Rename columns for clarity
            game_features.rename(columns={
                'user_id_count': 'game_user_count',
                'is_recommended_mean': 'game_recommendation_ratio',
                'is_recommended_sum': 'game_recommended_count',
                'hours_sum': 'game_total_hours',
                'hours_mean': 'game_avg_hours',
                'hours_max': 'game_max_hours'
            }, inplace=True)

            # 3. Process game tags
            if 'tags' in self.train_df.columns:
                # Extract top tags per game
                top_tags = {}
                tag_counts = {}

                for app_id, tags_str in zip(self.train_df['app_id'], self.train_df['tags']):
                    if pd.notna(tags_str) and tags_str:
                        tags = [tag.strip() for tag in tags_str.split(',')]
                        top_tags[app_id] = tags[:3]  # Keep top 3 tags
                        for tag in tags:
                            tag_counts[tag] = tag_counts.get(tag, 0) + 1

                # Get most common tags
                common_tags = sorted(tag_counts.items(), key=lambda x: x[1], reverse=True)[:50]
                common_tags = [tag for tag, _ in common_tags]

                # Create tag features for games
                for i, tag in enumerate(common_tags[:10]):
                    tag_col = f'top_tag_{i + 1}'
                    game_features[tag_col] = game_features['app_id'].apply(
                        lambda app_id: 1 if app_id in top_tags and tag in top_tags[app_id] else 0
                    )

            # 4. Create interaction features
            # Merge user and game features into training data
            self.train_df = self.train_df.merge(user_features, on='user_id', how='left')
            self.train_df = self.train_df.merge(game_features, on='app_id', how='left')

            # Do the same for test data
            self.test_df = self.test_df.merge(user_features, on='user_id', how='left')
            self.test_df = self.test_df.merge(game_features, on='app_id', how='left')

            # 5. Create user preference match features
            # Calculate preference match between user history and game tags
            if 'tags' in self.train_df.columns:
                # Function to extract user's preferred tags from history
                def get_user_preferred_tags(user_id):
                    user_games = self.train_df[self.train_df['user_id'] == user_id]
                    user_tags = []
                    for tags_str in user_games['tags'].dropna():
                        if tags_str:
                            user_tags.extend([tag.strip() for tag in tags_str.split(',')])

                    # Return most frequent tags
                    tag_counter = {}
                    for tag in user_tags:
                        tag_counter[tag] = tag_counter.get(tag, 0) + 1

                    # Sort by frequency
                    sorted_tags = sorted(tag_counter.items(), key=lambda x: x[1], reverse=True)
                    return [tag for tag, _ in sorted_tags[:10]]

                # Create user tag preferences
                user_tag_prefs = {}
                for user_id in self.train_df['user_id'].unique():
                    user_tag_prefs[user_id] = get_user_preferred_tags(user_id)

                # Calculate tag match ratio for each user-game pair
                def calc_tag_match(row):
                    user_id, tags_str = row['user_id'], row['tags']
                    if user_id not in user_tag_prefs or pd.isna(tags_str) or not tags_str:
                        return 0.0

                    game_tags = [tag.strip() for tag in tags_str.split(',')]
                    user_tags = user_tag_prefs[user_id]

                    matches = len(set(game_tags) & set(user_tags))
                    total = len(set(game_tags) | set(user_tags)) if len(set(game_tags) | set(user_tags)) > 0 else 1

                    return matches / total

                # Add tag match feature
                self.train_df['tag_match'] = self.train_df.apply(calc_tag_match, axis=1)
                self.test_df['tag_match'] = self.test_df.apply(calc_tag_match, axis=1)

            # 6. Create user data dictionary with time information
            user_data = {}
            for user_id in self.train_df['user_id'].unique():
                user_interactions = self.train_df[self.train_df['user_id'] == user_id]

                # 构建交互列表
                interactions = []
                for _, row in user_interactions.iterrows():
                    interaction = {
                        'app_id': row['app_id'],
                        'is_recommended': row['is_recommended'],
                        'hours': row['hours'],
                    }

                    # 如果有日期信息，添加到交互中
                    if 'date' in row and pd.notna(row['date']):
                        interaction['date'] = row['date']

                    interactions.append(interaction)

                # 获取用户喜欢的游戏和标签
                liked_games = user_interactions[user_interactions['is_recommended'] == True]['app_id'].tolist()

                tag_preferences = {}
                for _, row in user_interactions.iterrows():
                    if 'tags' in row and pd.notna(row['tags']) and row['tags']:
                        tags = [tag.strip() for tag in row['tags'].split(',')]
                        for tag in tags:
                            if tag not in tag_preferences:
                                tag_preferences[tag] = 0
                            weight = 2 if row['is_recommended'] else 1
                            if pd.notna(row['hours']):
                                weight *= min(row['hours'] / 10, 3)
                            tag_preferences[tag] += weight

                sorted_tags = sorted(tag_preferences.items(), key=lambda x: x[1], reverse=True)
                top_tags = [tag for tag, _ in sorted_tags[:10]]

                user_data[user_id] = {
                    'interactions': interactions,
                    'liked_games': liked_games,
                    'top_tags': top_tags,
                    'tag_preferences': dict(sorted_tags)
                }

            # 保存用户数据
            self.user_data = user_data

            logger.info("Feature engineering completed")
            return True

        except Exception as e:
            logger.error(f"Error in feature engineering: {str(e)}")
            logger.error(traceback.format_exc())
            return False

    def create_sequence_features(self):
        """Create sequential features based on user history"""
        if not hasattr(self, 'train_df') or self.train_df is None:
            logger.error("No training data available for sequence features")
            return

        try:
            # Sort data by user and timestamp (if available)
            logger.info("Creating sequence features from user history...")

            if 'date' in self.train_df.columns:
                self.train_df = self.train_df.sort_values(['user_id', 'date'])
            else:
                self.train_df = self.train_df.sort_values('user_id')

            # 首先创建新列
            self.train_df['prev_apps'] = None
            self.train_df['prev_ratings'] = None
            self.train_df['prev_hours'] = None

            # 同样为测试数据创建列
            self.test_df['prev_apps'] = None
            self.test_df['prev_ratings'] = None
            self.test_df['prev_hours'] = None

            # Create user history sequences
            user_sequences = {}

            for user_id in self.train_df['user_id'].unique():
                user_data = self.train_df[self.train_df['user_id'] == user_id]

                # Create sequences of previous games, ratings, and hours
                prev_apps = []
                prev_ratings = []
                prev_hours = []

                for idx, row in user_data.iterrows():
                    # Store current interaction's sequence data
                    self.train_df.at[idx, 'prev_apps'] = prev_apps[-self.config['max_seq_length']:] if prev_apps else []
                    self.train_df.at[idx, 'prev_ratings'] = prev_ratings[
                                                            -self.config['max_seq_length']:] if prev_ratings else []
                    self.train_df.at[idx, 'prev_hours'] = prev_hours[
                                                          -self.config['max_seq_length']:] if prev_hours else []

                    # Update sequences for next interaction
                    prev_apps.append(row['app_id'])
                    prev_ratings.append(1 if row['is_recommended'] else 0)
                    prev_hours.append(row['hours'])

            # Create similar sequence features for test data
            if 'date' in self.test_df.columns:
                self.test_df = self.test_df.sort_values(['user_id', 'date'])
            else:
                self.test_df = self.test_df.sort_values('user_id')

            for user_id in self.test_df['user_id'].unique():
                # Get user history from training data
                user_train_data = self.train_df[self.train_df['user_id'] == user_id]

                if len(user_train_data) > 0:
                    # Extract sequences from training data
                    prev_apps = user_train_data['app_id'].tolist()
                    prev_ratings = [1 if r else 0 for r in user_train_data['is_recommended'].tolist()]
                    prev_hours = user_train_data['hours'].tolist()
                else:
                    prev_apps = []
                    prev_ratings = []
                    prev_hours = []

                # Add sequence data to each test interaction
                user_test_data = self.test_df[self.test_df['user_id'] == user_id]

                for idx, row in user_test_data.iterrows():
                    self.test_df.at[idx, 'prev_apps'] = prev_apps[-self.config['max_seq_length']:] if prev_apps else []
                    self.test_df.at[idx, 'prev_ratings'] = prev_ratings[
                                                           -self.config['max_seq_length']:] if prev_ratings else []
                    self.test_df.at[idx, 'prev_hours'] = prev_hours[
                                                         -self.config['max_seq_length']:] if prev_hours else []

                    # Update sequences
                    prev_apps.append(row['app_id'])
                    prev_ratings.append(1 if row['is_recommended'] else 0)
                    prev_hours.append(row['hours'])

            # Calculate sequence statistics features
            def calc_sequence_stats(row):
                prev_ratings = row['prev_ratings'] if isinstance(row['prev_ratings'], list) else []
                prev_hours = row['prev_hours'] if isinstance(row['prev_hours'], list) else []

                # Calculate decayed average rating and hours
                rating_sum, rating_weight_sum = 0, 0
                hours_sum, hours_weight_sum = 0, 0

                for i, (rating, hours) in enumerate(zip(prev_ratings, prev_hours)):
                    # Apply time decay - more recent interactions have higher weight
                    weight = self.config['time_decay_factor'] ** (len(prev_ratings) - i - 1)
                    rating_sum += rating * weight
                    rating_weight_sum += weight
                    hours_sum += hours * weight
                    hours_weight_sum += weight

                avg_prev_rating = rating_sum / rating_weight_sum if rating_weight_sum > 0 else 0
                avg_prev_hours = hours_sum / hours_weight_sum if hours_weight_sum > 0 else 0

                return pd.Series({
                    'prev_game_count': len(row['prev_apps']) if isinstance(row['prev_apps'], list) else 0,
                    'avg_prev_rating': avg_prev_rating,
                    'avg_prev_hours': avg_prev_hours
                })

            # Add sequence statistics to dataframes
            seq_stats_train = self.train_df.apply(calc_sequence_stats, axis=1)
            self.train_df = pd.concat([self.train_df, seq_stats_train], axis=1)

            seq_stats_test = self.test_df.apply(calc_sequence_stats, axis=1)
            self.test_df = pd.concat([self.test_df, seq_stats_test], axis=1)

            # Store sequence feature columns for model training
            self.sequence_feature_columns = [
                'prev_game_count', 'avg_prev_rating', 'avg_prev_hours',
                'user_game_count', 'user_recommendation_ratio', 'user_total_hours'
            ]

            logger.info("Sequence features created successfully")

        except Exception as e:
            logger.error(f"Error creating sequence features: {str(e)}")
            logger.error(traceback.format_exc())

    def split_train_test(self, test_size=0.2, random_state=42, by_time=True):
        """Split data into training and test sets"""
        from sklearn.model_selection import train_test_split

        # 使用 self.df 而不是参数传入的 df
        df = self.df

        if by_time and 'date' in df.columns:
            # 按时间划分
            df = df.sort_values('date')
            split_idx = int(len(df) * (1 - test_size))
            train_df = df.iloc[:split_idx].copy()
            test_df = df.iloc[split_idx:].copy()
            logger.info(
                f"按时间划分数据，训练集截止日期: {train_df['date'].max()}, 测试集起始日期: {test_df['date'].min()}")
        else:
            # 随机划分
            stratify_col = df['is_recommended'] if 'is_recommended' in df.columns else None
            train_df, test_df = train_test_split(
                df, test_size=test_size, random_state=random_state, stratify=stratify_col
            )
            logger.info(f"随机划分数据，训练集: {len(train_df)}条, 测试集: {len(test_df)}条")

        return train_df, test_df

    def get_processed_data(self):
        """Return processed data"""
        return {
            'full_df': self.df,
            'train_df': self.train_df,
            'test_df': self.test_df
        }

    def split_data_into_train_test(self, test_size=0.2, random_state=42):
        """Split data into training and test sets"""
        logger.info("Splitting data into training and test sets")

        try:
            if not hasattr(self, 'df') or self.df is None or len(self.df) == 0:
                logger.error("No data available for splitting")
                return False

            # 使用已有的 split_train_test 方法
            self.train_df, self.test_df = self.split_train_test(
                test_size=test_size, random_state=random_state
            )

            # 添加这些关键日志
            logger.info(
                f"Data split completed: Training set: {len(self.train_df)} samples, Test set: {len(self.test_df)} samples")

            # 检查测试集中的唯一用户数量
            if self.test_df is not None:
                unique_test_users = self.test_df['user_id'].nunique()
                logger.info(f"Test set contains {unique_test_users} unique users")

                # 查看测试集中的必要列
                test_columns = self.test_df.columns.tolist()
                logger.info(f"Test set columns: {test_columns}")

                # 确认 is_recommended 列存在且有推荐的物品
                if 'is_recommended' in self.test_df.columns:
                    recommended_count = self.test_df['is_recommended'].sum()
                    logger.info(f"Test set contains {recommended_count} recommended items")
            else:
                logger.error("Test DataFrame is None after splitting")

            return True
        except Exception as e:
            logger.error(f"Error splitting data: {str(e)}")
            logger.error(traceback.format_exc())
            return False

    def extract_user_preferences(self):
        """Extract and analyze user preferences"""
        logger.info("Extracting user preferences...")

        user_preferences = {}
        item_tags = {}

        # 处理物品标签
        if hasattr(self, 'train_df') and 'tags' in self.train_df.columns:
            for _, row in self.train_df.drop_duplicates('app_id').iterrows():
                app_id = row['app_id']
                if pd.notna(row['tags']) and row['tags']:
                    item_tags[app_id] = [tag.strip() for tag in row['tags'].split(',')]

        # 处理用户偏好
        for user_id in self.train_df['user_id'].unique():
            user_data = self.train_df[self.train_df['user_id'] == user_id]

            # 喜欢的游戏
            liked_games = user_data[user_data['is_recommended'] == True]['app_id'].tolist()

            # 标签偏好
            tag_preferences = {}
            for _, row in user_data.iterrows():
                if pd.notna(row['tags']) and row['tags']:
                    tags = [tag.strip() for tag in row['tags'].split(',')]
                    for tag in tags:
                        if tag not in tag_preferences:
                            tag_preferences[tag] = 0

                        # 计算标签权重
                        weight = 2 if row['is_recommended'] else 0.5

                        # 考虑游戏时长
                        if pd.notna(row['hours']):
                            playtime_factor = min(row['hours'] / 10, 3)  # 最多3倍权重
                            weight *= playtime_factor

                        tag_preferences[tag] += weight

            # 储存用户偏好
            user_preferences[user_id] = {
                'liked_games': liked_games,
                'tag_preferences': tag_preferences
            }

        # 保存处理结果
        self.user_preferences = user_preferences
        self.item_tags = item_tags

        logger.info(f"Extracted preferences for {len(user_preferences)} users and tags for {len(item_tags)} items")
        return user_preferences