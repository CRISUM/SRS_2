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

    def load_data(self, data_path, chunk_size=500000):
        """Load data in chunks"""
        logger.info(f"Loading data in chunks: {data_path}")

        # First read header to get column names
        try:
            header_df = pd.read_csv(data_path, nrows=1)
            column_names = header_df.columns.tolist()
            logger.info(f"Data columns: {column_names}")

            # Check if required columns exist
            required_cols = ['user_id', 'app_id', 'is_recommended', 'hours', 'title', 'tags']
            missing_cols = [col for col in required_cols if col not in column_names]
            if missing_cols:
                logger.error(f"CSV missing required columns: {missing_cols}")

                # Look for possible matching columns (case-insensitive)
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

        # Initialize statistics tracking
        unique_users = set()
        unique_games = set()
        user_stats = {}
        game_stats = {}
        chunk_count = 0

        try:
            for chunk in pd.read_csv(self.data_path, chunksize=chunk_size):
                chunk_count += 1
                logger.info(f"Processing data chunk {chunk_count}")

                # Ensure necessary columns exist
                for col in required_cols:
                    if col not in chunk.columns:
                        logger.warning(f"Chunk {chunk_count} missing column '{col}', skipping")
                        continue

                # Update statistics
                unique_users.update(chunk['user_id'].unique())
                unique_games.update(chunk['app_id'].unique())

                # Process user statistics
                for _, row in chunk.iterrows():
                    user_id = row['user_id']
                    app_id = row['app_id']

                    # Handle possible missing values
                    try:
                        is_recommended = row['is_recommended']
                        hours = float(row['hours']) if not pd.isna(row['hours']) else 0.0
                    except (KeyError, ValueError):
                        is_recommended = False
                        hours = 0.0

                    # Update user stats
                    if user_id not in user_stats:
                        user_stats[user_id] = {'game_count': 0, 'total_hours': 0, 'recommended_count': 0}
                    user_stats[user_id]['game_count'] += 1
                    user_stats[user_id]['total_hours'] += hours
                    if is_recommended:
                        user_stats[user_id]['recommended_count'] += 1

                    # Update game stats
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

                # Free memory
                del chunk
                gc.collect()

            logger.info(f"Processed {chunk_count} data chunks")

            # Convert statistics to DataFrames
            self.user_df = pd.DataFrame.from_dict(user_stats, orient='index')
            self.user_df.reset_index(inplace=True)
            self.user_df.rename(columns={'index': 'user_id'}, inplace=True)

            self.game_df = pd.DataFrame.from_dict(game_stats, orient='index')
            self.game_df.reset_index(inplace=True)
            self.game_df.rename(columns={'index': 'app_id'}, inplace=True)

            # Calculate additional statistics
            self.user_df['recommendation_ratio'] = self.user_df['recommended_count'] / self.user_df['game_count']
            self.game_df['recommendation_ratio'] = self.game_df['recommended_count'] / self.game_df['user_count']
            self.game_df['avg_hours'] = self.game_df['total_hours'] / self.game_df['user_count']

            # Sample data for training
            self._create_training_sample()

            logger.info(
                f"Data loading complete, found {len(unique_users)} unique users and {len(unique_games)} games")
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

            # 6. Create sequence features
            self.create_sequence_features()

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
        df = self.train_df

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
                df, test_size=test_size, random_state=42, stratify=stratify_col
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