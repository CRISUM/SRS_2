#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Py_GetModel_src/models/sequence_model.py - Sequential recommendation model
Author: YourName
Date: 2025-04-29
Description: Implements deep learning based game-to-game sequential recommendation model
             Optimized for sparse user data by focusing on game associations
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import logging
import os
import pickle
import traceback
from collections import defaultdict

from .base_model import BaseRecommenderModel

logger = logging.getLogger(__name__)


class GameSequenceModel(nn.Module):
    """Neural network model for sequential game recommendations"""

    def __init__(self, num_features, hidden_dim=128, num_layers=2, dropout=0.2):
        """Initialize model architecture"""
        super().__init__()
        self.input_size = num_features
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout

        # Define model layers with批归一化
        self.fc1 = nn.Linear(num_features, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)  # 添加批归一化
        self.relu = nn.ReLU()
        self.dropout_layer = nn.Dropout(dropout)

        # Create additional hidden layers if num_layers > 1
        self.hidden_layers = nn.ModuleList()
        self.bn_layers = nn.ModuleList()  # 批归一化层
        for _ in range(num_layers - 1):
            self.hidden_layers.append(nn.Linear(hidden_dim, hidden_dim))
            self.bn_layers.append(nn.BatchNorm1d(hidden_dim))

        # Output layer
        self.output = nn.Linear(hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """Forward pass

        Args:
            x: Input tensor of shape (batch_size, input_size)

        Returns:
            tensor: Predicted scores (0-1)
        """
        # First layer with批归一化
        x = self.fc1(x)
        # 处理批量大小为1的情况
        if x.size(0) > 1:
            x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout_layer(x)

        # Additional hidden layers
        for i, layer in enumerate(self.hidden_layers):
            x = layer(x)
            # 处理批量大小为1的情况
            if x.size(0) > 1:
                x = self.bn_layers[i](x)
            x = self.relu(x)
            x = self.dropout_layer(x)

        # Output layer
        x = self.output(x)
        x = self.sigmoid(x)

        return x.squeeze()

class SequenceRecommender(BaseRecommenderModel):
    """Recommendation model using sequential game-to-game relationships
       Addresses data sparsity by focusing on game associations rather than user sequences
    """

    def __init__(self, hidden_dim=128, num_layers=2, dropout=0.2,
                 learning_rate=0.001, batch_size=64, epochs=10, device=None):
        """Initialize game sequence recommender"""
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.feature_columns = None
        self.training_history = {'loss': []}

        # Game data
        self.game_data = {}
        self.game_indices = {}

        # Game sequences
        self.game_sequences = defaultdict(list)  # Game -> List of next games
        self.game_transitions = {}  # (game1, game2) -> count

        # User histories for context
        self.user_history = {}

        # Popular games for fallback
        self.popular_games = []

        # Default features
        self.default_features = None

    def _extract_game_sequences(self, df):
        """Extract game-to-game transition sequences from user play histories

        Args:
            df (DataFrame): User-item interactions with user_id, app_id columns

        Returns:
            dict: Game transition counts
        """
        logger.info("Extracting game-to-game transitions from user histories...")

        transitions = defaultdict(int)
        sequences = defaultdict(list)
        user_histories = {}

        # Process each user's history to extract game transitions
        for user_id in df['user_id'].unique():
            user_data = df[df['user_id'] == user_id].copy()

            # Sort by date if available, otherwise use default order
            if 'date' in user_data.columns:
                user_data = user_data.sort_values('date')

            # Get sequence of games
            user_games = user_data['app_id'].tolist()
            user_histories[user_id] = user_games

            # Skip if user has only one game
            if len(user_games) <= 1:
                continue

            # Extract transitions between consecutive games
            for i in range(len(user_games) - 1):
                game1 = user_games[i]
                game2 = user_games[i + 1]

                transitions[(game1, game2)] += 1
                sequences[game1].append(game2)

        return transitions, sequences, user_histories

    def _prepare_training_data(self, df, transitions):
        """Prepare training data for the neural network

        Args:
            df (DataFrame): User-item interactions
            transitions (dict): Game transition counts

        Returns:
            tuple: (feature_matrix, labels)
        """
        logger.info("Preparing training data from game transitions...")

        # Collect features for training
        game_features = {}
        game_tags = {}

        # Extract game metadata
        for _, row in df.drop_duplicates('app_id').iterrows():
            game_id = row['app_id']

            # Extract tags if available
            if 'tags' in row and pd.notna(row['tags']):
                tags = [t.strip() for t in str(row['tags']).split(',')]
                game_tags[game_id] = tags
            else:
                game_tags[game_id] = []

            # Store basic game features
            game_features[game_id] = {}

            # Get hours if available
            if 'hours' in df.columns:
                avg_hours = df[df['app_id'] == game_id]['hours'].mean()
                game_features[game_id]['avg_hours'] = avg_hours if not pd.isna(avg_hours) else 0

            # Get recommendation ratio if available
            if 'is_recommended' in df.columns:
                rec_ratio = df[df['app_id'] == game_id]['is_recommended'].mean()
                game_features[game_id]['rec_ratio'] = rec_ratio if not pd.isna(rec_ratio) else 0.5

        # Convert tags to one-hot features
        if game_tags:
            # Find most common tags
            tag_counter = defaultdict(int)
            for tags in game_tags.values():
                for tag in tags:
                    tag_counter[tag] += 1

            # Take top 50 tags
            top_tags = sorted(tag_counter.items(), key=lambda x: x[1], reverse=True)[:50]
            top_tag_set = {tag for tag, _ in top_tags}

            # Create one-hot encoding
            for game_id, tags in game_tags.items():
                for tag in top_tag_set:
                    game_features[game_id][f'tag_{tag}'] = 1 if tag in tags else 0

        # Collect all feature keys
        all_feature_keys = set()
        for features in game_features.values():
            all_feature_keys.update(features.keys())

        # Convert to sorted list for consistent ordering
        feature_columns = sorted(all_feature_keys)
        self.feature_columns = feature_columns

        # Prepare transition samples
        X_samples = []
        y_samples = []

        # Process all possible game pairs
        game_ids = list(game_features.keys())
        for i, game1 in enumerate(game_ids):
            for game2 in game_ids:
                if game1 == game2:
                    continue

                # Create feature vector
                feature_vector = []
                for feature in feature_columns:
                    # Get feature for game1
                    game1_value = game_features[game1].get(feature, 0)

                    # Get feature for game2
                    game2_value = game_features[game2].get(feature, 0)

                    # Add both values
                    feature_vector.append(game1_value)
                    feature_vector.append(game2_value)

                    # Add interaction term (product)
                    feature_vector.append(game1_value * game2_value)

                # Create label from transition count
                count = transitions.get((game1, game2), 0)
                label = 1.0 if count > 0 else 0.0

                X_samples.append(feature_vector)
                y_samples.append(label)

        # Convert to numpy arrays
        X = np.array(X_samples)
        y = np.array(y_samples)

        logger.info(f"Prepared {len(X)} training samples with {X.shape[1]} features")
        return X, y

    def fit(self, data):
        """Train the model with provided data

        Args:
            data (DataFrame or dict): Training data

        Returns:
            self: Trained model
        """
        logger.info("Training simplified game sequence model...")

        try:
            # Handle different input formats
            if isinstance(data, dict) and 'df' in data:
                df = data['df']
            elif isinstance(data, pd.DataFrame):
                df = data
            else:
                logger.error("Invalid input data format")
                return self

            # Extract game transitions from user histories
            transitions, sequences, user_histories = self._extract_game_sequences(df)
            self.game_transitions = transitions
            self.game_sequences = sequences
            self.user_history = user_histories

            # Get popular games for fallback
            game_counts = df['app_id'].value_counts()
            self.popular_games = [(game_id, count / len(df['user_id'].unique()))
                                  for game_id, count in game_counts.items()]
            self.popular_games.sort(key=lambda x: x[1], reverse=True)

            # 简化: 不再使用神经网络，而是基于游戏转移概率
            # 计算游戏转移概率矩阵
            self.transition_probs = defaultdict(dict)

            for (game1, game2), count in transitions.items():
                # 获取从game1出发的所有转移总数
                total_transitions = sum(count for (g1, _), count in transitions.items() if g1 == game1)

                # 计算条件概率 P(game2|game1)
                if total_transitions > 0:
                    self.transition_probs[game1][game2] = count / total_transitions

            # 保存游戏数据
            for game_id in df['app_id'].unique():
                self.game_data[game_id] = {'id': game_id}
                self.game_indices[game_id] = len(self.game_indices)

            logger.info(f"Simplified game sequence model trained with {len(transitions)} game transitions")
            return self

        except Exception as e:
            logger.error(f"Error training simplified game sequence model: {str(e)}")
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
        try:
            # Check if we have user history
            if user_id not in self.user_history or not self.user_history[user_id]:
                # Fall back to popularity
                for game_id, popularity in self.popular_games:
                    if game_id == item_id:
                        return min(0.9, popularity)
                return 0.5

            # Get user's most recent games (up to 3)
            user_games = self.user_history[user_id][-3:]

            # Check for direct transitions
            transition_scores = []
            for game in user_games:
                count = self.game_transitions.get((game, item_id), 0)
                if count > 0:
                    # Scale transition count
                    transition_scores.append(min(0.9, 0.5 + 0.1 * count))

            # If we have transition scores, use the maximum
            if transition_scores:
                return max(transition_scores)

            # If we have a neural model, try using it
            if self.model is not None and hasattr(self, 'feature_columns') and self.feature_columns:
                # Try the neural model for the most recent game
                recent_game = user_games[-1]

                try:
                    # Extract features
                    features = self._extract_game_pair_features(recent_game, item_id)

                    # Convert to tensor
                    features_tensor = torch.FloatTensor(features).unsqueeze(0).to(self.device)

                    # Make prediction
                    with torch.no_grad():
                        prediction = self.model(features_tensor).item()

                    return prediction
                except Exception as e:
                    logger.debug(f"Neural prediction error: {str(e)}")

            # Fall back to rule-based score
            for game_id, sequence in self.game_sequences.items():
                if item_id in sequence and game_id in user_games:
                    return 0.7  # There is some association

            # No associations found, use popularity as fallback
            for game_id, popularity in self.popular_games:
                if game_id == item_id:
                    return min(0.7, 0.3 + 0.4 * popularity)

            return 0.5  # Default score

        except Exception as e:
            logger.error(f"Error in game sequence prediction: {str(e)}")
            logger.error(traceback.format_exc())
            return 0.5

    def recommend(self, user_id, n=10):
        """Generate top-N recommendations for a user

        Args:
            user_id: User ID
            n (int): Number of recommendations

        Returns:
            list: List of (item_id, score) tuples
        """
        try:
            # Check if user has history
            if user_id not in self.user_history or not self.user_history[user_id]:
                return self.popular_games[:n]

            # Get user's recent games (last 3)
            recent_games = self.user_history[user_id][-3:]

            # Collect candidate recommendations based on transition probabilities
            candidates = {}

            for game in recent_games:
                if game in self.transition_probs:
                    # Add games that frequently follow this game
                    for next_game, prob in self.transition_probs[game].items():
                        # Skip games user already has
                        if next_game in self.user_history[user_id]:
                            continue

                        # Update with highest probability
                        candidates[next_game] = max(
                            candidates.get(next_game, 0),
                            min(0.95, prob)  # Cap at 0.95
                        )

            # If not enough candidates, add popular games
            if len(candidates) < n:
                for game_id, pop in self.popular_games:
                    if game_id not in self.user_history[user_id] and game_id not in candidates:
                        # Use scaled popularity as score
                        candidates[game_id] = 0.3 + (0.4 * pop)

                        if len(candidates) >= n * 2:
                            break

            # Sort candidates by score and add diversity
            sorted_candidates = sorted(candidates.items(), key=lambda x: x[1], reverse=True)

            # Add diversity by avoiding too similar games
            selected = []
            selected_ids = set()

            # First select top recommendations
            top_k = max(1, n // 3)
            for i in range(min(top_k, len(sorted_candidates))):
                game_id, score = sorted_candidates[i]
                selected.append((game_id, score))
                selected_ids.add(game_id)

            # Then add diverse recommendations
            for game_id, score in sorted_candidates[top_k:]:
                if len(selected) >= n:
                    break

                # Check if too similar to already selected games
                too_similar = False
                for sel_id, _ in selected:
                    # If direct transition exists in either direction, might be too similar
                    if (sel_id in self.transition_probs and game_id in self.transition_probs[sel_id]) or \
                            (game_id in self.transition_probs and sel_id in self.transition_probs[game_id]):
                        too_similar = True
                        break

                # Add if not too similar or if we need more recommendations
                if not too_similar or len(selected) < n * 2 / 3:
                    selected.append((game_id, score))
                    selected_ids.add(game_id)

            # Fill remaining slots with other candidates if needed
            if len(selected) < n:
                for game_id, score in sorted_candidates:
                    if game_id not in selected_ids:
                        selected.append((game_id, score))
                        selected_ids.add(game_id)

                        if len(selected) >= n:
                            break

            return selected[:n]

        except Exception as e:
            logger.error(f"Error generating sequence recommendations: {str(e)}")
            logger.error(traceback.format_exc())
            return self.popular_games[:n]
        
    def _extract_game_pair_features(self, game1, game2):
        """Extract features for a pair of games

        Args:
            game1: First game ID
            game2: Second game ID

        Returns:
            array: Feature vector
        """
        # If no feature columns defined, return default features
        if not self.feature_columns or self.default_features is None:
            logger.warning("No feature columns defined, using defaults")
            return self.default_features if self.default_features is not None else np.zeros(self.model.input_size)

        # Get game data
        game1_data = self.game_data.get(game1, {})
        game2_data = self.game_data.get(game2, {})

        # Create feature vector
        feature_vector = []
        for feature in self.feature_columns:
            # Get feature for game1
            game1_value = game1_data.get(feature, 0)

            # Get feature for game2
            game2_value = game2_data.get(feature, 0)

            # Add both values
            feature_vector.append(game1_value)
            feature_vector.append(game2_value)

            # Add interaction term (product)
            feature_vector.append(game1_value * game2_value)

        return np.array(feature_vector)

    def update(self, new_data):
        """Update model with new data (incremental learning)

        Args:
            new_data (DataFrame or dict): New data for updating

        Returns:
            self: Updated model
        """
        logger.info("Updating game sequence model...")

        try:
            # Handle different input formats
            if isinstance(new_data, dict) and 'df' in new_data:
                df = new_data['df']
            elif isinstance(new_data, pd.DataFrame):
                df = new_data
            else:
                logger.error("Invalid input data format for update")
                return self

            # Extract new transitions
            new_transitions, new_sequences, new_user_histories = self._extract_game_sequences(df)

            # Update user histories
            for user_id, history in new_user_histories.items():
                if user_id in self.user_history:
                    # Append new history (maintaining order)
                    self.user_history[user_id].extend(history)
                else:
                    # Create new history
                    self.user_history[user_id] = history

            # Update game transitions
            for (game1, game2), count in new_transitions.items():
                self.game_transitions[(game1, game2)] = self.game_transitions.get((game1, game2), 0) + count

            # Update game sequences
            for game, next_games in new_sequences.items():
                self.game_sequences[game].extend(next_games)

            # Update game data and indices
            for game_id in df['app_id'].unique():
                if game_id not in self.game_data:
                    self.game_data[game_id] = {'id': game_id}
                    self.game_indices[game_id] = len(self.game_indices)

            # Update popular games
            game_counts = df['app_id'].value_counts()
            new_popular = [(game_id, count / len(df['user_id'].unique()))
                           for game_id, count in game_counts.items()]

            # Merge with existing popular games (weighted average)
            popular_dict = dict(self.popular_games)
            for game_id, score in new_popular:
                if game_id in popular_dict:
                    # Update with 70% weight to new data
                    popular_dict[game_id] = 0.3 * popular_dict[game_id] + 0.7 * score
                else:
                    popular_dict[game_id] = score

            # Convert back to sorted list
            self.popular_games = sorted(popular_dict.items(), key=lambda x: x[1], reverse=True)

            # Retrain neural model if we have a significant amount of new data
            if len(new_transitions) >= 20 and self.model is not None:
                # Prepare training data from all transitions
                X, y = self._prepare_training_data(df, self.game_transitions)

                # Fine-tune model
                X_tensor = torch.FloatTensor(X).to(self.device)
                y_tensor = torch.FloatTensor(y).to(self.device)

                # Setup optimizer with lower learning rate
                optimizer = optim.Adam(
                    self.model.parameters(),
                    lr=self.learning_rate * 0.5  # Reduced learning rate for fine-tuning
                )
                criterion = nn.BCELoss()

                # Training loop (fewer epochs)
                update_epochs = max(2, self.epochs // 3)

                self.model.train()
                for epoch in range(update_epochs):
                    # Mini-batch training
                    total_loss = 0
                    num_batches = 0

                    # Create random permutation for shuffling
                    indices = torch.randperm(len(X_tensor))

                    for i in range(0, len(X_tensor), self.batch_size):
                        # Get batch indices
                        batch_indices = indices[i:i + self.batch_size]

                        # Get batch
                        X_batch = X_tensor[batch_indices]
                        y_batch = y_tensor[batch_indices]

                        # Forward pass
                        optimizer.zero_grad()
                        outputs = self.model(X_batch)

                        # Calculate loss
                        loss = criterion(outputs, y_batch)
                        total_loss += loss.item()

                        # Backward pass
                        loss.backward()
                        optimizer.step()

                        num_batches += 1

                    # Record loss
                    epoch_loss = total_loss / num_batches if num_batches > 0 else 0
                    self.training_history['loss'].append(epoch_loss)
                    logger.info(f"Update epoch {epoch + 1}/{update_epochs}, Loss: {epoch_loss:.4f}")

                # Update default features
                self.default_features = np.mean(X, axis=0)

                # Set model to evaluation mode
                self.model.eval()

            logger.info("Game sequence model updated successfully")
            return self

        except Exception as e:
            logger.error(f"Error updating game sequence model: {str(e)}")
            logger.error(traceback.format_exc())
            return self

    def save(self, path):
        """Save model to disk

        Args:
            path (str): Directory path

        Returns:
            bool: Success
        """
        logger.info(f"Saving game sequence model to {path}")

        try:
            os.makedirs(path, exist_ok=True)

            # Save model state dict if we have a neural model
            if self.model is not None:
                torch.save(self.model.state_dict(), os.path.join(path, 'sequence_model.pt'))

            # Save game data
            with open(os.path.join(path, 'game_transitions.pkl'), 'wb') as f:
                pickle.dump(self.game_transitions, f)

            with open(os.path.join(path, 'game_sequences.pkl'), 'wb') as f:
                pickle.dump(dict(self.game_sequences), f)

            with open(os.path.join(path, 'game_data.pkl'), 'wb') as f:
                pickle.dump(self.game_data, f)

            with open(os.path.join(path, 'game_indices.pkl'), 'wb') as f:
                pickle.dump(self.game_indices, f)

            with open(os.path.join(path, 'user_history.pkl'), 'wb') as f:
                pickle.dump(self.user_history, f)

            with open(os.path.join(path, 'popular_games.pkl'), 'wb') as f:
                pickle.dump(self.popular_games, f)

            # Save model metadata
            metadata = {
                'hidden_dim': self.hidden_dim,
                'num_layers': self.num_layers,
                'dropout': self.dropout,
                'learning_rate': self.learning_rate,
                'batch_size': self.batch_size,
                'epochs': self.epochs,
                'feature_columns': self.feature_columns,
                'training_history': self.training_history,
                'default_features': self.default_features
            }

            with open(os.path.join(path, 'sequence_metadata.pkl'), 'wb') as f:
                pickle.dump(metadata, f)

            logger.info("Game sequence model saved successfully")
            return True

        except Exception as e:
            logger.error(f"Error saving game sequence model: {str(e)}")
            logger.error(traceback.format_exc())
            return False

    def load(self, path):
        """Load model from disk

        Args:
            path (str): Directory path

        Returns:
            self: Loaded model
        """
        logger.info(f"Loading game sequence model from {path}")

        try:
            # Load metadata
            with open(os.path.join(path, 'sequence_metadata.pkl'), 'rb') as f:
                metadata = pickle.load(f)

            self.hidden_dim = metadata['hidden_dim']
            self.num_layers = metadata['num_layers']
            self.dropout = metadata['dropout']
            self.learning_rate = metadata['learning_rate']
            self.batch_size = metadata['batch_size']
            self.epochs = metadata['epochs']
            self.feature_columns = metadata['feature_columns']
            self.training_history = metadata['training_history']
            self.default_features = metadata['default_features']

            # Load game transitions and sequences
            with open(os.path.join(path, 'game_transitions.pkl'), 'rb') as f:
                self.game_transitions = pickle.load(f)

            with open(os.path.join(path, 'game_sequences.pkl'), 'rb') as f:
                sequences_dict = pickle.load(f)
                self.game_sequences = defaultdict(list)
                for game, seq in sequences_dict.items():
                    self.game_sequences[game] = seq

            with open(os.path.join(path, 'game_data.pkl'), 'rb') as f:
                self.game_data = pickle.load(f)

            with open(os.path.join(path, 'game_indices.pkl'), 'rb') as f:
                self.game_indices = pickle.load(f)

            with open(os.path.join(path, 'user_history.pkl'), 'rb') as f:
                self.user_history = pickle.load(f)

            with open(os.path.join(path, 'popular_games.pkl'), 'rb') as f:
                self.popular_games = pickle.load(f)

            # Initialize and load neural model if exists
            model_path = os.path.join(path, 'sequence_model.pt')
            if os.path.exists(model_path) and self.default_features is not None:
                # Initialize model
                input_size = len(self.default_features)
                self.model = GameSequenceModel(
                    num_features=input_size,
                    hidden_dim=self.hidden_dim,
                    num_layers=self.num_layers,
                    dropout=self.dropout
                ).to(self.device)

                # Load state dict
                state_dict = torch.load(model_path, map_location=self.device)
                self.model.load_state_dict(state_dict)
                self.model.eval()
            else:
                logger.info("No neural model found, using rule-based sequence model only")
                self.model = None

            logger.info("Game sequence model loaded successfully")
            return self

        except Exception as e:
            logger.error(f"Error loading game sequence model: {str(e)}")
            logger.error(traceback.format_exc())
            return None