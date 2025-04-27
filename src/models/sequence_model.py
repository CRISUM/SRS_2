#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
src/models/sequence_model.py - Sequential recommendation model
Author: YourName
Date: 2025-04-27
Description: Implements deep learning based sequential recommendation model
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import logging
import os
import pickle

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

        # Define model layers
        self.fc1 = nn.Linear(num_features, hidden_dim)
        self.relu = nn.ReLU()
        self.dropout_layer = nn.Dropout(dropout)

        # Create additional hidden layers if num_layers > 1
        self.hidden_layers = nn.ModuleList()
        for _ in range(num_layers - 1):
            self.hidden_layers.append(nn.Linear(hidden_dim, hidden_dim))

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
        # First layer
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout_layer(x)

        # Additional hidden layers
        for layer in self.hidden_layers:
            x = layer(x)
            x = self.relu(x)
            x = self.dropout_layer(x)

        # Output layer
        x = self.output(x)
        x = self.sigmoid(x)

        return x.squeeze()


class SequenceRecommender(BaseRecommenderModel):
    """Recommendation model using sequential user behavior"""

    def __init__(self, hidden_dim=128, num_layers=2, dropout=0.2,
                 learning_rate=0.001, batch_size=64, epochs=10, device=None):
        """Initialize sequence recommender"""
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
        self.user_data = {}
        self.item_data = {}
        self.item_indices = {}
        self.default_features = None

    def fit(self, data):
        """Train the model with provided data

        Args:
            data (dict): Dictionary containing training data with keys:
                         'X': feature matrix
                         'y': target values
                         'feature_columns': list of feature column names
                         'user_data': optional user data dictionary
                         'item_data': optional item data dictionary

        Returns:
            self: Trained model
        """
        logger.info("Training sequence model...")

        if not isinstance(data, dict) or 'X' not in data or 'y' not in data:
            raise ValueError("Data must be a dictionary with 'X' and 'y' keys")

        X = data['X']
        y = data['y']

        # Store feature columns
        if 'feature_columns' in data:
            self.feature_columns = data['feature_columns']

        # Store user and item data if provided
        if 'user_data' in data:
            self.user_data = data['user_data']

        if 'item_data' in data:
            self.item_data = data['item_data']
            self.item_indices = {item_id: i for i, item_id in enumerate(self.item_data)}

        # Initialize model
        input_size = X.shape[1]
        self.model = GameSequenceModel(
            num_features=input_size,
            hidden_dim=self.hidden_dim,
            num_layers=self.num_layers,
            dropout=self.dropout
        ).to(self.device)

        # Setup optimizer and loss function
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        criterion = nn.BCELoss()

        # Convert to tensors
        X_tensor = torch.FloatTensor(X).to(self.device)
        y_tensor = torch.FloatTensor(y).to(self.device)

        # Training loop
        self.model.train()
        for epoch in range(self.epochs):
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

            # Calculate average loss for epoch
            epoch_loss = total_loss / num_batches
            self.training_history['loss'].append(epoch_loss)
            logger.info(f"Epoch {epoch + 1}/{self.epochs}, Loss: {epoch_loss:.4f}")

        # Save default feature vector for cold start
        self.default_features = np.mean(X, axis=0)

        # Set model to evaluation mode
        self.model.eval()
        logger.info("Sequence model training completed")
        return self

    def predict(self, user_id, item_id):
        """Predict rating for a user-item pair

        Args:
            user_id: User ID
            item_id: Item ID

        Returns:
            float: Predicted rating (0-1 scale)
        """
        if self.model is None:
            logger.warning("Model not trained yet")
            return 0.5

        try:
            # Extract features for prediction
            features = self.extract_features(user_id, item_id)

            # Convert to tensor
            features_tensor = torch.FloatTensor(features).unsqueeze(0).to(self.device)

            # Make prediction
            self.model.eval()
            with torch.no_grad():
                prediction = self.model(features_tensor).item()

            return prediction
        except Exception as e:
            logger.error(f"Error in sequence model prediction: {str(e)}")
            return 0.5

    def recommend(self, user_id, n=10):
        """Generate top-N recommendations for a user

        Args:
            user_id: User ID
            n (int): Number of recommendations

        Returns:
            list: List of (item_id, score) tuples
        """
        if self.model is None:
            logger.warning("Model not trained yet")
            return []

        try:
            # Get all possible items
            if self.item_data:
                candidate_items = list(self.item_data.keys())
            elif self.item_indices:
                candidate_items = list(self.item_indices.keys())
            else:
                logger.warning("No item data available for recommendations")
                return []

            # Get user's existing items to filter them out
            user_items = set()
            if user_id in self.user_data and 'items' in self.user_data[user_id]:
                user_items = set(self.user_data[user_id]['items'])

            # Calculate scores for all candidate items
            item_scores = []
            for item_id in candidate_items:
                # Skip already seen items
                if item_id in user_items:
                    continue

                # Predict score
                score = self.predict(user_id, item_id)
                item_scores.append((item_id, score))

            # Sort by score in descending order
            item_scores.sort(key=lambda x: x[1], reverse=True)

            # Return top N
            return item_scores[:n]
        except Exception as e:
            logger.error(f"Error in sequence model recommendation: {str(e)}")
            return []

    def update(self, new_data):
        """Update model with new data (incremental learning)

        Args:
            new_data (dict): Dictionary containing new training data with keys:
                            'X': feature matrix
                            'y': target values
                            'feature_columns': optional list of feature column names
                            'user_data': optional user data dictionary update
                            'item_data': optional item data dictionary update

        Returns:
            self: Updated model
        """
        logger.info("Updating sequence model...")

        if not isinstance(new_data, dict) or 'X' not in new_data or 'y' not in new_data:
            raise ValueError("New data must be a dictionary with 'X' and 'y' keys")

        X = new_data['X']
        y = new_data['y']

        # Update feature columns if provided
        if 'feature_columns' in new_data:
            self.feature_columns = new_data['feature_columns']

        # Update user and item data if provided
        if 'user_data' in new_data:
            self.user_data.update(new_data['user_data'])

        if 'item_data' in new_data:
            self.item_data.update(new_data['item_data'])
            self.item_indices = {item_id: i for i, item_id in enumerate(self.item_data)}

        # Check if model exists
        if self.model is None:
            # Train from scratch
            return self.fit(new_data)

        # Setup optimizer and loss function
        optimizer = optim.Adam(self.model.parameters(),
                               lr=self.learning_rate * 0.5)  # Lower learning rate for fine-tuning
        criterion = nn.BCELoss()

        # Convert to tensors
        X_tensor = torch.FloatTensor(X).to(self.device)
        y_tensor = torch.FloatTensor(y).to(self.device)

        # Fine-tuning loop (fewer epochs)
        update_epochs = max(2, self.epochs // 3)  # Fewer epochs for update

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

            # Calculate average loss for epoch
            epoch_loss = total_loss / num_batches
            self.training_history['loss'].append(epoch_loss)
            logger.info(f"Update epoch {epoch + 1}/{update_epochs}, Loss: {epoch_loss:.4f}")

        # Update default feature vector
        self.default_features = (self.default_features + np.mean(X, axis=0)) / 2

        # Set model to evaluation mode
        self.model.eval()
        logger.info("Sequence model update completed")
        return self

    def save(self, path):
        """Save model to disk

        Args:
            path (str): Directory path

        Returns:
            bool: Success
        """
        logger.info(f"Saving sequence model to {path}")

        os.makedirs(path, exist_ok=True)

        try:
            # Save model state dict
            torch.save(self.model.state_dict(), os.path.join(path, 'sequence_model.pt'))

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

            # Save user and item data
            with open(os.path.join(path, 'sequence_data.pkl'), 'wb') as f:
                pickle.dump({
                    'user_data': self.user_data,
                    'item_data': self.item_data,
                    'item_indices': self.item_indices
                }, f)

            logger.info("Sequence model saved successfully")
            return True
        except Exception as e:
            logger.error(f"Error saving sequence model: {str(e)}")
            return False

    def load(self, path):
        """Load model from disk

        Args:
            path (str): Directory path

        Returns:
            self: Loaded model
        """
        logger.info(f"Loading sequence model from {path}")

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

            # Load user and item data
            with open(os.path.join(path, 'sequence_data.pkl'), 'rb') as f:
                data = pickle.load(f)

            self.user_data = data['user_data']
            self.item_data = data['item_data']
            self.item_indices = data['item_indices']

            # Initialize model
            input_size = len(self.default_features)
            self.model = GameSequenceModel(
                num_features=input_size,
                hidden_dim=self.hidden_dim,
                num_layers=self.num_layers,
                dropout=self.dropout
            ).to(self.device)

            # Load model state dict
            state_dict = torch.load(os.path.join(path, 'sequence_model.pt'), map_location=self.device)
            self.model.load_state_dict(state_dict)
            self.model.eval()

            logger.info("Sequence model loaded successfully")
            return self
        except Exception as e:
            logger.error(f"Error loading sequence model: {str(e)}")
            return None

    def extract_features(self, user_id, item_id):
        """Extract features for prediction

        Args:
            user_id: User ID
            item_id: Item ID

        Returns:
            ndarray: Feature vector
        """
        # If no feature columns defined, return default features
        if not self.feature_columns or self.default_features is None:
            logger.warning("No feature columns defined, using zeros")
            return np.zeros(self.model.input_size)

        # Initialize feature dictionary
        features = {}

        # Fill with default values
        for i, col in enumerate(self.feature_columns):
            features[col] = self.default_features[i]

        # Update with user-specific features
        if user_id in self.user_data:
            user_features = self.user_data[user_id]
            for col in self.feature_columns:
                if col in user_features:
                    features[col] = user_features[col]

        # Update with item-specific features
        if item_id in self.item_data:
            item_features = self.item_data[item_id]
            for col in self.feature_columns:
                if col in item_features:
                    features[col] = item_features[col]

        # Convert to array with consistent order
        feature_array = np.array([features[col] for col in self.feature_columns])

        return feature_array