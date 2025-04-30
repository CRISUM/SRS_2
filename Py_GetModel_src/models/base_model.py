#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Py_GetModel_src/models/base_model.py - Base model class for Steam Game Recommendation System
Author: YourName
Date: 2025-04-27
Description: Defines abstract base classes for all recommendation models
"""

import logging
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class BaseRecommenderModel(ABC):
    """Abstract base class for all recommendation models"""

    @abstractmethod
    def fit(self, data):
        """Train the model with provided data"""
        pass

    @abstractmethod
    def predict(self, user_id, item_id):
        """Predict rating for a user-item pair"""
        pass

    @abstractmethod
    def recommend(self, user_id, n=10):
        """Generate top-N recommendations for a user"""
        pass

    @abstractmethod
    def update(self, new_data):
        """Update model with new data (incremental learning)"""
        pass

    @abstractmethod
    def save(self, path):
        """Save model to disk"""
        pass

    @abstractmethod
    def load(self, path):
        """Load model from disk"""
        pass