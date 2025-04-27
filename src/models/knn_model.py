#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
src/models/knn_model.py - KNN-based collaborative filtering models
Author: YourName
Date: 2025-04-27
Description: Implements user-based and item-based KNN collaborative filtering
"""

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
import logging
import pickle

from models.base_model import BaseRecommenderModel

logger = logging.getLogger(__name__)


class KNNModel(BaseRecommenderModel):
    """KNN-based collaborative filtering model"""

    def __init__(self, type='user', n_neighbors=20, metric='cosine', algorithm='brute'):
        """Initialize KNN model

        Args:
            type (str): 'user' for user-based CF, 'item' for item-based CF
            n_neighbors (int): Number of neighbors to consider
            metric (str): Distance metric
            algorithm (str): KNN algorithm
        """
        self.type = type
        self.n_neighbors = n_neighbors
        self.metric = metric
        self.algorithm = algorithm
        self.model = None
        self.user_indices = {}
        self.item_indices = {}
        self.reversed_user_indices = {}
        self.reversed_item_indices = {}
        self.user_item_matrix = None
        self.sparse_matrix = None

    # Implement fit, predict, recommend, update, save, load methods
    # Extract from your existing KNN implementation in steam_recommender.py