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

        if not hasattr(self, 'train_df') or df is None:
            logger.error("No training data available")
            return None

        try:
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
                tag_embedding_dim = min(self.config['tag_embedding_dim'], tag_features.shape[1])
                svd = TruncatedSVD(n_components=tag_embedding_dim, random_state=42)
                tag_embeddings = svd.fit_transform(tag_features)

                # Create embedding dictionary
                self.game_embeddings = {}
                for i, app_id in enumerate(games_df['app_id']):
                    self.game_embeddings[app_id] = tag_embeddings[i]

                logger.info(f"Created tag-based embeddings for {len(self.game_embeddings)} games")

            # Enhance embeddings with collaborative information if SVD model is available
            if hasattr(self, 'svd_model') and self.svd_model is not None:
                # Get game factors from SVD model
                item_factors = self.svd_model.item_factors

                # Create or enhance game embeddings
                if not hasattr(self, 'game_embeddings') or self.game_embeddings is None:
                    self.game_embeddings = {}

                for app_id, idx in self.svd_model.item_map.items():
                    # Get SVD factors for this game
                    svd_embedding = item_factors[idx]

                    # If game already has tag-based embedding, concatenate with SVD factors
                    if app_id in self.game_embeddings:
                        tag_embedding = self.game_embeddings[app_id]
                        # Normalize both embeddings
                        tag_norm = np.linalg.norm(tag_embedding)
                        svd_norm = np.linalg.norm(svd_embedding)

                        if tag_norm > 0 and svd_norm > 0:
                            tag_embedding = tag_embedding / tag_norm
                            svd_embedding = svd_embedding / svd_norm

                        # Concatenate and store
                        self.game_embeddings[app_id] = np.concatenate([tag_embedding, svd_embedding])
                    else:
                        # Store SVD factors directly
                        self.game_embeddings[app_id] = svd_embedding

                logger.info(f"Enhanced embeddings with collaborative information")

            # Store number of games with embeddings
            self.training_history['games_with_embeddings'] = len(self.game_embeddings) if hasattr(self,
                                                                                                  'game_embeddings') else 0
            return self.game_embeddings

        except Exception as e:
            logger.error(f"Error creating game embeddings: {str(e)}")
            logger.error(traceback.format_exc())
            return None


    def get_similarity_matrix(self, embeddings):
        """Create item-item similarity matrix from embeddings"""
        # Implement similarity matrix calculation