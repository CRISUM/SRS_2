#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Py_GetModel_src/kafka/kafka_consumer_service.py - Kafka consumer service
Author: YourName
Date: 2025-04-27
Description: Service that consumes Kafka messages and updates the recommender
"""

import logging
import json
import pandas as pd

from kafka.kafka_manager import KafkaManager

logger = logging.getLogger(__name__)


class KafkaRecommenderService:
    """Kafka recommendation service supporting incremental training"""

    def __init__(self, recommender, kafka_config=None, batch_size=100, save_model_path=None):
        """Initialize service

        Args:
            recommender: SteamRecommender instance
            kafka_config: Kafka configuration
            batch_size: Threshold for triggering incremental training
            save_model_path: Model save path
        """
        # Extract from your existing KafkaRecommenderService implementation
        # Implement the service using the new KafkaManager