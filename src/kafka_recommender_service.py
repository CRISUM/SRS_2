#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Steam游戏推荐Kafka服务 - 支持增量训练
作者: 融合团队
日期: 2025-04-24
描述: 从Kafka接收消息并使用Steam推荐系统生成推荐，支持增量训练
"""

import argparse
import logging
import json
import os
from steam_recommender import SteamRecommender
from kafka_consumer import KafkaRecommenderService

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)



def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Steam Game Recommendation Kafka Service')
    parser.add_argument('--data', type=str, default=None, help='Data file path')
    parser.add_argument('--model', type=str, default=None, help='Model path')
    parser.add_argument('--save-model', type=str, default='recommender_model', help='Model save path')
    parser.add_argument('--topic', type=str, default='SGR_topic_0', help='Kafka topic')
    parser.add_argument('--max', type=int, default=None, help='Maximum message count')
    parser.add_argument('--batch-size', type=int, default=100, help='Incremental training batch size')
    parser.add_argument('--config', type=str, default=None, help='Configuration file path')

    args = parser.parse_args()

    # Load configuration file
    config = {}
    if args.config and os.path.exists(args.config):
        with open(args.config, 'r') as f:
            config = json.load(f)

    # Command-line arguments override config file
    kafka_config = config.get('kafka_config', {})

    # Initialize recommendation system
    recommender = SteamRecommender(args.data)

    # If model path is provided, try to load the model
    if args.model:
        success = recommender.load_model(args.model)
        if success:
            logger.info(f"Successfully loaded model from {args.model}")
        else:
            logger.warning(f"Failed to load model from {args.model}")

            # If data is provided, try to train a new model
            if args.data:
                logger.info("Training new model with provided data")
                recommender.load_data()
                recommender.engineer_features()
                # Use the new hybrid recommendation approach
                recommender.train_knn_model()
                recommender.train_svd_model()
                recommender.train_simple_model()
                recommender.train_sequence_model()
                recommender.create_game_embeddings()
                recommender.train_content_model()
    elif args.data:
        # If only data is provided, load and train model
        logger.info("Training new model with provided data")
        recommender.load_data()
        recommender.engineer_features()
        # Use the new hybrid recommendation approach
        recommender.train_knn_model()
        recommender.train_svd_model()
        recommender.train_simple_model()
        recommender.train_sequence_model()
        recommender.create_game_embeddings()
        recommender.train_content_model()
    else:
        logger.warning("No data or model provided, will use popular game recommendations")

    # Create Kafka service
    kafka_service = KafkaRecommenderService(
        recommender=recommender,
        kafka_config=kafka_config,
        batch_size=args.batch_size,
        save_model_path=args.save_model
    )

    # Subscribe to topic
    kafka_service.subscribe(args.topic)

    # Start consuming messages
    kafka_service.start_consuming(args.max)

if __name__ == "__main__":
    main()