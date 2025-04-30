#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Py_GetModel_src/main.py - Main entry point for Steam Recommender
Author: YourName
Date: 2025-04-27
Description: Command-line interface for the Steam Game Recommender System
"""

import argparse
import logging
import json
import os
from Py_GetModel_src.recommender import SteamRecommender
from kafka.kafka_consumer_service import KafkaRecommenderService


def setup_logging(level=logging.INFO):
    """Set up logging configuration"""
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


def main():
    """Main entry point"""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Steam Game Recommendation System')
    parser.add_argument('--data', type=str, default=None, help='Data file path')
    parser.add_argument('--model', type=str, default=None, help='Model path')
    parser.add_argument('--save-model', type=str, default='recommender_model', help='Model save path')
    parser.add_argument('--mode', type=str, default='train',
                        choices=['train', 'optimal-train', 'recommend', 'evaluate', 'serve'],
                        help='Operation mode')
    parser.add_argument('--topic', type=str, default='SGR_topic_0', help='Kafka topic')
    parser.add_argument('--max', type=int, default=None, help='Maximum message count')
    parser.add_argument('--batch-size', type=int, default=100, help='Incremental training batch size')
    parser.add_argument('--config', type=str, default=None, help='Configuration file path')
    parser.add_argument('--test-knn', action='store_true', help='测试不同KNN邻居数量的效果')
    parser.add_argument('--min-neighbors', type=int, default=10, help='KNN测试的最小邻居数量')
    parser.add_argument('--max-neighbors', type=int, default=60, help='KNN测试的最大邻居数量')
    parser.add_argument('--step-neighbors', type=int, default=10, help='KNN测试的邻居数量步长')

    args = parser.parse_args()

    # Set up logging
    setup_logging()

    # Load configuration if provided
    config = {}
    if args.config and os.path.exists(args.config):
        with open(args.config, 'r') as f:
            config = json.load(f)

    # Initialize recommender
    recommender = SteamRecommender(args.data, config)

    # Execute based on mode
    if args.mode == 'train':
        # Training mode
        if args.data:
            recommender.load_data()
            recommender.engineer_features()
            recommender.train_models()
            recommender.evaluate_recommendations()
            recommender.visualize_results()
            recommender.save_model(args.save_model)
        else:
            logging.error("Training requires data file path")

    # 在模式选择中添加最佳参数训练选项
    elif args.mode == 'optimal-train':
        # 训练模式 - 使用最佳参数
        if args.data:
            recommender.load_data()
            recommender.engineer_features()
            recommender.train_with_optimal_parameters()  # 使用最佳参数训练
            recommender.evaluate_recommendations()
            recommender.visualize_results()
            recommender.save_model(args.save_model)
        else:
            logging.error("Training requires data file path")

    elif args.mode == 'recommend':
        # Recommendation mode - interactive CLI for recommendations
        if args.model:
            recommender.load_model(args.model)

            # Interactive recommendations
            print("Steam Game Recommender - Interactive Mode")
            print("Enter user ID to get recommendations (or 'quit' to exit)")

            while True:
                user_input = input("User ID> ")
                if user_input.lower() in ('quit', 'exit'):
                    break

                try:
                    user_id = int(user_input)
                    recommendations = recommender.generate_recommendations(user_id, 10)

                    print(f"\nTop 10 recommendations for user {user_id}:")
                    for i, (game_id, score) in enumerate(recommendations, 1):
                        game_title = "Unknown"
                        if hasattr(recommender, 'data_processor') and hasattr(recommender.data_processor, 'df'):
                            game_data = recommender.data_processor.df[
                                recommender.data_processor.df['app_id'] == game_id
                                ]
                            if not game_data.empty and 'title' in game_data.columns:
                                game_title = game_data['title'].iloc[0]

                        print(f"{i}. {game_title} (ID: {game_id}, Score: {score:.4f})")
                    print()

                except ValueError:
                    print("Invalid user ID. Please enter a numeric ID.")
        else:
            logging.error("Recommendation requires model path")

    elif args.mode == 'evaluate':
        # Evaluation mode
        if args.model:
            recommender.load_model(args.model)
            results = recommender.evaluate_recommendations()
            recommender.visualize_results()

            # Print results
            print("\nEvaluation Results:")
            for metric, values in results.items():
                if isinstance(values, dict):
                    print(f"\n{metric.capitalize()}:")
                    for k, v in values.items():
                        print(f"  @{k}: {v:.4f}")
                else:
                    print(f"\n{metric.capitalize()}: {values:.4f}")
        else:
            logging.error("Evaluation requires model path")

        # 添加KNN参数测试
        if args.test_knn:
            print("\n测试不同KNN参数的效果:")
            user_range = range(args.min_neighbors, args.max_neighbors + 1, args.step_neighbors)
            item_range = range(args.min_neighbors, args.max_neighbors + 1, args.step_neighbors)

            knn_results = recommender.test_knn_clustering(
                user_neighbors_range=list(user_range),
                item_neighbors_range=list(item_range)
            )

            if knn_results:
                print("\nKNN参数测试结果:")
                for config, metrics in knn_results.items():
                    print(f"配置 {config}: NDCG@10 = {metrics['ndcg'][10]:.4f}")

    elif args.mode == 'serve':
        # Kafka service mode
        if args.model:
            recommender.load_model(args.model)
        elif args.data:
            recommender.load_data()
            recommender.engineer_features()
            recommender.train_and_optimize()
            recommender.save_model(args.save_model)
        else:
            logging.error("Service requires either model path or data file")
            return

        # Create Kafka service
        kafka_config = config.get('kafka_config', {})
        service = KafkaRecommenderService(
            recommender=recommender,
            kafka_config=kafka_config,
            batch_size=args.batch_size,
            save_model_path=args.save_model
        )

        # Start service
        service.subscribe(args.topic)
        service.start_consuming(args.max)


if __name__ == "__main__":
    main()