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


# 修改kafka_recommender_service.py中的main函数

def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='Steam游戏推荐Kafka服务')
    parser.add_argument('--data', type=str, default=None, help='数据文件路径')
    parser.add_argument('--model', type=str, default=None, help='模型路径')
    parser.add_argument('--save-model', type=str, default='recommender_model', help='模型保存路径')
    parser.add_argument('--topic', type=str, default='SGR_topic_0', help='Kafka主题')
    parser.add_argument('--max', type=int, default=None, help='最大消息数')
    parser.add_argument('--batch-size', type=int, default=100, help='增量训练批次大小')
    parser.add_argument('--config', type=str, default=None, help='配置文件路径')

    args = parser.parse_args()

    # 加载配置文件
    config = {}
    if args.config and os.path.exists(args.config):
        with open(args.config, 'r') as f:
            config = json.load(f)

    # 命令行参数覆盖配置文件
    kafka_config = config.get('kafka_config', {})

    # 初始化推荐系统
    recommender = SteamRecommender(args.data)

    # 如果提供了模型路径，尝试加载模型
    if args.model:
        success = recommender.load_model(args.model)
        if success:
            logger.info(f"成功从 {args.model} 加载模型")
        else:
            logger.warning(f"从 {args.model} 加载模型失败")

            # 如果提供了数据，尝试训练模型
            if args.data:
                logger.info("使用提供的数据训练新模型")
                recommender.load_data()
                recommender.engineer_features()
                # 使用KNN模型代替LightGBM模型
                recommender.train_knn_model()
                recommender.train_sequence_model()
                recommender.create_game_embeddings()
                recommender.train_content_model()
    elif args.data:
        # 如果只提供了数据，加载并训练模型
        logger.info("使用提供的数据训练新模型")
        recommender.load_data()
        recommender.engineer_features()
        # 使用KNN模型代替LightGBM模型
        recommender.train_knn_model()
        recommender.train_sequence_model()
        recommender.create_game_embeddings()
        recommender.train_content_model()
    else:
        logger.warning("未提供数据和模型，将使用热门游戏推荐")

    # 创建Kafka服务
    kafka_service = KafkaRecommenderService(
        recommender=recommender,
        kafka_config=kafka_config,
        batch_size=args.batch_size,
        save_model_path=args.save_model
    )

    # 订阅主题
    kafka_service.subscribe(args.topic)

    # 开始消费消息
    kafka_service.start_consuming(args.max)

if __name__ == "__main__":
    main()