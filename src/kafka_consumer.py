# 第一行注释：创建新文件 kafka_consumer.py

from confluent_kafka import Consumer, KafkaException
import json
import logging
import pandas as pd
from steam_recommender import SteamRecommender

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# 修改 kafka_consumer.py 文件

class KafkaRecommenderService:
    """Kafka推荐服务类，支持增量训练"""

    def __init__(self, recommender, kafka_config=None, batch_size=100, save_model_path=None):
        """
        初始化服务

        参数:
            recommender: SteamRecommender实例
            kafka_config: Kafka配置
            batch_size: 触发增量训练的消息数量阈值
            save_model_path: 模型保存路径
        """
        self.recommender = recommender
        self.batch_size = batch_size
        self.save_model_path = save_model_path

        # 累积数据容器
        self.accumulated_games = []
        self.accumulated_users = []
        self.accumulated_interactions = []
        self.message_count = 0

        # 默认Kafka配置
        self.kafka_config = {
            'bootstrap.servers': 'pkc-312o0.ap-southeast-1.aws.confluent.cloud:9092',
            'group.id': 'realtime-recommender',
            'auto.offset.reset': 'earliest',
            'security.protocol': 'SASL_SSL',
            'sasl.mechanisms': 'PLAIN',
            'sasl.username': '2ONZ7HJKO3ADC23J',
            'sasl.password': '7gprOxjuBXtDXxGAqeXFOjogHI4xCm+Jw1EH+hYoucMWze3t2K71+Es/mLMaKsIV'
        }

        # 更新配置
        if kafka_config:
            self.kafka_config.update(kafka_config)

        # 初始化Kafka消费者
        self.consumer = Consumer(self.kafka_config)

    def subscribe(self, topic='SGR_topic_0'):
        """订阅Kafka主题"""
        self.consumer.subscribe([topic])
        logger.info(f"已订阅主题: {topic}")

    def process_game_data(self, game_data):
        """
        处理游戏数据，更新推荐器的游戏信息并累积数据

        参数:
            game_data: 游戏数据列表
        """
        if not game_data:
            return

        # 累积游戏数据用于增量训练
        self.accumulated_games.extend(game_data)

        # 转换为DataFrame
        games_df = pd.DataFrame(game_data)

        # 如果推荐器没有数据，直接使用这些数据
        if not hasattr(self.recommender, 'df') or self.recommender.df is None:
            self.recommender.df = pd.DataFrame()

        # 将新游戏数据合并到推荐器的数据中
        for game in game_data:
            app_id = game['app_id']
            # 检查游戏是否已存在
            if 'app_id' in self.recommender.df.columns and app_id in self.recommender.df['app_id'].values:
                continue

            # 添加新游戏
            self.recommender.df = pd.concat([self.recommender.df, pd.DataFrame([game])])

    def process_user_data(self, user_data):
        """
        处理用户数据，更新推荐器的用户信息并累积数据

        参数:
            user_data: 用户数据列表
        """
        if not user_data:
            return

        # 累积用户数据用于增量训练
        self.accumulated_users.extend(user_data)

        # 更新用户元数据
        if not hasattr(self.recommender, 'user_metadata'):
            self.recommender.user_metadata = {}

        for user in user_data:
            user_id = user['user_id']
            self.recommender.user_metadata[user_id] = user

    def process_metadata(self, metadata):
        """
        处理元数据，更新推荐器的游戏详细信息

        参数:
            metadata: 元数据列表
        """
        if not metadata:
            return

        # 更新游戏标签和描述等信息
        for meta in metadata:
            app_id = meta['app_id']
            if 'app_id' in self.recommender.df.columns and app_id in self.recommender.df['app_id'].values:
                # 更新现有游戏信息
                mask = self.recommender.df['app_id'] == app_id
                if 'tags' in meta and meta['tags']:
                    self.recommender.df.loc[mask, 'tags'] = meta['tags']
                if 'description' in meta and meta['description']:
                    self.recommender.df.loc[mask, 'description'] = meta['description']

    def process_recommendations(self, recommendations):
        """
        处理推荐数据，提取用户-游戏交互并累积

        参数:
            recommendations: 推荐数据列表
        """
        if not recommendations:
            return

        # 累积交互数据用于增量训练
        self.accumulated_interactions.extend(recommendations)

        # 更新原始数据集中的交互信息
        for rec in recommendations:
            user_id = rec.get('user_id')
            app_id = rec.get('app_id')
            is_recommended = rec.get('is_recommended', False)
            hours = rec.get('hours', 0)

            if not user_id or not app_id:
                continue

            # 构建交互记录
            interaction = {
                'user_id': user_id,
                'app_id': app_id,
                'is_recommended': is_recommended,
                'hours': hours
            }

            # 如果有时间戳，也添加进去
            if 'timestamp' in rec:
                interaction['date'] = rec['timestamp']

            # 添加到原始数据
            if hasattr(self.recommender, 'df'):
                # 检查是否已存在这条交互
                mask = (
                        (self.recommender.df['user_id'] == user_id) &
                        (self.recommender.df['app_id'] == app_id)
                )
                if 'user_id' in self.recommender.df.columns and sum(mask) > 0:
                    # 更新现有记录
                    self.recommender.df.loc[mask, 'is_recommended'] = is_recommended
                    self.recommender.df.loc[mask, 'hours'] = hours
                else:
                    # 添加新记录
                    self.recommender.df = pd.concat([self.recommender.df, pd.DataFrame([interaction])])

    def check_incremental_training(self):
        """
        检查是否需要执行增量训练，并在必要时触发训练
        """
        # 检查是否达到增量训练的阈值
        if self.message_count >= self.batch_size and self.accumulated_interactions:
            logger.info(f"已处理 {self.message_count} 条消息，执行增量训练...")

            # 将累积的数据转换为DataFrame
            interactions_df = pd.DataFrame(self.accumulated_interactions)
            games_df = pd.DataFrame(self.accumulated_games) if self.accumulated_games else None
            users_df = pd.DataFrame(self.accumulated_users) if self.accumulated_users else None

            # 执行增量训练
            self.perform_incremental_training(interactions_df, games_df, users_df)

            # 重置累积数据
            self.accumulated_games = []
            self.accumulated_users = []
            self.accumulated_interactions = []
            self.message_count = 0

            logger.info("增量训练完成")

    # 在kafka_consumer.py中修改perform_incremental_training方法

    def perform_incremental_training(self, interactions_df, games_df=None, users_df=None):
        """
        Execute incremental training

        Parameters:
            interactions_df: User-game interaction data
            games_df: Game data
            users_df: User data
        """
        try:
            # First make sure recommender system is initialized
            if not hasattr(self.recommender, 'df') or self.recommender.df is None:
                logger.warning("Recommender system not initialized, cannot perform incremental training")
                return

            # If this is the first training, perform full model training
            if (not hasattr(self.recommender, 'user_knn_model') or self.recommender.user_knn_model is None or
                    not hasattr(self.recommender, 'item_knn_model') or self.recommender.item_knn_model is None):
                logger.info("Performing first complete model training...")
                self.recommender.engineer_features()
                # Use KNN models and alternative methods instead of LightGBM
                self.recommender.train_knn_model()
                self.recommender.train_svd_model()  # New SVD-based collaborative filtering
                self.recommender.train_simple_model()  # Simple classifier replacing LightGBM
                self.recommender.train_sequence_model()
                self.recommender.create_game_embeddings()
                self.recommender.train_content_model()
            else:
                # Otherwise, perform incremental update
                logger.info("Executing incremental model update...")

                # Update KNN models
                self.recommender.update_knn_model(interactions_df)

                # Update SVD model
                if hasattr(self.recommender, 'update_svd_model'):
                    self.recommender.update_svd_model(interactions_df)

                # Update simple model
                if hasattr(self.recommender, 'update_simple_model'):
                    self.recommender.update_simple_model(interactions_df)

                # Update sequence model
                if hasattr(self.recommender, 'update_sequence_model'):
                    self.recommender.update_sequence_model(interactions_df)

                # Update content model
                if games_df is not None and len(games_df) > 0 and hasattr(self.recommender, 'update_content_model'):
                    self.recommender.update_content_model(games_df)

                # Update game embeddings
                if hasattr(self.recommender, 'create_game_embeddings'):
                    self.recommender.create_game_embeddings()

            # Save updated model
            if self.save_model_path:
                self.recommender.save_model(self.save_model_path)
                logger.info(f"Updated model saved to {self.save_model_path}")

        except Exception as e:
            logger.error(f"Error in incremental training: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())

    def handle_message(self, data):
        """
        处理Kafka消息

        参数:
            data: 消息数据
        """
        try:
            # 提取各部分数据
            recommendations = data.get('recommendations', [])
            games = data.get('games', [])
            users = data.get('users', [])
            metadata = data.get('metadata', [])

            # 处理游戏数据
            self.process_game_data(games)

            # 处理用户数据
            self.process_user_data(users)

            # 处理元数据
            self.process_metadata(metadata)

            # 处理交互数据（用于增量训练）
            self.process_recommendations(recommendations)

            # 增加消息计数
            self.message_count += 1

            # 检查是否需要增量训练
            self.check_incremental_training()

            # 处理推荐请求
            for rec in recommendations:
                user_id = rec.get('user_id')
                app_id = rec.get('app_id')

                if not user_id:
                    logger.warning("消息中没有user_id，跳过处理")
                    continue

                logger.info(f"处理用户 {user_id} 的推荐请求")

                # 生成推荐
                try:
                    user_recs = self.recommender.generate_recommendations(user_id, top_n=5)

                    if not user_recs:
                        logger.warning(f"用户 {user_id} 没有推荐结果，使用备选策略")
                        if app_id:
                            # 使用内容推荐作为备选
                            user_recs = self.recommender.get_content_recommendations(app_id, top_n=5)
                        else:
                            # 使用热门游戏作为备选
                            user_recs = self.recommender.get_popular_games(top_n=5)

                    # 打印推荐结果
                    logger.info(f"为用户 {user_id} 生成的推荐:")
                    for i, (game_id, score) in enumerate(user_recs, 1):
                        title = self.get_game_title(game_id)
                        logger.info(f"{i}. {title} (ID: {game_id}, 得分: {score:.4f})")

                except Exception as e:
                    logger.error(f"生成推荐时出错: {str(e)}")

        except Exception as e:
            logger.error(f"处理消息时出错: {str(e)}")

    def get_game_title(self, game_id):
        """获取游戏标题"""
        if hasattr(self.recommender,
                   'df') and 'app_id' in self.recommender.df.columns and 'title' in self.recommender.df.columns:
            game = self.recommender.df[self.recommender.df['app_id'] == game_id]
            if not game.empty:
                return game['title'].iloc[0]
        return f"游戏 {game_id}"

    def start_consuming(self, max_messages=None):
        """
        开始消费Kafka消息

        参数:
            max_messages: 最大消息数，None表示持续消费
        """
        logger.info("开始消费Kafka消息...")
        total_message_count = 0

        try:
            while True:
                msg = self.consumer.poll(1.0)

                if msg is None:
                    continue

                if msg.error():
                    logger.error(f"Kafka错误: {msg.error()}")
                    continue

                try:
                    # 解析消息
                    logger.info(f"[{total_message_count + 1}] 收到Kafka消息")
                    value = msg.value().decode('utf-8')

                    data = json.loads(value)

                    # 处理消息
                    self.handle_message(data)

                    total_message_count += 1
                    if max_messages and total_message_count >= max_messages:
                        logger.info(f"已达到最大消息数 {max_messages}，停止消费")
                        break

                except json.JSONDecodeError as e:
                    logger.error(f"JSON解析错误: {str(e)}")
                except Exception as e:
                    logger.error(f"处理消息时出错: {str(e)}")

        except KeyboardInterrupt:
            logger.info("用户中断，停止消费")
        finally:
            # 在结束前执行一次增量训练，确保所有累积的数据都被使用
            if self.accumulated_interactions:
                logger.info("执行最终增量训练...")
                interactions_df = pd.DataFrame(self.accumulated_interactions)
                games_df = pd.DataFrame(self.accumulated_games) if self.accumulated_games else None
                users_df = pd.DataFrame(self.accumulated_users) if self.accumulated_users else None
                self.perform_incremental_training(interactions_df, games_df, users_df)

            self.consumer.close()
            logger.info(f"已处理 {total_message_count} 条消息，消费者已关闭")