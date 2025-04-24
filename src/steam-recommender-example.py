#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Steam游戏推荐系统使用示例
作者: Claude
日期: 2025-04-24
描述: 演示如何使用基于LightGBM和序列行为的Steam游戏推荐系统
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from steam_recommender import SteamRecommender

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def run_training_pipeline(data_path, config=None):
    """
    运行完整的训练流程

    参数:
        data_path (str): 数据文件路径
        config (dict, optional): 配置参数
    """
    logger.info("开始训练流程...")

    # 默认配置
    if config is None:
        config = {
            'lgbm_params': {
                'objective': 'binary',
                'metric': 'auc',
                'boosting_type': 'gbdt',
                'learning_rate': 0.05,
                'num_leaves': 31,
                'max_depth': -1,
                'min_child_samples': 20,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'random_state': 42,
                'n_estimators': 500,  # 减少迭代次数，加快训练
                'early_stopping_rounds': 50,
                'verbose': -1
            },
            'sequence_params': {
                'embedding_dim': 32,  # 降低嵌入维度
                'hidden_dim': 64,  # 降低隐藏层维度
                'num_layers': 1,  # 减少层数
                'dropout': 0.2,
                'batch_size': 64,
                'learning_rate': 0.001,
                'epochs': 5  # 减少训练轮数
            },
            'tag_embedding_dim': 30,
            'text_embedding_dim': 50,
            'max_seq_length': 10,
            'time_decay_factor': 0.9,
            'n_recommendations': 10,
            'content_weight': 0.3,
            'sequence_weight': 0.3,
            'lgbm_weight': 0.4
        }

    # 初始化推荐系统
    recommender = SteamRecommender(data_path, config)

    # 执行训练流程
    recommender.load_data()
    recommender.engineer_features()
    recommender.train_lgbm_model()
    recommender.train_sequence_model()
    recommender.create_game_embeddings()
    recommender.train_content_model()

    # 评估模型
    evaluation_results = recommender.evaluate_recommendations(k_values=[5, 10])
    recommender.evaluation_results = evaluation_results

    # 可视化结果
    recommender.visualize_results()

    # 保存模型
    recommender.save_model('trained_model')

    logger.info("训练流程完成")
    return recommender


def run_inference(model_path, user_id=None, top_n=10):
    """
    使用训练好的模型为用户生成推荐

    参数:
        model_path (str): 模型保存路径
        user_id (int, optional): 用户ID，如果为None则使用示例用户
        top_n (int): 推荐数量
    """
    logger.info(f"为用户 {user_id} 生成推荐...")

    # 加载模型
    recommender = SteamRecommender(None)  # 创建空推荐器
    success = recommender.load_model(model_path)

    if not success:
        logger.error("模型加载失败")
        return

    # 如果没有指定用户ID，使用数据集中的第一个用户
    if user_id is None and hasattr(recommender, 'df'):
        user_id = recommender.df['user_id'].iloc[0]
        logger.info(f"使用示例用户ID: {user_id}")

    # 生成推荐
    recommendations = recommender.generate_recommendations(user_id, top_n)

    # 打印推荐结果
    print(f"\n为用户 {user_id} 的推荐:")
    for i, (game_id, score) in enumerate(recommendations, 1):
        if hasattr(recommender, 'df'):
            try:
                game_title = recommender.df[recommender.df['app_id'] == game_id]['title'].iloc[0]
                print(f"{i}. {game_title} (ID: {game_id}, Score: {score:.4f})")
            except:
                print(f"{i}. 游戏ID: {game_id} (Score: {score:.4f})")
        else:
            print(f"{i}. 游戏ID: {game_id} (Score: {score:.4f})")

    return recommendations


def analyze_user_behavior(data_path, user_id):
    """
    分析用户行为模式

    参数:
        data_path (str): 数据文件路径
        user_id (int): 用户ID
    """
    logger.info(f"分析用户 {user_id} 的行为...")

    # 读取数据
    df = pd.read_csv(data_path)

    # 获取用户数据
    user_data = df[df['user_id'] == user_id]

    if len(user_data) == 0:
        logger.error(f"未找到用户 {user_id} 的数据")
        return

    # 1. 基本统计
    total_games = len(user_data)
    avg_hours = user_data['hours'].mean()
    total_hours = user_data['hours'].sum()
    recommended_ratio = user_data['is_recommended'].mean() if 'is_recommended' in user_data.columns else None

    print(f"\n用户 {user_id} 的游戏行为分析:")
    print(f"总游戏数: {total_games}")
    print(f"平均游戏时间: {avg_hours:.2f} 小时")
    print(f"总游戏时间: {total_hours:.2f} 小时")
    if recommended_ratio is not None:
        print(f"推荐比例: {recommended_ratio:.2%}")

    # 2. 游戏类型偏好
    if 'tags' in user_data.columns:
        all_tags = []
        for tags_str in user_data['tags'].dropna():
            all_tags.extend([tag.strip() for tag in tags_str.split(',')])

        tag_counts = pd.Series(all_tags).value_counts()
        top_tags = tag_counts.head(10)

        print("\n用户游戏类型偏好 (前10):")
        for tag, count in top_tags.items():
            print(f"{tag}: {count} 次")

        # 可视化标签偏好
        plt.figure(figsize=(12, 6))
        top_tags.plot(kind='bar')
        plt.title(f'用户 {user_id} 的游戏类型偏好')
        plt.ylabel('频次')
        plt.xlabel('游戏标签')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(f'user_{user_id}_tag_preferences.png')
        plt.close()

    # 3. 游戏时间分布
    plt.figure(figsize=(10, 6))
    sns.histplot(user_data['hours'], bins=20)
    plt.title(f'用户 {user_id} 的游戏时间分布')
    plt.xlabel('游戏时间 (小时)')
    plt.ylabel('频次')
    plt.tight_layout()
    plt.savefig(f'user_{user_id}_hours_distribution.png')
    plt.close()

    # 4. 时间序列分析
    if 'date' in user_data.columns:
        user_data['date'] = pd.to_datetime(user_data['date'])
        user_data = user_data.sort_values('date')

        # 按月统计游戏活动
        user_data['month'] = user_data['date'].dt.to_period('M')
        monthly_activity = user_data.groupby('month').size()

        plt.figure(figsize=(12, 6))
        monthly_activity.plot(kind='line', marker='o')
        plt.title(f'用户 {user_id} 的月度游戏活动')
        plt.ylabel('游戏评论数')
        plt.xlabel('月份')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f'user_{user_id}_monthly_activity.png')
        plt.close()

    logger.info(f"用户 {user_id} 的行为分析完成")


def analyze_game_popularity(data_path, top_n=20):
    """
    分析游戏流行度

    参数:
        data_path (str): 数据文件路径
        top_n (int): 显示前N个最受欢迎的游戏
    """
    logger.info("分析游戏流行度...")

    # 读取数据
    df = pd.read_csv(data_path)

    # 计算游戏流行度指标
    game_stats = df.groupby('app_id').agg({
        'user_id': 'count',  # 评论数
        'hours': ['mean', 'sum'],  # 游戏时间
        'is_recommended': ['mean', 'sum'] if 'is_recommended' in df.columns else [],  # 推荐率
        'rating_new': ['mean'] if 'rating_new' in df.columns else []  # 评分
    })

    # 展平多级索引列名
    game_stats.columns = ['_'.join(col).strip() for col in game_stats.columns.values]
    game_stats = game_stats.reset_index()

    # 合并游戏标题
    game_titles = df[['app_id', 'title']].drop_duplicates('app_id')
    game_stats = game_stats.merge(game_titles, on='app_id', how='left')

    # 计算综合流行度分数
    # 规范化各指标
    for col in ['user_id_count', 'hours_sum']:
        if col in game_stats.columns:
            game_stats[f'{col}_norm'] = game_stats[col] / game_stats[col].max()

    # 创建综合分数
    rating_col = 'rating_new_mean' if 'rating_new_mean' in game_stats.columns else 'is_recommended_mean'
    if rating_col in game_stats.columns:
        game_stats['popularity_score'] = (
                game_stats['user_id_count_norm'] * 0.6 +
                game_stats['hours_sum_norm'] * 0.2 +
                game_stats[rating_col] * 0.2
        )
    else:
        game_stats['popularity_score'] = (
                game_stats['user_id_count_norm'] * 0.7 +
                game_stats['hours_sum_norm'] * 0.3
        )

    # 排序并获取最受欢迎的游戏
    popular_games = game_stats.sort_values('popularity_score', ascending=False).head(top_n)

    print("\n最受欢迎的游戏:")
    for idx, row in popular_games.iterrows():
        print(f"{row['title']} (ID: {row['app_id']}, 流行度分数: {row['popularity_score']:.4f})")

    # 可视化
    plt.figure(figsize=(12, 8))
    sns.barplot(x='popularity_score', y='title', data=popular_games.head(15))
    plt.title('最受欢迎的15款游戏')
    plt.xlabel('流行度分数')
    plt.tight_layout()
    plt.savefig('popular_games.png')
    plt.close()

    logger.info("游戏流行度分析完成")


def compare_recommendation_methods(data_path, user_id, top_n=10):
    """
    比较不同推荐方法的结果

    参数:
        data_path (str): 数据文件路径
        user_id (int): 用户ID
        top_n (int): 推荐数量
    """
    logger.info(f"比较为用户 {user_id} 的不同推荐方法...")

    # 初始化推荐系统
    recommender = SteamRecommender(data_path)

    # 加载并处理数据
    recommender.load_data()
    recommender.engineer_features()

    # 训练不同的模型
    recommender.train_lgbm_model()
    recommender.train_sequence_model()
    recommender.train_content_model()

    # 1. 仅使用LightGBM的推荐
    original_weights = recommender.config.copy()
    recommender.config['lgbm_weight'] = 1.0
    recommender.config['sequence_weight'] = 0.0
    recommender.config['content_weight'] = 0.0
    lgbm_recommendations = recommender.generate_recommendations(user_id, top_n)

    # 2. 仅使用序列模型的推荐
    recommender.config['lgbm_weight'] = 0.0
    recommender.config['sequence_weight'] = 1.0
    recommender.config['content_weight'] = 0.0
    sequence_recommendations = recommender.generate_recommendations(user_id, top_n)

    # 3. 仅使用内容模型的推荐
    recommender.config['lgbm_weight'] = 0.0
    recommender.config['sequence_weight'] = 0.0
    recommender.config['content_weight'] = 1.0
    content_recommendations = recommender.generate_recommendations(user_id, top_n)

    # 4. 混合模型的推荐
    recommender.config = original_weights
    hybrid_recommendations = recommender.generate_recommendations(user_id, top_n)

    # 获取游戏标题
    game_titles = {}
    for game_id in recommender.df['app_id'].unique():
        title = recommender.df[recommender.df['app_id'] == game_id]['title'].iloc[0]
        game_titles[game_id] = title

    # 打印比较结果
    methods = [
        ("LightGBM", lgbm_recommendations),
        ("序列模型", sequence_recommendations),
        ("内容模型", content_recommendations),
        ("混合模型", hybrid_recommendations)
    ]

    print(f"\n为用户 {user_id} 的不同推荐方法比较:")
    for method_name, recs in methods:
        print(f"\n{method_name} 推荐:")
        for i, (game_id, score) in enumerate(recs, 1):
            title = game_titles.get(game_id, f"未知游戏 (ID: {game_id})")
            print(f"{i}. {title} - 得分: {score:.4f}")

    # 计算推荐重叠度
    def jaccard_similarity(list1, list2):
        set1 = set([game_id for game_id, _ in list1])
        set2 = set([game_id for game_id, _ in list2])
        return len(set1.intersection(set2)) / len(set1.union(set2))

    print("\n不同方法之间的推荐重叠度 (Jaccard相似度):")
    for i, (method1, recs1) in enumerate(methods):
        for j, (method2, recs2) in enumerate(methods):
            if i < j:
                similarity = jaccard_similarity(recs1, recs2)
                print(f"{method1} vs {method2}: {similarity:.2f}")

    # 可视化不同方法的游戏类型分布
    if 'tags' in recommender.df.columns:
        method_tags = {}

        for method_name, recs in methods:
            method_tags[method_name] = []
            for game_id, _ in recs:
                game_data = recommender.df[recommender.df['app_id'] == game_id]
                if not game_data.empty and pd.notna(game_data['tags'].iloc[0]):
                    tags = [tag.strip() for tag in game_data['tags'].iloc[0].split(',')]
                    method_tags[method_name].extend(tags)

        # 创建标签分布数据
        tag_data = []
        for method, tags in method_tags.items():
            tag_counter = pd.Series(tags).value_counts().head(10)
            for tag, count in tag_counter.items():
                tag_data.append({'方法': method, '标签': tag, '数量': count})

        tag_df = pd.DataFrame(tag_data)

        # 绘制堆叠条形图
        plt.figure(figsize=(15, 8))
        sns.barplot(x='方法', y='数量', hue='标签', data=tag_df)
        plt.title('不同推荐方法的游戏类型分布 (前10个标签)')
        plt.xticks(rotation=0)
        plt.legend(title='游戏标签', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(f'user_{user_id}_recommendation_methods_comparison.png')
        plt.close()

    logger.info("推荐方法比较完成")
    return methods


def analyze_game_similarity(data_path, game_id, n=10):
    """
    分析游戏相似度，找出与指定游戏最相似的其他游戏

    参数:
        data_path (str): 数据文件路径
        game_id (int): 游戏ID
        n (int): 返回最相似的游戏数量
    """
    logger.info(f"分析与游戏 {game_id} 最相似的游戏...")

    # 读取数据
    df = pd.read_csv(data_path)

    # 检查游戏是否存在
    if game_id not in df['app_id'].values:
        logger.error(f"游戏ID {game_id} 不存在于数据集中")
        return

    # 获取游戏标题
    game_title = df[df['app_id'] == game_id]['title'].iloc[0]

    # 基于标签的相似度
    if 'tags' in df.columns:
        # 获取所有游戏的标签
        game_tags = {}
        for _, row in df.drop_duplicates('app_id').iterrows():
            if pd.notna(row['tags']):
                tags = set(tag.strip() for tag in row['tags'].split(','))
                game_tags[row['app_id']] = tags

        # 计算与目标游戏的Jaccard相似度
        if game_id in game_tags:
            target_tags = game_tags[game_id]
            similarities = []

            for other_id, other_tags in game_tags.items():
                if other_id != game_id and other_tags:  # 跳过目标游戏和没有标签的游戏
                    jaccard = len(target_tags & other_tags) / len(target_tags | other_tags)
                    similarities.append((other_id, jaccard))

            # 排序并获取最相似的游戏
            similar_games = sorted(similarities, key=lambda x: x[1], reverse=True)[:n]

            print(f"\n与游戏 '{game_title}' (ID: {game_id}) 最相似的游戏 (基于标签):")
            for similar_id, score in similar_games:
                similar_title = df[df['app_id'] == similar_id]['title'].iloc[0]
                print(f"{similar_title} (ID: {similar_id}) - 相似度: {score:.4f}")

            # 可视化共同标签
            if similar_games:
                top_similar_id = similar_games[0][0]
                top_similar_title = df[df['app_id'] == top_similar_id]['title'].iloc[0]

                common_tags = game_tags[game_id] & game_tags[top_similar_id]
                only_target = game_tags[game_id] - game_tags[top_similar_id]
                only_similar = game_tags[top_similar_id] - game_tags[game_id]

                print(f"\n'{game_title}' 与 '{top_similar_title}' 的标签比较:")
                print(f"共同标签 ({len(common_tags)}): {', '.join(common_tags)}")
                print(f"仅 '{game_title}' 有的标签 ({len(only_target)}): {', '.join(only_target)}")
                print(f"仅 '{top_similar_title}' 有的标签 ({len(only_similar)}): {', '.join(only_similar)}")

                # 绘制Venn图
                plt.figure(figsize=(10, 6))
                from matplotlib_venn import venn2
                venn2([game_tags[game_id], game_tags[top_similar_id]],
                      set_labels=[game_title, top_similar_title])
                plt.title('标签重叠比较')
                plt.tight_layout()
                plt.savefig(f'game_{game_id}_tag_comparison.png')
                plt.close()

    # 基于用户行为的相似度（协同过滤）
    # 创建用户-游戏矩阵
    user_game_matrix = pd.pivot_table(
        df,
        values='is_recommended' if 'is_recommended' in df.columns else 'hours',
        index='user_id',
        columns='app_id',
        aggfunc='mean',
        fill_value=0
    )

    # 计算游戏间余弦相似度
    from sklearn.metrics.pairwise import cosine_similarity
    game_similarity = cosine_similarity(user_game_matrix.T)

    # 创建游戏索引映射
    game_idx = {game: i for i, game in enumerate(user_game_matrix.columns)}
    reverse_idx = {i: game for game, i in game_idx.items()}

    # 获取相似游戏
    if game_id in game_idx:
        idx = game_idx[game_id]
        similar_indices = game_similarity[idx].argsort()[::-1][1:n + 1]  # 跳过自己

        cf_similar_games = [(reverse_idx[i], game_similarity[idx][i]) for i in similar_indices]

        print(f"\n与游戏 '{game_title}' (ID: {game_id}) 最相似的游戏 (基于协同过滤):")
        for similar_id, score in cf_similar_games:
            similar_title = df[df['app_id'] == similar_id]['title'].iloc[0]
            print(f"{similar_title} (ID: {similar_id}) - 相似度: {score:.4f}")

    logger.info(f"游戏相似度分析完成")


def cold_start_analysis(data_path):
    """
    冷启动问题分析和处理

    参数:
        data_path (str): 数据文件路径
    """
    logger.info("分析冷启动问题...")

    # 读取数据
    df = pd.read_csv(data_path)

    # 初始化推荐系统
    recommender = SteamRecommender(data_path)
    recommender.load_data()
    recommender.engineer_features()

    # 训练模型
    recommender.train_lgbm_model()
    recommender.train_sequence_model()
    recommender.train_content_model()

    # 1. 新用户冷启动
    print("\n新用户冷启动问题分析:")
    new_user_recs = recommender.handle_cold_start_user(10)

    print("为新用户推荐的热门游戏:")
    for i, (game_id, score) in enumerate(new_user_recs, 1):
        game_title = df[df['app_id'] == game_id]['title'].iloc[0]
        print(f"{i}. {game_title} (得分: {score:.4f})")

    # 2. 新游戏冷启动
    # 模拟一个新游戏，从数据中随机选择一个游戏并删除其所有交互
    all_games = df['app_id'].unique()
    if len(all_games) > 1:
        # 选择一个游戏作为"新游戏"
        new_game_id = np.random.choice(all_games)
        new_game_title = df[df['app_id'] == new_game_id]['title'].iloc[0]

        print(f"\n新游戏冷启动分析 (模拟游戏: '{new_game_title}', ID: {new_game_id}):")

        # 创建一个没有该游戏交互数据的数据集
        df_without_game = df[df['app_id'] != new_game_id].copy()

        # 初始化一个新的推荐器，使用修改后的数据集
        cold_recommender = SteamRecommender(None)
        cold_recommender.df = df_without_game
        cold_recommender.train_df = cold_recommender.df.sample(frac=0.8, random_state=42)
        cold_recommender.test_df = cold_recommender.df.drop(cold_recommender.train_df.index)

        # 特征工程和模型训练
        cold_recommender.engineer_features()
        cold_recommender.train_content_model()

        # 从原始数据集获取新游戏的标签
        if 'tags' in df.columns:
            new_game_tags = df[df['app_id'] == new_game_id]['tags'].iloc[0]
            print(f"游戏标签: {new_game_tags}")

            # 根据标签相似度推荐
            print("\n基于内容的类似游戏推荐:")

            # 从所有游戏中找出标签最相似的
            game_tags = {}
            for _, row in df.drop_duplicates('app_id').iterrows():
                if pd.notna(row['tags']):
                    tags = set(tag.strip() for tag in row['tags'].split(','))
                    game_tags[row['app_id']] = tags

            # 计算标签相似度
            if new_game_id in game_tags:
                target_tags = game_tags[new_game_id]
                tag_similarities = []

                for other_id, other_tags in game_tags.items():
                    if other_id != new_game_id and other_tags:
                        jaccard = len(target_tags & other_tags) / len(target_tags | other_tags)
                        tag_similarities.append((other_id, jaccard))

                # 获取最相似的游戏
                similar_games = sorted(tag_similarities, key=lambda x: x[1], reverse=True)[:5]

                for i, (similar_id, score) in enumerate(similar_games, 1):
                    similar_title = df[df['app_id'] == similar_id]['title'].iloc[0]
                    print(f"{i}. {similar_title} (相似度: {score:.4f})")

    logger.info("冷启动问题分析完成")


def time_based_recommendation_analysis(data_path, user_id):
    """
    基于时间的推荐分析，研究用户品味随时间的变化

    参数:
        data_path (str): 数据文件路径
        user_id (int): 用户ID
    """
    logger.info(f"分析用户 {user_id} 的时间序列推荐...")

    # 读取数据
    df = pd.read_csv(data_path)

    # 检查用户是否存在
    if user_id not in df['user_id'].values:
        logger.error(f"用户ID {user_id} 不存在于数据集中")
        return

    # 确保日期列存在
    if 'date' not in df.columns:
        logger.error("缺少日期列，无法进行时间序列分析")
        return

    # 转换日期列
    df['date'] = pd.to_datetime(df['date'])

    # 获取用户数据，按时间排序
    user_data = df[df['user_id'] == user_id].sort_values('date')

    # 按季度分组
    user_data['quarter'] = user_data['date'].dt.to_period('Q')

    # 分析用户在不同时期的品味变化
    print(f"\n用户 {user_id} 随时间的游戏偏好变化:")

    for quarter, group in user_data.groupby('quarter'):
        if len(group) > 0:
            print(f"\n{quarter} 季度 ({len(group)} 条记录):")

            # 获取此季度玩的游戏
            games = []
            for _, row in group.iterrows():
                game_title = row['title']
                is_recommended = row['is_recommended'] if 'is_recommended' in row else 'N/A'
                hours = row['hours'] if 'hours' in row else 'N/A'
                games.append(f"{game_title} (推荐: {is_recommended}, 游戏时间: {hours}小时)")

            for game in games:
                print(f"- {game}")

            # 分析标签偏好
            if 'tags' in group.columns:
                all_tags = []
                for tags_str in group['tags'].dropna():
                    all_tags.extend([tag.strip() for tag in tags_str.split(',')])

                if all_tags:
                    tag_counts = pd.Series(all_tags).value_counts().head(5)
                    print(f"热门标签: {', '.join([f'{tag} ({count})' for tag, count in tag_counts.items()])}")

    # 时间衰减因子分析
    # 比较不同时间衰减因子对推荐结果的影响
    print("\n时间衰减因子对推荐的影响:")

    # 创建推荐系统实例并训练模型
    recommender = SteamRecommender(data_path)
    recommender.load_data()
    recommender.engineer_features()
    recommender.train_lgbm_model()
    recommender.train_sequence_model()

    # 测试不同的时间衰减因子
    decay_factors = [0.5, 0.7, 0.9, 0.99]
    decay_recommendations = {}

    # 原始衰减因子
    original_decay = recommender.config['time_decay_factor']

    for decay in decay_factors:
        # 设置衰减因子
        recommender.config['time_decay_factor'] = decay

        # 重新生成序列特征
        recommender.create_sequence_features()

        # 生成推荐
        recs = recommender.generate_recommendations(user_id, 5)
        decay_recommendations[decay] = recs

    # 恢复原始衰减因子
    recommender.config['time_decay_factor'] = original_decay

    # 比较不同衰减因子的推荐差异
    for decay, recs in decay_recommendations.items():
        print(f"\n衰减因子 {decay}:")
        for i, (game_id, score) in enumerate(recs, 1):
            game_title = df[df['app_id'] == game_id]['title'].iloc[0]
            print(f"{i}. {game_title} (得分: {score:.4f})")

    # 计算不同衰减因子之间的推荐重叠度
    print("\n不同衰减因子之间的推荐重叠度 (Jaccard相似度):")
    decay_list = sorted(decay_factors)
    for i in range(len(decay_list)):
        for j in range(i + 1, len(decay_list)):
            decay1 = decay_list[i]
            decay2 = decay_list[j]

            set1 = set([game_id for game_id, _ in decay_recommendations[decay1]])
            set2 = set([game_id for game_id, _ in decay_recommendations[decay2]])

            similarity = len(set1 & set2) / len(set1 | set2)
            print(f"衰减因子 {decay1} vs {decay2}: {similarity:.2f}")

    logger.info("时间序列推荐分析完成")


def optimize_recommender(data_path):
    """
    优化推荐器超参数

    参数:
        data_path (str): 数据文件路径
    """
    logger.info("开始优化推荐器超参数...")

    # 读取数据
    df = pd.read_csv(data_path)

    # 设定要测试的参数
    test_configs = [
        # 基线配置
        {
            'lgbm_params': {'learning_rate': 0.05, 'num_leaves': 31, 'max_depth': -1},
            'lgbm_weight': 0.4,
            'sequence_weight': 0.3,
            'content_weight': 0.3
        },
        # 偏向LightGBM
        {
            'lgbm_params': {'learning_rate': 0.05, 'num_leaves': 31, 'max_depth': -1},
            'lgbm_weight': 0.6,
            'sequence_weight': 0.2,
            'content_weight': 0.2
        },
        # 偏向序列模型
        {
            'lgbm_params': {'learning_rate': 0.05, 'num_leaves': 31, 'max_depth': -1},
            'lgbm_weight': 0.2,
            'sequence_weight': 0.6,
            'content_weight': 0.2
        },
        # 偏向内容模型
        {
            'lgbm_params': {'learning_rate': 0.05, 'num_leaves': 31, 'max_depth': -1},
            'lgbm_weight': 0.2,
            'sequence_weight': 0.2,
            'content_weight': 0.6
        },
        # 调整LightGBM参数1
        {
            'lgbm_params': {'learning_rate': 0.1, 'num_leaves': 63, 'max_depth': 10},
            'lgbm_weight': 0.4,
            'sequence_weight': 0.3,
            'content_weight': 0.3
        },
        # 调整LightGBM参数2
        {
            'lgbm_params': {'learning_rate': 0.01, 'num_leaves': 15, 'max_depth': 5},
            'lgbm_weight': 0.4,
            'sequence_weight': 0.3,
            'content_weight': 0.3
        }
    ]

    # 初始化结果存储
    results = []

    # 对每个配置进行评估
    for i, config in enumerate(test_configs):
        logger.info(f"测试配置 {i + 1}/{len(test_configs)}")

        # 创建推荐系统实例
        recommender = SteamRecommender(data_path, config)

        # 执行训练流程
        recommender.load_data()
        recommender.engineer_features()
        recommender.train_lgbm_model()
        recommender.train_sequence_model()
        recommender.train_content_model()

        # 评估性能
        metrics = recommender.evaluate_recommendations(k_values=[5, 10])

        # 存储结果
        config_name = f"配置{i + 1}"
        if i == 0:
            config_name += " (基线)"
        elif i == 1:
            config_name += " (偏向LightGBM)"
        elif i == 2:
            config_name += " (偏向序列模型)"
        elif i == 3:
            config_name += " (偏向内容模型)"
        elif i == 4:
            config_name += " (调整LightGBM-1)"
        elif i == 5:
            config_name += " (调整LightGBM-2)"

        results.append({
            'config_name': config_name,
            'config': config,
            'metrics': metrics
        })

    # 比较结果
    print("\n超参数优化结果比较:")
    for result in results:
        print(f"\n{result['config_name']}:")
        metrics = result['metrics']
        print(f"Precision@5: {metrics['precision'][5]:.4f}, Precision@10: {metrics['precision'][10]:.4f}")
        print(f"Recall@5: {metrics['recall'][5]:.4f}, Recall@10: {metrics['recall'][10]:.4f}")
        print(f"NDCG@5: {metrics['ndcg'][5]:.4f}, NDCG@10: {metrics['ndcg'][10]:.4f}")
        print(f"Diversity@5: {metrics['diversity'][5]:.4f}, Diversity@10: {metrics['diversity'][10]:.4f}")
        print(f"Coverage: {metrics['coverage']:.4f}")

    # 可视化比较
    # 准备可视化数据
    vis_data = []
    for result in results:
        config_name = result['config_name']
        metrics = result['metrics']

        for k in [5, 10]:
            for metric in ['precision', 'recall', 'ndcg', 'diversity']:
                vis_data.append({
                    'Config': config_name,
                    'Metric': f"{metric.capitalize()}@{k}",
                    'Value': metrics[metric][k]
                })

        vis_data.append({
            'Config': config_name,
            'Metric': 'Coverage',
            'Value': metrics['coverage']
        })

    vis_df = pd.DataFrame(vis_data)

    # 绘制性能比较图
    plt.figure(figsize=(15, 10))

    metrics_to_plot = ['Precision@5', 'Precision@10', 'Recall@5', 'Recall@10', 'NDCG@5', 'NDCG@10']
    for i, metric in enumerate(metrics_to_plot, 1):
        plt.subplot(2, 3, i)
        data = vis_df[vis_df['Metric'] == metric]
        sns.barplot(x='Config', y='Value', data=data)
        plt.title(metric)
        plt.xticks(rotation=45, ha='right')
        plt.ylim(0, data['Value'].max() * 1.2)

        # 在柱状图上显示数值
        for j, v in enumerate(data['Value']):
            plt.text(j, v + 0.01, f"{v:.3f}", ha='center')

    plt.tight_layout()
    plt.savefig('recommender_optimization_comparison.png')
    plt.close()

    # 找出最佳配置
    best_configs = {}
    for metric in ['precision', 'recall', 'ndcg', 'diversity']:
        for k in [5, 10]:
            metric_at_k = f"{metric.capitalize()}@{k}"
            best_idx = vis_df[vis_df['Metric'] == metric_at_k]['Value'].idxmax()
            best_config = vis_df.loc[best_idx, 'Config']
            best_configs[metric_at_k] = best_config

    best_coverage_idx = vis_df[vis_df['Metric'] == 'Coverage']['Value'].idxmax()
    best_configs['Coverage'] = vis_df.loc[best_coverage_idx, 'Config']

    print("\n各指标的最佳配置:")
    for metric, config in best_configs.items():
        print(f"{metric}: {config}")

    logger.info("超参数优化完成")
    return results


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description='Steam游戏推荐系统示例')
    parser.add_argument('--data', type=str, default='steam_data.csv', help='数据文件路径')
    parser.add_argument('--mode', type=str, default='train',
                        choices=['train', 'recommend', 'evaluate', 'analyze', 'optimize'],
                        help='运行模式')
    parser.add_argument('--user_id', type=int, help='用户ID (用于推荐或分析)')
    parser.add_argument('--game_id', type=int, help='游戏ID (用于游戏相似度分析)')
    parser.add_argument('--model_path', type=str, default='trained_model', help='模型保存/加载路径')
    parser.add_argument('--top_n', type=int, default=10, help='推荐数量')

    args = parser.parse_args()

    # 根据运行模式执行相应功能
    if args.mode == 'train':
        # 训练模式
        print("开始训练推荐系统...")
        recommender = run_training_pipeline(args.data)
        print("训练完成")

    elif args.mode == 'recommend':
        # 推荐模式
        if args.user_id is None:
            print("请提供用户ID")
            return

        print(f"为用户ID {args.user_id} 生成推荐...")
        recs = run_inference(args.model_path, args.user_id, args.top_n)

    elif args.mode == 'evaluate':
        # 评估模式
        print("开始评估推荐系统...")
        recommender = SteamRecommender(args.data)
        recommender.load_data()
        recommender.engineer_features()
        recommender.train_lgbm_model()
        recommender.train_sequence_model()
        recommender.train_content_model()

        evaluation_results = recommender.evaluate_recommendations()
        recommender.evaluation_results = evaluation_results
        recommender.visualize_results()
        print("评估完成")

    elif args.mode == 'analyze':
        # 分析模式
        if args.user_id is not None:
            # 用户行为分析
            print(f"分析用户ID {args.user_id}...")
            analyze_user_behavior(args.data, args.user_id)

            # 时间序列分析
            time_based_recommendation_analysis(args.data, args.user_id)

            # 推荐方法比较
            compare_recommendation_methods(args.data, args.user_id, args.top_n)

        elif args.game_id is not None:
            # 游戏相似度分析
            print(f"分析游戏ID {args.game_id}...")
            analyze_game_similarity(args.data, args.game_id, args.top_n)

        else:
            # 整体分析
            print("进行整体数据分析...")
            analyze_game_popularity(args.data, args.top_n)
            cold_start_analysis(args.data)

    elif args.mode == 'optimize':
        # 优化模式
        print("开始优化推荐系统超参数...")
        optimize_recommender(args.data)
        print("优化完成")


if __name__ == "__main__":
    main()