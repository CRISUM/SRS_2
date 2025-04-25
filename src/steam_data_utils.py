#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Steam游戏推荐系统 - 数据处理工具
作者: Claude
日期: 2025-04-24
描述: 提供Steam游戏数据处理和特征工程的辅助函数
"""

import pandas as pd
import numpy as np
from datetime import datetime
import re
from collections import Counter
import logging

logger = logging.getLogger(__name__)


def load_and_preprocess_data(file_path):
    """
    加载并预处理Steam游戏数据

    参数:
        file_path (str): 数据文件路径

    返回:
        DataFrame: 预处理后的数据
    """
    logger.info(f"加载数据: {file_path}")

    # 读取CSV文件
    df = pd.read_csv(file_path)

    # 基本数据清洗
    df = clean_data(df)

    # 特征转换
    df = transform_features(df)

    logger.info(f"数据预处理完成，最终形状: {df.shape}")
    return df


def clean_data(df):
    """
    清洗数据，处理缺失值、异常值等

    参数:
        df (DataFrame): 原始数据

    返回:
        DataFrame: 清洗后的数据
    """
    logger.info("开始数据清洗...")

    # 复制数据，避免修改原始数据
    df = df.copy()

    # 处理布尔型列
    bool_columns = ['is_recommended', 'win', 'mac', 'linux', 'steam_deck']
    for col in bool_columns:
        if col in df.columns:
            df[col] = df[col].map({'True': True, 'False': False, True: True, False: False})

    # 处理日期列
    date_columns = ['date', 'date_release']
    for col in date_columns:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col])

    # 填充缺失值
    df['hours'] = df['hours'].fillna(0)
    if 'is_recommended' in df.columns:
        df['is_recommended'] = df['is_recommended'].fillna(False)

    # 移除重复记录
    df_size_before = len(df)
    df = df.drop_duplicates()
    df_size_after = len(df)
    if df_size_before > df_size_after:
        logger.info(f"移除了 {df_size_before - df_size_after} 条重复记录")

    # 处理异常值
    if 'hours' in df.columns:
        # 剔除游戏时间异常的记录（例如超过10000小时）
        hours_threshold = 10000
        outliers = df[df['hours'] > hours_threshold]
        if len(outliers) > 0:
            logger.info(f"发现 {len(outliers)} 条游戏时间异常记录 (>{hours_threshold}小时)")
            df = df[df['hours'] <= hours_threshold]

    # 处理文本中的特殊字符
    text_columns = ['title', 'description', 'tags', 'review']
    for col in text_columns:
        if col in df.columns:
            df[col] = df[col].astype(str).apply(lambda x: clean_text(x))

    logger.info(f"数据清洗完成，剩余记录数: {len(df)}")
    return df


def clean_text(text):
    """
    清洗文本，移除特殊字符等

    参数:
        text (str): 原始文本

    返回:
        str: 清洗后的文本
    """
    if pd.isna(text):
        return ""

    # 替换HTML标签
    text = re.sub(r'<.*?>', '', text)

    # 移除多余空格
    text = re.sub(r'\s+', ' ', text).strip()

    return text


def transform_features(df):
    """
    转换和创建特征

    参数:
        df (DataFrame): 清洗后的数据

    返回:
        DataFrame: 添加新特征后的数据
    """
    logger.info("开始特征转换...")

    # 复制数据，避免修改原始数据
    df = df.copy()

    # 添加基本时间特征
    if 'date' in df.columns:
        df['day_of_week'] = df['date'].dt.dayofweek
        df['month'] = df['date'].dt.month
        df['year'] = df['date'].dt.year

        if 'date_release' in df.columns:
            df['days_since_release'] = (df['date'] - df['date_release']).dt.days
            df['days_since_release'] = df['days_since_release'].clip(lower=0)

    # 创建评分特征
    if 'rating_new' in df.columns:
        df['rating'] = df['rating_new']
    elif 'is_recommended' in df.columns:
        df['rating'] = df['is_recommended'].astype(int) * 10

    # 处理游戏价格
    if 'price_final' in df.columns and 'price_original' in df.columns:
        # 计算折扣率
        if 'discount' not in df.columns:
            df['discount'] = 1 - (df['price_final'] / df['price_original'].replace(0, 1))
            df['discount'] = df['discount'].clip(lower=0, upper=1)

        # 价格分类
        price_bins = [0, 5, 10, 20, 30, 50, float('inf')]
        price_labels = ['免费/特低', '低价', '中低价', '中价', '中高价', '高价']
        df['price_category'] = pd.cut(df['price_final'], bins=price_bins, labels=price_labels)

    # 处理游戏时长
    if 'hours' in df.columns:
        # 游戏时长的非线性变换
        df['hours_log'] = np.log1p(df['hours'])
        df['hours_sqrt'] = np.sqrt(df['hours'])

        # 时长分类
        hour_bins = [0, 1, 5, 10, 20, 50, 100, float('inf')]
        hour_labels = ['未玩', '尝鲜', '简单体验', '一般通关', '深度体验', '充分探索', '重度沉迷']
        df['hours_category'] = pd.cut(df['hours'], bins=hour_bins, labels=hour_labels)

    # 处理标签
    if 'tags' in df.columns:
        # 统计标签频率
        all_tags = []
        for tags_str in df['tags'].dropna():
            all_tags.extend([tag.strip() for tag in tags_str.split(',')])

        tag_counts = Counter(all_tags)

        # 提取前N个最常见标签
        top_n = 100
        common_tags = [tag for tag, _ in tag_counts.most_common(top_n)]

        logger.info(f"提取了 {len(common_tags)} 个最常见标签")

    logger.info("特征转换完成")
    return df


def split_train_test(df, test_size=0.2, by_time=True):
    """
    划分训练集和测试集

    参数:
        df (DataFrame): 数据集
        test_size (float): 测试集比例
        by_time (bool): 是否按时间划分

    返回:
        tuple: (train_df, test_df)
    """
    from sklearn.model_selection import train_test_split

    if by_time and 'date' in df.columns:
        # 按时间划分
        df = df.sort_values('date')
        split_idx = int(len(df) * (1 - test_size))
        train_df = df.iloc[:split_idx].copy()
        test_df = df.iloc[split_idx:].copy()
        logger.info(f"按时间划分数据，训练集截止日期: {train_df['date'].max()}, 测试集起始日期: {test_df['date'].min()}")
    else:
        # 随机划分
        stratify_col = df['is_recommended'] if 'is_recommended' in df.columns else None
        train_df, test_df = train_test_split(
            df, test_size=test_size, random_state=42, stratify=stratify_col
        )
        logger.info(f"随机划分数据，训练集: {len(train_df)}条, 测试集: {len(test_df)}条")

    return train_df, test_df


def analyze_user_behavior(df, user_id=None):
    """
    分析用户行为特征

    参数:
        df (DataFrame): 数据集
        user_id (int, optional): 特定用户ID，如果为None则分析所有用户

    返回:
        DataFrame: 用户特征
    """
    if user_id:
        user_df = df[df['user_id'] == user_id]
        if len(user_df) == 0:
            logger.warning(f"未找到用户ID: {user_id}")
            return None

        logger.info(f"分析用户ID: {user_id}, 记录数: {len(user_df)}")
    else:
        user_df = df
        logger.info(f"分析所有用户行为, 用户数: {user_df['user_id'].nunique()}")

    # 按用户分组
    grouped = user_df.groupby('user_id')

    # 创建用户特征
    user_features = pd.DataFrame({
        'user_id': user_df['user_id'].unique(),
        'game_count': grouped['app_id'].nunique(),
        'total_hours': grouped['hours'].sum(),
        'avg_hours': grouped['hours'].mean(),
        'max_hours': grouped['hours'].max(),
        'min_hours': grouped['hours'].min()
    })

    if 'is_recommended' in user_df.columns:
        user_features['recommend_ratio'] = grouped['is_recommended'].mean()

    if 'date' in user_df.columns:
        user_dates = grouped['date'].agg(['min', 'max'])
        user_features['first_activity'] = user_dates['min']
        user_features['last_activity'] = user_dates['max']
        user_features['activity_days'] = (user_dates['max'] - user_dates['min']).dt.days

    # 分析偏好标签
    if 'tags' in user_df.columns:
        user_tags = {}
        for user, group in grouped:
            all_tags = []
            for tags_str in group['tags'].dropna():
                all_tags.extend([tag.strip() for tag in tags_str.split(',')])

            if all_tags:
                tag_counter = Counter(all_tags)
                top_tags = [tag for tag, _ in tag_counter.most_common(3)]
                user_tags[user] = top_tags

        for i in range(min(3, max([len(tags) for tags in user_tags.values()], default=0))):
            col_name = f'top_tag_{i + 1}'
            user_features[col_name] = user_features['user_id'].map(
                lambda uid: user_tags.get(uid, [''])[i] if uid in user_tags and i < len(user_tags[uid]) else ''
            )

    return user_features


def analyze_game_popularity(df):
    """
    分析游戏流行度

    参数:
        df (DataFrame): 数据集

    返回:
        DataFrame: 游戏流行度特征
    """
    logger.info("分析游戏流行度...")

    # 按游戏分组
    grouped = df.groupby('app_id')

    # 创建基本统计特征
    game_features = pd.DataFrame({
        'app_id': df['app_id'].unique(),
        'review_count': grouped.size(),
        'unique_users': grouped['user_id'].nunique(),
        'avg_hours': grouped['hours'].mean(),
        'total_hours': grouped['hours'].sum()
    })

    if 'is_recommended' in df.columns:
        game_features['recommend_ratio'] = grouped['is_recommended'].mean()

    if 'rating' in df.columns:
        game_features['avg_rating'] = grouped['rating'].mean()

    # 合并游戏标题
    if 'title' in df.columns:
        game_titles = df[['app_id', 'title']].drop_duplicates('app_id')
        game_features = game_features.merge(game_titles, on='app_id', how='left')

    # 计算流行度分数
    if 'recommend_ratio' in game_features.columns:
        # 归一化特征
        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler()
        features_to_scale = ['review_count', 'unique_users', 'total_hours', 'recommend_ratio']
        game_features[features_to_scale] = scaler.fit_transform(game_features[features_to_scale])

        # 计算加权得分
        game_features['popularity_score'] = (
                game_features['review_count'] * 0.3 +
                game_features['unique_users'] * 0.3 +
                game_features['total_hours'] * 0.2 +
                game_features['recommend_ratio'] * 0.2
        )
    elif 'avg_rating' in game_features.columns:
        # 如果没有推荐率，使用评分
        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler()
        features_to_scale = ['review_count', 'unique_users', 'total_hours', 'avg_rating']
        game_features[features_to_scale] = scaler.fit_transform(game_features[features_to_scale])

        game_features['popularity_score'] = (
                game_features['review_count'] * 0.3 +
                game_features['unique_users'] * 0.3 +
                game_features['total_hours'] * 0.2 +
                game_features['avg_rating'] * 0.2
        )
    else:
        # 如果既没有推荐率也没有评分
        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler()
        features_to_scale = ['review_count', 'unique_users', 'total_hours']
        game_features[features_to_scale] = scaler.fit_transform(game_features[features_to_scale])

        game_features['popularity_score'] = (
                game_features['review_count'] * 0.4 +
                game_features['unique_users'] * 0.4 +
                game_features['total_hours'] * 0.2
        )

    # 排序
    game_features = game_features.sort_values('popularity_score', ascending=False)

    logger.info(f"游戏流行度分析完成，共有 {len(game_features)} 款游戏")
    return game_features


def extract_game_content_features(df):
    """
    提取游戏内容特征，包括标签和描述

    参数:
        df (DataFrame): 数据集

    返回:
        DataFrame: 游戏内容特征
    """
    logger.info("提取游戏内容特征...")

    # 获取唯一游戏及其标签/描述
    game_data = df[['app_id', 'title']].drop_duplicates('app_id')

    # 添加标签
    if 'tags' in df.columns:
        game_tags = df.groupby('app_id')['tags'].first().reset_index()
        game_data = game_data.merge(game_tags, on='app_id', how='left')

        # 处理标签，创建标签集合
        game_data['tag_list'] = game_data['tags'].apply(
            lambda x: set([tag.strip() for tag in str(x).split(',')]) if pd.notna(x) else set()
        )

    # 添加描述
    if 'description' in df.columns:
        game_desc = df.groupby('app_id')['description'].first().reset_index()
        game_data = game_data.merge(game_desc, on='app_id', how='left')

    # 使用TF-IDF处理标签
    if 'tags' in df.columns:
        from sklearn.feature_extraction.text import TfidfVectorizer

        # 准备标签文本
        game_data['tags_text'] = game_data['tags'].fillna('').apply(
            lambda x: ' '.join([tag.strip() for tag in str(x).split(',')])
        )

        # 应用TF-IDF
        tfidf = TfidfVectorizer(max_features=100)
        tag_features = tfidf.fit_transform(game_data['tags_text'])

        # 创建特征名称
        feature_names = [f'tag_{i}' for i in range(tag_features.shape[1])]

        # 转换为DataFrame
        tag_df = pd.DataFrame(tag_features.toarray(), columns=feature_names)
        tag_df['app_id'] = game_data['app_id'].values

        # 合并回原始数据
        game_data = game_data.merge(tag_df, on='app_id', how='left')

        logger.info(f"创建了 {len(feature_names)} 个标签特征")

    # 使用TF-IDF处理描述文本
    if 'description' in df.columns:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.decomposition import TruncatedSVD

        # 准备描述文本
        game_data['description'] = game_data['description'].fillna('')

        # 应用TF-IDF
        tfidf = TfidfVectorizer(max_features=1000, stop_words='english')
        desc_features = tfidf.fit_transform(game_data['description'])

        # 使用SVD降维
        svd = TruncatedSVD(n_components=20)
        desc_svd = svd.fit_transform(desc_features)

        # 创建特征名称
        feature_names = [f'desc_{i}' for i in range(desc_svd.shape[1])]

        # 转换为DataFrame
        desc_df = pd.DataFrame(desc_svd, columns=feature_names)
        desc_df['app_id'] = game_data['app_id'].values

        # 合并回原始数据
        game_data = game_data.merge(desc_df, on='app_id', how='left')

        logger.info(f"创建了 {len(feature_names)} 个描述文本特征")

    logger.info(f"游戏内容特征提取完成，共有 {len(game_data)} 款游戏")
    return game_data


def calculate_game_similarity(game_data):
    """
    计算游戏间的相似度矩阵

    参数:
        game_data (DataFrame): 包含游戏特征的数据集

    返回:
        dict: 包含相似度矩阵和映射的字典
    """
    logger.info("计算游戏相似度矩阵...")

    # 提取特征列
    feature_cols = [col for col in game_data.columns
                    if col.startswith(('tag_', 'desc_')) and col != 'tag_list']

    if not feature_cols:
        logger.warning("没有找到合适的特征列，无法计算相似度")
        return None

    # 计算余弦相似度
    from sklearn.metrics.pairwise import cosine_similarity

    # 构建特征矩阵
    feature_matrix = game_data[feature_cols].values

    # 计算相似度
    similarity_matrix = cosine_similarity(feature_matrix)

    # 创建游戏ID到索引的映射
    game_idx = {game_id: idx for idx, game_id in enumerate(game_data['app_id'])}

    result = {
        'matrix': similarity_matrix,
        'game_idx': game_idx,
        'titles': dict(zip(game_data['app_id'], game_data['title']))
    }

    logger.info(f"相似度矩阵计算完成，大小: {similarity_matrix.shape}")
    return result


def find_similar_games(similarity_data, game_id, n=10):
    """
    找出与指定游戏最相似的游戏

    参数:
        similarity_data (dict): 相似度矩阵数据
        game_id (int): 目标游戏ID
        n (int): 返回相似游戏数量

    返回:
        list: 相似游戏列表，每个元素为(game_id, title, similarity_score)
    """
    if similarity_data is None:
        logger.error("相似度数据为空")
        return []

    if game_id not in similarity_data['game_idx']:
        logger.error(f"游戏ID {game_id} 不在相似度矩阵中")
        return []

    # 获取游戏索引
    idx = similarity_data['game_idx'][game_id]

    # 获取相似度分数
    similarities = similarity_data['matrix'][idx]

    # 找出最相似的游戏（排除自己）
    similar_indices = similarities.argsort()[::-1][1:n + 1]

    # 转换回游戏ID和标题
    reverse_idx = {idx: game_id for game_id, idx in similarity_data['game_idx'].items()}
    titles = similarity_data['titles']

    similar_games = []
    for idx in similar_indices:
        similar_id = reverse_idx[idx]
        similar_title = titles.get(similar_id, f"Unknown Game {similar_id}")
        similar_score = similarities[idx]
        similar_games.append((similar_id, similar_title, similar_score))

    return similar_games


def create_train_test_files(df, train_output='train_data.csv', test_output='test_data.csv', test_size=0.2):
    """
    创建训练集和测试集文件

    参数:
        df (DataFrame): 数据集
        train_output (str): 训练集输出文件路径
        test_output (str): 测试集输出文件路径
        test_size (float): 测试集比例

    返回:
        tuple: (train_df, test_df)
    """
    train_df, test_df = split_train_test(df, test_size=test_size)

    train_df.to_csv(train_output, index=False)
    test_df.to_csv(test_output, index=False)

    logger.info(f"训练集已保存到 {train_output}")
    logger.info(f"测试集已保存到 {test_output}")

    return train_df, test_df


if __name__ == "__main__":
    # 设置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # 测试功能
    data_file = "steam_top_100000.csv"
    try:
        df = load_and_preprocess_data(data_file)
        print(f"数据加载成功，形状: {df.shape}")

        # 分析游戏流行度
        popular_games = analyze_game_popularity(df)
        print("\n最流行的10款游戏:")
        for _, row in popular_games.head(10).iterrows():
            print(f"{row['title']} (ID: {row['app_id']}, 得分: {row['popularity_score']:.4f})")

        # 提取内容特征
        game_features = extract_game_content_features(df)

        # 计算相似度
        similarity_data = calculate_game_similarity(game_features)

        # 找出相似游戏
        if len(df['app_id'].unique()) > 0:
            example_game = df['app_id'].iloc[0]
            example_title = df[df['app_id'] == example_game]['title'].iloc[0]

            print(f"\n与游戏 '{example_title}' 最相似的游戏:")
            similar_games = find_similar_games(similarity_data, example_game, n=5)

            for game_id, title, score in similar_games:
                print(f"{title} (ID: {game_id}, 相似度: {score:.4f})")

    except Exception as e:
        logger.error(f"运行错误: {str(e)}")