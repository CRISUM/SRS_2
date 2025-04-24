#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Steam游戏推荐系统 - 模型工具
作者: Claude
日期: 2025-04-24
描述: 提供Steam游戏推荐系统模型训练和评估的辅助函数
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score, precision_score, recall_score, ndcg_score
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import os
import pickle
import json

logger = logging.getLogger(__name__)


class SequenceDataset(Dataset):
    """游戏序列数据集类"""

    def __init__(self, sequences, targets):
        """
        初始化数据集

        参数:
            sequences (list): 游戏ID序列列表
            targets (list): 目标值列表
        """
        self.sequences = sequences
        self.targets = targets

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx], self.targets[idx]


class GameSequenceModel(nn.Module):
    """游戏序列模型，使用LSTM处理用户历史游戏序列"""

    def __init__(self, num_games, embedding_dim=64, hidden_dim=128, num_layers=2, dropout=0.2):
        """
        初始化序列模型

        参数:
            num_games (int): 游戏总数（嵌入层大小）
            embedding_dim (int): 嵌入维度
            hidden_dim (int): LSTM隐藏层维度
            num_layers (int): LSTM层数
            dropout (float): Dropout比例
        """
        super().__init__()
        self.game_embedding = nn.Embedding(num_games + 1, embedding_dim)  # +1用于填充和OOV
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, game_seq, seq_lengths):
        """
        前向传播

        参数:
            game_seq (tensor): 游戏ID序列
            seq_lengths (tensor): 序列长度

        返回:
            tensor: 预测分数
        """
        # 嵌入游戏ID
        embedded = self.game_embedding(game_seq)

        # 打包序列
        packed = nn.utils.rnn.pack_padded_sequence(
            embedded, seq_lengths.cpu(), batch_first=True, enforce_sorted=False
        )

        # LSTM处理
        _, (hidden, _) = self.lstm(packed)

        # 获取最后一层隐藏状态
        last_hidden = hidden[-1]

        # 线性层和sigmoid激活函数
        output = self.fc(self.dropout(last_hidden))
        return self.sigmoid(output).squeeze()


def train_sequence_model(train_dataloader, val_dataloader=None, num_games=None, config=None, device='cpu'):
    """
    训练序列模型

    参数:
        train_dataloader (DataLoader): 训练数据加载器
        val_dataloader (DataLoader): 验证数据加载器
        num_games (int): 游戏总数
        config (dict): 配置参数
        device (str): 设备（'cpu'或'cuda'）

    返回:
        GameSequenceModel: 训练好的模型
    """
    logger.info("开始训练序列模型...")

    # 默认配置
    if config is None:
        config = {
            'embedding_dim': 64,
            'hidden_dim': 128,
            'num_layers': 2,
            'dropout': 0.2,
            'learning_rate': 0.001,
            'epochs': 10
        }

    # 初始化模型
    model = GameSequenceModel(
        num_games=num_games,
        embedding_dim=config['embedding_dim'],
        hidden_dim=config['hidden_dim'],
        num_layers=config['num_layers'],
        dropout=config['dropout']
    ).to(device)

    # 定义损失函数和优化器
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])

    # 训练循环
    epochs = config['epochs']
    best_loss = float('inf')
    best_model = None

    for epoch in range(epochs):
        # 训练阶段
        model.train()
        train_loss = 0

        for sequences, seq_lengths, targets in train_dataloader:
            # 移动数据到设备
            sequences = sequences.to(device)
            seq_lengths = seq_lengths.to(device)
            targets = targets.to(device)

            # 前向传播
            optimizer.zero_grad()
            outputs = model(sequences, seq_lengths)

            # 计算损失
            loss = criterion(outputs, targets)

            # 反向传播
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_dataloader)

        # 验证阶段
        if val_dataloader:
            model.eval()
            val_loss = 0

            with torch.no_grad():
                for sequences, seq_lengths, targets in val_dataloader:
                    sequences = sequences.to(device)
                    seq_lengths = seq_lengths.to(device)
                    targets = targets.to(device)

                    outputs = model(sequences, seq_lengths)
                    loss = criterion(outputs, targets)
                    val_loss += loss.item()

            avg_val_loss = val_loss / len(val_dataloader)
            logger.info(f"Epoch {epoch + 1}/{epochs} - Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

            # 保存最佳模型
            if avg_val_loss < best_loss:
                best_loss = avg_val_loss
                best_model = model.state_dict().copy()
        else:
            logger.info(f"Epoch {epoch + 1}/{epochs} - Train Loss: {avg_train_loss:.4f}")

    # 加载最佳模型
    if best_model is not None:
        model.load_state_dict(best_model)

    logger.info("序列模型训练完成")
    return model


def collate_sequences(batch):
    """
    序列数据整理函数，用于DataLoader

    参数:
        batch: 批次数据

    返回:
        tuple: (padded_sequences, seq_lengths, targets)
    """
    # 提取序列和目标
    sequences, targets = zip(*batch)

    # 计算每个序列的长度
    seq_lengths = torch.tensor([len(seq) for seq in sequences])

    # 填充序列到相同长度
    max_len = max(seq_lengths).item()
    padded_sequences = torch.zeros(len(sequences), max_len, dtype=torch.long)

    for i, seq in enumerate(sequences):
        end = seq_lengths[i]
        padded_sequences[i, :end] = torch.tensor(seq[:end])

    # 转换目标为张量
    targets = torch.tensor(targets, dtype=torch.float32)

    return padded_sequences, seq_lengths, targets


def train_lightgbm_model(X_train, y_train, X_val=None, y_val=None, categorical_features=None, config=None):
    """
    训练LightGBM模型

    参数:
        X_train (DataFrame): 训练特征
        y_train (Series): 训练目标
        X_val (DataFrame): 验证特征
        y_val (Series): 验证目标
        categorical_features (list): 分类特征列表
        config (dict): 配置参数

    返回:
        lgb.Booster: 训练好的模型
    """
    logger.info("开始训练LightGBM模型...")

    # 默认配置
    if config is None:
        config = {
            'objective': 'binary',
            'metric': 'auc',
            'boosting_type': 'gbdt',
            'learning_rate': 0.05,
            'num_leaves': 31,
            'max_depth': -1,
            'min_child_samples': 20,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'n_estimators': 1000,
            'early_stopping_rounds': 50,
            'random_state': 42,
            'verbose': -1
        }

    # 转换数据格式
    train_data = lgb.Dataset(
        X_train,
        label=y_train,
        categorical_feature=categorical_features
    )

    valid_data = None
    if X_val is not None and y_val is not None:
        valid_data = lgb.Dataset(
            X_val,
            label=y_val,
            categorical_feature=categorical_features,
            reference=train_data
        )

    # 训练模型
    valid_sets = [train_data]
    valid_names = ['train']

    if valid_data:
        valid_sets.append(valid_data)
        valid_names.append('valid')

    model = lgb.train(
        config,
        train_data,
        valid_sets=valid_sets,
        valid_names=valid_names,
        early_stopping_rounds=config.get('early_stopping_rounds', 50),
        verbose_eval=50 if config.get('verbose', -1) >= 0 else False,
        feature_name='auto'
    )

    # 输出特征重要性
    feature_importance = pd.DataFrame({
        'Feature': X_train.columns,
        'Importance': model.feature_importance(importance_type='gain')
    }).sort_values(by='Importance', ascending=False)

    logger.info("前10个重要特征:")
    for idx, row in feature_importance.head(10).iterrows():
        logger.info(f"{row['Feature']}: {row['Importance']}")

    logger.info("LightGBM模型训练完成")
    return model, feature_importance


def evaluate_recommendations(model_func, test_users, test_df, df, k_values=[5, 10, 20]):
    """
    评估推荐系统性能

    参数:
        model_func (callable): 推荐函数，接收user_id和n，返回推荐列表
        test_users (array): 测试用户数组
        test_df (DataFrame): 测试数据集
        df (DataFrame): 完整数据集
        k_values (list): 评估的K值列表

    返回:
        dict: 评估结果
    """
    logger.info("开始评估推荐系统...")

    # 限制评估用户数以提高效率
    from tqdm import tqdm
    max_test_users = min(100, len(test_users))
    test_users = np.random.choice(test_users, max_test_users, replace=False)

    logger.info(f"使用 {len(test_users)} 个测试用户进行评估")

    # 初始化评估指标
    metrics = {
        'precision': {k: [] for k in k_values},
        'recall': {k: [] for k in k_values},
        'ndcg': {k: [] for k in k_values},
        'diversity': {k: [] for k in k_values},
        'coverage': []
    }

    # 所有推荐的游戏集合（用于计算覆盖率）
    all_recommended_games = set()
    all_games = set(df['app_id'].unique())

    # 评估每个测试用户
    for user_id in tqdm(test_users, desc="评估用户"):
        # 获取用户实际喜欢的游戏
        user_liked_games = set(test_df[
                                   (test_df['user_id'] == user_id) &
                                   (test_df['is_recommended'] == True)
                                   ]['app_id'].values)

        # 如果用户没有喜欢的游戏，跳过
        if not user_liked_games:
            continue

        # 生成推荐
        max_k = max(k_values)
        recommendations = model_func(user_id, max_k)
        recommended_games = [game_id for game_id, _ in recommendations]

        # 更新所有推荐的游戏集合
        all_recommended_games.update(recommended_games)

        # 计算每个K值的指标
        for k in k_values:
            top_k_games = recommended_games[:k]

            # 计算精确率
            hits = len(set(top_k_games) & user_liked_games)
            precision = hits / k if k > 0 else 0
            metrics['precision'][k].append(precision)

            # 计算召回率
            recall = hits / len(user_liked_games) if user_liked_games else 0
            metrics['recall'][k].append(recall)

            # 计算NDCG
            # 创建相关性数组（1表示相关，0表示不相关）
            relevance = [1 if game in user_liked_games else 0 for game in top_k_games]
            # 理想情况下的排序
            ideal_relevance = sorted(relevance, reverse=True)

            if sum(relevance) > 0:
                ndcg = ndcg_score([ideal_relevance], [relevance])
                metrics['ndcg'][k].append(ndcg)

            # 计算多样性
            # 使用游戏标签计算推荐列表中游戏的不同类型
            if 'tags' in df.columns:
                game_tags = {}
                for game_id in top_k_games:
                    tags = df[df['app_id'] == game_id]['tags'].iloc[0] if game_id in df['app_id'].values else ""
                    if pd.notna(tags):
                        game_tags[game_id] = set(tag.strip() for tag in tags.split(','))
                    else:
                        game_tags[game_id] = set()

                # 计算平均两两Jaccard距离
                if len(game_tags) >= 2:
                    diversity_scores = []
                    for i, (game1, tags1) in enumerate(game_tags.items()):
                        for game2, tags2 in list(game_tags.items())[i + 1:]:
                            if tags1 and tags2:  # 只有当两个游戏都有标签时
                                jaccard_similarity = len(tags1 & tags2) / len(tags1 | tags2)
                                jaccard_distance = 1 - jaccard_similarity
                                diversity_scores.append(jaccard_distance)

                    if diversity_scores:
                        diversity = sum(diversity_scores) / len(diversity_scores)
                        metrics['diversity'][k].append(diversity)

    # 计算覆盖率
    coverage = len(all_recommended_games) / len(all_games) if all_games else 0
    metrics['coverage'] = coverage

    # 计算平均指标
    results = {}
    for metric in ['precision', 'recall', 'ndcg', 'diversity']:
        results[metric] = {k: np.mean(metrics[metric][k]) for k in k_values}
    results['coverage'] = metrics['coverage']

    # 打印结果
    logger.info("评估结果:")
    for metric in ['precision', 'recall', 'ndcg', 'diversity']:
        logger.info(f"{metric.capitalize()}:")
        for k in k_values:
            logger.info(f"  @{k}: {results[metric][k]:.4f}")
    logger.info(f"Coverage: {results['coverage']:.4f}")

    return results


def visualize_results(evaluation_results, feature_importance=None):
    """
    可视化评估结果和特征重要性

    参数:
        evaluation_results (dict): 评估结果
        feature_importance (DataFrame): 特征重要性
    """
    plt.figure(figsize=(15, 10))

    # 绘制评估指标
    metrics = ['precision', 'recall', 'ndcg', 'diversity']
    k_values = sorted(evaluation_results['precision'].keys())

    for i, metric in enumerate(metrics, 1):
        plt.subplot(2, 2, i)
        values = [evaluation_results[metric][k] for k in k_values]
        plt.plot(k_values, values, marker='o')
        plt.title(f'{metric.capitalize()} at different k')
        plt.xlabel('k')
        plt.ylabel(metric)
        plt.grid(True)

    plt.tight_layout()
    plt.savefig('evaluation_metrics.png')
    plt.close()

    # 绘制特征重要性
    if feature_importance is not None:
        plt.figure(figsize=(12, 8))
        top_features = feature_importance.head(20)
        sns.barplot(x='Importance', y='Feature', data=top_features)
        plt.title('Top 20 Features by Importance')
        plt.tight_layout()
        plt.savefig('feature_importance.png')
        plt.close()

    logger.info("结果可视化完成")


def save_model_components(lgbm_model=None, sequence_model=None, encoders=None, scaler=None,
                          content_similarity=None, config=None, path='models'):
    """
    保存模型和相关组件

    参数:
        lgbm_model: LightGBM模型
        sequence_model: 序列模型
        encoders: 标签编码器
        scaler: 特征缩放器
        content_similarity: 内容相似度矩阵
        config: 配置参数
        path: 保存路径
    """
    logger.info(f"保存模型到 {path}...")

    # 创建保存目录
    os.makedirs(path, exist_ok=True)

    # 保存LightGBM模型
    if lgbm_model is not None:
        lgbm_model.save_model(os.path.join(path, 'lgbm_model.txt'))

    # 保存序列模型
    if sequence_model is not None:
        torch.save(sequence_model.state_dict(), os.path.join(path, 'sequence_model.pt'))

    # 保存编码器
    if encoders is not None:
        with open(os.path.join(path, 'encoders.pkl'), 'wb') as f:
            pickle.dump(encoders, f)

    # 保存缩放器
    if scaler is not None:
        with open(os.path.join(path, 'scaler.pkl'), 'wb') as f:
            pickle.dump(scaler, f)

    # 保存内容相似度
    if content_similarity is not None:
        with open(os.path.join(path, 'content_similarity.pkl'), 'wb') as f:
            pickle.dump(content_similarity, f)

    # 保存配置
    if config is not None:
        with open(os.path.join(path, 'config.json'), 'w') as f:
            json.dump(config, f, indent=4)

    logger.info(f"模型保存完成")


def load_model_components(path='models'):
    """
    加载模型和相关组件

    参数:
        path: 加载路径

    返回:
        dict: 加载的模型和组件
    """
    logger.info(f"从 {path} 加载模型...")

    result = {}

    # 检查目录
    if not os.path.exists(path):
        logger.error(f"模型目录 {path} 不存在")
        return result

    # 加载LightGBM模型
    lgbm_path = os.path.join(path, 'lgbm_model.txt')
    if os.path.exists(lgbm_path):
        result['lgbm_model'] = lgb.Booster(model_file=lgbm_path)

    # 加载编码器
    encoders_path = os.path.join(path, 'encoders.pkl')
    if os.path.exists(encoders_path):
        with open(encoders_path, 'rb') as f:
            result['encoders'] = pickle.load(f)

    # 加载缩放器
    scaler_path = os.path.join(path, 'scaler.pkl')
    if os.path.exists(scaler_path):
        with open(scaler_path, 'rb') as f:
            result['scaler'] = pickle.load(f)

    # 加载内容相似度
    similarity_path = os.path.join(path, 'content_similarity.pkl')
    if os.path.exists(similarity_path):
        with open(similarity_path, 'rb') as f:
            result['content_similarity'] = pickle.load(f)

    # 加载配置
    config_path = os.path.join(path, 'config.json')
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            result['config'] = json.load(f)

    # 序列模型需要重建架构后再加载权重
    sequence_path = os.path.join(path, 'sequence_model.pt')
    if os.path.exists(sequence_path):
        logger.info("序列模型权重存在，但需要手动重建模型架构后再加载")
        result['sequence_model_path'] = sequence_path

    logger.info(f"模型加载完成, 加载了 {len(result)} 个组件")
    return result


def rebuild_sequence_model(weights_path, num_games, config=None):
    """
    重建序列模型并加载权重

    参数:
        weights_path: 权重文件路径
        num_games: 游戏数量
        config: 模型配置

    返回:
        GameSequenceModel: 重建的模型
    """
    if config is None:
        config = {
            'embedding_dim': 64,
            'hidden_dim': 128,
            'num_layers': 2,
            'dropout': 0.2
        }

    # 创建模型
    model = GameSequenceModel(
        num_games=num_games,
        embedding_dim=config.get('embedding_dim', 64),
        hidden_dim=config.get('hidden_dim', 128),
        num_layers=config.get('num_layers', 2),
        dropout=config.get('dropout', 0.2)
    )

    # 加载权重
    model.load_state_dict(torch.load(weights_path, map_location=torch.device('cpu')))
    model.eval()

    return model


def get_popular_games(df, n=10):
    """
    获取最流行的游戏

    参数:
        df: 数据集
        n: 返回数量

    返回:
        list: (game_id, score) 元组列表
    """
    # 计算游戏流行度
    game_popularity = df.groupby('app_id').agg({
        'user_id': 'count',  # 评论数
        'rating': 'mean' if 'rating' in df.columns else None  # 平均评分
    }).reset_index()

    # 移除None列
    game_popularity = game_popularity.dropna(axis=1)

    # 重命名列
    game_popularity.columns = ['app_id'] + [col if col == 'app_id' else col for col in game_popularity.columns if
                                            col != 'app_id']

    # 规范化
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()

    if 'rating' in game_popularity.columns:
        norm_cols = ['user_id', 'rating']
        game_popularity[norm_cols] = scaler.fit_transform(game_popularity[norm_cols])

        # 计算综合得分
        game_popularity['popularity_score'] = game_popularity['user_id'] * 0.7 + game_popularity['rating'] * 0.3
    else:
        game_popularity['user_id'] = scaler.fit_transform(game_popularity[['user_id']])
        game_popularity['popularity_score'] = game_popularity['user_id']

    # 排序并获取前N个
    popular_games = game_popularity.sort_values('popularity_score', ascending=False).head(n)

    # 返回(game_id, score)元组列表
    return [(row['app_id'], row['popularity_score']) for _, row in popular_games.iterrows()]


def optimize_model_hyperparameters(X_train, y_train, X_val, y_val, param_grid=None):
    """
    优化LightGBM模型超参数

    参数:
        X_train: 训练特征
        y_train: 训练标签
        X_val: 验证特征
        y_val: 验证标签
        param_grid: 参数网格

    返回:
        tuple: (best_params, best_score)
    """
    from sklearn.model_selection import GridSearchCV
    import lightgbm as lgb

    logger.info("开始超参数优化...")

    if param_grid is None:
        param_grid = {
            'learning_rate': [0.01, 0.05, 0.1],
            'num_leaves': [31, 63, 127],
            'max_depth': [-1, 5, 10],
            'min_child_samples': [10, 20, 50],
            'subsample': [0.7, 0.8, 0.9],
            'colsample_bytree': [0.7, 0.8, 0.9]
        }

    # 创建模型
    model = lgb.LGBMClassifier(
        objective='binary',
        metric='auc',
        boosting_type='gbdt',
        random_state=42,
        n_jobs=-1,
        verbose=-1
    )

    # 网格搜索
    grid_search = GridSearchCV(
        model,
        param_grid,
        cv=3,
        scoring='roc_auc',
        verbose=1,
        n_jobs=-1
    )

    grid_search.fit(X_train, y_train, eval_set=[(X_val, y_val)], early_stopping_rounds=50, verbose=False)

    logger.info(f"最佳参数: {grid_search.best_params_}")
    logger.info(f"最佳得分: {grid_search.best_score_:.4f}")

    return grid_search.best_params_, grid_search.best_score_


def compare_models(models, X_test, y_test, model_names=None):
    """
    比较多个模型的性能

    参数:
        models: 模型列表
        X_test: 测试特征
        y_test: 测试标签
        model_names: 模型名称列表

    返回:
        DataFrame: 比较结果
    """
    if model_names is None:
        model_names = [f"Model {i + 1}" for i in range(len(models))]

    results = []

    for model, name in zip(models, model_names):
        # 预测
        y_pred_proba = model.predict(X_test)
        y_pred = (y_pred_proba > 0.5).astype(int)

        # 计算指标
        auc = roc_auc_score(y_test, y_pred_proba)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)

        results.append({
            'Model': name,
            'AUC': auc,
            'Precision': precision,
            'Recall': recall
        })

    # 创建DataFrame
    results_df = pd.DataFrame(results)

    # 可视化
    plt.figure(figsize=(10, 6))

    metrics = ['AUC', 'Precision', 'Recall']
    for i, metric in enumerate(metrics, 1):
        plt.subplot(1, 3, i)
        sns.barplot(x='Model', y=metric, data=results_df)
        plt.title(metric)
        plt.ylim(0, 1)
        plt.xticks(rotation=45)

    plt.tight_layout()
    plt.savefig('model_comparison.png')
    plt.close()

    return results_df


if __name__ == "__main__":
    # 设置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # 简单测试
    try:
        # 创建测试数据
        import numpy as np

        print("测试序列模型...")
        num_games = 1000
        embedding_dim = 32

        # 创建序列模型
        model = GameSequenceModel(num_games, embedding_dim=embedding_dim)
        print(f"序列模型创建成功: {model}")

        # 测试前向传播
        batch_size = 4
        seq_len = 10
        sequences = torch.randint(0, num_games, (batch_size, seq_len))
        seq_lengths = torch.tensor([seq_len] * batch_size)

        outputs = model(sequences, seq_lengths)
        print(f"模型输出形状: {outputs.shape}")
        print("测试完成!")

    except Exception as e:
        logger.error(f"测试错误: {str(e)}")