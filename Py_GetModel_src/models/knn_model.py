#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Py_GetModel_src/models/knn_model.py - KNN-based collaborative filtering models
Author: YourName
Date: 2025-04-27
Description: Implements user-based and item-based KNN collaborative filtering
"""
import traceback

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
import logging
import pickle
import os

from .base_model import (BaseRecommenderModel)

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

    # 修改 knn_model.py 中的 fit 方法，使用稀疏矩阵直接拟合
    def fit(self, data):
        """Train the model with provided data with balanced speed-accuracy tradeoff"""
        logger.info(f"Training {self.type}-based KNN model with balanced optimization...")

        if isinstance(data, pd.DataFrame):
            original_size = len(data)

            # 1. 适度采样 - 不要过度缩减数据
            if len(data) > 400000:  # 更大的阈值
                logger.info(f"Large dataset ({len(data)} rows), sampling to 400,000 rows")
                data = data.sample(n=400000, random_state=42)

            # 2. 更温和的过滤 - 保留更多的有价值数据
            item_counts = data['app_id'].value_counts()
            user_counts = data['user_id'].value_counts()

            # 保留至少有2次交互的物品（更少的过滤）
            popular_items = item_counts[item_counts >= 2].index
            # 更大的物品集
            if len(popular_items) > 8000:  # 增加上限
                popular_items = item_counts.nlargest(8000).index

            # 对用户更少的限制
            if self.type == 'user':
                # 对于用户KNN，不过滤用户，但可能限制物品
                filtered_data = data[data['app_id'].isin(popular_items)]
            else:
                # 对于物品KNN，保留更多的用户数据
                active_users = user_counts[user_counts >= 1].index  # 更少的过滤
                if len(active_users) > 20000:  # 增加上限
                    active_users = user_counts.nlargest(20000).index

                filtered_data = data[data['app_id'].isin(popular_items) & data['user_id'].isin(active_users)]

            logger.info(f"Filtered from {original_size} to {len(filtered_data)} rows (more moderate filtering)")
            logger.info(
                f"Working with {filtered_data['user_id'].nunique()} users and {filtered_data['app_id'].nunique()} items")

            # 使用过滤后的数据
            data = filtered_data

            # 创建用户和物品的索引映射
            user_ids = data['user_id'].astype('category')
            item_ids = data['app_id'].astype('category')

            # 保存索引映射
            self.user_indices = {user: i for i, user in enumerate(user_ids.cat.categories)}
            self.item_indices = {item: i for i, item in enumerate(item_ids.cat.categories)}

            # 反向索引映射
            self.reversed_user_indices = {i: user for user, i in self.user_indices.items()}
            self.reversed_item_indices = {i: item for item, i in self.item_indices.items()}

            # 3. 保留原始评分尺度 - 不二值化
            if 'is_recommended' in data.columns:
                # 使用更细粒度的评分
                ratings = data['is_recommended'].astype(int) * 5  # 0或5，而不是0或1
            else:
                # 保留更多小时数信息
                ratings = data['hours'].fillna(0).clip(0, 100)  # 裁剪极端值但保留连续尺度

            # 创建稀疏矩阵
            row = user_ids.cat.codes
            col = item_ids.cat.codes
            self.sparse_matrix = csr_matrix((ratings, (row, col)),
                                            shape=(len(self.user_indices), len(self.item_indices)))

            # 保存为 user_item_matrix 以兼容现有代码
            self.user_item_matrix = pd.DataFrame.sparse.from_spmatrix(
                self.sparse_matrix,
                index=user_ids.cat.categories,
                columns=item_ids.cat.categories)

            logger.info(
                f"Sparse matrix shape: {self.sparse_matrix.shape}, density: {self.sparse_matrix.nnz / (self.sparse_matrix.shape[0] * self.sparse_matrix.shape[1]):.6f}")

            # 4. 根据数据量动态调整邻居数
            max_dimension = max(self.sparse_matrix.shape)
            if max_dimension > 10000:
                dynamic_neighbors = 20  # 较大数据集用较少邻居
            else:
                dynamic_neighbors = min(30, self.n_neighbors)  # 较小数据集用较多邻居

            if self.n_neighbors > dynamic_neighbors:
                logger.info(f"Adjusting n_neighbors from {self.n_neighbors} to {dynamic_neighbors} based on data size")
                self.n_neighbors = dynamic_neighbors

            # 5. 根据数据大小选择算法
            if self.metric == 'jaccard' and max_dimension > 10000:
                logger.info(f"Large dataset with dimension {max_dimension}, switching to optimized computation...")

                # 导入必要的库
                from joblib import Parallel, delayed
                import numpy as np
                import os

                # 决定使用哪种计算模式
                if max_dimension > 20000:
                    # 5.1 对于极大规模的数据，使用余弦相似度+批处理
                    logger.info("Dataset too large for Jaccard, using optimized cosine similarity")
                    self.model = NearestNeighbors(
                        n_neighbors=self.n_neighbors,
                        metric='cosine',
                        algorithm='auto',
                        n_jobs=-1
                    )

                    if self.type == 'user':
                        self.model.fit(self.sparse_matrix)
                    else:
                        self.model.fit(self.sparse_matrix.T)

                else:
                    # 5.2 对于中等规模的数据，使用优化的Jaccard计算
                    logger.info("Using optimized Jaccard computation")

                    # 定义Jaccard计算函数
                    def compute_jaccard_row(i, sparse_matrix, indices):
                        """计算一行的Jaccard相似度"""
                        results = []
                        if self.type == 'user':
                            i_vector = sparse_matrix[i].toarray().ravel() > 0
                        else:
                            i_vector = sparse_matrix[:, i].toarray().ravel() > 0

                        for j in indices:
                            if i == j:
                                results.append(0)  # 同一项相似度为0
                                continue

                            if self.type == 'user':
                                j_vector = sparse_matrix[j].toarray().ravel() > 0
                            else:
                                j_vector = sparse_matrix[:, j].toarray().ravel() > 0

                            # 计算Jaccard相似度
                            intersection = np.sum(i_vector & j_vector)
                            union = np.sum(i_vector | j_vector)
                            similarity = intersection / union if union > 0 else 0

                            results.append(similarity)

                        return results

                    # 确定样本数量和批次
                    if self.type == 'user':
                        n_samples = self.sparse_matrix.shape[0]
                    else:
                        n_samples = self.sparse_matrix.shape[1]

                    # 限制样本数量，但保持较大的集合
                    max_samples = 5000  # 更大的样本集
                    if n_samples > max_samples:
                        indices = np.random.choice(n_samples, max_samples, replace=False)
                        sim_indices = indices
                    else:
                        indices = np.arange(n_samples)
                        sim_indices = indices

                    # 计算相似度矩阵
                    logger.info(f"Computing Jaccard similarity for {len(indices)} samples...")

                    # 使用多进程计算，但限制核心数避免过载
                    n_jobs = min(8, os.cpu_count() or 1)
                    logger.info(f"Using {n_jobs} parallel workers")

                    # 分批计算以减少内存使用
                    batch_size = 100
                    similarity_matrix = np.zeros((len(indices), len(indices)))

                    for start in range(0, len(indices), batch_size):
                        end = min(start + batch_size, len(indices))
                        batch_indices = indices[start:end]
                        logger.info(
                            f"Processing batch {start // batch_size + 1}/{(len(indices) + batch_size - 1) // batch_size}")

                        # 并行计算这一批的相似度
                        batch_similarities = Parallel(n_jobs=n_jobs, verbose=1)(
                            delayed(compute_jaccard_row)(i, self.sparse_matrix, sim_indices)
                            for i in batch_indices
                        )

                        # 更新相似度矩阵
                        for i, similarities in zip(range(start, end), batch_similarities):
                            similarity_matrix[i - start] = similarities

                        # 手动垃圾回收
                        import gc
                        gc.collect()

                    # 创建并训练KNN模型
                    self.model = NearestNeighbors(
                        n_neighbors=min(self.n_neighbors, len(indices) - 1),
                        metric='precomputed',
                        algorithm='brute'
                    )
                    self.model.fit(1 - similarity_matrix)  # 距离 = 1 - 相似度

                    # 保存采样索引，用于推荐阶段
                    self.sampled_indices = sim_indices

            else:
                # 6. 标准方法训练KNN模型
                logger.info(f"Using standard method with {self.metric} metric")
                self.model = NearestNeighbors(
                    n_neighbors=self.n_neighbors,
                    metric=self.metric,
                    algorithm='auto',
                    n_jobs=-1
                )

                if self.type == 'user':
                    self.model.fit(self.sparse_matrix)
                else:
                    self.model.fit(self.sparse_matrix.T)

            logger.info(f"{self.type}-based KNN model trained successfully")
            return self

    def predict(self, user_id, item_id):
        """Predict rating for a user-item pair

        Args:
            user_id: User ID
            item_id: Item ID

        Returns:
            float: Predicted rating (0-1 scale)
        """
        if self.model is None:
            logger.warning("Model not trained yet")
            return 0.5

        # Check if user and item are in the model
        if user_id not in self.user_indices or item_id not in self.item_indices:
            return 0.5  # Default score for cold-start cases

        try:
            user_idx = self.user_indices[user_id]
            item_idx = self.item_indices[item_id]

            if self.type == 'user':
                # User-based prediction
                # Get user vector
                user_vector = self.sparse_matrix[user_idx].toarray().reshape(1, -1)

                # Find most similar users
                distances, indices = self.model.kneighbors(user_vector, n_neighbors=min(10, self.model.n_neighbors))

                # Calculate weighted score from similar users
                similar_users = indices[0]
                similarities = 1 - distances[0]  # Convert distance to similarity

                # Filter out the user itself
                if user_idx in similar_users:
                    user_idx_pos = np.where(similar_users == user_idx)[0][0]
                    similar_users = np.delete(similar_users, user_idx_pos)
                    similarities = np.delete(similarities, user_idx_pos)

                if len(similar_users) == 0:
                    return 0.5

                # Get similar users' ratings for the target item
                ratings = []
                for i, similar_user_idx in enumerate(similar_users):
                    rating = self.sparse_matrix[similar_user_idx, item_idx]
                    if rating > 0:  # Only consider non-zero ratings
                        ratings.append((rating, similarities[i]))

                if not ratings:
                    return 0.5

                # Weighted average of ratings
                weighted_sum = sum(rating * similarity for rating, similarity in ratings)
                similarity_sum = sum(similarity for _, similarity in ratings)

                if similarity_sum == 0:
                    return 0.5

                predicted_rating = weighted_sum / similarity_sum

                # Normalize to 0-1 range
                max_rating = 10  # Assuming ratings are on a 0-10 scale
                normalized_rating = predicted_rating / max_rating

                return normalized_rating

            else:
                # Item-based prediction
                # Get item vector
                item_vector = self.sparse_matrix[:, item_idx].T.toarray().reshape(1, -1)

                # Find most similar items
                distances, indices = self.model.kneighbors(item_vector, n_neighbors=min(10, self.model.n_neighbors))

                # Calculate weighted score from similar items
                similar_items = indices[0]
                similarities = 1 - distances[0]  # Convert distance to similarity

                # Filter out the item itself
                if item_idx in similar_items:
                    item_idx_pos = np.where(similar_items == item_idx)[0][0]
                    similar_items = np.delete(similar_items, item_idx_pos)
                    similarities = np.delete(similarities, item_idx_pos)

                if len(similar_items) == 0:
                    return 0.5

                # Get user's ratings for similar items
                ratings = []
                for i, similar_item_idx in enumerate(similar_items):
                    rating = self.sparse_matrix[user_idx, similar_item_idx]
                    if rating > 0:  # Only consider non-zero ratings
                        ratings.append((rating, similarities[i]))

                if not ratings:
                    return 0.5

                # Weighted average of ratings
                weighted_sum = sum(rating * similarity for rating, similarity in ratings)
                similarity_sum = sum(similarity for _, similarity in ratings)

                if similarity_sum == 0:
                    return 0.5

                predicted_rating = weighted_sum / similarity_sum

                # Normalize to 0-1 range
                max_rating = 10  # Assuming ratings are on a 0-10 scale
                normalized_rating = predicted_rating / max_rating

                return normalized_rating

        except Exception as e:
            logger.error(f"Error in KNN prediction: {str(e)}")
            return 0.5

    def recommend(self, user_id, n=10):
        """Generate recommendations handling sampled indices if necessary"""
        if self.model is None:
            logger.warning("Model not trained yet")
            return []

        # Check if user exists in the model
        if user_id not in self.user_indices:
            # logger.warning(f"User {user_id} not found in the model")
            return []

        try:
            user_idx = self.user_indices[user_id]

            # 如果我们使用了采样，需要特殊处理
            if hasattr(self, 'sampled_indices'):
                # 检查用户是否在采样中（对于用户KNN）
                if self.type == 'user':
                    if user_idx not in self.sampled_indices:
                        logger.debug(f"User {user_id} not in sampled indices, using fallback")
                        return []  # 或使用备选推荐方法

                    # 映射到采样索引
                    sampled_idx = np.where(self.sampled_indices == user_idx)[0][0]

                    # 获取相似用户
                    distances, indices = self.model.kneighbors(
                        np.array([sampled_idx]).reshape(1, -1),
                        n_neighbors=min(self.n_neighbors, len(self.sampled_indices) - 1)
                    )

                    # 映射回原始索引
                    similar_users = [self.sampled_indices[idx] for idx in indices[0]]
                    similarities = 1 - distances[0]  # 相似度 = 1 - 距离
                else:
                    # 正常流程，但使用采样索引
                    similar_items = []
                    for item_id, item_idx in self.item_indices.items():
                        if item_id in self.user_history.get(user_id, []):
                            continue  # 跳过用户已有物品

                        # 检查物品是否在采样中
                        if item_idx not in self.sampled_indices:
                            continue

                        # 计算得分
                        score = self.predict(user_id, item_id)
                        similar_items.append((item_id, score))

                    # 按分数排序
                    similar_items.sort(key=lambda x: x[1], reverse=True)
                    return similar_items[:n]
            else:
                # 标准KNN推荐流程
                # Get user's already seen items
                user_vector = self.sparse_matrix[user_idx].toarray().ravel()
                seen_indices = np.where(user_vector > 0)[0]
                seen_items = {self.reversed_item_indices[idx] for idx in seen_indices}

                # 根据KNN类型使用不同策略
                if self.type == 'user':
                    # User-based recommendation
                    user_vector = user_vector.reshape(1, -1)

                    # Find most similar users
                    distances, indices = self.model.kneighbors(
                        user_vector,
                        n_neighbors=min(self.n_neighbors, self.sparse_matrix.shape[0] - 1)
                    )

                    # Calculate weighted score from similar users
                    similar_users = indices[0]
                    similarities = 1 - distances[0]  # Convert distance to similarity

                    # Filter out the user itself
                    if user_idx in similar_users:
                        user_idx_pos = np.where(similar_users == user_idx)[0][0]
                        similar_users = np.delete(similar_users, user_idx_pos)
                        similarities = np.delete(similarities, user_idx_pos)

                    if len(similar_users) == 0:
                        return []

                    # Calculate predicted scores for all items
                    item_scores = {}

                    for item_id, item_idx in self.item_indices.items():
                        # Skip already seen items
                        if item_id in seen_items:
                            continue

                        # Get similar users' ratings for this item
                        ratings = []
                        for i, similar_user_idx in enumerate(similar_users):
                            rating = self.sparse_matrix[similar_user_idx, item_idx]
                            if rating > 0:  # Only consider non-zero ratings
                                ratings.append((rating, similarities[i]))

                        if not ratings:
                            continue

                        # Weighted average of ratings
                        weighted_sum = sum(rating * similarity for rating, similarity in ratings)
                        similarity_sum = sum(similarity for _, similarity in ratings)

                        if similarity_sum == 0:
                            continue

                        predicted_rating = weighted_sum / similarity_sum

                        # Normalize and store
                        max_rating = 10  # Assuming ratings are on a 0-10 scale
                        item_scores[item_id] = predicted_rating / max_rating

                else:
                    # Item-based recommendation
                    # Calculate predicted scores for all items
                    item_scores = {}

                    for item_id, item_idx in self.item_indices.items():
                        # Skip already seen items
                        if item_id in seen_items:
                            continue

                        # Get item vector
                        item_vector = self.sparse_matrix[:, item_idx].T.toarray().reshape(1, -1)

                        # Find most similar items that the user has rated
                        distances, indices = self.model.kneighbors(
                            item_vector,
                            n_neighbors=min(self.n_neighbors, self.sparse_matrix.shape[1] - 1)
                        )

                        similar_items = indices[0]
                        similarities = 1 - distances[0]  # Convert distance to similarity

                        # Filter out the item itself
                        if item_idx in similar_items:
                            item_idx_pos = np.where(similar_items == item_idx)[0][0]
                            similar_items = np.delete(similar_items, item_idx_pos)
                            similarities = np.delete(similarities, item_idx_pos)

                        if len(similar_items) == 0:
                            continue

                        # Get user's ratings for similar items
                        ratings = []
                        for i, similar_item_idx in enumerate(similar_items):
                            rating = self.sparse_matrix[user_idx, similar_item_idx]
                            if rating > 0:  # Only consider non-zero ratings
                                ratings.append((rating, similarities[i]))

                        if not ratings:
                            continue

                        # Weighted average of ratings
                        weighted_sum = sum(rating * similarity for rating, similarity in ratings)
                        similarity_sum = sum(similarity for _, similarity in ratings)

                        if similarity_sum == 0:
                            continue

                        predicted_rating = weighted_sum / similarity_sum

                        # Normalize and store
                        max_rating = 10  # Assuming ratings are on a 0-10 scale
                        item_scores[item_id] = predicted_rating / max_rating

                # Sort by score in descending order
                sorted_items = sorted(item_scores.items(), key=lambda x: x[1], reverse=True)

                # Return top N
                return sorted_items[:n]

        except Exception as e:
            logger.error(f"Error in KNN recommendation: {str(e)}")
            logger.error(traceback.format_exc())
            return []

    def update(self, new_data):
        """Update model with new data (incremental learning)

        Args:
            new_data (DataFrame): New interactions data

        Returns:
            self: Updated model
        """
        logger.info(f"Updating {self.type}-based KNN model...")

        if self.user_item_matrix is None:
            # If model isn't trained yet, just do a full training
            return self.fit(new_data)

        # Determine rating column
        if 'rating' in new_data.columns:
            rating_col = 'rating'
        elif 'is_recommended' in new_data.columns:
            # Convert boolean to numeric value
            new_data['rating_value'] = new_data['is_recommended'].astype(int) * 10
            rating_col = 'rating_value'
        else:
            # Use hours as interaction value
            rating_col = 'hours'

        # Process new users and items
        new_users = set(new_data['user_id']) - set(self.user_indices.keys())
        new_items = set(new_data['app_id']) - set(self.item_indices.keys())

        # If there are new users or items, we need to expand the matrix
        if new_users or new_items:
            # Create updated index and columns
            updated_index = list(self.user_item_matrix.index) + list(new_users)
            updated_columns = list(self.user_item_matrix.columns) + list(new_items)

            # Create expanded matrix
            expanded_matrix = pd.DataFrame(
                0,
                index=updated_index,
                columns=updated_columns
            )

            # Fill in existing values
            expanded_matrix.loc[
                self.user_item_matrix.index, self.user_item_matrix.columns] = self.user_item_matrix.values

            # Update user-item matrix
            self.user_item_matrix = expanded_matrix

            # Update mappings
            self.user_indices = {user: i for i, user in enumerate(self.user_item_matrix.index)}
            self.item_indices = {item: i for i, item in enumerate(self.user_item_matrix.columns)}
            self.reversed_user_indices = {i: user for user, i in self.user_indices.items()}
            self.reversed_item_indices = {i: item for item, i in self.item_indices.items()}

        # Update matrix with new ratings
        for _, row in new_data.iterrows():
            user_id = row['user_id']
            item_id = row['app_id']
            rating = row[rating_col]

            # Update rating in user-item matrix
            self.user_item_matrix.loc[user_id, item_id] = rating

        # Convert to sparse matrix
        self.sparse_matrix = csr_matrix(self.user_item_matrix.values)

        # Retrain the model
        if self.type == 'user':
            # User-based KNN
            n_neighbors = min(self.n_neighbors, len(self.user_indices))
            self.model = NearestNeighbors(
                n_neighbors=n_neighbors,
                metric=self.metric,
                algorithm=self.algorithm,
                n_jobs=-1
            )
            self.model.fit(self.sparse_matrix)
        else:
            # Item-based KNN
            n_neighbors = min(self.n_neighbors, len(self.item_indices))
            self.model = NearestNeighbors(
                n_neighbors=n_neighbors,
                metric=self.metric,
                algorithm=self.algorithm,
                n_jobs=-1
            )
            self.model.fit(self.sparse_matrix.T)  # Transpose for item similarity

        logger.info(f"{self.type}-based KNN model updated successfully")
        return self

    def save(self, path):
        """Save model to disk

        Args:
            path (str): Directory path

        Returns:
            bool: Success
        """
        logger.info(f"Saving {self.type}-based KNN model to {path}")

        os.makedirs(path, exist_ok=True)

        try:
            # Save model data
            model_data = {
                'type': self.type,
                'n_neighbors': self.n_neighbors,
                'metric': self.metric,
                'algorithm': self.algorithm,
                'user_indices': self.user_indices,
                'item_indices': self.item_indices,
                'reversed_user_indices': self.reversed_user_indices,
                'reversed_item_indices': self.reversed_item_indices
            }

            with open(os.path.join(path, 'knn_model_metadata.pkl'), 'wb') as f:
                pickle.dump(model_data, f)

            # Save user-item matrix
            if self.user_item_matrix is not None:
                self.user_item_matrix.to_pickle(os.path.join(path, 'user_item_matrix.pkl'))

            # Save sparse matrix
            if self.sparse_matrix is not None:
                with open(os.path.join(path, 'sparse_matrix.pkl'), 'wb') as f:
                    pickle.dump(self.sparse_matrix, f)

            logger.info(f"{self.type}-based KNN model saved successfully")
            return True
        except Exception as e:
            logger.error(f"Error saving KNN model: {str(e)}")
            return False

    def load(self, path):
        """Load model from disk

        Args:
            path (str): Directory path

        Returns:
            self: Loaded model
        """
        logger.info(f"Loading KNN model from {path}")

        try:
            # Load model metadata
            with open(os.path.join(path, 'knn_model_metadata.pkl'), 'rb') as f:
                model_data = pickle.load(f)

            self.type = model_data['type']
            self.n_neighbors = model_data['n_neighbors']
            self.metric = model_data['metric']
            self.algorithm = model_data['algorithm']
            self.user_indices = model_data['user_indices']
            self.item_indices = model_data['item_indices']
            self.reversed_user_indices = model_data['reversed_user_indices']
            self.reversed_item_indices = model_data['reversed_item_indices']

            # Load user-item matrix
            user_item_matrix_path = os.path.join(path, 'user_item_matrix.pkl')
            if os.path.exists(user_item_matrix_path):
                self.user_item_matrix = pd.read_pickle(user_item_matrix_path)

            # Load sparse matrix
            sparse_matrix_path = os.path.join(path, 'sparse_matrix.pkl')
            if os.path.exists(sparse_matrix_path):
                with open(sparse_matrix_path, 'rb') as f:
                    self.sparse_matrix = pickle.load(f)

            # Recreate the model
            if self.sparse_matrix is not None:
                if self.type == 'user':
                    n_neighbors = min(self.n_neighbors, self.sparse_matrix.shape[0])
                    self.model = NearestNeighbors(
                        n_neighbors=n_neighbors,
                        metric=self.metric,
                        algorithm=self.algorithm,
                        n_jobs=-1
                    )
                    self.model.fit(self.sparse_matrix)
                else:
                    n_neighbors = min(self.n_neighbors, self.sparse_matrix.shape[1])
                    self.model = NearestNeighbors(
                        n_neighbors=n_neighbors,
                        metric=self.metric,
                        algorithm=self.algorithm,
                        n_jobs=-1
                    )
                    self.model.fit(self.sparse_matrix.T)

            logger.info("KNN model loaded successfully")
            return self
        except Exception as e:
            logger.error(f"Error loading KNN model: {str(e)}")
            return None

