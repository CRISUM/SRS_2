#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
app.py - Modified Flask API for Steam Game Recommender using CSV data
"""

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os
import json
import logging
import pandas as pd
import numpy as np
import time
from werkzeug.security import generate_password_hash, check_password_hash
from flask_jwt_extended import JWTManager, create_access_token, jwt_required, get_jwt_identity
import uuid

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 初始化Flask应用
app = Flask(__name__, static_folder='../frontend/build')
CORS(app)  # 启用跨域请求

# 配置JWT
app.config['JWT_SECRET_KEY'] = os.environ.get('JWT_SECRET_KEY', 'your-secret-key')
app.config['JWT_ACCESS_TOKEN_EXPIRES'] = 86400  # 令牌24小时过期
jwt = JWTManager(app)

# 数据文件路径
DATA_DIR = os.environ.get('DATA_DIR', '../SRS_2/data/origin')
GAMES_CSV = os.path.join(DATA_DIR, 'games.csv')
GAMES_METADATA = os.path.join(DATA_DIR, 'games_metadata.json')
RECOMMENDATIONS_CSV = os.path.join(DATA_DIR, 'recommendations.csv')
USERS_CSV = os.path.join(DATA_DIR, 'users.csv')

# 全局数据
games_df = None
games_metadata = {}
recommendations_df = None
users_df = None

# 用户数据 (在实际应用中应使用数据库)
users_db = {
    "demo": {
        "id": "12345-demo-user",
        "password_hash": generate_password_hash("password"),
        "created_at": time.time()
    }
}
user_preferences = {}
recommendation_cache = {}

# 预设用户偏好
DEFAULT_USER_PREFERENCES = {
    "12345-demo-user": {
        "liked_games": [],
        "disliked_games": [],
        "played_games": []
    }
}


def load_data():
    """加载CSV和JSON数据"""
    global games_df, games_metadata, recommendations_df, users_df, user_preferences

    try:
        # 加载游戏CSV
        logger.info(f"Loading games data from {GAMES_CSV}")
        games_df = pd.read_csv(GAMES_CSV)
        logger.info(f"Loaded {len(games_df)} games")

        # 加载游戏元数据JSON
        logger.info(f"Loading games metadata from {GAMES_METADATA}")
        with open(GAMES_METADATA, 'r') as f:
            for line in f:
                try:
                    game_data = json.loads(line.strip())
                    games_metadata[game_data['app_id']] = game_data
                except json.JSONDecodeError:
                    continue
        logger.info(f"Loaded metadata for {len(games_metadata)} games")

        # 加载推荐CSV
        logger.info(f"Loading recommendations data from {RECOMMENDATIONS_CSV}")
        recommendations_df = pd.read_csv(RECOMMENDATIONS_CSV)
        logger.info(f"Loaded {len(recommendations_df)} recommendations")

        # 加载用户CSV
        logger.info(f"Loading users data from {USERS_CSV}")
        users_df = pd.read_csv(USERS_CSV)
        logger.info(f"Loaded {len(users_df)} users")

        # 为CSV中的用户创建账户
        for _, row in users_df.iterrows():
            user_id = str(row['user_id'])
            users_db[user_id] = {
                "id": user_id,
                "password_hash": generate_password_hash("password"),  # 所有用户使用相同密码
                "created_at": time.time()
            }

        # 初始化用户偏好
        user_preferences.update(DEFAULT_USER_PREFERENCES)

        # 从推荐中提取喜好
        for _, row in recommendations_df.iterrows():
            user_id = str(row['user_id'])
            app_id = row['app_id']
            is_recommended = row['is_recommended']

            if user_id not in user_preferences:
                user_preferences[user_id] = {
                    "liked_games": [],
                    "disliked_games": [],
                    "played_games": []
                }

            # 添加到已玩游戏
            if app_id not in user_preferences[user_id]["played_games"]:
                user_preferences[user_id]["played_games"].append(app_id)

            # 添加到喜欢/不喜欢
            if is_recommended and app_id not in user_preferences[user_id]["liked_games"]:
                user_preferences[user_id]["liked_games"].append(app_id)
            elif not is_recommended and app_id not in user_preferences[user_id]["disliked_games"]:
                user_preferences[user_id]["disliked_games"].append(app_id)

        return True

    except Exception as e:
        logger.error(f"Error loading data: {str(e)}")
        return False


def get_game_info(game_id):
    """获取游戏信息"""
    try:
        game_id = int(game_id)
    except ValueError:
        return None

    # 从DataFrame中查找
    if games_df is not None:
        game_data = games_df[games_df['app_id'] == game_id]

        if len(game_data) > 0:
            game_info = {
                'id': int(game_id),
                'title': game_data['title'].iloc[0],
                'tags': []
            }

            # 添加标签
            if game_id in games_metadata and 'tags' in games_metadata[game_id]:
                game_info['tags'] = games_metadata[game_id]['tags']

            # 添加描述
            if game_id in games_metadata and 'description' in games_metadata[game_id]:
                game_info['description'] = games_metadata[game_id]['description']

            # 从CSV添加其他可用信息
            for col in ['date_release', 'win', 'mac', 'linux', 'rating', 'positive_ratio', 'price_final',
                        'price_original']:
                if col in game_data.columns and pd.notna(game_data[col].iloc[0]):
                    game_info[col] = game_data[col].iloc[0]

            return game_info

    # 如果找不到游戏，返回None
    return None


def get_popular_games(n=10):
    """获取流行游戏"""
    if recommendations_df is not None:
        # 计算每个游戏的推荐数
        game_counts = recommendations_df['app_id'].value_counts()

        # 计算正面评价率
        game_ratings = {}
        for game_id in game_counts.index:
            game_data = recommendations_df[recommendations_df['app_id'] == game_id]
            pos_ratio = game_data['is_recommended'].mean()
            game_ratings[game_id] = pos_ratio

        # 综合得分 = 0.7*流行度 + 0.3*好评率
        popular_items = []
        total_reviews = len(recommendations_df)

        for game_id, count in game_counts.items():
            pop_score = count / total_reviews
            rating_score = game_ratings.get(game_id, 0.5)
            final_score = (pop_score * 0.7) + (rating_score * 0.3)
            popular_items.append((game_id, final_score))

        # 排序并返回前N个
        popular_items.sort(key=lambda x: x[1], reverse=True)
        return popular_items[:n]

    # 如果没有推荐数据，返回前N个游戏
    elif games_df is not None:
        return [(row['app_id'], 0.5) for _, row in games_df.head(n).iterrows()]

    return []


def get_similar_games(game_id, n=5):
    """获取相似游戏

    基于标签相似度的简单实现
    """
    try:
        game_id = int(game_id)
    except ValueError:
        return []

    if game_id not in games_metadata or 'tags' not in games_metadata[game_id]:
        return []

    target_tags = set(games_metadata[game_id]['tags'])
    if not target_tags:
        return []

    # 计算与其他游戏的标签相似度
    similarities = []

    for other_id, metadata in games_metadata.items():
        if other_id == game_id or 'tags' not in metadata:
            continue

        other_tags = set(metadata['tags'])
        if not other_tags:
            continue

        # 计算Jaccard相似度
        intersection = len(target_tags.intersection(other_tags))
        union = len(target_tags.union(other_tags))

        if union > 0:
            similarity = intersection / union
            similarities.append((other_id, similarity))

    # 排序并返回前N个
    similarities.sort(key=lambda x: x[1], reverse=True)
    return similarities[:n]


def generate_recommendations(user_id, n=10):
    """生成游戏推荐

    简单实现：基于用户喜好和标签相似度
    """
    if user_id not in user_preferences:
        return get_popular_games(n)

    user_prefs = user_preferences[user_id]
    liked_games = user_prefs.get('liked_games', [])

    if not liked_games:
        return get_popular_games(n)

    # 从用户喜欢的游戏中收集标签
    user_tags = {}
    for game_id in liked_games:
        if game_id in games_metadata and 'tags' in games_metadata[game_id]:
            for tag in games_metadata[game_id]['tags']:
                user_tags[tag] = user_tags.get(tag, 0) + 1

    # 如果没有标签信息，返回热门游戏
    if not user_tags:
        return get_popular_games(n)

    # 为每个游戏计算标签匹配分数
    game_scores = {}
    played_games = set(user_prefs.get('played_games', []))

    for game_id, metadata in games_metadata.items():
        # 跳过已经玩过的游戏
        if game_id in played_games:
            continue

        if 'tags' not in metadata or not metadata['tags']:
            continue

        # 计算标签匹配分数
        score = 0
        for tag in metadata['tags']:
            if tag in user_tags:
                score += user_tags[tag]

        # 标准化分数
        game_scores[game_id] = score / len(metadata['tags'])

    # 排序并返回前N个
    recommendations = [(game_id, score) for game_id, score in game_scores.items()]
    recommendations.sort(key=lambda x: x[1], reverse=True)

    # 如果推荐不足N个，添加热门游戏
    if len(recommendations) < n:
        popular = get_popular_games(n)
        popular_ids = [p[0] for p in popular]

        for game_id, score in popular:
            if game_id not in [r[0] for r in recommendations] and game_id not in played_games:
                recommendations.append((game_id, score * 0.8))  # 降低权重

                if len(recommendations) >= n:
                    break

    # 再次排序并返回前N个
    recommendations.sort(key=lambda x: x[1], reverse=True)
    return recommendations[:n]


# API端点
@app.route('/api/health', methods=['GET'])
def health_check():
    """健康检查API"""
    if games_df is not None:
        return jsonify({'status': 'ok', 'message': 'API service is running'})
    else:
        return jsonify({'status': 'error', 'message': 'Data not loaded'}), 500


@app.route('/api/register', methods=['POST'])
def register():
    """用户注册API"""
    data = request.get_json()
    username = data.get('username')
    password = data.get('password')

    if not username or not password:
        return jsonify({'status': 'error', 'message': 'Username and password are required'}), 400

    if username in users_db:
        return jsonify({'status': 'error', 'message': 'Username already exists'}), 409

    # 生成用户ID并存储用户信息
    user_id = str(uuid.uuid4())
    users_db[username] = {
        'id': user_id,
        'password_hash': generate_password_hash(password),
        'created_at': time.time()
    }

    # 创建用户偏好
    user_preferences[user_id] = {
        'liked_games': [],
        'disliked_games': [],
        'played_games': []
    }

    # 生成令牌
    access_token = create_access_token(identity=user_id)

    return jsonify({
        'status': 'success',
        'message': 'User registered successfully',
        'token': access_token,
        'user': {
            'id': user_id,
            'username': username
        }
    })


@app.route('/api/login', methods=['POST'])
def login():
    """用户登录API"""
    data = request.get_json()
    username = data.get('username')
    password = data.get('password')

    if not username or not password:
        return jsonify({'status': 'error', 'message': 'Username and password are required'}), 400

    # 预设demo用户自动登录
    if username == "demo" and password == "password":
        user = users_db["demo"]
        access_token = create_access_token(identity=user['id'])
        return jsonify({
            'status': 'success',
            'message': 'Login successful',
            'token': access_token,
            'user': {
                'id': user['id'],
                'username': 'demo'
            }
        })

    user = users_db.get(username)
    if not user or not check_password_hash(user['password_hash'], password):
        return jsonify({'status': 'error', 'message': 'Invalid credentials'}), 401

    # 生成令牌
    access_token = create_access_token(identity=user['id'])

    return jsonify({
        'status': 'success',
        'message': 'Login successful',
        'token': access_token,
        'user': {
            'id': user['id'],
            'username': username
        }
    })


@app.route('/api/recommendations', methods=['GET'])
@jwt_required()
def get_recommendations():
    """获取游戏推荐API"""
    user_id = get_jwt_identity()

    # 检查缓存
    if user_id in recommendation_cache:
        cached_results = recommendation_cache[user_id]
        # 如果缓存不超过1小时，使用缓存
        if time.time() - cached_results['timestamp'] < 3600:
            return jsonify({
                'status': 'success',
                'recommendations': cached_results['recommendations'],
                'cached': True
            })

    try:
        # 获取推荐
        n_recommendations = int(request.args.get('count', 10))
        recs = generate_recommendations(user_id, n_recommendations)

        # 转换推荐为带有游戏信息的列表
        recommendation_list = []
        for game_id, score in recs:
            game_info = get_game_info(game_id)
            if game_info:
                game_info['score'] = float(score)
                recommendation_list.append(game_info)

        # 更新缓存
        recommendation_cache[user_id] = {
            'recommendations': recommendation_list,
            'timestamp': time.time()
        }

        return jsonify({
            'status': 'success',
            'recommendations': recommendation_list,
            'cached': False
        })

    except Exception as e:
        logger.error(f"Error generating recommendations: {str(e)}")
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/api/games/<int:game_id>', methods=['GET'])
def get_game(game_id):
    """获取单个游戏信息API"""
    try:
        game_info = get_game_info(game_id)
        if game_info:
            return jsonify({'status': 'success', 'game': game_info})
        else:
            return jsonify({'status': 'error', 'message': 'Game not found'}), 404
    except Exception as e:
        logger.error(f"Error getting game info: {str(e)}")
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/api/games', methods=['GET'])
def get_games():
    """获取游戏列表API"""
    try:
        # 分页参数
        page = int(request.args.get('page', 1))
        limit = int(request.args.get('limit', 20))

        # 搜索参数
        search = request.args.get('search', '').lower()

        if games_df is None:
            return jsonify({'status': 'error', 'message': 'Game data not available'}), 500

        # 筛选游戏
        filtered_games = games_df
        if search:
            filtered_games = filtered_games[filtered_games['title'].str.lower().str.contains(search, na=False)]

        # 计算总页数
        total_games = len(filtered_games)
        total_pages = (total_games + limit - 1) // limit

        # 分页
        start = (page - 1) * limit
        end = min(start + limit, total_games)

        # 提取当前页的游戏
        current_page_games = filtered_games.iloc[start:end]

        # 转换为JSON
        games_list = []
        for _, row in current_page_games.iterrows():
            game_id = row['app_id']
            game_info = {
                'id': int(game_id),
                'title': row['title'],
                'tags': []
            }

            # 添加标签
            if game_id in games_metadata and 'tags' in games_metadata[game_id]:
                game_info['tags'] = games_metadata[game_id]['tags']

            # 添加其他可用信息
            for col in ['date_release', 'win', 'mac', 'linux', 'rating', 'positive_ratio', 'price_final',
                        'price_original']:
                if col in row and pd.notna(row[col]):
                    game_info[col] = row[col]

            games_list.append(game_info)

        return jsonify({
            'status': 'success',
            'games': games_list,
            'pagination': {
                'page': page,
                'limit': limit,
                'total_games': total_games,
                'total_pages': total_pages
            }
        })

    except Exception as e:
        logger.error(f"Error getting games: {str(e)}")
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/api/user/preferences', methods=['GET'])
@jwt_required()
def get_user_preferences():
    """获取用户偏好API"""
    user_id = get_jwt_identity()

    # 获取用户偏好
    prefs = user_preferences.get(user_id, {
        'liked_games': [],
        'disliked_games': [],
        'played_games': []
    })

    # 添加游戏详情
    prefs_with_details = {
        'liked_games': [get_game_info(game_id) for game_id in prefs.get('liked_games', []) if get_game_info(game_id)],
        'disliked_games': [get_game_info(game_id) for game_id in prefs.get('disliked_games', []) if
                           get_game_info(game_id)],
        'played_games': [get_game_info(game_id) for game_id in prefs.get('played_games', []) if get_game_info(game_id)]
    }

    return jsonify({
        'status': 'success',
        'preferences': prefs_with_details
    })


@app.route('/api/user/preferences', methods=['POST'])
@jwt_required()
def update_user_preferences():
    """更新用户偏好API"""
    user_id = get_jwt_identity()
    data = request.get_json()

    # 获取现有偏好
    prefs = user_preferences.get(user_id, {
        'liked_games': [],
        'disliked_games': [],
        'played_games': []
    })

    # 更新偏好
    action = data.get('action')
    game_id = data.get('game_id')

    if not action or not game_id:
        return jsonify({'status': 'error', 'message': 'Action and game_id are required'}), 400

    try:
        game_id = int(game_id)
    except ValueError:
        return jsonify({'status': 'error', 'message': 'Invalid game_id'}), 400

    # 根据动作更新偏好
    if action == 'like':
        if game_id not in prefs['liked_games']:
            prefs['liked_games'].append(game_id)
        # 如果游戏在不喜欢列表中，移除
        if game_id in prefs['disliked_games']:
            prefs['disliked_games'].remove(game_id)

    elif action == 'dislike':
        if game_id not in prefs['disliked_games']:
            prefs['disliked_games'].append(game_id)
        # 如果游戏在喜欢列表中，移除
        if game_id in prefs['liked_games']:
            prefs['liked_games'].remove(game_id)

    elif action == 'play':
        if game_id not in prefs['played_games']:
            prefs['played_games'].append(game_id)

    elif action == 'unlike':
        if game_id in prefs['liked_games']:
            prefs['liked_games'].remove(game_id)

    elif action == 'undislike':
        if game_id in prefs['disliked_games']:
            prefs['disliked_games'].remove(game_id)

    else:
        return jsonify({'status': 'error', 'message': 'Invalid action'}), 400

    # 更新用户偏好
    user_preferences[user_id] = prefs

    # 清除推荐缓存
    if user_id in recommendation_cache:
        del recommendation_cache[user_id]

    return jsonify({
        'status': 'success',
        'message': 'Preferences updated',
        'preferences': prefs
    })


@app.route('/api/similar-games/<int:game_id>', methods=['GET'])
def get_similar_games_api(game_id):
    """获取相似游戏API"""
    try:
        # 获取相似游戏数量
        n_similar = int(request.args.get('count', 5))

        # 获取相似游戏
        similar_games = get_similar_games(game_id, n_similar)

        # 转换为带有游戏信息的列表
        similar_list = []
        for similar_id, similarity in similar_games:
            game_info = get_game_info(similar_id)
            if game_info:
                game_info['similarity'] = float(similarity)
                similar_list.append(game_info)

        return jsonify({
            'status': 'success',
            'similar_games': similar_list
        })

    except Exception as e:
        logger.error(f"Error getting similar games: {str(e)}")
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/api/popular-games', methods=['GET'])
def get_popular_games_api():
    """获取流行游戏API"""
    try:
        # 获取流行游戏数量
        n_popular = int(request.args.get('count', 10))

        # 获取流行游戏
        popular_games = get_popular_games(n_popular)

        # 转换为带有游戏信息的列表
        popular_list = []
        for game_id, popularity in popular_games:
            game_info = get_game_info(game_id)
            if game_info:
                game_info['popularity'] = float(popularity)
                popular_list.append(game_info)

        return jsonify({
            'status': 'success',
            'popular_games': popular_list
        })

    except Exception as e:
        logger.error(f"Error getting popular games: {str(e)}")
        return jsonify({'status': 'error', 'message': str(e)}), 500


# 前端应用路由
@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def serve(path):
    if path and os.path.exists(os.path.join(app.static_folder, path)):
        return send_from_directory(app.static_folder, path)
    else:
        return send_from_directory(app.static_folder, 'index.html')

# 自定义推荐
@app.route('/api/custom-recommendations', methods=['POST'])
def custom_recommendations():
    """基于用户操作生成自定义推荐API"""
    try:
        # 获取用户操作
        user_actions = request.get_json()

        # 记录用户操作信息
        logger.info(f"Generating custom recommendations based on user actions:")
        logger.info(f"Liked games: {len(user_actions.get('liked', []))} games")
        logger.info(f"Purchased games: {len(user_actions.get('purchased', []))} games")
        logger.info(f"Recommended games: {len(user_actions.get('recommended', []))} games")

        # 提取所有用户有操作的游戏ID
        all_action_game_ids = set()
        all_action_game_ids.update(user_actions.get('liked', []))
        all_action_game_ids.update(user_actions.get('purchased', []))
        all_action_game_ids.update(user_actions.get('recommended', []))

        # 如果用户没有任何操作，返回热门游戏
        if not all_action_game_ids:
            popular_games = get_popular_games(10)
            popular_list = []
            for game_id, popularity in popular_games:
                game_info = get_game_info(game_id)
                if game_info:
                    game_info['score'] = float(popularity)
                    popular_list.append(game_info)

            return jsonify({
                'status': 'success',
                'recommendations': popular_list,
                'is_popular': True
            })

        # 查找与用户操作过的游戏相似的游戏
        similar_game_scores = {}  # 游戏ID -> 累计相似度分数
        games_excluded = set(all_action_game_ids)  # 排除用户已经操作过的游戏

        # 不同操作类型的权重
        action_weights = {
            'liked': 3.0,  # 喜欢的游戏权重最高
            'purchased': 2.0,  # 购买的游戏权重次之
            'recommended': 2.5  # 推荐的游戏权重也较高
        }

        # 为每种操作类型获取相似游戏
        for action_type, game_ids in user_actions.items():
            if action_type not in action_weights or not game_ids:
                continue

            weight = action_weights[action_type]

            for game_id in game_ids:
                # 获取相似游戏
                similar_games = get_similar_games(game_id, 5)

                # 累加相似度分数
                for similar_id, similarity in similar_games:
                    if similar_id not in games_excluded:
                        # 加权相似度
                        weighted_similarity = similarity * weight
                        if similar_id in similar_game_scores:
                            similar_game_scores[similar_id] += weighted_similarity
                        else:
                            similar_game_scores[similar_id] = weighted_similarity

        # 获取标签信息以增加多样性
        if games_df is not None and 'tags' in games_df.columns:
            # 提取用户操作过的游戏标签
            user_game_tags = set()
            for game_id in all_action_game_ids:
                game_data = games_df[games_df['app_id'] == game_id]
                if len(game_data) > 0 and 'tags' in game_data.columns and pd.notna(game_data['tags'].iloc[0]):
                    tags = [tag.strip() for tag in game_data['tags'].iloc[0].split(',')]
                    user_game_tags.update(tags)

            # 为有相同标签的游戏增加分数
            for similar_id in list(similar_game_scores.keys()):
                game_data = games_df[games_df['app_id'] == similar_id]
                if len(game_data) > 0 and 'tags' in game_data.columns and pd.notna(game_data['tags'].iloc[0]):
                    tags = [tag.strip() for tag in game_data['tags'].iloc[0].split(',')]
                    # 计算与用户标签的重叠
                    common_tags = set(tags) & user_game_tags
                    if common_tags:
                        # 标签匹配加分
                        tag_bonus = 0.2 * len(common_tags)
                        similar_game_scores[similar_id] += tag_bonus

        # 排序并获取前10个推荐
        sorted_recommendations = sorted(similar_game_scores.items(), key=lambda x: x[1], reverse=True)[:10]

        # 如果推荐不足10个，添加一些流行游戏
        if len(sorted_recommendations) < 10:
            already_recommended = set(game_id for game_id, _ in sorted_recommendations)
            popular_games = get_popular_games(20)

            for game_id, popularity in popular_games:
                if game_id not in already_recommended and game_id not in games_excluded:
                    sorted_recommendations.append((game_id, popularity * 0.7))  # 降低权重
                    if len(sorted_recommendations) >= 10:
                        break

        # 获取游戏详情
        recommendations_with_details = []
        for game_id, score in sorted_recommendations:
            game_info = get_game_info(game_id)
            if game_info:
                # 归一化分数到0-1范围
                normalized_score = min(score / 5.0, 1.0)
                game_info['score'] = float(normalized_score)
                recommendations_with_details.append(game_info)

        return jsonify({
            'status': 'success',
            'recommendations': recommendations_with_details,
            'is_popular': False
        })

    except Exception as e:
        logger.error(f"Error generating custom recommendations: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': 'Failed to generate recommendations. Please try again later.'
        }), 500

# 启动应用
if __name__ == '__main__':
    # 加载数据
    load_data()

    # 启动Flask应用
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)