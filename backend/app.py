#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
app.py - Improved Flask API for Steam Game Recommender using CSV data
Enhanced with better error handling and fallback sample data
"""
from flask import Flask, request, jsonify, send_from_directory, current_app, Response
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
import traceback
from functools import lru_cache
from sample_data import generate_sample_games, generate_sample_recommendations, generate_sample_users

# Custom JSON encoder class that handles NumPy types
class NumpyJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyJSONEncoder, self).default(obj)

# Create a custom jsonify function that uses our encoder
def custom_jsonify(*args, **kwargs):
    """Custom jsonify function that handles NumPy types"""
    if args and kwargs:
        raise TypeError('jsonify() behavior undefined when passed both args and kwargs')
    elif len(args) == 1:  # single args are passed directly to dumps()
        data = args[0]
    else:
        data = args or kwargs

    # Use the NumpyJSONEncoder for serialization
    response = Response(
        json.dumps(data, cls=NumpyJSONEncoder) + '\n',
        mimetype='application/json'
    )
    return response

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__, static_folder='../frontend/build')
CORS(app)  # Enable CORS

# Configure JWT
app.config['JWT_SECRET_KEY'] = os.environ.get('JWT_SECRET_KEY', 'your-secret-key')
app.config['JWT_ACCESS_TOKEN_EXPIRES'] = 86400  # 24-hour token expiration
jwt = JWTManager(app)

# Data file paths
DATA_DIR = os.environ.get('DATA_DIR', '../data')
GAMES_CSV = os.path.join(DATA_DIR, 'games.csv')
GAMES_METADATA = os.path.join(DATA_DIR, 'games_metadata.json')
RECOMMENDATIONS_CSV = os.path.join(DATA_DIR, 'recommendations.csv')
USERS_CSV = os.path.join(DATA_DIR, 'users.csv')

# Flag to indicate whether to use sample data
USE_SAMPLE_DATA = os.environ.get('USE_SAMPLE_DATA', 'false').lower() == 'true'

# Global data
games_df = None
games_metadata = {}
recommendations_df = None
users_df = None
game_info_cache = {}

# User data (would use a database in a real application)
users_db = {
    "demo": {
        "id": "12345-demo-user",
        "password_hash": generate_password_hash("password"),
        "created_at": time.time()
    }
}
user_preferences = {}
recommendation_cache = {}

# Default user preferences
DEFAULT_USER_PREFERENCES = {
    "12345-demo-user": {
        "liked_games": [],
        "disliked_games": [],
        "played_games": []
    }
}

def get_game_info(game_id):
    """Get game information with caching for better performance"""
    # Try to convert to int for consistent lookup
    try:
        game_id = int(game_id)
    except ValueError:
        return None

    # Check cache first
    if game_id in game_info_cache:
        return game_info_cache[game_id]

    # Not in cache, need to look up
    game_info = None

    # Try to find in DataFrame
    if games_df is not None:
        # Use boolean indexing to find the game
        game_data = games_df[games_df['app_id'] == game_id]

        if len(game_data) > 0:
            first_row = game_data.iloc[0]
            game_info = {
                'id': int(game_id),
                'title': first_row['title'] if 'title' in first_row and pd.notna(
                    first_row['title']) else f"Game {game_id}"
            }

            # Add tags (handle different possible formats safely)
            game_tags = []
            if 'tags' in game_data.columns:
                # Get the first row's tags value
                tags_value = first_row['tags']

                # Check if it's not NA (avoiding the array truth value error)
                if isinstance(tags_value, list) or (isinstance(tags_value, (str, float, int)) and pd.notna(tags_value)):
                    # Handle different tag formats
                    if isinstance(tags_value, list):
                        game_tags = tags_value
                    elif isinstance(tags_value, str):
                        game_tags = [tag.strip() for tag in tags_value.split(',')]

            # If no tags found in DataFrame, try from metadata
            if not game_tags and game_id in games_metadata and 'tags' in games_metadata[game_id]:
                metadata_tags = games_metadata[game_id]['tags']
                if isinstance(metadata_tags, list):
                    game_tags = metadata_tags
                elif isinstance(metadata_tags, str):
                    game_tags = [tag.strip() for tag in metadata_tags.split(',')]

            # Assign the tags to game_info
            game_info['tags'] = game_tags

            # Add description
            if 'description' in game_data.columns and pd.notna(first_row['description']):
                game_info['description'] = first_row['description']
            elif game_id in games_metadata and 'description' in games_metadata[game_id]:
                game_info['description'] = games_metadata[game_id]['description']
            else:
                game_info['description'] = ""  # Default empty description

            # Add other available info from DataFrame - non-boolean columns
            for col in ['date_release', 'rating', 'positive_ratio', 'price_final', 'price_original']:
                if col in game_data.columns and pd.notna(first_row[col]):
                    game_info[col] = first_row[col]

            # Handle boolean columns separately
            for bool_col in ['win', 'mac', 'linux']:
                if bool_col in game_data.columns and pd.notna(first_row[bool_col]):
                    # Convert Python boolean to JSON-compatible boolean
                    bool_value = first_row[bool_col]
                    # Convert to a proper boolean if it's a string representation
                    if isinstance(bool_value, str):
                        bool_value = bool_value.lower() == 'true'
                    # Now store it as a Python bool which Flask can properly serialize
                    game_info[bool_col] = bool(bool_value)

    # If not found in DataFrame, try metadata
    if game_info is None and game_id in games_metadata:
        metadata = games_metadata[game_id]
        game_info = {
            'id': int(game_id),
            'title': metadata.get('title', f'Game {game_id}'),
            'tags': metadata.get('tags', []),
            'description': metadata.get('description', '')
        }

        # Add other available info - handling non-boolean columns
        for key in ['date_release', 'rating', 'positive_ratio', 'price_final', 'price_original']:
            if key in metadata:
                game_info[key] = metadata[key]

        # Handle boolean columns separately
        for bool_col in ['win', 'mac', 'linux']:
            if bool_col in metadata:
                bool_value = metadata[bool_col]
                # Convert string representations to proper booleans
                if isinstance(bool_value, str):
                    bool_value = bool_value.lower() == 'true'
                # Store as a Python bool which Flask can properly serialize
                game_info[bool_col] = bool(bool_value)

    # Cache the result (even if None)
    if game_info is not None:
        game_info_cache[game_id] = game_info

    return game_info

def convert_numpy_types(obj):
    """Convert numpy types to native Python types recursively."""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_numpy_types(i) for i in obj]
    return obj


def load_data():
    """Load CSV and JSON data with enhanced error handling and fallback to sample data"""
    global games_df, games_metadata, recommendations_df, users_df, user_preferences

    # Initialize with None to ensure clean state on reload
    games_df = None
    games_metadata = {}
    recommendations_df = None
    users_df = None

    try:
        # If sample data is explicitly requested, skip loading real data
        if USE_SAMPLE_DATA:
            logger.info("Using sample data as requested by environment variable")
            return load_sample_data()

        # Load games CSV
        logger.info(f"Loading games data from {GAMES_CSV}")
        if os.path.exists(GAMES_CSV):
            games_df = pd.read_csv(GAMES_CSV)
            logger.info(f"Loaded {len(games_df)} games")
        else:
            logger.warning(f"Games CSV not found at {GAMES_CSV}")
            games_df = None

        # Load games metadata JSON with explicit UTF-8 encoding
        logger.info(f"Loading games metadata from {GAMES_METADATA}")
        if os.path.exists(GAMES_METADATA):
            try:
                with open(GAMES_METADATA, 'r', encoding='utf-8') as f:  # Explicitly specify UTF-8 encoding
                    for line in f:
                        try:
                            game_data = json.loads(line.strip())
                            games_metadata[game_data['app_id']] = game_data
                        except json.JSONDecodeError:
                            continue
                logger.info(f"Loaded metadata for {len(games_metadata)} games")
            except Exception as e:
                logger.error(f"Error loading games metadata: {str(e)}")
                logger.error(traceback.format_exc())
                games_metadata = {}
        else:
            logger.warning(f"Games metadata not found at {GAMES_METADATA}")
            games_metadata = {}

        # Load recommendations CSV
        logger.info(f"Loading recommendations data from {RECOMMENDATIONS_CSV}")
        if os.path.exists(RECOMMENDATIONS_CSV):
            recommendations_df = pd.read_csv(RECOMMENDATIONS_CSV)
            logger.info(f"Loaded {len(recommendations_df)} recommendations")
        else:
            logger.warning(f"Recommendations CSV not found at {RECOMMENDATIONS_CSV}")
            recommendations_df = None

        # Load users CSV
        logger.info(f"Loading users data from {USERS_CSV}")
        if os.path.exists(USERS_CSV):
            users_df = pd.read_csv(USERS_CSV)
            logger.info(f"Loaded {len(users_df)} users")
        else:
            logger.warning(f"Users CSV not found at {USERS_CSV}")
            users_df = None

        # Check if we need to fall back to sample data
        if games_df is None or len(games_df) == 0 or recommendations_df is None:
            logger.warning("Insufficient real data loaded. Falling back to sample data.")
            return load_sample_data()

        # Create accounts for users in the CSV
        if users_df is not None:
            for _, row in users_df.iterrows():
                user_id = str(row['user_id'])
                users_db[user_id] = {
                    "id": user_id,
                    "password_hash": generate_password_hash("password"),  # Default password for all users
                    "created_at": time.time()
                }

        # Initialize user preferences
        user_preferences.update(DEFAULT_USER_PREFERENCES)

        # Extract preferences from recommendations
        if recommendations_df is not None:
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

                # Add to played games
                if app_id not in user_preferences[user_id]["played_games"]:
                    user_preferences[user_id]["played_games"].append(app_id)

                # Add to liked/disliked
                if is_recommended and app_id not in user_preferences[user_id]["liked_games"]:
                    user_preferences[user_id]["liked_games"].append(app_id)
                elif not is_recommended and app_id not in user_preferences[user_id]["disliked_games"]:
                    user_preferences[user_id]["disliked_games"].append(app_id)

        return True
    except Exception as e:
        logger.error(f"Error loading data: {str(e)}")
        logger.error(traceback.format_exc())
        return load_sample_data()


def load_sample_data():
    """Load sample data when real data isn't available"""
    global games_df, games_metadata, recommendations_df, users_df, user_preferences

    try:
        logger.info("Loading sample data...")

        # Generate sample games
        sample_games = generate_sample_games(100)
        games_df = pd.DataFrame(sample_games)

        # Create metadata
        games_metadata = {game["app_id"]: game for game in sample_games}

        # Generate sample recommendations
        sample_recommendations = generate_sample_recommendations(20, 10)
        recommendations_df = pd.DataFrame(sample_recommendations)

        # Generate sample users
        sample_users = generate_sample_users(20)
        users_df = pd.DataFrame(sample_users)

        # Create accounts for sample users
        for user in sample_users:
            user_id = str(user["user_id"])
            users_db[user_id] = {
                "id": user_id,
                "password_hash": generate_password_hash("password"),
                "created_at": time.time()
            }

        # Initialize user preferences
        user_preferences.update(DEFAULT_USER_PREFERENCES)

        # Extract preferences from recommendations
        for rec in sample_recommendations:
            user_id = str(rec['user_id'])
            app_id = rec['app_id']
            is_recommended = rec['is_recommended']

            if user_id not in user_preferences:
                user_preferences[user_id] = {
                    "liked_games": [],
                    "disliked_games": [],
                    "played_games": []
                }

            # Add to played games
            if app_id not in user_preferences[user_id]["played_games"]:
                user_preferences[user_id]["played_games"].append(app_id)

            # Add to liked/disliked
            if is_recommended and app_id not in user_preferences[user_id]["liked_games"]:
                user_preferences[user_id]["liked_games"].append(app_id)
            elif not is_recommended and app_id not in user_preferences[user_id]["disliked_games"]:
                user_preferences[user_id]["disliked_games"].append(app_id)

        logger.info("Sample data loaded successfully")
        return True
    except Exception as e:
        logger.error(f"Error loading sample data: {str(e)}")
        logger.error(traceback.format_exc())
        return False


def get_game_info(game_id):
    """Get game information with improved error handling for tags and NA values"""
    try:
        game_id = int(game_id)
    except ValueError:
        return None

    # Try to find in DataFrame
    if games_df is not None:
        # Use boolean indexing to find the game
        game_data = games_df[games_df['app_id'] == game_id]

        if len(game_data) > 0:
            first_row = game_data.iloc[0]
            game_info = {
                'id': int(game_id),
                'title': first_row['title'] if 'title' in first_row and pd.notna(
                    first_row['title']) else f"Game {game_id}"
            }

            # Add tags (handle different possible formats safely)
            game_tags = []
            if 'tags' in game_data.columns:
                # Get the first row's tags value
                tags_value = first_row['tags']

                # Check if it's not NA (avoiding the array truth value error)
                if isinstance(tags_value, list) or (isinstance(tags_value, (str, float, int)) and pd.notna(tags_value)):
                    # Handle different tag formats
                    if isinstance(tags_value, list):
                        game_tags = tags_value
                    elif isinstance(tags_value, str):
                        game_tags = [tag.strip() for tag in tags_value.split(',')]

            # If no tags found in DataFrame, try from metadata
            if not game_tags and game_id in games_metadata and 'tags' in games_metadata[game_id]:
                metadata_tags = games_metadata[game_id]['tags']
                if isinstance(metadata_tags, list):
                    game_tags = metadata_tags
                elif isinstance(metadata_tags, str):
                    game_tags = [tag.strip() for tag in metadata_tags.split(',')]

            # Assign the tags to game_info
            game_info['tags'] = game_tags

            # Add description
            if 'description' in game_data.columns and pd.notna(first_row['description']):
                game_info['description'] = first_row['description']
            elif game_id in games_metadata and 'description' in games_metadata[game_id]:
                game_info['description'] = games_metadata[game_id]['description']
            else:
                game_info['description'] = ""  # Default empty description

            # Add other available info from DataFrame - non-boolean columns
            for col in ['date_release', 'rating', 'positive_ratio', 'price_final', 'price_original']:
                if col in game_data.columns and pd.notna(first_row[col]):
                    game_info[col] = first_row[col]

            # Handle boolean columns separately
            for bool_col in ['win', 'mac', 'linux']:
                if bool_col in game_data.columns and pd.notna(first_row[bool_col]):
                    # Convert Python boolean to JSON-compatible boolean
                    bool_value = first_row[bool_col]
                    # Convert to a proper boolean if it's a string representation
                    if isinstance(bool_value, str):
                        bool_value = bool_value.lower() == 'true'
                    # Now store it as a Python bool which Flask can properly serialize
                    game_info[bool_col] = bool(bool_value)

            return game_info

    # If not found in DataFrame, try metadata
    if game_id in games_metadata:
        metadata = games_metadata[game_id]
        game_info = {
            'id': int(game_id),
            'title': metadata.get('title', f'Game {game_id}'),
            'tags': metadata.get('tags', []),
            'description': metadata.get('description', '')
        }

        # Add other available info - handling non-boolean columns
        for key in ['date_release', 'rating', 'positive_ratio', 'price_final', 'price_original']:
            if key in metadata:
                game_info[key] = metadata[key]

        # Handle boolean columns separately
        for bool_col in ['win', 'mac', 'linux']:
            if bool_col in metadata:
                bool_value = metadata[bool_col]
                # Convert string representations to proper booleans
                if isinstance(bool_value, str):
                    bool_value = bool_value.lower() == 'true'
                # Store as a Python bool which Flask can properly serialize
                game_info[bool_col] = bool(bool_value)

        return game_info

    # Game not found
    return None


def get_popular_games(n=10):
    """Get popular games - FIXED optimized version with caching"""
    # Use a simple in-memory cache to avoid recalculating popular games

    # Check if we have a cached result that's less than 1 hour old
    current_time = time.time()
    if hasattr(get_popular_games, '_cache') and hasattr(get_popular_games, '_cache_time'):
        if current_time - get_popular_games._cache_time < 3600:  # 1 hour cache validity
            # Return cached results if available for requested size
            if len(get_popular_games._cache) >= n:
                return get_popular_games._cache[:n]

    # If no valid cache, calculate popular games
    if recommendations_df is not None and len(recommendations_df) > 0:
        # For large datasets, use a more efficient approach
        if len(recommendations_df) > 100000:
            # Take a random sample of 100,000 recommendations for faster processing
            sample_size = min(100000, len(recommendations_df))
            sample_df = recommendations_df.sample(n=sample_size, random_state=42)

            # Calculate game counts from the sample
            game_counts = sample_df['app_id'].value_counts()

            # Calculate positive rating ratio from the sample
            game_ratings = {}
            # Get the top 100 games by count
            top_games = game_counts.head(100).index.tolist()

            for game_id in top_games:
                game_data = sample_df[sample_df['app_id'] == game_id]
                if 'is_recommended' in game_data.columns:
                    pos_ratio = game_data['is_recommended'].mean()
                    game_ratings[game_id] = pos_ratio
                else:
                    game_ratings[game_id] = 0.5  # Default if no ratings available

            # Calculate combined score: 70% popularity + 30% ratings
            popular_items = []
            total_reviews = sample_size

            for game_id in top_games:
                count = game_counts[game_id]
                pop_score = count / total_reviews
                rating_score = game_ratings.get(game_id, 0.5)
                final_score = (pop_score * 0.7) + (rating_score * 0.3)
                popular_items.append((game_id, final_score))
        else:
            # Original approach for smaller datasets
            game_counts = recommendations_df['app_id'].value_counts()

            # Get the top 100 games by count
            top_games = game_counts.head(100).index.tolist()

            # Calculate positive rating ratio
            game_ratings = {}
            for game_id in top_games:
                game_data = recommendations_df[recommendations_df['app_id'] == game_id]
                if 'is_recommended' in game_data.columns:
                    pos_ratio = game_data['is_recommended'].mean()
                    game_ratings[game_id] = pos_ratio
                else:
                    game_ratings[game_id] = 0.5

            # Calculate combined score
            popular_items = []
            total_reviews = len(recommendations_df)

            for game_id in top_games:
                count = game_counts[game_id]
                pop_score = count / total_reviews
                rating_score = game_ratings.get(game_id, 0.5)
                final_score = (pop_score * 0.7) + (rating_score * 0.3)
                popular_items.append((game_id, final_score))

        # Sort and get top N
        popular_items.sort(key=lambda x: x[1], reverse=True)
        result = popular_items[:100]  # Cache more than requested for future calls

    # Fallback to games sorted by rating if no recommendation data
    elif games_df is not None and len(games_df) > 0:
        if 'rating' in games_df.columns:
            # Handle the case where rating column is not numeric
            try:
                # First attempt to convert the rating column to numeric if possible
                games_df['rating_numeric'] = pd.to_numeric(games_df['rating'], errors='coerce')
                # Sort by the converted column
                top_games_df = games_df.sort_values('rating_numeric', ascending=False).head(100)
            except:
                # If conversion fails, just take the first 100 games
                top_games_df = games_df.head(100)
        else:
            # Just take the first N games
            top_games_df = games_df.head(100)

        # Calculate normalized scores between 0.5 and 1.0
        max_idx = len(top_games_df)
        result = [(row['app_id'], 1.0 - (0.5 * idx / max_idx)) for idx, (_, row) in enumerate(top_games_df.iterrows())]
    else:
        # Fallback: empty list
        result = []

    # Update cache
    get_popular_games._cache = result
    get_popular_games._cache_time = current_time

    return result[:n]


def get_similar_games(game_id, n=5):
    """Get similar games based on tag similarity with improved error handling"""
    try:
        game_id = int(game_id)
    except ValueError:
        return []

    # Get the target game's tags
    target_tags = set()

    # Try from games_metadata
    if game_id in games_metadata and 'tags' in games_metadata[game_id]:
        tags_value = games_metadata[game_id]['tags']
        if isinstance(tags_value, list):
            target_tags = set(tags_value)
        elif isinstance(tags_value, str):
            target_tags = set(tag.strip() for tag in tags_value.split(','))

    # Try from games_df if no tags found yet
    if not target_tags and games_df is not None:
        game_rows = games_df[games_df['app_id'] == game_id]
        if len(game_rows) > 0 and 'tags' in game_rows.columns:
            # Get the first row's tags value
            tags_value = game_rows.iloc[0]['tags']
            # Safe check for NA
            if isinstance(tags_value, list) or (isinstance(tags_value, (str, float, int)) and pd.notna(tags_value)):
                if isinstance(tags_value, list):
                    target_tags = set(tags_value)
                elif isinstance(tags_value, str):
                    target_tags = set(tag.strip() for tag in tags_value.split(','))

    # If still no tags, return empty list
    if not target_tags:
        return []

    # Calculate similarity with other games
    similarities = []

    # Process games from DataFrame
    if games_df is not None:
        for _, row in games_df.iterrows():
            other_id = row['app_id']

            # Skip the same game
            if other_id == game_id:
                continue

            # Get other game's tags
            other_tags = set()
            if 'tags' in row:
                tags_value = row['tags']
                # Safe check for NA
                if isinstance(tags_value, list) or (isinstance(tags_value, (str, float, int)) and pd.notna(tags_value)):
                    if isinstance(tags_value, list):
                        other_tags = set(tags_value)
                    elif isinstance(tags_value, str):
                        other_tags = set(tag.strip() for tag in tags_value.split(','))

            # Skip if no tags
            if not other_tags:
                continue

            # Calculate Jaccard similarity
            intersection = len(target_tags.intersection(other_tags))
            union = len(target_tags.union(other_tags))

            if union > 0:
                similarity = intersection / union
                similarities.append((other_id, similarity))

    # Process games from metadata if they're not in the DataFrame
    for other_id, metadata in games_metadata.items():
        # Skip if already processed or it's the same game
        if other_id == game_id or any(sim_id == other_id for sim_id, _ in similarities):
            continue

        # Get tags
        other_tags = set()
        if 'tags' in metadata:
            tags_value = metadata['tags']
            if isinstance(tags_value, list):
                other_tags = set(tags_value)
            elif isinstance(tags_value, str):
                other_tags = set(tag.strip() for tag in tags_value.split(','))

        # Skip if no tags
        if not other_tags:
            continue

        # Calculate Jaccard similarity
        intersection = len(target_tags.intersection(other_tags))
        union = len(target_tags.union(other_tags))

        if union > 0:
            similarity = intersection / union
            similarities.append((other_id, similarity))

    # Sort by similarity and return top N
    similarities.sort(key=lambda x: x[1], reverse=True)
    return similarities[:n]


def generate_recommendations(user_id, n=10):
    """Generate game recommendations based on user preferences with improved error handling"""
    if user_id not in user_preferences:
        return get_popular_games(n)

    user_prefs = user_preferences[user_id]
    liked_games = user_prefs.get('liked_games', [])

    if not liked_games:
        return get_popular_games(n)

    # Collect tags from liked games
    user_tags = {}
    for game_id in liked_games:
        # Get game tags
        game_tags = []

        # Try from games_metadata
        if game_id in games_metadata and 'tags' in games_metadata[game_id]:
            tags_value = games_metadata[game_id]['tags']
            if isinstance(tags_value, list):
                game_tags = tags_value
            elif isinstance(tags_value, str):
                game_tags = [tag.strip() for tag in tags_value.split(',')]

        # Try from games_df if no tags found yet
        if not game_tags and games_df is not None:
            game_rows = games_df[games_df['app_id'] == game_id]
            if len(game_rows) > 0 and 'tags' in game_rows.columns:
                # Get the first row's tags value safely
                tags_value = game_rows.iloc[0]['tags']
                # Check if it's not NA
                if isinstance(tags_value, list) or (isinstance(tags_value, (str, float, int)) and pd.notna(tags_value)):
                    if isinstance(tags_value, list):
                        game_tags = tags_value
                    elif isinstance(tags_value, str):
                        game_tags = [tag.strip() for tag in tags_value.split(',')]

        # Count tag occurrences
        for tag in game_tags:
            user_tags[tag] = user_tags.get(tag, 0) + 1

    # If no tags, return popular games
    if not user_tags:
        return get_popular_games(n)

    # Calculate tag match scores for all games
    game_scores = {}
    played_games = set(user_prefs.get('played_games', []))

    # Process games from DataFrame
    if games_df is not None:
        for _, row in games_df.iterrows():
            game_id = row['app_id']

            # Skip already played games
            if game_id in played_games:
                continue

            # Get game tags safely
            game_tags = []
            if 'tags' in row:
                tags_value = row['tags']
                # Safe check for NA
                if isinstance(tags_value, list) or (isinstance(tags_value, (str, float, int)) and pd.notna(tags_value)):
                    if isinstance(tags_value, list):
                        game_tags = tags_value
                    elif isinstance(tags_value, str):
                        game_tags = [tag.strip() for tag in tags_value.split(',')]

            # Skip if no tags
            if not game_tags:
                continue

            # Calculate tag match score
            score = 0
            for tag in game_tags:
                if tag in user_tags:
                    score += user_tags[tag]

            # Normalize score
            if len(game_tags) > 0:
                game_scores[game_id] = score / len(game_tags)

    # Process games from metadata if they're not in the DataFrame
    for game_id, metadata in games_metadata.items():
        # Skip if we've already processed this game or it's already played
        if game_id in game_scores or game_id in played_games:
            continue

        # Get tags
        game_tags = []
        if 'tags' in metadata:
            tags_value = metadata['tags']
            if isinstance(tags_value, list):
                game_tags = tags_value
            elif isinstance(tags_value, str):
                game_tags = [tag.strip() for tag in tags_value.split(',')]

        # Skip if no tags
        if not game_tags:
            continue

        # Calculate tag match score
        score = 0
        for tag in game_tags:
            if tag in user_tags:
                score += user_tags[tag]

        # Normalize score
        if len(game_tags) > 0:
            game_scores[game_id] = score / len(game_tags)

    # Convert to list and sort
    recommendations = [(game_id, score) for game_id, score in game_scores.items()]
    recommendations.sort(key=lambda x: x[1], reverse=True)

    # If not enough recommendations, add popular games
    if len(recommendations) < n:
        popular = get_popular_games(n)
        popular_ids = [p[0] for p in popular]

        for game_id, score in popular:
            if game_id not in [r[0] for r in recommendations] and game_id not in played_games:
                # Lower the score for popular recommendations
                recommendations.append((game_id, score * 0.8))

                if len(recommendations) >= n:
                    break

    # Sort again and return top N
    recommendations.sort(key=lambda x: x[1], reverse=True)
    return recommendations[:n]

# API endpoints
@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check API"""
    if games_df is not None:
        return custom_jsonify({'status': 'ok', 'message': 'API service is running'})
    else:
        return custom_jsonify({'status': 'error', 'message': 'Data not loaded'}), 500


@app.route('/api/register', methods=['POST'])
def register():
    """User registration API"""
    data = request.get_json()
    username = data.get('username')
    password = data.get('password')

    if not username or not password:
        return custom_jsonify({'status': 'error', 'message': 'Username and password are required'}), 400

    if username in users_db:
        return custom_jsonify({'status': 'error', 'message': 'Username already exists'}), 409

    # Generate user ID and store user info
    user_id = str(uuid.uuid4())
    users_db[username] = {
        'id': user_id,
        'password_hash': generate_password_hash(password),
        'created_at': time.time()
    }

    # Create user preferences
    user_preferences[user_id] = {
        'liked_games': [],
        'disliked_games': [],
        'played_games': []
    }

    # Generate token
    access_token = create_access_token(identity=user_id)

    return custom_jsonify({
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
    """User login API"""
    data = request.get_json()
    username = data.get('username')
    password = data.get('password')

    if not username or not password:
        return custom_jsonify({'status': 'error', 'message': 'Username and password are required'}), 400

    # Auto-login for demo user
    if username == "demo" and password == "password":
        user = users_db["demo"]
        access_token = create_access_token(identity=user['id'])
        return custom_jsonify({
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
        return custom_jsonify({'status': 'error', 'message': 'Invalid credentials'}), 401

    # Generate token
    access_token = create_access_token(identity=user['id'])

    return custom_jsonify({
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
    """Get game recommendations API"""
    user_id = get_jwt_identity()

    # Check cache
    if user_id in recommendation_cache:
        cached_results = recommendation_cache[user_id]
        # Use cache if less than 1 hour old
        if time.time() - cached_results['timestamp'] < 3600:
            return custom_jsonify({
                'status': 'success',
                'recommendations': cached_results['recommendations'],
                'cached': True
            })

    try:
        # Get recommendations
        n_recommendations = int(request.args.get('count', 10))
        recs = generate_recommendations(user_id, n_recommendations)

        # Convert recommendations to list with game info
        recommendation_list = []
        for game_id, score in recs:
            game_info = get_game_info(game_id)
            if game_info:
                game_info['score'] = float(score)
                recommendation_list.append(game_info)

        # Update cache
        recommendation_cache[user_id] = {
            'recommendations': recommendation_list,
            'timestamp': time.time()
        }

        return custom_jsonify({
            'status': 'success',
            'recommendations': recommendation_list,
            'cached': False
        })

    except Exception as e:
        logger.error(f"Error generating recommendations: {str(e)}")
        logger.error(traceback.format_exc())
        return custom_jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/api/games/<int:game_id>', methods=['GET'])
def get_game(game_id):
    """Get single game info API"""
    try:
        game_info = get_game_info(game_id)
        if game_info:
            return custom_jsonify({'status': 'success', 'game': game_info})
        else:
            return custom_jsonify({'status': 'error', 'message': 'Game not found'}), 404
    except Exception as e:
        logger.error(f"Error getting game info: {str(e)}")
        logger.error(traceback.format_exc())
        return custom_jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/api/games', methods=['GET'])
def get_games():
    """Get games list API with improved error handling"""
    try:
        # Check if we have games data
        if games_df is None or len(games_df) == 0:
            logger.warning("No games data available, attempting to reload")
            if not load_data():
                return custom_jsonify({'status': 'error', 'message': 'Game data not available'}), 500

        # Pagination parameters
        page = int(request.args.get('page', 1))
        limit = int(request.args.get('limit', 20))

        # Search parameter
        search = request.args.get('search', '').lower()

        # Filter games
        filtered_games = games_df.copy()
        if search and 'title' in filtered_games.columns:
            # Handle potential missing values
            title_column = filtered_games['title'].fillna('')
            filtered_games = filtered_games[title_column.str.lower().str.contains(search, na=False)]

        # Calculate total pages
        total_games = len(filtered_games)
        total_pages = max(1, (total_games + limit - 1) // limit)

        # Ensure valid page number
        page = max(1, min(page, total_pages))

        # Pagination
        start = (page - 1) * limit
        end = min(start + limit, total_games)

        # Extract current page games
        if start < end:
            current_page_games = filtered_games.iloc[start:end]
        else:
            current_page_games = filtered_games.head(0)  # Empty DataFrame with same structure

        # Convert to JSON
        games_list = []
        for _, row in current_page_games.iterrows():
            game_id = row['app_id']

            # Get game info using our helper function
            game_info = get_game_info(game_id)
            if game_info:
                games_list.append(game_info)

        # Use convert_numpy_types before jsonify
        response_data = convert_numpy_types({
            'status': 'success',
            'games': games_list,
            'pagination': {
                'page': page,
                'limit': limit,
                'total_games': total_games,
                'total_pages': total_pages
            }
        })

        return custom_jsonify(response_data)

    except Exception as e:
        logger.error(f"Error getting games: {str(e)}")
        logger.error(traceback.format_exc())
        return custom_jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/api/user/preferences', methods=['GET'])
@jwt_required()
def get_user_preferences():
    """Get user preferences API"""
    user_id = get_jwt_identity()

    # Get user preferences
    prefs = user_preferences.get(user_id, {
        'liked_games': [],
        'disliked_games': [],
        'played_games': []
    })

    # Add game details
    prefs_with_details = {
        'liked_games': [get_game_info(game_id) for game_id in prefs.get('liked_games', []) if get_game_info(game_id)],
        'disliked_games': [get_game_info(game_id) for game_id in prefs.get('disliked_games', []) if
                           get_game_info(game_id)],
        'played_games': [get_game_info(game_id) for game_id in prefs.get('played_games', []) if get_game_info(game_id)]
    }

    return custom_jsonify({
        'status': 'success',
        'preferences': prefs_with_details
    })


@app.route('/api/user/preferences', methods=['POST'])
@jwt_required()
def update_user_preferences():
    """Update user preferences API"""
    user_id = get_jwt_identity()
    data = request.get_json()

    # Get existing preferences
    prefs = user_preferences.get(user_id, {
        'liked_games': [],
        'disliked_games': [],
        'played_games': []
    })

    # Update preferences
    action = data.get('action')
    game_id = data.get('game_id')

    if not action or not game_id:
        return custom_jsonify({'status': 'error', 'message': 'Action and game_id are required'}), 400

    try:
        game_id = int(game_id)
    except ValueError:
        return custom_jsonify({'status': 'error', 'message': 'Invalid game_id'}), 400

    # Update based on action
    if action == 'like':
        if game_id not in prefs['liked_games']:
            prefs['liked_games'].append(game_id)
        # If game in disliked list, remove
        if game_id in prefs['disliked_games']:
            prefs['disliked_games'].remove(game_id)

    elif action == 'dislike':
        if game_id not in prefs['disliked_games']:
            prefs['disliked_games'].append(game_id)
        # If game in liked list, remove
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
        return custom_jsonify({'status': 'error', 'message': 'Invalid action'}), 400

    # Update user preferences
    user_preferences[user_id] = prefs

    # Clear recommendation cache
    if user_id in recommendation_cache:
        del recommendation_cache[user_id]

    return custom_jsonify({
        'status': 'success',
        'message': 'Preferences updated',
        'preferences': prefs
    })


@app.route('/api/similar-games/<int:game_id>', methods=['GET'])
def get_similar_games_api(game_id):
    """Get similar games API"""
    try:
        # Get number of similar games
        n_similar = int(request.args.get('count', 5))

        # Get similar games
        similar_games = get_similar_games(game_id, n_similar)

        # Convert to list with game info
        similar_list = []
        for similar_id, similarity in similar_games:
            game_info = get_game_info(similar_id)
            if game_info:
                game_info['similarity'] = float(similarity)
                similar_list.append(game_info)

        return custom_jsonify({
            'status': 'success',
            'similar_games': similar_list
        })

    except Exception as e:
        logger.error(f"Error getting similar games: {str(e)}")
        logger.error(traceback.format_exc())
        return custom_jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/api/popular-games', methods=['GET'])
def get_popular_games_api():
    """Get popular games API"""
    try:
        # Get number of popular games
        n_popular = int(request.args.get('count', 10))

        # Get popular games
        popular_games = get_popular_games(n_popular)

        # Convert to list with game info
        popular_list = []
        for game_id, popularity in popular_games:
            game_info = get_game_info(game_id)
            if game_info:
                game_info['popularity'] = float(popularity)
                popular_list.append(game_info)

        return custom_jsonify({
            'status': 'success',
            'popular_games': popular_list
        })

    except Exception as e:
        logger.error(f"Error getting popular games: {str(e)}")
        logger.error(traceback.format_exc())
        return custom_jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/api/custom-recommendations', methods=['POST'])
def custom_recommendations():
    """Generate custom recommendations based on user actions - FIXED version"""
    try:
        # Get user actions
        user_actions = request.get_json()

        # Log user action info
        logger.info(f"Generating custom recommendations based on user actions:")
        logger.info(f"Liked games: {len(user_actions.get('liked', []))} games")
        logger.info(f"Purchased games: {len(user_actions.get('purchased', []))} games")
        logger.info(f"Recommended games: {len(user_actions.get('recommended', []))} games")

        # Extract all game IDs with user actions
        all_action_game_ids = set()
        all_action_game_ids.update(user_actions.get('liked', []))
        all_action_game_ids.update(user_actions.get('purchased', []))
        all_action_game_ids.update(user_actions.get('recommended', []))

        # If no user actions, return popular games
        if not all_action_game_ids:
            # Use the existing popular games function but limit to 10
            popular_games = get_popular_games(10)
            popular_list = []
            for game_id, popularity in popular_games:
                game_info = get_game_info(game_id)
                if game_info:
                    game_info['score'] = float(popularity)
                    popular_list.append(game_info)

            return custom_jsonify({
                'status': 'success',
                'recommendations': popular_list,
                'is_popular': True
            })

        # ---- OPTIMIZATION: Limit the number of games to process ----
        # Instead of processing the entire games_df, just process a reasonable subset
        MAX_GAMES_TO_PROCESS = 2000

        # Approach 1: If we're dealing with metadata, limit the games to consider
        games_to_process = set()

        # First, add direct neighbors from games_metadata as highest priority
        for action_game_id in all_action_game_ids:
            # Add similar games from metadata first
            if action_game_id in games_metadata and 'tags' in games_metadata[action_game_id]:
                action_game_tags = set(games_metadata[action_game_id]['tags']) if isinstance(
                    games_metadata[action_game_id]['tags'], list) else set()

                # Find games with at least one matching tag (limited)
                tag_matches = 0
                for other_id, metadata in games_metadata.items():
                    if tag_matches >= 100:  # Limit to 100 per action game
                        break
                    if other_id not in all_action_game_ids and 'tags' in metadata:
                        other_tags = set(metadata['tags']) if isinstance(metadata['tags'], list) else set()
                        if action_game_tags.intersection(other_tags):
                            games_to_process.add(other_id)
                            tag_matches += 1

        # If we still need more games, add popular ones
        if len(games_to_process) < MAX_GAMES_TO_PROCESS:
            # Get popular games
            popular_games = get_popular_games(MAX_GAMES_TO_PROCESS - len(games_to_process))
            # Extract just the game IDs from the popular games
            for game_id, _ in popular_games:
                games_to_process.add(game_id)

        # ---- Build a more efficient similarity calculation ----
        similar_game_scores = {}  # game_id -> cumulative similarity score

        # Action type weights
        action_weights = {
            'liked': 3.0,  # Highest weight for liked games
            'purchased': 2.0,  # Medium weight for purchased games
            'recommended': 2.5  # High weight for recommended games
        }

        # Collect all tags from user's action games for faster processing
        user_action_tags = {}
        for action_type, game_ids in user_actions.items():
            if action_type not in action_weights or not game_ids:
                continue

            weight = action_weights[action_type]

            for game_id in game_ids:
                # Get game tags
                game_tags = []

                # Try from games_metadata
                if game_id in games_metadata and 'tags' in games_metadata[game_id]:
                    tags_value = games_metadata[game_id]['tags']
                    if isinstance(tags_value, list):
                        game_tags = tags_value
                    elif isinstance(tags_value, str):
                        game_tags = [tag.strip() for tag in tags_value.split(',')]

                # Count tag occurrences with weight
                for tag in game_tags:
                    user_action_tags[tag] = user_action_tags.get(tag, 0) + weight

        # Fast similarity calculation using collected tags
        for game_id in games_to_process:
            if game_id in all_action_game_ids:
                continue  # Skip games user already interacted with

            # Get game tags
            game_tags = []

            # Try from games_metadata
            if game_id in games_metadata and 'tags' in games_metadata[game_id]:
                tags_value = games_metadata[game_id]['tags']
                if isinstance(tags_value, list):
                    game_tags = tags_value
                elif isinstance(tags_value, str):
                    game_tags = [tag.strip() for tag in tags_value.split(',')]

            # Calculate tag match score
            score = 0
            for tag in game_tags:
                if tag in user_action_tags:
                    score += user_action_tags[tag]

            # Normalize score by number of tags
            if len(game_tags) > 0:
                similar_game_scores[game_id] = score / len(game_tags)

        # Sort recommendations by score
        sorted_recommendations = sorted(similar_game_scores.items(), key=lambda x: x[1], reverse=True)[:20]

        # If not enough recommendations, add popular games
        if len(sorted_recommendations) < 10:
            already_recommended = set(game_id for game_id, _ in sorted_recommendations)
            popular_games = get_popular_games(20)

            for game_id, popularity in popular_games:
                if game_id not in already_recommended and game_id not in all_action_game_ids:
                    sorted_recommendations.append((game_id, popularity * 0.7))  # Lower weight for popular games
                    if len(sorted_recommendations) >= 20:
                        break

        # Get game details for the top 10
        recommendations_with_details = []
        for game_id, score in sorted_recommendations[:10]:
            game_info = get_game_info(game_id)
            if game_info:
                # Normalize score to 0-1 range
                normalized_score = min(score / 5.0, 1.0)
                game_info['score'] = float(normalized_score)
                recommendations_with_details.append(game_info)

        return custom_jsonify({
            'status': 'success',
            'recommendations': recommendations_with_details,
            'is_popular': False
        })

    except Exception as e:
        logger.error(f"Error generating custom recommendations: {str(e)}")
        logger.error(traceback.format_exc())
        return custom_jsonify({
            'status': 'error',
            'message': 'Failed to generate recommendations. Please try again later.'
        }), 500


# Frontend application routes
@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def serve(path):
    if path and os.path.exists(os.path.join(app.static_folder, path)):
        return send_from_directory(app.static_folder, path)
    else:
        return send_from_directory(app.static_folder, 'index.html')


# Application startup
if __name__ == '__main__':
    # Load data
    load_data()

    # Start Flask app
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)