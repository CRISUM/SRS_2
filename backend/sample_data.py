#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
backend/sample_data.py - Generate sample data when real data isn't available
Author: YourName
Date: 2025-04-30
Description: Provides fallback data for the API when CSV files can't be loaded
"""

import random
import logging
import time
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

# Sample game tags for generating realistic data
GAME_TAGS = [
    "Action", "Adventure", "RPG", "Strategy", "Simulation", "Sports", "Racing",
    "Puzzle", "Indie", "Casual", "Multiplayer", "FPS", "Horror", "Survival",
    "Open World", "Story Rich", "Atmospheric", "Sci-fi", "Fantasy", "Platformer"
]

# Sample game titles and their associated tags
SAMPLE_GAMES = [
    {
        "app_id": 1001,
        "title": "Cosmic Explorer",
        "tags": ["Space", "Adventure", "Sci-fi", "Exploration", "Open World"],
        "description": "Explore the vast cosmos, discover new planets and build your own space empire.",
        "date_release": "2024-06-15",
        "win": True,
        "mac": True,
        "linux": True,
        "rating": 8.7,
        "positive_ratio": 0.92,
        "price_final": 29.99,
        "price_original": 39.99
    },
    {
        "app_id": 1002,
        "title": "Dragon Quest: Legends",
        "tags": ["RPG", "Fantasy", "Adventure", "Story Rich", "Magic"],
        "description": "Embark on an epic quest to slay dragons and save the kingdom from darkness.",
        "date_release": "2023-11-05",
        "win": True,
        "mac": True,
        "linux": False,
        "rating": 9.1,
        "positive_ratio": 0.95,
        "price_final": 49.99,
        "price_original": 49.99
    },
    {
        "app_id": 1003,
        "title": "Urban Warfare",
        "tags": ["FPS", "Action", "Multiplayer", "Tactical", "Shooter"],
        "description": "Experience intense urban combat with realistic weapons and destructible environments.",
        "date_release": "2024-03-22",
        "win": True,
        "mac": False,
        "linux": False,
        "rating": 8.2,
        "positive_ratio": 0.87,
        "price_final": 39.99,
        "price_original": 59.99
    },
    {
        "app_id": 1004,
        "title": "Farming Valley",
        "tags": ["Simulation", "Farming", "Relaxing", "Life Sim", "Building"],
        "description": "Build your dream farm, raise animals, grow crops and become part of a vibrant community.",
        "date_release": "2023-08-10",
        "win": True,
        "mac": True,
        "linux": True,
        "rating": 8.9,
        "positive_ratio": 0.94,
        "price_final": 19.99,
        "price_original": 24.99
    },
    {
        "app_id": 1005,
        "title": "Race Masters",
        "tags": ["Racing", "Sports", "Multiplayer", "Competitive", "Arcade"],
        "description": "Feel the speed in this adrenaline-pumping racing game with realistic physics.",
        "date_release": "2024-01-18",
        "win": True,
        "mac": True,
        "linux": False,
        "rating": 7.8,
        "positive_ratio": 0.82,
        "price_final": 29.99,
        "price_original": 29.99
    },
    {
        "app_id": 1006,
        "title": "Dungeon Delvers",
        "tags": ["RPG", "Dungeon Crawler", "Fantasy", "Co-op", "Loot"],
        "description": "Dive into procedurally generated dungeons, collect epic loot and defeat fearsome monsters.",
        "date_release": "2023-10-03",
        "win": True,
        "mac": True,
        "linux": True,
        "rating": 8.5,
        "positive_ratio": 0.9,
        "price_final": 24.99,
        "price_original": 24.99
    },
    {
        "app_id": 1007,
        "title": "City Builder 2025",
        "tags": ["Simulation", "Strategy", "Building", "Management", "Economy"],
        "description": "Create and manage your own thriving metropolis with advanced economic systems.",
        "date_release": "2024-02-05",
        "win": True,
        "mac": True,
        "linux": True,
        "rating": 8.8,
        "positive_ratio": 0.91,
        "price_final": 34.99,
        "price_original": 34.99
    },
    {
        "app_id": 1008,
        "title": "Spooky Mansion",
        "tags": ["Horror", "Adventure", "Puzzle", "Atmospheric", "Mystery"],
        "description": "Explore a haunted mansion filled with terrifying secrets and challenging puzzles.",
        "date_release": "2023-12-13",
        "win": True,
        "mac": True,
        "linux": False,
        "rating": 8.0,
        "positive_ratio": 0.85,
        "price_final": 19.99,
        "price_original": 19.99
    },
    {
        "app_id": 1009,
        "title": "Platformer Paradise",
        "tags": ["Platformer", "Indie", "Pixel Graphics", "Difficult", "Action"],
        "description": "Jump, dash and wall-climb through challenging levels in this precision platformer.",
        "date_release": "2023-09-28",
        "win": True,
        "mac": True,
        "linux": True,
        "rating": 8.6,
        "positive_ratio": 0.93,
        "price_final": 14.99,
        "price_original": 14.99
    },
    {
        "app_id": 1010,
        "title": "Zombie Apocalypse",
        "tags": ["Survival", "Horror", "Open World", "Crafting", "Zombies"],
        "description": "Survive in a world overrun by zombies by scavenging resources and building defenses.",
        "date_release": "2024-05-07",
        "win": True,
        "mac": False,
        "linux": False,
        "rating": 7.9,
        "positive_ratio": 0.84,
        "price_final": 29.99,
        "price_original": 39.99
    }
]


def generate_sample_games(count=50):
    """Generate a larger set of sample games by combining elements from the base set

    Args:
        count: Number of games to generate

    Returns:
        list: List of game dictionaries
    """
    logger.info(f"Generating {count} sample games")
    games = list(SAMPLE_GAMES)  # Start with the base set

    # Generate additional games if needed
    while len(games) < count:
        # Get a random base game to use as a template
        base_game = random.choice(SAMPLE_GAMES)

        # Create a new game with some variations
        new_id = 1000 + len(games) + 1

        # Modify the title (add a sequel number, remix words, etc.)
        title_options = [
            f"{base_game['title']} {random.choice(['II', 'III', 'IV', 'V', 'Remastered', 'Definitive Edition'])}",
            f"{base_game['title']}: {random.choice(['New Dawn', 'Legacy', 'Origins', 'Evolution', 'Revolution'])}",
            f"{random.choice(['Super', 'Ultimate', 'Extreme', 'Epic'])} {base_game['title']}",
            f"{base_game['title']} {random.randint(2, 5)}"
        ]

        # Modify tags by keeping some and adding new ones
        base_tags = set(base_game['tags'])
        new_tags = set(random.sample(GAME_TAGS, random.randint(2, 5)))
        combined_tags = list(base_tags.union(new_tags))[:5]  # Keep max 5 tags

        # Generate a new release date between 1-24 months ago
        months_ago = random.randint(1, 24)
        release_date = (datetime.now() - timedelta(days=30 * months_ago)).strftime("%Y-%m-%d")

        # Create the new game
        new_game = {
            "app_id": new_id,
            "title": random.choice(title_options),
            "tags": combined_tags,
            "description": base_game["description"],
            "date_release": release_date,
            "win": True,
            "mac": random.choice([True, False]),
            "linux": random.choice([True, False]),
            "rating": round(random.uniform(6.0, 9.5), 1),
            "positive_ratio": round(random.uniform(0.7, 0.98), 2),
            "price_final": round(random.choice([14.99, 19.99, 24.99, 29.99, 39.99, 49.99, 59.99]), 2),
            "price_original": None  # Will be set below
        }

        # 30% chance of the game being on sale
        if random.random() < 0.3:
            new_game["price_original"] = round(new_game["price_final"] * random.uniform(1.2, 2.0), 2)
        else:
            new_game["price_original"] = new_game["price_final"]

        games.append(new_game)

    logger.info(f"Generated {len(games)} sample games")
    return games


def generate_sample_recommendations(user_count=10, games_per_user=5):
    """Generate sample recommendation data

    Args:
        user_count: Number of users to generate
        games_per_user: Number of games per user

    Returns:
        list: List of recommendation dictionaries
    """
    logger.info(f"Generating sample recommendations for {user_count} users")

    recommendations = []
    sample_games = generate_sample_games(30)  # Ensure we have enough games
    game_ids = [game["app_id"] for game in sample_games]

    for user_id in range(1, user_count + 1):
        # Select random games for this user
        user_games = random.sample(game_ids, min(games_per_user, len(game_ids)))

        for game_id in user_games:
            # 70% chance of recommending the game
            is_recommended = random.random() < 0.7

            # Random playtime between 0.5 and 100 hours
            hours = round(random.uniform(0.5, 100.0), 1)

            recommendation = {
                "user_id": user_id,
                "app_id": game_id,
                "is_recommended": is_recommended,
                "hours": hours
            }

            recommendations.append(recommendation)

    logger.info(f"Generated {len(recommendations)} sample recommendations")
    return recommendations


def generate_sample_users(count=20):
    """Generate sample user data

    Args:
        count: Number of users to generate

    Returns:
        list: List of user dictionaries
    """
    logger.info(f"Generating {count} sample users")

    users = []
    for user_id in range(1, count + 1):
        user = {
            "user_id": user_id,
            "username": f"user_{user_id}",
            "join_date": (datetime.now() - timedelta(days=random.randint(1, 365 * 3))).strftime("%Y-%m-%d"),
            "games_owned": random.randint(5, 100),
            "reviews_written": random.randint(0, 50)
        }
        users.append(user)

    logger.info(f"Generated {len(users)} sample users")
    return users