#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
src/visualization/visualizer.py - Visualization utilities
Author: YourName
Date: 2025-04-27
Description: Creates visualizations for model performance and recommendations
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import logging
import os
import traceback

logger = logging.getLogger(__name__)


class RecommenderVisualizer:
    """Visualizer for recommendation system"""

    def __init__(self, output_dir=None):
        """Initialize visualizer

        Args:
            output_dir (str): Directory to save visualizations
        """
        self.output_dir = output_dir or 'visualizations'
        os.makedirs(self.output_dir, exist_ok=True)

        # Set matplotlib style
        plt.style.use('ggplot')

        # Set color palette
        sns.set_palette("viridis")

    def visualize_metrics(self, metrics):
        """Visualize evaluation metrics

        Args:
            metrics (dict): Evaluation metrics from RecommenderEvaluator
        """
        logger.info("Creating evaluation metrics visualizations...")

        try:
            # Create figure for metrics
            plt.figure(figsize=(15, 10))

            # Get k values
            k_values = None
            for metric in ['precision', 'recall', 'ndcg', 'diversity']:
                if metric in metrics and isinstance(metrics[metric], dict):
                    k_values = sorted([int(k) for k in metrics[metric].keys()])
                    break

            if not k_values:
                logger.warning("No k values found in metrics")
                return

            # Plot precision, recall, NDCG, diversity
            plot_metrics = ['precision', 'recall', 'ndcg', 'diversity']
            for i, metric in enumerate(plot_metrics, 1):
                if metric in metrics and isinstance(metrics[metric], dict):
                    plt.subplot(2, 2, i)

                    # Get values for each k
                    values = []
                    for k in k_values:
                        k_str = str(k)
                        if k_str in metrics[metric]:
                            values.append(metrics[metric][k_str])
                        elif k in metrics[metric]:
                            values.append(metrics[metric][k])
                        else:
                            values.append(0)

                    # Plot
                    plt.plot(k_values, values, 'o-', linewidth=2, markersize=8)
                    plt.title(f'{metric.capitalize()} at different k', fontsize=12)
                    plt.xlabel('k', fontsize=10)
                    plt.ylabel(metric.capitalize(), fontsize=10)
                    plt.xticks(k_values)
                    plt.ylim(0, max(1.0, max(values) * 1.1))  # Ensure upper limit is at least 1.0
                    plt.grid(True, alpha=0.3)

                    # Add value labels
                    for x, y in zip(k_values, values):
                        plt.text(x, y + 0.02, f'{y:.3f}', ha='center', va='bottom', fontsize=9)

            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, 'evaluation_metrics.png'), dpi=300, bbox_inches='tight')
            plt.close()

            # Create coverage visualization
            plt.figure(figsize=(8, 6))

            if 'coverage' in metrics:
                coverage = metrics['coverage']
                plt.bar(['Coverage'], [coverage], color='teal', alpha=0.7)
                plt.title('Recommendation Coverage', fontsize=14)
                plt.ylim(0, 1.1)
                plt.ylabel('Score', fontsize=12)
                plt.grid(axis='y', alpha=0.3)

                # Add value label
                plt.text(0, coverage + 0.05, f'{coverage:.3f}', ha='center', va='bottom', fontsize=12)

                plt.tight_layout()
                plt.savefig(os.path.join(self.output_dir, 'coverage_metric.png'), dpi=300, bbox_inches='tight')

            plt.close()
            logger.info("Evaluation metrics visualizations created successfully")

        except Exception as e:
            logger.error(f"Error creating metrics visualizations: {str(e)}")
            logger.error(traceback.format_exc())

    def visualize_training_history(self, history, model_name='model'):
        """Visualize model training history

        Args:
            history (dict): Training history with loss values
            model_name (str): Name of the model
        """
        logger.info(f"Creating training history visualization for {model_name}...")

        try:
            if 'loss' not in history or not history['loss']:
                logger.warning(f"No loss values found in training history for {model_name}")
                return

            # Plot training loss
            plt.figure(figsize=(10, 6))

            loss_values = history['loss']
            epochs = range(1, len(loss_values) + 1)

            plt.plot(epochs, loss_values, 'b-', linewidth=2)
            plt.title(f'Training Loss for {model_name.capitalize()}', fontsize=14)
            plt.xlabel('Epoch', fontsize=12)
            plt.ylabel('Loss', fontsize=12)
            plt.grid(True, alpha=0.3)

            # Add markers for min loss
            min_loss = min(loss_values)
            min_epoch = loss_values.index(min_loss) + 1
            plt.scatter(min_epoch, min_loss, c='red', s=100, zorder=3)
            plt.annotate(f'Min: {min_loss:.4f}',
                         (min_epoch, min_loss),
                         xytext=(10, -20),
                         textcoords='offset points',
                         arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=.2'))

            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, f'{model_name}_training_history.png'), dpi=300,
                        bbox_inches='tight')
            plt.close()
            logger.info(f"Training history visualization for {model_name} created successfully")

        except Exception as e:
            logger.error(f"Error creating training history visualization: {str(e)}")
            logger.error(traceback.format_exc())

    def visualize_user_recommendations(self, user_id, recommendations, df):
        """Visualize recommendations for a user

        Args:
            user_id: User ID
            recommendations: List of (item_id, score) tuples
            df: DataFrame with item metadata
        """
        logger.info(f"Creating recommendation visualization for user {user_id}...")

        try:
            # Check if recommendations exist
            if not recommendations:
                logger.warning(f"No recommendations found for user {user_id}")
                return

            # Get game information
            games = []
            scores = []

            for game_id, score in recommendations:
                game_data = df[df['app_id'] == game_id]

                if len(game_data) > 0 and 'title' in game_data.columns:
                    title = game_data['title'].iloc[0]
                else:
                    title = f"Game {game_id}"

                games.append(title)
                scores.append(score)

            # Create horizontal bar chart
            plt.figure(figsize=(12, max(6, len(games) * 0.4)))

            # Sort for better visualization
            sorted_indices = np.argsort(scores)
            sorted_games = [games[i] for i in sorted_indices]
            sorted_scores = [scores[i] for i in sorted_indices]

            # Plot
            bars = plt.barh(sorted_games, sorted_scores, alpha=0.7)
            plt.title(f'Top Recommendations for User {user_id}', fontsize=14)
            plt.xlabel('Recommendation Score', fontsize=12)
            plt.xlim(0, max(1.0, max(scores) * 1.1))

            # Add score labels
            for i, (score, bar) in enumerate(zip(sorted_scores, bars)):
                plt.text(score + 0.01, i, f'{score:.3f}', va='center')

            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, f'user_{user_id}_recommendations.png'), dpi=300,
                        bbox_inches='tight')
            plt.close()
            logger.info(f"User recommendation visualization created successfully")

        except Exception as e:
            logger.error(f"Error creating user recommendations visualization: {str(e)}")
            logger.error(traceback.format_exc())

    def visualize_game_similarity(self, game_id, similar_games, df):
        """Visualize game similarity

        Args:
            game_id: Game ID
            similar_games: List of (game_id, similarity) tuples
            df: DataFrame with game metadata
        """
        logger.info(f"Creating game similarity visualization for game {game_id}...")

        try:
            # Check if similar games exist
            if not similar_games:
                logger.warning(f"No similar games found for game {game_id}")
                return

            # Get target game title
            target_game_title = "Unknown Game"
            target_game_data = df[df['app_id'] == game_id]
            if len(target_game_data) > 0 and 'title' in target_game_data.columns:
                target_game_title = target_game_data['title'].iloc[0]

            # Get similar game information
            titles = []
            similarities = []

            for similar_id, similarity in similar_games:
                game_data = df[df['app_id'] == similar_id]

                if len(game_data) > 0 and 'title' in game_data.columns:
                    title = game_data['title'].iloc[0]
                else:
                    title = f"Game {similar_id}"

                titles.append(title)
                similarities.append(similarity)

            # Create horizontal bar chart
            plt.figure(figsize=(12, max(6, len(titles) * 0.4)))

            # Sort for better visualization
            sorted_indices = np.argsort(similarities)
            sorted_titles = [titles[i] for i in sorted_indices]
            sorted_similarities = [similarities[i] for i in sorted_indices]

            # Plot
            bars = plt.barh(sorted_titles, sorted_similarities, alpha=0.7)
            plt.title(f'Games Similar to "{target_game_title}"', fontsize=14)
            plt.xlabel('Similarity Score', fontsize=12)
            plt.xlim(0, max(1.0, max(similarities) * 1.1))

            # Add score labels
            for i, (similarity, bar) in enumerate(zip(sorted_similarities, bars)):
                plt.text(similarity + 0.01, i, f'{similarity:.3f}', va='center')

            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, f'game_{game_id}_similarity.png'), dpi=300, bbox_inches='tight')
            plt.close()
            logger.info(f"Game similarity visualization created successfully")

        except Exception as e:
            logger.error(f"Error creating game similarity visualization: {str(e)}")
            logger.error(traceback.format_exc())

    def visualize_feature_importance(self, importance_df, model_name='model'):
        """Visualize feature importance

        Args:
            importance_df: DataFrame with Feature and Importance columns
            model_name: Name of the model
        """
        logger.info(f"Creating feature importance visualization for {model_name}...")

        try:
            # Check if feature importance exists
            if not isinstance(importance_df,
                              pd.DataFrame) or 'Feature' not in importance_df.columns or 'Importance' not in importance_df.columns:
                logger.warning(f"Invalid feature importance data for {model_name}")
                return

            # Sort features by importance
            sorted_importance = importance_df.sort_values('Importance', ascending=False)

            # Limit to top 20 features
            if len(sorted_importance) > 20:
                sorted_importance = sorted_importance.head(20)

            # Create horizontal bar chart
            plt.figure(figsize=(12, max(6, len(sorted_importance) * 0.4)))

            # Plot
            plt.barh(sorted_importance['Feature'], sorted_importance['Importance'], alpha=0.7)
            plt.title(f'Feature Importance for {model_name.capitalize()}', fontsize=14)
            plt.xlabel('Importance', fontsize=12)

            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, f'{model_name}_feature_importance.png'), dpi=300,
                        bbox_inches='tight')
            plt.close()
            logger.info(f"Feature importance visualization for {model_name} created successfully")

        except Exception as e:
            logger.error(f"Error creating feature importance visualization: {str(e)}")
            logger.error(traceback.format_exc())

    def visualize_user_game_matrix(self, user_game_matrix, max_size=1000, title=None):
        """Visualize user-game interaction matrix sparsity

        Args:
            user_game_matrix: User-item matrix (users as rows, items as columns)
            max_size: Maximum size to display (will sample if larger)
            title: Optional title for the plot
        """
        logger.info("Creating user-game matrix visualization...")

        try:
            # Check if matrix exists
            if user_game_matrix is None or not isinstance(user_game_matrix, pd.DataFrame):
                logger.warning("Invalid user-game matrix")
                return

            # Sample matrix if too large
            matrix = user_game_matrix
            if matrix.shape[0] > max_size or matrix.shape[1] > max_size:
                logger.info(f"Matrix too large ({matrix.shape}), sampling to max size {max_size}...")

                # Sample users and items
                if matrix.shape[0] > max_size:
                    users = np.random.choice(matrix.index, max_size, replace=False)
                    matrix = matrix.loc[users]

                if matrix.shape[1] > max_size:
                    items = np.random.choice(matrix.columns, max_size, replace=False)
                    matrix = matrix[items]

            # Create figure
            plt.figure(figsize=(10, 8))

            # Calculate and display sparsity
            total_cells = matrix.shape[0] * matrix.shape[1]
            non_zero_cells = np.count_nonzero(matrix.values)
            sparsity = 1 - (non_zero_cells / total_cells)

            # Plot matrix
            plt.spy(matrix, markersize=0.5, aspect='auto')

            # Set title and labels
            if title:
                plt.title(f"{title}\nSparsity: {sparsity:.2%}")
            else:
                plt.title(f"User-Game Matrix Sparsity: {sparsity:.2%}")

            plt.xlabel(f"Games ({matrix.shape[1]})")
            plt.ylabel(f"Users ({matrix.shape[0]})")

            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, 'user_game_matrix_sparsity.png'), dpi=300, bbox_inches='tight')
            plt.close()
            logger.info("User-game matrix visualization created successfully")

        except Exception as e:
            logger.error(f"Error creating user-game matrix visualization: {str(e)}")
            logger.error(traceback.format_exc())

    def visualize_tag_distribution(self, df, top_n=20):
        """Visualize distribution of game tags

        Args:
            df: DataFrame with game data including 'tags' column
            top_n: Number of top tags to display
        """
        logger.info("Creating tag distribution visualization...")

        try:
            # Check if tags exist
            if 'tags' not in df.columns:
                logger.warning("Tags column not found in DataFrame")
                return

            # Extract all tags
            all_tags = []
            for tags_str in df['tags'].dropna():
                if isinstance(tags_str, str):
                    all_tags.extend([tag.strip() for tag in tags_str.split(',')])

            # Count tag frequencies
            tag_counts = pd.Series(all_tags).value_counts()

            # Get top N tags
            top_tags = tag_counts.head(top_n)

            # Create bar chart
            plt.figure(figsize=(12, 8))

            # Plot
            plt.bar(top_tags.index, top_tags.values, alpha=0.7)
            plt.title(f'Top {top_n} Game Tags', fontsize=14)
            plt.xlabel('Tag', fontsize=12)
            plt.ylabel('Frequency', fontsize=12)
            plt.xticks(rotation=45, ha='right')
            plt.grid(axis='y', alpha=0.3)

            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, 'tag_distribution.png'), dpi=300, bbox_inches='tight')
            plt.close()
            logger.info("Tag distribution visualization created successfully")

        except Exception as e:
            logger.error(f"Error creating tag distribution visualization: {str(e)}")
            logger.error(traceback.format_exc())

    def visualize_user_activity(self, df, user_id=None, max_users=10):
        """Visualize user activity patterns

        Args:
            df: DataFrame with user interaction data
            user_id: Specific user to visualize (if None, shows top active users)
            max_users: Maximum number of users to display
        """
        logger.info("Creating user activity visualization...")

        try:
            # Create user activity summary
            user_activity = df.groupby('user_id').agg({
                'app_id': 'count',
                'is_recommended': 'mean' if 'is_recommended' in df.columns else None,
                'hours': 'sum' if 'hours' in df.columns else None
            }).reset_index()

            user_activity.columns = ['user_id', 'interaction_count',
                                     'recommendation_ratio' if 'is_recommended' in df.columns else None,
                                     'total_hours' if 'hours' in df.columns else None]

            # Remove None columns
            user_activity = user_activity.loc[:, ~user_activity.columns.isnull()]

            if user_id is not None:
                # Visualize specific user
                if user_id not in user_activity['user_id'].values:
                    logger.warning(f"User {user_id} not found in data")
                    return

                # Get user's game interactions
                user_games = df[df['user_id'] == user_id]

                # If date column exists, visualize activity over time
                if 'date' in user_games.columns:
                    # Convert to datetime if needed
                    if not pd.api.types.is_datetime64_dtype(user_games['date']):
                        user_games['date'] = pd.to_datetime(user_games['date'])

                    # Aggregate by month
                    user_games['month'] = user_games['date'].dt.to_period('M')
                    monthly_activity = user_games.groupby('month').size()

                    # Plot
                    plt.figure(figsize=(12, 6))
                    monthly_activity.plot(kind='bar', alpha=0.7)
                    plt.title(f'Monthly Activity for User {user_id}', fontsize=14)
                    plt.xlabel('Month', fontsize=12)
                    plt.ylabel('Number of Interactions', fontsize=12)
                    plt.xticks(rotation=45)
                    plt.grid(axis='y', alpha=0.3)

                    plt.tight_layout()
                    plt.savefig(os.path.join(self.output_dir, f'user_{user_id}_monthly_activity.png'), dpi=300,
                                bbox_inches='tight')
                    plt.close()

                # Visualize hours distribution if available
                if 'hours' in user_games.columns:
                    plt.figure(figsize=(10, 6))
                    sns.histplot(user_games['hours'], bins=20, kde=True)
                    plt.title(f'Hours Distribution for User {user_id}', fontsize=14)
                    plt.xlabel('Hours', fontsize=12)
                    plt.ylabel('Frequency', fontsize=12)
                    plt.grid(alpha=0.3)

                    plt.tight_layout()
                    plt.savefig(os.path.join(self.output_dir, f'user_{user_id}_hours_distribution.png'), dpi=300,
                                bbox_inches='tight')
                    plt.close()
            else:
                # Visualize top active users
                top_users = user_activity.sort_values('interaction_count', ascending=False).head(max_users)

                plt.figure(figsize=(12, 6))
                plt.bar(top_users['user_id'].astype(str), top_users['interaction_count'], alpha=0.7)
                plt.title(f'Top {max_users} Most Active Users', fontsize=14)
                plt.xlabel('User ID', fontsize=12)
                plt.ylabel('Number of Interactions', fontsize=12)
                plt.xticks(rotation=45)
                plt.grid(axis='y', alpha=0.3)

                plt.tight_layout()
                plt.savefig(os.path.join(self.output_dir, 'top_active_users.png'), dpi=300, bbox_inches='tight')
                plt.close()

                # Visualize hours vs interactions if hours available
                if 'total_hours' in top_users.columns:
                    plt.figure(figsize=(10, 8))
                    plt.scatter(top_users['interaction_count'], top_users['total_hours'], alpha=0.7, s=100)

                    # Add labels for each user
                    for i, user in top_users.iterrows():
                        plt.annotate(str(user['user_id']),
                                     (user['interaction_count'], user['total_hours']),
                                     xytext=(5, 5),
                                     textcoords='offset points')

                    plt.title('Hours vs Interactions for Top Users', fontsize=14)
                    plt.xlabel('Number of Interactions', fontsize=12)
                    plt.ylabel('Total Hours', fontsize=12)
                    plt.grid(alpha=0.3)

                    plt.tight_layout()
                    plt.savefig(os.path.join(self.output_dir, 'hours_vs_interactions.png'), dpi=300,
                                bbox_inches='tight')
                    plt.close()

            logger.info("User activity visualization created successfully")

        except Exception as e:
            logger.error(f"Error creating user activity visualization: {str(e)}")
            logger.error(traceback.format_exc())

    # !/usr/bin/env python
    # -*- coding: utf-8 -*-

    def visualize_knn_optimization(self, knn_results):
        """Visualize KNN parameter optimization results

        Args:
            knn_results (dict): Dictionary mapping parameter configs to metrics
        """
        logger.info("Creating KNN parameter optimization visualization...")

        try:
            if not knn_results:
                logger.warning("No KNN optimization results to visualize")
                return

            import numpy as np

            # Extract parameter combinations and NDCG values
            user_neighbors = []
            item_neighbors = []
            ndcg_values = []

            for config_key, metrics in knn_results.items():
                # Parse config key format "user_X_item_Y"
                parts = config_key.split('_')
                if len(parts) >= 4:
                    user_n = int(parts[1])
                    item_n = int(parts[3])
                    if isinstance(metrics, dict) and 'ndcg' in metrics and 10 in metrics['ndcg']:
                        ndcg = metrics['ndcg'][10]  # Use NDCG@10

                        user_neighbors.append(user_n)
                        item_neighbors.append(item_n)
                        ndcg_values.append(ndcg)

            # Get unique neighbor values
            unique_user_n = sorted(set(user_neighbors))
            unique_item_n = sorted(set(item_neighbors))

            # Create heatmap matrix
            heatmap_data = np.zeros((len(unique_user_n), len(unique_item_n)))

            # Fill heatmap data
            for u, i, ndcg in zip(user_neighbors, item_neighbors, ndcg_values):
                u_idx = unique_user_n.index(u)
                i_idx = unique_item_n.index(i)
                heatmap_data[u_idx, i_idx] = ndcg

            # Create heatmap
            plt.figure(figsize=(12, 10))
            sns.heatmap(
                heatmap_data,
                annot=True,
                fmt=".4f",
                cmap="viridis",
                xticklabels=unique_item_n,
                yticklabels=unique_user_n,
                cbar_kws={'label': 'NDCG@10'}
            )

            plt.title('KNN Parameter Optimization - NDCG@10', fontsize=16)
            plt.xlabel('Item Neighbors', fontsize=14)
            plt.ylabel('User Neighbors', fontsize=14)

            # Save chart
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, 'knn_parameter_optimization.png'), dpi=300, bbox_inches='tight')
            plt.close()

            # Create line chart - Impact of user neighbors
            plt.figure(figsize=(12, 6))

            # Plot a line for each item_n value
            for item_n in unique_item_n:
                item_ndcg = []
                for user_n in unique_user_n:
                    try:
                        config_key = f"user_{user_n}_item_{item_n}"
                        if config_key in knn_results and 'ndcg' in knn_results[config_key]:
                            item_ndcg.append(knn_results[config_key]['ndcg'][10])
                        else:
                            item_ndcg.append(None)  # Missing data
                    except:
                        item_ndcg.append(None)

                # Plot line, ignoring missing values
                valid_indices = [i for i, v in enumerate(item_ndcg) if v is not None]
                valid_user_n = [unique_user_n[i] for i in valid_indices]
                valid_ndcg = [item_ndcg[i] for i in valid_indices]

                if valid_ndcg:
                    plt.plot(valid_user_n, valid_ndcg, marker='o', label=f'Item Neighbors={item_n}')

            plt.title('Impact of User Neighbors on NDCG@10', fontsize=16)
            plt.xlabel('User Neighbors', fontsize=14)
            plt.ylabel('NDCG@10', fontsize=14)
            plt.grid(True, alpha=0.3)
            plt.legend()

            plt.savefig(os.path.join(self.output_dir, 'user_neighbors_impact.png'), dpi=300, bbox_inches='tight')
            plt.close()

            # Create line chart - Impact of item neighbors
            plt.figure(figsize=(12, 6))

            # Plot a line for each user_n value
            for user_n in unique_user_n:
                user_ndcg = []
                for item_n in unique_item_n:
                    try:
                        config_key = f"user_{user_n}_item_{item_n}"
                        if config_key in knn_results and 'ndcg' in knn_results[config_key]:
                            user_ndcg.append(knn_results[config_key]['ndcg'][10])
                        else:
                            user_ndcg.append(None)  # Missing data
                    except:
                        user_ndcg.append(None)

                # Plot line, ignoring missing values
                valid_indices = [i for i, v in enumerate(user_ndcg) if v is not None]
                valid_item_n = [unique_item_n[i] for i in valid_indices]
                valid_ndcg = [user_ndcg[i] for i in valid_indices]

                if valid_ndcg:
                    plt.plot(valid_item_n, valid_ndcg, marker='o', label=f'User Neighbors={user_n}')

            plt.title('Impact of Item Neighbors on NDCG@10', fontsize=16)
            plt.xlabel('Item Neighbors', fontsize=14)
            plt.ylabel('NDCG@10', fontsize=14)
            plt.grid(True, alpha=0.3)
            plt.legend()

            plt.savefig(os.path.join(self.output_dir, 'item_neighbors_impact.png'), dpi=300, bbox_inches='tight')
            plt.close()

            # Find and highlight best parameters
            best_ndcg = np.max(heatmap_data)
            best_indices = np.where(heatmap_data == best_ndcg)
            best_user_n = unique_user_n[best_indices[0][0]]
            best_item_n = unique_item_n[best_indices[1][0]]

            # Create summary figure
            plt.figure(figsize=(8, 6))

            # Create a simple bar chart with the best configuration
            plt.bar(['Best Configuration'], [best_ndcg], color='teal', alpha=0.7)
            plt.title(f'Best KNN Configuration: User={best_user_n}, Item={best_item_n}', fontsize=14)
            plt.ylabel('NDCG@10', fontsize=12)
            plt.ylim(0, min(1.0, best_ndcg * 1.2))  # Set reasonable y limit

            # Add value label
            plt.text(0, best_ndcg + 0.02, f'{best_ndcg:.4f}', ha='center', va='bottom', fontsize=12)

            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, 'knn_best_configuration.png'), dpi=300, bbox_inches='tight')
            plt.close()

            logger.info("KNN parameter optimization visualization created successfully")

        except Exception as e:
            logger.error(f"Error creating KNN parameter visualization: {str(e)}")
            logger.error(traceback.format_exc())