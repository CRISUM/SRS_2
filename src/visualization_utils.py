#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Steam Game Recommendation System - Visualization Utilities
Date: 2025-04-26
Description: Utility functions for visualizing recommender system training, evaluation, and results
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.ticker as mtick
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import logging
import os
import json
import time
from datetime import datetime

logger = logging.getLogger(__name__)

# Set plot style
plt.style.use('ggplot')
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']


def plot_training_history(history, metrics=None, save_path=None):
    """
    Plot training history metrics over epochs

    Parameters:
        history (dict): Dictionary containing training history (loss, metrics per epoch)
        metrics (list): List of metrics to plot (default: all metrics in history)
        save_path (str): Path to save the figure (default: None, just display)
    """
    if not history:
        logger.warning("No training history provided")
        return

    if metrics is None:
        # Plot all metrics except validation ones
        metrics = [m for m in history.keys() if not m.startswith('val_')]

    n_metrics = len(metrics)
    if n_metrics == 0:
        logger.warning("No metrics to plot")
        return

    # Create subplots
    fig, axes = plt.subplots(1, n_metrics, figsize=(5 * n_metrics, 5))
    if n_metrics == 1:
        axes = [axes]

    # Plot each metric
    for i, metric in enumerate(metrics):
        ax = axes[i]
        ax.plot(history[metric], label=f'Training {metric}', marker='o')

        # Plot validation metric if available
        val_metric = f'val_{metric}'
        if val_metric in history:
            ax.plot(history[val_metric], label=f'Validation {metric}', marker='x')

        ax.set_title(f'{metric.capitalize()} Over Epochs')
        ax.set_xlabel('Epoch')
        ax.set_ylabel(metric)
        ax.legend()
        ax.grid(True)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Training history plot saved to {save_path}")
    else:
        plt.show()

    plt.close()


def plot_knn_evaluation(distances, optimal_k=None, save_path=None):
    """
    Plot KNN model evaluation to help determine optimal K value

    Parameters:
        distances (dict): Dictionary with k values as keys and average distances as values
        optimal_k (int): Optimal K value to highlight (default: None)
        save_path (str): Path to save the figure (default: None, just display)
    """
    if not distances:
        logger.warning("No KNN distance data provided")
        return

    k_values = list(distances.keys())
    avg_distances = list(distances.values())

    plt.figure(figsize=(10, 6))
    plt.plot(k_values, avg_distances, marker='o', linestyle='-', linewidth=2)

    if optimal_k is not None:
        # Highlight optimal K
        opt_dist = distances.get(optimal_k)
        if opt_dist is not None:
            plt.scatter([optimal_k], [opt_dist], color='red', s=100, zorder=5)
            plt.annotate(f'Optimal k={optimal_k}',
                         xy=(optimal_k, opt_dist),
                         xytext=(optimal_k + 2, opt_dist),
                         arrowprops=dict(facecolor='red', shrink=0.05))

    plt.title('Average Distance vs. Number of Neighbors (k)')
    plt.xlabel('Number of Neighbors (k)')
    plt.ylabel('Average Distance')
    plt.grid(True)
    plt.xticks(k_values)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"KNN evaluation plot saved to {save_path}")
    else:
        plt.show()

    plt.close()


def plot_evaluation_metrics(metrics, k_values=None, save_path=None):
    """
    Plot evaluation metrics for recommendation system

    Parameters:
        metrics (dict): Dictionary with metrics and their values
        k_values (list): K values for which metrics were computed
        save_path (str): Path to save the figure (default: None, just display)
    """
    if not metrics:
        logger.warning("No metrics provided for plotting")
        return

    # If k_values not specified but metrics contain k values
    if k_values is None and 'precision' in metrics and isinstance(metrics['precision'], dict):
        k_values = sorted(metrics['precision'].keys())

    # Prepare data for plotting
    if k_values is not None:
        # Multi-k plot (precision@k, recall@k, etc.)
        metrics_to_plot = ['precision', 'recall', 'ndcg', 'diversity']
        available_metrics = [m for m in metrics_to_plot if m in metrics]

        n_metrics = len(available_metrics)
        if n_metrics == 0:
            logger.warning("No valid metrics found for plotting")
            return

        fig, axes = plt.subplots(2, (n_metrics + 1) // 2, figsize=(15, 10))
        axes = axes.flatten()

        for i, metric in enumerate(available_metrics):
            if i < len(axes):
                ax = axes[i]

                # Extract values for each k
                values = [metrics[metric][k] for k in k_values]

                ax.plot(k_values, values, marker='o', linestyle='-', linewidth=2)
                ax.set_title(f'{metric.capitalize()} at different k')
                ax.set_xlabel('k')
                ax.set_ylabel(metric)
                ax.grid(True)

                # Add data labels
                for x, y in zip(k_values, values):
                    ax.annotate(f"{y:.3f}", (x, y), textcoords="offset points",
                                xytext=(0, 10), ha='center')

        # If coverage is available, add it to the last subplot
        if 'coverage' in metrics and i + 1 < len(axes):
            ax = axes[i + 1]
            ax.bar(['Coverage'], [metrics['coverage']], color=colors[0])
            ax.set_title('Coverage')
            ax.set_ylabel('Value')
            ax.set_ylim(0, min(1.0, metrics['coverage'] * 1.2))
            ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))

            # Add data label
            ax.annotate(f"{metrics['coverage']:.3f}", (0, metrics['coverage']),
                        textcoords="offset points", xytext=(0, 10), ha='center')

        # Remove any unused subplots
        for j in range(i + 2, len(axes)):
            fig.delaxes(axes[j])

    else:
        # Single value metrics
        metrics_names = list(metrics.keys())
        metrics_values = list(metrics.values())

        plt.figure(figsize=(10, 6))
        bars = plt.bar(metrics_names, metrics_values, color=colors[:len(metrics_names)])

        plt.title('Recommendation System Evaluation Metrics')
        plt.xlabel('Metric')
        plt.ylabel('Value')
        plt.ylim(0, max(metrics_values) * 1.2)

        # Add data labels
        for bar in bars:
            height = bar.get_height()
            plt.annotate(f"{height:.3f}",
                         xy=(bar.get_x() + bar.get_width() / 2, height),
                         xytext=(0, 3),  # 3 points vertical offset
                         textcoords="offset points",
                         ha='center', va='bottom')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Evaluation metrics plot saved to {save_path}")
    else:
        plt.show()

    plt.close()


def visualize_user_item_embeddings(embeddings, labels=None, n_samples=1000, method='tsne', save_path=None):
    """
    Visualize user or item embeddings in 2D space

    Parameters:
        embeddings (numpy.ndarray): Embeddings matrix
        labels (list): Labels for coloring points (e.g., item categories)
        n_samples (int): Number of samples to visualize (default: 1000)
        method (str): Dimensionality reduction method ('pca' or 'tsne')
        save_path (str): Path to save the figure (default: None, just display)
    """
    if embeddings is None or len(embeddings) == 0:
        logger.warning("No embeddings provided for visualization")
        return

    # Sample if there are too many points
    if len(embeddings) > n_samples:
        indices = np.random.choice(len(embeddings), n_samples, replace=False)
        sample_embeddings = embeddings[indices]
        if labels is not None:
            sample_labels = [labels[i] for i in indices]
        else:
            sample_labels = None
    else:
        sample_embeddings = embeddings
        sample_labels = labels

    # Reduce to 2D
    logger.info(f"Reducing {len(sample_embeddings)} embeddings to 2D using {method.upper()}")
    start_time = time.time()

    if method.lower() == 'tsne':
        reducer = TSNE(n_components=2, random_state=42, n_jobs=-1)
    else:  # default to PCA
        reducer = PCA(n_components=2, random_state=42)

    embedding_2d = reducer.fit_transform(sample_embeddings)

    logger.info(f"Dimensionality reduction completed in {time.time() - start_time:.2f} seconds")

    # Create plot
    plt.figure(figsize=(12, 10))

    if sample_labels is not None:
        # Color by label
        unique_labels = list(set(sample_labels))
        cmap = plt.cm.get_cmap('tab10', len(unique_labels))

        for i, label in enumerate(unique_labels):
            idx = [j for j, l in enumerate(sample_labels) if l == label]
            plt.scatter(embedding_2d[idx, 0], embedding_2d[idx, 1],
                        c=[cmap(i)], label=label, alpha=0.7)

        plt.legend(title="Categories", bbox_to_anchor=(1.05, 1), loc='upper left')
    else:
        # Single color
        plt.scatter(embedding_2d[:, 0], embedding_2d[:, 1], alpha=0.7)

    plt.title(f'2D {method.upper()} Projection of Embeddings')
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    plt.grid(True)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Embeddings visualization saved to {save_path}")
    else:
        plt.show()

    plt.close()

    return embedding_2d  # Return 2D embeddings for potential further use


def plot_similarity_heatmap(similarity_matrix, labels=None, top_n=50, save_path=None):
    """
    Plot a heatmap of the similarity matrix

    Parameters:
        similarity_matrix (numpy.ndarray): Similarity matrix
        labels (list): Item labels for the axes
        top_n (int): Number of items to include (default: 50)
        save_path (str): Path to save the figure (default: None, just display)
    """
    if similarity_matrix is None:
        logger.warning("No similarity matrix provided")
        return

    # Sample if matrix is too large
    if similarity_matrix.shape[0] > top_n:
        # Select top_n items with highest average similarity
        avg_sim = np.mean(similarity_matrix, axis=1)
        top_indices = np.argsort(-avg_sim)[:top_n]
        sample_matrix = similarity_matrix[top_indices][:, top_indices]

        if labels is not None:
            sample_labels = [labels[i] for i in top_indices]
        else:
            sample_labels = [f"Item {i}" for i in top_indices]
    else:
        sample_matrix = similarity_matrix
        if labels is not None:
            sample_labels = labels
        else:
            sample_labels = [f"Item {i}" for i in range(len(similarity_matrix))]

    # Create heatmap
    plt.figure(figsize=(12, 10))
    sns.heatmap(sample_matrix, annot=False, cmap='viridis',
                xticklabels=sample_labels, yticklabels=sample_labels)

    plt.title('Item Similarity Heatmap')
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Similarity heatmap saved to {save_path}")
    else:
        plt.show()

    plt.close()


def plot_user_game_matrix_sparsity(matrix, save_path=None):
    """
    Visualize the sparsity of the user-game interaction matrix

    Parameters:
        matrix: User-game interaction matrix (pandas DataFrame or sparse matrix)
        save_path (str): Path to save the figure (default: None, just display)
    """
    if matrix is None:
        logger.warning("No matrix provided")
        return

    # If matrix is too large, sample it
    if isinstance(matrix, pd.DataFrame):
        if matrix.shape[0] > 1000 or matrix.shape[1] > 1000:
            sample_rows = min(1000, matrix.shape[0])
            sample_cols = min(1000, matrix.shape[1])

            # Sample rows and columns
            row_indices = np.random.choice(matrix.shape[0], sample_rows, replace=False)
            col_indices = np.random.choice(matrix.shape[1], sample_cols, replace=False)

            sample_matrix = matrix.iloc[row_indices, col_indices].values
        else:
            sample_matrix = matrix.values
    else:
        # Assume it's already a numpy array or similar
        if matrix.shape[0] > 1000 or matrix.shape[1] > 1000:
            sample_rows = min(1000, matrix.shape[0])
            sample_cols = min(1000, matrix.shape[1])

            # Sample rows and columns
            row_indices = np.random.choice(matrix.shape[0], sample_rows, replace=False)
            col_indices = np.random.choice(matrix.shape[1], sample_cols, replace=False)

            sample_matrix = matrix[row_indices][:, col_indices]
        else:
            sample_matrix = matrix

    # Calculate sparsity
    sparsity = 1.0 - (np.count_nonzero(sample_matrix) / float(sample_matrix.size))

    plt.figure(figsize=(10, 8))
    plt.spy(sample_matrix, markersize=0.1, aspect='auto')

    plt.title(f'User-Game Matrix Sparsity: {sparsity:.2%}')
    plt.xlabel('Games')
    plt.ylabel('Users')

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Matrix sparsity plot saved to {save_path}")
    else:
        plt.show()

    plt.close()

    return sparsity


def visualize_recommendation_diversity(recommendations, game_data, user_id, save_path=None):
    """
    Visualize diversity of recommendations for a user

    Parameters:
        recommendations (list): List of (game_id, score) tuples
        game_data (DataFrame): Game metadata including tags, genres, etc.
        user_id: User ID for which recommendations were generated
        save_path (str): Path to save the figure (default: None, just display)
    """
    if not recommendations or game_data is None:
        logger.warning("No recommendations or game data provided")
        return

    # Extract game IDs
    game_ids = [game_id for game_id, _ in recommendations]

    # Get game information
    game_info = game_data[game_data['app_id'].isin(game_ids)].copy()

    if 'tags' not in game_info.columns:
        logger.warning("Game tags not available for diversity visualization")
        return

    # Process tags
    all_tags = []
    for tags_str in game_info['tags'].dropna():
        if isinstance(tags_str, str):
            all_tags.extend([tag.strip() for tag in tags_str.split(',')])

    # Count tags
    tag_counts = pd.Series(all_tags).value_counts()
    top_tags = tag_counts.head(10)

    # Plot tag distribution
    plt.figure(figsize=(12, 8))
    bars = plt.bar(top_tags.index, top_tags.values, color=colors[:len(top_tags)])

    plt.title(f'Tag Distribution in Recommendations for User {user_id}')
    plt.xlabel('Tags')
    plt.ylabel('Count')
    plt.xticks(rotation=45, ha='right')

    # Add counts above bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2., height + 0.1,
                 f'{int(height)}', ha='center', va='bottom')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Recommendation diversity plot saved to {save_path}")
    else:
        plt.show()

    plt.close()

    # Create a second plot for tag similarity network
    if len(game_ids) > 1:
        plt.figure(figsize=(12, 10))

        # Create tag sets for each game
        game_tags = {}
        for _, row in game_info.iterrows():
            if pd.notna(row['tags']):
                game_tags[row['app_id']] = set(tag.strip() for tag in row['tags'].split(','))
            else:
                game_tags[row['app_id']] = set()

        # Calculate tag similarity between games
        similarity_matrix = np.zeros((len(game_ids), len(game_ids)))

        for i, game1 in enumerate(game_ids):
            for j, game2 in enumerate(game_ids):
                if i != j and game1 in game_tags and game2 in game_tags:
                    # Jaccard similarity
                    similarity = len(game_tags[game1] & game_tags[game2]) / len(
                        game_tags[game1] | game_tags[game2]) if len(game_tags[game1] | game_tags[game2]) > 0 else 0
                    similarity_matrix[i, j] = similarity

        # Plot similarity heatmap
        game_titles = []
        for game_id in game_ids:
            title_row = game_info[game_info['app_id'] == game_id]['title']
            if not title_row.empty:
                # Truncate long titles
                title = title_row.iloc[0]
                if len(title) > 20:
                    title = title[:17] + '...'
                game_titles.append(title)
            else:
                game_titles.append(f"Game {game_id}")

        sns.heatmap(similarity_matrix, annot=True, cmap='viridis', fmt='.2f',
                    xticklabels=game_titles, yticklabels=game_titles)

        plt.title(f'Tag Similarity Between Recommended Games for User {user_id}')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()

        if save_path:
            similarity_path = save_path.replace('.png', '_similarity.png')
            plt.savefig(similarity_path, dpi=300, bbox_inches='tight')
            logger.info(f"Recommendation similarity matrix saved to {similarity_path}")
        else:
            plt.show()

        plt.close()


def create_training_dashboard(history, evaluation_metrics, feature_importance=None, save_dir='visualizations'):
    """
    Create a comprehensive dashboard of the training process and results

    Parameters:
        history (dict): Training history
        evaluation_metrics (dict): Model evaluation metrics
        feature_importance (DataFrame): Feature importance data
        save_dir (str): Directory to save dashboard HTML file
    """
    os.makedirs(save_dir, exist_ok=True)

    # Create a plotly dashboard
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Training Progress', 'Evaluation Metrics',
                        'Feature Importance', 'Model Performance')
    )

    # 1. Training Progress
    if history:
        epochs = list(range(1, len(history.get('loss', [])) + 1))
        metrics = [m for m in history.keys() if not m.startswith('val_')]

        for metric in metrics:
            fig.add_trace(
                go.Scatter(x=epochs, y=history[metric], mode='lines+markers', name=f'Train {metric}'),
                row=1, col=1
            )

            # Add validation metrics if available
            val_metric = f'val_{metric}'
            if val_metric in history:
                fig.add_trace(
                    go.Scatter(x=epochs, y=history[val_metric], mode='lines+markers',
                               name=f'Validation {metric}', line=dict(dash='dash')),
                    row=1, col=1
                )

    # 2. Evaluation Metrics
    if evaluation_metrics:
        if 'precision' in evaluation_metrics and isinstance(evaluation_metrics['precision'], dict):
            # Multi-k evaluation
            k_values = sorted(evaluation_metrics['precision'].keys())
            metrics_to_plot = ['precision', 'recall', 'ndcg']

            for metric in metrics_to_plot:
                if metric in evaluation_metrics:
                    values = [evaluation_metrics[metric][k] for k in k_values]
                    fig.add_trace(
                        go.Scatter(x=k_values, y=values, mode='lines+markers', name=f'{metric.capitalize()}@k'),
                        row=1, col=2
                    )
        else:
            # Single-value metrics
            metrics_names = list(evaluation_metrics.keys())
            metrics_values = list(evaluation_metrics.values())

            fig.add_trace(
                go.Bar(x=metrics_names, y=metrics_values, name='Metrics'),
                row=1, col=2
            )

    # 3. Feature Importance
    if feature_importance is not None and not feature_importance.empty:
        top_features = feature_importance.head(10)

        fig.add_trace(
            go.Bar(
                y=top_features['Feature'],
                x=top_features['Importance'],
                orientation='h',
                name='Feature Importance'
            ),
            row=2, col=1
        )

    # 4. Model Performance (ROC Curve, confusion matrix, etc.)
    # This would depend on specific performance data available
    # For now, add a placeholder
    fig.add_trace(
        go.Scatter(
            x=[0, 0.2, 0.4, 0.6, 0.8, 1.0],
            y=[0, 0.3, 0.5, 0.7, 0.9, 1.0],
            mode='lines',
            name='ROC Curve (placeholder)'
        ),
        row=2, col=2
    )

    # Update layout
    fig.update_layout(
        title_text='Steam Game Recommendation System Training Dashboard',
        height=900,
        width=1200,
        showlegend=True
    )

    # Save dashboard
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    dashboard_path = os.path.join(save_dir, f'training_dashboard_{timestamp}.html')
    fig.write_html(dashboard_path)

    logger.info(f"Training dashboard created and saved to {dashboard_path}")
    return dashboard_path


def plot_model_comparison(models_metrics, metrics=None, save_path=None):
    """
    Plot comparison of multiple models

    Parameters:
        models_metrics (dict): Dictionary with model names as keys and metrics as values
        metrics (list): List of metrics to include in comparison
        save_path (str): Path to save the figure (default: None, just display)
    """
    if not models_metrics:
        logger.warning("No model metrics provided for comparison")
        return

    model_names = list(models_metrics.keys())

    if metrics is None:
        # Find common metrics across all models
        all_metrics = set()
        for model_metrics in models_metrics.values():
            all_metrics.update(model_metrics.keys())

        # Filter for numeric metrics
        metrics = []
        for metric in all_metrics:
            all_numeric = True
            for model_metrics in models_metrics.values():
                if metric in model_metrics:
                    val = model_metrics[metric]
                    if isinstance(val, dict):  # Handle metrics@k
                        # Take mean across k values
                        all_numeric = all(isinstance(v, (int, float)) for v in val.values())
                    elif not isinstance(val, (int, float)):
                        all_numeric = False

            if all_numeric:
                metrics.append(metric)

    if not metrics:
        logger.warning("No valid metrics found for comparison")
        return

    # Prepare data for plotting
    comparison_data = []

    for model_name, model_metrics in models_metrics.items():
        for metric in metrics:
            if metric in model_metrics:
                val = model_metrics[metric]
                if isinstance(val, dict):  # metrics@k
                    # Average across k values
                    avg_val = sum(val.values()) / len(val)
                    comparison_data.append({
                        'Model': model_name,
                        'Metric': f"avg_{metric}",
                        'Value': avg_val
                    })

                    # Also add individual k values
                    for k, k_val in val.items():
                        comparison_data.append({
                            'Model': model_name,
                            'Metric': f"{metric}@{k}",
                            'Value': k_val
                        })
                else:
                    comparison_data.append({
                        'Model': model_name,
                        'Metric': metric,
                        'Value': val
                    })

    if not comparison_data:
        logger.warning("No data available for comparison after processing")
        return

    # Convert to DataFrame
    df = pd.DataFrame(comparison_data)

    # Plot
    plt.figure(figsize=(15, 10))

    # Group by metric
    for i, metric in enumerate(df['Metric'].unique()):
        plt.subplot(2, (len(df['Metric'].unique()) + 1) // 2, i + 1)

        metric_data = df[df['Metric'] == metric]
        bars = plt.bar(metric_data['Model'], metric_data['Value'], color=colors[:len(metric_data)])

        plt.title(metric)
        plt.ylabel('Value')
        plt.xticks(rotation=45, ha='right')

        # Add data labels
        for bar in bars:
            height = bar.get_height()
            plt.annotate(f"{height:.3f}",
                         xy=(bar.get_x() + bar.get_width() / 2, height),
                         xytext=(0, 3),  # 3 points vertical offset
                         textcoords="offset points",
                         ha='center', va='bottom')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Model comparison plot saved to {save_path}")
    else:
        plt.show()

    plt.close()

    # Create interactive visualization with Plotly
    fig = px.bar(df, x='Model', y='Value', color='Model', facet_col='Metric',
                 barmode='group', title='Model Comparison')

    fig.update_layout(
        height=600,
        width=200 * len(df['Metric'].unique()),
    )

    if save_path:
        interactive_path = save_path.replace('.png', '_interactive.html')
        fig.write_html(interactive_path)
        logger.info(f"Interactive model comparison saved to {interactive_path}")

    return df


def plot_training_convergence(history, metrics=['loss'], title=None, save_path=None):
    """
    Plot convergence of training metrics over iterations or epochs

    Parameters:
        history (dict): Training history with metrics
        metrics (list): Metrics to plot
        title (str): Plot title
        save_path (str): Path to save the figure
    """
    if not history or not metrics:
        logger.warning("No history data or metrics provided")
        return

    available_metrics = [m for m in metrics if m in history]
    if not available_metrics:
        logger.warning(f"None of the requested metrics {metrics} found in history")
        return

    plt.figure(figsize=(10, 6))

    for metric in available_metrics:
        values = history[metric]
        iterations = list(range(1, len(values) + 1))

        plt.plot(iterations, values, marker='o', linestyle='-', linewidth=2, label=metric)

        # Add validation metric if available
        val_metric = f'val_{metric}'
        if val_metric in history:
            plt.plot(iterations, history[val_metric], marker='x', linestyle='--',
                     linewidth=2, label=f'Validation {metric}')

    if title:
        plt.title(title)
    else:
        plt.title('Training Convergence')

    plt.xlabel('Iteration/Epoch')
    plt.ylabel('Value')
    plt.grid(True)
    plt.legend()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Convergence plot saved to {save_path}")
    else:
        plt.show()

    plt.close()


def plot_svd_explained_variance(svd_model, save_path=None):
    """
    Plot explained variance ratio for SVD components

    Parameters:
        svd_model: Trained SVD model
        save_path (str): Path to save the figure
    """
    if not hasattr(svd_model, 'explained_variance_ratio_'):
        logger.warning("SVD model does not have explained_variance_ratio_ attribute")
        return

    # Get explained variance ratio
    explained_variance = svd_model.explained_variance_ratio_
    cumulative_variance = np.cumsum(explained_variance)
    components = range(1, len(explained_variance) + 1)

    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Plot individual explained variance
    ax1.bar(components, explained_variance, alpha=0.7)
    ax1.set_title('Individual Explained Variance')
    ax1.set_xlabel('Component')
    ax1.set_ylabel('Explained Variance Ratio')
    ax1.grid(True)

    # Plot cumulative explained variance
    ax2.plot(components, cumulative_variance, marker='o', linestyle='-', linewidth=2)
    ax2.set_title('Cumulative Explained Variance')
    ax2.set_xlabel('Number of Components')
    ax2.set_ylabel('Cumulative Explained Variance')
    ax2.grid(True)

    # Add threshold lines
    for threshold in [0.8, 0.9, 0.95]:
        # Find first component to exceed threshold
        try:
            idx = np.where(cumulative_variance >= threshold)[0][0]
            ax2.axhline(y=threshold, color='r', linestyle='--', alpha=0.5)
            ax2.axvline(x=idx + 1, color='r', linestyle='--', alpha=0.5)
            ax2.annotate(f'{threshold * 100:.0f}% at {idx + 1} components',
                         xy=(idx + 1, threshold),
                         xytext=(idx + 1 + 2, threshold + 0.05),
                         arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=8))
        except IndexError:
            pass

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"SVD explained variance plot saved to {save_path}")
    else:
        plt.show()

    plt.close()


def create_recommendation_report(recommendations, user_data, game_data, save_path=None):
    """
    Create a visual report of recommendations for a user

    Parameters:
        recommendations (list): List of (game_id, score) tuples
        user_data (DataFrame): User data with past interactions
        game_data (DataFrame): Game metadata
        save_path (str): Path to save the report
    """
    if not recommendations or game_data is None:
        logger.warning("No recommendations or game data provided")
        return

    # Get game info for recommendations
    game_ids = [game_id for game_id, _ in recommendations]
    scores = [score for _, score in recommendations]

    recommended_games = game_data[game_data['app_id'].isin(game_ids)].copy()
    if len(recommended_games) == 0:
        logger.warning("Could not find game data for recommendations")
        return

    # Add recommendation scores
    id_to_score = dict(recommendations)
    recommended_games['score'] = recommended_games['app_id'].map(id_to_score)

    # Sort by score
    recommended_games = recommended_games.sort_values('score', ascending=False)

    # Create figure
    fig = plt.figure(figsize=(12, len(recommendations) * 0.8 + 3))

    # Plot recommendation scores
    plt.barh(recommended_games['title'], recommended_games['score'], color='skyblue')
    plt.title('Recommendation Scores')
    plt.xlabel('Score')
    plt.ylabel('Game')
    plt.xlim(0, 1)
    plt.grid(True, axis='x')

    # Add value labels
    for i, score in enumerate(recommended_games['score']):
        plt.text(score + 0.02, i, f"{score:.3f}", va='center')

    plt.tight_layout()

    # Save or show
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Recommendation report saved to {save_path}")
    else:
        plt.show()

    plt.close()

    # Return for potential further analysis
    return recommended_games


def main():
    """Test the visualization functions with synthetic data"""
    # Generate some sample data
    history = {
        'loss': [0.6, 0.5, 0.45, 0.4, 0.38],
        'val_loss': [0.65, 0.55, 0.5, 0.48, 0.47],
        'accuracy': [0.7, 0.75, 0.78, 0.8, 0.82],
        'val_accuracy': [0.68, 0.72, 0.75, 0.76, 0.78]
    }

    evaluation_metrics = {
        'precision': {5: 0.82, 10: 0.76, 20: 0.68},
        'recall': {5: 0.31, 10: 0.42, 20: 0.56},
        'ndcg': {5: 0.85, 10: 0.80, 20: 0.75},
        'diversity': {5: 0.72, 10: 0.68, 20: 0.65},
        'coverage': 0.45
    }

    # Test training history plot
    plot_training_history(history, save_path='test_training_history.png')

    # Test evaluation metrics plot
    plot_evaluation_metrics(evaluation_metrics, save_path='test_evaluation_metrics.png')

    # Test model comparison
    models_metrics = {
        'KNN': {
            'precision': {5: 0.82, 10: 0.76},
            'recall': {5: 0.31, 10: 0.42},
            'ndcg': {5: 0.85, 10: 0.80},
            'coverage': 0.45
        },
        'SVD': {
            'precision': {5: 0.79, 10: 0.73},
            'recall': {5: 0.33, 10: 0.45},
            'ndcg': {5: 0.83, 10: 0.78},
            'coverage': 0.48
        },
        'Hybrid': {
            'precision': {5: 0.85, 10: 0.80},
            'recall': {5: 0.35, 10: 0.48},
            'ndcg': {5: 0.88, 10: 0.83},
            'coverage': 0.52
        }
    }

    plot_model_comparison(models_metrics, save_path='test_model_comparison.png')

    print("Visualization tests completed")


if __name__ == "__main__":
    main()