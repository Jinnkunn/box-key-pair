"""
Cluster analysis visualization functions.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from typing import Dict, Any, List, Tuple
import pandas as pd
import os
from ..utils import (
    create_figure,
    apply_style_to_axis,
    add_statistical_annotations,
    save_figure,
    set_axis_style
)
from ..config import (
    PARAM_COLORS,
    COLOR_PALETTE,
    FIGURE_SIZES,
    ALPHA_VALUES,
    CUSTOM_COLORMAPS,
    STAT_SETTINGS
)
from ..performance.plots import calculate_success_probability

def perform_clustering(results_df: pd.DataFrame, n_clusters: int = 3) -> Tuple[pd.DataFrame, np.ndarray]:
    """Perform clustering analysis on the parameter space."""
    # Select features for clustering
    features = ['theta_mean', 'omega_mean', 'r_mean', 'success_rate']
    X = results_df[features].values
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Perform clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(X_scaled)
    
    # Add cluster labels to dataframe
    results_df_with_clusters = results_df.copy()
    results_df_with_clusters['cluster'] = clusters
    
    return results_df_with_clusters, X_scaled

def create_cluster_plot(results_df: pd.DataFrame, output_dir: str) -> None:
    """Create visualization of cluster analysis in parameter space."""
    # Perform clustering
    results_df_clustered, X_scaled = perform_clustering(results_df)
    
    # Create figure with adjusted size and spacing
    fig = plt.figure(figsize=FIGURE_SIZES['large'])
    gs = plt.GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3)
    
    # Plot 1: PCA visualization
    ax1 = fig.add_subplot(gs[0, 0])
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    
    scatter = ax1.scatter(X_pca[:, 0], X_pca[:, 1],
                         c=results_df_clustered['cluster'],
                         cmap=CUSTOM_COLORMAPS['cluster'],
                         alpha=ALPHA_VALUES['scatter'])
    
    # Add explained variance ratio
    var_ratio = pca.explained_variance_ratio_
    ax1.set_xlabel(f'PC1 ({var_ratio[0]:.2%} variance)')
    ax1.set_ylabel(f'PC2 ({var_ratio[1]:.2%} variance)')
    set_axis_style(ax1, style='clean')
    apply_style_to_axis(ax1, title='Cluster Distribution (PCA)')
    
    # Plot 2-4: Parameter space visualization
    param_pairs = [('theta_mean', 'omega_mean'),
                  ('omega_mean', 'r_mean'),
                  ('r_mean', 'success_rate')]
    param_titles = ['Learning vs Social', 'Social vs Exploration', 'Exploration vs Success']
    
    positions = [(0, 1), (1, 0), (1, 1)]  # 定义每个子图的位置
    
    for i, ((param1, param2), title) in enumerate(zip(param_pairs, param_titles)):
        row, col = positions[i]
        ax = fig.add_subplot(gs[row, col])
        scatter = ax.scatter(results_df_clustered[param1],
                           results_df_clustered[param2],
                           c=results_df_clustered['cluster'],
                           cmap=CUSTOM_COLORMAPS['cluster'],
                           alpha=ALPHA_VALUES['scatter'])
        
        ax.set_xlabel(param1.replace('_mean', ''))
        ax.set_ylabel(param2.replace('_mean', ''))
        set_axis_style(ax, style='clean')
        apply_style_to_axis(ax, title=title)
    
    # Add colorbar
    plt.colorbar(scatter, ax=fig.axes, label='Cluster')
    
    # Save figure
    save_figure(fig, os.path.join(output_dir, 'visualization/cluster/cluster_analysis.png'))

def create_cluster_profiles_plot(results_df: pd.DataFrame, output_dir: str) -> None:
    """Create visualization of cluster profiles."""
    # Perform clustering
    results_df_clustered, _ = perform_clustering(results_df)
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=FIGURE_SIZES['wide'])
    
    # Plot 1: Parameter distributions by cluster
    param_cols = ['theta_mean', 'omega_mean', 'r_mean']
    param_labels = ['Learning (θ)', 'Social (ω)', 'Exploration (r)']
    
    cluster_means = []
    cluster_stds = []
    
    for cluster in range(3):
        cluster_data = results_df_clustered[results_df_clustered['cluster'] == cluster]
        means = [cluster_data[col].mean() for col in param_cols]
        stds = [cluster_data[col].std() for col in param_cols]
        cluster_means.append(means)
        cluster_stds.append(stds)
    
    cluster_means = np.array(cluster_means)
    cluster_stds = np.array(cluster_stds)
    
    x = np.arange(len(param_labels))
    width = 0.25
    
    for i in range(3):
        ax1.bar(x + i*width, cluster_means[i],
                width, yerr=cluster_stds[i],
                label=f'Cluster {i}',
                color=COLOR_PALETTE[f'cluster_{i}'],
                alpha=ALPHA_VALUES['main_plot'])
    
    ax1.set_xticks(x + width)
    ax1.set_xticklabels(param_labels)
    ax1.legend()
    set_axis_style(ax1, style='clean')
    apply_style_to_axis(ax1, title='Parameter Profiles by Cluster')
    
    # Plot 2: Performance metrics by cluster
    metrics = ['success_rate', 'num_unlock', 'unlock_time']
    metric_labels = ['Success Rate', 'Number of Unlocks', 'Completion Time']
    
    # Create box plots for each metric
    positions = np.arange(len(metrics))
    
    for i, (metric, label) in enumerate(zip(metrics, metric_labels)):
        # Create box plot
        sns.boxplot(data=results_df_clustered,
                   x='cluster',
                   y=metric,
                   hue='cluster',
                   palette=[COLOR_PALETTE[f'cluster_{j}'] for j in range(3)],
                   legend=False,
                   ax=ax2)
        
        # Add individual points
        sns.stripplot(data=results_df_clustered,
                     x='cluster',
                     y=metric,
                     color=COLOR_PALETTE['neutral'],
                     alpha=ALPHA_VALUES['scatter'],
                     size=4,
                     jitter=0.2,
                     ax=ax2)
    
    # Style plot 2
    ax2.set_xlabel('Cluster')
    ax2.set_ylabel('Values')
    set_axis_style(ax2, style='clean')
    apply_style_to_axis(ax2, title='Performance by Cluster')
    
    plt.tight_layout()
    save_figure(fig, os.path.join(output_dir, 'visualization/cluster/cluster_profiles.png'))

def create_cluster_characteristics_plot(results_df: pd.DataFrame, output_dir: str) -> None:
    """Create visualization of cluster characteristics and demographics."""
    # Perform clustering
    results_df_clustered, _ = perform_clustering(results_df)
    
    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=FIGURE_SIZES['large'])
    axes = axes.ravel()
    
    # Plot 1: Age distribution by cluster
    sns.boxplot(data=results_df_clustered,
               x='cluster',
               y='age',
               hue='cluster',
               palette=CUSTOM_COLORMAPS['cluster'],
               legend=False,
               ax=axes[0])
    
    sns.stripplot(data=results_df_clustered,
                 x='cluster',
                 y='age',
                 color=COLOR_PALETTE['neutral'],
                 alpha=ALPHA_VALUES['scatter'],
                 size=4,
                 jitter=0.2,
                 ax=axes[0])
    
    set_axis_style(axes[0], style='clean')
    apply_style_to_axis(axes[0], title='Age Distribution by Cluster')
    
    # Plot 2: Strategy usage by cluster
    strategy_cols = ['color_match_rate', 'num_match_rate', 'shape_match_rate']
    strategy_means = []
    strategy_stds = []
    
    for cluster in range(3):
        cluster_data = results_df_clustered[results_df_clustered['cluster'] == cluster]
        means = [cluster_data[col].mean() for col in strategy_cols]
        stds = [cluster_data[col].std() for col in strategy_cols]
        strategy_means.append(means)
        strategy_stds.append(stds)
    
    strategy_means = np.array(strategy_means)
    strategy_stds = np.array(strategy_stds)
    
    x = np.arange(len(strategy_cols))
    width = 0.25
    
    for i in range(3):
        axes[1].bar(x + i*width, strategy_means[i],
                   width, yerr=strategy_stds[i],
                   label=f'Cluster {i}',
                   color=COLOR_PALETTE[f'cluster_{i}'],
                   alpha=ALPHA_VALUES['main_plot'])
    
    axes[1].set_xticks(x + width)
    axes[1].set_xticklabels(['Color', 'Number', 'Shape'])
    axes[1].legend()
    set_axis_style(axes[1], style='clean')
    apply_style_to_axis(axes[1], title='Strategy Usage by Cluster')
    
    # Plot 3: Gender distribution by cluster
    gender_props = []
    for cluster in range(3):
        cluster_data = results_df_clustered[results_df_clustered['cluster'] == cluster]
        gender_prop = cluster_data['gender'].value_counts(normalize=True)
        gender_props.append(gender_prop)
    
    gender_df = pd.DataFrame(gender_props).fillna(0)
    gender_df.plot(kind='bar', stacked=True,
                  color=[COLOR_PALETTE['boys'], COLOR_PALETTE['girls']],
                  ax=axes[2])
    
    axes[2].set_xlabel('Cluster')
    axes[2].set_ylabel('Proportion')
    axes[2].legend(title='Gender')
    set_axis_style(axes[2], style='clean')
    apply_style_to_axis(axes[2], title='Gender Distribution by Cluster')
    
    # Plot 4: Learning trajectories by cluster
    for cluster in range(3):
        cluster_data = results_df_clustered[results_df_clustered['cluster'] == cluster]
        success_trajectory = cluster_data.groupby('total_trials')['success_rate'].mean()
        axes[3].plot(success_trajectory.index,
                    success_trajectory.values,
                    label=f'Cluster {cluster}',
                    color=COLOR_PALETTE[f'cluster_{cluster}'],
                    alpha=ALPHA_VALUES['main_plot'])
    
    axes[3].legend()
    axes[3].set_xlabel('Trial Number')
    axes[3].set_ylabel('Success Rate')
    set_axis_style(axes[3], style='clean')
    apply_style_to_axis(axes[3], title='Learning Trajectories by Cluster')
    
    plt.tight_layout()
    save_figure(fig, os.path.join(output_dir, 'visualization/cluster/cluster_characteristics.png'))

def create_cluster_size_completion_plot(results_df: pd.DataFrame, output_dir: str) -> None:
    """Create visualization of cluster size and completion distribution."""
    # Perform clustering if not already done
    results_df_clustered, _ = perform_clustering(results_df)
    
    # Create figure
    fig, ax = create_figure(figsize='single')
    
    # Calculate cluster completion rates
    cluster_completion = pd.crosstab(results_df_clustered['cluster'], results_df_clustered['solved'])
    
    # Create stacked bar plot
    cluster_completion.plot(kind='bar', stacked=True,
                          color=[COLOR_PALETTE['failure'], COLOR_PALETTE['success']],
                          alpha=ALPHA_VALUES['main_plot'],
                          ax=ax)
    
    # Add value labels
    for c in ax.containers:
        ax.bar_label(c, fmt='%d', label_type='center')
    
    # Style the plot
    set_axis_style(ax, style='clean')
    apply_style_to_axis(ax,
                       title='Cluster Size and Completion Distribution',
                       xlabel='Cluster',
                       ylabel='Number of Participants')
    ax.legend(['Incomplete', 'Complete'])
    
    plt.tight_layout()
    save_figure(fig, os.path.join(output_dir, 'visualization/cluster/cluster_size_completion.png'))

def add_stats_if_sufficient(data1, data2, ax, min_samples=2):
    """添加统计检验结果，但仅在样本量足够时进行。"""
    if len(data1) >= min_samples and len(data2) >= min_samples:
        stat, pval = stats.ttest_ind(data1, data2)
        stats_dict = {
            't-stat': stat,
            'p-value': pval
        }
        add_statistical_annotations(ax, stats_dict)
    else:
        stats_dict = {
            'warning': 'Insufficient sample size'
        }
        add_statistical_annotations(ax, stats_dict)

def create_success_vs_completion_plot(results_df: pd.DataFrame, output_dir: str) -> None:
    """Create visualization of success rate vs completion time by cluster."""
    # Perform clustering if not already done
    results_df_clustered, _ = perform_clustering(results_df)
    
    # Create figure
    fig, ax = create_figure(figsize='single')
    
    # Create scatter plot
    scatter = ax.scatter(results_df_clustered['success_rate'],
                        results_df_clustered['unlock_time'],
                        c=results_df_clustered['cluster'],
                        cmap=CUSTOM_COLORMAPS['cluster'],
                        alpha=ALPHA_VALUES['scatter'],
                        s=100)
    
    # Add correlation statistics if sufficient samples
    if len(results_df_clustered) >= 2:
        corr, pval = stats.pearsonr(results_df_clustered['success_rate'],
                                   results_df_clustered['unlock_time'])
        stats_dict = {
            'r': corr,
            'p-value': pval
        }
    else:
        stats_dict = {
            'warning': 'Insufficient sample size'
        }
    add_statistical_annotations(ax, stats_dict)
    
    # Style the plot
    set_axis_style(ax, style='clean')
    apply_style_to_axis(ax,
                       title='Success Rate vs Completion Time by Cluster',
                       xlabel='Success Rate',
                       ylabel='Completion Time (s)')
    plt.colorbar(scatter, label='Cluster')
    
    plt.tight_layout()
    save_figure(fig, os.path.join(output_dir, 'visualization/cluster/success_vs_completion.png'))

def create_unlocks_by_cluster_plot(results_df: pd.DataFrame, output_dir: str) -> None:
    """Create visualization of unlock counts by cluster."""
    # Perform clustering
    results_df_clustered, _ = perform_clustering(results_df)
    
    # Create figure
    fig, ax = create_figure(figsize='single')
    
    # Create box plot
    sns.boxplot(data=results_df_clustered,
                x='cluster',
                y='num_unlock',
                hue='cluster',
                palette=CUSTOM_COLORMAPS['cluster'],
                legend=False,
                ax=ax)
    
    # Add individual points
    sns.stripplot(data=results_df_clustered,
                 x='cluster',
                 y='num_unlock',
                 color=COLOR_PALETTE['neutral'],
                 alpha=ALPHA_VALUES['scatter'],
                 size=4,
                 jitter=0.2,
                 ax=ax)
    
    # Style the plot
    set_axis_style(ax, style='clean')
    apply_style_to_axis(ax,
                       title='Unlock Count Distribution by Cluster',
                       xlabel='Cluster',
                       ylabel='Number of Unlocks')
    
    plt.tight_layout()
    save_figure(fig, os.path.join(output_dir, 'visualization/cluster/unlocks_by_cluster.png')) 