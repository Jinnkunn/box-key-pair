"""
Strategy analysis visualization functions.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from typing import Dict, Any
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

def create_strategy_usage_plot(results_df: pd.DataFrame, output_dir: str) -> None:
    """Create visualization of strategy usage patterns."""
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=FIGURE_SIZES['wide'])
    
    # Plot 1: Overall strategy usage
    strategy_cols = ['color_match_rate', 'num_match_rate', 'shape_match_rate']
    strategy_names = ['Color', 'Number', 'Shape']
    
    # Calculate mean usage rates
    usage_rates = [results_df[col].mean() for col in strategy_cols]
    usage_std = [results_df[col].std() for col in strategy_cols]
    
    # Create bar plot
    bars = ax1.bar(strategy_names, usage_rates,
                   yerr=usage_std,
                   color=[COLOR_PALETTE['boys'], COLOR_PALETTE['girls'], COLOR_PALETTE['accent']],
                   alpha=ALPHA_VALUES['main_plot'])
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}',
                ha='center', va='bottom')
    
    # Style plot 1
    ax1.set_ylim(0, 1)
    set_axis_style(ax1, style='clean')
    apply_style_to_axis(ax1,
                       title='Strategy Usage Distribution',
                       xlabel='Strategy Type',
                       ylabel='Usage Rate')
    
    # Plot 2: Strategy correlations
    corr_matrix = results_df[strategy_cols].corr()
    sns.heatmap(corr_matrix,
                ax=ax2,
                cmap=CUSTOM_COLORMAPS['correlation'],
                annot=True,
                xticklabels=strategy_names,
                yticklabels=strategy_names,
                vmin=-1, vmax=1,
                center=0)
    
    # Calculate and add significance markers
    for i in range(len(strategy_cols)):
        for j in range(len(strategy_cols)):
            if i < j:
                stat, pval = stats.pearsonr(results_df[strategy_cols[i]], 
                                          results_df[strategy_cols[j]])
                if pval < 0.05:
                    ax2.text(j + 0.5, i + 0.5, '*',
                            ha='center', va='center',
                            color='black')
    
    # Style plot 2
    apply_style_to_axis(ax2,
                       title='Strategy Correlations',
                       grid=False)
    
    plt.tight_layout()
    save_figure(fig, os.path.join(output_dir, 'visualization/strategy/strategy_usage.png'))

def create_strategy_effectiveness_plot(results_df: pd.DataFrame, output_dir: str) -> None:
    """Create visualization of strategy effectiveness."""
    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=FIGURE_SIZES['large'])
    axes = axes.ravel()
    
    strategy_cols = ['color_match_rate', 'num_match_rate', 'shape_match_rate']
    strategy_names = ['Color Matching', 'Number Matching', 'Shape Matching']
    performance_metrics = ['success_rate', 'solved']
    metric_labels = ['Success Rate', 'Task Completion']
    
    plot_idx = 0
    for metric, metric_label in zip(performance_metrics, metric_labels):
        for strategy, strategy_name in zip(strategy_cols, strategy_names):
            ax = axes[plot_idx]
            
            # Create scatter plot with regression line
            sns.regplot(data=results_df,
                       x=strategy,
                       y=metric,
                       color=PARAM_COLORS['theta'],
                       scatter_kws={'alpha': ALPHA_VALUES['scatter']},
                       ax=ax)
            
            # Add correlation statistics
            corr, pval = stats.pearsonr(results_df[strategy], results_df[metric])
            stats_dict = {
                'r': corr,
                'p-value': pval
            }
            add_statistical_annotations(ax, stats_dict)
            
            # Style the plot
            ax.set_xlabel(f'{strategy_name} Rate')
            ax.set_ylabel(metric_label)
            set_axis_style(ax, style='clean')
            apply_style_to_axis(ax)
            
            plot_idx += 1
            if plot_idx >= 4:
                break
        if plot_idx >= 4:
            break
    
    plt.tight_layout()
    save_figure(fig, os.path.join(output_dir, 'visualization/strategy/strategy_effectiveness.png'))

def create_strategy_heatmap(merged_data: pd.DataFrame, output_dir: str) -> None:
    """Create heatmap visualization of strategy transitions."""
    # Calculate strategy transitions
    strategies = ['ColorMatch', 'NumMatch', 'ShapeMatch']
    n_strategies = len(strategies)
    transition_matrix = np.zeros((n_strategies, n_strategies))
    
    for subject_id in merged_data['ID'].unique():
        subject_data = merged_data[merged_data['ID'] == subject_id]
        
        for i in range(len(subject_data) - 1):
            current_strat = None
            next_strat = None
            
            # Find current strategy
            for idx, strat in enumerate(strategies):
                if subject_data.iloc[i][strat] == 1:
                    current_strat = idx
                    break
            
            # Find next strategy
            for idx, strat in enumerate(strategies):
                if subject_data.iloc[i + 1][strat] == 1:
                    next_strat = idx
                    break
            
            if current_strat is not None and next_strat is not None:
                transition_matrix[current_strat, next_strat] += 1
    
    # Normalize transitions
    row_sums = transition_matrix.sum(axis=1, keepdims=True)
    transition_matrix = np.divide(transition_matrix, row_sums,
                                where=row_sums != 0)
    
    # Create figure
    fig, ax = create_figure(figsize='square')
    
    # Create heatmap
    sns.heatmap(transition_matrix,
                annot=True,
                fmt='.2f',
                cmap=CUSTOM_COLORMAPS['heatmap'],
                xticklabels=['Color', 'Number', 'Shape'],
                yticklabels=['Color', 'Number', 'Shape'],
                ax=ax)
    
    # Style the plot
    apply_style_to_axis(ax,
                       title='Strategy Transition Probabilities',
                       xlabel='Next Strategy',
                       ylabel='Current Strategy',
                       grid=False)
    
    plt.tight_layout()
    save_figure(fig, os.path.join(output_dir, 'visualization/strategy/strategy_heatmap.png'))

def create_strategy_sequence(merged_data: pd.DataFrame, output_dir: str) -> None:
    """Create visualization of strategy sequences over time."""
    # Create figure
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=FIGURE_SIZES['tall'])
    
    # Plot 1: Strategy usage over trials
    strategies = ['ColorMatch', 'NumMatch', 'ShapeMatch']
    strategy_names = ['Color', 'Number', 'Shape']
    colors = [COLOR_PALETTE['boys'], COLOR_PALETTE['girls'], COLOR_PALETTE['accent']]
    
    for strategy, name, color in zip(strategies, strategy_names, colors):
        # Calculate moving average of strategy usage
        usage_rate = merged_data.groupby('ID')[strategy].rolling(window=5).mean()
        mean_rate = usage_rate.groupby(level=0).mean()
        
        # Plot average usage rate
        ax1.plot(mean_rate.index, mean_rate.values,
                 label=name,
                 color=color,
                 alpha=ALPHA_VALUES['main_plot'])
    
    # Style plot 1
    ax1.set_xlabel('Trial Number')
    ax1.set_ylabel('Strategy Usage Rate')
    set_axis_style(ax1, style='clean')
    apply_style_to_axis(ax1,
                       title='Strategy Usage Evolution',
                       legend=True)
    
    # Plot 2: Success rate by strategy
    success_rates = []
    for strategy in strategies:
        success_rate = merged_data[merged_data[strategy] == 1]['Worked'].mean()
        success_rates.append(success_rate)
    
    # Create bar plot
    bars = ax2.bar(strategy_names, success_rates,
                   color=colors,
                   alpha=ALPHA_VALUES['main_plot'])
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}',
                ha='center', va='bottom')
    
    # Style plot 2
    ax2.set_ylim(0, 1)
    set_axis_style(ax2, style='clean')
    apply_style_to_axis(ax2,
                       title='Strategy Success Rates',
                       xlabel='Strategy Type',
                       ylabel='Success Rate')
    
    plt.tight_layout()
    save_figure(fig, os.path.join(output_dir, 'visualization/strategy/strategy_sequence.png')) 