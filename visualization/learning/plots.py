"""
Learning dynamics visualization functions.
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
    STAT_SETTINGS,
    preprocess_trajectories,
    calculate_average_curve,
    calculate_confidence_intervals
)

def create_learning_curves(results_df: pd.DataFrame, 
                         all_trajectories: Dict[str, Any],
                         output_dir: str) -> None:
    """Create learning curves visualization."""
    fig, axes = plt.subplots(2, 2, figsize=FIGURE_SIZES['large'])
    axes = axes.ravel()
    
    params = ['theta', 'omega', 'r', 'weights']
    param_labels = ['Learning Ability (θ)', 'Social Influence (ω)', 
                   'Exploration Rate (r)', 'Particle Weights']
    
    for i, (param, label) in enumerate(zip(params, param_labels)):
        ax = axes[i]
        
        # Process trajectory data
        processed_data = preprocess_trajectories(all_trajectories, param)
        
        # Calculate average curve and confidence intervals
        avg_curve = calculate_average_curve(processed_data)
        ci_lower, ci_upper = calculate_confidence_intervals(processed_data)
        
        # Plot individual trajectories
        for j in range(len(processed_data)):
            ax.plot(processed_data[j], 
                   alpha=ALPHA_VALUES['background'],
                   color=PARAM_COLORS[param.split('_')[0] if param != 'weights' else 'motor'])
        
        # Plot average curve and confidence interval
        x = np.arange(len(avg_curve))
        ax.fill_between(x, ci_lower, ci_upper, 
                       alpha=ALPHA_VALUES['background'],
                       color=PARAM_COLORS[param.split('_')[0] if param != 'weights' else 'motor'])
        ax.plot(x, avg_curve, 
                color=PARAM_COLORS[param.split('_')[0] if param != 'weights' else 'motor'],
                linewidth=2,
                label='Average')
        
        # Style the plot
        ax.set_xlabel('Trial Number')
        ax.set_ylabel(label)
        set_axis_style(ax, style='clean')
        apply_style_to_axis(ax)
        
        # Add convergence statistics
        final_values = processed_data[:, -1]
        stats_dict = {
            'Final Mean': np.mean(final_values),
            'Final Std': np.std(final_values),
            'CV': np.std(final_values) / np.mean(final_values)
        }
        add_statistical_annotations(ax, stats_dict)
    
    plt.tight_layout()
    save_figure(fig, os.path.join(output_dir, 'visualization/learning/learning_curves.png'))

def create_learning_speed_plot(results_df: pd.DataFrame,
                             all_trajectories: Dict[str, Any],
                             output_dir: str) -> None:
    """Create visualization of learning speed analysis."""
    # Calculate learning speeds
    learning_speeds = {}
    for subject_id, trajectories in all_trajectories.items():
        theta_trajectory = np.average(trajectories['theta'], axis=1)
        # Calculate speed as the maximum rate of change
        learning_speeds[subject_id] = np.max(np.abs(np.diff(theta_trajectory)))
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=FIGURE_SIZES['wide'])
    
    # Plot 1: Learning speed distribution
    sns.histplot(data=list(learning_speeds.values()),
                ax=ax1,
                color=PARAM_COLORS['theta'],
                alpha=ALPHA_VALUES['main_plot'])
    
    # Add distribution statistics
    speeds = np.array(list(learning_speeds.values()))
    stats_dict = {
        'Mean': np.mean(speeds),
        'Std': np.std(speeds),
        'Median': np.median(speeds)
    }
    add_statistical_annotations(ax1, stats_dict)
    
    # Style plot 1
    ax1.set_xlabel('Learning Speed')
    ax1.set_ylabel('Count')
    set_axis_style(ax1, style='clean')
    apply_style_to_axis(ax1, title='Distribution of Learning Speeds')
    
    # Plot 2: Learning speed vs performance
    performance = results_df['success_rate'].values
    speeds = np.array([learning_speeds[id] for id in results_df['ID']])
    
    sns.regplot(x=speeds, y=performance,
                ax=ax2,
                color=PARAM_COLORS['theta'],
                scatter_kws={'alpha': ALPHA_VALUES['scatter']})
    
    # Add correlation statistics
    corr, pval = stats.pearsonr(speeds, performance)
    stats_dict = {
        'r': corr,
        'p-value': pval
    }
    add_statistical_annotations(ax2, stats_dict)
    
    # Style plot 2
    ax2.set_xlabel('Learning Speed')
    ax2.set_ylabel('Success Rate')
    set_axis_style(ax2, style='clean')
    apply_style_to_axis(ax2, title='Learning Speed vs Performance')
    
    plt.tight_layout()
    save_figure(fig, os.path.join(output_dir, 'visualization/learning/learning_speed.png'))

def create_parameter_evolution_plot(results_df: pd.DataFrame,
                                  all_trajectories: Dict[str, Any],
                                  output_dir: str) -> None:
    """Create visualization of parameter evolution patterns."""
    fig, axes = plt.subplots(2, 2, figsize=FIGURE_SIZES['large'])
    axes = axes.ravel()
    
    params = ['theta', 'omega', 'r', 'weights']
    param_labels = ['Learning Ability (θ)', 'Social Influence (ω)', 
                   'Exploration Rate (r)', 'Particle Weights']
    
    for i, (param, label) in enumerate(zip(params, param_labels)):
        ax = axes[i]
        
        # Process data
        processed_data = preprocess_trajectories(all_trajectories, param)
        
        # Calculate evolution patterns
        mean_trajectory = np.mean(processed_data, axis=0)
        std_trajectory = np.std(processed_data, axis=0)
        
        # Plot evolution pattern
        x = np.arange(len(mean_trajectory))
        ax.fill_between(x, 
                       mean_trajectory - std_trajectory,
                       mean_trajectory + std_trajectory,
                       alpha=ALPHA_VALUES['background'],
                       color=PARAM_COLORS[param.split('_')[0] if param != 'weights' else 'motor'])
        ax.plot(x, mean_trajectory,
                color=PARAM_COLORS[param.split('_')[0] if param != 'weights' else 'motor'],
                linewidth=2)
        
        # Add trend analysis
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, mean_trajectory)
        trend_line = slope * x + intercept
        ax.plot(x, trend_line, '--', 
                color=COLOR_PALETTE['neutral'],
                alpha=ALPHA_VALUES['main_plot'],
                label=f'Trend (slope={slope:.3f})')
        
        # Add statistics
        stats_dict = {
            'Slope': slope,
            'R²': r_value**2,
            'p-value': p_value
        }
        add_statistical_annotations(ax, stats_dict)
        
        # Style the plot
        ax.set_xlabel('Trial Number')
        ax.set_ylabel(label)
        set_axis_style(ax, style='clean')
        apply_style_to_axis(ax)
    
    plt.tight_layout()
    save_figure(fig, os.path.join(output_dir, 'visualization/learning/parameter_evolution.png')) 