"""
Performance analysis visualization functions.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from typing import Dict, Any
import pandas as pd
import os
from ...utils import (
    create_figure,
    apply_style_to_axis,
    add_statistical_annotations,
    save_figure,
    set_axis_style
)
from ...config import (
    PARAM_COLORS,
    COLOR_PALETTE,
    FIGURE_SIZES,
    ALPHA_VALUES,
    CUSTOM_COLORMAPS,
    STAT_SETTINGS
)

def calculate_success_probability(theta: float, motor: float, omega: float, r: float) -> float:
    """
    Calculate the probability of success based on model parameters.
    
    Args:
        theta: Learning ability parameter
        motor: Motor skill parameter
        omega: Social influence parameter
        r: Exploration rate parameter
    
    Returns:
        float: Predicted probability of success
    """
    base_prob = theta * motor
    social_effect = omega * (1 - base_prob)
    exploration_effect = r * (1 - base_prob - social_effect)
    return base_prob + social_effect + exploration_effect

def add_stats_if_sufficient(data1, data2, ax, min_samples=2):
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

def create_success_rate_plot(results_df: pd.DataFrame, output_dir: str) -> None:
    """Create visualization of success rates and their distribution."""
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=FIGURE_SIZES['wide'])
    
    # Plot 1: Success rate distribution
    sns.histplot(data=results_df['success_rate'],
                ax=ax1,
                color=COLOR_PALETTE['success'],
                alpha=ALPHA_VALUES['main_plot'])
    
    # Add distribution statistics
    stats_dict = {
        'Mean': results_df['success_rate'].mean(),
        'Std': results_df['success_rate'].std(),
        'Median': results_df['success_rate'].median()
    }
    add_statistical_annotations(ax1, stats_dict)
    
    # Style plot 1
    ax1.set_xlabel('Success Rate')
    ax1.set_ylabel('Count')
    set_axis_style(ax1, style='clean')
    apply_style_to_axis(ax1, title='Success Rate Distribution')
    
    # Plot 2: Success rate by completion status
    sns.boxplot(data=results_df,
                x='solved',
                y='success_rate',
                hue='solved',
                palette=[COLOR_PALETTE['failure'], COLOR_PALETTE['success']],
                legend=False,
                ax=ax2)
    
    # Add individual points
    sns.stripplot(data=results_df,
                 x='solved',
                 y='success_rate',
                 color=COLOR_PALETTE['neutral'],
                 alpha=ALPHA_VALUES['scatter'],
                 size=4,
                 jitter=0.2,
                 ax=ax2)
    
    # Add statistical test
    complete = results_df[results_df['solved'] == 1]['success_rate']
    incomplete = results_df[results_df['solved'] == 0]['success_rate']
    add_stats_if_sufficient(complete, incomplete, ax2)
    
    # Style plot 2
    ax2.set_xticks([0, 1])
    ax2.set_xticklabels(['Incomplete', 'Complete'])
    ax2.set_xlabel('Task Completion')
    ax2.set_ylabel('Success Rate')
    set_axis_style(ax2, style='clean')
    apply_style_to_axis(ax2, title='Success Rate by Completion')
    
    plt.tight_layout()
    save_figure(fig, os.path.join(output_dir, 'visualization/performance/success_rate.png'))

def create_performance_by_age_plot(results_df: pd.DataFrame, output_dir: str) -> None:
    """Create visualization of performance metrics by age."""
    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=FIGURE_SIZES['large'])
    axes = axes.ravel()
    
    metrics = ['success_rate', 'num_unlock', 'unlock_time', 'total_trials']
    metric_labels = ['Success Rate', 'Number of Unlocks', 'Completion Time', 'Total Trials']
    
    for i, (metric, label) in enumerate(zip(metrics, metric_labels)):
        ax = axes[i]
        
        # Create scatter plot with regression line
        sns.regplot(data=results_df,
                   x='age',
                   y=metric,
                   color=PARAM_COLORS['theta'],
                   scatter_kws={'alpha': ALPHA_VALUES['scatter']},
                   ax=ax)
        
        # Add correlation statistics
        corr, pval = stats.pearsonr(results_df['age'], results_df[metric])
        stats_dict = {
            'r': corr,
            'p-value': pval
        }
        add_statistical_annotations(ax, stats_dict)
        
        # Style the plot
        ax.set_xlabel('Age (years)')
        ax.set_ylabel(label)
        set_axis_style(ax, style='clean')
        apply_style_to_axis(ax)
    
    plt.tight_layout()
    save_figure(fig, os.path.join(output_dir, 'visualization/performance/performance_by_age.png'))

def create_completion_analysis_plot(results_df: pd.DataFrame, output_dir: str) -> None:
    """Create visualization of task completion patterns."""
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=FIGURE_SIZES['wide'])
    
    # Plot 1: Completion time distribution
    sns.histplot(data=results_df,
                x='unlock_time',
                hue='solved',
                multiple='stack',
                palette=[COLOR_PALETTE['failure'], COLOR_PALETTE['success']],
                alpha=ALPHA_VALUES['main_plot'],
                ax=ax1)
    
    # Add distribution statistics
    stats_dict = {
        'Mean Time': results_df['unlock_time'].mean(),
        'Median Time': results_df['unlock_time'].median(),
        'Std Time': results_df['unlock_time'].std()
    }
    add_statistical_annotations(ax1, stats_dict)
    
    # Style plot 1
    ax1.set_xlabel('Completion Time (s)')
    ax1.set_ylabel('Count')
    set_axis_style(ax1, style='clean')
    apply_style_to_axis(ax1, title='Completion Time Distribution')
    
    # Plot 2: Completion rate by number of unlocks
    sns.boxplot(data=results_df,
                x='num_unlock',
                y='success_rate',
                color=COLOR_PALETTE['success'],
                ax=ax2)
    
    # Add individual points
    sns.stripplot(data=results_df,
                 x='num_unlock',
                 y='success_rate',
                 color=COLOR_PALETTE['neutral'],
                 alpha=ALPHA_VALUES['scatter'],
                 size=4,
                 jitter=0.2,
                 ax=ax2)
    
    # Add correlation statistics
    corr, pval = stats.pearsonr(results_df['num_unlock'], results_df['success_rate'])
    stats_dict = {
        'r': corr,
        'p-value': pval
    }
    add_statistical_annotations(ax2, stats_dict)
    
    # Style plot 2
    ax2.set_xlabel('Number of Unlocks')
    ax2.set_ylabel('Success Rate')
    set_axis_style(ax2, style='clean')
    apply_style_to_axis(ax2, title='Success Rate by Number of Unlocks')
    
    plt.tight_layout()
    save_figure(fig, os.path.join(output_dir, 'visualization/performance/completion_analysis.png'))

def create_model_evaluation_plot(results_df: pd.DataFrame, output_dir: str) -> None:
    """Create visualization of model evaluation metrics."""
    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=FIGURE_SIZES['large'])
    axes = axes.ravel()
    
    # Plot 1: Parameter uncertainties
    param_stds = ['theta_std', 'omega_std', 'r_std']
    param_means = ['theta_mean', 'omega_mean', 'r_mean']
    param_labels = ['Learning (θ)', 'Social (ω)', 'Exploration (r)']
    
    ax = axes[0]
    for std, mean, label in zip(param_stds, param_means, param_labels):
        sns.regplot(data=results_df,
                   x=mean,
                   y=std,
                   label=label,
                   scatter_kws={'alpha': ALPHA_VALUES['scatter']},
                   ax=ax)
    
    ax.legend()
    ax.set_xlabel('Parameter Value')
    ax.set_ylabel('Standard Deviation')
    set_axis_style(ax, style='clean')
    apply_style_to_axis(ax, title='Parameter Uncertainties')
    
    # Plot 2: Success rate prediction
    ax = axes[1]
    predicted_success = results_df.apply(lambda row: calculate_success_probability(
        row['theta_mean'], row['motor'], row['omega_mean'], row['r_mean']), axis=1)
    
    sns.regplot(x=predicted_success,
                y=results_df['success_rate'],
                color=PARAM_COLORS['theta'],
                scatter_kws={'alpha': ALPHA_VALUES['scatter']},
                ax=ax)
    
    # Add correlation statistics
    corr, pval = stats.pearsonr(predicted_success, results_df['success_rate'])
    stats_dict = {
        'r': corr,
        'p-value': pval
    }
    add_statistical_annotations(ax, stats_dict)
    
    ax.set_xlabel('Predicted Success Rate')
    ax.set_ylabel('Actual Success Rate')
    set_axis_style(ax, style='clean')
    apply_style_to_axis(ax, title='Model Prediction Accuracy')
    
    # Plot 3: Parameter convergence
    ax = axes[2]
    param_ranges = {
        'theta': (0, 1),
        'omega': (1, 10),
        'r': (0, 1)
    }
    
    for param, (min_val, max_val) in param_ranges.items():
        density = stats.gaussian_kde(results_df[f'{param}_mean'])
        x = np.linspace(min_val, max_val, 100)
        ax.plot(x, density(x), label=param)
    
    ax.legend()
    ax.set_xlabel('Parameter Value')
    ax.set_ylabel('Density')
    set_axis_style(ax, style='clean')
    apply_style_to_axis(ax, title='Parameter Convergence')
    
    # Plot 4: Model residuals
    ax = axes[3]
    residuals = results_df['success_rate'] - predicted_success
    
    sns.histplot(residuals,
                color=COLOR_PALETTE['neutral'],
                alpha=ALPHA_VALUES['main_plot'],
                ax=ax)
    
    # Add residual statistics
    stats_dict = {
        'Mean': np.mean(residuals),
        'Std': np.std(residuals),
        'Skew': stats.skew(residuals)
    }
    add_statistical_annotations(ax, stats_dict)
    
    ax.set_xlabel('Residual')
    ax.set_ylabel('Count')
    set_axis_style(ax, style='clean')
    apply_style_to_axis(ax, title='Model Residuals')
    
    plt.tight_layout()
    save_figure(fig, os.path.join(output_dir, 'visualization/performance/model_evaluation.png'))

def create_performance_by_gender_plot(results_df: pd.DataFrame, output_dir: str) -> None:
    """Create visualization of performance metrics by gender."""
    # Create figure
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=FIGURE_SIZES['large'])
    
    # Plot 1: Success rate by gender
    sns.boxplot(data=results_df,
                x='gender',
                y='success_rate',
                hue='gender',
                palette=[COLOR_PALETTE['boys'], COLOR_PALETTE['girls']],
                legend=False,
                ax=ax1)
    
    sns.stripplot(data=results_df,
                  x='gender',
                  y='success_rate',
                  color=COLOR_PALETTE['neutral'],
                  alpha=ALPHA_VALUES['scatter'],
                  size=4,
                  jitter=0.2,
                  ax=ax1)
    
    # Add statistical test
    male_success = results_df[results_df['gender'] == 'M']['success_rate']
    female_success = results_df[results_df['gender'] == 'F']['success_rate']
    add_stats_if_sufficient(male_success, female_success, ax1)
    
    # Style plot 1
    ax1.set_xlabel('Gender')
    ax1.set_ylabel('Success Rate')
    set_axis_style(ax1, style='clean')
    apply_style_to_axis(ax1)
    
    # Plot 2: Completion by gender
    completion_data = pd.crosstab(results_df['gender'], results_df['solved'])
    completion_data.plot(kind='bar', stacked=True,
                        color=[COLOR_PALETTE['failure'], COLOR_PALETTE['success']],
                        alpha=ALPHA_VALUES['main_plot'],
                        ax=ax2)
    
    # Add value labels
    for c in ax2.containers:
        ax2.bar_label(c, fmt='%d', label_type='center')
    
    # Style plot 2
    ax2.set_xticks([0, 1])
    ax2.set_xticklabels(['Male', 'Female'])
    ax2.set_xlabel('Gender')
    ax2.set_ylabel('Number of Participants')
    set_axis_style(ax2, style='clean')
    apply_style_to_axis(ax2)
    ax2.legend(['Incomplete', 'Complete'])
    
    # Plot 3: Number of unlocks by gender
    sns.boxplot(data=results_df,
                x='gender',
                y='num_unlock',
                hue='gender',
                palette=[COLOR_PALETTE['boys'], COLOR_PALETTE['girls']],
                legend=False,
                ax=ax3)
    
    sns.stripplot(data=results_df,
                  x='gender',
                  y='num_unlock',
                  color=COLOR_PALETTE['neutral'],
                  alpha=ALPHA_VALUES['scatter'],
                  size=4,
                  jitter=0.2,
                  ax=ax3)
    
    # Add statistical test
    male_unlocks = results_df[results_df['gender'] == 'M']['num_unlock']
    female_unlocks = results_df[results_df['gender'] == 'F']['num_unlock']
    add_stats_if_sufficient(male_unlocks, female_unlocks, ax3)
    
    # Style plot 3
    ax3.set_xlabel('Gender')
    ax3.set_ylabel('Number of Unlocks')
    set_axis_style(ax3, style='clean')
    apply_style_to_axis(ax3)
    
    # Plot 4: Completion time by gender
    sns.boxplot(data=results_df,
                x='gender',
                y='unlock_time',
                hue='gender',
                palette=[COLOR_PALETTE['boys'], COLOR_PALETTE['girls']],
                ax=ax4,
                legend=False)
    
    sns.stripplot(data=results_df,
                  x='gender',
                  y='unlock_time',
                  color=COLOR_PALETTE['neutral'],
                  alpha=ALPHA_VALUES['scatter'],
                  size=4,
                  jitter=0.2,
                  ax=ax4)
    
    # Add statistical test
    male_time = results_df[results_df['gender'] == 'M']['unlock_time']
    female_time = results_df[results_df['gender'] == 'F']['unlock_time']
    add_stats_if_sufficient(male_time, female_time, ax4)
    
    # Style plot 4
    ax4.set_xlabel('Gender')
    ax4.set_ylabel('Completion Time')
    set_axis_style(ax4, style='clean')
    apply_style_to_axis(ax4)
    
    plt.tight_layout()
    save_figure(fig, os.path.join(output_dir, 'visualization/performance/performance_by_gender.png')) 