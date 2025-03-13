"""
Parameter analysis visualization functions for generative model.
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
    set_axis_style,
    safe_statistical_test,
    format_statistical_result
)
from ...config import (
    PARAM_COLORS,
    COLOR_PALETTE,
    FIGURE_SIZES,
    ALPHA_VALUES,
    CUSTOM_COLORMAPS,
    STAT_SETTINGS,
    FONT_SIZES
)

def create_parameter_distributions_plot(results_df: pd.DataFrame, output_dir: str) -> None:
    """Create visualization of parameter distributions."""
    params = ['theta', 'omega', 'r', 'motor_skill']
    param_labels = ['Learning (θ)', 'Social (ω)', 'Exploration (r)', 'Motor Skill']
    fig, axes = plt.subplots(2, 2, figsize=FIGURE_SIZES['large'])
    axes = axes.ravel()
    
    # Plot 1: Parameter distributions
    for i, (param, label) in enumerate(zip(params, param_labels)):
        sns.histplot(data=results_df[param], bins=30,
                    color=COLOR_PALETTE['primary'],
                    alpha=ALPHA_VALUES['main_plot'],
                    ax=axes[i])
        
        # Add statistical summary
        stats_text = f"{label}:\n"
        stats_text += f"Mean: {results_df[param].mean():.3f}\n"
        stats_text += f"Std: {results_df[param].std():.3f}\n"
        stats_text += f"Range: [{results_df[param].min():.3f}, {results_df[param].max():.3f}]"
        
        axes[i].text(0.5, -0.15, stats_text,
                    horizontalalignment='center',
                    transform=axes[i].transAxes,
                    fontsize=8)
        
        axes[i].set_title(f'{label} Distribution')
        axes[i].set_xlabel(label)
        axes[i].set_ylabel('Count')
        set_axis_style(axes[i], style='clean')
        apply_style_to_axis(axes[i])
    
    plt.tight_layout()
    
    # Save figure
    vis_path = os.path.join(output_dir, 'visualization/parameter/parameter_distributions.png')
    
    os.makedirs(os.path.dirname(vis_path), exist_ok=True)
    
    save_figure(fig, vis_path)
    
    # Create correlation plot
    fig, ax = plt.subplots(figsize=FIGURE_SIZES['square'])
    
    # Calculate correlations with handling for zero variance
    param_data = results_df[params]
    n_params = len(params)
    corr_matrix = np.zeros((n_params, n_params))
    
    for i in range(n_params):
        for j in range(n_params):
            x = param_data.iloc[:, i]
            y = param_data.iloc[:, j]
            
            # Check for non-zero variance in both parameters
            if x.std() > 1e-10 and y.std() > 1e-10:
                # Remove any NaN values
                valid_mask = ~(np.isnan(x) | np.isnan(y))
                x_valid = x[valid_mask]
                y_valid = y[valid_mask]
                
                if len(x_valid) > 1:  # Need at least 2 points for correlation
                    corr = np.corrcoef(x_valid, y_valid)[0, 1]
                    if not np.isnan(corr) and not np.isinf(corr):
                        corr_matrix[i, j] = corr
    
    sns.heatmap(corr_matrix,
                xticklabels=param_labels,
                yticklabels=param_labels,
                cmap=CUSTOM_COLORMAPS['diverging'],
                center=0,
                annot=True,
                fmt='.2f',
                ax=ax)
    
    ax.set_title('Parameter Correlations')
    set_axis_style(ax, style='clean')
    apply_style_to_axis(ax)
    
    plt.tight_layout()
    
    # Save figure
    vis_path = os.path.join(output_dir, 'visualization/parameter/parameter_correlations.png')
    
    os.makedirs(os.path.dirname(vis_path), exist_ok=True)
    
    save_figure(fig, vis_path)

def create_parameter_relationships_3d_plot(results_df: pd.DataFrame, output_dir: str) -> None:
    """Create 3D visualization of parameter relationships."""
    fig = plt.figure(figsize=FIGURE_SIZES['large'])
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot 3D scatter plot
    scatter = ax.scatter(results_df['theta'],
                        results_df['omega'],
                        results_df['r'],
                        c=results_df['success'],
                        cmap=CUSTOM_COLORMAPS['sequential'],
                        alpha=ALPHA_VALUES['scatter'])
    
    # Add colorbar
    plt.colorbar(scatter, label='Success Rate')
    
    # Set labels and title
    ax.set_xlabel('Learning (θ)')
    ax.set_ylabel('Social (ω)')
    ax.set_zlabel('Exploration (r)')
    ax.set_title('3D Parameter Relationships')
    
    # Add statistical annotations
    for param in ['theta', 'omega', 'r']:
        stat, p_val, msg = safe_statistical_test('pearsonr',
                                               results_df[param].values,
                                               results_df['success'].values)
        stats_dict = {
            'r': stat,
            'p-value': p_val,
            'note': msg
        }
        add_statistical_annotations(ax, stats_dict)
    
    plt.tight_layout()
    
    # Save figure
    vis_path = os.path.join(output_dir, 'visualization/parameter/parameter_relationships_3d.png')
    
    os.makedirs(os.path.dirname(vis_path), exist_ok=True)
    
    save_figure(fig, vis_path)

def create_parameter_age_evolution_plot(results_df: pd.DataFrame, output_dir: str) -> None:
    """Create visualization of parameter evolution with age."""
    fig, axes = plt.subplots(2, 2, figsize=FIGURE_SIZES['large'])
    axes = axes.ravel()
    
    # Plot 1: Parameter values by age
    params = ['theta', 'omega', 'r']
    param_labels = ['Learning (θ)', 'Social (ω)', 'Exploration (r)']
    
    for i, (param, label) in enumerate(zip(params, param_labels)):
        param_by_age = results_df.groupby('age')[param].mean()
        param_std_by_age = results_df.groupby('age')[param].std()
        
        axes[0].plot(param_by_age.index, param_by_age.values,
                    color=PARAM_COLORS[param],
                    alpha=ALPHA_VALUES['main_plot'],
                    label=label)
        axes[0].fill_between(param_by_age.index,
                           param_by_age.values - param_std_by_age.values,
                           param_by_age.values + param_std_by_age.values,
                           color=PARAM_COLORS[param],
                           alpha=ALPHA_VALUES['background'])
        
        # Add correlation statistics
        stat, p_val, msg = safe_statistical_test('pearsonr',
                                               results_df['age'].values,
                                               results_df[param].values)
        stats_dict = {
            'r': stat,
            'p-value': p_val,
            'note': msg
        }
        add_statistical_annotations(axes[0], stats_dict)
    
    axes[0].set_title('Parameter Values by Age')
    axes[0].set_xlabel('Age')
    axes[0].set_ylabel('Value')
    axes[0].legend()
    set_axis_style(axes[0], style='clean')
    apply_style_to_axis(axes[0])
    
    # Plot 2: Parameter distributions by age group
    results_df['age_group'] = pd.qcut(results_df['age'], q=4, labels=['Young', 'Mid-young', 'Mid-old', 'Old'])
    
    for i, (param, label) in enumerate(zip(params, param_labels)):
        sns.violinplot(data=results_df, x='age_group', y=param,
                      color=PARAM_COLORS[param],
                      alpha=ALPHA_VALUES['main_plot'],
                      ax=axes[1])
        
        # Add statistical test
        for j, age_group in enumerate(results_df['age_group'].unique()):
            group_data = results_df[results_df['age_group'] == age_group][param]
            other_data = results_df[results_df['age_group'] != age_group][param]
            stat, p_val, msg = safe_statistical_test('ttest_ind', group_data, other_data)
            stats_dict = {
                't-stat': stat,
                'p-value': p_val,
                'note': msg
            }
            add_statistical_annotations(axes[1], stats_dict, position=(j, group_data.mean()))
    
    axes[1].set_title('Parameter Distributions by Age Group')
    axes[1].set_xlabel('Age Group')
    axes[1].set_ylabel('Value')
    set_axis_style(axes[1], style='clean')
    apply_style_to_axis(axes[1])
    
    # Plot 3: Parameter correlations by age
    for i, (param, label) in enumerate(zip(params, param_labels)):
        corr_by_age = results_df.groupby('age')[param].corr(results_df['success'])
        axes[2].plot(corr_by_age.index, corr_by_age.values,
                    color=PARAM_COLORS[param],
                    alpha=ALPHA_VALUES['main_plot'],
                    label=label)
    
    axes[2].set_title('Parameter-Success Correlations by Age')
    axes[2].set_xlabel('Age')
    axes[2].set_ylabel('Correlation')
    axes[2].legend()
    set_axis_style(axes[2], style='clean')
    apply_style_to_axis(axes[2])
    
    # Plot 4: Parameter stability by age
    for i, (param, label) in enumerate(zip(params, param_labels)):
        stability_by_age = results_df.groupby('age')[param].std()
        axes[3].plot(stability_by_age.index, stability_by_age.values,
                    color=PARAM_COLORS[param],
                    alpha=ALPHA_VALUES['main_plot'],
                    label=label)
    
    axes[3].set_title('Parameter Stability by Age')
    axes[3].set_xlabel('Age')
    axes[3].set_ylabel('Standard Deviation')
    axes[3].legend()
    set_axis_style(axes[3], style='clean')
    apply_style_to_axis(axes[3])
    
    plt.tight_layout()
    
    # Save figure
    vis_path = os.path.join(output_dir, 'visualization/parameter/parameter_age_evolution.png')
    
    os.makedirs(os.path.dirname(vis_path), exist_ok=True)
    
    save_figure(fig, vis_path)
