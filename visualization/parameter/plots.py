"""
Parameter visualization functions.
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
    set_axis_style,
    setup_3d_axis
)
from ..config import (
    PARAM_COLORS,
    COLOR_PALETTE,
    FIGURE_SIZES,
    ALPHA_VALUES,
    CUSTOM_COLORMAPS,
    STAT_SETTINGS,
    FONT_SIZES
)

def create_correlation_heatmap(results_df: pd.DataFrame, output_dir: str) -> None:
    """Create correlation heatmap for parameters."""
    # Calculate correlations
    params = ['theta_mean', 'omega_mean', 'r_mean', 'motor']
    corr_matrix = results_df[params].corr()
    pvalues = pd.DataFrame(np.zeros_like(corr_matrix), 
                          index=corr_matrix.index, 
                          columns=corr_matrix.columns)
    
    # Calculate p-values
    for i in range(len(params)):
        for j in range(len(params)):
            if i != j:
                stat, pval = stats.pearsonr(results_df[params[i]], 
                                          results_df[params[j]])
                pvalues.iloc[i, j] = pval
    
    # Create figure
    fig, ax = create_figure(figsize='square')
    
    # Create heatmap
    sns.heatmap(corr_matrix, 
                annot=True, 
                cmap=CUSTOM_COLORMAPS['correlation'],
                vmin=-1, vmax=1,
                center=0,
                ax=ax)
    
    # Add significance markers
    for i in range(len(params)):
        for j in range(len(params)):
            if pvalues.iloc[i, j] < 0.05:
                ax.text(j + 0.5, i + 0.5, '*',
                       ha='center', va='center',
                       color='black',
                       fontsize=FONT_SIZES['annotation'])
    
    # Style the plot
    param_labels = ['Learning (θ)', 'Social (ω)', 'Exploration (r)', 'Motor']
    ax.set_xticks(np.arange(len(param_labels)))
    ax.set_yticks(np.arange(len(param_labels)))
    ax.set_xticklabels(param_labels, rotation=45)
    ax.set_yticklabels(param_labels, rotation=0)
    apply_style_to_axis(ax, 
                       title='Parameter Correlations',
                       grid=False)
    
    # Save figure
    save_figure(fig, os.path.join(output_dir, 'visualization/correlation_analysis/parameter_correlations.png'))

def create_parameter_distributions_plot(results_df: pd.DataFrame, all_posteriors: Dict, output_dir: str) -> None:
    """Create visualization of parameter distributions."""
    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=FIGURE_SIZES['large'])
    axes = axes.ravel()
    
    params = ['theta', 'omega', 'r']
    param_labels = ['Learning (θ)', 'Social (ω)', 'Exploration (r)']
    
    for i, (param, label) in enumerate(zip(params, param_labels)):
        ax = axes[i]
        
        # Plot individual distributions
        for subject_id, posteriors in all_posteriors.items():
            posterior = posteriors[param]
            weights = np.ones_like(posterior) / len(posterior)
            sns.kdeplot(data=pd.DataFrame({param: posterior}),
                       x=param,
                       alpha=ALPHA_VALUES['background'],
                       color=PARAM_COLORS[param],
                       ax=ax)
        
        # Plot population distribution
        population_data = results_df[f'{param}_mean']
        sns.kdeplot(data=pd.DataFrame({param: population_data}),
                   x=param,
                   color=PARAM_COLORS[param],
                   linewidth=2,
                   ax=ax)
        
        # Add distribution statistics
        stats_dict = {
            'Mean': population_data.mean(),
            'Std': population_data.std(),
            'Median': population_data.median()
        }
        add_statistical_annotations(ax, stats_dict)
        
        # Style the plot
        ax.set_xlabel(label)
        ax.set_ylabel('Density')
        set_axis_style(ax, style='clean')
        apply_style_to_axis(ax)
    
    # Plot 4: Success rate distribution
    ax = axes[3]
    sns.kdeplot(data=pd.DataFrame({'Success Rate': results_df['success_rate']}),
                x='Success Rate',
                color=COLOR_PALETTE['success'],
                ax=ax)
    
    # Add distribution statistics
    stats_dict = {
        'Mean': results_df['success_rate'].mean(),
        'Std': results_df['success_rate'].std(),
        'Median': results_df['success_rate'].median()
    }
    add_statistical_annotations(ax, stats_dict)
    
    # Style the plot
    ax.set_xlabel('Success Rate')
    ax.set_ylabel('Density')
    set_axis_style(ax, style='clean')
    apply_style_to_axis(ax)
    
    plt.tight_layout()
    save_figure(fig, os.path.join(output_dir, 'visualization/parameter/parameter_distributions.png'))

def create_parameter_qq_plots(results_df: pd.DataFrame, output_dir: str) -> None:
    """Create Q-Q plots for parameter distributions."""
    fig, axes = plt.subplots(2, 2, figsize=FIGURE_SIZES['large'])
    axes = axes.ravel()
    
    params = ['theta_mean', 'omega_mean', 'r_mean', 'motor']
    param_labels = ['Learning Ability (θ)', 'Social Influence (ω)', 
                   'Exploration Rate (r)', 'Motor Skill']
    
    for i, (param, label) in enumerate(zip(params, param_labels)):
        ax = axes[i]
        data = results_df[param]
        
        # Create Q-Q plot
        stats.probplot(data, dist="norm", plot=ax)
        
        # Add Shapiro-Wilk test result
        stat, pval = stats.shapiro(data)
        stats_dict = {
            'Shapiro-Wilk': stat,
            'p-value': pval
        }
        add_statistical_annotations(ax, stats_dict)
        
        # Style the plot
        ax.set_title(label)
        set_axis_style(ax, style='clean')
        apply_style_to_axis(ax)
    
    plt.tight_layout()
    save_figure(fig, os.path.join(output_dir, 'visualization/distribution_analysis/parameter_qq_plots.png'))

def create_parameter_violin_plots(results_df: pd.DataFrame, output_dir: str) -> None:
    """Create violin plots for parameter distributions by gender."""
    fig, axes = plt.subplots(2, 2, figsize=FIGURE_SIZES['large'])
    axes = axes.ravel()
    
    params = ['theta_mean', 'omega_mean', 'r_mean', 'motor']
    param_labels = ['Learning Ability (θ)', 'Social Influence (ω)', 
                   'Exploration Rate (r)', 'Motor Skill']
    
    for i, (param, label) in enumerate(zip(params, param_labels)):
        ax = axes[i]
        
        # Create violin plot
        sns.violinplot(data=results_df, x='gender', y=param, hue='gender',
                      palette=[COLOR_PALETTE['boys'], COLOR_PALETTE['girls']],
                      alpha=ALPHA_VALUES['violin_plot'], ax=ax, legend=False)
        
        # Add individual points
        sns.stripplot(data=results_df, x='gender', y=param,
                     color=COLOR_PALETTE['neutral'],
                     alpha=ALPHA_VALUES['scatter'],
                     size=4, jitter=0.2, ax=ax)
        
        # Add statistical test
        male_data = results_df[results_df['gender'] == 'M'][param]
        female_data = results_df[results_df['gender'] == 'F'][param]
        stat, pval = stats.ttest_ind(male_data, female_data)
        
        stats_dict = {
            't-stat': stat,
            'p-value': pval
        }
        add_statistical_annotations(ax, stats_dict)
        
        # Style the plot
        ax.set_xlabel('Gender')
        ax.set_ylabel(label)
        set_axis_style(ax, style='clean')
        apply_style_to_axis(ax)
    
    plt.tight_layout()
    save_figure(fig, os.path.join(output_dir, 'visualization/distribution_analysis/parameter_violin_plots.png'))

def create_parameter_relationships_3d(results_df: pd.DataFrame, output_dir: str) -> None:
    """Create 3D visualization of parameter relationships."""
    fig = plt.figure(figsize=FIGURE_SIZES['large'])
    ax = fig.add_subplot(111, projection='3d')
    
    # Get data
    x = results_df['theta_mean']
    y = results_df['omega_mean']
    z = results_df['r_mean']
    colors = results_df['success_rate']
    
    # Create scatter plot
    scatter = ax.scatter(x, y, z, c=colors, cmap=CUSTOM_COLORMAPS['continuous'],
                        alpha=ALPHA_VALUES['scatter'])
    
    # Add colorbar
    plt.colorbar(scatter, label='Success Rate')
    
    # Style the plot
    setup_3d_axis(ax,
                  xlabel='Learning Ability (θ)',
                  ylabel='Social Influence (ω)',
                  zlabel='Exploration Rate (r)')
    
    # Save figure
    save_figure(fig, os.path.join(output_dir, 'visualization/correlation_analysis/parameter_relationships_3d.png'))

def create_parameter_age_evolution(results_df: pd.DataFrame, output_dir: str) -> None:
    """Create visualization of parameter evolution with age."""
    fig, axes = plt.subplots(2, 2, figsize=FIGURE_SIZES['large'])
    axes = axes.ravel()
    
    params = ['theta_mean', 'omega_mean', 'r_mean', 'motor']
    param_labels = ['Learning Ability (θ)', 'Social Influence (ω)', 
                   'Exploration Rate (r)', 'Motor Skill']
    
    for i, (param, label) in enumerate(zip(params, param_labels)):
        ax = axes[i]
        
        # Create scatter plot
        sns.regplot(data=results_df, x='age', y=param,
                   scatter_kws={'alpha': ALPHA_VALUES['scatter'],
                              'color': PARAM_COLORS[param.split('_')[0]]},
                   line_kws={'color': COLOR_PALETTE['neutral']},
                   ax=ax)
        
        # Calculate correlation
        corr, pval = stats.pearsonr(results_df['age'], results_df[param])
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
    save_figure(fig, os.path.join(output_dir, 'visualization/parameter/parameter_age_evolution.png')) 