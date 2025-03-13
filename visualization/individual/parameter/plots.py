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
from ...utils import (
    create_figure,
    apply_style_to_axis,
    add_statistical_annotations,
    save_figure,
    set_axis_style,
    setup_3d_axis,
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

def create_correlation_heatmap(results_df: pd.DataFrame, output_dir: str) -> None:
    """Create correlation heatmap for parameters."""
    # Calculate correlations
    params = ['theta_mean', 'omega_mean', 'r_mean', 'motor']
    corr_matrix = results_df[params].corr()
    pvalues = pd.DataFrame(np.zeros_like(corr_matrix), 
                          index=corr_matrix.index, 
                          columns=corr_matrix.columns)
    
    # Calculate p-values using safe_statistical_test
    for i in range(len(params)):
        for j in range(len(params)):
            if i != j:
                stat, p_val, msg = safe_statistical_test('pearsonr', 
                                                       results_df[params[i]].values, 
                                                       results_df[params[j]].values)
                pvalues.iloc[i, j] = p_val if p_val is not None else 1.0
    
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
    vis_path = os.path.join(output_dir, 'visualization/individual/parameter/correlation_heatmap.png')
    report_path = os.path.join(output_dir, 'reports/individual/parameter/correlation_heatmap.png')
    
    # Create directories if they don't exist
    os.makedirs(os.path.dirname(vis_path), exist_ok=True)
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    
    # Save to both visualization and reports directories
    save_figure(fig, vis_path)
    save_figure(fig, report_path)

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
        
        # Add Shapiro-Wilk test result using safe_statistical_test
        stat, p_val, msg = safe_statistical_test('shapiro', data.values)
        stats_dict = {
            'Shapiro-Wilk': stat,
            'p-value': p_val,
            'note': msg
        }
        add_statistical_annotations(ax, stats_dict)
        
        # Style the plot
        ax.set_title(label)
        set_axis_style(ax, style='clean')
        apply_style_to_axis(ax)
    
    plt.tight_layout()
    
    # Save figure to both directories
    vis_path = os.path.join(output_dir, 'visualization/individual/parameter/qq_plots.png')
    report_path = os.path.join(output_dir, 'reports/individual/parameter/qq_plots.png')
    
    os.makedirs(os.path.dirname(vis_path), exist_ok=True)
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    
    save_figure(fig, vis_path)
    save_figure(fig, report_path)

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
        
        # Add statistical test using safe_statistical_test
        male_data = results_df[results_df['gender'] == 'M'][param].values
        female_data = results_df[results_df['gender'] == 'F'][param].values
        stat, p_val, msg = safe_statistical_test('ttest_ind', male_data, female_data)
        
        stats_dict = {
            't-stat': stat,
            'p-value': p_val,
            'note': msg
        }
        add_statistical_annotations(ax, stats_dict)
        
        # Style the plot
        ax.set_xlabel('Gender')
        ax.set_ylabel(label)
        set_axis_style(ax, style='clean')
        apply_style_to_axis(ax)
    
    plt.tight_layout()
    
    # Save figure to both directories
    vis_path = os.path.join(output_dir, 'visualization/individual/parameter/violin_plots.png')
    report_path = os.path.join(output_dir, 'reports/individual/parameter/violin_plots.png')
    
    os.makedirs(os.path.dirname(vis_path), exist_ok=True)
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    
    save_figure(fig, vis_path)
    save_figure(fig, report_path)

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
    
    # Save figure to both directories
    vis_path = os.path.join(output_dir, 'visualization/individual/parameter/relationships_3d.png')
    report_path = os.path.join(output_dir, 'reports/individual/parameter/relationships_3d.png')
    
    os.makedirs(os.path.dirname(vis_path), exist_ok=True)
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    
    save_figure(fig, vis_path)
    save_figure(fig, report_path)

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
        
        # Calculate correlation using safe_statistical_test
        stat, p_val, msg = safe_statistical_test('pearsonr', 
                                               results_df['age'].values, 
                                               results_df[param].values)
        stats_dict = {
            'r': stat,
            'p-value': p_val,
            'note': msg
        }
        add_statistical_annotations(ax, stats_dict)
        
        # Style the plot
        ax.set_xlabel('Age (years)')
        ax.set_ylabel(label)
        set_axis_style(ax, style='clean')
        apply_style_to_axis(ax)
    
    plt.tight_layout()
    
    # Save figure to both directories
    vis_path = os.path.join(output_dir, 'visualization/individual/parameter/age_evolution.png')
    report_path = os.path.join(output_dir, 'reports/individual/parameter/age_evolution.png')
    
    os.makedirs(os.path.dirname(vis_path), exist_ok=True)
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    
    save_figure(fig, vis_path)
    save_figure(fig, report_path)

def create_parameter_distributions_plot(results_df: pd.DataFrame, all_posteriors: Dict[str, Any], output_dir: str) -> None:
    """Create parameter distributions visualization."""
    plt.figure(figsize=FIGURE_SIZES['single'])
    
    # Parameters to plot
    params = ['theta_mean', 'omega_mean', 'r_mean']
    param_labels = ['Learning Ability (θ)', 'Social Influence (ω)', 'Exploration (r)']
    
    for i, (param, label) in enumerate(zip(params, param_labels)):
        plt.subplot(1, 3, i+1)
        
        # Create violin plot
        sns.violinplot(y=results_df[param], 
                      color=PARAM_COLORS[param.split('_')[0]])
        
        # Add individual points
        sns.swarmplot(y=results_df[param], 
                     color=COLOR_PALETTE['neutral'],
                     alpha=ALPHA_VALUES['swarm_plot'])
        
        # Add statistical summary
        stats_text = f"Mean: {results_df[param].mean():.3f}\n"
        stats_text += f"Std: {results_df[param].std():.3f}\n"
        stats_text += f"Range: [{results_df[param].min():.3f}, {results_df[param].max():.3f}]"
        
        plt.text(0.5, -0.15, stats_text,
                horizontalalignment='center',
                transform=plt.gca().transAxes,
                fontsize=8)
        
        plt.title(f'{label} Distribution')
        plt.ylabel(label)
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save figure to both directories
    vis_path = os.path.join(output_dir, 'visualization/individual/parameter/distributions.png')
    report_path = os.path.join(output_dir, 'reports/individual/parameter/distributions.png')
    
    os.makedirs(os.path.dirname(vis_path), exist_ok=True)
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    
    save_figure(plt.gcf(), vis_path)
    save_figure(plt.gcf(), report_path)
    plt.close()

def create_distribution_comparison_plot(results_df, output_dir):
    """
    Create distribution comparison visualization
    
    Args:
        results_df: DataFrame containing analysis results
        output_dir: Directory to save the plot
    """
    plt.figure(figsize=FIGURE_SIZES['default'])
    
    # Parameters to compare
    params = ['theta_mean', 'omega_mean', 'r_mean']
    param_labels = ['Learning Ability (θ)', 'Social Influence (ω)', 'Exploration (r)']
    
    # Create box plots for each parameter
    data = []
    labels = []
    for param, label in zip(params, param_labels):
        data.append(results_df[param])
        labels.extend([label] * len(results_df))
    
    # Create violin plot
    sns.violinplot(data=data, inner='box')
    
    # Customize plot
    plt.xticks(range(len(params)), param_labels)
    plt.title('Parameter Distribution Comparison')
    plt.ylabel('Value')
    plt.grid(True, alpha=0.3)
    
    # Add statistical test results
    for i in range(len(params)-1):
        t_stat, p_val = stats.ttest_ind(data[i], data[i+1])
        plt.text(i+0.5, -0.15, f't = {t_stat:.3f}, p = {p_val:.3f}',
                horizontalalignment='center',
                transform=plt.gca().transAxes,
                fontsize=8)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'distribution_analysis/distribution_comparison.png'),
                dpi=300, bbox_inches='tight')
    plt.close() 