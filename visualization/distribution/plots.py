import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from scipy import stats
import os
from visualization.config import (
    PARAM_COLORS, COLOR_PALETTE, FIGURE_SIZES, ALPHA_VALUES
)

def create_parameter_distributions_plot(results_df, output_dir):
    """
    Create parameter distributions visualization
    
    Args:
        results_df: DataFrame containing analysis results
        output_dir: Directory to save the plot
    """
    plt.figure(figsize=FIGURE_SIZES['default'])
    
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
    plt.savefig(os.path.join(output_dir, 'distribution_analysis/parameter_distributions.png'),
                dpi=300, bbox_inches='tight')
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