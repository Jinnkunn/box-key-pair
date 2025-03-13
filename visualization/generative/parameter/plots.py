import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from typing import Dict

def plot_recovery_comparison(true_params: pd.DataFrame, estimated_params: pd.DataFrame, output_dir: str):
    """
    Plot comparison between true and estimated parameters
    """
    parameters = ['theta', 'omega', 'r', 'motor_skill']
    fig, axes = plt.subplots(2, 2, figsize=(15, 15))
    
    for ax, param in zip(axes.flat, parameters):
        # Create scatter plot
        ax.scatter(true_params[param], estimated_params[param], alpha=0.5)
        
        # Add perfect recovery line
        min_val = min(true_params[param].min(), estimated_params[param].min())
        max_val = max(true_params[param].max(), estimated_params[param].max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect Recovery')
        
        # Calculate and add R² value
        r2 = np.corrcoef(true_params[param], estimated_params[param])[0, 1]**2
        ax.text(0.05, 0.95, f'R² = {r2:.3f}',
                transform=ax.transAxes, verticalalignment='top')
        
        ax.set_title(f'{param} Recovery')
        ax.set_xlabel('True Value')
        ax.set_ylabel('Estimated Value')
        ax.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'recovery_comparison.png'))
    plt.close()

def plot_bias_distributions(bias_metrics: Dict, output_dir: str):
    """
    Plot distributions of parameter estimation bias
    """
    parameters = list(bias_metrics.keys())
    
    # Create figure with multiple subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 15))
    
    metrics = ['mean_bias', 'median_bias', 'std_bias', 'relative_bias']
    titles = ['Mean Bias', 'Median Bias', 'Bias Std Dev', 'Relative Bias']
    
    for ax, metric, title in zip(axes.flat, metrics, titles):
        values = [bias_metrics[param][metric] for param in parameters]
        
        # Create bar plot
        x = np.arange(len(parameters))
        bars = ax.bar(x, values)
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}',
                   ha='center', va='bottom')
        
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels(parameters)
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'bias_distributions.png'))
    plt.close()

def plot_consistency_intervals(consistency_metrics: Dict, output_dir: str):
    """
    Plot consistency intervals for recovery metrics
    """
    parameters = list(consistency_metrics.keys())
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Plot R² confidence intervals
    y_pos = np.arange(len(parameters))
    r2_means = [(ci[0] + ci[1])/2 for ci in [consistency_metrics[param]['r2_ci'] for param in parameters]]
    r2_errors = [(ci[1] - ci[0])/2 for ci in [consistency_metrics[param]['r2_ci'] for param in parameters]]
    
    ax1.errorbar(r2_means, y_pos, xerr=r2_errors, fmt='o')
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(parameters)
    ax1.set_xlabel('R² Score')
    ax1.set_title('R² Score with 95% CI')
    
    # Plot RMSE confidence intervals
    rmse_means = [(ci[0] + ci[1])/2 for ci in [consistency_metrics[param]['rmse_ci'] for param in parameters]]
    rmse_errors = [(ci[1] - ci[0])/2 for ci in [consistency_metrics[param]['rmse_ci'] for param in parameters]]
    
    ax2.errorbar(rmse_means, y_pos, xerr=rmse_errors, fmt='o')
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(parameters)
    ax2.set_xlabel('RMSE')
    ax2.set_title('RMSE with 95% CI')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'consistency_intervals.png'))
    plt.close()

def plot_bootstrap_distributions(consistency_metrics: Dict, output_dir: str):
    """
    Plot bootstrap distributions of recovery metrics
    """
    parameters = list(consistency_metrics.keys())
    fig, axes = plt.subplots(2, 2, figsize=(15, 15))
    
    for ax, param in zip(axes.flat, parameters):
        # Plot R² bootstrap distribution
        sns.histplot(consistency_metrics[param]['r2_bootstrap'], 
                    label='R²', alpha=0.5, ax=ax)
        
        # Plot RMSE bootstrap distribution
        sns.histplot(consistency_metrics[param]['rmse_bootstrap'],
                    label='RMSE', alpha=0.5, ax=ax)
        
        ax.set_title(f'Bootstrap Distributions for {param}')
        ax.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'bootstrap_distributions.png'))
    plt.close()

def create_plots(true_params: pd.DataFrame,
              estimated_params: pd.DataFrame,
              analysis_results: Dict,
              output_dir: str = 'output/generative/visualization/parameter'):
    """
    Create all parameter analysis visualizations
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Plot parameter recovery scatter plots
    parameters = ['theta', 'omega', 'r', 'motor_skill']
    fig, axes = plt.subplots(2, 2, figsize=(15, 15))
    axes = axes.flatten()
    
    for ax, param in zip(axes, parameters):
        true_values = true_params[param]
        estimated_values = estimated_params[param]
        
        # Scatter plot
        ax.scatter(true_values, estimated_values, alpha=0.5)
        
        # Add identity line
        min_val = min(true_values.min(), estimated_values.min())
        max_val = max(true_values.max(), estimated_values.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', label='Identity')
        
        # Add correlation coefficient
        corr = np.corrcoef(true_values, estimated_values)[0, 1]
        ax.text(0.05, 0.95, f'r = {corr:.3f}', 
                transform=ax.transAxes, 
                verticalalignment='top')
        
        ax.set_xlabel(f'True {param}')
        ax.set_ylabel(f'Estimated {param}')
        ax.set_title(f'{param} Recovery')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'parameter_recovery.png'))
    plt.close()
    
    # Plot correlation matrix
    correlations = np.zeros((len(parameters), len(parameters)))
    for i, param1 in enumerate(parameters):
        for j, param2 in enumerate(parameters):
            correlations[i, j] = np.corrcoef(true_params[param1], estimated_params[param2])[0, 1]
    
    plt.figure(figsize=(8, 6))
    im = plt.imshow(correlations, cmap='coolwarm', vmin=-1, vmax=1)
    plt.colorbar(im)
    
    # Set ticks and labels
    ax = plt.gca()
    ax.set_xticks(np.arange(len(parameters)))
    ax.set_yticks(np.arange(len(parameters)))
    ax.set_xticklabels(parameters)
    ax.set_yticklabels(parameters)
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    plt.title('Parameter Recovery Correlations')
    
    # Add correlation values as text
    for i in range(len(parameters)):
        for j in range(len(parameters)):
            plt.text(j, i, f'{correlations[i,j]:.2f}', 
                    ha='center', va='center')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'recovery_correlations.png'))
    plt.close()
    
    # Plot bias distributions
    plot_bias_distributions(analysis_results['bias'], output_dir)
    
    # Plot consistency intervals
    plot_consistency_intervals(analysis_results['consistency'], output_dir)
    
    # Plot bootstrap distributions
    plot_bootstrap_distributions(analysis_results['consistency'], output_dir) 