import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from typing import Dict

def plot_uncertainty_intervals(uncertainties: Dict, output_dir: str):
    """
    Plot uncertainty intervals for each parameter
    """
    parameters = ['theta', 'omega', 'r', 'motor_skill']
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    y_pos = np.arange(len(parameters))
    means = [uncertainties[param]['ci_50_lower'] + 
             (uncertainties[param]['ci_50_upper'] - uncertainties[param]['ci_50_lower'])/2 
             for param in parameters]
    
    # Plot means
    ax.scatter(means, y_pos, color='blue', zorder=3)
    
    # Plot 95% CIs
    for i, param in enumerate(parameters):
        ax.hlines(y=i, 
                 xmin=uncertainties[param]['ci_95_lower'],
                 xmax=uncertainties[param]['ci_95_upper'],
                 color='lightgray', alpha=0.5, label='95% CI' if i == 0 else "")
    
    # Plot 50% CIs
    for i, param in enumerate(parameters):
        ax.hlines(y=i,
                 xmin=uncertainties[param]['ci_50_lower'],
                 xmax=uncertainties[param]['ci_50_upper'],
                 color='gray', alpha=0.8, linewidth=3, label='50% CI' if i == 0 else "")
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(parameters)
    ax.set_xlabel('Parameter Value')
    ax.set_title('Parameter Estimates with Uncertainty Intervals')
    ax.legend()
    
    plt.tight_layout()
    save_path = os.path.join(output_dir, 'visualization/uncertainty/uncertainty_intervals.png')
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()

def plot_convergence_diagnostics(results_data: pd.DataFrame, output_dir: str):
    """
    Plot convergence diagnostics for each parameter
    """
    parameters = ['theta', 'omega', 'r', 'motor_skill']
    fig, axes = plt.subplots(2, 2, figsize=(15, 15))
    
    for ax, param in zip(axes.flat, parameters):
        # Calculate running statistics
        running_mean = results_data[param].expanding().mean()
        running_std = results_data[param].expanding().std()
        
        # Plot running mean
        ax.plot(running_mean, label='Running Mean', color='blue')
        
        # Plot running mean ± running std
        ax.fill_between(range(len(running_mean)),
                       running_mean - running_std,
                       running_mean + running_std,
                       alpha=0.2, color='blue',
                       label='±1 Std Dev')
        
        ax.set_title(f'Convergence of {param}')
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Parameter Value')
        ax.legend()
    
    plt.tight_layout()
    save_path = os.path.join(output_dir, 'visualization/uncertainty/convergence_diagnostics.png')
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()

def plot_ess_comparison(ess: Dict[str, float], output_dir: str):
    """
    Plot comparison of effective sample sizes
    """
    parameters = list(ess.keys())
    values = list(ess.values())
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(parameters, values)
    
    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}',
                ha='center', va='bottom')
    
    plt.title('Effective Sample Size by Parameter')
    plt.ylabel('ESS')
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    save_path = os.path.join(output_dir, 'visualization/uncertainty/effective_sample_sizes.png')
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()

def plot_coefficient_variation(uncertainties: Dict, output_dir: str):
    """
    Plot coefficient of variation for each parameter
    """
    parameters = list(uncertainties.keys())
    cv_values = [uncertainties[param]['cv'] for param in parameters]
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(parameters, cv_values)
    
    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}',
                ha='center', va='bottom')
    
    plt.title('Coefficient of Variation by Parameter')
    plt.ylabel('CV (σ/μ)')
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    save_path = os.path.join(output_dir, 'visualization/uncertainty/coefficient_variation.png')
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()

def create_plots(results_data: pd.DataFrame,
              analysis_results: Dict,
              output_dir: str = 'output/generative'):
    """
    Create all uncertainty analysis visualizations
    """
    # Create visualizations
    plot_uncertainty_intervals(analysis_results['uncertainties'], output_dir)
    plot_convergence_diagnostics(results_data, output_dir)
    plot_ess_comparison(analysis_results['ess'], output_dir)
    plot_coefficient_variation(analysis_results['uncertainties'], output_dir) 