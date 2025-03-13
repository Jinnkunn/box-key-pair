import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from typing import Dict

def plot_marginal_likelihood_distributions(likelihoods: Dict, output_dir: str):
    """
    Plot bootstrap distributions of marginal likelihoods
    """
    parameters = ['theta', 'omega', 'r', 'motor_skill']
    fig, axes = plt.subplots(2, 2, figsize=(15, 15))
    
    for ax, param in zip(axes.flat, parameters):
        # Get bootstrap estimates
        estimates = likelihoods[param]['bootstrap_estimates']
        
        # Plot histogram with KDE
        sns.histplot(estimates, kde=True, ax=ax)
        
        # Add vertical lines for mean and CI
        ax.axvline(likelihoods[param]['mean_log_likelihood'], 
                  color='red', linestyle='--', label='Mean')
        ax.axvline(likelihoods[param]['ci_lower'],
                  color='gray', linestyle=':', label='95% CI')
        ax.axvline(likelihoods[param]['ci_upper'],
                  color='gray', linestyle=':')
        
        ax.set_title(f'Marginal Likelihood Distribution for {param}')
        ax.set_xlabel('Log Likelihood')
        ax.legend()
    
    plt.tight_layout()
    save_path = os.path.join(output_dir, 'visualization/likelihood/marginal_likelihood_distributions.png')
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()

def plot_likelihood_ratio_comparison(ratios: Dict, output_dir: str):
    """
    Plot comparison of likelihood ratios
    """
    comparisons = list(ratios.keys())
    log_ratios = [stats['log_likelihood_ratio'] for stats in ratios.values()]
    
    plt.figure(figsize=(12, 6))
    bars = plt.barh(comparisons, log_ratios)
    
    # Add zero line
    plt.axvline(x=0, color='black', linestyle='--', alpha=0.3)
    
    # Add value labels
    for bar in bars:
        width = bar.get_width()
        plt.text(width, bar.get_y() + bar.get_height()/2,
                f'{width:.2f}',
                ha='left' if width >= 0 else 'right', va='center')
    
    plt.title('Log Likelihood Ratios Between Parameters')
    plt.xlabel('Log Likelihood Ratio')
    
    plt.tight_layout()
    save_path = os.path.join(output_dir, 'visualization/likelihood/likelihood_ratios.png')
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()

def plot_bic_comparison(bic: Dict[str, float], output_dir: str):
    """
    Plot comparison of BIC values
    """
    parameters = list(bic.keys())
    values = list(bic.values())
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(parameters, values)
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.0f}',
                ha='center', va='bottom')
    
    plt.title('Bayesian Information Criterion by Parameter')
    plt.ylabel('BIC')
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    save_path = os.path.join(output_dir, 'visualization/likelihood/bic_comparison.png')
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()

def plot_likelihood_surface(results_data: pd.DataFrame, output_dir: str):
    """
    Plot 2D likelihood surface for pairs of parameters
    """
    parameters = ['theta', 'omega', 'r', 'motor_skill']
    
    # Create all possible pairs
    for i, param1 in enumerate(parameters):
        for param2 in parameters[i+1:]:
            plt.figure(figsize=(10, 8))
            
            # Create 2D KDE plot
            sns.kdeplot(data=results_data, x=param1, y=param2,
                       fill=True, cmap='viridis')
            
            plt.title(f'Likelihood Surface: {param1} vs {param2}')
            
            plt.tight_layout()
            save_path = os.path.join(output_dir, f'visualization/likelihood/likelihood_surface_{param1}_{param2}.png')
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path)
            plt.close()

def create_plots(results_data: pd.DataFrame,
              analysis_results: Dict,
              output_dir: str = 'output/generative'):
    """
    Create all likelihood analysis visualizations
    """
    # Create visualizations
    plot_marginal_likelihood_distributions(analysis_results['marginal_likelihoods'], output_dir)
    plot_likelihood_ratio_comparison(analysis_results['likelihood_ratios'], output_dir)
    plot_bic_comparison(analysis_results['bic_values'], output_dir)
    plot_likelihood_surface(results_data, output_dir) 