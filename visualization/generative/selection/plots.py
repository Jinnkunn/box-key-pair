import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from typing import Dict

def plot_information_criteria(comparison_metrics: Dict[str, Dict[str, float]], 
                           output_dir: str):
    """
    Plot information criteria comparison
    """
    models = list(comparison_metrics.keys())
    metrics = ['aic', 'bic', 'dic']
    
    fig, axes = plt.subplots(len(metrics), 1, figsize=(10, 12))
    
    for ax, metric in zip(axes, metrics):
        values = [comparison_metrics[model][metric] for model in models]
        ax.bar(models, values)
        ax.set_title(f'{metric.upper()} Comparison')
        ax.set_ylabel(metric.upper())
        ax.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    save_path = os.path.join(output_dir, 'visualization/selection/information_criteria.png')
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()

def plot_predictive_performance(prediction_metrics: Dict[str, Dict[str, float]], 
                              output_dir: str):
    """
    Plot predictive performance comparison
    """
    models = list(prediction_metrics.keys())
    metrics = ['rmse', 'r2', 'correlation']
    
    fig, axes = plt.subplots(len(metrics), 1, figsize=(10, 12))
    
    for ax, metric in zip(axes, metrics):
        values = [prediction_metrics[model][metric] for model in models]
        ax.bar(models, values)
        ax.set_title(f'{metric.upper()} Comparison')
        ax.set_ylabel(metric.upper())
        ax.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    save_path = os.path.join(output_dir, 'visualization/selection/predictive_performance.png')
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()

def plot_parameter_stability(stability_metrics: Dict[str, Dict[str, Dict[str, float]]], 
                           output_dir: str):
    """
    Plot parameter stability comparison
    """
    parameters = list(stability_metrics.keys())
    models = list(stability_metrics[parameters[0]].keys())
    metrics = ['mean', 'std', 'cv']
    
    for param in parameters:
        fig, axes = plt.subplots(len(metrics), 1, figsize=(10, 12))
        
        for ax, metric in zip(axes, metrics):
            values = [stability_metrics[param][model][metric] for model in models]
            ax.bar(models, values)
            ax.set_title(f'{param} {metric.upper()}')
            ax.set_ylabel(metric.upper())
            ax.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        save_path = os.path.join(output_dir, f'visualization/selection/stability_{param}.png')
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        plt.close()

def plot_model_rankings(comparison_metrics: Dict[str, Dict[str, float]], 
                      output_dir: str):
    """
    Plot model rankings based on different criteria
    """
    models = list(comparison_metrics.keys())
    metrics = ['aic', 'bic', 'dic']
    
    # Create figure with subplots
    fig, axes = plt.subplots(len(metrics), 1, figsize=(10, 12))
    
    for ax, metric in zip(axes, metrics):
        values = [comparison_metrics[model][metric] for model in models]
        
        # Sort models by metric value
        sorted_indices = np.argsort(values)
        sorted_models = [models[i] for i in sorted_indices]
        sorted_values = [values[i] for i in sorted_indices]
        
        # Create horizontal bar plot
        ax.barh(sorted_models, sorted_values)
        ax.set_title(f'Model Ranking by {metric.upper()}')
        ax.set_xlabel(metric.upper())
    
    plt.tight_layout()
    save_path = os.path.join(output_dir, 'visualization/selection/model_rankings.png')
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()

def create_plots(model_variants: Dict[str, pd.DataFrame],
              analysis_results: Dict,
              output_dir: str = 'output/generative'):
    """
    Create all model selection analysis visualizations
    """
    # Plot information criteria comparison
    plot_information_criteria(analysis_results['comparison'], output_dir)
    
    # Plot model rankings
    plot_model_rankings(analysis_results['comparison'], output_dir)
    
    # Plot predictive performance comparison
    plot_predictive_performance(analysis_results['prediction'], output_dir)
    
    # Plot parameter stability comparison
    plot_parameter_stability(analysis_results['stability'], output_dir) 