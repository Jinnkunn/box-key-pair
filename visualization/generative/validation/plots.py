import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from typing import Dict

def create_plots(results_data: pd.DataFrame,
              analysis_results: Dict,
              output_dir: str = 'output/generative/visualization/validation'):
    """
    Create all validation analysis visualizations
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Set style for all plots
    plt.style.use('default')
    sns.set_theme(style="whitegrid")
    plt.rcParams.update({
        'font.size': 12,
        'axes.labelsize': 14,
        'axes.titlesize': 16,
        'figure.titlesize': 18
    })
    
    # Create prediction vs actual plot
    plt.figure(figsize=(10, 6))
    plt.scatter(results_data['predicted_success'], results_data['success'], alpha=0.5)
    plt.plot([0, 1], [0, 1], 'r--', label='Perfect prediction')
    plt.xlabel('Predicted Success')
    plt.ylabel('Actual Success')
    plt.title('Prediction vs Actual')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'prediction_vs_actual.png'))
    plt.close()
    
    # Create calibration plot
    plt.figure(figsize=(10, 6))
    n_bins = 10
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_indices = np.digitize(results_data['predicted_success'], bin_edges) - 1
    
    bin_means = []
    true_means = []
    for i in range(n_bins):
        mask = bin_indices == i
        if mask.any():
            bin_means.append(results_data['predicted_success'][mask].mean())
            true_means.append(results_data['success'][mask].mean())
    
    plt.scatter(bin_means, true_means, alpha=0.5)
    plt.plot([0, 1], [0, 1], 'r--', label='Perfect calibration')
    plt.xlabel('Predicted Success Rate')
    plt.ylabel('Actual Success Rate')
    plt.title('Calibration Plot')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'calibration.png'))
    plt.close()
    
    # Create reliability plot
    plt.figure(figsize=(10, 6))
    reliability_score = analysis_results['reliability']['reliability_score']
    resolution_score = analysis_results['reliability']['resolution_score']
    uncertainty_score = analysis_results['reliability']['uncertainty_score']
    
    metrics = ['Reliability', 'Resolution', 'Uncertainty']
    values = [reliability_score, resolution_score, uncertainty_score]
    
    plt.bar(metrics, values)
    plt.title('Model Reliability Metrics')
    plt.ylabel('Score')
    plt.xticks(range(len(metrics)), metrics, rotation=0)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'reliability.png'))
    plt.close()
    
    # Create prediction metrics plot
    plt.figure(figsize=(10, 6))
    metrics = analysis_results['prediction']
    metric_names = ['RMSE', 'MAE', 'R²']
    metric_values = [metrics['rmse'], metrics['mae'], metrics['r2']]
    
    plt.bar(metric_names, metric_values)
    plt.title('Prediction Performance Metrics')
    plt.ylabel('Value')
    plt.xticks(range(len(metric_names)), metric_names, rotation=0)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'prediction_metrics.png'))
    plt.close() 