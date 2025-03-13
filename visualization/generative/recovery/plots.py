import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from typing import Dict, List

def plot_recovery_accuracy(accuracy_metrics: Dict[str, Dict[str, float]], 
                         true_params: pd.DataFrame,
                         estimated_params: pd.DataFrame,
                         output_dir: str):
    """
    绘制参数恢复准确性图
    """
    parameters = list(accuracy_metrics.keys())
    
    # 创建散点图矩阵
    fig, axes = plt.subplots(2, 2, figsize=(15, 15))
    axes = axes.flatten()
    
    for i, param in enumerate(parameters):
        # 绘制散点图
        ax = axes[i]
        ax.scatter(true_params[param], estimated_params[param], alpha=0.5)
        
        # 添加对角线
        lims = [
            np.min([ax.get_xlim(), ax.get_ylim()]),
            np.max([ax.get_xlim(), ax.get_ylim()])
        ]
        ax.plot(lims, lims, 'r--', alpha=0.75, zorder=0)
        
        # 添加相关系数和RMSE
        corr = accuracy_metrics[param]['correlation']
        rmse = accuracy_metrics[param]['rmse']
        ax.text(0.05, 0.95, f'r = {corr:.3f}\nRMSE = {rmse:.3f}',
                transform=ax.transAxes, verticalalignment='top')
        
        ax.set_title(f'{param} Recovery')
        ax.set_xlabel('True Value')
        ax.set_ylabel('Estimated Value')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'recovery_accuracy.png'))
    plt.close()

def plot_recovery_consistency(consistency_metrics: Dict[str, Dict[str, Dict[str, float]]], 
                            output_dir: str):
    """
    绘制参数恢复一致性图
    """
    parameters = list(consistency_metrics.keys())
    metrics = ['correlation', 'rmse', 'bias']
    
    fig, axes = plt.subplots(len(metrics), 1, figsize=(10, 15))
    
    for i, metric in enumerate(metrics):
        means = []
        errors = []
        
        for param in parameters:
            mean = consistency_metrics[param][metric]['mean']
            ci = consistency_metrics[param][metric]['ci_95']
            means.append(mean)
            errors.append([mean - ci[0], ci[1] - mean])
        
        # 绘制误差条
        axes[i].errorbar(means, parameters, xerr=np.array(errors).T, fmt='o')
        axes[i].set_title(f'Bootstrap {metric.capitalize()} Distribution')
        axes[i].grid(True)
        
        if metric == 'correlation':
            axes[i].set_xlim(-1, 1)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'recovery_consistency.png'))
    plt.close()

def plot_recovery_robustness(robustness_metrics: Dict[str, Dict[str, Dict[float, float]]], 
                           output_dir: str):
    """
    绘制参数恢复稳健性图
    """
    parameters = list(robustness_metrics.keys())
    metrics = ['rmse', 'correlation', 'bias']
    noise_levels = sorted(list(robustness_metrics[parameters[0]]['rmse'].keys()))
    
    fig, axes = plt.subplots(len(metrics), 1, figsize=(10, 15))
    
    for i, metric in enumerate(metrics):
        for param in parameters:
            values = [robustness_metrics[param][metric][noise] for noise in noise_levels]
            axes[i].plot(noise_levels, values, 'o-', label=param)
        
        axes[i].set_title(f'{metric.capitalize()} vs Noise Level')
        axes[i].set_xlabel('Noise Level')
        axes[i].set_ylabel(metric.capitalize())
        axes[i].grid(True)
        axes[i].legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'recovery_robustness.png'))
    plt.close()

def plot_recovery_summary(accuracy_metrics: Dict[str, Dict[str, float]], 
                         output_dir: str):
    """
    绘制参数恢复总结图
    """
    parameters = list(accuracy_metrics.keys())
    metrics = ['rmse', 'mae', 'correlation', 'relative_bias']
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 15))
    axes = axes.flatten()
    
    for i, metric in enumerate(metrics):
        values = [accuracy_metrics[param][metric] for param in parameters]
        
        # 创建条形图
        ax = axes[i]
        bars = ax.bar(parameters, values)
        
        # 添加数值标签
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}',
                   ha='center', va='bottom')
        
        ax.set_title(f'Parameter {metric.replace("_", " ").title()}')
        ax.set_ylabel(metric.upper())
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'recovery_summary.png'))
    plt.close()

def create_plots(true_params: pd.DataFrame,
                estimated_params: pd.DataFrame,
                analysis_results: Dict,
                output_dir: str = 'output/generative/visualization/recovery'):
    """
    Create all parameter recovery analysis visualizations
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
        corr = analysis_results['accuracy'][param]['correlation']
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
    correlations = np.array([[analysis_results['accuracy'][p]['correlation'] for p in parameters] 
                            for p in parameters])
    
    plt.figure(figsize=(8, 6))
    im = plt.imshow(correlations, cmap='coolwarm', vmin=-1, vmax=1)
    plt.colorbar(im)
    
    plt.xticks(range(len(parameters)), parameters, rotation=45)
    plt.yticks(range(len(parameters)), parameters)
    plt.title('Parameter Recovery Correlations')
    
    # Add correlation values as text
    for i in range(len(parameters)):
        for j in range(len(parameters)):
            plt.text(j, i, f'{correlations[i,j]:.2f}', 
                    ha='center', va='center')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'recovery_correlations.png'))
    plt.close() 