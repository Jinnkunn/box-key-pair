import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from typing import Dict, List
import arviz as az

def plot_posterior_distributions(posterior_metrics: Dict[str, Dict[str, float]], 
                               results_data: pd.DataFrame,
                               output_dir: str):
    """
    绘制后验分布图
    """
    parameters = list(posterior_metrics.keys())
    n_params = len(parameters)
    
    # 创建子图网格
    fig = plt.figure(figsize=(15, 5))
    
    for i, param in enumerate(parameters, 1):
        plt.subplot(1, n_params, i)
        
        # 绘制直方图和核密度估计
        sns.histplot(results_data[param], kde=True)
        
        # 添加均值和可信区间
        plt.axvline(posterior_metrics[param]['mean'], color='r', linestyle='--', label='Mean')
        plt.axvline(posterior_metrics[param]['ci_95_lower'], color='g', linestyle=':', label='95% CI')
        plt.axvline(posterior_metrics[param]['ci_95_upper'], color='g', linestyle=':')
        
        plt.title(f'{param} Posterior Distribution')
        plt.xlabel('Value')
        plt.ylabel('Density')
        plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'posterior_distributions.png'))
    plt.close()

def plot_posterior_correlations(correlation_metrics: Dict[str, Dict[str, Dict[str, float]]], 
                              results_data: pd.DataFrame,
                              output_dir: str):
    """
    Plot correlation heatmap for posterior parameters
    """
    # Create correlation matrix
    parameters = ['theta', 'omega', 'r', 'motor_skill']
    n_params = len(parameters)
    corr_matrix = np.zeros((n_params, n_params))
    
    # Fill correlation matrix
    for i, param1 in enumerate(parameters):
        for j, param2 in enumerate(parameters):
            if i < j and param1 in correlation_metrics and param2 in correlation_metrics[param1]:
                corr = correlation_metrics[param1][param2]['correlation']
                corr_matrix[i, j] = corr
                corr_matrix[j, i] = corr
            elif i == j:
                corr_matrix[i, j] = 1.0
    
    # Create figure
    plt.figure(figsize=(10, 8))
    sns.heatmap(pd.DataFrame(corr_matrix, index=parameters, columns=parameters),
                annot=True, cmap='RdBu_r', vmin=-1, vmax=1, center=0)
    plt.title('Parameter Correlations')
    plt.tight_layout()
    
    # Save plot
    plt.savefig(os.path.join(output_dir, 'correlations.png'))
    plt.close()
    
    # 使用 ArviZ 绘制联合分布图
    data_dict = {param: results_data[param].values for param in parameters}
    az_data = az.convert_to_dataset(data_dict)
    az.plot_pair(az_data, var_names=list(data_dict.keys()),
                 kind='scatter', marginals=True, figsize=(12, 12))
    plt.savefig(os.path.join(output_dir, 'posterior_joint.png'))
    plt.close()

def plot_convergence_diagnostics(convergence_metrics: Dict[str, Dict[str, float]], 
                               results_data: pd.DataFrame,
                               output_dir: str):
    """
    Plot convergence diagnostics for posterior parameters
    """
    parameters = list(convergence_metrics.keys())
    
    # Plot R-hat and ESS
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    # R-hat plot
    r_hat_values = [metrics['r_hat'] for metrics in convergence_metrics.values()]
    ax1.bar(parameters, r_hat_values)
    ax1.axhline(y=1.1, color='r', linestyle='--', label='R-hat threshold')
    ax1.set_title('R-hat Values')
    ax1.set_ylabel('R-hat')
    ax1.tick_params(axis='x', rotation=45)
    ax1.legend()
    
    # ESS plot
    ess_values = [metrics['n_eff'] for metrics in convergence_metrics.values()]
    ax2.bar(parameters, ess_values)
    ax2.set_title('Effective Sample Size')
    ax2.set_ylabel('ESS')
    ax2.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'convergence_diagnostics.png'))
    plt.close()
    
    # Plot parameter traces
    for param in parameters:
        plt.figure(figsize=(10, 6))
        chain = results_data[param].values
        
        # Plot moving average to smooth the trace
        window_size = 50
        moving_avg = pd.Series(chain).rolling(window=window_size).mean()
        
        plt.plot(chain, alpha=0.3, label='Raw trace')
        plt.plot(moving_avg, alpha=0.8, label=f'Moving average (window={window_size})')
        
        plt.title(f'{param} Parameter Trace')
        plt.xlabel('Iteration')
        plt.ylabel('Value')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'trace_{param}.png'))
        plt.close()

def plot_posterior_predictive(predictive_metrics: Dict[str, Dict[str, float]], 
                            results_data: pd.DataFrame,
                            true_data: pd.DataFrame,
                            output_dir: str):
    """
    Plot posterior predictive check
    """
    # Plot scatter plot of predicted vs actual values
    plt.figure(figsize=(8, 8))
    plt.scatter(true_data['Worked'], results_data['predicted_success'], alpha=0.5)
    plt.plot([0, 1], [0, 1], 'r--', label='Perfect prediction')
    plt.xlabel('Observed Success')
    plt.ylabel('Predicted Success')
    plt.title('Posterior Predictive Check')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'posterior_predictive_scatter.png'))
    plt.close()
    
    # Plot comparison of predicted and actual distributions
    plt.figure(figsize=(10, 6))
    sns.kdeplot(data=true_data['Worked'], label='Observed', color='blue')
    sns.kdeplot(data=results_data['predicted_success'], label='Predicted', color='red')
    plt.xlabel('Success Rate')
    plt.ylabel('Density')
    plt.title('Observed vs Predicted Distributions')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'posterior_predictive_dist.png'))
    plt.close()

def create_plots(results_data: pd.DataFrame,
                true_data: pd.DataFrame,
                analysis_results: Dict,
                output_dir: str = 'output/generative/visualization/posterior'):
    """
    Create all posterior analysis visualizations
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Set plotting style
    plt.style.use('default')
    sns.set_theme(style="whitegrid")
    plt.rcParams.update({
        'font.size': 12,
        'axes.labelsize': 14,
        'axes.titlesize': 16,
        'figure.titlesize': 18
    })
    
    # 绘制后验分布图
    plot_posterior_distributions(analysis_results['distributions'], 
                               results_data, output_dir)
    
    # 绘制相关性图
    plot_posterior_correlations(analysis_results['correlations'], 
                              results_data, output_dir)
    
    # 绘制收敛诊断图
    plot_convergence_diagnostics(analysis_results['convergence'], 
                               results_data, output_dir)
    
    # 绘制后验预测检验图
    plot_posterior_predictive(analysis_results['predictive'], 
                            results_data, true_data, output_dir) 