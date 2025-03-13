import numpy as np
import pandas as pd
from scipy import stats
import os
from typing import Dict, List, Tuple

def analyze_recovery_accuracy(true_params: pd.DataFrame,
                            estimated_params: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    """
    Analyze the accuracy of parameter recovery
    """
    parameters = ['theta', 'omega', 'r', 'motor_skill']
    accuracy_metrics = {}
    
    for param in parameters:
        if param in true_params.columns and param in estimated_params.columns:
            mse = ((true_params[param] - estimated_params[param])**2).mean()
            rmse = np.sqrt(mse)
            mae = np.abs(true_params[param] - estimated_params[param]).mean()
            
            correlation = stats.pearsonr(true_params[param], estimated_params[param])
            
            bias = (estimated_params[param] - true_params[param]).mean()
            relative_bias = bias / true_params[param].mean()
            
            accuracy_metrics[param] = {
                'mse': mse,
                'rmse': rmse,
                'mae': mae,
                'correlation': correlation[0],
                'correlation_p': correlation[1],
                'bias': bias,
                'relative_bias': relative_bias
            }
    
    return accuracy_metrics

def analyze_recovery_consistency(true_params: pd.DataFrame,
                               estimated_params: pd.DataFrame,
                               n_bootstrap: int = 1000) -> Dict[str, Dict[str, List[float]]]:
    """
    Analyze the consistency of parameter recovery
    """
    parameters = ['theta', 'omega', 'r', 'motor_skill']
    consistency_metrics = {}
    
    for param in parameters:
        if param in true_params.columns and param in estimated_params.columns:
            bootstrap_metrics = {
                'correlation': [],
                'rmse': [],
                'bias': []
            }
            
            for _ in range(n_bootstrap):
                indices = np.random.choice(len(true_params), len(true_params), replace=True)
                true_sample = true_params.iloc[indices][param]
                est_sample = estimated_params.iloc[indices][param]
                
                bootstrap_metrics['correlation'].append(
                    stats.pearsonr(true_sample, est_sample)[0]
                )
                bootstrap_metrics['rmse'].append(
                    np.sqrt(((true_sample - est_sample)**2).mean())
                )
                bootstrap_metrics['bias'].append(
                    (est_sample - true_sample).mean()
                )
            
            consistency_metrics[param] = {
                metric: {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'ci_95': np.percentile(values, [2.5, 97.5])
                }
                for metric, values in bootstrap_metrics.items()
            }
    
    return consistency_metrics

def analyze_recovery_robustness(true_params: pd.DataFrame,
                              estimated_params: pd.DataFrame,
                              noise_levels: List[float] = [0.1, 0.2, 0.3]) -> Dict[str, Dict[str, Dict[float, float]]]:
    """
    Analyze the robustness of parameter recovery
    """
    parameters = ['theta', 'omega', 'r', 'motor_skill']
    robustness_metrics = {}
    
    for param in parameters:
        if param in true_params.columns and param in estimated_params.columns:
            param_metrics = {'rmse': {}, 'correlation': {}, 'bias': {}}
            
            for noise in noise_levels:
                # 添加噪声
                noisy_estimates = estimated_params[param] + \
                                np.random.normal(0, noise * estimated_params[param].std(), 
                                               len(estimated_params))
                
                # 计算指标
                param_metrics['rmse'][noise] = np.sqrt(
                    ((true_params[param] - noisy_estimates)**2).mean()
                )
                param_metrics['correlation'][noise] = stats.pearsonr(
                    true_params[param], noisy_estimates
                )[0]
                param_metrics['bias'][noise] = (
                    noisy_estimates - true_params[param]
                ).mean()
            
            robustness_metrics[param] = param_metrics
    
    return robustness_metrics

def generate_report(true_params: pd.DataFrame,
                   estimated_params: pd.DataFrame,
                   output_dir: str = 'output/generative/reports/recovery') -> Dict:
    """
    Generate comprehensive parameter recovery analysis report
    """
    os.makedirs(output_dir, exist_ok=True)
    
    accuracy_metrics = analyze_recovery_accuracy(true_params, estimated_params)
    consistency_metrics = analyze_recovery_consistency(true_params, estimated_params)
    robustness_metrics = analyze_recovery_robustness(true_params, estimated_params)
    
    with open(os.path.join(output_dir, 'report.txt'), 'w') as f:
        f.write("Parameter Recovery Accuracy\n")
        f.write("========================\n\n")
        for param, metrics in accuracy_metrics.items():
            f.write(f"{param}:\n")
            f.write(f"  RMSE: {metrics['rmse']:.3f}\n")
            f.write(f"  MAE: {metrics['mae']:.3f}\n")
            f.write(f"  Correlation: {metrics['correlation']:.3f} ")
            f.write(f"(p = {metrics['correlation_p']:.3f})\n")
            f.write(f"  Bias: {metrics['bias']:.3f}\n")
            f.write(f"  Relative Bias: {metrics['relative_bias']:.3f}\n\n")
        
        f.write("\nParameter Recovery Consistency\n")
        f.write("===========================\n\n")
        for param, metrics in consistency_metrics.items():
            f.write(f"{param}:\n")
            for metric, values in metrics.items():
                f.write(f"  {metric}:\n")
                f.write(f"    Mean: {values['mean']:.3f}\n")
                f.write(f"    Std Dev: {values['std']:.3f}\n")
                f.write(f"    95% CI: [{values['ci_95'][0]:.3f}, {values['ci_95'][1]:.3f}]\n")
            f.write("\n")
        
        f.write("\nParameter Recovery Robustness\n")
        f.write("==========================\n\n")
        for param, metrics in robustness_metrics.items():
            f.write(f"{param}:\n")
            for metric, noise_values in metrics.items():
                f.write(f"  {metric}:\n")
                for noise, value in noise_values.items():
                    f.write(f"    Noise {noise:.1f}: {value:.3f}\n")
            f.write("\n")
    
    results = {
        'accuracy': accuracy_metrics,
        'consistency': consistency_metrics,
        'robustness': robustness_metrics
    }
    
    return results 