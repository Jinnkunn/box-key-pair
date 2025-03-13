import numpy as np
import pandas as pd
from scipy import stats
import os
from typing import Dict, List, Tuple
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler

def analyze_posterior_recovery(true_params: pd.DataFrame, estimated_params: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    """
    Analyze the quality of parameter posterior recovery
    """
    parameters = ['theta', 'omega', 'r', 'motor_skill']
    recovery_metrics = {}
    
    for param in parameters:
        # Calculate correlation coefficient
        correlation = stats.pearsonr(true_params[param], estimated_params[param])
        
        # Calculate R² score
        r2 = r2_score(true_params[param], estimated_params[param])
        
        # Calculate root mean squared error
        rmse = np.sqrt(np.mean((true_params[param] - estimated_params[param])**2))
        
        # Calculate normalized RMSE
        scaler = StandardScaler()
        true_scaled = scaler.fit_transform(true_params[param].values.reshape(-1, 1)).flatten()
        est_scaled = scaler.transform(estimated_params[param].values.reshape(-1, 1)).flatten()
        nrmse = np.sqrt(np.mean((true_scaled - est_scaled)**2))
        
        # Store results
        recovery_metrics[param] = {
            'correlation': correlation[0],
            'correlation_p': correlation[1],
            'r2': r2,
            'rmse': rmse,
            'nrmse': nrmse
        }
    
    return recovery_metrics

def analyze_parameter_bias(true_params: pd.DataFrame, estimated_params: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    """
    Analyze bias in parameter estimates
    """
    parameters = ['theta', 'omega', 'r', 'motor_skill']
    bias_metrics = {}
    
    for param in parameters:
        # Calculate bias statistics
        errors = estimated_params[param] - true_params[param]
        
        bias_metrics[param] = {
            'mean_bias': np.mean(errors),
            'median_bias': np.median(errors),
            'std_bias': np.std(errors),
            'relative_bias': np.mean(errors) / np.mean(true_params[param])
        }
    
    return bias_metrics

def analyze_recovery_consistency(true_params: pd.DataFrame, estimated_params: pd.DataFrame,
                              n_bootstrap: int = 1000) -> Dict[str, Dict[str, List[float]]]:
    """
    Analyze recovery consistency using bootstrap
    """
    parameters = ['theta', 'omega', 'r', 'motor_skill']
    consistency_metrics = {}
    
    for param in parameters:
        bootstrap_r2 = []
        bootstrap_rmse = []
        
        for _ in range(n_bootstrap):
            # Bootstrap sampling
            indices = np.random.choice(len(true_params), size=len(true_params), replace=True)
            true_boot = true_params[param].iloc[indices]
            est_boot = estimated_params[param].iloc[indices]
            
            # Calculate metrics
            r2 = r2_score(true_boot, est_boot)
            rmse = np.sqrt(np.mean((true_boot - est_boot)**2))
            
            bootstrap_r2.append(r2)
            bootstrap_rmse.append(rmse)
        
        # Calculate bootstrap confidence intervals
        r2_ci = np.percentile(bootstrap_r2, [2.5, 97.5])
        rmse_ci = np.percentile(bootstrap_rmse, [2.5, 97.5])
        
        consistency_metrics[param] = {
            'r2_ci': r2_ci,
            'rmse_ci': rmse_ci,
            'r2_bootstrap': bootstrap_r2,
            'rmse_bootstrap': bootstrap_rmse
        }
    
    return consistency_metrics

def generate_report(true_params: pd.DataFrame,
                estimated_params: pd.DataFrame,
                output_dir: str = 'output/generative/reports/parameter') -> Dict:
    """
    Generate comprehensive parameter analysis report
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Perform analyses
    recovery_metrics = analyze_posterior_recovery(true_params, estimated_params)
    bias_metrics = analyze_parameter_bias(true_params, estimated_params)
    consistency_metrics = analyze_recovery_consistency(true_params, estimated_params)
    
    # Save results
    with open(os.path.join(output_dir, 'report.txt'), 'w') as f:
        # Write recovery metrics
        f.write("Parameter Recovery Analysis\n")
        f.write("=========================\n\n")
        for param, metrics in recovery_metrics.items():
            f.write(f"{param}:\n")
            f.write(f"  Correlation: {metrics['correlation']:.3f} (p = {metrics['correlation_p']:.3f})\n")
            f.write(f"  R² Score: {metrics['r2']:.3f}\n")
            f.write(f"  RMSE: {metrics['rmse']:.3f}\n")
            f.write(f"  Normalized RMSE: {metrics['nrmse']:.3f}\n\n")
        
        # Write bias analysis
        f.write("\nParameter Estimation Bias Analysis\n")
        f.write("================================\n\n")
        for param, metrics in bias_metrics.items():
            f.write(f"{param}:\n")
            f.write(f"  Mean Bias: {metrics['mean_bias']:.3f}\n")
            f.write(f"  Median Bias: {metrics['median_bias']:.3f}\n")
            f.write(f"  Bias Std Dev: {metrics['std_bias']:.3f}\n")
            f.write(f"  Relative Bias: {metrics['relative_bias']:.3%}\n\n")
        
        # Write consistency analysis
        f.write("\nRecovery Consistency Analysis\n")
        f.write("===========================\n\n")
        for param, metrics in consistency_metrics.items():
            f.write(f"{param}:\n")
            f.write(f"  R² 95% CI: [{metrics['r2_ci'][0]:.3f}, {metrics['r2_ci'][1]:.3f}]\n")
            f.write(f"  RMSE 95% CI: [{metrics['rmse_ci'][0]:.3f}, {metrics['rmse_ci'][1]:.3f}]\n\n")
    
    # Save numerical results for later use
    results = {
        'recovery': recovery_metrics,
        'bias': bias_metrics,
        'consistency': consistency_metrics
    }
    
    return results 