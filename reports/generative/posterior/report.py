import numpy as np
import pandas as pd
from scipy import stats
import os
from typing import Dict, List, Tuple
import arviz as az

def analyze_posterior_distributions(results_data: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    """
    Analyze characteristics of posterior distributions
    """
    parameters = ['theta', 'omega', 'r', 'motor_skill']
    posterior_metrics = {}
    
    for param in parameters:
        if param in results_data.columns:
            # Calculate basic statistics
            mean = results_data[param].mean()
            std = results_data[param].std()
            median = results_data[param].median()
            
            # Calculate credible intervals
            ci_95 = np.percentile(results_data[param], [2.5, 97.5])
            ci_50 = np.percentile(results_data[param], [25, 75])
            
            # Calculate distribution characteristics
            skewness = stats.skew(results_data[param])
            kurtosis = stats.kurtosis(results_data[param])
            
            posterior_metrics[param] = {
                'mean': mean,
                'std': std,
                'median': median,
                'ci_95_lower': ci_95[0],
                'ci_95_upper': ci_95[1],
                'ci_50_lower': ci_50[0],
                'ci_50_upper': ci_50[1],
                'skewness': skewness,
                'kurtosis': kurtosis
            }
    
    return posterior_metrics

def analyze_posterior_correlations(results_data: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    """
    Analyze posterior correlations between parameters
    """
    parameters = ['theta', 'omega', 'r', 'motor_skill']
    correlation_metrics = {}
    
    for i, param1 in enumerate(parameters):
        if param1 not in results_data.columns:
            continue
            
        correlation_metrics[param1] = {}
        for param2 in parameters[i+1:]:
            if param2 not in results_data.columns:
                continue
                
            # Calculate correlation coefficient and p-value
            corr, p_value = stats.pearsonr(results_data[param1], 
                                         results_data[param2])
            
            correlation_metrics[param1][param2] = {
                'correlation': corr,
                'p_value': p_value
            }
    
    return correlation_metrics

def analyze_posterior_convergence(results_data: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    """
    Analyze convergence of posterior distributions
    """
    parameters = ['theta', 'omega', 'r', 'motor_skill']
    convergence_metrics = {}
    
    for param in parameters:
        if param not in results_data.columns:
            continue
            
        # Get parameter chain
        chain = results_data[param].values
        
        # Split single chain into 4 segments for convergence metrics
        n_samples = len(chain)
        segment_size = n_samples // 4
        chains = chain[:segment_size * 4].reshape(4, -1)
        
        # Calculate Gelman-Rubin statistic
        r_hat = az.rhat(chains)
        
        # Calculate effective sample size
        n_eff = az.ess(chains)
        
        # Calculate Monte Carlo standard error
        mcse = az.mcse(chains)
        
        convergence_metrics[param] = {
            'r_hat': float(r_hat),
            'n_eff': float(n_eff),
            'mcse': float(mcse)
        }
    
    return convergence_metrics

def analyze_posterior_predictive(results_data: pd.DataFrame, 
                               true_data: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    """
    Analyze posterior predictive checks
    """
    predictive_metrics = {}
    
    # Calculate statistics of predictive distribution
    pred_mean = results_data['predicted_success'].mean()
    pred_std = results_data['predicted_success'].std()
    
    # Calculate comparison metrics with true data
    mse = ((results_data['predicted_success'] - true_data['Worked'])**2).mean()
    rmse = np.sqrt(mse)
    
    # Calculate prediction interval coverage
    ci_95 = np.percentile(results_data['predicted_success'], [2.5, 97.5], axis=0)
    coverage_95 = np.mean((true_data['Worked'] >= ci_95[0]) & 
                         (true_data['Worked'] <= ci_95[1]))
    
    predictive_metrics['summary'] = {
        'pred_mean': pred_mean,
        'pred_std': pred_std,
        'mse': mse,
        'rmse': rmse,
        'coverage_95': coverage_95
    }
    
    return predictive_metrics

def generate_report(results_data: pd.DataFrame,
                   data: pd.DataFrame,
                   output_dir: str = 'output/generative/reports/posterior') -> Dict:
    """
    Generate comprehensive posterior analysis report
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Perform analyses
    posterior_metrics = analyze_posterior_distributions(results_data)
    correlation_metrics = analyze_posterior_correlations(results_data)
    convergence_metrics = analyze_posterior_convergence(results_data)
    predictive_metrics = analyze_posterior_predictive(results_data, data)
    
    # Save results
    with open(os.path.join(output_dir, 'report.txt'), 'w') as f:
        # Write posterior distribution analysis
        f.write("Posterior Distribution Analysis\n")
        f.write("============================\n\n")
        for param, metrics in posterior_metrics.items():
            f.write(f"{param}:\n")
            f.write(f"  Mean: {metrics['mean']:.3f}\n")
            f.write(f"  Std Dev: {metrics['std']:.3f}\n")
            f.write(f"  Median: {metrics['median']:.3f}\n")
            f.write(f"  95% CI: [{metrics['ci_95_lower']:.3f}, {metrics['ci_95_upper']:.3f}]\n")
            f.write(f"  50% CI: [{metrics['ci_50_lower']:.3f}, {metrics['ci_50_upper']:.3f}]\n")
            f.write(f"  Skewness: {metrics['skewness']:.3f}\n")
            f.write(f"  Kurtosis: {metrics['kurtosis']:.3f}\n\n")
        
        # Write correlation analysis
        f.write("\nPosterior Correlation Analysis\n")
        f.write("===========================\n\n")
        for param1, correlations in correlation_metrics.items():
            for param2, metrics in correlations.items():
                f.write(f"{param1} - {param2}:\n")
                f.write(f"  Correlation: {metrics['correlation']:.3f}\n")
                f.write(f"  P-value: {metrics['p_value']:.3f}\n\n")
        
        # Write convergence analysis
        f.write("\nConvergence Analysis\n")
        f.write("===================\n\n")
        for param, metrics in convergence_metrics.items():
            f.write(f"{param}:\n")
            f.write(f"  R-hat: {metrics['r_hat']:.3f}\n")
            f.write(f"  Effective Sample Size: {metrics['n_eff']:.0f}\n")
            f.write(f"  MCSE: {metrics['mcse']:.3f}\n\n")
        
        # Write predictive analysis
        f.write("\nPosterior Predictive Analysis\n")
        f.write("===========================\n\n")
        metrics = predictive_metrics['summary']
        f.write(f"Prediction Mean: {metrics['pred_mean']:.3f}\n")
        f.write(f"Prediction Std Dev: {metrics['pred_std']:.3f}\n")
        f.write(f"RMSE: {metrics['rmse']:.3f}\n")
        f.write(f"95% Coverage: {metrics['coverage_95']:.3f}\n")
    
    # Save numerical results for later use
    results = {
        'distributions': posterior_metrics,
        'correlations': correlation_metrics,
        'convergence': convergence_metrics,
        'predictive': predictive_metrics
    }
    
    return results 