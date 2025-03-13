import numpy as np
import pandas as pd
from scipy import stats
import os
from typing import Dict, List, Tuple
from sklearn.neighbors import KernelDensity

def estimate_marginal_likelihood(results_data: pd.DataFrame, n_bootstrap: int = 100) -> Dict[str, Dict[str, float]]:
    """
    Estimate marginal likelihood using bootstrap and kernel density estimation
    """
    parameters = ['theta', 'omega', 'r', 'motor_skill']
    likelihoods = {}
    
    for param in parameters:
        # Get parameter values
        values = results_data[param].values.reshape(-1, 1)
        
        # Initialize bootstrap estimates
        bootstrap_estimates = np.zeros(n_bootstrap)
        
        # Fit KDE to full dataset first
        kde = KernelDensity(kernel='gaussian', bandwidth='scott').fit(values)
        base_ll = kde.score(values)
        
        # Perform bootstrap estimation
        for i in range(n_bootstrap):
            # Sample with replacement
            boot_sample = np.random.choice(values.flatten(), size=len(values))
            
            # Fit KDE to bootstrap sample
            kde = KernelDensity(kernel='gaussian', bandwidth='scott').fit(boot_sample.reshape(-1, 1))
            
            # Evaluate log likelihood at original points
            log_likelihood = kde.score(values)
            bootstrap_estimates[i] = log_likelihood
        
        # Calculate statistics
        mean_ll = np.mean(bootstrap_estimates)
        std_ll = np.std(bootstrap_estimates)
        ci = np.percentile(bootstrap_estimates, [2.5, 97.5])
        
        likelihoods[param] = {
            'mean_log_likelihood': mean_ll,
            'std_log_likelihood': std_ll,
            'ci_lower': ci[0],
            'ci_upper': ci[1],
            'base_log_likelihood': base_ll,
            'bootstrap_estimates': bootstrap_estimates
        }
    
    return likelihoods

def calculate_likelihood_ratios(results_data: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    """
    Calculate likelihood ratios between different parameter combinations
    """
    parameters = ['theta', 'omega', 'r', 'motor_skill']
    ratios = {}
    
    # Calculate log likelihoods for each parameter once
    param_lls = {}
    for param in parameters:
        values = results_data[param].values.reshape(-1, 1)
        kde = KernelDensity(kernel='gaussian', bandwidth='scott').fit(values)
        param_lls[param] = kde.score(values)
    
    # Calculate ratios using pre-computed log likelihoods
    for i, param1 in enumerate(parameters):
        for param2 in parameters[i+1:]:
            # Calculate ratio using pre-computed values
            ll1 = param_lls[param1]
            ll2 = param_lls[param2]
            ratio = np.exp(ll1 - ll2)
            
            ratios[f"{param1}_vs_{param2}"] = {
                'log_likelihood_ratio': ll1 - ll2,
                'likelihood_ratio': ratio
            }
    
    return ratios

def calculate_bic(results_data: pd.DataFrame) -> Dict[str, float]:
    """
    Calculate Bayesian Information Criterion for each parameter
    """
    parameters = ['theta', 'omega', 'r', 'motor_skill']
    bic = {}
    
    n_samples = len(results_data)
    
    for param in parameters:
        # Get parameter values
        values = results_data[param].values.reshape(-1, 1)
        
        # Fit KDE
        kde = KernelDensity(kernel='gaussian', bandwidth='scott').fit(values)
        
        # Calculate log likelihood
        log_likelihood = kde.score(values)
        
        # Calculate BIC (k=2 for mean and bandwidth)
        bic[param] = -2 * log_likelihood + 2 * np.log(n_samples)
    
    return bic

def generate_report(results_data: pd.DataFrame) -> Dict:
    """
    Generate comprehensive likelihood analysis report
    """
    output_dir = 'reports/likelihood'
    os.makedirs(output_dir, exist_ok=True)
    
    # Perform analyses
    marginal_likelihoods = estimate_marginal_likelihood(results_data)
    likelihood_ratios = calculate_likelihood_ratios(results_data)
    bic_values = calculate_bic(results_data)
    
    # Save results
    with open(os.path.join(output_dir, 'report.txt'), 'w') as f:
        # Write marginal likelihood analysis
        f.write("Marginal Likelihood Analysis\n")
        f.write("=========================\n\n")
        for param, metrics in marginal_likelihoods.items():
            f.write(f"{param}:\n")
            f.write(f"  Mean Log Likelihood: {metrics['mean_log_likelihood']:.3f}\n")
            f.write(f"  Std Dev: {metrics['std_log_likelihood']:.3f}\n")
            f.write(f"  95% CI: [{metrics['ci_lower']:.3f}, {metrics['ci_upper']:.3f}]\n\n")
        
        # Write likelihood ratios
        f.write("\nLikelihood Ratio Analysis\n")
        f.write("=======================\n\n")
        for param1, ratios in likelihood_ratios.items():
            for param2, ratio in ratios.items():
                f.write(f"{param1} vs {param2}: {ratio:.3f}\n")
            f.write("\n")
        
        # Write BIC analysis
        f.write("\nBIC Analysis\n")
        f.write("============\n\n")
        for param, bic in bic_values.items():
            f.write(f"{param}: {bic:.3f}\n")
    
    # Save numerical results for later use
    results = {
        'marginal_likelihoods': marginal_likelihoods,
        'likelihood_ratios': likelihood_ratios,
        'bic_values': bic_values
    }
    
    return results 