import numpy as np
import pandas as pd
from scipy import stats
import os
from typing import Dict, List, Tuple
import arviz as az

def calculate_parameter_uncertainties(results_data: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    """
    Calculate various uncertainty metrics for model parameters
    """
    parameters = ['theta', 'omega', 'r', 'motor_skill']
    uncertainties = {}
    
    for param in parameters:
        # Calculate standard deviation and variance
        std = results_data[param].std()
        var = results_data[param].var()
        
        # Calculate credible intervals using percentiles
        ci_95 = np.percentile(results_data[param], [2.5, 97.5])
        ci_50 = np.percentile(results_data[param], [25, 75])
        
        # Calculate coefficient of variation
        cv = std / results_data[param].mean()
        
        # Store results
        uncertainties[param] = {
            'std': std,
            'variance': var,
            'ci_95_lower': ci_95[0],
            'ci_95_upper': ci_95[1],
            'ci_50_lower': ci_50[0],
            'ci_50_upper': ci_50[1],
            'cv': cv
        }
    
    return uncertainties

def calculate_effective_sample_size(results_data: pd.DataFrame) -> Dict[str, float]:
    """
    Calculate effective sample size for each parameter using autocorrelation
    """
    parameters = ['theta', 'omega', 'r', 'motor_skill']
    ess = {}
    
    for param in parameters:
        # Get the parameter values
        values = results_data[param].values
        n = len(values)
        
        # Calculate autocorrelation up to lag 100 or n/3, whichever is smaller
        max_lag = min(100, n // 3)
        rho = np.zeros(max_lag)
        
        # Calculate mean and variance
        mean = np.mean(values)
        var = np.var(values)
        
        # Calculate autocorrelation for each lag
        for lag in range(max_lag):
            c = np.mean((values[lag:] - mean) * (values[:-lag if lag else None] - mean))
            rho[lag] = c / var
        
        # Find where autocorrelation drops below 0.05 or use all lags
        cutoff = np.where(np.abs(rho) < 0.05)[0]
        tau = len(rho) if len(cutoff) == 0 else cutoff[0]
        
        # Calculate ESS
        ess[param] = n / (1 + 2 * np.sum(rho[:tau]))
    
    return ess

def analyze_convergence(results_data: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    """
    Analyze convergence of parameter estimates
    """
    parameters = ['theta', 'omega', 'r', 'motor_skill']
    convergence = {}
    
    for param in parameters:
        # Calculate running means
        running_mean = results_data[param].expanding().mean()
        
        # Calculate running standard deviations
        running_std = results_data[param].expanding().std()
        
        # Calculate relative change in last 10% of samples
        n = len(results_data)
        last_10_percent = int(n * 0.1)
        relative_change = abs(running_mean.iloc[-1] - running_mean.iloc[-last_10_percent]) / running_mean.iloc[-1]
        
        convergence[param] = {
            'final_mean': running_mean.iloc[-1],
            'final_std': running_std.iloc[-1],
            'relative_change': relative_change
        }
    
    return convergence

def generate_report(results_data: pd.DataFrame,
                output_dir: str = 'output/generative/reports/uncertainty') -> Dict:
    """
    Generate comprehensive uncertainty analysis report
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Perform analyses
    uncertainties = calculate_parameter_uncertainties(results_data)
    ess = calculate_effective_sample_size(results_data)
    convergence = analyze_convergence(results_data)
    
    # Save results
    with open(os.path.join(output_dir, 'report.txt'), 'w') as f:
        # Write parameter uncertainties
        f.write("Parameter Uncertainties\n")
        f.write("=====================\n\n")
        for param, metrics in uncertainties.items():
            f.write(f"{param}:\n")
            f.write(f"  Standard Deviation: {metrics['std']:.3f}\n")
            f.write(f"  Variance: {metrics['variance']:.3f}\n")
            f.write(f"  95% CI: [{metrics['ci_95_lower']:.3f}, {metrics['ci_95_upper']:.3f}]\n")
            f.write(f"  50% CI: [{metrics['ci_50_lower']:.3f}, {metrics['ci_50_upper']:.3f}]\n")
            f.write(f"  CV: {metrics['cv']:.3f}\n\n")
        
        # Write effective sample size
        f.write("\nEffective Sample Size\n")
        f.write("===================\n\n")
        for param, value in ess.items():
            f.write(f"{param}: {value:.0f}\n")
        
        # Write convergence analysis
        f.write("\nConvergence Analysis\n")
        f.write("===================\n\n")
        for param, metrics in convergence.items():
            f.write(f"{param}:\n")
            f.write(f"  Final Mean: {metrics['final_mean']:.3f}\n")
            f.write(f"  Final Std: {metrics['final_std']:.3f}\n")
            f.write(f"  Relative Change: {metrics['relative_change']:.3f}\n\n")
    
    # Save numerical results for later use
    results = {
        'uncertainties': uncertainties,
        'ess': ess,
        'convergence': convergence
    }
    
    return results 