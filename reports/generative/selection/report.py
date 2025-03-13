import numpy as np
import pandas as pd
from scipy import stats
import os
from typing import Dict, List, Tuple
from sklearn.metrics import r2_score, mean_squared_error
import arviz as az

def analyze_model_comparison(model_results: Dict[str, pd.DataFrame]) -> Dict[str, Dict[str, float]]:
    """
    Compare different model variants using information criteria
    """
    comparison_metrics = {}
    
    for model_name, results in model_results.items():
        # Calculate log likelihood
        log_likelihood = results['log_likelihood'].sum()
        
        # Get number of parameters
        n_params = len([col for col in results.columns if col not in ['log_likelihood', 'trial_id', 'participant_id', 'predicted_success']])
        
        # Get number of observations
        n_obs = len(results)
        
        # Calculate information criteria
        aic = -2 * log_likelihood + 2 * n_params
        bic = -2 * log_likelihood + np.log(n_obs) * n_params
        
        # Calculate DIC (Deviance Information Criterion)
        deviance = -2 * results['log_likelihood']
        dic = 2 * deviance.mean() - deviance.var()
        
        comparison_metrics[model_name] = {
            'log_likelihood': log_likelihood,
            'n_params': n_params,
            'n_obs': n_obs,
            'aic': aic,
            'bic': bic,
            'dic': dic
        }
    
    return comparison_metrics

def analyze_model_predictions(model_results: Dict[str, pd.DataFrame], 
                           true_data: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    """
    Analyze predictive performance of different model variants
    """
    prediction_metrics = {}
    
    for model_name, results in model_results.items():
        # Calculate prediction metrics
        mse = mean_squared_error(true_data['Worked'], results['predicted_success'])
        rmse = np.sqrt(mse)
        r2 = r2_score(true_data['Worked'], results['predicted_success'])
        
        # Calculate correlation
        correlation = stats.pearsonr(true_data['Worked'], results['predicted_success'])
        
        prediction_metrics[model_name] = {
            'mse': mse,
            'rmse': rmse,
            'r2': r2,
            'correlation': correlation[0],
            'correlation_p': correlation[1]
        }
    
    return prediction_metrics

def analyze_parameter_stability(model_results: Dict[str, pd.DataFrame]) -> Dict[str, Dict[str, Dict[str, float]]]:
    """
    Analyze parameter stability across different model variants
    """
    stability_metrics = {}
    parameters = ['theta', 'omega', 'r', 'motor_skill']
    
    for param in parameters:
        param_metrics = {}
        
        for model_name, results in model_results.items():
            if param in results.columns:
                mean_val = results[param].mean()
                std_val = results[param].std()
                
                # Calculate CV only if mean is not zero
                cv = std_val / mean_val if abs(mean_val) > 1e-10 else np.nan
                
                # Calculate stability metrics
                param_metrics[model_name] = {
                    'mean': mean_val,
                    'std': std_val,
                    'cv': cv,
                    'ci_lower': np.percentile(results[param], 2.5),
                    'ci_upper': np.percentile(results[param], 97.5)
                }
        
        stability_metrics[param] = param_metrics
    
    return stability_metrics

def generate_report(model_variants: Dict[str, pd.DataFrame],
                data: pd.DataFrame,
                output_dir: str = 'output/generative/reports/selection') -> Dict:
    """
    Generate comprehensive model selection analysis report
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Perform analyses
    comparison_metrics = analyze_model_comparison(model_variants)
    prediction_metrics = analyze_model_predictions(model_variants, data)
    stability_metrics = analyze_parameter_stability(model_variants)
    
    # Save results
    with open(os.path.join(output_dir, 'report.txt'), 'w') as f:
        # Write model comparison metrics
        f.write("Model Comparison Analysis\n")
        f.write("=======================\n\n")
        for model_name, metrics in comparison_metrics.items():
            f.write(f"{model_name}:\n")
            f.write(f"  Log Likelihood: {metrics['log_likelihood']:.3f}\n")
            f.write(f"  Number of Parameters: {metrics['n_params']}\n")
            f.write(f"  AIC: {metrics['aic']:.3f}\n")
            f.write(f"  BIC: {metrics['bic']:.3f}\n")
            f.write(f"  DIC: {metrics['dic']:.3f}\n\n")
        
        # Write prediction metrics
        f.write("\nPredictive Performance Analysis\n")
        f.write("=============================\n\n")
        for model_name, metrics in prediction_metrics.items():
            f.write(f"{model_name}:\n")
            f.write(f"  RMSE: {metrics['rmse']:.3f}\n")
            f.write(f"  R²: {metrics['r2']:.3f}\n")
            f.write(f"  Correlation: {metrics['correlation']:.3f} ")
            f.write(f"(p = {metrics['correlation_p']:.3f})\n\n")
        
        # Write parameter stability analysis
        f.write("\nParameter Stability Analysis\n")
        f.write("==========================\n\n")
        for param, model_metrics in stability_metrics.items():
            f.write(f"{param}:\n")
            for model_name, metrics in model_metrics.items():
                f.write(f"  {model_name}:\n")
                f.write(f"    Mean: {metrics['mean']:.3f}\n")
                f.write(f"    Std Dev: {metrics['std']:.3f}\n")
                f.write(f"    CV: {metrics['cv']:.3f}\n")
                f.write(f"    95% CI: [{metrics['ci_lower']:.3f}, {metrics['ci_upper']:.3f}]\n\n")
    
    # Save numerical results for later use
    results = {
        'comparison': comparison_metrics,
        'prediction': prediction_metrics,
        'stability': stability_metrics
    }
    
    return results 