import numpy as np
import pandas as pd
from scipy import stats
import os
from typing import Dict, List, Tuple
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, r2_score

def perform_k_fold_cv(results_data: pd.DataFrame, n_splits: int = 5) -> Dict[str, Dict[str, List[float]]]:
    """
    Perform k-fold cross-validation for parameter estimation
    """
    parameters = ['theta', 'omega', 'r', 'motor_skill']
    cv_results = {param: {'mse': [], 'r2': [], 'predictions': []} for param in parameters}
    
    # Create k-fold splitter
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    for param in parameters:
        for train_idx, test_idx in kf.split(results_data):
            # Split data
            train_data = results_data.iloc[train_idx]
            test_data = results_data.iloc[test_idx]
            
            # Calculate mean parameter value from training data
            train_mean = train_data[param].mean()
            
            # Make predictions on test data
            predictions = np.full(len(test_data), train_mean)
            
            # Calculate metrics
            mse = mean_squared_error(test_data[param], predictions)
            r2 = r2_score(test_data[param], predictions)
            
            # Store results
            cv_results[param]['mse'].append(mse)
            cv_results[param]['r2'].append(r2)
            cv_results[param]['predictions'].extend(predictions)
    
    return cv_results

def analyze_prediction_errors(results_data: pd.DataFrame, cv_results: Dict) -> Dict[str, Dict[str, float]]:
    """
    Analyze prediction errors from cross-validation
    """
    parameters = ['theta', 'omega', 'r', 'motor_skill']
    error_analysis = {}
    
    for param in parameters:
        # Calculate prediction errors
        true_values = results_data[param]
        predicted_values = cv_results[param]['predictions']
        errors = true_values - predicted_values
        
        # Calculate error statistics
        error_analysis[param] = {
            'mean_error': np.mean(errors),
            'std_error': np.std(errors),
            'rmse': np.sqrt(np.mean(errors**2)),
            'mae': np.mean(np.abs(errors)),
            'r2_mean': np.mean(cv_results[param]['r2'])
        }
    
    return error_analysis

def analyze_fold_stability(cv_results: Dict) -> Dict[str, Dict[str, float]]:
    """
    Analyze stability of cross-validation results across folds
    """
    parameters = ['theta', 'omega', 'r', 'motor_skill']
    stability = {}
    
    for param in parameters:
        # Calculate statistics across folds
        mse_values = cv_results[param]['mse']
        r2_values = cv_results[param]['r2']
        
        stability[param] = {
            'mse_mean': np.mean(mse_values),
            'mse_std': np.std(mse_values),
            'mse_cv': np.std(mse_values) / np.mean(mse_values),  # Coefficient of variation
            'r2_mean': np.mean(r2_values),
            'r2_std': np.std(r2_values),
            'r2_cv': np.std(r2_values) / np.mean(r2_values)
        }
    
    return stability

def analyze_predictions(results_data: pd.DataFrame,
                      true_data: pd.DataFrame) -> Dict[str, float]:
    """
    Analyze prediction accuracy
    """
    # Calculate prediction metrics
    mse = ((results_data['predicted_success'] - true_data['Worked'])**2).mean()
    rmse = np.sqrt(mse)
    mae = np.abs(results_data['predicted_success'] - true_data['Worked']).mean()
    r2 = r2_score(true_data['Worked'], results_data['predicted_success'])
    
    # Calculate correlation
    correlation = stats.pearsonr(true_data['Worked'], results_data['predicted_success'])
    
    return {
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'correlation': correlation[0],
        'correlation_p': correlation[1]
    }

def analyze_calibration(results_data: pd.DataFrame,
                       true_data: pd.DataFrame) -> Dict[str, float]:
    """
    Analyze model calibration
    """
    # Calculate calibration error
    n_bins = 10
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_indices = np.digitize(results_data['predicted_success'], bin_edges) - 1
    
    calibration_error = 0
    for i in range(n_bins):
        mask = bin_indices == i
        if mask.any():
            pred_prob = results_data['predicted_success'][mask].mean()
            true_prob = true_data['Worked'][mask].mean()
            calibration_error += np.abs(pred_prob - true_prob)
    
    calibration_error /= n_bins
    
    # Calculate Brier score
    brier_score = ((results_data['predicted_success'] - true_data['Worked'])**2).mean()
    
    # Calculate log loss
    eps = 1e-15  # Small epsilon to avoid log(0)
    pred_probs = np.clip(results_data['predicted_success'], eps, 1 - eps)
    log_loss = -(true_data['Worked'] * np.log(pred_probs) + 
                 (1 - true_data['Worked']) * np.log(1 - pred_probs)).mean()
    
    return {
        'calibration_error': calibration_error,
        'brier_score': brier_score,
        'log_loss': log_loss
    }

def analyze_reliability(results_data: pd.DataFrame,
                       true_data: pd.DataFrame) -> Dict[str, float]:
    """
    Analyze model reliability
    """
    # Calculate reliability score
    n_bins = 10
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_indices = np.digitize(results_data['predicted_success'], bin_edges) - 1
    
    reliability_score = 0
    resolution_score = 0
    overall_success_rate = true_data['Worked'].mean()
    
    for i in range(n_bins):
        mask = bin_indices == i
        if mask.any():
            pred_prob = results_data['predicted_success'][mask].mean()
            true_prob = true_data['Worked'][mask].mean()
            n_samples = mask.sum()
            
            # Reliability score (calibration)
            reliability_score += n_samples * (pred_prob - true_prob)**2
            
            # Resolution score (discrimination)
            resolution_score += n_samples * (true_prob - overall_success_rate)**2
    
    reliability_score /= len(results_data)
    resolution_score /= len(results_data)
    
    # Calculate uncertainty score
    uncertainty_score = overall_success_rate * (1 - overall_success_rate)
    
    return {
        'reliability_score': reliability_score,
        'resolution_score': resolution_score,
        'uncertainty_score': uncertainty_score
    }

def generate_report(results_data: pd.DataFrame,
                   data: pd.DataFrame,
                   output_dir: str = 'output/generative/reports/validation') -> Dict:
    """
    Generate comprehensive validation analysis report
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Perform analyses
    prediction_metrics = analyze_predictions(results_data, data)
    calibration_metrics = analyze_calibration(results_data, data)
    reliability_metrics = analyze_reliability(results_data, data)
    
    # Save results
    with open(os.path.join(output_dir, 'report.txt'), 'w') as f:
        # Write prediction metrics
        f.write("Prediction Analysis\n")
        f.write("==================\n\n")
        f.write(f"RMSE: {prediction_metrics['rmse']:.3f}\n")
        f.write(f"MAE: {prediction_metrics['mae']:.3f}\n")
        f.write(f"R²: {prediction_metrics['r2']:.3f}\n")
        f.write(f"Correlation: {prediction_metrics['correlation']:.3f} ")
        f.write(f"(p = {prediction_metrics['correlation_p']:.3f})\n\n")
        
        # Write calibration metrics
        f.write("\nCalibration Analysis\n")
        f.write("===================\n\n")
        f.write(f"Calibration Error: {calibration_metrics['calibration_error']:.3f}\n")
        f.write(f"Brier Score: {calibration_metrics['brier_score']:.3f}\n")
        f.write(f"Log Loss: {calibration_metrics['log_loss']:.3f}\n\n")
        
        # Write reliability metrics
        f.write("\nReliability Analysis\n")
        f.write("===================\n\n")
        f.write(f"Reliability Score: {reliability_metrics['reliability_score']:.3f}\n")
        f.write(f"Resolution Score: {reliability_metrics['resolution_score']:.3f}\n")
        f.write(f"Uncertainty Score: {reliability_metrics['uncertainty_score']:.3f}\n")
    
    # Save numerical results for later use
    results = {
        'prediction': prediction_metrics,
        'calibration': calibration_metrics,
        'reliability': reliability_metrics
    }
    
    return results 