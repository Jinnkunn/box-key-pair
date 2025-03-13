"""
Learning dynamics report generation functions.
"""

import numpy as np
import pandas as pd
from scipy import stats
import os
from typing import Dict, Any
from ...utils import (
    safe_statistical_test,
    format_statistical_result,
    create_report_header,
    create_section_header,
    format_descriptive_stats
)

def generate_learning_dynamics_report(results_df: pd.DataFrame, all_trajectories: dict, output_dir: str):
    """
    Generate learning dynamics analysis report
    
    Args:
        results_df: DataFrame containing analysis results
        all_trajectories: Dictionary containing parameter trajectories
        output_dir: Directory to save the report
    """
    report = []
    report.append("Learning Dynamics Analysis Report")
    report.append("=" * 50)
    
    # 1. Learning Speed Analysis
    report.append("\n1. Learning Speed Analysis")
    report.append("-" * 30)
    
    # Calculate learning speed for each participant
    learning_speeds = []
    for subject_id, trajectories in all_trajectories.items():
        theta_trajectory = np.average(trajectories['theta'], axis=1)
        learning_speed = np.mean(np.diff(theta_trajectory))
        learning_speeds.append({
            'ID': subject_id,
            'learning_speed': learning_speed,
            'solved': results_df[results_df['ID'] == subject_id]['solved'].iloc[0]
        })
    
    learning_speeds_df = pd.DataFrame(learning_speeds)
    
    # Compare learning speeds between successful and unsuccessful participants
    successful = learning_speeds_df[learning_speeds_df['solved'] == 1]['learning_speed']
    unsuccessful = learning_speeds_df[learning_speeds_df['solved'] == 0]['learning_speed']
    
    t_stat, p_val = stats.ttest_ind(successful, unsuccessful)
    report.append("\nLearning Speed Comparison:")
    report.append(f"Successful participants: {successful.mean():.3f} ± {successful.std():.3f}")
    report.append(f"Unsuccessful participants: {unsuccessful.mean():.3f} ± {unsuccessful.std():.3f}")
    report.append(f"T-test results: t={t_stat:.3f}, p={p_val:.3f}")
    
    # 2. Parameter Evolution Analysis
    report.append("\n2. Parameter Evolution Analysis")
    report.append("-" * 30)
    
    params = ['theta', 'omega', 'r']
    param_labels = {
        'theta': 'Learning Ability (θ)',
        'omega': 'Social Influence (ω)',
        'r': 'Exploration Rate (r)'
    }
    
    for param in params:
        report.append(f"\n{param_labels[param]} Evolution Analysis:")
        
        # Calculate parameter evolution
        max_trials = max(traj[param].shape[0] for traj in all_trajectories.values())
        param_data = np.zeros((len(all_trajectories), max_trials))
        
        for i, (subject_id, trajectories) in enumerate(all_trajectories.items()):
            n_trials = trajectories[param].shape[0]
            param_data[i, :n_trials] = np.average(trajectories[param], axis=1)
            param_data[i, n_trials:] = param_data[i, n_trials-1]
        
        # Calculate statistics
        mean_evolution = np.nanmean(param_data, axis=0)
        std_evolution = np.nanstd(param_data, axis=0)
        
        report.append(f"Initial value: {mean_evolution[0]:.3f} ± {std_evolution[0]:.3f}")
        report.append(f"Final value: {mean_evolution[-1]:.3f} ± {std_evolution[-1]:.3f}")
        report.append(f"Change: {mean_evolution[-1] - mean_evolution[0]:.3f}")
        
        # Calculate trend
        slope, intercept, r_value, p_value, std_err = stats.linregress(
            range(len(mean_evolution)), mean_evolution
        )
        report.append(f"Linear trend: slope={slope:.3f}, R²={r_value**2:.3f}, p={p_value:.3f}")
    
    # 3. Learning Strategy Analysis
    report.append("\n3. Learning Strategy Analysis")
    report.append("-" * 30)
    
    strategies = ['color_match_rate', 'num_match_rate', 'shape_match_rate']
    strategy_labels = {
        'color_match_rate': 'Color Matching',
        'num_match_rate': 'Number Matching',
        'shape_match_rate': 'Shape Matching'
    }
    
    # Calculate strategy usage and success rates
    for strategy in strategies:
        report.append(f"\n{strategy_labels[strategy]} Strategy:")
        usage_rate = results_df[strategy].mean()
        success_rate = results_df[results_df[strategy] > 0]['success_rate'].mean()
        report.append(f"Usage rate: {usage_rate:.2%}")
        report.append(f"Success rate: {success_rate:.2%}")
        
        # Correlation analysis
        corr, p = stats.pearsonr(results_df[strategy], results_df['success_rate'])
        report.append(f"Correlation with success rate: r={corr:.3f}, p={p:.3f}")
    
    # 4. Age and Gender Effects on Learning
    report.append("\n4. Age and Gender Effects on Learning")
    report.append("-" * 30)
    
    # Age correlations
    age_corr = {}
    for param in ['theta_mean', 'omega_mean', 'r_mean']:
        corr, p = stats.pearsonr(results_df['age'], results_df[param])
        age_corr[param] = (corr, p)
    
    report.append("\nAge Correlation Analysis:")
    report.append(f"Learning ability vs Age: r={age_corr['theta_mean'][0]:.3f}, p={age_corr['theta_mean'][1]:.3f}")
    report.append(f"Social influence vs Age: r={age_corr['omega_mean'][0]:.3f}, p={age_corr['omega_mean'][1]:.3f}")
    report.append(f"Exploration rate vs Age: r={age_corr['r_mean'][0]:.3f}, p={age_corr['r_mean'][1]:.3f}")
    
    # Gender differences
    report.append("\nGender Differences Analysis:")
    for param in ['theta_mean', 'omega_mean', 'r_mean']:
        boys = results_df[results_df['gender'] == 'Boy'][param]
        girls = results_df[results_df['gender'] == 'Girl'][param]
        t_stat, p_val = stats.ttest_ind(boys, girls)
        report.append(f"\n{param_labels[param.split('_')[0]]}:")
        report.append(f"Boys: {boys.mean():.3f} ± {boys.std():.3f}")
        report.append(f"Girls: {girls.mean():.3f} ± {girls.std():.3f}")
        report.append(f"T-test results: t={t_stat:.3f}, p={p_val:.3f}")
    
    # Save report
    os.makedirs(os.path.join(output_dir, 'reports/learning'), exist_ok=True)
    with open(os.path.join(output_dir, 'reports/learning/learning_dynamics_report.txt'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(report))
    
    return report 