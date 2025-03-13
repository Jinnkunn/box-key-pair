"""
Parameter analysis report generation functions.
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

def generate_parameter_analysis_report(results_df: pd.DataFrame, output_dir: str):
    """
    Generate parameter analysis report
    
    Args:
        results_df: DataFrame containing analysis results
        output_dir: Directory to save the report
    """
    report = []
    report.append("Parameter Analysis Report")
    report.append("=" * 50)
    
    params = ['theta_mean', 'omega_mean', 'r_mean']
    param_labels = {
        'theta_mean': 'Learning Ability (θ)',
        'omega_mean': 'Social Influence (ω)',
        'r_mean': 'Exploration Rate (r)'
    }
    
    # 1. Distribution Analysis
    report.append("\n1. Distribution Analysis")
    report.append("-" * 30)
    
    for param in params:
        name = param_labels[param]
        report.append(f"\n{name}:")
        
        # Descriptive statistics
        stats_desc = results_df[param].describe()
        report.append(f"Mean: {stats_desc['mean']:.3f}")
        report.append(f"Std: {stats_desc['std']:.3f}")
        report.append(f"Range: {stats_desc['min']:.3f} - {stats_desc['max']:.3f}")
        report.append(f"Quartiles: {stats_desc['25%']:.3f}, {stats_desc['50%']:.3f}, {stats_desc['75%']:.3f}")
        
        # Normality test
        stat, p_val, msg = safe_statistical_test('shapiro', results_df[param].values)
        report.append(format_statistical_result(stat, p_val, msg, "Shapiro-Wilk Normality Test"))
        report.append(f"Distribution: {'Normal' if p_val > 0.05 else 'Non-normal'}")
    
    # 2. Gender Differences
    report.append("\n2. Gender Differences")
    report.append("-" * 30)
    
    for param in params:
        name = param_labels[param]
        report.append(f"\n{name}:")
        
        # Calculate gender statistics
        gender_stats = results_df.groupby('gender')[param].agg(['mean', 'std', 'count'])
        for gender, stats in gender_stats.iterrows():
            report.append(f"{gender}: {stats['mean']:.3f} ± {stats['std']:.3f} (n={int(stats['count'])})")
        
        # T-test for gender differences
        boys = results_df[results_df['gender'] == 'Boy'][param]
        girls = results_df[results_df['gender'] == 'Girl'][param]
        stat, p_val, msg = safe_statistical_test('ttest_ind', boys.values, girls.values)
        report.append(format_statistical_result(stat, p_val, msg, "Gender Comparison T-test"))
    
    # 3. Age Correlations
    report.append("\n3. Age Correlations")
    report.append("-" * 30)
    
    for param in params:
        name = param_labels[param]
        report.append(f"\n{name}:")
        
        # Calculate correlation with age
        stat, p_val, msg = safe_statistical_test('pearsonr', results_df['age'].values, results_df[param].values)
        report.append(format_statistical_result(stat, p_val, msg, "Age Correlation"))
    
    # 4. Parameter Relationships
    report.append("\n4. Parameter Relationships")
    report.append("-" * 30)
    
    for i, param1 in enumerate(params):
        for param2 in params[i+1:]:
            name1 = param_labels[param1]
            name2 = param_labels[param2]
            report.append(f"\n{name1} vs {name2}:")
            
            # Calculate correlation
            stat, p_val, msg = safe_statistical_test('pearsonr', results_df[param1].values, results_df[param2].values)
            report.append(format_statistical_result(stat, p_val, msg, "Correlation"))
    
    # 5. Performance Correlations
    report.append("\n5. Performance Correlations")
    report.append("-" * 30)
    
    metrics = ['success_rate', 'num_unlock', 'unlock_time']
    metric_labels = {
        'success_rate': 'Success Rate',
        'num_unlock': 'Number of Unlocks',
        'unlock_time': 'Completion Time'
    }
    
    for param in params:
        name = param_labels[param]
        report.append(f"\n{name}:")
        
        for metric in metrics:
            metric_name = metric_labels[metric]
            stat, p_val, msg = safe_statistical_test('pearsonr', results_df[param].values, results_df[metric].values)
            report.append(format_statistical_result(stat, p_val, msg, f"{metric_name} Correlation"))
    
    # Save report
    os.makedirs(os.path.join(output_dir, 'reports/parameter'), exist_ok=True)
    with open(os.path.join(output_dir, 'reports/parameter/parameter_analysis_report.txt'), 'w') as f:
        f.write('\n'.join(report))
    
    return report

def generate_correlation_report(results_df, output_dir):
    """
    Generate a report on parameter correlations.
    
    Args:
        results_df: DataFrame containing analysis results
        output_dir: Directory to save the report
    """
    output_file = os.path.join(output_dir, 'reports/parameter/correlation_report.txt')
    
    with open(output_file, 'w') as f:
        # Create report header
        create_report_header(
            title="Parameter Correlation Analysis Report",
            analysis_type="Correlation Analysis",
            file=f
        )
        
        # Parameter Correlations Section
        create_section_header("Parameter Correlations", f)
        params = ['theta_mean', 'omega_mean', 'r_mean', 'motor']
        param_names = ['Learning Rate (θ)', 'Social Influence (ω)', 
                      'Exploration Rate (r)', 'Motor Skill']
        
        for i, (param1, name1) in enumerate(zip(params, param_names)):
            for param2, name2 in zip(params[i+1:], param_names[i+1:]):
                r_val, p_val = stats.pearsonr(results_df[param1], results_df[param2])
                f.write(format_statistical_result(
                    f"Correlation: {name1} vs {name2}",
                    r_val,
                    p_val,
                    {
                        "Correlation Strength": "Strong" if abs(r_val) > 0.5 else
                                              "Moderate" if abs(r_val) > 0.3 else
                                              "Weak"
                    }
                ))
                f.write("\n")
        
        # Performance Correlations Section
        create_section_header("Parameter-Performance Correlations", f)
        perf_metrics = ['success_rate', 'num_unlock', 'unlock_time']
        perf_names = ['Success Rate', 'Number of Unlocks', 'Completion Time']
        
        for param, param_name in zip(params, param_names):
            for metric, metric_name in zip(perf_metrics, perf_names):
                r_val, p_val = stats.pearsonr(results_df[param], results_df[metric])
                f.write(format_statistical_result(
                    f"{param_name} vs {metric_name}",
                    r_val,
                    p_val,
                    {
                        "Effect Direction": "Positive" if r_val > 0 else "Negative",
                        "Practical Significance": "High" if abs(r_val) > 0.5 else
                                                "Medium" if abs(r_val) > 0.3 else
                                                "Low"
                    }
                ))
                f.write("\n") 