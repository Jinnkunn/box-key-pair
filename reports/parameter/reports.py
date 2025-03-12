"""
Parameter analysis report generation module.
"""

import os
import numpy as np
from scipy import stats
from ..utils import (
    create_report_header,
    create_section_header,
    format_statistical_result,
    format_descriptive_stats
)

def generate_parameter_analysis_report(results_df, output_dir):
    """
    Generate a comprehensive report on parameter analysis.
    
    Args:
        results_df: DataFrame containing analysis results
        output_dir: Directory to save the report
    """
    output_file = os.path.join(output_dir, 'reports/parameter/parameter_analysis_report.txt')
    
    with open(output_file, 'w') as f:
        # Create report header
        create_report_header(
            title="Parameter Analysis Report",
            analysis_type="Individual Differences Analysis",
            file=f
        )
        
        # Basic Statistics Section
        create_section_header("Basic Parameter Statistics", f)
        params = ['theta_mean', 'omega_mean', 'r_mean', 'motor']
        param_names = ['Learning Rate (θ)', 'Social Influence (ω)', 
                      'Exploration Rate (r)', 'Motor Skill']
        
        for param, name in zip(params, param_names):
            f.write(format_descriptive_stats(results_df[param], name))
            f.write("\n")
        
        # Distribution Analysis Section
        create_section_header("Distribution Analysis", f)
        for param, name in zip(params, param_names):
            # Normality test
            w_stat, p_val = stats.shapiro(results_df[param])
            f.write(format_statistical_result(
                f"{name} Normality Test",
                w_stat,
                p_val,
                {"Interpretation": "Normal" if p_val > 0.05 else "Non-normal"}
            ))
            f.write("\n")
        
        # Gender Differences Section
        create_section_header("Gender Differences in Parameters", f)
        for param, name in zip(params, param_names):
            boys = results_df[results_df['gender'] == 'Boy'][param]
            girls = results_df[results_df['gender'] == 'Girl'][param]
            
            t_stat, p_val = stats.ttest_ind(boys, girls)
            f.write(format_statistical_result(
                f"{name} Gender Comparison",
                t_stat,
                p_val,
                {
                    "Boys Mean": boys.mean(),
                    "Girls Mean": girls.mean(),
                    "Effect Size": (boys.mean() - girls.mean()) / np.sqrt((boys.var() + girls.var()) / 2)
                }
            ))
            f.write("\n")
        
        # Age Effects Section
        create_section_header("Age Effects on Parameters", f)
        for param, name in zip(params, param_names):
            r_val, p_val = stats.pearsonr(results_df['age'], results_df[param])
            f.write(format_statistical_result(
                f"{name} Age Correlation",
                r_val,
                p_val,
                {"Interpretation": "Significant" if p_val < 0.05 else "Non-significant"}
            ))
            f.write("\n")

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