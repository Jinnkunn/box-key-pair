import numpy as np
import pandas as pd
from scipy import stats
import os

def generate_performance_analysis_report(results_df: pd.DataFrame, output_dir: str):
    """
    Generate performance analysis report
    
    Args:
        results_df: DataFrame containing analysis results
        output_dir: Directory to save the report
    """
    report = []
    report.append("Performance Analysis Report")
    report.append("=" * 50)
    
    # 1. Overall Performance Statistics
    report.append("\n1. Overall Performance Statistics")
    report.append("-" * 30)
    
    success_stats = results_df['success_rate'].describe()
    report.append("\nSuccess Rate Statistics:")
    report.append(f"Mean: {success_stats['mean']:.3f}")
    report.append(f"Std: {success_stats['std']:.3f}")
    report.append(f"Range: {success_stats['min']:.3f} - {success_stats['max']:.3f}")
    report.append(f"Quartiles: {success_stats['25%']:.3f}, {success_stats['50%']:.3f}, {success_stats['75%']:.3f}")
    
    # Test against chance level
    t_stat, p_val = stats.ttest_1samp(results_df['success_rate'], 0.5)
    report.append("\nComparison with Chance Level (0.5):")
    report.append(f"T-test: t = {t_stat:.3f}, p = {p_val:.3f}")
    
    # 2. Age-Related Performance
    report.append("\n2. Age-Related Performance")
    report.append("-" * 30)
    
    # Create age groups
    results_df['AgeGroup'] = pd.qcut(results_df['age'], q=4, labels=['Q1', 'Q2', 'Q3', 'Q4'])
    age_performance = results_df.groupby('AgeGroup')['success_rate'].agg(['mean', 'std', 'count'])
    
    report.append("\nPerformance by Age Quartile:")
    for idx, row in age_performance.iterrows():
        report.append(f"{idx}: {row['mean']:.3f} ± {row['std']:.3f} (n={int(row['count'])})")
    
    # ANOVA test for age groups
    age_groups = [group['success_rate'].values for name, group in results_df.groupby('AgeGroup')]
    f_stat, p_val = stats.f_oneway(*age_groups)
    report.append(f"\nAge Group ANOVA: F = {f_stat:.3f}, p = {p_val:.3f}")
    
    # 3. Gender Differences
    report.append("\n3. Gender Differences")
    report.append("-" * 30)
    
    gender_performance = results_df.groupby('gender')['success_rate'].agg(['mean', 'std', 'count'])
    report.append("\nPerformance by Gender:")
    for idx, row in gender_performance.iterrows():
        report.append(f"{idx}: {row['mean']:.3f} ± {row['std']:.3f} (n={int(row['count'])})")
    
    # T-test for gender differences
    boys = results_df[results_df['gender'] == 'Boy']['success_rate']
    girls = results_df[results_df['gender'] == 'Girl']['success_rate']
    t_stat, p_val = stats.ttest_ind(boys, girls)
    report.append(f"\nGender Comparison T-test: t = {t_stat:.3f}, p = {p_val:.3f}")
    
    # 4. Completion Analysis
    report.append("\n4. Completion Analysis")
    report.append("-" * 30)
    
    completion_rate = results_df['solved'].mean()
    report.append(f"\nOverall Completion Rate: {completion_rate:.1%}")
    
    # Completion by gender
    gender_completion = results_df.groupby('gender')['solved'].agg(['mean', 'count'])
    report.append("\nCompletion Rate by Gender:")
    for idx, row in gender_completion.iterrows():
        report.append(f"{idx}: {row['mean']:.1%} (n={int(row['count'])})")
    
    # Completion time analysis
    completion_times = results_df[results_df['solved'] == 1]['unlock_time']
    time_stats = completion_times.describe()
    report.append("\nCompletion Time Statistics (successful participants only):")
    report.append(f"Mean: {time_stats['mean']:.2f} seconds")
    report.append(f"Std: {time_stats['std']:.2f} seconds")
    report.append(f"Range: {time_stats['min']:.2f} - {time_stats['max']:.2f} seconds")
    
    # Save report
    os.makedirs(os.path.join(output_dir, 'reports/performance'), exist_ok=True)
    with open(os.path.join(output_dir, 'reports/performance/performance_analysis_report.txt'), 'w') as f:
        f.write('\n'.join(report))
    
    return report

def generate_model_evaluation_report(results_df: pd.DataFrame, output_dir: str):
    """
    Generate model evaluation report
    
    Args:
        results_df: DataFrame containing analysis results
        output_dir: Directory to save the report
    """
    report = []
    report.append("Model Evaluation Report")
    report.append("=" * 50)
    
    # 1. Model Fit Assessment
    report.append("\n1. Model Fit Assessment")
    report.append("-" * 30)
    
    # Calculate R-squared for success rate prediction
    slope, intercept, r_value, p_value, std_err = stats.linregress(
        results_df['theta_mean'], results_df['success_rate']
    )
    r_squared = r_value ** 2
    
    report.append("\nSuccess Rate Prediction:")
    report.append(f"R-squared: {r_squared:.3f}")
    report.append(f"Slope: {slope:.3f} ± {std_err:.3f}")
    report.append(f"P-value: {p_value:.3f}")
    
    # 2. Parameter Estimation Results
    report.append("\n2. Parameter Estimation Results")
    report.append("-" * 30)
    
    params = ['theta_mean', 'omega_mean', 'r_mean']
    param_labels = {
        'theta_mean': 'Learning Ability (θ)',
        'omega_mean': 'Social Influence (ω)',
        'r_mean': 'Exploration Rate (r)'
    }
    
    for param in params:
        stats_desc = results_df[param].describe()
        report.append(f"\n{param_labels[param]}:")
        report.append(f"Mean: {stats_desc['mean']:.3f}")
        report.append(f"Std: {stats_desc['std']:.3f}")
        report.append(f"Range: {stats_desc['min']:.3f} - {stats_desc['max']:.3f}")
        
        # Test reliability
        if param + '_std' in results_df.columns:
            avg_uncertainty = results_df[param + '_std'].mean()
            report.append(f"Average estimation uncertainty: {avg_uncertainty:.3f}")
    
    # 3. Model Comparison
    report.append("\n3. Model Comparison")
    report.append("-" * 30)
    
    # Compare parameter correlations with performance
    for param in params:
        corr, p_val = stats.pearsonr(results_df[param], results_df['success_rate'])
        report.append(f"\n{param_labels[param]} correlation with success rate:")
        report.append(f"Correlation: {corr:.3f}")
        report.append(f"P-value: {p_val:.3f}")
    
    # Save report
    os.makedirs(os.path.join(output_dir, 'reports/performance'), exist_ok=True)
    with open(os.path.join(output_dir, 'reports/performance/model_evaluation_report.txt'), 'w') as f:
        f.write('\n'.join(report))
    
    return report 