"""
Detailed parameter analysis report for generative model.
"""

import pandas as pd
import numpy as np
from scipy import stats
import os

def generate_detailed_parameter_report(results_df: pd.DataFrame, output_dir: str) -> dict:
    """
    Generate detailed parameter analysis report for generative model.
    
    Args:
        results_df: DataFrame containing model results
        output_dir: Directory to save the report
    
    Returns:
        dict: Analysis results
    """
    # Create output directory
    report_dir = os.path.join(output_dir, 'reports/generative/detailed/parameter')
    os.makedirs(report_dir, exist_ok=True)
    
    # Print data quality information
    print("\nData Quality Check:")
    print(f"Total rows: {len(results_df)}")
    print("\nMissing values:")
    print(results_df[['theta', 'omega', 'r', 'motor_skill']].isnull().sum())
    print("\nParameter statistics:")
    print(results_df[['theta', 'omega', 'r', 'motor_skill']].describe())
    
    # Clean data - remove rows with NaN values
    results_df = results_df.dropna(subset=['theta', 'omega', 'r', 'motor_skill'])
    print(f"\nRows after cleaning: {len(results_df)}")
    
    # Initialize results dictionary
    results = {
        'parameter_distributions': {},
        'parameter_correlations': {},
        'age_effects': {},
        'gender_effects': {}
    }
    
    # Analyze parameter distributions
    params = ['theta', 'omega', 'r', 'motor_skill']
    for param in params:
        dist_stats = {
            'mean': results_df[param].mean(),
            'std': results_df[param].std(),
            'median': results_df[param].median(),
            'q25': results_df[param].quantile(0.25),
            'q75': results_df[param].quantile(0.75),
            'skew': stats.skew(results_df[param]),
            'kurtosis': stats.kurtosis(results_df[param]),
            'shapiro_stat': stats.shapiro(results_df[param])[0],
            'shapiro_p': stats.shapiro(results_df[param])[1]
        }
        results['parameter_distributions'][param] = dist_stats
    
    # Analyze parameter correlations
    param_corr = results_df[params].corr()
    for p1 in params:
        for p2 in params:
            if p1 != p2:
                # Check if we have valid data for correlation
                data1 = results_df[p1]
                data2 = results_df[p2]
                
                # Skip if either variable has zero standard deviation
                if data1.std() == 0 or data2.std() == 0:
                    corr_stats = {
                        'correlation': None,
                        'pearson_r': None,
                        'pearson_p': None,
                        'spearman_r': None,
                        'spearman_p': None
                    }
                else:
                    corr_stats = {
                        'correlation': param_corr.loc[p1, p2],
                        'pearson_r': stats.pearsonr(data1, data2)[0],
                        'pearson_p': stats.pearsonr(data1, data2)[1],
                        'spearman_r': stats.spearmanr(data1, data2)[0],
                        'spearman_p': stats.spearmanr(data1, data2)[1]
                    }
                results['parameter_correlations'][f'{p1}_vs_{p2}'] = corr_stats
    
    # Analyze age effects
    for param in params:
        age_corr = stats.pearsonr(results_df['age'], results_df[param])
        age_stats = {
            'correlation': age_corr[0],
            'p_value': age_corr[1],
            'slope': np.polyfit(results_df['age'], results_df[param], 1)[0],
            'intercept': np.polyfit(results_df['age'], results_df[param], 1)[1]
        }
        results['age_effects'][param] = age_stats
    
    # Analyze gender effects
    for param in params:
        male_data = results_df[results_df['gender'] == 'Boy'][param]
        female_data = results_df[results_df['gender'] == 'Girl'][param]
        ttest = stats.ttest_ind(male_data, female_data)
        gender_stats = {
            'male_mean': male_data.mean(),
            'female_mean': female_data.mean(),
            'male_std': male_data.std(),
            'female_std': female_data.std(),
            't_stat': ttest[0],
            'p_value': ttest[1]
        }
        results['gender_effects'][param] = gender_stats
    
    # Save results to file
    results_df = pd.DataFrame.from_dict(results, orient='index')
    results_df.to_csv(os.path.join(report_dir, 'parameter_analysis.csv'))
    
    return results 