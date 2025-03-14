"""
Cluster analysis report generation functions.
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

def generate_cluster_analysis_report(results_df: pd.DataFrame, output_dir: str):
    """
    Generate cluster analysis report
    
    Args:
        results_df: DataFrame containing analysis results
        output_dir: Directory to save the report
    """
    # Perform clustering first
    from visualization.individual.cluster.plots import perform_clustering
    results_df_clustered, _ = perform_clustering(results_df)
    
    report = []
    report.append("Cluster Analysis Report")
    report.append("=" * 50)
    
    # 1. Cluster Size Analysis
    report.append("\n1. Cluster Size Analysis")
    report.append("-" * 30)
    
    cluster_sizes = results_df_clustered['cluster'].value_counts().sort_index()
    report.append("\nCluster Sizes:")
    for cluster, size in cluster_sizes.items():
        report.append(f"Cluster {cluster}: {size} participants ({size/len(results_df_clustered):.1%})")
    
    # 2. Cluster Characteristics
    report.append("\n2. Cluster Characteristics")
    report.append("-" * 30)
    
    features = ['theta_mean', 'omega_mean', 'r_mean', 'motor', 'success_rate']
    feature_labels = {
        'theta_mean': 'Learning Ability (θ)',
        'omega_mean': 'Social Influence (ω)',
        'r_mean': 'Exploration Rate (r)',
        'motor': 'Motor Skill',
        'success_rate': 'Success Rate'
    }
    
    cluster_means = results_df_clustered.groupby('cluster', observed=True)[features].mean()
    cluster_stds = results_df_clustered.groupby('cluster', observed=True)[features].std()
    
    for cluster in sorted(results_df_clustered['cluster'].unique()):
        report.append(f"\nCluster {cluster} Characteristics:")
        for feature in features:
            mean = cluster_means.loc[cluster, feature]
            std = cluster_stds.loc[cluster, feature]
            report.append(f"{feature_labels[feature]}: {mean:.3f} ± {std:.3f}")
    
    # 3. Cluster Comparisons
    report.append("\n3. Cluster Comparisons")
    report.append("-" * 30)
    
    for feature in features:
        report.append(f"\n{feature_labels[feature]} Comparison:")
        # Get groups for ANOVA
        groups = [group[feature].values for name, group in results_df_clustered.groupby('cluster', observed=True)]
        
        # Perform ANOVA with safety checks
        stat, p, msg = safe_statistical_test('f_oneway', *groups, min_samples=3)
        report.append(format_statistical_result(stat, p, msg, "ANOVA"))
    
    # 4. Completion Analysis by Cluster
    report.append("\n4. Completion Analysis by Cluster")
    report.append("-" * 30)
    
    completion_rates = results_df_clustered.groupby('cluster', observed=True)['solved'].agg(['count', 'mean'])
    report.append("\nCompletion Rates by Cluster:")
    for cluster in sorted(results_df_clustered['cluster'].unique()):
        count = completion_rates.loc[cluster, 'count']
        rate = completion_rates.loc[cluster, 'mean']
        report.append(f"Cluster {cluster}: {rate:.1%} ({int(rate * count)}/{count})")
    
    # Save report
    os.makedirs(os.path.join(output_dir, 'reports/cluster'), exist_ok=True)
    with open(os.path.join(output_dir, 'reports/cluster/cluster_analysis_report.txt'), 'w') as f:
        f.write('\n'.join(report))
    
    return report 