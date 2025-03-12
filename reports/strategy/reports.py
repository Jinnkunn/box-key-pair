import numpy as np
import pandas as pd
from scipy import stats
import os

def generate_strategy_analysis_report(results_df: pd.DataFrame, output_dir: str):
    """
    Generate strategy analysis report
    
    Args:
        results_df: DataFrame containing analysis results
        output_dir: Directory to save the report
    """
    report = []
    report.append("Strategy Analysis Report")
    report.append("=" * 50)
    
    # 1. Strategy Usage Patterns
    report.append("\n1. Strategy Usage Patterns")
    report.append("-" * 30)
    
    strategies = ['color_match_rate', 'num_match_rate', 'shape_match_rate']
    strategy_labels = {
        'color_match_rate': 'Color Matching',
        'num_match_rate': 'Number Matching',
        'shape_match_rate': 'Shape Matching'
    }
    
    for strategy in strategies:
        stats_desc = results_df[strategy].describe()
        report.append(f"\n{strategy_labels[strategy]}:")
        report.append(f"Mean usage rate: {stats_desc['mean']:.3f}")
        report.append(f"Std: {stats_desc['std']:.3f}")
        report.append(f"Range: {stats_desc['min']:.3f} - {stats_desc['max']:.3f}")
        report.append(f"Users with non-zero usage: {(results_df[strategy] > 0).sum()}/{len(results_df)}")
    
    # 2. Strategy Effectiveness
    report.append("\n2. Strategy Effectiveness")
    report.append("-" * 30)
    
    for strategy in strategies:
        # Calculate success rate for trials using this strategy
        users_with_strategy = results_df[results_df[strategy] > 0]
        success_rate = users_with_strategy['success_rate'].mean()
        std_err = users_with_strategy['success_rate'].std() / np.sqrt(len(users_with_strategy))
        
        report.append(f"\n{strategy_labels[strategy]}:")
        report.append(f"Success rate: {success_rate:.3f} ± {std_err:.3f}")
        report.append(f"Sample size: {len(users_with_strategy)}")
        
        # Compare with non-users
        users_without_strategy = results_df[results_df[strategy] == 0]
        if len(users_without_strategy) > 0 and len(users_with_strategy) > 0:
            t_stat, p_val = stats.ttest_ind(
                users_with_strategy['success_rate'],
                users_without_strategy['success_rate']
            )
            report.append(f"Comparison with non-users:")
            report.append(f"Non-users success rate: {users_without_strategy['success_rate'].mean():.3f}")
            report.append(f"T-test: t = {t_stat:.3f}, p = {p_val:.3f}")
    
    # 3. Strategy Transitions
    report.append("\n3. Strategy Transitions")
    report.append("-" * 30)
    
    # Calculate dominant strategy for each participant
    dominant_strategies = results_df[strategies].idxmax().map({
        'color_match_rate': 'Color',
        'num_match_rate': 'Number',
        'shape_match_rate': 'Shape'
    })
    
    strategy_counts = dominant_strategies.value_counts()
    report.append("\nDominant Strategy Distribution:")
    for strategy, count in strategy_counts.items():
        report.append(f"{strategy}: {count} participants ({count/len(results_df):.1%})")
    
    # 4. Strategy Combinations
    report.append("\n4. Strategy Combinations")
    report.append("-" * 30)
    
    # Calculate strategy combinations
    strategy_combinations = []
    for _, row in results_df.iterrows():
        active_strategies = [strategy_labels[s] for s in strategies if row[s] > 0.1]
        strategy_combinations.append('+'.join(sorted(active_strategies)) if active_strategies else 'None')
    
    combination_counts = pd.Series(strategy_combinations).value_counts()
    report.append("\nStrategy Combination Distribution:")
    for combination, count in combination_counts.items():
        report.append(f"{combination}: {count} participants ({count/len(results_df):.1%})")
    
    # Save report
    os.makedirs(os.path.join(output_dir, 'reports/strategy'), exist_ok=True)
    with open(os.path.join(output_dir, 'reports/strategy/strategy_analysis_report.txt'), 'w') as f:
        f.write('\n'.join(report))
    
    return report 