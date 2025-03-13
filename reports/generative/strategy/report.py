from itertools import combinations
import numpy as np
import pandas as pd
from scipy import stats
import os
from typing import Dict, List, Tuple
from sklearn.metrics import confusion_matrix

def analyze_strategy_usage(results_data: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    """
    Analyze strategy usage patterns
    """
    strategy_metrics = {}
    strategies = ['ColorMatch', 'NumMatch', 'ShapeMatch']
    
    # Calculate overall usage frequencies
    total_trials = len(results_data)
    for strategy in strategies:
        usage_count = results_data[strategy].sum()
        frequency = usage_count / total_trials
        
        # Calculate success rate for each strategy
        strategy_data = results_data[results_data[strategy] == 1]
        success_rate = strategy_data['Worked'].mean() if len(strategy_data) > 0 else 0
        error_rate = strategy_data['Error'].mean() if len(strategy_data) > 0 else 0
        
        strategy_metrics[strategy] = {
            'usage_count': usage_count,
            'frequency': frequency,
            'success_rate': success_rate,
            'error_rate': error_rate
        }
    
    return strategy_metrics

def analyze_strategy_transitions(results_data: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    """
    Analyze transitions between different strategies
    """
    strategies = ['ColorMatch', 'NumMatch', 'ShapeMatch']
    transition_matrix = {}
    
    # Create transition counts matrix
    for from_strategy in strategies:
        transition_matrix[from_strategy] = {}
        for to_strategy in strategies:
            # Count transitions from one strategy to another
            transitions = 0
            for i in range(len(results_data)-1):
                if results_data[from_strategy].iloc[i] == 1 and results_data[to_strategy].iloc[i+1] == 1:
                    transitions += 1
            
            # Calculate transition probability
            total_from = results_data[from_strategy].sum()
            prob = transitions / total_from if total_from > 0 else 0
            
            transition_matrix[from_strategy][to_strategy] = prob
    
    return transition_matrix

def analyze_strategy_learning(results_data: pd.DataFrame) -> Dict[str, Dict[str, List[float]]]:
    """
    Analyze how strategy usage and success change over time
    """
    strategies = ['ColorMatch', 'NumMatch', 'ShapeMatch']
    learning_metrics = {}
    
    # Split data into time bins
    n_bins = 10
    bin_size = len(results_data) // n_bins
    
    for strategy in strategies:
        usage_over_time = []
        success_over_time = []
        error_over_time = []
        
        for i in range(n_bins):
            start_idx = i * bin_size
            end_idx = (i + 1) * bin_size if i < n_bins - 1 else len(results_data)
            bin_data = results_data.iloc[start_idx:end_idx]
            
            # Calculate metrics for this time bin
            usage_rate = bin_data[strategy].mean()
            strategy_data = bin_data[bin_data[strategy] == 1]
            success_rate = strategy_data['Worked'].mean() if len(strategy_data) > 0 else 0
            error_rate = strategy_data['Error'].mean() if len(strategy_data) > 0 else 0
            
            usage_over_time.append(usage_rate)
            success_over_time.append(success_rate)
            error_over_time.append(error_rate)
        
        learning_metrics[strategy] = {
            'usage': usage_over_time,
            'success': success_over_time,
            'error': error_over_time
        }
    
    return learning_metrics

def analyze_strategy_combinations(results_data: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    """
    Analyze the effectiveness of different strategy combinations
    """
    strategies = ['ColorMatch', 'NumMatch', 'ShapeMatch']
    combination_metrics = {}
    
    # Analyze individual strategies and their combinations
    for r in range(1, len(strategies) + 1):
        for combo in combinations(strategies, r):
            combo_name = '+'.join(combo)
            
            # Create mask for trials using this combination
            mask = pd.Series(True, index=results_data.index)
            for strategy in combo:
                mask &= (results_data[strategy] == 1)
            
            combo_data = results_data[mask]
            
            if len(combo_data) > 0:
                combination_metrics[combo_name] = {
                    'frequency': len(combo_data) / len(results_data),
                    'success_rate': combo_data['Worked'].mean(),
                    'error_rate': combo_data['Error'].mean()
                }
    
    return combination_metrics

def generate_report(data: pd.DataFrame,
                output_dir: str = 'reports/strategy') -> Dict:
    """
    Generate comprehensive strategy analysis report
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Perform analyses
    strategy_metrics = analyze_strategy_usage(data)
    transition_metrics = analyze_strategy_transitions(data)
    learning_metrics = analyze_strategy_learning(data)
    combination_metrics = analyze_strategy_combinations(data)
    
    # Save results
    with open(os.path.join(output_dir, 'report.txt'), 'w') as f:
        # Write strategy usage analysis
        f.write("Strategy Usage Analysis\n")
        f.write("=====================\n\n")
        for strategy, metrics in strategy_metrics.items():
            f.write(f"{strategy}:\n")
            f.write(f"  Usage Count: {metrics['usage_count']}\n")
            f.write(f"  Frequency: {metrics['frequency']:.3f}\n")
            f.write(f"  Success Rate: {metrics['success_rate']:.3f}\n")
            f.write(f"  Error Rate: {metrics['error_rate']:.3f}\n\n")
        
        # Write transition analysis
        f.write("\nStrategy Transition Analysis\n")
        f.write("=========================\n\n")
        for from_strategy, transitions in transition_metrics.items():
            f.write(f"From {from_strategy}:\n")
            for to_strategy, prob in transitions.items():
                f.write(f"  To {to_strategy}: {prob:.3f}\n")
            f.write("\n")
        
        # Write learning analysis
        f.write("\nStrategy Learning Analysis\n")
        f.write("========================\n\n")
        for strategy, metrics in learning_metrics.items():
            f.write(f"{strategy}:\n")
            f.write("  Usage over time: ")
            f.write(", ".join([f"{x:.3f}" for x in metrics['usage']]))
            f.write("\n  Success over time: ")
            f.write(", ".join([f"{x:.3f}" for x in metrics['success']]))
            f.write("\n  Error over time: ")
            f.write(", ".join([f"{x:.3f}" for x in metrics['error']]))
            f.write("\n\n")
        
        # Write combination analysis
        f.write("\nStrategy Combination Analysis\n")
        f.write("===========================\n\n")
        for combo, metrics in combination_metrics.items():
            f.write(f"{combo}:\n")
            f.write(f"  Frequency: {metrics['frequency']:.3f}\n")
            f.write(f"  Success Rate: {metrics['success_rate']:.3f}\n")
            f.write(f"  Error Rate: {metrics['error_rate']:.3f}\n\n")
    
    # Save numerical results for later use
    results = {
        'usage': strategy_metrics,
        'transitions': transition_metrics,
        'learning': learning_metrics,
        'combinations': combination_metrics
    }
    
    return results 