import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
import numpy as np

def plot_parameter_distributions(df: pd.DataFrame, output_dir: str):
    """Plot distributions of generated parameters"""
    params = ['theta', 'omega', 'r', 'motor_skill']
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    
    for ax, param in zip(axes.flat, params):
        sns.histplot(data=df, x=param, ax=ax, bins=30)
        ax.set_title(f'Distribution of {param}')
        ax.set_xlabel(param)
        ax.set_ylabel('Count')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'visualization/generative/parameter_distributions.png'))
    plt.close()

def plot_age_effects(df: pd.DataFrame, output_dir: str):
    """Plot parameter variations across ages"""
    params = ['theta', 'omega', 'r', 'motor_skill']
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    
    for ax, param in zip(axes.flat, params):
        sns.boxplot(data=df, x='age', y=param, ax=ax)
        ax.set_title(f'{param} by Age')
        ax.set_xlabel('Age')
        ax.set_ylabel(param)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'visualization/generative/age_effects.png'))
    plt.close()

def plot_success_rates(df: pd.DataFrame, output_dir: str):
    """Plot success rates by age and trial"""
    # Success rate by age
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df, x='age', y='success')
    plt.title('Success Rate by Age')
    plt.xlabel('Age')
    plt.ylabel('Success Rate')
    plt.savefig(os.path.join(output_dir, 'visualization/generative/success_by_age.png'))
    plt.close()
    
    # Success rate over trials
    plt.figure(figsize=(10, 6))
    success_by_trial = df.groupby('trial')['success'].mean()
    plt.plot(success_by_trial.index, success_by_trial.values)
    plt.title('Average Success Rate over Trials')
    plt.xlabel('Trial Number')
    plt.ylabel('Success Rate')
    plt.savefig(os.path.join(output_dir, 'visualization/generative/success_over_trials.png'))
    plt.close()

def plot_strategy_usage(df: pd.DataFrame, output_dir: str):
    """Plot strategy usage patterns"""
    # Overall strategy distribution
    plt.figure(figsize=(10, 6))
    sns.countplot(data=df, x='strategy')
    plt.title('Overall Strategy Distribution')
    plt.xlabel('Strategy')
    plt.ylabel('Count')
    plt.savefig(os.path.join(output_dir, 'visualization/generative/strategy_distribution.png'))
    plt.close()
    
    # Strategy evolution over trials
    plt.figure(figsize=(12, 6))
    strategy_counts = df.groupby(['trial'])['strategy'].value_counts(normalize=True).unstack()
    strategy_counts.plot(kind='area', stacked=True)
    plt.title('Strategy Evolution over Trials')
    plt.xlabel('Trial Number')
    plt.ylabel('Strategy Proportion')
    plt.savefig(os.path.join(output_dir, 'visualization/generative/strategy_evolution.png'))
    plt.close()

def plot_parameter_correlations(df: pd.DataFrame, output_dir: str):
    """Plot correlations between parameters"""
    params = ['theta', 'omega', 'r', 'motor_skill']
    
    # Correlation matrix
    plt.figure(figsize=(10, 8))
    corr_matrix = df[params].corr()
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0)
    plt.title('Parameter Correlations')
    plt.savefig(os.path.join(output_dir, 'visualization/generative/parameter_correlations.png'))
    plt.close()
    
    # Pairwise scatter plots
    sns.pairplot(df[params])
    plt.savefig(os.path.join(output_dir, 'visualization/generative/parameter_pairplot.png'))
    plt.close()

def plot_learning_curves(df: pd.DataFrame, output_dir: str):
    """Plot learning curves for different age groups"""
    plt.figure(figsize=(12, 6))
    
    for age in df['age'].unique():
        age_data = df[df['age'] == age]
        success_rate = age_data.groupby('trial')['success'].mean()
        plt.plot(success_rate.index, success_rate.values, label=f'Age {age}')
    
    plt.title('Learning Curves by Age')
    plt.xlabel('Trial Number')
    plt.ylabel('Success Rate')
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'visualization/generative/learning_curves.png'))
    plt.close()

def create_all_generative_plots(df: pd.DataFrame, output_dir: str):
    """Create all visualizations for the generative model"""
    print("\nGenerating visualizations for generative model...")
    
    # Parameter distributions
    print("- Plotting parameter distributions...")
    plot_parameter_distributions(df, output_dir)
    
    # Age effects
    print("- Plotting age effects...")
    plot_age_effects(df, output_dir)
    
    # Success rates
    print("- Plotting success rates...")
    plot_success_rates(df, output_dir)
    
    # Strategy usage
    print("- Plotting strategy usage patterns...")
    plot_strategy_usage(df, output_dir)
    
    # Parameter correlations
    print("- Plotting parameter correlations...")
    plot_parameter_correlations(df, output_dir)
    
    # Learning curves
    print("- Plotting learning curves...")
    plot_learning_curves(df, output_dir)
    
    print("All generative model visualizations have been saved to visualization/generative/") 