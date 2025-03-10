import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from typing import List, Tuple, Dict
import pandas as pd
import seaborn as sns
import pymc as pm
import arviz as az

def calculate_success_probability(theta: float, motor_skill: float, omega: float, r: float) -> float:
    """Calculate success probability based on learning ability, motor skill, and strategy weights"""
    # Add small epsilon to avoid perfect motor skill
    # motor_skill = pm.math.clip(motor_skill, 0.01, 0.99)
    
    base_prob = theta * motor_skill
    social_influence = omega * base_prob
    heuristic_influence = r * (1 - base_prob)
    return pm.math.clip(base_prob + social_influence + heuristic_influence, 0, 1)

def fit_parameters_bayesian(subject_data: pd.DataFrame, n_samples: int = 2000) -> Dict:
    """Fit parameters using Bayesian MCMC sampling"""
    # Calculate motor skill based on Error rate
    errors = subject_data['Error'].sum()
    total = len(subject_data)
    motor = 1 - (errors / total)
    
    # # Handle perfect performance case
    # if motor >= 0.99:
    #     motor = 0.99
    
    with pm.Model() as model:
        # More conservative priors
        theta = pm.Beta('theta', alpha=1, beta=1, initval=0.5)  # Uniform prior
        omega = pm.HalfNormal('omega', sigma=0.5, initval=0.5)  # More conservative prior
        r = pm.Uniform('r', lower=0, upper=1, initval=0.5)
        
        # Success probability
        p = pm.Deterministic('p', calculate_success_probability(theta, motor, omega, r))
        
        # Likelihood
        y = pm.Bernoulli('y', p=p, observed=subject_data['Worked'])
        
        # Debug the model
        print("\nModel Debug Information:")
        print("-" * 50)
        print(f"Motor skill: {motor}")
        print(f"Data summary:")
        print(f"  Total trials: {total}")
        print(f"  Success rate: {subject_data['Worked'].mean():.4f}")
        print(f"  Error rate: {errors/total:.4f}")
        
        # MCMC sampling with better initialization
        trace = pm.sample(
            n_samples,
            return_inferencedata=True,
            cores=4,
            init='jitter+adapt_diag',
            random_seed=42,
            progressbar=True,
            target_accept=0.95,
            tune=2000,
            chains=4
        )
    
    # Extract posterior samples
    posterior = trace.posterior
    
    return {
        'theta': posterior['theta'].values,
        'omega': posterior['omega'].values,
        'r': posterior['r'].values,
        'motor': motor
    }

def analyze_gender_differences():
    """Analyze learning differences between boys and girls using Bayesian methods"""
    # Load data
    df = pd.read_csv('data/dollhouse.csv')
    
    # Group data by gender
    boys_data = []
    girls_data = []
    
    # Process each subject's data
    for subject_id in df['ID'].unique():
        subject_data = df[df['ID'] == subject_id]
        gender = subject_data['Gender'].iloc[0]
        
        print(f"\nProcessing subject {subject_id} ({gender})")
        print("-" * 50)
        
        # Fit parameters using Bayesian MCMC
        results = fit_parameters_bayesian(subject_data)
        
        # Store results
        result = {
            'theta_mean': np.mean(results['theta']),
            'theta_std': np.std(results['theta']),
            'omega_mean': np.mean(results['omega']),
            'omega_std': np.std(results['omega']),
            'r_mean': np.mean(results['r']),
            'r_std': np.std(results['r']),
            'motor': results['motor'],
            'age': subject_data['Age'].iloc[0],
            'total_trials': len(subject_data),
            'success_rate': subject_data['Worked'].mean(),
            'color_match_rate': subject_data['ColorMatch'].mean(),
            'num_match_rate': subject_data['NumMatch'].mean(),
            'shape_match_rate': subject_data['ShapeMatch'].mean()
        }
        
        if gender == 'Boy':
            boys_data.append(result)
        else:
            girls_data.append(result)
    
    # Convert to DataFrames
    boys_df = pd.DataFrame(boys_data)
    girls_df = pd.DataFrame(girls_data)
    
    # Visualize results
    plt.figure(figsize=(15, 10))
    
    # Learning ability comparison
    plt.subplot(2, 2, 1)
    plt.boxplot([boys_df['theta_mean'], girls_df['theta_mean']], labels=['Boys', 'Girls'])
    plt.title('Learning Ability Comparison (Bayesian)')
    plt.ylabel('Learning Ability (θ)')
    
    # Social prior weight comparison
    plt.subplot(2, 2, 2)
    plt.boxplot([boys_df['omega_mean'], girls_df['omega_mean']], labels=['Boys', 'Girls'])
    plt.title('Social Prior Weight Comparison (Bayesian)')
    plt.ylabel('Social Prior Weight (ω)')
    
    # Heuristic exploration weight comparison
    plt.subplot(2, 2, 3)
    plt.boxplot([boys_df['r_mean'], girls_df['r_mean']], labels=['Boys', 'Girls'])
    plt.title('Heuristic Exploration Weight Comparison (Bayesian)')
    plt.ylabel('Heuristic Exploration Weight (r)')
    
    # Motor skill comparison
    plt.subplot(2, 2, 4)
    plt.boxplot([boys_df['motor'], girls_df['motor']], labels=['Boys', 'Girls'])
    plt.title('Motor Skill Comparison')
    plt.ylabel('Motor Skill (M)')
    
    plt.tight_layout()
    plt.savefig('bayesian_comparison_results.png')
    plt.close()
    
    # Calculate statistical significance using Bayesian approach
    print("\nBayesian Analysis Results:")
    print("-" * 50)
    
    # Calculate probability of difference between groups
    for var, name in [('theta_mean', 'Learning Ability (θ)'),
                      ('omega_mean', 'Social Prior Weight (ω)'),
                      ('r_mean', 'Heuristic Exploration Weight (r)'),
                      ('motor', 'Motor Skill (M)')]:
        diff = boys_df[var] - girls_df[var]
        prob_diff = np.mean(diff > 0)
        print(f"\n{name}:")
        print(f"  Probability of boys > girls: {prob_diff:.4f}")
        print(f"  Boys: {boys_df[var].mean():.4f} ± {boys_df[var].std():.4f}")
        print(f"  Girls: {girls_df[var].mean():.4f} ± {girls_df[var].std():.4f}")

def analyze_individual_differences():
    """Analyze individual differences and strategy combinations"""
    # Load data
    df = pd.read_csv('data/dollhouse.csv')
    
    # Store results for all subjects
    all_results = []
    
    # Process each subject's data
    for subject_id in df['ID'].unique():
        subject_data = df[df['ID'] == subject_id]
        
        print(f"\nProcessing subject {subject_id}")
        print("-" * 50)
        
        # Fit parameters using Bayesian MCMC
        results = fit_parameters_bayesian(subject_data)
        
        # Store results
        result = {
            'ID': subject_id,
            'theta_mean': np.mean(results['theta']),
            'theta_std': np.std(results['theta']),
            'omega_mean': np.mean(results['omega']),
            'omega_std': np.std(results['omega']),
            'r_mean': np.mean(results['r']),
            'r_std': np.std(results['r']),
            'motor': results['motor'],
            'age': subject_data['Age'].iloc[0],
            'total_trials': len(subject_data),
            'success_rate': subject_data['Worked'].mean(),
            'color_match_rate': subject_data['ColorMatch'].mean(),
            'num_match_rate': subject_data['NumMatch'].mean(),
            'shape_match_rate': subject_data['ShapeMatch'].mean()
        }
        all_results.append(result)
    
    # Convert to DataFrame
    results_df = pd.DataFrame(all_results)
    
    # 1. Correlation Analysis
    correlation_vars = ['theta_mean', 'omega_mean', 'r_mean', 'motor', 'success_rate', 
                       'color_match_rate', 'num_match_rate', 'shape_match_rate']
    correlation_matrix = results_df[correlation_vars].corr()
    
    # Plot correlation heatmap
    plt.figure(figsize=(12, 10))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
    plt.title('Correlation Matrix of Key Variables')
    plt.tight_layout()
    plt.savefig('correlation_heatmap.png')
    plt.close()
    
    # 2. Strategy Combination Analysis
    # Create strategy profiles based on parameter combinations
    results_df['strategy_profile'] = results_df.apply(
        lambda x: 'High Learning' if x['theta_mean'] > 0.5 else 'Low Learning', axis=1
    )
    results_df['strategy_profile'] = results_df.apply(
        lambda x: x['strategy_profile'] + '_High Social' if x['omega_mean'] > 0.5 
        else x['strategy_profile'] + '_Low Social', axis=1
    )
    results_df['strategy_profile'] = results_df.apply(
        lambda x: x['strategy_profile'] + '_High Explore' if x['r_mean'] > 0.5 
        else x['strategy_profile'] + '_Low Explore', axis=1
    )
    
    # Analyze success rates by strategy profile
    strategy_success = results_df.groupby('strategy_profile')['success_rate'].agg(['mean', 'std', 'count'])
    
    # Plot strategy profile comparison
    plt.figure(figsize=(12, 6))
    strategy_success['mean'].plot(kind='bar')
    plt.errorbar(range(len(strategy_success)), strategy_success['mean'], 
                yerr=strategy_success['std'], fmt='none', color='black', capsize=5)
    plt.title('Success Rate by Strategy Profile')
    plt.xlabel('Strategy Profile')
    plt.ylabel('Success Rate')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('strategy_profile_comparison.png')
    plt.close()
    
    # Print analysis results
    print("\nCorrelation Analysis Results:")
    print("-" * 50)
    print("\nKey Correlations:")
    for var in correlation_vars:
        if var != 'success_rate':
            corr = correlation_matrix.loc[var, 'success_rate']
            print(f"{var} vs success_rate: {corr:.4f}")
    
    print("\nStrategy Profile Analysis:")
    print("-" * 50)
    print(strategy_success)

if __name__ == "__main__":
    analyze_individual_differences()
