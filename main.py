import numpy as np
from scipy import stats
from typing import Dict, Tuple
import pandas as pd
import os
import pickle
from visualization.config import setup_plot_style
from visualization.parameter.plots import (
    create_correlation_heatmap,
    create_parameter_distributions_plot,
    create_parameter_qq_plots,
    create_parameter_violin_plots,
    create_parameter_relationships_3d,
    create_parameter_age_evolution
)
from visualization.learning.plots import (
    create_learning_curves,
    create_learning_speed_plot,
    create_parameter_evolution_plot
)
from visualization.strategy.plots import (
    create_strategy_usage_plot,
    create_strategy_effectiveness_plot,
    create_strategy_heatmap,
    create_strategy_sequence
)
from visualization.performance.plots import (
    create_success_rate_plot,
    create_performance_by_age_plot,
    create_completion_analysis_plot,
    create_model_evaluation_plot
)
from visualization.cluster.plots import (
    create_cluster_plot,
    create_cluster_profiles_plot,
    create_cluster_characteristics_plot,
    create_cluster_size_completion_plot,
    create_success_vs_completion_plot,
    create_unlocks_by_cluster_plot
)

def partial_correlation(x, y, control):
    """Calculate partial correlation between x and y controlling for control variable."""
    # First regression: x ~ control
    slope_x, intercept_x, _, _, _ = stats.linregress(control, x)
    residuals_x = x - (slope_x * control + intercept_x)
    
    # Second regression: y ~ control
    slope_y, intercept_y, _, _, _ = stats.linregress(control, y)
    residuals_y = y - (slope_y * control + intercept_y)
    
    # Correlation between residuals
    return stats.pearsonr(residuals_x, residuals_y)[0]

output_dir = './output_smc'

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Setup plot style
setup_plot_style()

def load_and_merge_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load and merge long-format and short-format data directly from original files
    
    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: (merged_data, summary_data)
        merged_data: Long-format data containing detailed information for each attempt
        summary_data: Short-format data containing overall performance for each participant
    """
    # Load data from original files
    long_data = pd.read_csv('data/dollhouse_long.csv')
    short_data = pd.read_csv('data/dollhouse_short.csv')
    
    # Add gender and age information from short_data to long_data
    gender_age_dict = dict(zip(short_data['ID'], zip(short_data['Gender'], short_data['Age'])))
    
    # Update long_data with gender and age information
    for idx, row in long_data.iterrows():
        if row['ID'] in gender_age_dict:
            gender, age = gender_age_dict[row['ID']]
            long_data.at[idx, 'Gender'] = gender
            long_data.at[idx, 'Age'] = age
    
    # Merge data
    merged_data = pd.merge(
        long_data,
        short_data[['ID', 'NumUnlock', 'Solved', 'UnlockTime']],
        on='ID',
        how='left'
    )
    
    return merged_data, short_data

def calculate_success_probability(theta: float, motor_skill: float, omega: float, r: float) -> float:
    """
    Calculate success probability based on learning ability, motor skill, and strategy weights
    
    Args:
        theta: Learning ability parameter
        motor_skill: Motor skill parameter
        omega: Social influence parameter
        r: Exploration rate parameter
    
    Returns:
        float: Probability of success
    """
    eps = 1e-7  # Small epsilon for numerical stability
    
    # Clip parameters to valid ranges
    theta = np.clip(theta, eps, 1.0 - eps)
    motor_skill = np.clip(motor_skill, eps, 1.0 - eps)
    r = np.clip(r, eps, 1.0 - eps)
    omega = np.clip(omega, 1.0 + eps, 10.0)
    
    # Calculate base probability from learning ability and motor skill
    base_prob = theta * motor_skill
    base_prob = np.clip(base_prob, eps, 1.0 - eps)
    
    # Add social influence and exploration effects
    social_influence = omega * base_prob
    heuristic_influence = r * (1.0 - base_prob)
    
    # Calculate final probability
    final_prob = np.clip(base_prob + social_influence + heuristic_influence, eps, 1.0 - eps)
    
    return final_prob

def fit_parameters_sequential(subject_data: pd.DataFrame, n_particles: int = 1000) -> Dict:
    """
    Fit parameters using Sequential Monte Carlo (Particle Filter)
    
    Args:
        subject_data: DataFrame containing participant data
        n_particles: Number of particles
    
    Returns:
        Dict: Dictionary containing parameter posterior distributions
    """
    # Get completion information
    solved = subject_data['Solved'].iloc[0]
    num_unlock = subject_data['NumUnlock'].iloc[0]
    unlock_time = subject_data['UnlockTime'].iloc[0]
    
    # Calculate motor skill (error rate)
    errors = subject_data['Error'].sum()
    total = len(subject_data)
    motor = 1 - (errors / total)
    
    # Initialize particle filter
    particles = {
        'theta': np.random.beta(2, 2, n_particles),  # Learning ability
        'omega': np.random.pareto(3.0, n_particles) + 1.0,  # Social influence
        'r': np.random.uniform(0, 1, n_particles),  # Exploration rate
        'weights': np.ones(n_particles) / n_particles
    }
    
    # Store parameter trajectories
    trajectories = {
        'theta': np.zeros((len(subject_data), n_particles)),
        'omega': np.zeros((len(subject_data), n_particles)),
        'r': np.zeros((len(subject_data), n_particles)),
        'weights': np.zeros((len(subject_data), n_particles))
    }
    
    # Process each trial sequentially
    for t in range(len(subject_data)):
        trial = subject_data.iloc[t]
        
        # Calculate likelihood for each particle
        success_probs = np.array([
            calculate_success_probability(
                theta=particles['theta'][i],
                motor_skill=motor,
                omega=particles['omega'][i],
                r=particles['r'][i]
            ) for i in range(n_particles)
        ])
        
        # Update weights based on observation
        likelihood = success_probs if trial['Worked'] == 1 else (1 - success_probs)
        particles['weights'] *= likelihood
        particles['weights'] /= particles['weights'].sum()  # Normalize weights
        
        # Store current state
        for param in ['theta', 'omega', 'r', 'weights']:
            trajectories[param][t] = particles[param]
        
        # Resample if effective sample size is too low
        n_eff = 1 / (particles['weights'] ** 2).sum()
        if n_eff < n_particles / 2:
            indices = np.random.choice(n_particles, size=n_particles, p=particles['weights'])
            for param in ['theta', 'omega', 'r']:
                particles[param] = particles[param][indices]
            particles['weights'] = np.ones(n_particles) / n_particles
            
            # Add small noise to parameters (parameter evolution)
            particles['theta'] += np.random.normal(0, 0.01, n_particles)
            particles['theta'] = np.clip(particles['theta'], 0.001, 0.999)
            
            particles['omega'] += np.random.normal(0, 0.1, n_particles)
            particles['omega'] = np.maximum(particles['omega'], 1.0)
            
            particles['r'] += np.random.normal(0, 0.01, n_particles)
            particles['r'] = np.clip(particles['r'], 0.001, 0.999)
    
    # Calculate final parameter estimates and uncertainties
    final_estimates = {
        'theta': np.average(particles['theta'], weights=particles['weights']),
        'theta_std': np.sqrt(np.average((particles['theta'] - np.average(particles['theta'], weights=particles['weights']))**2, 
                                      weights=particles['weights'])),
        'omega': np.average(particles['omega'], weights=particles['weights']),
        'omega_std': np.sqrt(np.average((particles['omega'] - np.average(particles['omega'], weights=particles['weights']))**2,
                                      weights=particles['weights'])),
        'r': np.average(particles['r'], weights=particles['weights']),
        'r_std': np.sqrt(np.average((particles['r'] - np.average(particles['r'], weights=particles['weights']))**2,
                                  weights=particles['weights'])),
        'motor': motor,
        'completion_info': {
            'solved': solved,
            'num_unlock': num_unlock,
            'unlock_time': unlock_time,
        },
        'trajectories': trajectories,
        'particles': particles  # Add complete particle information
    }
    
    return final_estimates

def create_output_directories():
    """Create all necessary output directories."""
    subdirs = [
        'data',
        'reports/learning',
        'reports/strategy',
        'reports/cluster',
        'reports/performance',
        'reports/parameter',
        'visualization/learning',
        'visualization/strategy',
        'visualization/cluster',
        'visualization/performance',
        'visualization/parameter',
        'visualization/distribution_analysis',
        'visualization/correlation_analysis'
    ]
    
    for subdir in subdirs:
        os.makedirs(os.path.join(output_dir, subdir), exist_ok=True)

def analyze_individual_differences(generate_plots=True):
    """Analyze individual differences and strategy combinations"""
    # Create output directories first
    create_output_directories()
    
    # Load data
    merged_data, summary_data = load_and_merge_data()
    
    # Store results
    all_results = []
    all_posteriors = {}
    all_trajectories = {}
    
    # Process data for each participant
    for subject_id in merged_data['ID'].unique():
        subject_data = merged_data[merged_data['ID'] == subject_id]
        
        print(f"\nProcessing participant {subject_id}")
        print("-" * 50)
        
        # Fit parameters using Sequential MCMC
        results = fit_parameters_sequential(subject_data)
        
        # Store trajectories
        all_trajectories[subject_id] = results['trajectories']
        
        # Store complete posterior distributions
        all_posteriors[subject_id] = {
            'theta': results['particles']['theta'],
            'omega': results['particles']['omega'],
            'r': results['particles']['r'],
            'weights': results['particles']['weights']
        }
        
        # Get completion information
        completion_info = results['completion_info']
        
        # Store summary statistics
        result = {
            'ID': subject_id,
            'theta_mean': results['theta'],
            'theta_std': results['theta_std'],
            'omega_mean': results['omega'],
            'omega_std': results['omega_std'],
            'r_mean': results['r'],
            'r_std': results['r_std'],
            'motor': results['motor'],
            'age': subject_data['Age'].iloc[0],
            'gender': subject_data['Gender'].iloc[0],
            'total_trials': len(subject_data),
            'success_rate': subject_data['Worked'].mean(),
            'color_match_rate': subject_data['ColorMatch'].mean(),
            'num_match_rate': subject_data['NumMatch'].mean(),
            'shape_match_rate': subject_data['ShapeMatch'].mean(),
            'solved': completion_info['solved'],
            'num_unlock': completion_info['num_unlock'],
            'unlock_time': completion_info['unlock_time'],
        }
        all_results.append(result)
    
    # Convert to DataFrame
    results_df = pd.DataFrame(all_results)
    
    # Save results
    results_df.to_csv(os.path.join(output_dir, 'data/individual_results.csv'), index=False)
    with open(os.path.join(output_dir, 'data/posterior_distributions.pkl'), 'wb') as f:
        pickle.dump(all_posteriors, f)
    with open(os.path.join(output_dir, 'data/parameter_trajectories.pkl'), 'wb') as f:
        pickle.dump(all_trajectories, f)
    
    print(f"\nIndividual results have been saved to '{os.path.join(output_dir, 'data/individual_results.csv')}'")
    print(f"Posterior distributions have been saved to '{os.path.join(output_dir, 'data/posterior_distributions.pkl')}'")
    print(f"Parameter trajectories have been saved to '{os.path.join(output_dir, 'data/parameter_trajectories.pkl')}'")
    
    # Generate visualizations
    if generate_plots:
        print("\nGenerating visualizations...")
        
        # Parameter distributions and correlations
        print("Generating parameter visualizations...")
        create_parameter_distributions_plot(results_df, all_posteriors, output_dir)
        create_correlation_heatmap(results_df, output_dir)
        create_parameter_qq_plots(results_df, output_dir)
        create_parameter_violin_plots(results_df, output_dir)
        create_parameter_relationships_3d(results_df, output_dir)
        create_parameter_age_evolution(results_df, output_dir)
        
        # Learning dynamics
        print("Generating learning visualizations...")
        create_learning_curves(results_df, all_trajectories, output_dir)
        create_learning_speed_plot(results_df, all_trajectories, output_dir)
        create_parameter_evolution_plot(results_df, all_trajectories, output_dir)
        
        # Strategy analysis
        print("Generating strategy visualizations...")
        create_strategy_usage_plot(results_df, output_dir)
        create_strategy_effectiveness_plot(results_df, output_dir)
        create_strategy_heatmap(merged_data, output_dir)
        create_strategy_sequence(merged_data, output_dir)
        
        # Performance analysis
        print("Generating performance visualizations...")
        create_success_rate_plot(results_df, output_dir)
        create_performance_by_age_plot(results_df, output_dir)
        create_completion_analysis_plot(results_df, output_dir)
        create_model_evaluation_plot(results_df, output_dir)
        
        # Cluster analysis
        print("Generating cluster visualizations...")
        create_cluster_plot(results_df, output_dir)
        create_cluster_profiles_plot(results_df, output_dir)
        create_cluster_characteristics_plot(results_df, output_dir)
        create_cluster_size_completion_plot(results_df, output_dir)
        create_success_vs_completion_plot(results_df, output_dir)
        create_unlocks_by_cluster_plot(results_df, output_dir)
        
        print("\nAll visualizations have been generated successfully!")
    
    return results_df, all_posteriors, all_trajectories

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Run individual difference analysis')
    parser.add_argument('--plots-only', action='store_true', 
                      help='Only generate plots from saved files')
    
    args = parser.parse_args()
    
    if args.plots_only:
        print("Generating plots from saved files...")
        results_df = pd.read_csv(os.path.join(output_dir, 'data/individual_results.csv'))
        with open(os.path.join(output_dir, 'data/posterior_distributions.pkl'), 'rb') as f:
            all_posteriors = pickle.load(f)
        with open(os.path.join(output_dir, 'data/parameter_trajectories.pkl'), 'rb') as f:
            all_trajectories = pickle.load(f)
        analyze_individual_differences(generate_plots=True)
    else:
        print("Running full analysis...")
        analyze_individual_differences(generate_plots=True)