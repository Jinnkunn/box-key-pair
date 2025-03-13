import numpy as np
from scipy import stats
import pandas as pd
import os
import pickle
from visualization.individual.parameter.plots import (
    create_correlation_heatmap,
    create_parameter_distributions_plot,
    create_parameter_qq_plots,
    create_parameter_violin_plots,
    create_parameter_relationships_3d,
    create_parameter_age_evolution
)
from visualization.individual.learning.plots import (
    create_learning_curves,
    create_learning_speed_plot,
    create_parameter_evolution_plot
)
from visualization.individual.strategy.plots import (
    create_strategy_usage_plot,
    create_strategy_effectiveness_plot,
    create_strategy_heatmap,
    create_strategy_sequence
)
from visualization.individual.performance.plots import (
    create_success_rate_plot,
    create_performance_by_age_plot,
    create_completion_analysis_plot,
    create_model_evaluation_plot
)
from visualization.individual.cluster.plots import (
    create_cluster_plot,
    create_cluster_profiles_plot,
    create_cluster_characteristics_plot,
    create_cluster_size_completion_plot,
    create_success_vs_completion_plot,
    create_unlocks_by_cluster_plot
)
from reports.individual.parameter.reports import generate_parameter_analysis_report
from reports.individual.learning.reports import generate_learning_dynamics_report
from reports.individual.strategy.reports import generate_strategy_analysis_report
from reports.individual.performance.reports import generate_performance_analysis_report
from reports.individual.cluster.reports import generate_cluster_analysis_report

def load_and_merge_data() -> tuple[pd.DataFrame, pd.DataFrame]:
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

def fit_parameters_sequential(subject_data: pd.DataFrame, n_particles: int = 1000) -> dict:
    """
    Fit parameters using Sequential Monte Carlo (Particle Filter)
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

def analyze_individual_differences(data: pd.DataFrame = None, output_dir: str = 'output/individual', generate_plots: bool = True):
    """Analyze individual differences and strategy combinations"""
    # Load data if not provided
    if data is None:
        print("\nNo data provided, loading from files...")
        data, summary_data = load_and_merge_data()
    else:
        print(f"\nUsing provided data: {len(data)} trials from {len(data['ID'].unique())} participants")
        # Extract summary data from the full dataset
        summary_data = pd.DataFrame([{
            'ID': id_,
            'NumUnlock': data[data['ID'] == id_]['NumUnlock'].iloc[0],
            'Solved': data[data['ID'] == id_]['Solved'].iloc[0],
            'UnlockTime': data[data['ID'] == id_]['UnlockTime'].iloc[0],
            'Gender': data[data['ID'] == id_]['Gender'].iloc[0],
            'Age': data[data['ID'] == id_]['Age'].iloc[0]
        } for id_ in data['ID'].unique()])
    
    # Store results
    all_results = []
    all_posteriors = {}
    all_trajectories = {}
    
    # Process data for each participant
    for subject_id in data['ID'].unique():
        subject_data = data[data['ID'] == subject_id]
        
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
    data_dir = os.path.join(output_dir, 'data')
    os.makedirs(data_dir, exist_ok=True)
    
    results_df.to_csv(os.path.join(data_dir, 'individual_results.csv'), index=False)
    with open(os.path.join(data_dir, 'posterior_distributions.pkl'), 'wb') as f:
        pickle.dump(all_posteriors, f)
    with open(os.path.join(data_dir, 'parameter_trajectories.pkl'), 'wb') as f:
        pickle.dump(all_trajectories, f)
    
    print(f"\nIndividual results have been saved to '{os.path.join(data_dir, 'individual_results.csv')}'")
    print(f"Posterior distributions have been saved to '{os.path.join(data_dir, 'posterior_distributions.pkl')}'")
    print(f"Parameter trajectories have been saved to '{os.path.join(data_dir, 'parameter_trajectories.pkl')}'")
    
    # Generate reports and visualizations
    if generate_plots:
        print("\nGenerating reports and visualizations...")
        
        # Generate reports
        print("Generating analysis reports...")
        generate_parameter_analysis_report(results_df, output_dir)
        generate_learning_dynamics_report(results_df, all_trajectories, output_dir)
        generate_strategy_analysis_report(results_df, output_dir)
        generate_performance_analysis_report(results_df, output_dir)
        generate_cluster_analysis_report(results_df, output_dir)
        
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
        create_strategy_heatmap(data, output_dir)
        create_strategy_sequence(data, output_dir)
        
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