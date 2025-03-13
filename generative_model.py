import numpy as np
import pandas as pd
import os
from scipy import stats
import arviz as az
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple
from dataclasses import dataclass
from scipy.stats import multivariate_normal
from sklearn.preprocessing import StandardScaler
import json

from reports.generative import (
    generate_posterior_report,
    generate_recovery_report,
    generate_selection_report,
    generate_strategy_report,
    generate_validation_report,
    generate_uncertainty_report,
    generate_parameter_report,
    generate_likelihood_report
)
from reports.generative.detailed import (
    generate_detailed_parameter_report
)
from visualization.generative import (
    generate_all_visualizations,
    create_posterior_plots,
    create_recovery_plots,
    create_selection_plots,
    create_validation_plots,
    create_uncertainty_plots,
    create_parameter_plots,
    create_likelihood_plots
)

@dataclass
class Particle:
    """Represents a particle containing parameters and weight"""
    theta: float
    omega: float
    r: float
    motor_skill: float
    weight: float = 1.0
    
    def to_dict(self) -> dict:
        return {
            'theta': self.theta,
            'omega': self.omega,
            'r': self.r,
            'motor_skill': self.motor_skill
        }

class SMC:
    """Sequential Monte Carlo sampler"""
    def __init__(self, n_particles: int = 1000):
        self.n_particles = n_particles
        self.particles: List[Particle] = []
        self.effective_sample_size: float = n_particles
        
    def initialize_particles(self) -> None:
        """Initialize particles using the same distributions as individual model"""
        self.particles = []
        for _ in range(self.n_particles):
            particle = Particle(
                theta=np.random.beta(2, 2),  # Learning ability
                omega=np.random.pareto(3.0) + 1.0,  # Social influence
                r=np.random.uniform(0, 1),  # Exploration rate
                motor_skill=np.random.beta(2, 2)  # Motor skill
            )
            self.particles.append(particle)
    
    def predict(self) -> None:
        """Prediction step: apply state transition to each particle"""
        for particle in self.particles:
            # Add small noise to parameters (parameter evolution)
            particle.theta = np.clip(particle.theta + np.random.normal(0, 0.01), 0.001, 0.999)
            particle.omega = max(particle.omega + np.random.normal(0, 0.1), 1.0)
            particle.r = np.clip(particle.r + np.random.normal(0, 0.01), 0.001, 0.999)
            particle.motor_skill = np.clip(particle.motor_skill + np.random.normal(0, 0.01), 0.001, 0.999)
    
    def update(self, observation: bool, model) -> None:
        """Update step: update particle weights based on observation"""
        weights = []
        for particle in self.particles:
            # Calculate likelihood
            success_prob = model.calculate_success_probability(particle.to_dict())
            likelihood = success_prob if observation else (1 - success_prob)
            particle.weight *= likelihood
            weights.append(particle.weight)
        
        # Normalize weights
        weights = np.array(weights)
        weights /= np.sum(weights)
        for particle, weight in zip(self.particles, weights):
            particle.weight = weight
        
        # Calculate effective sample size
        self.effective_sample_size = 1.0 / np.sum(weights ** 2)
    
    def resample(self) -> None:
        """Resampling step"""
        if self.effective_sample_size < self.n_particles / 2:
            weights = np.array([p.weight for p in self.particles])
            indices = np.random.choice(
                self.n_particles,
                size=self.n_particles,
                p=weights
            )
            
            new_particles = []
            for idx in indices:
                old_particle = self.particles[idx]
                new_particle = Particle(
                    theta=old_particle.theta,
                    omega=old_particle.omega,
                    r=old_particle.r,
                    motor_skill=old_particle.motor_skill,
                    weight=1.0
                )
                new_particles.append(new_particle)
            
            self.particles = new_particles
            self.effective_sample_size = self.n_particles
            
    def get_model_parameters(self) -> dict:
        """获取当前模型的参数分布"""
        weights = np.array([p.weight for p in self.particles])
        
        # 计算加权平均和方差
        model_params = {
            'theta': {
                'mean': np.average([p.theta for p in self.particles], weights=weights),
                'std': np.sqrt(np.average([(p.theta - np.average([p.theta for p in self.particles], weights=weights))**2 
                                         for p in self.particles], weights=weights))
            },
            'omega': {
                'mean': np.average([p.omega for p in self.particles], weights=weights),
                'std': np.sqrt(np.average([(p.omega - np.average([p.omega for p in self.particles], weights=weights))**2 
                                         for p in self.particles], weights=weights))
            },
            'r': {
                'mean': np.average([p.r for p in self.particles], weights=weights),
                'std': np.sqrt(np.average([(p.r - np.average([p.r for p in self.particles], weights=weights))**2 
                                         for p in self.particles], weights=weights))
            },
            'motor_skill': {
                'mean': np.average([p.motor_skill for p in self.particles], weights=weights),
                'std': np.sqrt(np.average([(p.motor_skill - np.average([p.motor_skill for p in self.particles], weights=weights))**2 
                                         for p in self.particles], weights=weights))
            }
        }
        
        return model_params
    
    def get_predictive_distribution(self, model, trial_data: dict) -> tuple[float, float]:
        """使用所有粒子进行预测，返回预测值及其不确定性"""
        weights = np.array([p.weight for p in self.particles])
        
        # 使用所有粒子进行预测
        particle_predictions = [
            model.calculate_success_probability(p.to_dict())
            for p in self.particles
        ]
        
        # 计算加权平均预测和标准差
        mean_prediction = np.average(particle_predictions, weights=weights)
        std_prediction = np.sqrt(np.average((particle_predictions - mean_prediction)**2, weights=weights))
        
        return mean_prediction, std_prediction

class GenerativeModel:
    def __init__(self):
        # SMC settings
        self.n_particles = 1000
        self.smc = SMC(n_particles=self.n_particles)
        self.final_model = None  # 存储最终的模型参数
    
    def calculate_success_probability(self, params: dict) -> float:
        """Calculate success probability based on parameters"""
        eps = 1e-7  # Small epsilon for numerical stability
        
        # Clip parameters to valid ranges
        theta = np.clip(params['theta'], eps, 1.0 - eps)
        motor_skill = np.clip(params['motor_skill'], eps, 1.0 - eps)
        r = np.clip(params['r'], eps, 1.0 - eps)
        omega = np.clip(params['omega'], 1.0 + eps, 10.0)
        
        # Calculate base probability from learning ability and motor skill
        base_prob = theta * motor_skill
        base_prob = np.clip(base_prob, eps, 1.0 - eps)
        
        # Add social influence and exploration effects
        social_influence = omega * base_prob
        heuristic_influence = r * (1.0 - base_prob)
        
        # Calculate final probability
        final_prob = np.clip(base_prob + social_influence + heuristic_influence, eps, 1.0 - eps)
        
        return final_prob
    
    def load_data(self) -> pd.DataFrame:
        """Load and merge data from original files"""
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
        
        return merged_data
    
    def run_generative_model(self, data: pd.DataFrame = None, output_dir: str = 'output/generative', generate_plots: bool = True) -> pd.DataFrame:
        """Run the generative model using real data"""
        if data is None:
            print("\nNo data provided, loading from files...")
            data = self.load_data()
        
        print("\nLoading data files...")
        print(f"Loaded {len(data)} trials from {len(data['ID'].unique())} participants")
        print("\nData columns:", list(data.columns))
        
        print("\nStep 1: Running Generative Model")
        print(f"Processing {len(data)} trials from {len(data['ID'].unique())} participants")
        print("-" * 50)
        
        # Create output directories
        data_dir = os.path.join(output_dir, 'data')
        figures_dir = os.path.join(output_dir, 'visualization')
        reports_dir = os.path.join(output_dir, 'reports')
        for dir_path in [data_dir, figures_dir, reports_dir]:
            os.makedirs(dir_path, exist_ok=True)
        
        # Initialize SMC
        self.smc.initialize_particles()
        
        # Store all trials data
        all_trials = []
        
        # Process all trials sequentially
        total_trials = len(data)
        
        print("\nTraining model on all trials...")
        for idx, trial in data.iterrows():
            # Prediction step
            self.smc.predict()
            
            # 使用整个粒子集进行预测
            mean_pred, std_pred = self.smc.get_predictive_distribution(self, trial)
            
            # Update step using real observation
            self.smc.update(trial['Worked'] == 1, self)
            
            # Resampling step
            self.smc.resample()
            
            # 获取当前模型参数
            current_params = self.smc.get_model_parameters()
            
            # Record data
            trial_data = {
                'ID': trial['ID'],
                'age': trial['Age'],
                'trial': idx + 1,
                'success': trial['Worked'],
                'color_match': trial['ColorMatch'],
                'num_match': trial['NumMatch'],
                'shape_match': trial['ShapeMatch'],
                'error': trial['Error'],
                'theta_mean': current_params['theta']['mean'],
                'theta_std': current_params['theta']['std'],
                'omega_mean': current_params['omega']['mean'],
                'omega_std': current_params['omega']['std'],
                'r_mean': current_params['r']['mean'],
                'r_std': current_params['r']['std'],
                'motor_skill_mean': current_params['motor_skill']['mean'],
                'motor_skill_std': current_params['motor_skill']['std'],
                'effective_sample_size': self.smc.effective_sample_size,
                'predicted_success_mean': mean_pred,
                'predicted_success_std': std_pred,
                'gender': trial['Gender']
            }
            all_trials.append(trial_data)
            
            # Update progress
            if (idx + 1) % 1000 == 0:
                print(f"Processed {idx + 1}/{total_trials} trials")
        
        # 保存最终模型参数
        self.final_model = self.smc.get_model_parameters()
        
        # Convert to DataFrame
        results_df = pd.DataFrame(all_trials)
        
        # Save results
        results_path = os.path.join(data_dir, 'model_results.csv')
        results_df.to_csv(results_path, index=False)
        
        # Save final model parameters
        model_params_path = os.path.join(data_dir, 'final_model_parameters.json')
        with open(model_params_path, 'w') as f:
            json.dump(self.final_model, f, indent=4)
        
        # Generate participant summaries
        participant_summaries = []
        for participant_id in results_df['ID'].unique():
            participant_data = results_df[results_df['ID'] == participant_id]
            summary = {
                'ID': participant_id,
                'age': participant_data['age'].iloc[0],
                'mean_theta': participant_data['theta_mean'].mean(),
                'std_theta': participant_data['theta_std'].mean(),
                'mean_omega': participant_data['omega_mean'].mean(),
                'std_omega': participant_data['omega_std'].mean(),
                'mean_r': participant_data['r_mean'].mean(),
                'std_r': participant_data['r_std'].mean(),
                'mean_motor_skill': participant_data['motor_skill_mean'].mean(),
                'std_motor_skill': participant_data['motor_skill_std'].mean(),
                'success_rate': participant_data['success'].mean(),
                'mean_prediction': participant_data['predicted_success_mean'].mean(),
                'std_prediction': participant_data['predicted_success_std'].mean()
            }
            participant_summaries.append(summary)
        
        participant_summaries_df = pd.DataFrame(participant_summaries)
        summaries_path = os.path.join(data_dir, 'participant_summaries.csv')
        participant_summaries_df.to_csv(summaries_path, index=False)
        
        print(f"\nModel results have been saved to '{results_path}'")
        print(f"Final model parameters have been saved to '{model_params_path}'")
        print(f"Participant summaries have been saved to '{summaries_path}'")
        
        # Generate reports and plots if requested
        if generate_plots:
            self._generate_reports_and_plots(results_df, data_dir, figures_dir, reports_dir)
        
        return results_df
    
    def predict(self, new_data: pd.DataFrame) -> pd.DataFrame:
        """使用训练好的模型进行预测"""
        if self.final_model is None:
            raise ValueError("Model has not been trained yet. Please run run_generative_model first.")
        
        predictions = []
        for _, trial in new_data.iterrows():
            mean_pred, std_pred = self.smc.get_predictive_distribution(self, trial)
            predictions.append({
                'ID': trial['ID'],
                'predicted_success_mean': mean_pred,
                'predicted_success_std': std_pred
            })
        
        return pd.DataFrame(predictions)
    
    def _generate_reports_and_plots(self, results_df: pd.DataFrame, data_dir: str, figures_dir: str, reports_dir: str):
        """Generate all reports and plots"""
        print("\nGenerating analysis reports and plots...")
        
        # Generate all new visualizations
        print("\nGenerating new visualizations...")
        generate_all_visualizations(results_df, reports_dir)
        print("New visualizations created")
        
        # Generate detailed analysis reports
        print("\nGenerating detailed analysis reports...")
        parameter_analysis = generate_detailed_parameter_report(results_df, reports_dir)
        print("Detailed analysis reports created")
        
        # Create parameter dataframes for legacy visualizations
        true_params = pd.DataFrame({
            'theta': participant_summaries_df['mean_theta'],
            'omega': participant_summaries_df['mean_omega'],
            'r': participant_summaries_df['mean_r'],
            'motor_skill': participant_summaries_df['mean_motor_skill']
        })
        
        # Extract estimated parameters for each participant
        estimated_params_list = []
        for _, row in participant_summaries_df.iterrows():
            participant_data = results_df[results_df['ID'] == row['ID']]
            estimated_params_list.append({
                'theta': participant_data['theta_mean'].values,
                'omega': participant_data['omega_mean'].values,
                'r': participant_data['r_mean'].values,
                'motor_skill': participant_data['motor_skill_mean'].values
            })
        
        # Flatten estimated parameters into a DataFrame
        estimated_params = pd.DataFrame({
            'theta': np.concatenate([p['theta'] for p in estimated_params_list]),
            'omega': np.concatenate([p['omega'] for p in estimated_params_list]),
            'r': np.concatenate([p['r'] for p in estimated_params_list]),
            'motor_skill': np.concatenate([p['motor_skill'] for p in estimated_params_list])
        })
        
        # Repeat true parameters to match the length of estimated parameters
        true_params_repeated = pd.DataFrame({
            'theta': np.repeat(true_params['theta'].values, [len(p['theta']) for p in estimated_params_list]),
            'omega': np.repeat(true_params['omega'].values, [len(p['omega']) for p in estimated_params_list]),
            'r': np.repeat(true_params['r'].values, [len(p['r']) for p in estimated_params_list]),
            'motor_skill': np.repeat(true_params['motor_skill'].values, [len(p['motor_skill']) for p in estimated_params_list])
        })
        
        # Parameter analysis
        parameter_results = generate_parameter_report(true_params_repeated, estimated_params)
        create_parameter_plots(true_params_repeated, estimated_params, parameter_results)
        
        # Posterior analysis
        posterior_results = generate_posterior_report(results_df, data)
        create_posterior_plots(results_data=results_df, true_data=data, analysis_results=posterior_results)
        
        # Parameter recovery analysis
        print(f"\nTrue parameters shape: {true_params.shape}")
        print(f"Estimated parameters shape: {estimated_params.shape}")
        
        # Get final estimated parameters for each participant
        final_estimated_params = pd.DataFrame({
            'theta': [p['theta'][-1] for p in estimated_params_list],
            'omega': [p['omega'][-1] for p in estimated_params_list],
            'r': [p['r'][-1] for p in estimated_params_list],
            'motor_skill': [p['motor_skill'][-1] for p in estimated_params_list]
        })
        print(f"Final estimated parameters shape: {final_estimated_params.shape}")
        
        recovery_results = generate_recovery_report(true_params, final_estimated_params)
        create_recovery_plots(true_params, final_estimated_params, recovery_results)
        print("Recovery plots created")
        
        # Calculate log likelihood
        results_df['log_likelihood'] = np.log(
            np.where(data['Worked'] == 1,
                    results_df['predicted_success_mean'],
                    1 - results_df['predicted_success_mean'])
        )
        
        # Model selection analysis
        model_variants = {
            'full': results_df,
            'no_social': results_df.copy(),  # Variant without social influence
            'no_exploration': results_df.copy()  # Variant without exploration
        }
        
        # Modify variants
        model_variants['no_social']['omega_mean'] = 1.0  # Set social influence to minimum
        model_variants['no_exploration']['r_mean'] = 0.0  # Set exploration rate to zero
        
        # Recalculate predictions and log likelihood for variants
        for variant_name, variant_data in model_variants.items():
            if variant_name != 'full':
                variant_data['predicted_success_mean'] = variant_data.apply(
                    lambda row: self.calculate_success_probability({
                        'theta': row['theta_mean'],
                        'omega': row['omega_mean'],
                        'r': row['r_mean'],
                        'motor_skill': row['motor_skill_mean']
                    }),
                    axis=1
                )
                variant_data['log_likelihood'] = np.log(
                    np.where(data['Worked'] == 1,
                            variant_data['predicted_success_mean'],
                            1 - variant_data['predicted_success_mean'])
                )
        
        selection_results = generate_selection_report(model_variants, data)
        create_selection_plots(model_variants, selection_results)
        print("Selection plots created")

        # Validation analysis
        validation_results = generate_validation_report(results_df, data)
        create_validation_plots(results_df, validation_results)
        print("Validation plots created")
        
        # Uncertainty analysis
        uncertainty_results = generate_uncertainty_report(results_df)
        create_uncertainty_plots(results_df, uncertainty_results)
        print("Uncertainty plots created")
        
        # Likelihood analysis
        # likelihood_results = generate_likelihood_report(results_df)
        # create_likelihood_plots(results_df, data, likelihood_results)
        # print("Likelihood plots created")

        # Plot diagnostics
        self._plot_smc_diagnostics(results_df, figures_dir)
        print("SMC diagnostics plots created")  
        
        self._plot_parameter_trajectories(results_df, figures_dir)
        print("Parameter trajectories plots created")
        
        print("\nAnalysis completed successfully!") 