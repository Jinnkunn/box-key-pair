import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from typing import Dict, Tuple
import pandas as pd
import seaborn as sns
import pymc as pm
from sklearn.mixture import GaussianMixture
import os
import networkx as nx
from sklearn.metrics import roc_curve, auc
import pickle

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

# Unified color scheme
COLOR_PALETTE = {
    'primary': '#1976D2',    # Primary blue
    'secondary': '#43A047',  # Secondary green
    'accent': '#E53935',     # Accent red
    'neutral': '#757575',    # Neutral gray
    'highlight': '#FDD835',  # Highlight yellow
    'boys': '#2196F3',      # Boys color
    'girls': '#E91E63'      # Girls color
}

# Parameter color mapping
PARAM_COLORS = {
    'theta': COLOR_PALETTE['primary'],     # Learning ability
    'omega': COLOR_PALETTE['secondary'],   # Social influence
    'r': COLOR_PALETTE['accent'],         # Exploration rate
    'motor': COLOR_PALETTE['highlight']   # Motor skill
}

# Strategy color mapping
STRATEGY_COLORS = {
    'Color': '#1f77b4',
    'Number': '#2ca02c',
    'Shape': '#d62728',
    'None': COLOR_PALETTE['neutral']
}

# Unified chart style settings
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
    'axes.grid': True,
    'grid.alpha': 0.3,
    'axes.labelsize': 12,
    'axes.titlesize': 14,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'legend.frameon': True,
    'legend.framealpha': 0.8,
    'legend.edgecolor': 'lightgray',
    'savefig.dpi': 300,
    'savefig.bbox': 'tight'
})

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
    errors = subject_data['Error'].sum()
    total = len(subject_data)
    motor = 1 - (errors / total)
    
    solved = subject_data['Solved'].iloc[0]
    num_unlock = subject_data['NumUnlock'].iloc[0]
    unlock_time = subject_data['UnlockTime'].iloc[0]
    
    success_rate = subject_data['Worked'].mean()
    
    n_hypotheses = 3  # Color, Number, Shape
    
    # Initialize particles
    particles = {
        'theta': np.random.beta(2, 2, n_particles),
        'omega': np.random.pareto(3.0, n_particles) + 1.0,
        'r': np.random.uniform(0, 1, n_particles)
    }
    
    # Initialize weights
    weights = np.ones(n_particles) / n_particles
    
    # Initialize parameter trajectories
    parameter_trajectories = {k: [] for k in particles.keys()}
    
    # Sequential processing
    for _, row in subject_data.iterrows():
        # Update particles
        for i in range(n_particles):
            theta = particles['theta'][i]
            omega = particles['omega'][i]
            r = particles['r'][i]
            
            # Calculate success probability
            p = calculate_success_probability(theta, motor, omega, r)
            
            # Update weights
            if row['Worked']:
                weights[i] *= p
            else:
                weights[i] *= (1 - p)
        
        # Normalize weights
        weights /= np.sum(weights)
        
        # Resample particles
        indices = np.random.choice(n_particles, size=n_particles, p=weights)
        particles = {k: v[indices] for k, v in particles.items()}
        
        # Store parameter trajectories
        for k, v in particles.items():
            parameter_trajectories[k].append(np.mean(v))
    
    # Calculate final parameter estimates
    final_parameters = {k: np.mean(v) for k, v in particles.items()}
    
    # Calculate parameter uncertainties
    parameter_uncertainties = {k: np.std(v) for k, v in particles.items()}
    
    return {
        'theta': final_parameters['theta'],
        'omega': final_parameters['omega'],
        'r': final_parameters['r'],
        'motor': motor,
        'completion_info': {
            'solved': solved,
            'num_unlock': num_unlock,
            'unlock_time': unlock_time,
        },
        'parameter_trajectories': parameter_trajectories,
        'parameter_uncertainties': parameter_uncertainties
    }

def analyze_gender_differences():
    """Analyze learning differences between boys and girls using Bayesian methods"""
    # Load data directly from original files
    merged_data, _ = load_and_merge_data()
    
    # Group data by gender
    boys_data = []
    girls_data = []
    
    # Process each subject's data
    for subject_id in merged_data['ID'].unique():
        subject_data = merged_data[merged_data['ID'] == subject_id]
        gender = subject_data['Gender'].iloc[0]
        
        print(f"\nProcessing subject {subject_id} ({gender})")
        print("-" * 50)
        
        # Fit parameters using Sequential MCMC
        results = fit_parameters_sequential(subject_data)
        
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
            'shape_match_rate': subject_data['ShapeMatch'].mean(),
            'solved': results['completion_info']['solved'],
            'num_unlock': results['completion_info']['num_unlock'],
            'unlock_time': results['completion_info']['unlock_time'],
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
    plt.savefig(os.path.join(output_dir, 'bayesian_comparison_results.png'))
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

def create_strategy_heatmap(trial_data):
    """
    Create a heatmap visualization of strategy usage patterns across different phases of the task.
    Considers completion status in the visualization.
    """
    plt.figure(figsize=(15, 10))
    
    df = trial_data.copy()
    
    if 'Solved' not in df.columns:
        results_df = pd.read_csv(os.path.join(output_dir, 'individual_results.csv'))
        completion_dict = dict(zip(results_df['ID'], results_df['solved']))
        df['Solved'] = df['ID'].map(completion_dict)
    
    def get_primary_strategy(row):
        if row['ColorMatch']:
            return 'Color'
        elif row['NumMatch']:
            return 'Number'
        elif row['ShapeMatch']:
            return 'Shape'
        return 'None'
    
    df['Strategy'] = df.apply(get_primary_strategy, axis=1)
    
    # Define phases based on trial order and completion status
    df['Phase'] = df.groupby(['ID', 'Solved'])['Order'].transform(
        lambda x: pd.qcut(x, q=4, labels=['Early', 'Early-Mid', 'Mid-Late', 'Late'])
    )
    
    strategy_freq = df.groupby(['Phase', 'Solved', 'Strategy']).size().unstack(fill_value=0)
    strategy_freq = strategy_freq.groupby('Solved').transform(lambda x: x / x.sum())
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    
    completed_freq = strategy_freq.xs(1, level='Solved') if 1 in strategy_freq.index.get_level_values('Solved') else None
    if completed_freq is not None:
        sns.heatmap(completed_freq, ax=ax1, cmap='viridis', annot=True, fmt='.2f')
        ax1.set_title('Strategy Usage - Completed Tasks')
    else:
        ax1.text(0.5, 0.5, 'No completed tasks data available', ha='center', va='center')
    
    incomplete_freq = strategy_freq.xs(0, level='Solved') if 0 in strategy_freq.index.get_level_values('Solved') else None
    if incomplete_freq is not None:
        sns.heatmap(incomplete_freq, ax=ax2, cmap='viridis', annot=True, fmt='.2f')
        ax2.set_title('Strategy Usage - Incomplete Tasks')
    else:
        ax2.text(0.5, 0.5, 'No incomplete tasks data available', ha='center', va='center')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'strategy_analysis/strategy_heatmap.png'), dpi=300, bbox_inches='tight')    
    plt.close()

def create_strategy_sequence(df, n_subjects=6):
    """
    Create a visualization of strategy transitions for selected subjects,
    balanced between completed and incomplete cases
    
    Args:
        df: DataFrame containing trial data
        n_subjects: Number of subjects to display (will be split between completed/incomplete)
    """
    subject_trials = df.groupby('ID').agg({
        'Order': 'size',
        'Solved': 'first'
    }).reset_index()
    
    completed = subject_trials[subject_trials['Solved'] == 1].nlargest(n_subjects//2, 'Order')
    incomplete = subject_trials[subject_trials['Solved'] == 0].nlargest(n_subjects//2, 'Order')
    selected_subjects = pd.concat([completed, incomplete])['ID']
    
    fig, axes = plt.subplots(n_subjects, 1, figsize=(15, 3*n_subjects))
    fig.suptitle('Strategy Transition Sequences (Completed vs Incomplete)', fontsize=16, y=1.02)
    
    colors = {
        'Color': '#1f77b4',
        'Number': '#2ca02c',
        'Shape': '#d62728',
        'None': '#7f7f7f'
    }
    
    for idx, subject in enumerate(selected_subjects):
        subject_data = df[df['ID'] == subject].copy()
        
        def get_primary_strategy(row):
            if row['ColorMatch']:
                return 'Color'
            elif row['NumMatch']:
                return 'Number'
            elif row['ShapeMatch']:
                return 'Shape'
            return 'None'
        
        subject_data['PrimaryStrategy'] = subject_data.apply(get_primary_strategy, axis=1)
        
        ax = axes[idx]
        
        for strategy in colors:
            mask = subject_data['PrimaryStrategy'] == strategy
            ax.scatter(subject_data[mask]['Order'], 
                      [1] * mask.sum(),
                      c=colors[strategy],
                      label=strategy,
                      alpha=0.7,
                      s=100)
        
        # Add success/failure markers
        ax.scatter(subject_data[subject_data['Worked'] == 1]['Order'],
                  [1.2] * subject_data['Worked'].sum(),
                  marker='^',
                  c='green',
                  alpha=0.5,
                  label='Success')
        
        ax.scatter(subject_data[subject_data['Error'] == 1]['Order'],
                  [0.8] * subject_data['Error'].sum(),
                  marker='v',
                  c='red',
                  alpha=0.5,
                  label='Error')
        
        completion_status = "Completed" if subject_data['Solved'].iloc[0] else "Incomplete"
        num_unlocked = subject_data['NumUnlock'].iloc[0]
        ax.set_title(f'Subject {subject} ({completion_status}, {num_unlocked}/5 locks)')
        ax.set_xlabel('Trial Number')
        ax.set_yticks([])
        ax.set_ylim([0.5, 1.5])
        
        if idx == 0:
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'strategy_analysis/strategy_sequence.png'), dpi=300, bbox_inches='tight')   
    plt.close()

def create_parameter_trajectory_plot(subject_id, parameter_trajectories, parameter_uncertainties, subject_data):
    """
    Create a visualization of parameter trajectories for a given subject
    
    Args:
        subject_id: ID of the subject
        parameter_trajectories: Dictionary containing parameter trajectories
        parameter_uncertainties: Dictionary containing parameter uncertainties
        subject_data: DataFrame containing subject data
    """
    plt.figure(figsize=(15, 10))
    
    # Plot parameter trajectories
    for idx, (param, trajectory) in enumerate(parameter_trajectories.items()):
        plt.subplot(2, 2, idx + 1)
        
        # Plot parameter values
        plt.scatter(range(len(trajectory)), trajectory, 
                   alpha=0.6, 
                   color=PARAM_COLORS[param],
                   s=100)
        
        # Add uncertainty intervals
        uncertainty = parameter_uncertainties[param]
        plt.fill_between(range(len(trajectory)), 
                         [t - uncertainty for t in trajectory],
                         [t + uncertainty for t in trajectory],
                         color=PARAM_COLORS[param],
                         alpha=0.2)
        
        # Add success/failure markers
        plt.scatter(subject_data[subject_data['Worked'] == 1]['Order'],
                    [0.9] * subject_data['Worked'].sum(),
                    marker='^',
                    c='green',
                    alpha=0.5,
                    label='Success')
        
        plt.scatter(subject_data[subject_data['Error'] == 1]['Order'],
                    [0.1] * subject_data['Error'].sum(),
                    marker='v',
                    c='red',
                    alpha=0.5,
                    label='Error')
        
        plt.title(f'{param.capitalize()} Trajectory')
        plt.xlabel('Trial Number')
        plt.ylabel(param.capitalize())
        plt.grid(True, alpha=0.3)
        
        if idx == 0:
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'parameter_analysis/trajectories_{subject_id}.png'), dpi=300, bbox_inches='tight')    
    plt.close()

def create_parameter_evolution(df):
    """
    Create visualization of parameter evolution across trials
    
    Args:
        df: DataFrame containing trial data with calculated parameters
    """
    plt.figure(figsize=(15, 10))
    
    # Sort subjects by age for visualization
    df = df.sort_values('age')
    subject_indices = range(len(df))
    
    params = {
        'theta_mean': ('Learning Ability (θ)', '#1976D2'),
        'omega_mean': ('Social Influence (ω)', '#43A047'),
        'r_mean': ('Exploration (r)', '#E53935'),
        'motor': ('Motor Skill (M)', '#FDD835')
    }
    
    for idx, (param, (label, color)) in enumerate(params.items()):
        plt.subplot(2, 2, idx + 1)
        
        # Plot parameter values
        plt.scatter(df['age'], df[param], 
                   alpha=0.6, 
                   color=color,
                   s=100)
        
        # Add trend line
        z = np.polyfit(df['age'], df[param], 1)
        p = np.poly1d(z)
        age_range = np.linspace(df['age'].min(), df['age'].max(), 100)
        plt.plot(age_range, p(age_range), 
                color='black',
                linestyle='--',
                alpha=0.8)
        
        # Add error bars if std deviation is available
        std_col = param.replace('mean', 'std')
        if std_col in df.columns:
            plt.errorbar(df['age'], df[param],
                        yerr=df[std_col],
                        fmt='none',
                        color=color,
                        alpha=0.3)
        
        plt.title(f'{label} vs Age')
        plt.xlabel('Age')
        plt.ylabel(label)
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'parameter_analysis/parameter_age_evolution.png'), dpi=300, bbox_inches='tight')    
    plt.close()

def create_parameter_relationships_3d(df):
    """
    Create 3D visualization of parameter relationships, incorporating completion status
    
    Args:
        df: DataFrame containing analysis results
    """
    fig = plt.figure(figsize=(20, 5))
    
    # First plot: θ, ω, r colored by completion status
    ax1 = fig.add_subplot(141, projection='3d')
    scatter1 = ax1.scatter(df['theta_mean'], 
                          df['omega_mean'], 
                          df['r_mean'],
                          c=df['solved'],  # Color by completion status
                          cmap='viridis')
    ax1.set_xlabel('Learning Ability (θ)')
    ax1.set_ylabel('Social Influence (ω)')
    ax1.set_zlabel('Exploration (r)')
    plt.colorbar(scatter1, label='Completed All Tasks')
    ax1.set_title('Parameter Space by Completion')
    
    # Second plot: Parameters vs Number of Unlocked Locks
    ax2 = fig.add_subplot(142, projection='3d')
    scatter2 = ax2.scatter(df['theta_mean'], 
                          df['omega_mean'], 
                          df['r_mean'],
                          c=df['num_unlock'],
                          cmap='RdYlBu')
    ax2.set_xlabel('Learning Ability (θ)')
    ax2.set_ylabel('Social Influence (ω)')
    ax2.set_zlabel('Exploration (r)')
    plt.colorbar(scatter2, label='Number of Unlocks')
    ax2.set_title('Parameter Space by Unlocks')
    
    # Third plot: Parameters vs Completion Time
    ax3 = fig.add_subplot(143, projection='3d')
    scatter3 = ax3.scatter(df['theta_mean'], 
                          df['omega_mean'], 
                          df['r_mean'],
                          c=df['unlock_time'],
                          cmap='plasma')
    ax3.set_xlabel('Learning Ability (θ)')
    ax3.set_ylabel('Social Influence (ω)')
    ax3.set_zlabel('Exploration (r)')
    plt.colorbar(scatter3, label='Completion Time (s)')
    ax3.set_title('Parameter Space by Time')
    
    # Fourth plot: Success Rate Surface with completion overlay
    ax4 = fig.add_subplot(144, projection='3d')
    x = np.linspace(df['theta_mean'].min(), df['theta_mean'].max(), 20)
    y = np.linspace(df['omega_mean'].min(), df['omega_mean'].max(), 20)
    X, Y = np.meshgrid(x, y)
    
    # Fit a 2D surface for success rate
    from scipy.interpolate import griddata
    Z = griddata((df['theta_mean'], df['omega_mean']), 
                 df['success_rate'], 
                 (X, Y),
                 method='cubic')
    
    # Plot surface and scatter points
    surf = ax4.plot_surface(X, Y, Z, cmap='viridis',
                           linewidth=0, antialiased=True, alpha=0.6)
    
    # Overlay scatter points colored by completion
    completed = df[df['solved'] == 1]
    incomplete = df[df['solved'] == 0]
    
    ax4.scatter(completed['theta_mean'], 
                completed['omega_mean'], 
                completed['success_rate'],
                c='green', marker='^', s=100, label='Completed')
    
    ax4.scatter(incomplete['theta_mean'], 
                incomplete['omega_mean'], 
                incomplete['success_rate'],
                c='red', marker='v', s=100, label='Incomplete')
    
    ax4.set_xlabel('Learning Ability (θ)')
    ax4.set_ylabel('Social Influence (ω)')
    ax4.set_zlabel('Success Rate')
    ax4.legend()
    plt.colorbar(surf, label='Success Rate')
    ax4.set_title('Success Rate Surface with Completion')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'parameter_analysis/parameter_relationships_3d.png'), dpi=300, bbox_inches='tight') 
    plt.close()

def create_cluster_analysis(df):
    """
    Create visualization of parameter clusters with improved feature processing
    
    Args:
        df: DataFrame containing analysis results
    """
    # Create cluster output directory if it doesn't exist
    os.makedirs('./data/cluster', exist_ok=True)
    
    from sklearn.preprocessing import StandardScaler
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    
    # Prepare data for clustering with adjusted weights
    features = ['theta_mean', 'omega_mean', 'r_mean', 'motor']  # Primary features
    performance_features = ['num_unlock', 'unlock_time', 'success_rate']  # Performance features
    
    # Standardize primary features
    X_primary = StandardScaler().fit_transform(df[features])
    
    # Standardize performance features and apply lower weight
    X_performance = StandardScaler().fit_transform(df[performance_features]) * 0.5
    
    # Combine features
    X = np.hstack([X_primary, X_performance])
    
    # Find optimal number of clusters using silhouette score
    silhouette_scores = []
    K = range(2, 7)
    for k in K:
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(X)
        score = silhouette_score(X, kmeans.labels_)
        silhouette_scores.append(score)
    
    # Select optimal number of clusters
    optimal_k = K[np.argmax(silhouette_scores)]
    
    # Perform clustering with optimal k
    kmeans = KMeans(n_clusters=optimal_k, random_state=42)
    df['Cluster'] = kmeans.fit_predict(X)
    
    # Figure 1: Cluster Size and Completion
    plt.figure(figsize=(10, 6))
    cluster_completion = pd.crosstab(df['Cluster'], df['solved'])
    cluster_completion.plot(kind='bar', stacked=True,
                          color=['#3498db', '#e67e22'])
    plt.title('Cluster Size and Completion Distribution', fontsize=14, pad=20)
    plt.xlabel('Cluster', fontsize=12)
    plt.ylabel('Number of Participants', fontsize=12)
    plt.legend(['Incomplete', 'Completed'], bbox_to_anchor=(1.05, 1))
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'cluster/cluster_size_completion.png'), dpi=300, bbox_inches='tight')   
    plt.close()
    
    # Figure 2: Success Rate vs Completion Time
    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(df['success_rate'], 
                         df['unlock_time'],
                         c=df['Cluster'],
                         cmap='viridis',
                         alpha=0.6,
                         s=100)
    plt.xlabel('Success Rate', fontsize=12)
    plt.ylabel('Completion Time (s)', fontsize=12)
    plt.title('Success Rate vs Completion Time by Cluster', fontsize=14, pad=20)
    plt.legend(*scatter.legend_elements(), title="Cluster")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'cluster/success_vs_completion.png'), dpi=300, bbox_inches='tight') 
    plt.close()
    
    # Figure 3: Number of Unlocks by Cluster
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df, x='Cluster', y='num_unlock', palette='viridis')
    plt.title('Number of Unlocks by Cluster', fontsize=14, pad=20)
    plt.xlabel('Cluster', fontsize=12)
    plt.ylabel('Number of Unlocks', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'cluster/unlocks_by_cluster.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Figure 4: Cluster Characteristics Heatmap
    plt.figure(figsize=(12, 6))
    cluster_means = df.groupby('Cluster')[features].mean()
    cluster_means_norm = (cluster_means - cluster_means.mean()) / cluster_means.std()
    
    sns.heatmap(cluster_means_norm,
                annot=cluster_means.round(2),
                fmt='.2f',
                cmap='RdYlBu_r',
                center=0)
    plt.title('Cluster Characteristics (Z-scored)', fontsize=14, pad=20)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'cluster/cluster_characteristics.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Figure 5: Parameter Distribution by Cluster
    plt.figure(figsize=(12, 6))
    param_data = []
    for param in ['theta_mean', 'omega_mean', 'r_mean']:
        for cluster in df['Cluster'].unique():
            param_data.append({
                'Parameter': param,
                'Value': df[df['Cluster'] == cluster][param].mean(),
                'Cluster': f'Cluster {cluster}'
            })
    param_df = pd.DataFrame(param_data)
    
    sns.barplot(data=param_df, x='Parameter', y='Value', 
                hue='Cluster', palette='viridis')
    plt.title('Parameter Distribution by Cluster', fontsize=14, pad=20)
    plt.xlabel('Parameter', fontsize=12)
    plt.ylabel('Mean Value', fontsize=12)
    plt.legend(title="Cluster")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'cluster/parameter_distribution.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # Create a combined figure with proper spacing
    fig = plt.figure(figsize=(20, 15))
    gs = plt.GridSpec(3, 2, figure=fig, height_ratios=[1, 1, 1], 
                     hspace=0.4, wspace=0.3)
    
    # 1. Cluster Size and Completion
    ax1 = fig.add_subplot(gs[0, 0])
    cluster_completion.plot(kind='bar', stacked=True, ax=ax1,
                          color=['#3498db', '#e67e22'])
    ax1.set_title('Cluster Size and Completion Distribution', fontsize=14, pad=20)
    ax1.set_xlabel('Cluster', fontsize=12)
    ax1.set_ylabel('Number of Participants', fontsize=12)
    ax1.legend(['Incomplete', 'Completed'])
    ax1.grid(True, alpha=0.3)
    
    # 2. Success Rate vs Completion Time
    ax2 = fig.add_subplot(gs[0, 1])
    scatter = ax2.scatter(df['success_rate'], 
                         df['unlock_time'],
                         c=df['Cluster'],
                         cmap='viridis',
                         alpha=0.6,
                         s=100)
    ax2.set_xlabel('Success Rate', fontsize=12)
    ax2.set_ylabel('Completion Time (s)', fontsize=12)
    ax2.set_title('Success Rate vs Completion Time', fontsize=14, pad=20)
    ax2.legend(*scatter.legend_elements(), title="Cluster")
    ax2.grid(True, alpha=0.3)
    
    # 3. Number of Unlocks by Cluster
    ax3 = fig.add_subplot(gs[1, 0])
    sns.boxplot(data=df, x='Cluster', y='num_unlock', 
                palette='viridis', ax=ax3)
    ax3.set_title('Number of Unlocks by Cluster', fontsize=14, pad=20)
    ax3.set_xlabel('Cluster', fontsize=12)
    ax3.set_ylabel('Number of Unlocks', fontsize=12)
    ax3.grid(True, alpha=0.3)
    
    # 4. Cluster Characteristics Heatmap
    ax4 = fig.add_subplot(gs[1, 1])
    sns.heatmap(cluster_means_norm,
                annot=cluster_means.round(2),
                fmt='.2f',
                cmap='RdYlBu_r',
                center=0,
                ax=ax4)
    ax4.set_title('Cluster Characteristics', fontsize=14, pad=20)
    ax4.set_yticklabels(ax4.get_yticklabels(), rotation=0)
    
    # 5. Parameter Distribution
    ax5 = fig.add_subplot(gs[2, :])
    sns.barplot(data=param_df, x='Parameter', y='Value', 
                hue='Cluster', palette='viridis', ax=ax5)
    ax5.set_title('Parameter Distribution by Cluster', fontsize=14, pad=20)
    ax5.set_xlabel('Parameter', fontsize=12)
    ax5.set_ylabel('Mean Value', fontsize=12)
    ax5.legend(title="Cluster")
    ax5.grid(True, alpha=0.3)
    
    plt.savefig(os.path.join(output_dir, 'cluster/parameter_clusters.png'), 
                dpi=300, 
                bbox_inches='tight',
                facecolor='white')
    plt.close()
    
    # Update cluster analysis report
    report = []
    report.append("Cluster Analysis Report")
    report.append("=" * 50)
    report.append(f"\nOptimal number of clusters: {optimal_k}")
    report.append(f"Silhouette score: {max(silhouette_scores):.3f}")
    
    for cluster in range(optimal_k):
        cluster_data = df[df['Cluster'] == cluster]
        report.append(f"\nCluster {cluster}:")
        report.append("-" * 20)
        report.append(f"Size: {len(cluster_data)} participants")
        report.append(f"Completion rate: {cluster_data['solved'].mean():.2%}")
        report.append(f"Average unlocks: {cluster_data['num_unlock'].mean():.2f}")
        report.append(f"Average completion time: {cluster_data['unlock_time'].mean():.2f}s")
        report.append(f"Average success rate: {cluster_data['success_rate'].mean():.2%}")
        report.append("\nParameter averages:")
        report.append(f"Learning ability (θ): {cluster_data['theta_mean'].mean():.3f}")
        report.append(f"Social influence (ω): {cluster_data['omega_mean'].mean():.3f}")
        report.append(f"Exploration rate (r): {cluster_data['r_mean'].mean():.3f}")
        report.append(f"Motor skill: {cluster_data['motor'].mean():.3f}")
    
    # Save cluster data and report
    cluster_data = df[['ID', 'Cluster', 'theta_mean', 'omega_mean', 'r_mean', 
                      'motor', 'num_unlock', 'solved', 'unlock_time', 'success_rate']]
    cluster_data.to_csv(os.path.join(output_dir, 'data/cluster_assignments.csv'), index=False)
    
    with open(os.path.join(output_dir, 'reports/cluster_analysis_report.txt'), 'w') as f:
        f.write('\n'.join(report))
    
    # Save cluster statistics
    cluster_stats = df.groupby('Cluster').agg({
        'theta_mean': ['mean', 'std'],
        'omega_mean': ['mean', 'std'],
        'r_mean': ['mean', 'std'],
        'motor': ['mean', 'std'],
        'num_unlock': ['mean', 'std'],
        'solved': ['mean', 'sum'],
        'unlock_time': ['mean', 'std'],
        'success_rate': ['mean', 'std']
    }).round(3)
    
    cluster_stats.to_csv(os.path.join(output_dir, 'data/cluster_statistics.csv'))

def generate_analysis_report(results_df, all_posteriors):
    """
    Generate a comprehensive analysis report with enhanced statistical analysis
    
    Args:
        results_df: DataFrame containing analysis results
        all_posteriors: Dictionary containing posterior distributions
    """
    report = []
    
    # 1. Basic Statistics
    report.append("Basic Statistical Analysis")
    report.append("=" * 50)
    
    # Sample size and basic power analysis
    n_subjects = len(results_df)
    # Calculate approximate power for medium effect size
    effect_size = 0.5  # Cohen's d for medium effect
    df = n_subjects - 1
    alpha = 0.05
    nc = effect_size * np.sqrt(n_subjects)
    power = 1 - stats.t.cdf(stats.t.ppf(1-alpha/2, df), df, nc)
    
    report.append("\nSample Statistics and Power Analysis:")
    report.append(f"Total number of subjects: {n_subjects}")
    report.append(f"Statistical power (medium effect): {power:.3f}")
    report.append(f"Age range: {results_df['age'].min():.1f} - {results_df['age'].max():.1f} years")
    report.append(f"Average number of trials: {results_df['total_trials'].mean():.1f} ± {results_df['total_trials'].std():.1f}")
    
    # Age distribution analysis
    age_shapiro = stats.shapiro(results_df['age'])
    report.append("\nAge Distribution Analysis:")
    report.append(f"Age normality test (Shapiro-Wilk): W={age_shapiro.statistic:.3f}, p={age_shapiro.pvalue:.3f}")
    report.append(f"Age distribution: {'Normal' if age_shapiro.pvalue > 0.05 else 'Non-normal'}")
    
    # Success rate analysis
    report.append("\nSuccess Rate Analysis:")
    success_stats = stats.describe(results_df['success_rate'])
    report.append(f"Mean success rate: {success_stats.mean:.3f} ± {np.sqrt(success_stats.variance):.3f}")
    report.append(f"Success rate range: {success_stats.minmax[0]:.3f} - {success_stats.minmax[1]:.3f}")
    report.append(f"Skewness: {success_stats.skewness:.3f}")
    report.append(f"Kurtosis: {success_stats.kurtosis:.3f}")
    
    # One-sample t-test against chance level (0.5)
    t_stat, p_val = stats.ttest_1samp(results_df['success_rate'], 0.5)
    report.append(f"\nPerformance vs. Chance Level:")
    report.append(f"One-sample t-test: t={t_stat:.3f}, p={p_val:.3f}")
    report.append(f"Interpretation: {'Above' if t_stat > 0 else 'Below'} chance level (p {'<' if p_val < 0.05 else '>'} 0.05)")
    
    # 2. Parameter Analysis with Effect Sizes
    report.append("\nParameter Analysis")
    report.append("=" * 50)
    
    params = {
        'theta_mean': 'Learning Ability (θ)',
        'omega_mean': 'Social Influence Weight (ω)',
        'r_mean': 'Exploration Rate (r)',
        'motor': 'Motor Skill (M)'
    }
    
    for param, name in params.items():
        # Basic statistics
        param_stats = stats.describe(results_df[param])
        report.append(f"\n{name}:")
        report.append(f"Mean: {param_stats.mean:.3f} ± {np.sqrt(param_stats.variance):.3f}")
        report.append(f"Median: {np.median(results_df[param]):.3f}")
        report.append(f"Range: {param_stats.minmax[0]:.3f} - {param_stats.minmax[1]:.3f}")
        
        # Distribution analysis
        shapiro_test = stats.shapiro(results_df[param])
        report.append(f"Normality test: W={shapiro_test.statistic:.3f}, p={shapiro_test.pvalue:.3f}")
        
        # Effect size (Cohen's d) relative to neutral point (0.5)
        d = (param_stats.mean - 0.5) / np.sqrt(param_stats.variance)
        report.append(f"Effect size (vs. neutral): d={d:.3f}")
        report.append(f"Effect magnitude: {get_effect_size_interpretation(d)}")
    
    # 3. Enhanced Correlation Analysis with Statistical Tests
    report.append("\nCorrelation Analysis")
    report.append("=" * 50)
    
    # Define parameter pairs for correlation analysis
    param_pairs = [
        ('theta_mean', 'omega_mean'),
        ('theta_mean', 'r_mean'),
        ('theta_mean', 'motor'),
        ('omega_mean', 'r_mean'),
        ('omega_mean', 'motor'),
        ('r_mean', 'motor'),
        ('theta_mean', 'success_rate'),
        ('omega_mean', 'success_rate'),
        ('r_mean', 'success_rate'),
        ('motor', 'success_rate')
    ]
    
    for param1, param2 in param_pairs:
        # Calculate correlation
        r, p = stats.pearsonr(results_df[param1], results_df[param2])
        
        # Only report significant or strong correlations
        if abs(r) > 0.3 or p < 0.05:
            report.append(f"\n{params.get(param1, param1)} and {params.get(param2, param2)}:")
            report.append(f"Correlation coefficient: r={r:.3f}")
            report.append(f"Statistical significance: p={p:.3f}")
            report.append(f"Effect size interpretation: {get_correlation_interpretation(r)}")
            
            # Add regression analysis
            slope, intercept, r_value, p_value, std_err = stats.linregress(
                results_df[param1], results_df[param2]
            )
            report.append(f"Regression analysis:")
            report.append(f"  Slope: {slope:.3f} ± {std_err:.3f}")
            report.append(f"  R-squared: {r_value**2:.3f}")
            
            # Add partial correlations controlling for other variables
            other_params = [p for p in params.keys() if p not in [param1, param2]]
            if other_params:
                for control_param in other_params:
                    partial_r = partial_correlation(
                        results_df[param1],
                        results_df[param2],
                        results_df[control_param]
                    )
                    report.append(f"Partial correlation (controlling for {params[control_param]}): r={partial_r:.3f}")
    
    # 4. Strategy Analysis with Comparative Statistics
    report.append("\nStrategy Usage Analysis")
    report.append("=" * 50)
    
    strategies = ['color_match_rate', 'num_match_rate', 'shape_match_rate']
    strategy_names = {
        'color_match_rate': 'Color Matching',
        'num_match_rate': 'Number Matching',
        'shape_match_rate': 'Shape Matching'
    }
    
    # ANOVA test for strategy differences
    strategy_data = [results_df[strat] for strat in strategies]
    f_stat, p_val = stats.f_oneway(*strategy_data)
    report.append("\nStrategy Comparison (ANOVA):")
    report.append(f"F-statistic: {f_stat:.3f}")
    report.append(f"p-value: {p_val:.3f}")
    
    for strategy in strategies:
        report.append(f"\n{strategy_names[strategy]} Strategy:")
        strategy_stats = stats.describe(results_df[strategy])
        report.append(f"Usage rate: {strategy_stats.mean:.3f} ± {np.sqrt(strategy_stats.variance):.3f}")
        
        # Correlation with success rate and parameters
        for param, param_name in params.items():
            corr, p = stats.pearsonr(results_df[strategy], results_df[param])
            if abs(corr) > 0.3 or p < 0.05:
                report.append(f"Correlation with {param_name}: r={corr:.3f}, p={p:.3f}")
                if p < 0.05:
                    report.append(f"Effect size interpretation: {get_correlation_interpretation(corr)}")
    
    # 5. High Performers Analysis with Group Comparisons
    report.append("\nHigh Performers Analysis")
    report.append("=" * 50)
    
    success_threshold = results_df['success_rate'].mean() + results_df['success_rate'].std()
    high_performers = results_df[results_df['success_rate'] >= success_threshold]
    low_performers = results_df[results_df['success_rate'] < success_threshold]
    
    report.append(f"\nNumber of high performers: {len(high_performers)} ({len(high_performers)/len(results_df)*100:.1f}%)")
    
    # Compare high vs low performers
    report.append("\nHigh vs. Low Performers Comparison:")
    for param in params:
        t_stat, p_val = stats.ttest_ind(high_performers[param], low_performers[param])
        cohens_d = (high_performers[param].mean() - low_performers[param].mean()) / \
                   np.sqrt((high_performers[param].var() + low_performers[param].var()) / 2)
        report.append(f"\n{params[param]}:")
        report.append(f"High performers: {high_performers[param].mean():.3f} ± {high_performers[param].std():.3f}")
        report.append(f"Low performers: {low_performers[param].mean():.3f} ± {low_performers[param].std():.3f}")
        report.append(f"t-statistic: {t_stat:.3f}, p-value: {p_val:.3f}")
        report.append(f"Effect size (Cohen's d): {cohens_d:.3f}")
        report.append(f"Effect magnitude: {get_effect_size_interpretation(cohens_d)}")
    
    # Save report
    with open(os.path.join(output_dir, 'reports/analysis_report.txt'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(report))
    
    print(f"Analysis report has been saved to '{os.path.join(output_dir, 'reports/analysis_report.txt')}'")
    return report

def get_effect_size_interpretation(d):
    """Interpret Cohen's d effect size"""
    if abs(d) < 0.2:
        return "Negligible"
    elif abs(d) < 0.5:
        return "Small"
    elif abs(d) < 0.8:
        return "Medium"
    else:
        return "Large"

def get_correlation_interpretation(r):
    """Interpret correlation coefficient"""
    if abs(r) < 0.1:
        return "Negligible"
    elif abs(r) < 0.3:
        return "Weak"
    elif abs(r) < 0.5:
        return "Moderate"
    elif abs(r) < 0.7:
        return "Strong"
    else:
        return "Very strong"

def create_enhanced_strategy_analysis(df):
    """
    Create enhanced strategy analysis visualization
    
    Args:
        df: DataFrame containing trial data with strategy columns
    """
    # Create figure
    fig = plt.figure(figsize=(20, 10))
    gs = plt.GridSpec(2, 2)
    
    # 1. Strategy transition probability matrix
    ax1 = fig.add_subplot(gs[0, 0])
    
    # Calculate primary strategy for each trial
    df['PrimaryStrategy'] = df.apply(lambda row: 
        'Color' if row['ColorMatch'] else
        'Number' if row['NumMatch'] else
        'Shape' if row['ShapeMatch'] else 'None', axis=1)
    
    # Calculate strategy transition matrix
    transitions = []
    for subject_id in df['ID'].unique():
        subject_data = df[df['ID'] == subject_id]['PrimaryStrategy']
        for i in range(len(subject_data)-1):
            transitions.append((subject_data.iloc[i], subject_data.iloc[i+1]))
    
    transition_df = pd.DataFrame(transitions, columns=['From', 'To'])
    transition_matrix = pd.crosstab(transition_df['From'], transition_df['To'], normalize='index')
    
    # Create heatmap
    sns.heatmap(transition_matrix, annot=True, fmt='.2f', cmap='YlOrRd', ax=ax1)
    ax1.set_title('Strategy Transition Probabilities')
    
    # 2. Strategy usage evolution
    ax2 = fig.add_subplot(gs[0, 1])
    
    # Divide trials into 10 phases
    df['Phase'] = df.groupby('ID')['Order'].transform(
        lambda x: pd.qcut(x, q=10, labels=['P'+str(i) for i in range(1, 11)])
    )
    
    # Calculate strategy usage proportion for each phase
    phase_strategies = df.groupby('Phase')[['ColorMatch', 'NumMatch', 'ShapeMatch']].mean()
    
    # Plot strategy usage evolution
    for strategy in ['ColorMatch', 'NumMatch', 'ShapeMatch']:
        ax2.plot(range(10), phase_strategies[strategy], 
                marker='o', label=strategy.replace('Match', ''))
    
    ax2.set_xticks(range(10))
    ax2.set_xticklabels(['P'+str(i) for i in range(1, 11)])
    ax2.set_xlabel('Learning Phase')
    ax2.set_ylabel('Strategy Usage Probability')
    ax2.set_title('Strategy Usage Evolution')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Strategy success rate analysis
    ax3 = fig.add_subplot(gs[1, 0])
    
    # Calculate success and error rates for each strategy
    strategy_success = []
    for strategy in ['ColorMatch', 'NumMatch', 'ShapeMatch']:
        strategy_data = df[df[strategy] == 1]
        success_rate = strategy_data['Worked'].mean() if len(strategy_data) > 0 else 0
        error_rate = strategy_data['Error'].mean() if len(strategy_data) > 0 else 0
        strategy_success.append({
            'Strategy': strategy.replace('Match', ''),
            'Success': success_rate,
            'Error': error_rate,
            'Neutral': 1 - success_rate - error_rate
        })
    
    strategy_success_df = pd.DataFrame(strategy_success)
    
    # Create stacked bar plot
    strategy_success_df.plot(x='Strategy', y=['Success', 'Error', 'Neutral'], 
                         kind='bar', stacked=True, ax=ax3)
    ax3.set_title('Strategy Outcomes Analysis')
    ax3.set_ylabel('Proportion')
    
    # Add chi-square test results
    contingency = pd.crosstab(df['PrimaryStrategy'], df['Worked'])
    chi2, p_val = stats.chi2_contingency(contingency)[:2]
    ax3.text(0.05, -0.15, f'χ² = {chi2:.2f}, p = {p_val:.3f}', 
             transform=ax3.transAxes)
    
    # 4. Strategy combination analysis
    ax4 = fig.add_subplot(gs[1, 1])
    
    # Calculate strategy combinations
    df['StrategyCombo'] = df.apply(lambda row: 
        '+'.join(sorted([s.replace('Match', '') for s, v in 
                        {'Color': row['ColorMatch'], 
                         'Number': row['NumMatch'], 
                         'Shape': row['ShapeMatch']}.items() if v == 1])), axis=1)
    
    # Calculate usage frequency and success rate for each combination
    combo_stats = df.groupby('StrategyCombo').agg({
        'ID': 'count',
        'Worked': 'mean'
    }).reset_index()
    combo_stats['Usage'] = combo_stats['ID'] / len(df)
    
    # Create scatter plot
    scatter = ax4.scatter(combo_stats['Usage'], combo_stats['Worked'], 
                         s=1000*combo_stats['Usage'], alpha=0.6)
    
    # Add labels
    for i, row in combo_stats.iterrows():
        ax4.annotate(row['StrategyCombo'], 
                    (row['Usage'], row['Worked']),
                    xytext=(5, 5), textcoords='offset points')
    
    ax4.set_xlabel('Usage Frequency')
    ax4.set_ylabel('Success Rate')
    ax4.set_title('Strategy Combination Analysis')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'strategy_analysis/enhanced_strategy_analysis.png'), dpi=300, bbox_inches='tight')
    plt.close()

def create_gender_parameter_comparison(results_df, trial_data):
    """
    Create visualizations comparing parameter distributions between genders.
    
    Args:
        results_df (pd.DataFrame): DataFrame containing parameter results
        trial_data (pd.DataFrame): DataFrame containing trial data with gender information
    """
    # Initialize report list
    report = []
    report.append("Gender Parameter Comparison Report")
    report.append("=" * 50)
    
    # Correctly merge gender information
    gender_info = trial_data.groupby('ID')['Gender'].first()
    merged_df = results_df.copy()
    merged_df['Gender'] = merged_df['ID'].map(gender_info)
    
    # Parameters to analyze
    params = ['theta_mean', 'omega_mean', 'r_mean', 'motor']
    param_labels = ['Theta (Learning Rate)', 'Omega (Forgetting Rate)', 'R (Reward Sensitivity)', 'Motor Noise']
    
    # Create figure
    plt.figure(figsize=(15, 10))
    
    # Create violin plots for each parameter
    for i, (param, label) in enumerate(zip(params, param_labels), 1):
        plt.subplot(2, 2, i)
        
        # Get data for each gender
        boys_data = merged_df[merged_df['Gender'] == 'Boy'][param]
        girls_data = merged_df[merged_df['Gender'] == 'Girl'][param]
        
        # Create violin plot
        positions = [1, 2]
        parts = plt.violinplot([boys_data, girls_data], positions=positions, showmeans=True)
        
        # Customize violin plot colors
        for pc in parts['bodies']:
            pc.set_facecolor(COLOR_PALETTE['primary'])
            pc.set_alpha(0.3)
        parts['cmeans'].set_color(COLOR_PALETTE['accent'])
        parts['cmaxes'].set_color(COLOR_PALETTE['neutral'])
        parts['cmins'].set_color(COLOR_PALETTE['neutral'])
        parts['cbars'].set_color(COLOR_PALETTE['neutral'])
        
        # Add scatter points
        for j, (data, pos) in enumerate([(boys_data, 1), (girls_data, 2)]):
            plt.scatter(np.random.normal(pos, 0.05, size=len(data)), data, 
                       alpha=0.6, s=40, 
                       c=COLOR_PALETTE['boys'] if j == 0 else COLOR_PALETTE['girls'])
        
        # Perform Mann-Whitney U test
        stat, p_value = stats.mannwhitneyu(boys_data, girls_data, alternative='two-sided')
        
        # Calculate effect size (Cohen's d)
        d = (boys_data.mean() - girls_data.mean()) / np.sqrt((boys_data.var() + girls_data.var()) / 2)
        report.append(f"\nStatistics for {label}:")
        report.append(f"Boys (n={len(boys_data)}): mean={boys_data.mean():.3f}, std={boys_data.std():.3f}")
        report.append(f"Girls (n={len(girls_data)}): mean={girls_data.mean():.3f}, std={girls_data.std():.3f}")
        report.append(f"Mann-Whitney U test: p={p_value:.3f}")
        report.append(f"Effect size (Cohen's d): {d:.3f}")
        
        # Add plot labels
        plt.title(f'{label} by Gender')
        plt.xticks([1, 2], ['Boys', 'Girls'])
        plt.ylabel(label)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'gender_analysis/gender_parameter_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save the report
    with open(os.path.join(output_dir, 'reports/gender_parameter_comparison_report.txt'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(report))

def create_enhanced_parameter_distribution(results_df):
    """
    Create Q-Q plots for parameter distributions
    
    Args:
        results_df: DataFrame containing analysis results
    """
    # Create figure with adjusted size and spacing
    fig = plt.figure(figsize=(16, 4))
    
    params = {
        'theta_mean': 'Learning Ability (θ)',
        'omega_mean': 'Social Influence (ω)',
        'r_mean': 'Exploration (r)',
        'motor': 'Motor Skill (M)'
    }
    
    # Q-Q plots and normality tests
    for idx, (param, label) in enumerate(params.items()):
        ax = plt.subplot(1, 4, idx + 1)
        
        # Q-Q plot with custom colors
        stats.probplot(results_df[param], dist="norm", plot=ax)
        
        # Customize plot appearance
        ax.get_lines()[0].set_markerfacecolor(PARAM_COLORS[param.split('_')[0]])
        ax.get_lines()[0].set_markeredgecolor(COLOR_PALETTE['neutral'])
        ax.get_lines()[0].set_alpha(0.6)
        ax.get_lines()[1].set_color(COLOR_PALETTE['accent'])
        
        ax.set_title(f'{label}\nQ-Q Plot', pad=20)
        
        # Add Shapiro-Wilk test results with enhanced style
        stat, p_val = stats.shapiro(results_df[param])
        ax.text(0.05, 0.95, 
                f'Shapiro-Wilk Test:\nW={stat:.3f}\np={p_val:.3f}',
                transform=ax.transAxes,
                bbox=dict(facecolor='white',
                         alpha=0.9,
                         edgecolor=COLOR_PALETTE['neutral'],
                         boxstyle='round,pad=0.5'),
                verticalalignment='top')
        
        # Customize grid
        ax.grid(True, alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'distribution_analysis/parameter_qq_plots.png'), dpi=300, bbox_inches='tight')
    plt.close()

def create_learning_influence_network(results_df):
    """
    Create visualization of learning influence network
    
    Args:
        results_df: DataFrame containing analysis results
    """
    G = nx.Graph()
    
    # Use stricter success rate threshold
    success_threshold = results_df['success_rate'].mean() + 0.5 * results_df['success_rate'].std()
    key_subjects = results_df[results_df['success_rate'] >= success_threshold]
    
    # Add nodes with additional information
    for _, row in key_subjects.iterrows():
        G.add_node(row['ID'], 
                  omega=row['omega_mean'],
                  success_rate=row['success_rate'],
                  theta=row['theta_mean'],
                  r=row['r_mean'])
    
    # Calculate similarity based on multiple parameters
    for i, row1 in key_subjects.iterrows():
        for j, row2 in key_subjects.iterrows():
            if i < j:
                # Calculate comprehensive similarity
                omega_sim = 1 / (1 + abs(row1['omega_mean'] - row2['omega_mean']))
                theta_sim = 1 / (1 + abs(row1['theta_mean'] - row2['theta_mean']))
                r_sim = 1 / (1 + abs(row1['r_mean'] - row2['r_mean']))
                
                similarity = (omega_sim + theta_sim + r_sim) / 3
                if similarity > 0.85:  # Increase similarity threshold
                    G.add_edge(row1['ID'], row2['ID'], 
                             weight=similarity,
                             color='gray' if similarity < 0.9 else 'black')
    
    plt.figure(figsize=(12, 12))
    pos = nx.spring_layout(G, k=1.5, iterations=50)  # Increase node spacing
    
    # Draw edges
    edges = G.edges()
    edge_colors = [G[u][v].get('color', 'gray') for u, v in edges]
    edge_weights = [G[u][v]['weight'] * 2 for u, v in edges]
    nx.draw_networkx_edges(G, pos, 
                          edge_color=edge_colors,
                          width=edge_weights,
                          alpha=0.5)
    
    # Draw nodes
    node_colors = [G.nodes[node]['success_rate'] for node in G.nodes()]
    node_sizes = [G.nodes[node]['omega'] * 1000 for node in G.nodes()]  # Increase node size
    nodes = nx.draw_networkx_nodes(G, pos,
                                 node_color=node_colors,
                                 node_size=node_sizes,
                                 cmap=plt.cm.viridis,
                                 alpha=0.8)
    
    # Add node labels
    labels = {node: f"{node}\nω={G.nodes[node]['omega']:.2f}\nθ={G.nodes[node]['theta']:.2f}" 
             for node in G.nodes()}
    nx.draw_networkx_labels(G, pos, labels, font_size=8)
    
    # Add color bar
    plt.colorbar(nodes, label='Success Rate')
    
    plt.title('Key Learning Influence Network\n(High Success Rate Subjects)')
    plt.axis('off')
    plt.savefig(os.path.join(output_dir, 'correlation_analysis/learning_influence_network.png'), dpi=300, bbox_inches='tight')
    plt.close()

def create_model_evaluation_plot(results_df):
    """
    Create model evaluation visualization
    
    Args:
        results_df: DataFrame containing analysis results
    """
    actuals = np.array(results_df['success_rate'])
    predictions = np.array(results_df['theta_mean'])
    
    fpr, tpr, _ = roc_curve(actuals > actuals.mean(), predictions)
    roc_auc = auc(fpr, tpr)
    
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
    
    # ROC Curve
    ax1.plot(fpr, tpr, color='darkorange', lw=2, 
            label=f'ROC curve (AUC = {roc_auc:.2f})')
    ax1.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    ax1.set_xlim([0.0, 1.0])
    ax1.set_ylim([0.0, 1.05])
    ax1.set_xlabel('False Positive Rate')
    ax1.set_ylabel('True Positive Rate')
    ax1.set_title('ROC Curve')
    ax1.legend(loc="lower right")
    
    # Residual Analysis
    residuals = predictions - actuals
    ax2.scatter(predictions, residuals, alpha=0.5)
    ax2.axhline(y=0, color='r', linestyle='--')
    ax2.set_xlabel('Predicted Values')
    ax2.set_ylabel('Residuals')
    ax2.set_title('Residual Analysis')
    
    # Predicted vs Actual
    ax3.scatter(actuals, predictions, alpha=0.5)
    ax3.plot([actuals.min(), actuals.max()], 
             [actuals.min(), actuals.max()], 'r--', lw=2)
    ax3.set_xlabel('Actual Values')
    ax3.set_ylabel('Predicted Values')
    ax3.set_title('Predicted vs Actual Values')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'model_evaluation/model_evaluation.png'), dpi=300, bbox_inches='tight')
    plt.close()

def create_completion_analysis(results_df: pd.DataFrame):
    """
    Create detailed analysis visualization for task completion
    
    Args:
        results_df: DataFrame containing analysis results
    """
    plt.figure(figsize=(20, 15))
    
    # 1. Completion time distribution
    plt.subplot(3, 2, 1)
    sns.histplot(data=results_df, x='unlock_time', hue='solved', bins=20)
    plt.title('Completion Time Distribution')
    plt.xlabel('Completion Time (seconds)')
    plt.ylabel('Frequency')
    
    # 2. Age vs completion status
    plt.subplot(3, 2, 2)
    sns.boxplot(data=results_df, x='solved', y='age')
    plt.title('Age vs Completion Status')
    plt.xlabel('All Tasks Completed')
    plt.ylabel('Age')
    
    # 3. Strategy use vs completion status
    plt.subplot(3, 2, 3)
    strategy_data = pd.melt(results_df, 
                           value_vars=['color_match_rate', 'num_match_rate', 'shape_match_rate'],
                           id_vars=['solved'])
    sns.boxplot(data=strategy_data, x='variable', y='value', hue='solved')
    plt.title('Strategy Use vs Completion Status')
    plt.xlabel('Strategy Type')
    plt.ylabel('Usage Rate')
    
    # 4. Number of locks completed distribution
    plt.subplot(3, 2, 4)
    sns.countplot(data=results_df, x='num_unlock')
    plt.title('Distribution of Completed Locks')
    plt.xlabel('Number of Locks Completed')
    plt.ylabel('Number of Participants')
    
    # 5. Gender vs completion status (if gender column exists)
    plt.subplot(3, 2, 5)
    if 'gender' in results_df.columns:
        completion_by_gender = pd.crosstab(results_df['gender'], results_df['solved'], normalize='index')
        completion_by_gender.plot(kind='bar', stacked=True)
        plt.title('Gender vs Completion Status')
        plt.xlabel('Gender')
        plt.ylabel('Proportion')
    else:
        plt.text(0.5, 0.5, 'Gender data not available', 
                horizontalalignment='center',
                verticalalignment='center')
        plt.title('Gender Analysis (Data Not Available)')
    
    # 6. Parameters vs completion status
    plt.subplot(3, 2, 6)
    param_completion = results_df.groupby('solved')[['theta_mean', 'omega_mean', 'r_mean']].mean()
    param_completion.plot(kind='bar')
    plt.title('Parameters vs Completion Status')
    plt.xlabel('All Tasks Completed')
    plt.ylabel('Average Parameter Value')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'performance_analysis/completion_analysis.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create statistical report
    report = []
    report.append("Task Completion Analysis Report")
    report.append("=" * 50)
    
    # 1. Overall completion
    report.append("\n1. Overall Completion")
    report.append(f"Total participants: {len(results_df)}")
    report.append(f"Participants who completed all tasks: {results_df['solved'].sum()}")
    report.append(f"Completion rate: {results_df['solved'].mean():.2%}")
    
    # 2. Age analysis
    report.append("\n2. Age Analysis")
    age_stats = results_df.groupby('solved')['age'].agg(['mean', 'std'])
    report.append("Completers average age: {:.2f} ± {:.2f}".format(
        age_stats.loc[1, 'mean'], age_stats.loc[1, 'std']))
    report.append("Non-completers average age: {:.2f} ± {:.2f}".format(
        age_stats.loc[0, 'mean'], age_stats.loc[0, 'std']))
    
    # Perform t-test
    t_stat, p_val = stats.ttest_ind(
        results_df[results_df['solved'] == 1]['age'],
        results_df[results_df['solved'] == 0]['age']
    )
    report.append(f"Age difference t-test: t = {t_stat:.3f}, p = {p_val:.3f}")
    
    # 3. Gender analysis (if gender column exists)
    if 'gender' in results_df.columns:
        report.append("\n3. Gender Analysis")
        gender_completion = pd.crosstab(results_df['gender'], results_df['solved'])
        chi2, p_val, dof, expected = stats.chi2_contingency(gender_completion)
        report.append("Gender vs completion contingency table:")
        report.append(str(gender_completion))
        report.append(f"Chi-square test: χ² = {chi2:.3f}, p = {p_val:.3f}")
    else:
        report.append("\n3. Gender Analysis")
        report.append("Gender data not available")
    
    # 4. Strategy analysis
    report.append("\n4. Strategy Analysis")
    for strategy in ['color_match_rate', 'num_match_rate', 'shape_match_rate']:
        t_stat, p_val = stats.ttest_ind(
            results_df[results_df['solved'] == 1][strategy],
            results_df[results_df['solved'] == 0][strategy]
        )
        report.append(f"\n{strategy}:")
        report.append("Completers: {:.2%} ± {:.2%}".format(
            results_df[results_df['solved'] == 1][strategy].mean(),
            results_df[results_df['solved'] == 1][strategy].std()
        ))
        report.append("Non-completers: {:.2%} ± {:.2%}".format(
            results_df[results_df['solved'] == 0][strategy].mean(),
            results_df[results_df['solved'] == 0][strategy].std()
        ))
        report.append(f"t-test: t = {t_stat:.3f}, p = {p_val:.3f}")
    
    # 5. Completion time analysis
    report.append("\n5. Completion Time Analysis")
    time_stats = results_df.groupby('solved')['unlock_time'].agg(['mean', 'std', 'min', 'max'])
    report.append("Completers:")
    report.append("Average time: {:.2f} ± {:.2f} seconds".format(
        time_stats.loc[1, 'mean'], time_stats.loc[1, 'std']))
    report.append("Fastest time: {:.2f} seconds".format(time_stats.loc[1, 'min']))
    report.append("Slowest time: {:.2f} seconds".format(time_stats.loc[1, 'max']))
    
    # Save report
    with open(os.path.join(output_dir, 'reports/completion_analysis_report.txt'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(report))

def create_output_directories():
    """Create all necessary output directories."""
    subdirs = [
        'distribution_analysis', 
        'correlation_analysis', 
        'parameter_analysis',
        'strategy_analysis',
        'learning_dynamics', 
        'cluster', 
        'model_evaluation',
        'performance_analysis',
        'reports', 
        'data'
    ]
    
    for subdir in subdirs:
        os.makedirs(os.path.join(output_dir, subdir), exist_ok=True)

def generate_all_visualizations(results_df=None, all_posteriors=None):
    """Generate all visualizations for the analysis.
    
    Args:
        results_df: DataFrame containing analysis results. If None, will load from file.
        all_posteriors: Dictionary containing posterior distributions. If None, will load from file.
    """
    create_output_directories()
    
    # Load data from files if not provided
    if results_df is None:
        try:
            results_df = pd.read_csv(os.path.join(output_dir, 'data/individual_results.csv'))
        except FileNotFoundError:
            raise FileNotFoundError("Results file not found. Please run the analysis first.")
    
    if all_posteriors is None:
        try:
            with open(os.path.join(output_dir, 'data/posterior_distributions.pkl'), 'rb') as f:
                all_posteriors = pickle.load(f)
        except FileNotFoundError:
            raise FileNotFoundError("Posterior distributions file not found. Please run the analysis first.")
    
    print("Generating analysis report...")
    generate_analysis_report(results_df, all_posteriors)
    
    print("Generating parameter distribution plot...")
    fig = plt.figure(figsize=(15, 5))
    gs = plt.GridSpec(1, 3, wspace=0.3)
    
    params = {
        'theta_mean': ('Learning Ability (θ)', 'blue'),
        'omega_mean': ('Social Influence (ω)', 'green'),
        'r_mean': ('Exploration (r)', 'red')
    }
    
    for idx, (param, (label, color)) in enumerate(params.items()):
        ax = plt.subplot(gs[idx])
        
        # Plot individual distributions
        sns.kdeplot(data=results_df, x=param, alpha=0.5, color=color, 
                   label='Population Distribution')
        
        # Plot mean and std
        plt.axvline(results_df[param].mean(), color='black', linestyle='--', alpha=0.5)
        plt.axvspan(results_df[param].mean() - results_df[param].std(),
                   results_df[param].mean() + results_df[param].std(),
                   color=color, alpha=0.1)
        
        plt.title(f'Distribution of {label}')
        plt.xlabel(label)
        plt.ylabel('Density')
        plt.grid(True, alpha=0.3)
        plt.legend()
    
    plt.savefig(os.path.join(output_dir, 'distribution_analysis/parameter_distributions.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Generating correlation heatmap...")
    # Create correlation heatmap
    key_vars = ['theta_mean', 'omega_mean', 'r_mean', 'motor', 'success_rate']
    correlation_matrix = results_df[key_vars].corr()
    
    plt.figure(figsize=(10, 8))
    mask = np.triu(np.ones_like(correlation_matrix), k=1)
    sns.heatmap(correlation_matrix, 
                mask=mask,
                annot=True, 
                cmap='coolwarm', 
                center=0,
                square=True,
                fmt='.2f',
                vmin=-1, 
                vmax=1)
    
    plt.title('Correlation Matrix of Key Parameters')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'correlation_analysis/correlation_heatmap.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Generating parameter relationships plot...")
    # Create 3D parameter relationships plot
    create_parameter_relationships_3d(results_df)
    
    print("Generating cluster analysis...")
    # Create cluster analysis
    create_cluster_analysis(results_df)
    
    print("Generating strategy analysis...")
    # Load trial data for strategy analysis
    merged_data, _ = load_and_merge_data()
    create_strategy_heatmap(merged_data)
    create_strategy_sequence(merged_data)
    
    print("Generating parameter evolution plot...")
    create_parameter_evolution(results_df)
    
    print("Generating model evaluation plots...")
    create_model_evaluation_plot(results_df)
    
    print("Generating performance analysis plots...")
    create_completion_analysis(results_df)
    
    print("Generating learning dynamics plots...")
    # Load trajectories for learning dynamics
    try:
        with open(os.path.join(output_dir, 'data/parameter_trajectories.pkl'), 'rb') as f:
            all_trajectories = pickle.load(f)
        analyze_learning_dynamics(results_df, all_trajectories)
    except FileNotFoundError:
        print("Warning: Parameter trajectories file not found. Skipping learning dynamics analysis.")
    
    print("All visualizations completed!")

def analyze_individual_differences(generate_plots=True):
    """Analyze individual differences and strategy combinations"""
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
        
        # Store posterior distributions (approximated from final particle distribution)
        all_posteriors[subject_id] = {
            'theta_mean': results['theta'],
            'omega_mean': results['omega'],
            'r_mean': results['r']
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
        
        # Generate and save individual trajectory plots
        if generate_plots:
            fig = create_parameter_trajectory_plot(subject_data, results['trajectories'])
            plt.savefig(os.path.join(output_dir, f'parameter_analysis/trajectories_{subject_id}.png'), 
                       dpi=300, bbox_inches='tight')
            plt.close()
    
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
    
    # Print completion statistics
    print("\nCompletion Statistics:")
    print("-" * 50)
    print(f"Total participants: {len(results_df)}")
    print(f"Participants who completed all tasks: {results_df['solved'].sum()}")
    print(f"Completion rate: {results_df['solved'].mean():.2%}")
    print("\nCompletion by gender:")
    print(results_df.groupby('gender')['solved'].agg(['count', 'mean', 'sum']))
    print("\nCompletion by age group:")
    age_groups = pd.qcut(results_df['age'], q=4)
    print(results_df.groupby(age_groups)['solved'].agg(['count', 'mean', 'sum']))
    
    # Generate visualizations
    if generate_plots:
        print("Starting visualization generation...")
        generate_all_visualizations(results_df, all_posteriors)
        
        print("Analyzing learning dynamics...")
        analyze_learning_dynamics(results_df, all_trajectories)
    
    return results_df, all_posteriors

def analyze_learning_dynamics(results_df: pd.DataFrame, all_trajectories: Dict):
    """
    Analyze learning dynamics using parameter trajectories
    
    Args:
        results_df: DataFrame containing analysis results
        all_trajectories: Dictionary containing parameter trajectories for all subjects
    """
    # Create output directory
    os.makedirs(os.path.join(output_dir, 'learning_dynamics'), exist_ok=True) 
    
    # Initialize report
    report = []
    report.append("Learning Dynamics Analysis Report")
    report.append("=" * 50)
    
    # 1. Analyze learning phases
    plt.figure(figsize=(15, 10))
    
    # Calculate average parameter trajectories
    avg_trajectories = {
        'theta': [],
        'omega': [],
        'r': []
    }
    
    max_trials = max(traj['theta'].shape[0] for traj in all_trajectories.values())
    
    for param in avg_trajectories.keys():
        param_data = np.zeros((len(all_trajectories), max_trials))
        for i, (subject_id, trajectories) in enumerate(all_trajectories.items()):
            n_trials = trajectories[param].shape[0]
            param_data[i, :n_trials] = np.average(trajectories[param], axis=1)
            param_data[i, n_trials:] = param_data[i, n_trials-1]  # Pad with last value
        
        avg_trajectories[param] = np.nanmean(param_data, axis=0)
    
    # Plot average learning curves
    for i, (param, label) in enumerate([('theta', 'Learning Ability (θ)'),
                                      ('omega', 'Social Influence (ω)'),
                                      ('r', 'Exploration (r)')]):
        plt.subplot(3, 1, i+1)
        plt.plot(avg_trajectories[param], color=PARAM_COLORS[param], linewidth=2)
        
        # Add confidence intervals
        plt.fill_between(range(max_trials),
                        np.percentile(param_data, 25, axis=0),
                        np.percentile(param_data, 75, axis=0),
                        color=PARAM_COLORS[param],
                        alpha=0.2)
        
        plt.title(f'Average {label} Evolution')
        plt.xlabel('Trial Number')
        plt.ylabel(label)
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'learning_dynamics/average_learning_curves.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Identify learning patterns
    report.append("\nLearning Pattern Analysis:")
    report.append("-" * 30)
    
    # Calculate learning speed (rate of change in theta)
    learning_speeds = []
    for subject_id, trajectories in all_trajectories.items():
        theta_trajectory = np.average(trajectories['theta'], axis=1)
        learning_speed = np.mean(np.diff(theta_trajectory))
        learning_speeds.append({
            'ID': subject_id,
            'learning_speed': learning_speed,
            'solved': results_df[results_df['ID'] == subject_id]['solved'].iloc[0]
        })
    
    learning_speeds_df = pd.DataFrame(learning_speeds)
    
    # Compare learning speeds between successful and unsuccessful participants
    successful = learning_speeds_df[learning_speeds_df['solved'] == 1]['learning_speed']
    unsuccessful = learning_speeds_df[learning_speeds_df['solved'] == 0]['learning_speed']
    
    t_stat, p_val = stats.ttest_ind(successful, unsuccessful)
    report.append("\nLearning Speed Comparison:")
    report.append(f"Successful participants: {successful.mean():.3f} ± {successful.std():.3f}")
    report.append(f"Unsuccessful participants: {unsuccessful.mean():.3f} ± {unsuccessful.std():.3f}")
    report.append(f"T-test: t={t_stat:.3f}, p={p_val:.3f}")
    
    # 3. Analyze strategy adaptation
    plt.figure(figsize=(12, 6))
    
    # Calculate strategy changes
    strategy_changes = []
    for subject_id, trajectories in all_trajectories.items():
        omega_trajectory = np.average(trajectories['omega'], axis=1)
        r_trajectory = np.average(trajectories['r'], axis=1)
        
        # Calculate ratio of exploration to social learning
        strategy_ratio = r_trajectory / omega_trajectory
        strategy_changes.append({
            'ID': subject_id,
            'initial_ratio': strategy_ratio[0],
            'final_ratio': strategy_ratio[-1],
            'solved': results_df[results_df['ID'] == subject_id]['solved'].iloc[0]
        })
    
    strategy_changes_df = pd.DataFrame(strategy_changes)
    
    # Plot strategy adaptation
    plt.scatter(strategy_changes_df[strategy_changes_df['solved'] == 1]['initial_ratio'],
                strategy_changes_df[strategy_changes_df['solved'] == 1]['final_ratio'],
                color='green', alpha=0.6, label='Successful')
    plt.scatter(strategy_changes_df[strategy_changes_df['solved'] == 0]['initial_ratio'],
                strategy_changes_df[strategy_changes_df['solved'] == 0]['final_ratio'],
                color='red', alpha=0.6, label='Unsuccessful')
    
    plt.plot([0, 1], [0, 1], 'k--', alpha=0.5)
    plt.xlabel('Initial Exploration/Social Learning Ratio')
    plt.ylabel('Final Exploration/Social Learning Ratio')
    plt.title('Strategy Adaptation Patterns')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.savefig(os.path.join(output_dir, 'learning_dynamics/strategy_adaptation.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save report
    with open(os.path.join(output_dir, 'reports/learning_dynamics_report.txt'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(report))

def create_parameter_trajectory_plot(subject_data: pd.DataFrame, trajectories: Dict):
    """
    Create parameter trajectory visualization for individual participant
    
    Args:
        subject_data: DataFrame containing participant trial data
        trajectories: Dictionary containing parameter trajectories
    
    Returns:
        matplotlib.figure.Figure: Generated figure object
    """
    plt.figure(figsize=(15, 10))
    
    params = ['theta', 'omega', 'r']
    param_labels = ['Learning Ability (θ)', 'Social Influence (ω)', 'Exploration Rate (r)']
    
    for i, (param, label) in enumerate(zip(params, param_labels)):
        plt.subplot(3, 1, i+1)
        
        # Plot particle trajectories (low alpha)
        for p in range(min(50, trajectories[param].shape[1])):  # Only plot subset of particles for clarity
            plt.plot(trajectories[param][:, p], 
                    alpha=0.1, 
                    color=PARAM_COLORS[param])
        
        # Plot weighted mean trajectory
        weighted_mean = np.average(trajectories[param], 
                                 axis=1, 
                                 weights=trajectories['weights'])
        plt.plot(weighted_mean, 
                color='black', 
                linewidth=2, 
                label='Weighted Mean')
        
        # Add success/failure markers
        successes = subject_data[subject_data['Worked'] == 1].index - subject_data.index[0]
        failures = subject_data[subject_data['Worked'] == 0].index - subject_data.index[0]
        
        plt.scatter(successes, 
                   weighted_mean[successes], 
                   color='green', 
                   alpha=0.5, 
                   label='Success')
        plt.scatter(failures, 
                   weighted_mean[failures], 
                   color='red', 
                   alpha=0.5, 
                   label='Failure')
        
        # Add confidence intervals
        percentile_25 = np.percentile(trajectories[param], 25, axis=1)
        percentile_75 = np.percentile(trajectories[param], 75, axis=1)
        plt.fill_between(range(len(weighted_mean)), 
                        percentile_25, 
                        percentile_75, 
                        color=PARAM_COLORS[param], 
                        alpha=0.2)
        
        plt.title(f'{label} Evolution')
        plt.xlabel('Trial Number')
        plt.ylabel(label)
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    # Add overall information
    completion_status = "Completed" if subject_data['Solved'].iloc[0] else "Incomplete"
    num_unlocked = subject_data['NumUnlock'].iloc[0]
    plt.suptitle(f'Parameter Trajectories for Participant {subject_data["ID"].iloc[0]}\n'
                f'({completion_status}, {num_unlocked}/5 locks)', 
                y=1.02)
    
    plt.tight_layout()
    return plt.gcf()

def fit_parameters_sequential(subject_data: pd.DataFrame) -> Dict:
    """
    Fit parameters using Sequential Monte Carlo (Particle Filter) sampling
    
    Args:
        subject_data: DataFrame containing participant data
    
    Returns:
        Dict: Dictionary containing parameter estimates and trajectories
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
    n_particles = 1000
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
        'trajectories': trajectories
    }
    
    return final_estimates

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Run individual difference analysis')
    parser.add_argument('--plots-only', action='store_true', 
                      help='Only generate plots from saved files')
    
    args = parser.parse_args()
    
    if args.plots_only:
        print("Generating plots from saved files...")
        generate_all_visualizations()
    else:
        print("Running full analysis...")
        analyze_individual_differences(generate_plots=True)