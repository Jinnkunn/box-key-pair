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

# Create output directory if it doesn't exist
os.makedirs('./output', exist_ok=True)

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
    """Calculate success probability based on learning ability, motor skill, and strategy weights"""
    
    eps = 1e-7  # Small epsilon for numerical stability
    
    theta = pm.math.clip(theta, eps, 1.0 - eps)
    motor_skill = pm.math.clip(motor_skill, eps, 1.0 - eps)
    r = pm.math.clip(r, eps, 1.0 - eps)
    omega = pm.math.clip(omega, 1.0 + eps, 10.0) 
    
    base_prob = theta * motor_skill
    base_prob = pm.math.clip(base_prob, eps, 1.0 - eps)
    
    social_influence = omega * base_prob
    heuristic_influence = r * (1.0 - base_prob)
    
    final_prob = pm.math.clip(base_prob + social_influence + heuristic_influence, eps, 1.0 - eps)
    
    return final_prob

def fit_parameters_bayesian(subject_data: pd.DataFrame, n_samples: int = 4000) -> Dict:
    """
    Fit parameters using Bayesian MCMC sampling
    
    Args:
        subject_data: DataFrame containing participant data
        n_samples: Number of MCMC samples
    
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
    
    with pm.Model() as model:
        theta_i = pm.Beta('theta_i', alpha=2, beta=2, shape=n_hypotheses)
        hypothesis_weights = pm.Dirichlet('hypothesis_weights', a=np.ones(n_hypotheses))
        theta = pm.Deterministic('theta', pm.math.dot(theta_i, hypothesis_weights))
        r = pm.Uniform('r', lower=0, upper=1, initval=0.5)
        
        # Using Pareto distribution for omega
        omega = pm.Pareto('omega', 
                     alpha=3.0,  # Shape parameter
                     m=1.0,      # Minimum value
                     initval=1.5)
        
        p = pm.Deterministic('p', calculate_success_probability(theta, motor, omega, r))
        
        # Count successes and failures for each hypothesis
        for i in range(n_hypotheses):
            if i == 0:  # Color hypothesis
                successes = subject_data.loc[subject_data['ColorMatch'] == 1, 'Worked'].sum()
                trials = len(subject_data[subject_data['ColorMatch'] == 1])
            elif i == 1:  # Number hypothesis
                successes = subject_data.loc[subject_data['NumMatch'] == 1, 'Worked'].sum()
                trials = len(subject_data[subject_data['NumMatch'] == 1])
            else:  # Shape hypothesis
                successes = subject_data.loc[subject_data['ShapeMatch'] == 1, 'Worked'].sum()
                trials = len(subject_data[subject_data['ShapeMatch'] == 1])
            
            if trials > 0:
                pm.Binomial(f'obs_{i}', n=trials, p=theta_i[i], observed=successes)
        
        # Overall success probability
        y = pm.Bernoulli('y', p=p, observed=subject_data['Worked'])
        
        # Debug information
        print("\nModel Debug Information:")
        print("-" * 50)
        print(f"Participant ID: {subject_data['ID'].iloc[0]}")
        print(f"Completion status: {'Completed' if solved else 'Incomplete'} ({num_unlock}/5)")
        print(f"Completion time: {unlock_time:.2f} seconds")
        print(f"Basic statistics:")
        print(f"  Total attempts: {total}")
        print(f"  Success rate: {success_rate:.4f}")
        print(f"  Error rate: {errors/total:.4f}")
        
        # Use improved sampling settings
        trace = pm.sample(
            n_samples,
            return_inferencedata=True,
            cores=4,
            init='jitter+adapt_diag',
            random_seed=42,
            progressbar=True,
            target_accept=0.95,
            tune=1000,
            chains=4,
        )
    
    posterior = trace.posterior
    
    return {
        'theta': posterior['theta'].values,
        'theta_i': posterior['theta_i'].values,
        'hypothesis_weights': posterior['hypothesis_weights'].values,
        'omega': posterior['omega'].values,
        'r': posterior['r'].values,
        'motor': motor,
        'completion_info': {
            'solved': solved,
            'num_unlock': num_unlock,
            'unlock_time': unlock_time,
        }
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
    plt.savefig('./output/bayesian_comparison_results.png')
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
        results_df = pd.read_csv('./output/individual_results.csv')
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
    plt.savefig('./output/strategy_analysis/strategy_heatmap.png', dpi=300, bbox_inches='tight')
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
    plt.savefig('./output/strategy_analysis/strategy_sequence.png', dpi=300, bbox_inches='tight')
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
    plt.savefig('./output/parameter_analysis/parameter_age_evolution.png', dpi=300, bbox_inches='tight')
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
    plt.savefig('./output/parameter_analysis/parameter_relationships_3d.png', dpi=300, bbox_inches='tight')
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
    plt.savefig('./output/cluster/cluster_size_completion.png', dpi=300, bbox_inches='tight')
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
    plt.savefig('./output/cluster/success_vs_completion.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Figure 3: Number of Unlocks by Cluster
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df, x='Cluster', y='num_unlock', palette='viridis')
    plt.title('Number of Unlocks by Cluster', fontsize=14, pad=20)
    plt.xlabel('Cluster', fontsize=12)
    plt.ylabel('Number of Unlocks', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('./output/cluster/unlocks_by_cluster.png', dpi=300, bbox_inches='tight')
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
    plt.savefig('./output/cluster/cluster_characteristics.png', dpi=300, bbox_inches='tight')
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
    plt.savefig('./output/cluster/parameter_distribution.png', dpi=300, bbox_inches='tight')
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
    
    plt.savefig('./output/cluster/parameter_clusters.png', 
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
    cluster_data.to_csv('./output/cluster/cluster_assignments.csv', index=False)
    
    with open('./output/cluster_analysis_report.txt', 'w') as f:
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
    
    cluster_stats.to_csv('./output/cluster/cluster_statistics.csv')

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
    
    # 3. Correlation Analysis with Statistical Tests
    report.append("\nCorrelation Analysis")
    report.append("=" * 50)
    
    # Create correlation matrix
    correlations = results_df[list(params.keys()) + ['success_rate']].corr()
    
    for param1 in params:
        for param2 in params:
            if param1 < param2:  # Only report upper triangle matrix
                corr = correlations.loc[param1, param2]
                # Perform correlation test
                r, p = stats.pearsonr(results_df[param1], results_df[param2])
                
                if abs(corr) > 0.3:  # Only report significant correlations
                    report.append(f"\n{params[param1]} and {params[param2]}:")
                    report.append(f"Correlation coefficient: r={r:.3f}")
                    report.append(f"Statistical significance: p={p:.3f}")
                    report.append(f"Effect size interpretation: {get_correlation_interpretation(r)}")
                    
                    # Add regression analysis if correlation is significant
                    if p < 0.05:
                        slope, intercept, r_value, p_value, std_err = stats.linregress(
                            results_df[param1], results_df[param2]
                        )
                        report.append(f"Regression analysis:")
                        report.append(f"  Slope: {slope:.3f} ± {std_err:.3f}")
                        report.append(f"  R-squared: {r_value**2:.3f}")
    
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
        
        # Correlation with success rate
        corr, p = stats.pearsonr(results_df[strategy], results_df['success_rate'])
        report.append(f"Correlation with success rate: r={corr:.3f}, p={p:.3f}")
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
    with open('./output/analysis_report.txt', 'w', encoding='utf-8') as f:
        f.write('\n'.join(report))
    
    print("Analysis report has been saved to './output/analysis_report.txt'")
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
    plt.savefig('./output/strategy_analysis/enhanced_strategy_analysis.png', dpi=300, bbox_inches='tight')
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
    plt.savefig('./output/gender_analysis/gender_parameter_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save the report
    with open('./output/gender_parameter_comparison_report.txt', 'w', encoding='utf-8') as f:
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
    plt.savefig('./output/distribution_analysis/parameter_qq_plots.png', dpi=300, bbox_inches='tight')
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
    plt.savefig('./output/correlation_analysis/learning_influence_network.png', dpi=300, bbox_inches='tight')
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
    plt.savefig('./output/model_evaluation/model_evaluation.png', dpi=300, bbox_inches='tight')
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
    plt.savefig('./output/performance_analysis/completion_analysis.png', dpi=300, bbox_inches='tight')
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
    with open('./output/completion_analysis_report.txt', 'w', encoding='utf-8') as f:
        f.write('\n'.join(report))

def create_output_directories():
    """Create all necessary output directories"""
    directories = [
        './output/model_evaluation',
        './output/parameter_analysis',
        './output/gender_analysis',
        './output/age_analysis',
        './output/strategy_analysis',
        './output/performance_analysis',
        './output/correlation_analysis',
        './output/distribution_analysis',
        './output/cluster'
    ]
    for directory in directories:
        os.makedirs(directory, exist_ok=True)

def generate_all_visualizations(results_df=None, all_posteriors=None, load_from_files=False):
    """
    Generate all visualizations
    
    Args:
        results_df: DataFrame containing analysis results, load from file if None
        all_posteriors: Dictionary containing posterior distributions, load from file if None
        load_from_files: Whether to load data from files
    """
    # Create output directories
    create_output_directories()
    
    if load_from_files:
        print("Loading data from files...")
        try:
            results_df = pd.read_csv('./output/individual_results.csv')
            with open('./output/posterior_distributions.pkl', 'rb') as f:
                all_posteriors = pickle.load(f)
        except FileNotFoundError:
            print("Error: Required data files not found. Please run full analysis first.")
            return
    
    if results_df is None or all_posteriors is None:
        print("Error: Data must be provided or loaded from files.")
        return

    print("Starting visualization generation...")
    
    # Generate analysis report
    print("Generating analysis report...")
    generate_analysis_report(results_df, all_posteriors)
    
    # Load trial data for new visualizations
    try:
        merged_data, _ = load_and_merge_data()
        trial_data = merged_data
    except FileNotFoundError:
        print("Warning: Trial data not found. Skipping strategy visualizations.")
        trial_data = None
    
    # Generate parameter distribution plot
    print("Generating parameter distribution plot...")
    fig = plt.figure(figsize=(15, 5))
    gs = plt.GridSpec(1, 3, wspace=0.3)
    
    def fit_beta_mixture(data, n_components=2):
        data = data[~np.isnan(data)]
        data = np.clip(data, 0.001, 0.999)
        logit_data = np.log(data / (1 - data))
        logit_data = logit_data.reshape(-1, 1)
        
        gmm = GaussianMixture(n_components=n_components, random_state=42)
        gmm.fit(logit_data)
        
        def mixture_pdf(x):
            x = np.clip(x, 0.001, 0.999)
            total = np.zeros_like(x)
            for i in range(n_components):
                weight = gmm.weights_[i]
                mu, sigma = gmm.means_[i][0], np.sqrt(gmm.covariances_[i][0][0])
                component = weight * stats.norm.pdf(np.log(x/(1-x)), mu, sigma) / (x * (1-x))
                total += component
            return total
        
        return mixture_pdf
    
    def fit_gaussian_mixture(data, n_components=2):
        data = data[~np.isnan(data)]
        data = data.reshape(-1, 1)
        
        gmm = GaussianMixture(n_components=2, 
                             random_state=42, 
                             covariance_type='full',
                             reg_covar=1e-2)
        gmm.fit(data)
        
        def mixture_pdf(x):
            total = np.zeros_like(x)
            for i in range(gmm.n_components):
                weight = gmm.weights_[i]
                mu, sigma = gmm.means_[i][0], np.sqrt(gmm.covariances_[i][0][0])
                component = weight * stats.norm.pdf(x, mu, sigma)
                total += component
            return total
        
        return mixture_pdf
    
    params = {
        'theta_mean': ('Learning Ability (θ)', 'blue'),
        'omega_mean': ('Social Influence (ω)', 'green'),
        'r_mean': ('Exploration (r)', 'red')
    }
    
    for idx, (param, (label, color)) in enumerate(params.items()):
        ax = plt.subplot(gs[idx])
        
        # Plot individual distributions
        for subject_id in all_posteriors:
            if param == 'omega_mean':
                data = all_posteriors[subject_id][param]
                sns.kdeplot(data, alpha=0.03, color=color, label='_nolegend_', 
                           bw_adjust=0.4)
            else:
                sns.kdeplot(all_posteriors[subject_id][param], alpha=0.05, color=color, label='_nolegend_')
        
        # Plot one individual distribution for legend
        first_subject = list(all_posteriors.keys())[0]
        if param == 'omega_mean':
            data = all_posteriors[first_subject][param]
            ind_dist = sns.kdeplot(data, alpha=0.5, color=color, 
                        label='Individual Distributions', linewidth=2, 
                        bw_adjust=0.4)
        else:
            ind_dist = sns.kdeplot(all_posteriors[first_subject][param], alpha=0.5, color=color, 
                        label='Individual Distributions', linewidth=2)
        
        # Plot population histogram
        if param == 'omega_mean':
            data = results_df[param]
            
            # 使用更小的 bin 宽度
            iqr = np.percentile(data, 75) - np.percentile(data, 25)
            bin_width = iqr / (len(data) ** (1/3))  # 减小 bin 宽度
            bins = max(int((data.max() - data.min()) / bin_width), 20)  # 增加最小 bin 数
            
            hist = sns.histplot(data=results_df, x=param, bins=bins, alpha=0.3, 
                              color=color, stat='density', label='Population Distribution')
        else:
            hist = sns.histplot(data=results_df, x=param, bins=20, alpha=0.3, 
                              color=color, stat='density', label='Population Distribution')
        
        if param == 'omega_mean':
            data = results_df[param].values
            m = min(data) 
            alpha = len(data) / sum(np.log(data/m))
            
            x = np.linspace(m, max(data), 200)
            y = alpha * (m**alpha) / (x**(alpha + 1))  # Pareto PDF
            
            mix_line, = plt.plot(x, y, 'k--', 
                                alpha=0.7, 
                                label='Pareto Fit', 
                                linewidth=2)
            
            plt.xlim(0.9, max(data) * 1.1) 
            plt.ylim(0, max(y) * 1.2)
        else:
            mixture = fit_beta_mixture(results_df[param].values)
            x = np.linspace(0.001, 0.999, 200)
            y = mixture(x)
            mix_line, = plt.plot(x, y, 'k--', alpha=0.7, label='Mixture Model Fit', linewidth=2)
        
        plt.title(f'Distribution of {label}')
        plt.xlabel(label)
        plt.ylabel('Density')
        
        # Adjust y-axis limits based on the parameter
        if param == 'theta_mean':
            plt.ylim(0, 25)
        elif param == 'r_mean':
            plt.ylim(0, 40)
        
        plt.grid(True, alpha=0.3)
        plt.legend()
    
    plt.tight_layout()
    plt.savefig('./output/distribution_analysis/parameter_distributions.png', 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Correlation heatmap
    print("Generating correlation heatmap...")
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
    plt.savefig('./output/correlation_analysis/correlation_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Violin plots
    print("Generating violin plots...")
    plt.figure(figsize=(12, 6))
    
    param_names = ['theta_mean', 'omega_mean', 'r_mean', 'motor']
    param_labels = ['Learning\nAbility (θ)', 'Social\nInfluence (ω)', 
                   'Exploration (r)', 'Motor\nSkill (M)']
    violin_data = [results_df[param].values for param in param_names]
    
    violin = plt.violinplot(violin_data, points=100, vert=True, widths=0.7,
                          showmeans=True, showextrema=True, showmedians=True)
    
    colors = ['#1976D2', '#43A047', '#E53935', '#FDD835']
    for i, pc in enumerate(violin['bodies']):
        pc.set_facecolor(colors[i])
        pc.set_alpha(0.7)
    
    violin['cmeans'].set_color('black')
    violin['cmedians'].set_color('white')
    
    for idx, param in enumerate(param_names):
        z_scores = np.abs(stats.zscore(results_df[param]))
        outliers = z_scores > 2.5
        
        x_jitter = np.random.normal(idx + 1, 0.02, size=len(results_df))
        plt.scatter(x_jitter, results_df[param], alpha=0.2, color='black', s=15)
        
        for i, (is_outlier, x, y) in enumerate(zip(outliers, x_jitter, results_df[param])):
            if is_outlier and z_scores[i] > 3:
                plt.annotate(results_df['ID'].iloc[i], 
                           (x, y),
                           xytext=(5, 5),
                           textcoords='offset points',
                           fontsize=8,
                           alpha=0.7)
    
    plt.xticks(range(1, len(param_names) + 1), param_labels)
    plt.ylabel('Parameter Value')
    plt.title('Distribution of Individual Parameters')
    plt.grid(True, axis='y', linestyle='--', alpha=0.3)
    
    plt.savefig('./output/distribution_analysis/parameter_violin_plots.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. Learning influence network
    print("Generating learning influence network...")
    create_learning_influence_network(results_df)
    
    # 5. Model evaluation plot
    print("Generating model evaluation plot...")
    create_model_evaluation_plot(results_df)
    
    # Generate strategy visualizations if trial data is available
    if trial_data is not None:
        print("Generating strategy heatmap...")
        create_strategy_heatmap(trial_data)
        
        print("Generating strategy sequence plot...")
        create_strategy_sequence(trial_data)
    
    # Add new visualizations
    print("Generating parameter evolution plot...")
    create_parameter_evolution(results_df)
    
    print("Generating 3D parameter relationships...")
    create_parameter_relationships_3d(results_df)
    
    print("Generating cluster analysis...")
    create_cluster_analysis(results_df)
    
    # Add new statistical visualizations
    print("Generating enhanced strategy analysis...")
    create_enhanced_strategy_analysis(trial_data)
    
    print("Generating enhanced parameter distribution analysis...")
    create_enhanced_parameter_distribution(results_df)
    
    print("Generating gender parameter comparison...")
    create_gender_parameter_comparison(results_df, trial_data)
    
    print("Generating completion analysis...")
    create_completion_analysis(results_df)
    
    print("All visualizations completed!")


def analyze_individual_differences(generate_plots=True):
    """Analyze individual differences and strategy combinations"""
    # Load data
    merged_data, summary_data = load_and_merge_data()
    
    # Store results
    all_results = []
    all_posteriors = {}
    
    # Process data for each participant
    for subject_id in merged_data['ID'].unique():
        subject_data = merged_data[merged_data['ID'] == subject_id]
        
        print(f"\nProcessing participant {subject_id}")
        print("-" * 50)
        
        # Fit parameters using Bayesian MCMC
        results = fit_parameters_bayesian(subject_data)
        
        # Store complete posterior distributions
        all_posteriors[subject_id] = {
            'theta_mean': results['theta'].flatten(),
            'omega_mean': results['omega'].flatten(),
            'r_mean': results['r'].flatten()
        }
        
        # Get completion information
        completion_info = results['completion_info']
        
        # Store summary statistics
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
    results_df.to_csv('./output/individual_results.csv', index=False)
    with open('./output/posterior_distributions.pkl', 'wb') as f:
        pickle.dump(all_posteriors, f)
    
    print("\nIndividual results have been saved to './output/individual_results.csv'")
    print("Posterior distributions have been saved to './output/posterior_distributions.pkl'")
    
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
    
    return results_df, all_posteriors

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Run individual difference analysis')
    parser.add_argument('--plots-only', action='store_true', 
                      help='Only generate plots from saved files')
    
    args = parser.parse_args()
    
    if args.plots_only:
        generate_all_visualizations(load_from_files=True)
    else:
        analyze_individual_differences(generate_plots=True)