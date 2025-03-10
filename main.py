import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from typing import Dict
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
        
        # Use scaled Pareto for omega with more conservative parameters
        omega_raw = pm.Pareto('omega_raw', alpha=3, m=0.1, initval=0.5)
        omega = pm.Deterministic('omega', omega_raw * 0.5)  # Scale down the effect
        
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

def create_strategy_heatmap(df):
    """
    Create a heatmap showing strategy usage patterns across different trial phases
    
    Args:
        df: DataFrame containing trial data with strategy columns
    """
    # Define trial phases (divide trials into 3 phases for each subject)
    df['Phase'] = df.groupby('ID')['Order'].transform(
        lambda x: pd.qcut(x, q=3, labels=['Early', 'Middle', 'Late'])
    )
    
    # Calculate strategy usage frequency for each phase
    strategy_cols = ['ColorMatch', 'NumMatch', 'ShapeMatch']
    phase_strategy = df.groupby('Phase')[strategy_cols].mean()
    
    # Create heatmap
    plt.figure(figsize=(10, 6))
    sns.heatmap(phase_strategy.T, 
                annot=True, 
                fmt='.2f',
                cmap='YlOrRd',
                center=0.5,
                vmin=0, 
                vmax=1)
    
    plt.title('Strategy Usage Patterns Across Learning Phases')
    plt.xlabel('Learning Phase')
    plt.ylabel('Strategy Type')
    
    plt.tight_layout()
    plt.savefig('./output/strategy_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_strategy_sequence(df, n_subjects=6):
    """
    Create a visualization of strategy transitions for selected subjects
    
    Args:
        df: DataFrame containing trial data
        n_subjects: Number of subjects to display
    """
    # Select subjects with most trials for better visualization
    subject_trials = df.groupby('ID').size()
    selected_subjects = subject_trials.nlargest(n_subjects).index
    
    # Create figure with subplots for each subject
    fig, axes = plt.subplots(n_subjects, 1, figsize=(15, 3*n_subjects))
    fig.suptitle('Strategy Transition Sequences', fontsize=16, y=1.02)
    
    # Color mapping for strategies
    colors = {
        'Color': '#1f77b4',
        'Number': '#2ca02c',
        'Shape': '#d62728',
        'None': '#7f7f7f'
    }
    
    for idx, subject in enumerate(selected_subjects):
        subject_data = df[df['ID'] == subject].copy()
        
        # Determine primary strategy for each trial
        def get_primary_strategy(row):
            if row['ColorMatch']:
                return 'Color'
            elif row['NumMatch']:
                return 'Number'
            elif row['ShapeMatch']:
                return 'Shape'
            return 'None'
        
        subject_data['PrimaryStrategy'] = subject_data.apply(get_primary_strategy, axis=1)
        
        # Create strategy sequence plot
        ax = axes[idx]
        
        # Plot strategy markers
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
        
        # Customize plot
        ax.set_title(f'Subject {subject}')
        ax.set_xlabel('Trial Number')
        ax.set_yticks([])
        ax.set_ylim([0.5, 1.5])
        
        # Add legend only for the first subplot
        if idx == 0:
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    plt.savefig('./output/strategy_sequence.png', dpi=300, bbox_inches='tight')
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
        'r_mean': ('Exploration Rate (r)', '#E53935'),
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
    plt.savefig('./output/parameter_age_evolution.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_parameter_relationships_3d(df):
    """
    Create 3D visualization of parameter relationships
    
    Args:
        df: DataFrame containing analysis results
    """
    fig = plt.figure(figsize=(15, 5))
    
    # First plot: θ, ω, r
    ax1 = fig.add_subplot(121, projection='3d')
    scatter1 = ax1.scatter(df['theta_mean'], 
                          df['omega_mean'], 
                          df['r_mean'],
                          c=df['success_rate'],
                          cmap='viridis')
    ax1.set_xlabel('Learning Ability (θ)')
    ax1.set_ylabel('Social Influence (ω)')
    ax1.set_zlabel('Exploration (r)')
    plt.colorbar(scatter1, label='Success Rate')
    ax1.set_title('Parameter Space Distribution')
    
    # Second plot: Success Rate Surface
    ax2 = fig.add_subplot(122, projection='3d')
    x = np.linspace(df['theta_mean'].min(), df['theta_mean'].max(), 20)
    y = np.linspace(df['omega_mean'].min(), df['omega_mean'].max(), 20)
    X, Y = np.meshgrid(x, y)
    
    # Fit a 2D surface
    from scipy.interpolate import griddata
    Z = griddata((df['theta_mean'], df['omega_mean']), 
                 df['success_rate'], 
                 (X, Y),
                 method='cubic')
    
    surf = ax2.plot_surface(X, Y, Z, cmap='viridis',
                           linewidth=0, antialiased=True)
    ax2.set_xlabel('Learning Ability (θ)')
    ax2.set_ylabel('Social Influence (ω)')
    ax2.set_zlabel('Success Rate')
    plt.colorbar(surf, label='Success Rate')
    ax2.set_title('Success Rate Surface')
    
    plt.tight_layout()
    plt.savefig('./output/parameter_relationships_3d.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_cluster_analysis(df):
    """
    Create visualization of parameter clusters
    
    Args:
        df: DataFrame containing analysis results
    """
    from sklearn.preprocessing import StandardScaler
    from sklearn.cluster import KMeans
    
    # Prepare data for clustering
    features = ['theta_mean', 'omega_mean', 'r_mean', 'motor']
    X = StandardScaler().fit_transform(df[features])
    
    # Perform clustering
    kmeans = KMeans(n_clusters=3, random_state=42)
    df['Cluster'] = kmeans.fit_predict(X)
    
    # Create visualization
    plt.figure(figsize=(15, 5))
    
    # Plot 1: Parameter distribution by cluster
    plt.subplot(121)
    for cluster in range(3):
        cluster_data = df[df['Cluster'] == cluster]
        plt.scatter(cluster_data['theta_mean'], 
                   cluster_data['omega_mean'],
                   alpha=0.6,
                   label=f'Cluster {cluster}')
    
    plt.xlabel('Learning Ability (θ)')
    plt.ylabel('Social Influence (ω)')
    plt.title('Parameter Clusters')
    plt.legend()
    
    # Plot 2: Cluster characteristics
    plt.subplot(122)
    cluster_means = df.groupby('Cluster')[features].mean()
    
    sns.heatmap(cluster_means,
                annot=True,
                fmt='.2f',
                cmap='YlOrRd',
                center=0)
    
    plt.title('Cluster Characteristics')
    
    plt.tight_layout()
    plt.savefig('./output/parameter_clusters.png', dpi=300, bbox_inches='tight')
    plt.close()


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
    plt.savefig('./output/enhanced_strategy_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_gender_parameter_comparison(results_df, trial_data):
    """
    Create visualizations comparing parameter distributions between genders.
    
    Args:
        results_df (pd.DataFrame): DataFrame containing parameter results
        trial_data (pd.DataFrame): DataFrame containing trial data with gender information
    """
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
        
        # Add statistical results to plot with enhanced style
        plt.text(0.05, 0.95, 
                f'p = {p_value:.3f}\nd = {d:.2f}',
                transform=plt.gca().transAxes,
                verticalalignment='top',
                bbox=dict(facecolor='white',
                         alpha=0.9,
                         edgecolor=COLOR_PALETTE['neutral'],
                         boxstyle='round,pad=0.5'))
        
        # Customize plot
        plt.title(f'{label} by Gender', pad=20)
        plt.xticks(positions, ['Boys', 'Girls'])
        plt.ylabel('Value')
        
        # Add grid with custom style
        plt.grid(True, alpha=0.3, linestyle='--')
        
        # Print detailed statistics
        print(f"\nStatistics for {label}:")
        print(f"Boys (n={len(boys_data)}): mean={boys_data.mean():.3f}, std={boys_data.std():.3f}")
        print(f"Girls (n={len(girls_data)}): mean={girls_data.mean():.3f}, std={girls_data.std():.3f}")
        print(f"Mann-Whitney U test: p={p_value:.3f}")
        print(f"Effect size (Cohen's d): {d:.3f}")
    
    plt.tight_layout()
    plt.savefig('./output/gender_parameter_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

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
        'r_mean': 'Exploration Rate (r)',
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
    plt.savefig('./output/parameter_qq_plots.png', dpi=300, bbox_inches='tight')
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
    plt.savefig('./output/learning_influence_network.png', dpi=300, bbox_inches='tight')
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
    plt.savefig('./output/model_evaluation.png', dpi=300, bbox_inches='tight')
    plt.close()

def generate_all_visualizations(results_df=None, all_posteriors=None, load_from_files=False):
    """
    Generate all visualizations
    
    Args:
        results_df: DataFrame containing analysis results, load from file if None
        all_posteriors: Dictionary containing posterior distributions, load from file if None
        load_from_files: Whether to load data from files
    """
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
        trial_data = pd.read_csv('data/dollhouse.csv')
    except FileNotFoundError:
        print("Warning: Trial data not found. Skipping strategy visualizations.")
        trial_data = None
    
    # Generate existing visualizations
    # 1. Parameter distribution plot
    print("Generating parameter distribution plot...")
    fig = plt.figure(figsize=(15, 5))
    gs = plt.GridSpec(1, 3, wspace=0.3)
    
    def fit_beta_mixture(data, n_components=2):
        logit_data = np.log(data / (1 - data))
        gmm = GaussianMixture(n_components=n_components, random_state=42)
        gmm.fit(logit_data.reshape(-1, 1))
        
        def mixture_pdf(x):
            total = np.zeros_like(x)
            for i in range(n_components):
                weight = gmm.weights_[i]
                mu, sigma = gmm.means_[i][0], np.sqrt(gmm.covariances_[i][0][0])
                component = weight * stats.norm.pdf(np.log(x/(1-x)), mu, sigma) / (x * (1-x))
                total += component
            return total
        return mixture_pdf
    
    params = {
        'theta': ('Learning Ability (θ)', 'blue'),
        'omega': ('Social Influence (ω)', 'green'),
        'r': ('Exploration (r)', 'red')
    }
    
    for idx, (param, (label, color)) in enumerate(params.items()):
        ax = plt.subplot(gs[idx])
        for subject_id in all_posteriors:
            sns.kdeplot(all_posteriors[subject_id][param], alpha=0.05, color=color)
        
        sns.histplot(data=results_df, x=f'{param}_mean', bins=20, alpha=0.3, 
                    color=color, stat='density')
        
        mixture = fit_beta_mixture(results_df[f'{param}_mean'].values)
        x = np.linspace(0.001, 0.999, 200)
        y = mixture(x)
        plt.plot(x, y, 'k--', alpha=0.7)
        
        plt.title(f'Distribution of {label}')
        plt.xlabel(label)
        plt.ylabel('Density')
        if idx == 0:
            plt.legend(['Individual', 'Population', 'Mixture Model'])
    
    plt.savefig('./output/parameter_distributions.png', dpi=300, bbox_inches='tight')
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
    plt.savefig('./output/correlation_heatmap.png', dpi=300, bbox_inches='tight')
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
    
    plt.savefig('./output/parameter_violin_plots.png', dpi=300, bbox_inches='tight')
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
    
    print("All visualizations completed!")


def analyze_individual_differences(generate_plots=True):
    """Analyze individual differences and strategy combinations"""
    # Load data
    df = pd.read_csv('data/dollhouse.csv')
    
    # Store results
    all_results = []
    all_posteriors = {}
    
    # Process data for each subject
    for subject_id in df['ID'].unique():
        subject_data = df[df['ID'] == subject_id]
        
        print(f"\nProcessing subject {subject_id}")
        print("-" * 50)
        
        # Fit parameters using Bayesian MCMC
        results = fit_parameters_bayesian(subject_data)
        
        # Store complete posterior distributions
        all_posteriors[subject_id] = {
            'theta': results['theta'].flatten(),
            'omega': results['omega'].flatten(),
            'r': results['r'].flatten()
        }
        
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
            'total_trials': len(subject_data),
            'success_rate': subject_data['Worked'].mean(),
            'color_match_rate': subject_data['ColorMatch'].mean(),
            'num_match_rate': subject_data['NumMatch'].mean(),
            'shape_match_rate': subject_data['ShapeMatch'].mean()
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
    
    # Generate visualizations if needed
    if generate_plots:
        generate_all_visualizations(results_df, all_posteriors)
    
    return results_df, all_posteriors

if __name__ == "__main__":
    import argparse
    import pickle
    
    parser = argparse.ArgumentParser(description='Run individual difference analysis')
    parser.add_argument('--plots-only', action='store_true', 
                      help='Only generate plots from saved files')
    
    args = parser.parse_args()
    
    if args.plots_only:
        generate_all_visualizations(load_from_files=True)
    else:
        analyze_individual_differences(generate_plots=True)