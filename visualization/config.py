"""
Global visualization configuration settings.
"""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Color settings for different parameters
PARAM_COLORS = {
    'theta': '#2E86C1',    # Blue
    'omega': '#28B463',    # Green
    'r': '#E74C3C',        # Red
    'motor': '#8E44AD',    # Purple
    'success': '#2ECC71',  # Green
    'error': '#E74C3C'     # Red
}

# Color palette for different purposes
COLOR_PALETTE = {
    'boys': '#3498DB',       # Light blue
    'girls': '#E91E63',      # Pink
    'neutral': '#95A5A6',    # Gray
    'success': '#2ECC71',    # Green
    'failure': '#E74C3C',    # Red
    'highlight': '#F39C12',  # Orange
    'accent': '#9B59B6',     # Purple
    'background': '#ECF0F1', # Light gray
    'cluster_0': '#2E86C1',  # Blue for cluster 0
    'cluster_1': '#28B463',  # Green for cluster 1
    'cluster_2': '#E74C3C',  # Red for cluster 2
    'primary': '#3498DB',    # Primary color (blue)
    'secondary': '#2ECC71',  # Secondary color (green)
    'tertiary': '#F39C12',   # Tertiary color (orange)
    'categorical': ['#3498DB', '#2ECC71', '#E74C3C', '#9B59B6', '#F39C12']  # Colors for categorical data
}

# Figure sizes for different plot types
FIGURE_SIZES = {
    'single': (8, 6),      # Standard single plot
    'wide': (12, 6),       # Wide format for multiple plots
    'tall': (8, 10),       # Tall format for vertical plots
    'square': (8, 8),      # Square format for correlation plots
    'large': (15, 10),     # Large format for complex plots
    'small': (6, 4)        # Small format for simple plots
}

# Alpha values for different plot elements
ALPHA_VALUES = {
    'main_plot': 0.8,      # Main plot elements
    'background': 0.2,     # Background elements
    'grid': 0.3,          # Grid lines
    'scatter': 0.6,       # Scatter plot points
    'swarm_plot': 0.7,    # Swarm plot points
    'violin_plot': 0.8,   # Violin plot bodies
    'error_bars': 0.5,    # Error bars
    'overlay': 0.4        # Overlay elements
}

# Font sizes for different text elements
FONT_SIZES = {
    'title': 16,           # Plot titles
    'subtitle': 14,        # Subplot titles
    'axis_label': 12,      # Axis labels
    'tick_label': 10,      # Tick labels
    'legend': 10,          # Legend text
    'annotation': 8,       # Annotations
    'caption': 9          # Figure captions
}

# Line styles for different purposes
LINE_STYLES = {
    'main': '-',          # Main plot lines
    'reference': '--',    # Reference/fit lines
    'grid': ':',         # Grid lines
    'threshold': '-.',    # Threshold lines
    'error': '-',        # Error bars
    'trend': '--'        # Trend lines
}

# Marker styles for different categories
MARKER_STYLES = {
    'data_point': 'o',    # Regular data points
    'average': 's',       # Mean/average markers
    'outlier': 'x',       # Outlier points
    'special': '*',       # Special points
    'reference': '^'      # Reference points
}

# Plot style settings
PLOT_STYLE = {
    'axes.facecolor': COLOR_PALETTE['background'],
    'figure.facecolor': 'white',
    'grid.color': COLOR_PALETTE['neutral'],
    'grid.linestyle': LINE_STYLES['grid'],
    'grid.alpha': ALPHA_VALUES['grid'],
    'axes.grid': True,
    'axes.edgecolor': COLOR_PALETTE['neutral'],
    'axes.linewidth': 1.0,
    'axes.titlesize': FONT_SIZES['title'],
    'axes.labelsize': FONT_SIZES['axis_label'],
    'xtick.labelsize': FONT_SIZES['tick_label'],
    'ytick.labelsize': FONT_SIZES['tick_label'],
    'legend.fontsize': FONT_SIZES['legend'],
    'figure.titlesize': FONT_SIZES['title']
}

# Default settings for various plot elements
PLOT_DEFAULTS = {
    'grid': True,
    'grid_alpha': ALPHA_VALUES['grid'],
    'grid_style': LINE_STYLES['grid'],
    'legend_loc': 'best',
    'dpi': 300,
    'bbox_inches': 'tight'
}

# Seaborn specific settings
SEABORN_STYLE = {
    'style': 'whitegrid',
    'context': 'notebook',
    'palette': 'deep'
}

def setup_plot_style():
    """Set up the global plotting style."""
    # Set matplotlib style
    plt.style.use('default')  # Use default style instead of seaborn
    plt.rcParams.update(PLOT_STYLE)
    
    # Set seaborn style
    sns.set_style(**{'style': SEABORN_STYLE['style']})
    sns.set_context(SEABORN_STYLE['context'])
    sns.set_palette(SEABORN_STYLE['palette'])

# Additional color maps for specific visualizations
CUSTOM_COLORMAPS = {
    'correlation': 'coolwarm',
    'heatmap': 'YlOrRd',
    'sequence': 'viridis',
    'diverging': 'RdYlBu',
    'continuous': 'plasma',
    'cluster': 'viridis',
    'sequential': 'viridis'  # Sequential colormap for ordered data
}

# Statistical visualization settings
STAT_SETTINGS = {
    'ci': 95,              # Confidence interval
    'n_bootstrap': 1000,   # Number of bootstrap samples
    'kernel_bandwidth': 0.2  # KDE bandwidth
}

# 3D plot settings
PLOT_3D_SETTINGS = {
    'view_angle': (30, 45),  # Default view angle (elevation, azimuth)
    'surface_alpha': 0.7,    # Surface plot transparency
    'edge_alpha': 0.2,      # Edge transparency
    'cmap': 'viridis'       # Default colormap
}

def preprocess_trajectories(trajectories, param):
    """
    Preprocess trajectory data with unified method
    
    Args:
        trajectories: Dictionary of parameter trajectories
        param: Parameter name to process
    
    Returns:
        numpy.ndarray: Processed trajectory data
    """
    max_trials = max(traj[param].shape[0] for traj in trajectories.values())
    processed_data = np.zeros((len(trajectories), max_trials))
    
    for i, (subject_id, traj) in enumerate(trajectories.items()):
        n_trials = traj[param].shape[0]
        processed_data[i, :n_trials] = np.average(traj[param], axis=1)
        processed_data[i, n_trials:] = processed_data[i, n_trials-1]
    
    return processed_data

def calculate_average_curve(data, method='mean'):
    """
    Calculate average curve with unified method
    
    Args:
        data: Array of trajectory data
        method: Aggregation method ('mean' or 'median')
    
    Returns:
        numpy.ndarray: Average curve
    """
    if method == 'mean':
        return np.nanmean(data, axis=0)
    elif method == 'median':
        return np.nanmedian(data, axis=0)
    else:
        raise ValueError(f"Unsupported aggregation method: {method}")

def calculate_confidence_intervals(data, lower=25, upper=75):
    """
    Calculate confidence intervals with unified method
    
    Args:
        data: Array of trajectory data
        lower: Lower percentile
        upper: Upper percentile
    
    Returns:
        tuple: (lower_bound, upper_bound)
    """
    return np.percentile(data, [lower, upper], axis=0) 