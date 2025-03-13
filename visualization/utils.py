"""
Utility functions for visualization.
"""

import numpy as np
from typing import Tuple, List, Union, Dict, Optional
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from scipy import stats
import seaborn as sns
from .config import (
    PLOT_STYLE,
    SEABORN_STYLE,
    FONT_SIZES,
    COLOR_PALETTE,
    ALPHA_VALUES,
    LINE_STYLES,
    PLOT_DEFAULTS
)
import pandas as pd

def calculate_confidence_interval(
    data: np.ndarray,
    confidence: float = 0.95
) -> Tuple[float, float]:
    """
    Calculate confidence interval for the given data.
    
    Args:
        data: Input data array
        confidence: Confidence level (default: 0.95)
    
    Returns:
        Tuple[float, float]: Lower and upper bounds of the confidence interval
    """
    mean = np.mean(data)
    std_err = np.std(data, ddof=1) / np.sqrt(len(data))
    z_score = stats.norm.ppf((1 + confidence) / 2)
    margin = z_score * std_err
    return mean - margin, mean + margin

def format_pvalue(pvalue: float) -> str:
    """
    Format p-value for display in plots.
    
    Args:
        pvalue: P-value to format
    
    Returns:
        str: Formatted p-value string
    """
    if pvalue < 0.001:
        return "p < 0.001"
    elif pvalue < 0.01:
        return f"p < 0.01"
    elif pvalue < 0.05:
        return f"p < 0.05"
    else:
        return f"p = {pvalue:.3f}"

def add_significance_bars(
    ax,
    x1: Union[int, float],
    x2: Union[int, float],
    y: Union[int, float],
    height: float,
    pvalue: float
) -> None:
    """
    Add significance bars to plots.
    
    Args:
        ax: Matplotlib axis object
        x1: Start x position
        x2: End x position
        y: Y position
        height: Height of the significance bar
        pvalue: P-value to display
    """
    bar_height = height
    bar_tips = height * 0.05
    
    # Draw the bar
    ax.plot([x1, x1, x2, x2], [y, y + bar_height, y + bar_height, y], 'k-', linewidth=1)
    
    # Add p-value text
    ax.text((x1 + x2) * 0.5, y + bar_height, format_pvalue(pvalue),
            ha='center', va='bottom')

def apply_style_to_axis(ax: Axes, title: str = None, xlabel: str = None, ylabel: str = None,
                       legend: bool = True, grid: bool = True) -> None:
    """Apply unified style to a matplotlib axis."""
    if title:
        ax.set_title(title, fontsize=FONT_SIZES['title'], pad=20)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=FONT_SIZES['axis_label'])
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=FONT_SIZES['axis_label'])
    
    ax.tick_params(labelsize=FONT_SIZES['tick_label'])
    
    if grid:
        ax.grid(True, alpha=PLOT_DEFAULTS['grid_alpha'], linestyle=PLOT_DEFAULTS['grid_style'])
    
    if legend:
        handles, labels = ax.get_legend_handles_labels()
        if handles and labels:
            ax.legend(fontsize=FONT_SIZES['legend'])

def create_figure(figsize: str = 'single') -> Tuple[Figure, Axes]:
    """Create a figure with the specified size from FIGURE_SIZES."""
    from .config import FIGURE_SIZES
    fig = plt.figure(figsize=FIGURE_SIZES[figsize])
    ax = fig.add_subplot(111)
    return fig, ax

def setup_3d_axis(ax: Axes, title: str = None, xlabel: str = None, ylabel: str = None,
                  zlabel: str = None, elev: float = 30, azim: float = 45) -> None:
    """Setup a 3D axis with unified style."""
    from .config import PLOT_3D_SETTINGS
    
    if title:
        ax.set_title(title, fontsize=FONT_SIZES['title'], pad=20)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=FONT_SIZES['axis_label'])
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=FONT_SIZES['axis_label'])
    if zlabel:
        ax.set_zlabel(zlabel, fontsize=FONT_SIZES['axis_label'])
    
    ax.tick_params(labelsize=FONT_SIZES['tick_label'])
    ax.view_init(elev=elev, azim=azim)
    
    # Set background color
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False

def add_statistical_annotations(ax, stats_dict, loc='upper right', position=None):
    """Add statistical annotations to a plot."""
    # Format the statistics text
    if isinstance(stats_dict, dict):
        # Handle dictionary input
        stats_text = format_statistical_result_dict(stats_dict)
    else:
        # Handle tuple input (stat, p_val, msg)
        stat, p_val, msg = stats_dict
        stats_text = msg if msg else format_statistical_result(stat, p_val)
    
    # Set up text properties
    bbox_props = dict(
        boxstyle='round,pad=0.5',
        facecolor='white',
        alpha=0.8,
        edgecolor='gray'
    )
    
    # Check if the axis is 3D
    is_3d = hasattr(ax, 'get_zlim')
    
    if position is not None:
        # For annotations at specific positions
        x, y = position
        ax.text(x, y, stats_text,
               horizontalalignment='center',
               verticalalignment='bottom',
               fontsize=FONT_SIZES['annotation'],
               bbox=bbox_props)
    elif is_3d:
        # For 3D plots
        x_pos = 0.05 if loc == 'upper left' else 0.95
        y_pos = 0.95
        z_pos = ax.get_zlim()[1]  # Get the maximum z value
        ax.text(x_pos, y_pos, z_pos, stats_text,
               horizontalalignment='left' if loc == 'upper left' else 'right',
               verticalalignment='top',
               transform=ax.transAxes,
               fontsize=FONT_SIZES['annotation'],
               bbox=bbox_props)
    else:
        # For 2D plots
        ax.text(0.05 if loc == 'upper left' else 0.95,
               0.95,
               stats_text,
               horizontalalignment='left' if loc == 'upper left' else 'right',
               verticalalignment='top',
               transform=ax.transAxes,
               fontsize=FONT_SIZES['annotation'],
               bbox=bbox_props)

def format_statistical_result_dict(stats_dict: Dict) -> str:
    """Format statistical results dictionary into a string."""
    # Format each value based on its type
    formatted_items = []
    for k, v in stats_dict.items():
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            formatted_items.append(f'{k}: {v:.3f}')
        else:
            formatted_items.append(f'{k}: {v}')
    
    return '\n'.join(formatted_items)

def format_statistical_result(stat: float, p_val: float) -> str:
    """Format statistical test results."""
    if np.isnan(stat) or np.isnan(p_val):
        return "Invalid result"
        
    significance = ""
    if p_val < 0.001:
        significance = "***"
    elif p_val < 0.01:
        significance = "**"
    elif p_val < 0.05:
        significance = "*"
        
    return f"stat={stat:.3f}, p={p_val:.3f}{significance}"

def save_figure(fig: Figure, output_path: str, **kwargs) -> None:
    """Save figure with default settings."""
    fig.savefig(output_path,
                dpi=PLOT_DEFAULTS['dpi'],
                bbox_inches=PLOT_DEFAULTS['bbox_inches'],
                **kwargs)
    plt.close(fig)

def set_axis_style(ax: Axes, style: str = 'default') -> None:
    """Apply predefined styles to an axis."""
    if style == 'clean':
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    elif style == 'minimal':
        for spine in ax.spines.values():
            spine.set_visible(False)
    elif style == 'box':
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color(COLOR_PALETTE['neutral'])
            spine.set_linewidth(1.0)

def safe_statistical_test(test_type: str, x: np.ndarray, y: np.ndarray) -> Tuple[float, float, str]:
    """
    Safely perform statistical tests while handling edge cases.
    
    Args:
        test_type: Type of test ('ttest_ind', 'pearsonr', etc.)
        x: First data array
        y: Second data array
        
    Returns:
        Tuple of (statistic, p_value, message)
    """
    # Remove NaN values
    x = x[~np.isnan(x)]
    y = y[~np.isnan(y)]
    
    # Check minimum sample sizes
    min_samples = 2
    if len(x) < min_samples or len(y) < min_samples:
        return np.nan, np.nan, "Insufficient samples"
        
    try:
        if test_type == 'ttest_ind':
            # Check for constant values
            if np.std(x) == 0 or np.std(y) == 0:
                return np.nan, np.nan, "Constant values in samples"
            stat, p_val = stats.ttest_ind(x, y, nan_policy='omit')
        elif test_type == 'pearsonr':
            # Check for constant values
            if np.std(x) == 0 or np.std(y) == 0:
                return np.nan, np.nan, "Constant values in samples"
            stat, p_val = stats.pearsonr(x, y)
        else:
            raise ValueError(f"Unsupported test type: {test_type}")
            
        # Handle invalid results
        if np.isnan(stat) or np.isnan(p_val):
            return np.nan, np.nan, "Invalid test result"
            
        return stat, p_val, format_statistical_result(stat, p_val)
        
    except Exception as e:
        return np.nan, np.nan, f"Error: {str(e)}" 