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

def add_statistical_annotations(ax: Axes, stats_dict: Dict, loc: str = 'upper left',
                              box: bool = True) -> None:
    """Add statistical annotations to a plot."""
    # Format each value based on its type
    formatted_items = []
    for k, v in stats_dict.items():
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            formatted_items.append(f'{k}: {v:.3f}')
        else:
            formatted_items.append(f'{k}: {v}')
    
    text = '\n'.join(formatted_items)
    
    bbox_props = dict(
        facecolor='white',
        alpha=0.9,
        edgecolor=COLOR_PALETTE['neutral'],
        boxstyle='round,pad=0.5'
    ) if box else None
    
    ax.text(0.05 if loc == 'upper left' else 0.95,
            0.95,
            text,
            transform=ax.transAxes,
            verticalalignment='top',
            horizontalalignment='left' if loc == 'upper left' else 'right',
            fontsize=FONT_SIZES['annotation'],
            bbox=bbox_props)

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

def safe_statistical_test(test_type: str, *args, **kwargs) -> Tuple[float, float, str]:
    """
    Safely perform statistical tests with sample size checks.
    
    Args:
        test_type: Type of test ('ttest_ind', 'pearsonr', 'shapiro', etc.)
        *args: Arguments for the test
        **kwargs: Keyword arguments for the test
    
    Returns:
        Tuple[float, float, str]: (statistic, p_value, message)
    """
    try:
        # Check sample sizes
        min_samples = 3  # Minimum required samples for statistical tests
        
        if test_type == 'ttest_ind':
            group1, group2 = args
            if len(group1) < min_samples or len(group2) < min_samples:
                return np.nan, 1.0, f"Insufficient samples (n1={len(group1)}, n2={len(group2)})"
            stat, pval = stats.ttest_ind(*args, **kwargs)
            
        elif test_type == 'pearsonr':
            x, y = args
            if len(x) < min_samples:
                return np.nan, 1.0, f"Insufficient samples (n={len(x)})"
            stat, pval = stats.pearsonr(*args, **kwargs)
            
        elif test_type == 'shapiro':
            data = args[0]
            if len(data) < min_samples:
                return np.nan, 1.0, f"Insufficient samples (n={len(data)})"
            stat, pval = stats.shapiro(*args, **kwargs)
            
        elif test_type == 'f_oneway':
            min_group_size = min(len(group) for group in args)
            if min_group_size < min_samples:
                return np.nan, 1.0, f"Insufficient samples in one or more groups (min n={min_group_size})"
            stat, pval = stats.f_oneway(*args, **kwargs)
            
        else:
            raise ValueError(f"Unsupported test type: {test_type}")
        
        return stat, pval, "Test performed successfully"
        
    except Exception as e:
        return np.nan, 1.0, f"Error performing test: {str(e)}"

def format_statistical_result(stat: float, pval: float, msg: str, test_name: str = "") -> str:
    """
    Format statistical test results into a readable string.
    
    Args:
        stat: Test statistic
        pval: P-value
        msg: Message from the test
        test_name: Name of the test (optional)
    
    Returns:
        str: Formatted result string
    """
    if np.isnan(stat):
        return f"{test_name}: {msg}"
    
    significance = ""
    if pval < 0.001:
        significance = "***"
    elif pval < 0.01:
        significance = "**"
    elif pval < 0.05:
        significance = "*"
    
    result = f"{test_name}: stat={stat:.3f}, p={pval:.3f}{significance}"
    if msg != "Test performed successfully":
        result += f" ({msg})"
    
    return result 