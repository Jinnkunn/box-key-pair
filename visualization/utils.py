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
    text = '\n'.join([f'{k}: {v:.3f}' for k, v in stats_dict.items()])
    
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