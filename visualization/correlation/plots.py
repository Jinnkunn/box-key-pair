"""
Correlation analysis visualization functions.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import networkx as nx
from typing import Optional, Dict, List
from ..utils import (
    create_figure,
    apply_style_to_axis,
    add_statistical_annotations,
    save_figure
)
from ..config import (
    COLOR_PALETTE,
    CUSTOM_COLORMAPS,
    ALPHA_VALUES,
    FONT_SIZES
)

def create_correlation_matrix(
    data: pd.DataFrame,
    variables: Optional[List[str]] = None,
    method: str = 'pearson',
    output_dir: str = None
) -> None:
    """
    Create correlation matrix visualization.
    
    Args:
        data: DataFrame containing variables to correlate
        variables: List of variables to include (default: all numeric columns)
        method: Correlation method ('pearson', 'spearman', or 'kendall')
        output_dir: Directory to save the plot
    """
    if variables is None:
        variables = data.select_dtypes(include=[np.number]).columns.tolist()
    
    # Calculate correlation matrix
    corr_matrix = data[variables].corr(method=method)
    
    # Create figure
    fig, ax = create_figure(figsize='square')
    
    # Create heatmap
    sns.heatmap(corr_matrix,
                annot=True,
                cmap=CUSTOM_COLORMAPS['correlation'],
                vmin=-1,
                vmax=1,
                center=0,
                fmt='.2f',
                square=True,
                ax=ax)
    
    # Style the plot
    apply_style_to_axis(ax,
                       title=f'{method.capitalize()} Correlation Matrix',
                       xlabel='Variables',
                       ylabel='Variables')
    
    if output_dir:
        save_figure(fig, f'{output_dir}/correlation_matrix.png')

def create_correlation_heatmap(
    data: pd.DataFrame,
    x_vars: List[str],
    y_vars: List[str],
    output_dir: str = None
) -> None:
    """
    Create correlation heatmap between two sets of variables.
    
    Args:
        data: DataFrame containing variables
        x_vars: List of variables for x-axis
        y_vars: List of variables for y-axis
        output_dir: Directory to save the plot
    """
    # Calculate correlation matrix
    corr_matrix = pd.DataFrame(index=y_vars, columns=x_vars)
    p_matrix = pd.DataFrame(index=y_vars, columns=x_vars)
    
    for y in y_vars:
        for x in x_vars:
            r, p = stats.pearsonr(data[x], data[y])
            corr_matrix.loc[y, x] = r
            p_matrix.loc[y, x] = p
    
    # Create figure
    fig, ax = create_figure(figsize='wide')
    
    # Create heatmap
    sns.heatmap(corr_matrix,
                annot=True,
                cmap=CUSTOM_COLORMAPS['correlation'],
                vmin=-1,
                vmax=1,
                center=0,
                fmt='.2f',
                ax=ax)
    
    # Add significance markers
    for i, y in enumerate(y_vars):
        for j, x in enumerate(x_vars):
            if p_matrix.loc[y, x] < 0.001:
                ax.text(j + 0.5, i + 0.5, '***',
                       ha='center', va='center')
            elif p_matrix.loc[y, x] < 0.01:
                ax.text(j + 0.5, i + 0.5, '**',
                       ha='center', va='center')
            elif p_matrix.loc[y, x] < 0.05:
                ax.text(j + 0.5, i + 0.5, '*',
                       ha='center', va='center')
    
    # Style the plot
    apply_style_to_axis(ax,
                       title='Correlation Heatmap',
                       xlabel='Variables',
                       ylabel='Variables')
    
    if output_dir:
        save_figure(fig, f'{output_dir}/correlation_heatmap.png')

def create_correlation_network(
    data: pd.DataFrame,
    variables: List[str],
    threshold: float = 0.3,
    output_dir: str = None
) -> None:
    """
    Create correlation network visualization.
    
    Args:
        data: DataFrame containing variables
        variables: List of variables to include
        threshold: Minimum absolute correlation to show (default: 0.3)
        output_dir: Directory to save the plot
    """
    # Calculate correlation matrix
    corr_matrix = data[variables].corr()
    
    # Create network graph
    G = nx.Graph()
    
    # Add nodes
    for var in variables:
        G.add_node(var)
    
    # Add edges for correlations above threshold
    for i, var1 in enumerate(variables):
        for var2 in variables[i+1:]:
            corr = corr_matrix.loc[var1, var2]
            if abs(corr) >= threshold:
                G.add_edge(var1, var2, weight=abs(corr),
                          color='red' if corr < 0 else 'blue')
    
    # Create figure
    fig, ax = create_figure(figsize='square')
    
    # Draw network
    pos = nx.spring_layout(G)
    
    # Draw edges
    edges = G.edges()
    edge_colors = [G[u][v]['color'] for u, v in edges]
    edge_weights = [G[u][v]['weight'] * 2 for u, v in edges]
    
    nx.draw_networkx_edges(G, pos, edge_color=edge_colors,
                          width=edge_weights, alpha=0.6)
    
    # Draw nodes
    nx.draw_networkx_nodes(G, pos, node_size=1000,
                          node_color='white',
                          edgecolors=COLOR_PALETTE['neutral'])
    
    # Add labels
    nx.draw_networkx_labels(G, pos, font_size=FONT_SIZES['annotation'])
    
    # Style the plot
    ax.set_title('Correlation Network')
    ax.axis('off')
    
    if output_dir:
        save_figure(fig, f'{output_dir}/correlation_network.png') 