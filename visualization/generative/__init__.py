"""
Visualization functions for generative model analysis.

This package provides visualization tools for:
- Parameter analysis
- Posterior analysis
- Recovery analysis
- Model selection
- Strategy analysis
- Model validation
- Uncertainty analysis
- Likelihood analysis
"""

from .posterior.plots import create_plots as create_posterior_plots
from .recovery.plots import create_plots as create_recovery_plots
from .selection.plots import create_plots as create_selection_plots
from .validation.plots import create_plots as create_validation_plots
from .uncertainty.plots import create_plots as create_uncertainty_plots
from .parameter.plots import create_plots as create_parameter_plots
from .likelihood.plots import create_plots as create_likelihood_plots
from .parameter.analysis import (
    create_parameter_distributions_plot,
    create_parameter_relationships_3d_plot,
    create_parameter_age_evolution_plot
)
__all__ = [
    'create_posterior_plots',
    'create_recovery_plots',
    'create_selection_plots',
    'create_strategy_plots',
    'create_validation_plots',
    'create_uncertainty_plots',
    'create_parameter_plots',
    'create_likelihood_plots'
]

def generate_all_visualizations(results_df, output_dir: str = 'output') -> None:
    """Generate all visualizations for the generative model analysis."""
    # Parameter analysis
    create_parameter_distributions_plot(results_df, output_dir)
    create_parameter_relationships_3d_plot(results_df, output_dir)
    create_parameter_age_evolution_plot(results_df, output_dir)

    print("All generative model visualizations have been saved to output/visualization/")
