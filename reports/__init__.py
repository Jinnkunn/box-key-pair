"""
Reports package for analyzing learning dynamics data.

This package provides tools for:
- Individual differences analysis
- Generative model analysis
- Parameter analysis
- Learning dynamics
- Strategy analysis
- Model validation
- Uncertainty quantification
"""

from . import generative
from . import individual
from . import utils

__all__ = [
    'generative',
    'individual',
    'utils'
] 