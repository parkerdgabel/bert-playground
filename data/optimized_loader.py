"""Compatibility wrapper for optimized_loader.py - redirects to unified implementation."""

import warnings
from .unified_loader import *

warnings.warn(
    "optimized_loader.py is deprecated. Please import from data.unified_loader instead.",
    DeprecationWarning,
    stacklevel=2
)

# Re-export all symbols for backward compatibility
__all__ = [
    'OptimizedTitanicDataPipeline',
    'create_optimized_dataloaders'
]