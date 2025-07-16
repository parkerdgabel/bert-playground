"""Compatibility wrapper for titanic_loader.py - redirects to unified implementation."""

import warnings
from .unified_loader import (
    TitanicDataPipeline,
    create_data_loaders,
)

warnings.warn(
    "titanic_loader.py is deprecated. Please import from data.unified_loader instead.",
    DeprecationWarning,
    stacklevel=2,
)

# Re-export all symbols for backward compatibility
__all__ = ["TitanicDataPipeline", "create_data_loaders"]
