"""Compatibility wrapper for mlflow_utils.py - redirects to unified implementation."""

import warnings

from .mlflow_helper import (
    MLflowExperimentTracker,
    MLflowModelRegistry,
    UnifiedMLflowTracker,
    create_experiment_tracker,
    track_experiment,
)

warnings.warn(
    "mlflow_utils.py is deprecated. Please import from utils.mlflow_helper instead.",
    DeprecationWarning,
    stacklevel=2,
)

# Re-export all symbols for backward compatibility
__all__ = [
    "UnifiedMLflowTracker",
    "MLflowExperimentTracker",
    "MLflowModelRegistry",
    "create_experiment_tracker",
    "track_experiment",
]
