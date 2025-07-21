"""Compatibility wrapper for mlflow_config.py - redirects to unified implementation."""

import warnings

from .mlflow_helper import (
    EnhancedMLflowTracker,
    UnifiedMLflowTracker,
    create_experiment_tags,
    launch_mlflow_ui,
    setup_mlflow_tracking,
)

warnings.warn(
    "mlflow_config.py is deprecated. Please import from utils.mlflow_helper instead.",
    DeprecationWarning,
    stacklevel=2,
)

# Re-export all symbols for backward compatibility
__all__ = [
    "UnifiedMLflowTracker",
    "EnhancedMLflowTracker",
    "CentralizedMLflowConfig",
    "setup_mlflow_tracking",
    "create_experiment_tags",
    "launch_mlflow_ui",
]

# Compatibility aliases
CentralizedMLflowConfig = UnifiedMLflowTracker
