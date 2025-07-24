"""Monitoring adapters for tracking training progress and metrics.

This package provides various implementations of the MonitoringPort interface:
- MLflow: For experiment tracking and model registry
- TensorBoard: For visualization and metrics logging
- Console: For rich terminal output
- Composite: For using multiple monitors simultaneously
- Loguru: Legacy structured logging adapter
"""

from .base import BaseMonitoringAdapter, BaseProgressBar
from .mlflow import MLflowMonitoringAdapter, MLflowConfig
from .tensorboard import TensorBoardMonitoringAdapter
from .console import ConsoleMonitoringAdapter
from .composite import MultiMonitorAdapter

# Re-export the existing loguru adapter for backward compatibility
try:
    from .loguru import LoguruMonitoringAdapter, MLflowExperimentTracker
except ImportError:
    # loguru not available
    LoguruMonitoringAdapter = None
    MLflowExperimentTracker = None

__all__ = [
    # Base classes
    "BaseMonitoringAdapter",
    "BaseProgressBar",
    
    # Implementations
    "MLflowMonitoringAdapter",
    "MLflowConfig",
    "TensorBoardMonitoringAdapter",
    "ConsoleMonitoringAdapter",
    "MultiMonitorAdapter",
    
    # Legacy (for backward compatibility)
    "LoguruMonitoringAdapter",
    "MLflowExperimentTracker",
]


def create_monitoring_adapter(
    adapter_type: str = "console",
    **kwargs
) -> BaseMonitoringAdapter:
    """Factory function to create monitoring adapters.
    
    Args:
        adapter_type: Type of adapter to create (mlflow, tensorboard, console, multi)
        **kwargs: Additional arguments for the adapter
        
    Returns:
        Monitoring adapter instance
        
    Raises:
        ValueError: If adapter type is unknown
    """
    adapters = {
        "mlflow": MLflowMonitoringAdapter,
        "tensorboard": TensorBoardMonitoringAdapter,
        "console": ConsoleMonitoringAdapter,
        "multi": MultiMonitorAdapter,
        "loguru": LoguruMonitoringAdapter,  # Legacy
    }
    
    if adapter_type not in adapters:
        raise ValueError(
            f"Unknown adapter type: {adapter_type}. "
            f"Available types: {', '.join(adapters.keys())}"
        )
    
    return adapters[adapter_type](**kwargs)