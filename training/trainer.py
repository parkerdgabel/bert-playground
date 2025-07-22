"""Main trainer module - provides unified training interface.

This module provides both the new strategy-based training system and
backward compatibility with the legacy trainer interface.
"""

import warnings

# New strategy-based system (recommended)
from .strategies import (
    StandardTraining,
    GradientAccumulationTraining,
    MixedPrecisionTraining,
    MLXOptimizedTraining,
    get_strategy,
    get_default_strategy,
)

# Legacy trainers (deprecated but maintained for backward compatibility)
from .core.base import BaseTrainer as _BaseTrainer
from .kaggle.trainer import KaggleTrainer as _KaggleTrainer
from .compat import LegacyTrainerAdapter, create_legacy_trainer

# Backward compatibility wrappers
class BaseTrainer(_BaseTrainer):
    """Backward compatibility wrapper for BaseTrainer."""
    
    def __init__(self, *args, **kwargs):
        warnings.warn(
            "BaseTrainer is deprecated. Please use the new strategy-based system:\n"
            "from training.strategies import get_strategy\n"
            "strategy = get_strategy('StandardTraining')",
            DeprecationWarning,
            stacklevel=2
        )
        super().__init__(*args, **kwargs)


class KaggleTrainer(_KaggleTrainer):
    """Backward compatibility wrapper for KaggleTrainer."""
    
    def __init__(self, *args, **kwargs):
        warnings.warn(
            "KaggleTrainer is deprecated. Please use the new strategy-based system:\n"
            "from training.strategies import get_strategy\n"
            "strategy = get_strategy('MLXOptimizedTraining')",
            DeprecationWarning,
            stacklevel=2
        )
        super().__init__(*args, **kwargs)


# Main trainer class - now uses the strategy system but maintains compatibility
def Trainer(model=None, config=None, strategy="StandardTraining", **kwargs):
    """Main trainer factory function.
    
    This function provides a unified interface that can work with both
    the new strategy-based system and legacy configurations.
    
    Args:
        model: Model to train
        config: Training configuration
        strategy: Training strategy name or strategy instance
        **kwargs: Additional arguments
        
    Returns:
        Training strategy or legacy trainer adapter
        
    Examples:
        # New way (recommended)
        strategy = Trainer(strategy="MLXOptimizedTraining")
        pipeline = strategy.create_pipeline(context)
        
        # Legacy way (deprecated but supported)
        trainer = Trainer(model, config)  # Returns LegacyTrainerAdapter
    """
    # If model and config provided, assume legacy usage
    if model is not None and config is not None:
        warnings.warn(
            "Legacy Trainer(model, config) usage is deprecated. "
            "Please use the new strategy-based system.",
            DeprecationWarning,
            stacklevel=2
        )
        return LegacyTrainerAdapter(model, config, **kwargs)
    
    # New strategy-based usage
    if isinstance(strategy, str):
        return get_strategy(strategy)
    else:
        return strategy


# Export both new and legacy interfaces
__all__ = [
    # New strategy-based system (recommended)
    "StandardTraining",
    "GradientAccumulationTraining", 
    "MixedPrecisionTraining",
    "MLXOptimizedTraining",
    "get_strategy",
    "get_default_strategy",
    "Trainer",
    
    # Legacy interfaces (deprecated)
    "BaseTrainer",
    "KaggleTrainer",
    "create_legacy_trainer",
]