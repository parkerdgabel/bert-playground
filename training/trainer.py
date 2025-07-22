"""Main trainer module - provides unified training interface using hexagonal architecture.

This module provides the strategy-based training system using dependency injection
and ports/adapters pattern.
"""

# Strategy-based system using hexagonal architecture
from .strategies import (
    StandardTraining,
    GradientAccumulationTraining,
    MixedPrecisionTraining,
    MLXOptimizedTraining,
    get_strategy,
    get_default_strategy,
)

# Core trainer implementations
from .core.base import BaseTrainer as _BaseTrainer
from .kaggle.trainer import KaggleTrainer as _KaggleTrainer

# Direct trainer classes using hexagonal architecture
class BaseTrainer(_BaseTrainer):
    """Standard base trainer using hexagonal architecture and dependency injection."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class KaggleTrainer(_KaggleTrainer):
    """Kaggle-specific trainer using hexagonal architecture and dependency injection."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


# Main trainer interface using hexagonal architecture
def create_trainer(strategy_name: str = "StandardTraining", **kwargs):
    """
    Create a trainer using the strategy-based system with hexagonal architecture.
    
    Args:
        strategy_name: Name of the training strategy to use
        **kwargs: Additional configuration arguments
        
    Returns:
        Training strategy instance
        
    Examples:
        # Create specific strategy
        strategy = create_trainer("MLXOptimizedTraining")
        pipeline = strategy.create_pipeline(context)
        
        # Use default strategy
        strategy = create_trainer()
    """
    return get_strategy(strategy_name)(**kwargs)


# Export hexagonal architecture interfaces
__all__ = [
    # Strategy-based system using hexagonal architecture
    "StandardTraining",
    "GradientAccumulationTraining", 
    "MixedPrecisionTraining",
    "MLXOptimizedTraining",
    "get_strategy",
    "get_default_strategy",
    "create_trainer",
    
    # Direct trainer classes
    "BaseTrainer",
    "KaggleTrainer",
]