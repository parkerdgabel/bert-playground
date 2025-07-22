"""
Training strategies for advanced model training.

This module provides various training strategies using the strategy pattern,
allowing different training approaches to be plugged in easily.
"""

# Base strategy classes
from .base import (
    TrainingStrategy,
    BaseTrainingStrategy,
    StrategyConfig,
    StrategyManager,
    register_strategy,
    get_strategy,
    get_default_strategy,
    list_strategies,
)

# Standard strategies
from .standard import (
    StandardTraining,
    GradientAccumulationTraining,
    MixedPrecisionTraining,
    MLXOptimizedTraining,
)

# Advanced strategies
from .advanced import (
    CurriculumLearningTraining,
    AdversarialTraining,
    MultiTaskTraining,
)

# Legacy strategies (for backward compatibility)
from .multi_stage import (
    BERTTrainingStrategy,
    MultiStageBERTTrainer,
    TrainingStage,
)

# Register default strategies
register_strategy(StandardTraining(), set_as_default=True)
register_strategy(GradientAccumulationTraining())
register_strategy(MixedPrecisionTraining())
register_strategy(MLXOptimizedTraining())
register_strategy(CurriculumLearningTraining())
register_strategy(AdversarialTraining())
register_strategy(MultiTaskTraining())

__all__ = [
    # Base classes
    "TrainingStrategy",
    "BaseTrainingStrategy",
    "StrategyConfig",
    "StrategyManager",
    "register_strategy",
    "get_strategy",
    "get_default_strategy",
    "list_strategies",
    
    # Standard strategies
    "StandardTraining",
    "GradientAccumulationTraining", 
    "MixedPrecisionTraining",
    "MLXOptimizedTraining",
    
    # Advanced strategies
    "CurriculumLearningTraining",
    "AdversarialTraining",
    "MultiTaskTraining",
    
    # Legacy strategies
    "TrainingStage",
    "BERTTrainingStrategy",
    "MultiStageBERTTrainer",
]
