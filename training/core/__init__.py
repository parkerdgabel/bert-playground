"""
Core training components including protocols, base classes, and state management.
"""

from .protocols import (
    Trainer,
    TrainerConfig,
    TrainingState,
    TrainingResult,
    Optimizer,
    LRScheduler,
    DataLoader,
    Model,
)

from .base import (
    BaseTrainer,
    BaseTrainerConfig,
)

from .state import (
    TrainingStateManager,
    CheckpointManager,
)

from .optimization import (
    create_optimizer,
    create_lr_scheduler,
    GradientAccumulator,
)

__all__ = [
    # Protocols
    "Trainer",
    "TrainerConfig",
    "TrainingState",
    "TrainingResult",
    "Optimizer",
    "LRScheduler",
    "DataLoader",
    "Model",
    # Base implementations
    "BaseTrainer",
    "BaseTrainerConfig",
    # State management
    "TrainingStateManager",
    "CheckpointManager",
    # Optimization
    "create_optimizer",
    "create_lr_scheduler",
    "GradientAccumulator",
]