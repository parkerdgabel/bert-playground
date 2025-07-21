"""
Core training components including protocols, base classes, and state management.
"""

from .base import (
    BaseTrainer,
    BaseTrainerConfig,
)
from .optimization import (
    GradientAccumulator,
    create_lr_scheduler,
    create_optimizer,
)
from .protocols import (
    DataLoader,
    LRScheduler,
    Model,
    Optimizer,
    Trainer,
    TrainerConfig,
    TrainingResult,
    TrainingState,
)
from .state import (
    CheckpointManager,
    TrainingStateManager,
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
