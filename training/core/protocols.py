"""Training protocols - unified interfaces for training components in hexagonal architecture."""

# Re-export all training protocols from the centralized location
from core.protocols.training import (
    Callback,
    CheckpointManager,
    LRScheduler,
    Metric,
    MetricsCollector,
    Optimizer,
    Trainer,
    TrainerConfig,
    TrainingHook,
    TrainingResult,
    TrainingState,
)

# Also re-export commonly used protocols from other modules
from core.protocols.data import DataLoader
from core.protocols.models import Model

__all__ = [
    "Model",
    "DataLoader",
    "Optimizer",
    "LRScheduler",
    "TrainingState",
    "TrainingResult",
    "TrainerConfig",
    "Trainer",
    "TrainingHook",
    "Callback",
    "Metric",
    "MetricsCollector",
    "CheckpointManager",
]