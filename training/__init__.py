"""
Declarative training module for MLX-based BERT models optimized for Kaggle competitions.

This module provides a configuration-driven approach to training with:
- Protocol-based interfaces for maximum flexibility
- Native MLX optimizations for Apple Silicon
- First-class MLflow integration
- Kaggle competition-specific features
"""

from .callbacks import (
    # Callback protocols
    Callback,
    CallbackList,
    # Built-in callbacks
    EarlyStopping,
    LearningRateScheduler,
    MetricsLogger,
    MLflowCallback,
    ModelCheckpoint,
    ProgressBar,
)
from .core import (
    # Base implementations
    BaseTrainer,
    BaseTrainerConfig,
    # Core protocols
    Trainer,
    TrainerConfig,
    TrainingResult,
    TrainingState,
)
from .factory import (
    create_trainer,
    get_trainer_config,
    list_trainers,
    register_trainer,
)
from .kaggle import (
    CompetitionProfile,
    KaggleTrainer,
    KaggleTrainerConfig,
)
from .metrics import (
    AUC,
    # Built-in metrics
    Accuracy,
    F1Score,
    Loss,
    # Metrics protocols
    Metric,
    MetricsCollector,
)

__all__ = [
    # Core
    "Trainer",
    "TrainerConfig",
    "TrainingState",
    "TrainingResult",
    "BaseTrainer",
    "BaseTrainerConfig",
    # Callbacks
    "Callback",
    "CallbackList",
    "EarlyStopping",
    "ModelCheckpoint",
    "LearningRateScheduler",
    "ProgressBar",
    "MLflowCallback",
    "MetricsLogger",
    # Metrics
    "Metric",
    "MetricsCollector",
    "Accuracy",
    "F1Score",
    "AUC",
    "Loss",
    # Kaggle
    "KaggleTrainer",
    "KaggleTrainerConfig",
    "CompetitionProfile",
    # Factory
    "create_trainer",
    "register_trainer",
    "list_trainers",
    "get_trainer_config",
]
