"""
Declarative training module for MLX-based BERT models optimized for Kaggle competitions.

This module provides a configuration-driven approach to training with:
- Protocol-based interfaces for maximum flexibility
- Native MLX optimizations for Apple Silicon
- First-class MLflow integration
- Kaggle competition-specific features
"""

from .core import (
    # Core protocols
    Trainer,
    TrainerConfig,
    TrainingState,
    TrainingResult,
    # Base implementations
    BaseTrainer,
    BaseTrainerConfig,
)

from .callbacks import (
    # Callback protocols
    Callback,
    CallbackList,
    # Built-in callbacks
    EarlyStopping,
    ModelCheckpoint,
    LearningRateScheduler,
    ProgressBar,
    MLflowCallback,
    MetricsLogger,
)

from .metrics import (
    # Metrics protocols
    Metric,
    MetricsCollector,
    # Built-in metrics
    Accuracy,
    F1Score,
    AUC,
    Loss,
)

from .kaggle import (
    KaggleTrainer,
    KaggleTrainerConfig,
    CompetitionProfile,
)

from .factory import (
    create_trainer,
    register_trainer,
    list_trainers,
    get_trainer_config,
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