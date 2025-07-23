"""Domain services for training and evaluation.

This package contains pure business logic services that define:
- Training workflows and strategies
- Evaluation and metric calculation
- Model lifecycle management
- Tokenization and data processing
- Checkpointing and model persistence

All services are framework-agnostic and contain only business logic.
"""

from .training import ModelTrainingService
from .evaluation import EvaluationService
from .tokenization import TokenizationService
from .checkpointing import CheckpointingService

# Keep existing imports for backward compatibility
try:
    from .training_service import (
        TrainingPhase,
        OptimizerType,
        SchedulerType,
        TrainingConfig,
        TrainingState,
        LearningRateSchedule,
        LinearSchedule,
        CosineSchedule,
        TrainingStrategy,
        EpochStrategy,
        StepStrategy,
        TrainingMetrics,
        TrainingService,
    )
    
    from .evaluation_service import (
        MetricType,
        MetricConfig,
        EvaluationConfig,
        EvaluationResult,
        MetricCalculator,
        ClassificationMetrics,
        RegressionMetrics,
        EvaluationService as LegacyEvaluationService,
        ConfidenceEstimator,
        UncertaintyEstimator,
    )
    
    __all__ = [
        # New services
        "ModelTrainingService",
        "EvaluationService",
        "TokenizationService",
        "CheckpointingService",
        
        # Legacy exports
        "TrainingPhase",
        "OptimizerType",
        "SchedulerType",
        "TrainingConfig",
        "TrainingState",
        "LearningRateSchedule",
        "LinearSchedule",
        "CosineSchedule",
        "TrainingStrategy",
        "EpochStrategy",
        "StepStrategy",
        "TrainingMetrics",
        "TrainingService",
        "MetricType",
        "MetricConfig",
        "EvaluationConfig",
        "EvaluationResult",
        "MetricCalculator",
        "ClassificationMetrics",
        "RegressionMetrics",
        "LegacyEvaluationService",
        "ConfidenceEstimator",
        "UncertaintyEstimator",
    ]
except ImportError:
    # If legacy modules don't exist, just export new ones
    __all__ = [
        "ModelTrainingService",
        "EvaluationService",
        "TokenizationService",
        "CheckpointingService",
    ]