"""Domain services for training and evaluation.

This package contains pure business logic services that define:
- Training workflows and strategies
- Evaluation and metric calculation
- Model lifecycle management

All services are framework-agnostic and contain only business logic.
"""

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
    EvaluationService,
    ConfidenceEstimator,
    UncertaintyEstimator,
)

__all__ = [
    # Training
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
    
    # Evaluation
    "MetricType",
    "MetricConfig",
    "EvaluationConfig",
    "EvaluationResult",
    "MetricCalculator",
    "ClassificationMetrics",
    "RegressionMetrics",
    "EvaluationService",
    "ConfidenceEstimator",
    "UncertaintyEstimator",
]