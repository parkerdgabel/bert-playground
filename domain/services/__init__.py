"""Domain services containing business logic.

These services orchestrate domain entities and implement
business rules without any framework dependencies.
"""

# New framework-agnostic services
from .model_builder import ModelBuilder
from .training_orchestrator import (
    TrainingOrchestrator,
    TrainingPhase,
    TrainingPlan,
    TrainingProgress,
    StopReason,
)
from .evaluation_engine import (
    EvaluationEngine,
    EvaluationPlan,
    PredictionResult,
    MetricType,
)

# Existing services
from .training import ModelTrainingService
from .tokenization import TokenizationService
from .checkpointing import CheckpointingService

# Try to import evaluation service (may have different names)
try:
    from .evaluation import EvaluationService
except ImportError:
    try:
        from .evaluation import ModelEvaluationService as EvaluationService
    except ImportError:
        EvaluationService = None

# Keep existing imports for backward compatibility
try:
    from .training_service import (
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
        # New framework-agnostic services
        "ModelBuilder",
        "TrainingOrchestrator",
        "TrainingPhase",
        "TrainingPlan", 
        "TrainingProgress",
        "StopReason",
        "EvaluationEngine",
        "EvaluationPlan",
        "PredictionResult",
        "MetricType",
        
        # Existing services
        "ModelTrainingService",
        "TokenizationService",
        "CheckpointingService",
        
        # Legacy exports
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
    
    if EvaluationService:
        __all__.append("EvaluationService")
        
except ImportError:
    # If legacy modules don't exist, just export new ones
    __all__ = [
        # New framework-agnostic services
        "ModelBuilder",
        "TrainingOrchestrator",
        "TrainingPhase",
        "TrainingPlan", 
        "TrainingProgress",
        "StopReason",
        "EvaluationEngine",
        "EvaluationPlan",
        "PredictionResult",
        "MetricType",
        
        # Existing services
        "ModelTrainingService",
        "TokenizationService",
        "CheckpointingService",
    ]
    
    if EvaluationService:
        __all__.append("EvaluationService")