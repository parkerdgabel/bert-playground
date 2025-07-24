"""Domain layer - Pure business logic and entities.

This layer contains the core business logic of the application,
with zero dependencies on frameworks or infrastructure.

The domain layer follows hexagonal architecture principles:
- Contains only business logic and rules  
- No framework dependencies (MLX, PyTorch, etc.)
- No infrastructure concerns
- Focuses on the "what" not the "how"
"""

# Entities
from .entities.model import (
    BertModel,
    ModelArchitecture,
    ModelWeights,
    TaskHead,
    ModelSpecification,
    ModelType,
    TaskType,
    ActivationType,
    AttentionType,
)
from .entities.dataset import (
    Dataset,
    DataBatch,
    DataSample,
    DatasetStatistics,
    DatasetSpecification,
    DatasetSplit,
    DataFormat,
)
from .entities.training import (
    TrainingSession,
    TrainingState,
    TrainingConfig,
)
from .entities.metrics import (
    TrainingMetrics,
    EvaluationMetrics,
)

# Services
from .services import (
    ModelBuilder,
    TrainingOrchestrator,
    TrainingPhase,
    TrainingPlan,
    TrainingProgress,
    StopReason,
    EvaluationEngine,
    EvaluationPlan,
    PredictionResult,
    MetricType,
    ModelTrainingService,
    TokenizationService,
    CheckpointingService,
)

# Exceptions
from .exceptions import (
    DomainException,
    ValidationException,
    TrainingError,
    ModelNotInitializedError,
    CheckpointError,
    InvalidConfigurationError,
)

__all__ = [
    # Entities
    "BertModel",
    "ModelArchitecture", 
    "ModelWeights",
    "TaskHead",
    "ModelSpecification",
    "ModelType",
    "TaskType",
    "ActivationType",
    "AttentionType",
    "Dataset",
    "DataBatch",
    "DataSample",
    "DatasetStatistics",
    "DatasetSpecification",
    "DatasetSplit",
    "DataFormat",
    "TrainingSession",
    "TrainingState",
    "TrainingConfig",
    "TrainingMetrics",
    "EvaluationMetrics",
    
    # Services
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
    "ModelTrainingService",
    "TokenizationService",
    "CheckpointingService",
    
    # Exceptions
    "DomainException",
    "ValidationException",
    "TrainingError",
    "ModelNotInitializedError",
    "CheckpointError",
    "InvalidConfigurationError",
]