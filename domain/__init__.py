"""Domain layer for k-bert.

This package contains the pure business logic for the BERT implementation,
completely independent of any ML framework or infrastructure concerns.

The domain layer follows hexagonal architecture principles:
- Contains only business logic and rules
- No framework dependencies (MLX, PyTorch, etc.)
- Uses ports (protocols) for external interactions
- Focuses on the "what" not the "how"

Structure:
- entities/: Core domain entities (models, training, datasets, metrics)
- services/: Domain services (training, evaluation, tokenization, checkpointing)
- ports/: Port interfaces for external dependencies
- exceptions/: Domain-specific exceptions
- models/: Legacy domain models (BERT architecture, configurations)
- data/: Legacy data domain models

All framework-specific implementations are handled by adapters in the
infrastructure layer that implement the domain's port interfaces.
"""

# New architecture imports
from .entities import (
    BertModel,
    ModelArchitecture,
    ModelWeights,
    TrainingSession,
    TrainingConfig as NewTrainingConfig,
    TrainingState as NewTrainingState,
    Dataset,
    DataBatch,
    TokenSequence,
    TrainingMetrics,
    EvaluationMetrics,
)

from .services import (
    ModelTrainingService,
    EvaluationService as NewEvaluationService,
    TokenizationService,
    CheckpointingService,
)

from .ports import (
    ComputePort,
    DataLoaderPort,
    DatasetPort,
    TokenizerPort,
    MonitoringPort,
    StoragePort,
    CheckpointPort,
    MetricsCalculatorPort,
)

from .exceptions import (
    TrainingError,
    ModelNotInitializedError,
    CheckpointError,
    InvalidConfigurationError,
    DataError,
    MetricsError,
    EarlyStoppingError,
)

# Legacy imports for backward compatibility
try:
    from .models import (
        BertDomainConfig,
        BertConfigPresets,
        BertDomainOutput,
        HeadOutput,
        TaskType,
        HeadConfig,
        HeadFactory,
    )
    
    from .services import (
        TrainingConfig,
        TrainingState,
        TrainingService,
        EvaluationConfig,
        EvaluationResult,
        EvaluationService,
    )
    
    from .data import (
        DataConfig,
        DatasetType,
        TaskDataType,
        TextExample,
        DataService,
        DataPipeline,
    )
    
    __all__ = [
        # New entities
        "BertModel",
        "ModelArchitecture",
        "ModelWeights",
        "TrainingSession",
        "NewTrainingConfig",
        "NewTrainingState",
        "Dataset",
        "DataBatch",
        "TokenSequence",
        "TrainingMetrics",
        "EvaluationMetrics",
        
        # New services
        "ModelTrainingService",
        "NewEvaluationService",
        "TokenizationService",
        "CheckpointingService",
        
        # Ports
        "ComputePort",
        "DataLoaderPort",
        "DatasetPort",
        "TokenizerPort",
        "MonitoringPort",
        "StoragePort",
        "CheckpointPort",
        "MetricsCalculatorPort",
        
        # Exceptions
        "TrainingError",
        "ModelNotInitializedError",
        "CheckpointError",
        "InvalidConfigurationError",
        "DataError",
        "MetricsError",
        "EarlyStoppingError",
        
        # Legacy models
        "BertDomainConfig",
        "BertConfigPresets",
        "BertDomainOutput",
        "HeadOutput",
        "TaskType",
        "HeadConfig",
        "HeadFactory",
        
        # Legacy services
        "TrainingConfig",
        "TrainingState",
        "TrainingService",
        "EvaluationConfig",
        "EvaluationResult",
        "EvaluationService",
        
        # Legacy data
        "DataConfig",
        "DatasetType",
        "TaskDataType",
        "TextExample",
        "DataService",
        "DataPipeline",
    ]
except ImportError:
    # If legacy modules don't exist, export only new components
    __all__ = [
        # Entities
        "BertModel",
        "ModelArchitecture",
        "ModelWeights",
        "TrainingSession",
        "NewTrainingConfig",
        "NewTrainingState",
        "Dataset",
        "DataBatch",
        "TokenSequence",
        "TrainingMetrics",
        "EvaluationMetrics",
        
        # Services
        "ModelTrainingService",
        "NewEvaluationService",
        "TokenizationService",
        "CheckpointingService",
        
        # Ports
        "ComputePort",
        "DataLoaderPort",
        "DatasetPort",
        "TokenizerPort",
        "MonitoringPort",
        "StoragePort",
        "CheckpointPort",
        "MetricsCalculatorPort",
        
        # Exceptions
        "TrainingError",
        "ModelNotInitializedError",
        "CheckpointError",
        "InvalidConfigurationError",
        "DataError",
        "MetricsError",
        "EarlyStoppingError",
    ]