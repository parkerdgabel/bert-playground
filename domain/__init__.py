"""Domain layer for k-bert.

This package contains the pure business logic for the BERT implementation,
completely independent of any ML framework or infrastructure concerns.

The domain layer follows hexagonal architecture principles:
- Contains only business logic and rules
- No framework dependencies (MLX, PyTorch, etc.)
- Uses ports (protocols) for external interactions
- Focuses on the "what" not the "how"

Structure:
- models/: Core domain models (BERT architecture, configurations, outputs)
- services/: Domain services (training logic, evaluation logic)
- data/: Data domain models and processing logic

All framework-specific implementations are handled by adapters in the
infrastructure layer that implement the domain's port interfaces.
"""

# Re-export main domain components for convenience
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
    # Models
    "BertDomainConfig",
    "BertConfigPresets",
    "BertDomainOutput",
    "HeadOutput",
    "TaskType",
    "HeadConfig",
    "HeadFactory",
    
    # Services
    "TrainingConfig",
    "TrainingState",
    "TrainingService",
    "EvaluationConfig",
    "EvaluationResult",
    "EvaluationService",
    
    # Data
    "DataConfig",
    "DatasetType",
    "TaskDataType",
    "TextExample",
    "DataService",
    "DataPipeline",
]