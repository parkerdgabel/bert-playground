"""Application layer - orchestrates domain services and ports.

This layer contains:
- Commands (write operations) that orchestrate business logic
- Queries (read operations) for retrieving information
- Services for complex orchestration and coordination
- DTOs for data transfer between layers
"""

from .commands import (
    TrainModelCommand,
    EvaluateModelCommand,
    PredictCommand,
    ExportModelCommand,
)
from .queries import (
    GetModelInfoQuery,
    GetTrainingMetricsQuery,
)
from .services import (
    TrainingOrchestrator,
    DataPipelineService,
    ExperimentTracker,
)
from .dto import (
    TrainingRequestDTO,
    TrainingResponseDTO,
    EvaluationRequestDTO,
    EvaluationResponseDTO,
    PredictionRequestDTO,
    PredictionResponseDTO,
    ExportRequestDTO,
    ExportResponseDTO,
)

__all__ = [
    # Commands
    "TrainModelCommand",
    "EvaluateModelCommand",
    "PredictCommand",
    "ExportModelCommand",
    # Queries
    "GetModelInfoQuery",
    "GetTrainingMetricsQuery",
    # Services
    "TrainingOrchestrator",
    "DataPipelineService",
    "ExperimentTracker",
    # DTOs
    "TrainingRequestDTO",
    "TrainingResponseDTO",
    "EvaluationRequestDTO",
    "EvaluationResponseDTO",
    "PredictionRequestDTO",
    "PredictionResponseDTO",
    "ExportRequestDTO",
    "ExportResponseDTO",
]