"""Data Transfer Objects for the application layer.

These DTOs provide a stable interface for external actors to interact
with the application, independent of internal domain models.
"""

from .training import TrainingRequestDTO, TrainingResponseDTO
from .evaluation import EvaluationRequestDTO, EvaluationResponseDTO
from .prediction import PredictionRequestDTO, PredictionResponseDTO

__all__ = [
    "TrainingRequestDTO",
    "TrainingResponseDTO",
    "EvaluationRequestDTO",
    "EvaluationResponseDTO",
    "PredictionRequestDTO",
    "PredictionResponseDTO",
]