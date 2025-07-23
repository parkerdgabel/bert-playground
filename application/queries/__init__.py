"""Application queries - handle read operations."""

from .model_info import GetModelInfoQuery
from .metrics import GetTrainingMetricsQuery

__all__ = [
    "GetModelInfoQuery",
    "GetTrainingMetricsQuery",
]