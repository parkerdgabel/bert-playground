"""Core data module for BERT Kaggle playground.

This module provides the foundational classes and interfaces for handling
Kaggle datasets with MLX optimization and BERT integration.
"""

from .base import KaggleDataset, DatasetSpec, CompetitionType
from .metadata import CompetitionMetadata, DatasetAnalyzer
from .registry import DatasetRegistry

__all__ = [
    "KaggleDataset",
    "DatasetSpec", 
    "CompetitionType",
    "CompetitionMetadata",
    "DatasetAnalyzer",
    "DatasetRegistry",
]