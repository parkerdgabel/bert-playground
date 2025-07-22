"""Data module for BERT Kaggle playground.

This module provides a comprehensive data abstraction layer for Kaggle competitions,
optimized for BERT models and Apple Silicon MLX framework.

Key Features:
- Unified interface for all Kaggle competition types
- MLX-optimized data loading with unified memory
- Automatic dataset analysis and optimization
- BERT-compatible tokenization and batching
- Real-time competition management and submission
- Streaming data processing at 1000+ samples/sec
- Component-based architecture for clean separation of concerns
"""

from .core import (
    CompetitionMetadata,
    CompetitionType,
    DatasetAnalyzer,
    DatasetRegistry,
    DatasetSpec,
    KaggleDataset,
)

from .loaders import (
    MLXDataLoader,
)

# New composition-based components
from .components import (
    CacheManager,
    ColumnSelector,
    DataReader,
    DataTransformer,
    DataValidator,
    MetadataManager,
)

from .composed_dataset import ComposedDataset

# Dependency injection
from .di import (
    DataContainer,
    DataComponentBuilder,
    create_dataset as create_composed_dataset,
    get_default_container,
    set_default_container,
)

__version__ = "0.1.0"

# Factory functions
from .factory import (
    create_data_pipeline,
    create_dataloader,
    create_dataset,
)

__all__ = [
    # Core abstractions
    "KaggleDataset",
    "DatasetSpec",
    "CompetitionType",
    "CompetitionMetadata",
    "DatasetAnalyzer",
    "DatasetRegistry",
    # Kaggle integration (to be implemented)
    # "KaggleClient",
    # "KaggleCompetitionDataset",
    # "LeaderboardTracker",
    # "SubmissionManager",
    # MLX-optimized loaders
    "MLXDataLoader",
    # Text conversion now handled by augmentation module
    # Factory functions
    "create_dataset",
    "create_dataloader",
    "create_data_pipeline",
    # New composition-based components
    "ComposedDataset",
    "DataReader",
    "DataValidator",
    "DataTransformer",
    "MetadataManager",
    "CacheManager",
    "ColumnSelector",
    # Dependency injection
    "DataContainer",
    "DataComponentBuilder",
    "create_composed_dataset",
    "get_default_container",
    "set_default_container",
]
