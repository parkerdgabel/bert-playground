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
"""

from .core import (
    CompetitionMetadata,
    CompetitionType, 
    DatasetAnalyzer,
    DatasetRegistry,
    DatasetSpec,
    KaggleDataset,
)
# Kaggle integration (to be implemented)
# from .kaggle import (
#     KaggleClient,
#     KaggleCompetitionDataset,
#     LeaderboardTracker,
#     SubmissionManager,
# )
from .loaders import (
    MLXDataLoader,
    StreamingPipeline,
    UnifiedMemoryManager,
)
from .templates import (
    BERTTextConverter,
    CompetitionTextTemplate,
    TabularTextConverter,
    TextTemplateEngine,
)

__version__ = "0.1.0"

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
    "StreamingPipeline", 
    "UnifiedMemoryManager",
    
    # Text conversion
    "TextTemplateEngine",
    "TabularTextConverter",
    "BERTTextConverter",
    "CompetitionTextTemplate",
]