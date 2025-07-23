"""Unified protocol module for k-bert.

This module centralizes all protocol definitions used throughout the k-bert codebase,
providing a single source of truth for interface contracts.

The protocols are organized into logical groups:
- base: Common/shared protocols
- models: Model-related protocols
- data: Data pipeline protocols
- training: Training protocols
- plugins: Plugin system protocols
"""

# Import all protocols for convenient access
from .base import *
from .data import *
from .models import *
from .plugins import *
from .training import *

__all__ = [
    # Base protocols
    "Configurable",
    "Serializable",
    "Comparable",
    # Data protocols
    "Dataset",
    "DataLoader",
    "DataFormat",
    "BatchProcessor",
    "Augmenter",
    "FeatureAugmenter",
    # Model protocols
    "Model",
    "Head",
    "ModelConfig",
    # Training protocols
    "Trainer",
    "TrainerConfig",
    "TrainingState",
    "TrainingResult",
    "Optimizer",
    "LRScheduler",
    "TrainingHook",
    "MetricsCollector",
    "CheckpointManager",
    "Callback",
    "Metric",
    # Plugin protocols
    "Plugin",
    "PluginMetadata",
    "HeadPlugin",
    "AugmenterPlugin",
    "FeatureExtractorPlugin",
    "DataLoaderPlugin",
    "ModelPlugin",
    "MetricPlugin",
]