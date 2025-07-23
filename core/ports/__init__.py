"""Port interfaces for hexagonal architecture.

Ports define the boundaries between the core domain logic and external systems.
All external dependencies should be accessed through these port interfaces.
"""

from .compute import ComputeBackend, Array, Module
from .config import ConfigurationProvider
from .monitoring import MonitoringService
from .storage import StorageService
from .neural import (
    NeuralBackend,
    Module as NeuralModule,
    ActivationType,
    NormalizationType,
    LossType,
    create_neural_backend,
)
from .neural_types import (
    AttentionConfig,
    FeedForwardConfig,
    EmbeddingConfig,
    TransformerOutput,
    TransformerLayerOutput,
    InitializationType,
    PositionalEncoding,
    AttentionMask,
    AttentionMaskType,
)
from .neural_ops import NeuralOps, create_neural_ops

__all__ = [
    # Compute
    "ComputeBackend",
    "Array",
    "Module",
    # Config
    "ConfigurationProvider",
    # Monitoring 
    "MonitoringService",
    # Storage
    "StorageService",
    # Neural
    "NeuralBackend",
    "NeuralModule",
    "ActivationType",
    "NormalizationType", 
    "LossType",
    "create_neural_backend",
    # Neural types
    "AttentionConfig",
    "FeedForwardConfig",
    "EmbeddingConfig",
    "TransformerOutput",
    "TransformerLayerOutput",
    "InitializationType",
    "PositionalEncoding",
    "AttentionMask",
    "AttentionMaskType",
    # Neural ops
    "NeuralOps",
    "create_neural_ops",
]