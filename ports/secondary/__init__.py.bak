"""Secondary ports for hexagonal architecture.

Secondary ports (driven ports) are interfaces that the application core uses
to interact with external systems. These are implemented by adapters.

Examples:
- Compute backend: MLX, PyTorch, JAX implementations
- Storage: File system, cloud storage, database implementations
- Monitoring: Logging, metrics, experiment tracking implementations
"""

from .compute import (
    ComputeBackend,
    Array,
    ArrayLike,
    Device,
    DType,
    Shape,
    Module,
    DataType,
    NeuralOps,
)
from .storage import (
    StorageService,
    ModelStorageService,
    StorageKey,
    StorageValue,
    Metadata,
)
from .monitoring import (
    MonitoringService,
    ExperimentTracker,
    Timer,
    Span,
    LogSeverity,
    MetricValue,
    Tags,
    Context,
)
from .configuration import (
    ConfigurationProvider,
    ConfigRegistry,
    ConfigValue,
    ConfigDict,
    ConfigPath,
)
from .neural import (
    NeuralBackend,
    NeuralModule,
    ActivationType,
    NormalizationType,
    LossType,
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
from .tokenizer import (
    TokenizerPort,
    TokenizerConfig,
    TokenizerOutput,
    TokenizerVocabulary,
)
from .optimization import (
    Optimizer,
    LRScheduler,
    OptimizerConfig,
    SchedulerConfig,
)
from .metrics import (
    Metric,
    MetricsCollector,
    MetricType,
)
from .checkpointing import (
    CheckpointManager,
    CheckpointInfo,
)

__all__ = [
    # Compute
    "ComputeBackend",
    "Array",
    "ArrayLike",
    "Device",
    "DType",
    "Shape",
    "Module",
    "DataType",
    "NeuralOps",
    # Storage
    "StorageService",
    "ModelStorageService",
    "StorageKey",
    "StorageValue",
    "Metadata",
    # Monitoring
    "MonitoringService",
    "ExperimentTracker",
    "Timer",
    "Span",
    "LogSeverity",
    "MetricValue",
    "Tags",
    "Context",
    # Configuration
    "ConfigurationProvider",
    "ConfigRegistry",
    "ConfigValue",
    "ConfigDict",
    "ConfigPath",
    # Neural
    "NeuralBackend",
    "NeuralModule",
    "ActivationType",
    "NormalizationType",
    "LossType",
    "AttentionConfig",
    "FeedForwardConfig",
    "EmbeddingConfig",
    "TransformerOutput",
    "TransformerLayerOutput",
    "InitializationType",
    "PositionalEncoding",
    "AttentionMask",
    "AttentionMaskType",
    # Tokenizer
    "TokenizerPort",
    "TokenizerConfig",
    "TokenizerOutput",
    "TokenizerVocabulary",
    # Optimization
    "Optimizer",
    "LRScheduler",
    "OptimizerConfig",
    "SchedulerConfig",
    # Metrics
    "Metric",
    "MetricsCollector",
    "MetricType",
    # Checkpointing
    "CheckpointManager",
    "CheckpointInfo",
]