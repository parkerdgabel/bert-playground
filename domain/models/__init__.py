"""Domain models for BERT architecture and components.

This package contains pure business logic models that define:
- BERT architecture and configuration
- Task-specific heads
- Model outputs and data structures

All models are framework-agnostic and contain only business logic.
"""

from .bert_config import (
    BertDomainConfig,
    BertConfigPresets,
)

from .bert_output import (
    BertDomainOutput,
    HeadOutput,
    TrainingOutput,
    InferenceOutput,
    PoolingStrategy,
    OutputProcessor,
)

from .bert_architecture import (
    AttentionType,
    ActivationType,
    NormalizationType,
    LayerComponents,
    BertLayer,
    BertEmbeddings,
    BertArchitecture,
    BertPooler,
    AttentionPattern,
    ModelCapabilities,
)

from .task_heads import (
    TaskType,
    LossType,
    PoolingType,
    HeadConfig,
    TaskHead,
    ClassificationHead,
    RegressionHead,
    MultiLabelClassificationHead,
    TokenClassificationHead,
    SimilarityHead,
    HeadFactory,
    HeadSelectionCriteria,
)

__all__ = [
    # Configuration
    "BertDomainConfig",
    "BertConfigPresets",
    
    # Outputs
    "BertDomainOutput",
    "HeadOutput",
    "TrainingOutput", 
    "InferenceOutput",
    "PoolingStrategy",
    "OutputProcessor",
    
    # Architecture
    "AttentionType",
    "ActivationType",
    "NormalizationType",
    "LayerComponents",
    "BertLayer",
    "BertEmbeddings",
    "BertArchitecture",
    "BertPooler",
    "AttentionPattern",
    "ModelCapabilities",
    
    # Task Heads
    "TaskType",
    "LossType",
    "PoolingType",
    "HeadConfig",
    "TaskHead",
    "ClassificationHead",
    "RegressionHead",
    "MultiLabelClassificationHead",
    "TokenClassificationHead",
    "SimilarityHead",
    "HeadFactory",
    "HeadSelectionCriteria",
]