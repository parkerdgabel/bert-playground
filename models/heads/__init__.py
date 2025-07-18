"""Comprehensive BERT heads system for Kaggle competitions.

This module provides a complete heads system with:
- Abstract base classes and interfaces
- Multiple head types for different competition types
- Dynamic head selection and configuration
- Competition-specific optimizations
- Registry system for head management
- Comprehensive loss functions and metrics
"""

from .base_head import (
    BaseKaggleHead,
    HeadType,
    PoolingType,
    ActivationType,
    HeadConfig,
    create_head_config,
    get_default_config_for_head_type,
    MeanPooling,
    MaxPooling,
    AttentionPooling,
    WeightedMeanPooling,
    LastTokenPooling,
)

from .head_registry import (
    HeadRegistry,
    HeadSpec,
    CompetitionType,
    get_head_registry,
    register_head,
    create_head_from_competition,
    list_available_heads,
    get_registry_info,
    register_head_class,
    infer_competition_type,
    get_head_type_from_competition,
)

from .classification_heads import (
    BinaryClassificationHead,
    MulticlassClassificationHead,
    MultilabelClassificationHead,
)

from .regression_heads import (
    RegressionHead,
    OrdinalRegressionHead,
    TimeSeriesRegressionHead,
)

from .loss_functions import (
    CompetitionLoss,
    FocalLoss,
    LabelSmoothingLoss,
    WeightedLoss,
    ContrastiveLoss,
    TripletLoss,
    RankingLoss,
    UncertaintyLoss,
    DistillationLoss,
    LossFactory,
    compute_class_weights,
    get_competition_loss,
)

from .metrics import (
    CompetitionMetric,
    AccuracyMetric,
    AUCMetric,
    F1Metric,
    PrecisionMetric,
    RecallMetric,
    MAEMetric,
    MSEMetric,
    RMSEMetric,
    R2Metric,
    MAPMetric,
    NDCGMetric,
    HammingLossMetric,
    SubsetAccuracyMetric,
    KendallTauMetric,
    MAPEMetric,
    MetricComputer,
    get_competition_metrics,
    compute_competition_metrics,
)

__all__ = [
    # Base classes and types
    "BaseKaggleHead",
    "HeadType",
    "PoolingType", 
    "ActivationType",
    "HeadConfig",
    "create_head_config",
    "get_default_config_for_head_type",
    
    # Pooling layers
    "MeanPooling",
    "MaxPooling",
    "AttentionPooling",
    "WeightedMeanPooling",
    "LastTokenPooling",
    
    # Registry system
    "HeadRegistry",
    "HeadSpec",
    "CompetitionType",
    "get_head_registry",
    "register_head",
    "create_head_from_competition",
    "list_available_heads",
    "get_registry_info",
    "register_head_class",
    "infer_competition_type",
    "get_head_type_from_competition",
    
    # Classification heads
    "BinaryClassificationHead",
    "MulticlassClassificationHead",
    "MultilabelClassificationHead",
    
    # Regression heads
    "RegressionHead",
    "OrdinalRegressionHead",
    "TimeSeriesRegressionHead",
    
    # Loss functions
    "CompetitionLoss",
    "FocalLoss",
    "LabelSmoothingLoss",
    "WeightedLoss",
    "ContrastiveLoss",
    "TripletLoss",
    "RankingLoss",
    "UncertaintyLoss",
    "DistillationLoss",
    "LossFactory",
    "compute_class_weights",
    "get_competition_loss",
    
    # Metrics
    "CompetitionMetric",
    "AccuracyMetric",
    "AUCMetric",
    "F1Metric",
    "PrecisionMetric",
    "RecallMetric",
    "MAEMetric",
    "MSEMetric",
    "RMSEMetric",
    "R2Metric",
    "MAPMetric",
    "NDCGMetric",
    "HammingLossMetric",
    "SubsetAccuracyMetric",
    "KendallTauMetric",
    "MAPEMetric",
    "MetricComputer",
    "get_competition_metrics",
    "compute_competition_metrics",
]