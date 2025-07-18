"""BERT heads module for classification and regression tasks.

This module provides various head implementations for BERT models,
following clean architecture patterns with modular components.
"""

# Import configuration classes
from .config import (
    HeadConfig,
    ClassificationConfig,
    RegressionConfig,
    # Factory functions
    get_base_config,
    get_classification_config,
    get_regression_config,
    # Preset configurations
    get_binary_classification_config,
    get_multiclass_classification_config,
    get_multilabel_classification_config,
    get_regression_preset_config,
    get_ranking_config,
)

# Import base class
from .base import BaseHead

# Import classification heads
from .classification import (
    BinaryClassificationHead,
    MulticlassClassificationHead,
    MultilabelClassificationHead,
    # Factory functions
    create_classification_head,
    create_binary_classification_head,
    create_multiclass_classification_head,
    create_multilabel_classification_head,
)

# Import regression heads
from .regression import (
    RegressionHead,
    OrdinalRegressionHead,
    QuantileRegressionHead,
    # Factory functions
    create_regression_head,
    create_standard_regression_head,
    create_ordinal_regression_head,
    create_quantile_regression_head,
)

# Import layers
from .layers import (
    # Pooling layers
    MeanPooling,
    MaxPooling,
    AttentionPooling,
    WeightedMeanPooling,
    LastTokenPooling,
    CLSTokenPooling,
    create_pooling_layer,
)

# Import utilities
from .utils import (
    # Loss functions
    binary_cross_entropy_loss,
    cross_entropy_loss,
    multilabel_bce_loss,
    FocalLoss,
    LabelSmoothingLoss,
    ContrastiveLoss,
    TripletLoss,
    compute_class_weights,
    create_loss_function,
    # Metrics
    accuracy,
    auc,
    f1_score,
    precision,
    recall,
    mae,
    mse,
    rmse,
    r2_score,
    mape,
    mean_average_precision,
    ndcg,
    hamming_loss,
    subset_accuracy,
    kendall_tau,
    compute_metrics,
    get_metrics_for_task,
)


# High-level factory function
def create_head(
    head_type: str,
    input_size: int,
    output_size: int = None,
    **kwargs
) -> BaseHead:
    """Create a head based on type and configuration.
    
    Args:
        head_type: Type of head to create
        input_size: Size of input features
        output_size: Size of output (required for some head types)
        **kwargs: Additional configuration parameters
        
    Returns:
        Head instance
        
    Raises:
        ValueError: If head type is unknown or required parameters are missing
    """
    head_type = head_type.lower()
    
    # Classification heads
    if head_type in ["binary", "binary_classification"]:
        return create_binary_classification_head(input_size, **kwargs)
    
    elif head_type in ["multiclass", "multiclass_classification"]:
        if output_size is None:
            raise ValueError("output_size (num_classes) required for multiclass head")
        return create_multiclass_classification_head(input_size, output_size, **kwargs)
    
    elif head_type in ["multilabel", "multilabel_classification"]:
        if output_size is None:
            raise ValueError("output_size (num_labels) required for multilabel head")
        return create_multilabel_classification_head(input_size, output_size, **kwargs)
    
    # Regression heads
    elif head_type in ["regression", "standard_regression"]:
        return create_standard_regression_head(input_size, **kwargs)
    
    elif head_type in ["ordinal", "ordinal_regression"]:
        if output_size is None:
            raise ValueError("output_size (num_classes) required for ordinal regression")
        return create_ordinal_regression_head(input_size, output_size, **kwargs)
    
    elif head_type in ["quantile", "quantile_regression"]:
        quantiles = kwargs.pop("quantiles", None)
        return create_quantile_regression_head(input_size, quantiles, **kwargs)
    
    else:
        raise ValueError(f"Unknown head type: {head_type}")


__all__ = [
    # Configuration
    "HeadConfig",
    "ClassificationConfig",
    "RegressionConfig",
    # Base class
    "BaseHead",
    # Classification heads
    "BinaryClassificationHead",
    "MulticlassClassificationHead",
    "MultilabelClassificationHead",
    # Regression heads
    "RegressionHead",
    "OrdinalRegressionHead",
    "QuantileRegressionHead",
    # High-level factory
    "create_head",
    # Classification factories
    "create_classification_head",
    "create_binary_classification_head",
    "create_multiclass_classification_head",
    "create_multilabel_classification_head",
    # Regression factories
    "create_regression_head",
    "create_standard_regression_head",
    "create_ordinal_regression_head",
    "create_quantile_regression_head",
    # Configuration factories
    "get_base_config",
    "get_classification_config",
    "get_regression_config",
    # Preset configurations
    "get_binary_classification_config",
    "get_multiclass_classification_config",
    "get_multilabel_classification_config",
    "get_regression_preset_config",
    "get_ranking_config",
    # Selected utilities (most common)
    "create_pooling_layer",
    "create_loss_function",
    "compute_metrics",
    "get_metrics_for_task",
]