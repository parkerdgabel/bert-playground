"""BERT heads module for classification and regression tasks.

This module provides various head implementations for BERT models,
following clean architecture patterns with modular components.
"""

# Import configuration classes
# Import base class
from .base import BaseHead, get_default_config_for_head_type

# Import classification heads
from .classification import (
    BinaryClassificationHead,
    MulticlassClassificationHead,
    MultilabelClassificationHead,
    create_binary_classification_head,
    # Factory functions
    create_classification_head,
    create_multiclass_classification_head,
    create_multilabel_classification_head,
)
from .config import (
    ClassificationConfig,
    HeadConfig,
    RegressionConfig,
    # Factory functions
    get_base_config,
    # Preset configurations
    get_binary_classification_config,
    get_classification_config,
    get_multiclass_classification_config,
    get_multilabel_classification_config,
    get_ranking_config,
    get_regression_config,
    get_regression_preset_config,
)

# Import layers
from .layers import (
    AttentionPooling,
    CLSTokenPooling,
    LastTokenPooling,
    MaxPooling,
    # Pooling layers
    MeanPooling,
    WeightedMeanPooling,
    create_pooling_layer,
)

# Import regression heads
from .regression import (
    OrdinalRegressionHead,
    QuantileRegressionHead,
    RegressionHead,
    create_ordinal_regression_head,
    create_quantile_regression_head,
    # Factory functions
    create_regression_head,
    create_standard_regression_head,
)

# Import utilities
from .utils import (
    ContrastiveLoss,
    FocalLoss,
    LabelSmoothingLoss,
    TripletLoss,
    # Metrics
    accuracy,
    auc,
    # Loss functions
    binary_cross_entropy_loss,
    compute_class_weights,
    compute_metrics,
    create_loss_function,
    cross_entropy_loss,
    f1_score,
    get_metrics_for_task,
    hamming_loss,
    kendall_tau,
    mae,
    mape,
    mean_average_precision,
    mse,
    multilabel_bce_loss,
    ndcg,
    precision,
    r2_score,
    recall,
    rmse,
    subset_accuracy,
)


# High-level factory function
def create_head(
    head_type: str, input_size: int, output_size: int = None, **kwargs
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
        # Remove num_classes from kwargs if it exists to avoid duplicate arguments
        filtered_kwargs = {k: v for k, v in kwargs.items() if k != "num_classes"}
        return create_multiclass_classification_head(
            input_size, output_size, **filtered_kwargs
        )

    elif head_type in ["multilabel", "multilabel_classification"]:
        if output_size is None:
            raise ValueError("output_size (num_labels) required for multilabel head")
        return create_multilabel_classification_head(input_size, output_size, **kwargs)

    # Regression heads
    elif head_type in ["regression", "standard_regression"]:
        # Pass output_size if provided
        if output_size is not None:
            kwargs["output_size"] = output_size
        return create_standard_regression_head(input_size, **kwargs)

    elif head_type in ["ordinal", "ordinal_regression"]:
        if output_size is None:
            raise ValueError(
                "output_size (num_classes) required for ordinal regression"
            )
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
    "get_default_config_for_head_type",
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
    # Pooling layers
    "AttentionPooling",
    "CLSTokenPooling",
    "LastTokenPooling",
    "MaxPooling",
    "MeanPooling",
    "WeightedMeanPooling",
    # Loss functions
    "ContrastiveLoss",
    "FocalLoss",
    "LabelSmoothingLoss",
    "TripletLoss",
    "binary_cross_entropy_loss",
    "cross_entropy_loss",
    "multilabel_bce_loss",
    "compute_class_weights",
    # Metrics
    "accuracy",
    "auc",
    "f1_score",
    "hamming_loss",
    "kendall_tau",
    "mae",
    "mape",
    "mean_average_precision",
    "mse",
    "ndcg",
    "precision",
    "r2_score",
    "recall",
    "rmse",
    "subset_accuracy",
]
