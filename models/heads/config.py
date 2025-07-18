"""Configuration for BERT heads.

This module defines configuration classes for various head types,
following the patterns established in the BERT module.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Union


@dataclass
class HeadConfig:
    """Configuration for BERT heads.
    
    This class defines the common configuration parameters for all head types.
    """
    
    # Basic configuration
    input_size: int
    output_size: int
    head_type: str = "classification"  # Type of head
    
    # Architecture parameters
    hidden_sizes: List[int] = field(default_factory=list)
    dropout_prob: float = 0.1
    activation: str = "gelu"  # Activation function name
    use_bias: bool = True
    
    # Pooling configuration
    pooling_type: str = "cls"  # Type of pooling to use
    
    # Normalization
    use_layer_norm: bool = True
    layer_norm_eps: float = 1e-12
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        # Validate head type
        valid_head_types = {
            "classification", "binary_classification", "multiclass_classification",
            "multilabel_classification", "regression", "ranking"
        }
        if self.head_type not in valid_head_types:
            raise ValueError(f"Invalid head_type: {self.head_type}. Must be one of {valid_head_types}")
        
        # Validate pooling type
        valid_pooling_types = {"cls", "mean", "max", "attention", "weighted_mean", "last"}
        if self.pooling_type not in valid_pooling_types:
            raise ValueError(f"Invalid pooling_type: {self.pooling_type}. Must be one of {valid_pooling_types}")
        
        # Validate activation
        valid_activations = {"relu", "gelu", "tanh", "sigmoid", "silu", "leaky_relu", "none"}
        if self.activation not in valid_activations:
            raise ValueError(f"Invalid activation: {self.activation}. Must be one of {valid_activations}")
        
        # Auto-configure output size for specific head types
        if self.head_type == "binary_classification" and self.output_size != 2:
            self.output_size = 2
        elif self.head_type == "regression" and self.output_size != 1:
            self.output_size = 1


@dataclass
class ClassificationConfig(HeadConfig):
    """Configuration for classification heads."""
    
    # Classification-specific parameters
    num_classes: Optional[int] = None
    use_focal_loss: bool = False
    focal_loss_gamma: float = 2.0
    focal_loss_alpha: Optional[float] = None
    label_smoothing: float = 0.0
    
    def __post_init__(self):
        """Set classification-specific defaults."""
        if self.num_classes is None:
            self.num_classes = self.output_size
        
        # Ensure output_size matches num_classes
        self.output_size = self.num_classes
        
        # Call parent validation
        super().__post_init__()


@dataclass
class RegressionConfig(HeadConfig):
    """Configuration for regression heads."""
    
    # Regression-specific parameters
    loss_type: str = "mse"  # Loss function type: "mse", "mae", "huber"
    huber_delta: float = 1.0  # Delta for Huber loss
    
    def __post_init__(self):
        """Set regression-specific defaults."""
        self.head_type = "regression"
        self.output_size = 1  # Regression typically has single output
        
        # Regression often benefits from different pooling
        if not hasattr(self, "_pooling_set"):
            self.pooling_type = "mean"
        
        # Call parent validation
        super().__post_init__()


# Factory functions for creating configurations

def get_base_config(input_size: int, output_size: int, **kwargs) -> HeadConfig:
    """Create a base head configuration.
    
    Args:
        input_size: Size of input features
        output_size: Size of output
        **kwargs: Additional configuration parameters
        
    Returns:
        HeadConfig instance
    """
    return HeadConfig(
        input_size=input_size,
        output_size=output_size,
        **kwargs
    )


def get_classification_config(
    input_size: int,
    num_classes: int,
    head_type: str = "multiclass_classification",
    **kwargs
) -> ClassificationConfig:
    """Create a classification head configuration.
    
    Args:
        input_size: Size of input features
        num_classes: Number of output classes
        head_type: Type of classification head
        **kwargs: Additional configuration parameters
        
    Returns:
        ClassificationConfig instance
    """
    return ClassificationConfig(
        input_size=input_size,
        output_size=num_classes,
        num_classes=num_classes,
        head_type=head_type,
        **kwargs
    )


def get_regression_config(
    input_size: int,
    **kwargs
) -> RegressionConfig:
    """Create a regression head configuration.
    
    Args:
        input_size: Size of input features
        **kwargs: Additional configuration parameters
        
    Returns:
        RegressionConfig instance
    """
    return RegressionConfig(
        input_size=input_size,
        output_size=1,
        **kwargs
    )


# Preset configurations

def get_binary_classification_config(input_size: int) -> ClassificationConfig:
    """Get preset configuration for binary classification.
    
    Args:
        input_size: Size of input features
        
    Returns:
        Optimized configuration for binary classification
    """
    return ClassificationConfig(
        input_size=input_size,
        output_size=2,
        num_classes=2,
        head_type="binary_classification",
        hidden_sizes=[input_size // 2],
        dropout_prob=0.1,
        activation="gelu",
        pooling_type="cls",
    )


def get_multiclass_classification_config(input_size: int, num_classes: int) -> ClassificationConfig:
    """Get preset configuration for multiclass classification.
    
    Args:
        input_size: Size of input features
        num_classes: Number of classes
        
    Returns:
        Optimized configuration for multiclass classification
    """
    return ClassificationConfig(
        input_size=input_size,
        output_size=num_classes,
        num_classes=num_classes,
        head_type="multiclass_classification",
        hidden_sizes=[input_size // 2],
        dropout_prob=0.1,
        activation="gelu",
        pooling_type="cls",
    )


def get_multilabel_classification_config(input_size: int, num_labels: int) -> ClassificationConfig:
    """Get preset configuration for multilabel classification.
    
    Args:
        input_size: Size of input features
        num_labels: Number of labels
        
    Returns:
        Optimized configuration for multilabel classification
    """
    return ClassificationConfig(
        input_size=input_size,
        output_size=num_labels,
        num_classes=num_labels,
        head_type="multilabel_classification",
        hidden_sizes=[input_size // 2],
        dropout_prob=0.1,
        activation="gelu",
        pooling_type="attention",  # Attention pooling often works better for multilabel
    )


def get_regression_preset_config(input_size: int) -> RegressionConfig:
    """Get preset configuration for regression.
    
    Args:
        input_size: Size of input features
        
    Returns:
        Optimized configuration for regression
    """
    return RegressionConfig(
        input_size=input_size,
        output_size=1,
        hidden_sizes=[input_size // 2, input_size // 4],
        dropout_prob=0.2,
        activation="relu",
        pooling_type="mean",
        loss_type="mse",
    )


def get_ranking_config(input_size: int) -> HeadConfig:
    """Get preset configuration for ranking tasks.
    
    Args:
        input_size: Size of input features
        
    Returns:
        Optimized configuration for ranking
    """
    return HeadConfig(
        input_size=input_size,
        output_size=1,  # Ranking typically outputs a single score
        head_type="ranking",
        hidden_sizes=[input_size // 2, input_size // 4],
        dropout_prob=0.1,
        activation="gelu",
        pooling_type="attention",
    )


__all__ = [
    # Configuration classes
    "HeadConfig",
    "ClassificationConfig",
    "RegressionConfig",
    # Factory functions
    "get_base_config",
    "get_classification_config",
    "get_regression_config",
    # Preset configurations
    "get_binary_classification_config",
    "get_multiclass_classification_config",
    "get_multilabel_classification_config",
    "get_regression_preset_config",
    "get_ranking_config",
]