"""Base head interface for all BERT heads in Kaggle competitions.

This module defines the abstract base class and common functionality
for all head types in the comprehensive BERT heads system.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Tuple, Union
import mlx.core as mx
import mlx.nn as nn
from enum import Enum
from dataclasses import dataclass

from loguru import logger


class HeadType(Enum):
    """Types of heads supported by the system."""
    BINARY_CLASSIFICATION = "binary_classification"
    MULTICLASS_CLASSIFICATION = "multiclass_classification"
    MULTILABEL_CLASSIFICATION = "multilabel_classification"
    REGRESSION = "regression"
    ORDINAL_REGRESSION = "ordinal_regression"
    TIME_SERIES = "time_series"
    RANKING = "ranking"
    HIERARCHICAL = "hierarchical"
    CONTRASTIVE = "contrastive"
    ENSEMBLE = "ensemble"
    MULTI_TASK = "multi_task"
    ADAPTIVE = "adaptive"


class PoolingType(Enum):
    """Types of pooling strategies for sequence-level tasks."""
    CLS = "cls"                    # Use [CLS] token
    MEAN = "mean"                  # Average pooling
    MAX = "max"                    # Max pooling
    ATTENTION = "attention"        # Attention-based pooling
    WEIGHTED_MEAN = "weighted_mean"  # Weighted average with learned weights
    LAST = "last"                  # Last token


class ActivationType(Enum):
    """Types of activation functions."""
    RELU = "relu"
    GELU = "gelu"
    TANH = "tanh"
    SIGMOID = "sigmoid"
    SWISH = "swish"
    LEAKY_RELU = "leaky_relu"
    NONE = "none"


@dataclass
class HeadConfig:
    """Configuration for BERT heads."""
    
    # Basic configuration
    head_type: HeadType
    input_size: int
    output_size: int
    
    # Architecture parameters
    hidden_sizes: List[int] = None  # Hidden layer sizes
    dropout_prob: float = 0.1
    use_layer_norm: bool = True
    activation: ActivationType = ActivationType.GELU
    
    # Pooling configuration
    pooling_type: PoolingType = PoolingType.CLS
    
    # Competition-specific settings
    competition_metric: str = "accuracy"  # Primary metric to optimize
    use_competition_tricks: bool = True   # Enable Kaggle-specific optimizations
    
    # Advanced features
    use_residual_connections: bool = False
    use_batch_norm: bool = False
    use_weight_norm: bool = False
    
    # Ensemble settings
    ensemble_size: int = 1
    use_uncertainty: bool = False
    
    def __post_init__(self):
        """Validate and set defaults."""
        if self.hidden_sizes is None:
            self.hidden_sizes = []
        
        # Auto-configure based on head type
        if self.head_type == HeadType.BINARY_CLASSIFICATION:
            if self.output_size != 2:
                self.output_size = 2
        elif self.head_type == HeadType.REGRESSION:
            if self.output_size != 1:
                self.output_size = 1


class BaseKaggleHead(nn.Module, ABC):
    """Abstract base class for all Kaggle competition heads.
    
    This class provides the common interface and shared functionality
    for all head types in the system.
    """
    
    def __init__(self, config: HeadConfig):
        """Initialize the base head.
        
        Args:
            config: Head configuration
        """
        super().__init__()
        self.config = config
        self.head_type = config.head_type
        
        # Build the head architecture
        self._build_head()
        
        # Initialize shared components
        self._build_shared_components()
        
        logger.info(f"Initialized {self.__class__.__name__} with config: {config}")
    
    def _build_head(self):
        """Build the head architecture. Must be implemented by subclasses."""
        self._build_pooling_layer()
        self._build_projection_layers()
        self._build_output_layer()
        self._build_loss_function()
    
    def _build_pooling_layer(self):
        """Build the pooling layer for sequence-level tasks."""
        if self.config.pooling_type == PoolingType.CLS:
            # Use [CLS] token - no additional layer needed
            self.pooling = None
        elif self.config.pooling_type == PoolingType.MEAN:
            self.pooling = MeanPooling()
        elif self.config.pooling_type == PoolingType.MAX:
            self.pooling = MaxPooling()
        elif self.config.pooling_type == PoolingType.ATTENTION:
            self.pooling = AttentionPooling(self.config.input_size)
        elif self.config.pooling_type == PoolingType.WEIGHTED_MEAN:
            self.pooling = WeightedMeanPooling(self.config.input_size)
        elif self.config.pooling_type == PoolingType.LAST:
            self.pooling = LastTokenPooling()
        else:
            raise ValueError(f"Unknown pooling type: {self.config.pooling_type}")
    
    def _build_projection_layers(self):
        """Build intermediate projection layers."""
        layers = []
        
        input_size = self.config.input_size
        
        # Build hidden layers
        for hidden_size in self.config.hidden_sizes:
            layers.append(nn.Linear(input_size, hidden_size))
            
            # Add normalization
            if self.config.use_layer_norm:
                layers.append(nn.LayerNorm(hidden_size))
            elif self.config.use_batch_norm:
                layers.append(nn.BatchNorm(hidden_size))
            
            # Add activation
            if self.config.activation != ActivationType.NONE:
                layers.append(self._get_activation_layer(self.config.activation))
            
            # Add dropout
            if self.config.dropout_prob > 0:
                layers.append(nn.Dropout(self.config.dropout_prob))
            
            input_size = hidden_size
        
        self.projection = nn.Sequential(*layers) if layers else nn.Identity()
        self.projection_output_size = input_size
    
    def _build_output_layer(self):
        """Build the final output layer. Must be implemented by subclasses."""
        pass
    
    def _build_loss_function(self):
        """Build the loss function. Must be implemented by subclasses."""
        pass
    
    def _build_shared_components(self):
        """Build shared components like uncertainty estimation."""
        if self.config.use_uncertainty:
            self.uncertainty_head = nn.Linear(self.projection_output_size, 1)
    
    def _get_activation_layer(self, activation: ActivationType) -> nn.Module:
        """Get activation layer by type."""
        if activation == ActivationType.RELU:
            return nn.ReLU()
        elif activation == ActivationType.GELU:
            return nn.GELU()
        elif activation == ActivationType.TANH:
            return nn.Tanh()
        elif activation == ActivationType.SIGMOID:
            return nn.Sigmoid()
        elif activation == ActivationType.SWISH:
            return nn.SiLU()  # SiLU is the same as Swish
        elif activation == ActivationType.LEAKY_RELU:
            return nn.LeakyReLU(0.1)
        else:
            raise ValueError(f"Unknown activation type: {activation}")
    
    def _apply_pooling(self, hidden_states: mx.array, attention_mask: Optional[mx.array] = None) -> mx.array:
        """Apply pooling to sequence hidden states.
        
        Args:
            hidden_states: Input hidden states [batch_size, seq_len, hidden_size]
            attention_mask: Optional attention mask [batch_size, seq_len]
            
        Returns:
            Pooled representation [batch_size, hidden_size]
        """
        if self.config.pooling_type == PoolingType.CLS:
            # Use [CLS] token (first token)
            return hidden_states[:, 0, :]
        elif self.pooling is not None:
            return self.pooling(hidden_states, attention_mask)
        else:
            # Fallback to CLS token
            return hidden_states[:, 0, :]
    
    def __call__(self, hidden_states: mx.array, attention_mask: Optional[mx.array] = None, **kwargs) -> Dict[str, mx.array]:
        """Make the head callable.
        
        Args:
            hidden_states: Input hidden states from BERT
            attention_mask: Optional attention mask
            **kwargs: Additional arguments
            
        Returns:
            Dictionary containing predictions and additional outputs
        """
        return self.forward(hidden_states, attention_mask, **kwargs)
    
    @abstractmethod
    def forward(self, hidden_states: mx.array, attention_mask: Optional[mx.array] = None, **kwargs) -> Dict[str, mx.array]:
        """Forward pass through the head.
        
        Args:
            hidden_states: Input hidden states from BERT
            attention_mask: Optional attention mask
            **kwargs: Additional arguments
            
        Returns:
            Dictionary containing predictions and additional outputs
        """
        pass
    
    @abstractmethod
    def compute_loss(self, predictions: Dict[str, mx.array], targets: mx.array, **kwargs) -> mx.array:
        """Compute loss for the head.
        
        Args:
            predictions: Predictions from forward pass
            targets: Ground truth targets
            **kwargs: Additional arguments
            
        Returns:
            Loss value
        """
        pass
    
    def compute_metrics(self, predictions: Dict[str, mx.array], targets: mx.array, **kwargs) -> Dict[str, float]:
        """Compute evaluation metrics for the head.
        
        Args:
            predictions: Predictions from forward pass
            targets: Ground truth targets
            **kwargs: Additional arguments
            
        Returns:
            Dictionary of metric values
        """
        # Default implementation - subclasses should override
        return {}
    
    def get_config(self) -> HeadConfig:
        """Get the head configuration."""
        return self.config
    
    def get_head_type(self) -> HeadType:
        """Get the head type."""
        return self.head_type


# Pooling layer implementations
class MeanPooling(nn.Module):
    """Mean pooling over sequence dimension."""
    
    def __call__(self, hidden_states: mx.array, attention_mask: Optional[mx.array] = None) -> mx.array:
        return self.forward(hidden_states, attention_mask)
    
    def forward(self, hidden_states: mx.array, attention_mask: Optional[mx.array] = None) -> mx.array:
        if attention_mask is not None:
            # Mask out padding tokens
            attention_mask = attention_mask.astype(mx.float32)
            attention_mask = attention_mask[..., None]  # Add feature dimension
            hidden_states = hidden_states * attention_mask
            
            # Compute mean over non-padded tokens
            seq_lengths = attention_mask.sum(axis=1)
            return hidden_states.sum(axis=1) / seq_lengths
        else:
            return hidden_states.mean(axis=1)


class MaxPooling(nn.Module):
    """Max pooling over sequence dimension."""
    
    def __call__(self, hidden_states: mx.array, attention_mask: Optional[mx.array] = None) -> mx.array:
        return self.forward(hidden_states, attention_mask)
    
    def forward(self, hidden_states: mx.array, attention_mask: Optional[mx.array] = None) -> mx.array:
        if attention_mask is not None:
            # Mask out padding tokens with large negative values
            attention_mask = attention_mask.astype(mx.float32)
            attention_mask = attention_mask[..., None]  # Add feature dimension
            hidden_states = hidden_states + (1.0 - attention_mask) * -1e9
        
        return hidden_states.max(axis=1)


class AttentionPooling(nn.Module):
    """Attention-based pooling over sequence dimension."""
    
    def __init__(self, hidden_size: int):
        super().__init__()
        self.attention = nn.Linear(hidden_size, 1)
    
    def __call__(self, hidden_states: mx.array, attention_mask: Optional[mx.array] = None) -> mx.array:
        return self.forward(hidden_states, attention_mask)
    
    def forward(self, hidden_states: mx.array, attention_mask: Optional[mx.array] = None) -> mx.array:
        # Compute attention scores
        attention_scores = self.attention(hidden_states).squeeze(-1)  # [batch_size, seq_len]
        
        if attention_mask is not None:
            # Mask out padding tokens
            attention_mask = attention_mask.astype(mx.float32)
            attention_scores = attention_scores + (1.0 - attention_mask) * -1e9
        
        # Apply softmax
        attention_weights = mx.softmax(attention_scores, axis=-1)  # [batch_size, seq_len]
        
        # Apply weighted sum
        attention_weights = attention_weights[..., None]  # Add feature dimension
        return (hidden_states * attention_weights).sum(axis=1)


class WeightedMeanPooling(nn.Module):
    """Weighted mean pooling with learned weights."""
    
    def __init__(self, hidden_size: int):
        super().__init__()
        self.weights = nn.Linear(hidden_size, 1)
    
    def __call__(self, hidden_states: mx.array, attention_mask: Optional[mx.array] = None) -> mx.array:
        return self.forward(hidden_states, attention_mask)
    
    def forward(self, hidden_states: mx.array, attention_mask: Optional[mx.array] = None) -> mx.array:
        # Compute position weights
        position_weights = mx.sigmoid(self.weights(hidden_states)).squeeze(-1)  # [batch_size, seq_len]
        
        if attention_mask is not None:
            # Mask out padding tokens
            attention_mask = attention_mask.astype(mx.float32)
            position_weights = position_weights * attention_mask
        
        # Normalize weights
        weight_sum = position_weights.sum(axis=1, keepdims=True)
        position_weights = position_weights / (weight_sum + 1e-9)
        
        # Apply weighted sum
        position_weights = position_weights[..., None]  # Add feature dimension
        return (hidden_states * position_weights).sum(axis=1)


class LastTokenPooling(nn.Module):
    """Use the last non-padding token."""
    
    def __call__(self, hidden_states: mx.array, attention_mask: Optional[mx.array] = None) -> mx.array:
        return self.forward(hidden_states, attention_mask)
    
    def forward(self, hidden_states: mx.array, attention_mask: Optional[mx.array] = None) -> mx.array:
        if attention_mask is not None:
            # Find the last non-padding token for each sequence
            seq_lengths = attention_mask.sum(axis=1) - 1  # 0-indexed
            batch_size = hidden_states.shape[0]
            batch_indices = mx.arange(batch_size)
            return hidden_states[batch_indices, seq_lengths]
        else:
            # Use the last token
            return hidden_states[:, -1, :]


# Utility functions
def create_head_config(
    head_type: HeadType,
    input_size: int,
    output_size: int,
    **kwargs
) -> HeadConfig:
    """Create a head configuration with sensible defaults.
    
    Args:
        head_type: Type of head to create
        input_size: Input feature size
        output_size: Output size
        **kwargs: Additional configuration parameters
        
    Returns:
        HeadConfig instance
    """
    return HeadConfig(
        head_type=head_type,
        input_size=input_size,
        output_size=output_size,
        **kwargs
    )


def get_default_config_for_head_type(head_type: HeadType, input_size: int, output_size: int) -> HeadConfig:
    """Get default configuration for a specific head type.
    
    Args:
        head_type: Type of head
        input_size: Input feature size
        output_size: Output size
        
    Returns:
        HeadConfig with optimized defaults for the head type
    """
    config_map = {
        HeadType.BINARY_CLASSIFICATION: {
            "pooling_type": PoolingType.CLS,
            "hidden_sizes": [input_size // 2],
            "dropout_prob": 0.1,
            "activation": ActivationType.GELU,
            "competition_metric": "auc",
        },
        HeadType.MULTICLASS_CLASSIFICATION: {
            "pooling_type": PoolingType.CLS,
            "hidden_sizes": [input_size // 2],
            "dropout_prob": 0.1,
            "activation": ActivationType.GELU,
            "competition_metric": "accuracy",
        },
        HeadType.REGRESSION: {
            "pooling_type": PoolingType.MEAN,
            "hidden_sizes": [input_size // 2, input_size // 4],
            "dropout_prob": 0.2,
            "activation": ActivationType.RELU,
            "competition_metric": "rmse",
        },
        HeadType.MULTILABEL_CLASSIFICATION: {
            "pooling_type": PoolingType.ATTENTION,
            "hidden_sizes": [input_size // 2],
            "dropout_prob": 0.1,
            "activation": ActivationType.GELU,
            "competition_metric": "f1_macro",
        },
        HeadType.RANKING: {
            "pooling_type": PoolingType.ATTENTION,
            "hidden_sizes": [input_size // 2, input_size // 4],
            "dropout_prob": 0.1,
            "activation": ActivationType.GELU,
            "competition_metric": "ndcg",
        },
    }
    
    # Get defaults for this head type
    defaults = config_map.get(head_type, {})
    
    return HeadConfig(
        head_type=head_type,
        input_size=input_size,
        output_size=output_size,
        **defaults
    )