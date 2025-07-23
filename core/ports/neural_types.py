"""Type definitions and utilities for the neural network port.

This module provides additional type definitions, data classes, and utilities
that support the neural network abstraction layer.
"""

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Generic, TypeVar

from typing_extensions import TypeAlias

from .compute import Array, Shape

# Generic type variable for layer outputs
T = TypeVar('T')


class InitializationType(Enum):
    """Weight initialization methods."""
    
    XAVIER_UNIFORM = "xavier_uniform"
    XAVIER_NORMAL = "xavier_normal"
    KAIMING_UNIFORM = "kaiming_uniform"
    KAIMING_NORMAL = "kaiming_normal"
    NORMAL = "normal"
    UNIFORM = "uniform"
    ONES = "ones"
    ZEROS = "zeros"
    CONSTANT = "constant"


class PaddingType(Enum):
    """Padding types for convolutional layers."""
    
    VALID = "valid"
    SAME = "same"
    CONSTANT = "constant"
    REFLECT = "reflect"
    REPLICATE = "replicate"
    CIRCULAR = "circular"


class AttentionMaskType(Enum):
    """Types of attention masks."""
    
    PADDING = "padding"  # Mask for padding tokens
    CAUSAL = "causal"    # Causal (autoregressive) mask
    CUSTOM = "custom"    # Custom attention mask


@dataclass
class LayerConfig:
    """Base configuration for neural network layers."""
    
    name: str | None = None
    dtype: Any = None
    device: Any = None
    init_type: InitializationType = InitializationType.XAVIER_UNIFORM
    init_kwargs: dict[str, Any] = field(default_factory=dict)


@dataclass
class AttentionConfig:
    """Configuration for attention layers."""
    
    hidden_size: int
    num_attention_heads: int
    attention_dropout: float = 0.0
    use_bias: bool = True
    use_rope: bool = False
    rope_base: float = 10000.0
    max_position_embeddings: int = 8192
    attention_type: str = "global"  # global, local, alternating
    window_size: int = 128  # For local attention
    # Base layer config fields
    name: str | None = None
    dtype: Any = None
    device: Any = None
    init_type: InitializationType = InitializationType.XAVIER_UNIFORM
    init_kwargs: dict[str, Any] = field(default_factory=dict)


@dataclass
class FeedForwardConfig:
    """Configuration for feedforward layers."""
    
    hidden_size: int
    intermediate_size: int
    activation: str = "gelu"
    dropout: float = 0.0
    use_bias: bool = True
    use_gate: bool = False  # For GeGLU, SwiGLU
    gate_activation: str = "gelu"  # Activation for gating
    # Base layer config fields
    name: str | None = None
    dtype: Any = None
    device: Any = None
    init_type: InitializationType = InitializationType.XAVIER_UNIFORM
    init_kwargs: dict[str, Any] = field(default_factory=dict)


@dataclass
class EmbeddingConfig:
    """Configuration for embedding layers."""
    
    vocab_size: int
    embedding_dim: int
    padding_idx: int | None = None
    max_norm: float | None = None
    norm_type: float = 2.0
    scale_grad_by_freq: bool = False
    sparse: bool = False
    use_positional: bool = True
    max_position_embeddings: int = 512
    use_token_type: bool = True
    type_vocab_size: int = 2
    # Base layer config fields
    name: str | None = None
    dtype: Any = None
    device: Any = None
    init_type: InitializationType = InitializationType.XAVIER_UNIFORM
    init_kwargs: dict[str, Any] = field(default_factory=dict)


@dataclass
class TransformerLayerOutput:
    """Output from a transformer layer."""
    
    hidden_states: Array
    attention_weights: Array | None = None
    attention_scores: Array | None = None
    present_key_value: tuple[Array, Array] | None = None


@dataclass
class TransformerOutput:
    """Output from a full transformer model."""
    
    last_hidden_state: Array
    pooler_output: Array | None = None
    hidden_states: tuple[Array, ...] | None = None
    attentions: tuple[Array, ...] | None = None
    cross_attentions: tuple[Array, ...] | None = None


@dataclass
class LossOutput:
    """Output from loss computation."""
    
    loss: Array
    logits: Array | None = None
    labels: Array | None = None
    predictions: Array | None = None
    metrics: dict[str, float] = field(default_factory=dict)


class LayerNormalization:
    """Layer normalization configuration."""
    
    def __init__(
        self,
        normalized_shape: int | Shape,
        eps: float = 1e-5,
        elementwise_affine: bool = True,
        bias: bool = True
    ):
        self.normalized_shape = normalized_shape
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        self.bias = bias


class PositionalEncoding(Enum):
    """Types of positional encoding."""
    
    LEARNED = auto()      # Learned positional embeddings
    SINUSOIDAL = auto()   # Sinusoidal positional encoding
    ROPE = auto()         # Rotary positional embeddings
    ALIBI = auto()        # Attention with Linear Biases
    NONE = auto()         # No positional encoding


@dataclass
class AttentionMask:
    """Attention mask with metadata."""
    
    mask: Array
    mask_type: AttentionMaskType
    dtype: Any = None
    
    def to_additive(self) -> Array:
        """Convert boolean mask to additive mask for softmax."""
        # This would be implemented by the backend
        raise NotImplementedError("Backend must implement mask conversion")


class ModuleStatistics:
    """Statistics for a neural network module."""
    
    def __init__(self):
        self.input_mean: float | None = None
        self.input_std: float | None = None
        self.output_mean: float | None = None
        self.output_std: float | None = None
        self.gradient_mean: float | None = None
        self.gradient_std: float | None = None
        self.parameter_norm: float | None = None
        self.gradient_norm: float | None = None
    
    def update_forward(self, input: Array, output: Array) -> None:
        """Update statistics after forward pass."""
        # This would be implemented by monitoring utilities
        pass
    
    def update_backward(self, gradients: dict[str, Array]) -> None:
        """Update statistics after backward pass."""
        # This would be implemented by monitoring utilities
        pass


# Type aliases for common patterns
ModuleFactory: TypeAlias = Callable[..., Any]  # Returns a Module
LossFunction: TypeAlias = Callable[[Array, Array], Array]
MetricFunction: TypeAlias = Callable[[Array, Array], float]
InitFunction: TypeAlias = Callable[[Shape], Array]


class ModuleWrapper(Generic[T]):
    """Generic wrapper for framework-specific modules.
    
    This allows type-safe wrapping of backend-specific modules
    while maintaining the abstract interface.
    """
    
    def __init__(self, module: T):
        self._module = module
    
    @property
    def wrapped(self) -> T:
        """Get the wrapped module."""
        return self._module
    
    def __call__(self, *args, **kwargs) -> Any:
        """Forward call to wrapped module."""
        return self._module(*args, **kwargs)


def create_init_function(
    init_type: InitializationType,
    **kwargs
) -> InitFunction:
    """Create an initialization function.
    
    Args:
        init_type: Type of initialization
        **kwargs: Additional arguments for the initialization
        
    Returns:
        Initialization function that takes shape and returns array
    """
    def init_fn(shape: Shape) -> Array:
        # This would be implemented by the backend
        raise NotImplementedError(f"Backend must implement {init_type} initialization")
    
    return init_fn


# Activation function signatures
ActivationFunction: TypeAlias = Callable[[Array], Array]
GatedActivationFunction: TypeAlias = Callable[[Array, Array], Array]


@dataclass
class ModuleMetadata:
    """Metadata for a neural network module."""
    
    module_type: str
    version: str = "1.0"
    backend: str | None = None
    created_at: str | None = None
    config: dict[str, Any] = field(default_factory=dict)
    tags: list[str] = field(default_factory=list)
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "module_type": self.module_type,
            "version": self.version,
            "backend": self.backend,
            "created_at": self.created_at,
            "config": self.config,
            "tags": self.tags
        }
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ModuleMetadata":
        """Create from dictionary."""
        return cls(**data)


# Common shape patterns for BERT models
@dataclass
class BertShapes:
    """Common shape patterns in BERT models."""
    
    batch_size: int
    sequence_length: int
    hidden_size: int
    num_attention_heads: int
    num_hidden_layers: int
    intermediate_size: int
    vocab_size: int
    
    @property
    def attention_head_size(self) -> int:
        """Size of each attention head."""
        return self.hidden_size // self.num_attention_heads
    
    @property
    def all_head_size(self) -> int:
        """Total size of all attention heads."""
        return self.num_attention_heads * self.attention_head_size
    
    def attention_shape(self) -> Shape:
        """Shape of attention weights."""
        return (self.batch_size, self.num_attention_heads, 
                self.sequence_length, self.sequence_length)
    
    def hidden_shape(self) -> Shape:
        """Shape of hidden states."""
        return (self.batch_size, self.sequence_length, self.hidden_size)
    
    def logits_shape(self, num_classes: int) -> Shape:
        """Shape of classification logits."""
        return (self.batch_size, num_classes)