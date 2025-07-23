"""Secondary neural port - Neural network backend that the application depends on.

This port defines the neural network interface that the application core uses
for building and working with neural networks. It's a driven port implemented
by adapters for different frameworks (MLX, PyTorch, JAX, etc.).
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any, Protocol, runtime_checkable, Callable, Optional, Iterator, Dict

from typing_extensions import TypeAlias

# Import base types from compute port
from .compute import Array, Shape

# Type aliases
NeuralModule: TypeAlias = Any  # Framework-specific module type
Parameter: TypeAlias = Any  # Framework-specific parameter type
ParameterDict: TypeAlias = Dict[str, Parameter]
GradientDict: TypeAlias = Dict[str, Array]


class ActivationType(Enum):
    """Supported activation functions."""
    
    RELU = "relu"
    GELU = "gelu"
    SILU = "silu"
    TANH = "tanh"
    SIGMOID = "sigmoid"
    SOFTMAX = "softmax"
    SWIGLU = "swiglu"
    GEGLU = "geglu"


class NormalizationType(Enum):
    """Supported normalization types."""
    
    LAYER = "layer"
    BATCH = "batch"
    RMS = "rms"
    GROUP = "group"


class LossType(Enum):
    """Supported loss functions."""
    
    CROSS_ENTROPY = "cross_entropy"
    MSE = "mse"
    MAE = "mae"
    HUBER = "huber"
    COSINE_SIMILARITY = "cosine_similarity"


class InitializationType(Enum):
    """Weight initialization strategies."""
    
    NORMAL = "normal"
    UNIFORM = "uniform"
    XAVIER_UNIFORM = "xavier_uniform"
    XAVIER_NORMAL = "xavier_normal"
    KAIMING_UNIFORM = "kaiming_uniform"
    KAIMING_NORMAL = "kaiming_normal"


class PositionalEncoding(Enum):
    """Positional encoding types."""
    
    SINUSOIDAL = "sinusoidal"
    LEARNED = "learned"
    ROPE = "rope"
    ALIBI = "alibi"
    NONE = "none"


class AttentionMaskType(Enum):
    """Attention mask types."""
    
    CAUSAL = "causal"
    BIDIRECTIONAL = "bidirectional"
    PREFIX_LM = "prefix_lm"
    CUSTOM = "custom"


@dataclass
class AttentionConfig:
    """Configuration for attention layers."""
    
    num_heads: int
    head_dim: int
    dropout: float = 0.0
    use_bias: bool = True
    use_rotary: bool = False
    rotary_base: float = 10000.0
    rotary_interleaved: bool = False
    use_flash_attention: bool = False
    attention_scale: Optional[float] = None


@dataclass
class FeedForwardConfig:
    """Configuration for feed-forward layers."""
    
    hidden_dim: int
    activation: ActivationType = ActivationType.GELU
    dropout: float = 0.0
    use_bias: bool = True
    use_gated: bool = False  # For SwiGLU/GeGLU


@dataclass
class EmbeddingConfig:
    """Configuration for embedding layers."""
    
    vocab_size: int
    embedding_dim: int
    max_position_embeddings: int
    position_encoding: PositionalEncoding = PositionalEncoding.LEARNED
    dropout: float = 0.0
    padding_idx: Optional[int] = None
    scale_embeddings: bool = True


@dataclass
class TransformerLayerOutput:
    """Output from a single transformer layer."""
    
    hidden_states: Array
    attention_weights: Optional[Array] = None
    cross_attention_weights: Optional[Array] = None


@dataclass
class TransformerOutput:
    """Output from a full transformer model."""
    
    last_hidden_state: Array
    hidden_states: Optional[list[Array]] = None
    attentions: Optional[list[Array]] = None
    cross_attentions: Optional[list[Array]] = None


@dataclass
class AttentionMask:
    """Attention mask configuration."""
    
    mask_type: AttentionMaskType
    mask: Optional[Array] = None
    prefix_length: Optional[int] = None


@dataclass
class ModuleInfo:
    """Information about a neural module."""
    
    name: str
    module_type: str
    trainable_params: int
    total_params: int
    submodules: Optional[Dict[str, 'ModuleInfo']] = None


class Module(ABC):
    """Base class for neural network modules.
    
    This provides a framework-agnostic interface for neural network components.
    Adapters should wrap their framework-specific modules to implement this interface.
    """
    
    def __init__(self):
        """Initialize module."""
        self.training = True
        self._parameters: ParameterDict = {}
        self._modules: Dict[str, 'Module'] = {}
    
    @abstractmethod
    def forward(self, *args, **kwargs) -> Any:
        """Forward pass through the module."""
        ...
    
    def __call__(self, *args, **kwargs) -> Any:
        """Make module callable."""
        return self.forward(*args, **kwargs)
    
    def train(self, mode: bool = True) -> 'Module':
        """Set training mode."""
        self.training = mode
        for module in self._modules.values():
            module.train(mode)
        return self
    
    def eval(self) -> 'Module':
        """Set evaluation mode."""
        return self.train(False)
    
    def parameters(self) -> Iterator[Parameter]:
        """Get all parameters."""
        for param in self._parameters.values():
            yield param
        for module in self._modules.values():
            yield from module.parameters()
    
    def named_parameters(self, prefix: str = '') -> Iterator[tuple[str, Parameter]]:
        """Get all parameters with names."""
        for name, param in self._parameters.items():
            if prefix:
                yield f"{prefix}.{name}", param
            else:
                yield name, param
        
        for name, module in self._modules.items():
            submodule_prefix = f"{prefix}.{name}" if prefix else name
            yield from module.named_parameters(submodule_prefix)
    
    def add_module(self, name: str, module: Optional['Module']) -> None:
        """Add a submodule."""
        if module is not None:
            self._modules[name] = module
    
    def register_parameter(self, name: str, param: Optional[Parameter]) -> None:
        """Register a parameter."""
        if param is not None:
            self._parameters[name] = param
    
    def get_info(self) -> ModuleInfo:
        """Get module information."""
        trainable_params = sum(1 for _ in self.parameters())
        total_params = trainable_params  # In base implementation, all params are trainable
        
        submodules = {}
        for name, module in self._modules.items():
            submodules[name] = module.get_info()
        
        return ModuleInfo(
            name=getattr(self, '_name', 'unnamed'),
            module_type=self.__class__.__name__,
            trainable_params=trainable_params,
            total_params=total_params,
            submodules=submodules if submodules else None
        )


@runtime_checkable
class NeuralBackend(Protocol):
    """Secondary port for neural network operations.
    
    This interface provides high-level neural network building blocks
    that the application core uses. Implemented by framework-specific adapters.
    """

    @property
    def name(self) -> str:
        """Name of the neural backend."""
        ...

    @property
    def supports_mixed_precision(self) -> bool:
        """Whether backend supports mixed precision training."""
        ...

    def create_embeddings(
        self,
        config: EmbeddingConfig
    ) -> NeuralModule:
        """Create embedding layers.
        
        Args:
            config: Embedding configuration
            
        Returns:
            Embedding module
        """
        ...

    def create_attention(
        self,
        config: AttentionConfig,
        is_cross_attention: bool = False
    ) -> NeuralModule:
        """Create attention layer.
        
        Args:
            config: Attention configuration
            is_cross_attention: Whether this is cross-attention
            
        Returns:
            Attention module
        """
        ...

    def create_feed_forward(
        self,
        config: FeedForwardConfig
    ) -> NeuralModule:
        """Create feed-forward layer.
        
        Args:
            config: Feed-forward configuration
            
        Returns:
            Feed-forward module
        """
        ...

    def create_normalization(
        self,
        dim: int,
        norm_type: NormalizationType = NormalizationType.LAYER,
        eps: float = 1e-5
    ) -> NeuralModule:
        """Create normalization layer.
        
        Args:
            dim: Dimension to normalize
            norm_type: Type of normalization
            eps: Epsilon for numerical stability
            
        Returns:
            Normalization module
        """
        ...

    def create_activation(
        self,
        activation_type: ActivationType
    ) -> NeuralModule:
        """Create activation function.
        
        Args:
            activation_type: Type of activation
            
        Returns:
            Activation module
        """
        ...

    def create_dropout(
        self,
        p: float = 0.5
    ) -> NeuralModule:
        """Create dropout layer.
        
        Args:
            p: Dropout probability
            
        Returns:
            Dropout module
        """
        ...

    def create_linear(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        init_type: InitializationType = InitializationType.XAVIER_UNIFORM
    ) -> NeuralModule:
        """Create linear layer.
        
        Args:
            in_features: Input features
            out_features: Output features
            bias: Whether to use bias
            init_type: Weight initialization type
            
        Returns:
            Linear module
        """
        ...

    def create_loss_function(
        self,
        loss_type: LossType,
        **kwargs: Any
    ) -> Callable[[Array, Array], Array]:
        """Create loss function.
        
        Args:
            loss_type: Type of loss
            **kwargs: Loss-specific arguments
            
        Returns:
            Loss function
        """
        ...

    def apply_attention_mask(
        self,
        attention_scores: Array,
        mask: AttentionMask
    ) -> Array:
        """Apply attention mask to scores.
        
        Args:
            attention_scores: Raw attention scores
            mask: Attention mask configuration
            
        Returns:
            Masked attention scores
        """
        ...

    def create_position_encoding(
        self,
        max_length: int,
        dim: int,
        encoding_type: PositionalEncoding
    ) -> Array | NeuralModule:
        """Create positional encoding.
        
        Args:
            max_length: Maximum sequence length
            dim: Embedding dimension
            encoding_type: Type of encoding
            
        Returns:
            Position encoding (array or module)
        """
        ...

    def initialize_weights(
        self,
        module: NeuralModule,
        init_type: InitializationType,
        **kwargs: Any
    ) -> None:
        """Initialize module weights.
        
        Args:
            module: Module to initialize
            init_type: Initialization type
            **kwargs: Initialization-specific arguments
        """
        ...

    def count_parameters(
        self,
        module: NeuralModule,
        trainable_only: bool = True
    ) -> int:
        """Count module parameters.
        
        Args:
            module: Module to count parameters for
            trainable_only: Whether to count only trainable parameters
            
        Returns:
            Parameter count
        """
        ...

    def freeze_module(
        self,
        module: NeuralModule,
        freeze: bool = True
    ) -> None:
        """Freeze or unfreeze module parameters.
        
        Args:
            module: Module to freeze/unfreeze
            freeze: Whether to freeze
        """
        ...

    def get_module_device(
        self,
        module: NeuralModule
    ) -> str:
        """Get device of module.
        
        Args:
            module: Module to check
            
        Returns:
            Device string
        """
        ...

    def move_module_to_device(
        self,
        module: NeuralModule,
        device: str
    ) -> NeuralModule:
        """Move module to device.
        
        Args:
            module: Module to move
            device: Target device
            
        Returns:
            Module on new device
        """
        ...
# Alias for compatibility
NeuralPort = NeuralBackend
