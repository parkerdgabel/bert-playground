"""Secondary neural port - Neural network backend that the application depends on.

This port defines the neural network interface that the application core uses
for building and working with neural networks. It's a driven port implemented
by adapters for different frameworks (MLX, PyTorch, JAX, etc.).
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any, Protocol, runtime_checkable, Callable, Optional, Iterator, Dict, Tuple

from typing_extensions import TypeAlias
from infrastructure.di import port

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


@port()
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

    # Low-level neural operations (moved from compute port)
    
    def linear(
        self,
        input: Array,
        weight: Array,
        bias: Optional[Array] = None
    ) -> Array:
        """Linear transformation (fully connected layer).
        
        Args:
            input: Input array [*, in_features]
            weight: Weight matrix [out_features, in_features]
            bias: Optional bias [out_features]
            
        Returns:
            Output array [*, out_features]
        """
        ...

    def embedding(
        self,
        input: Array,
        weight: Array,
        padding_idx: Optional[int] = None
    ) -> Array:
        """Embedding lookup.
        
        Args:
            input: Input indices [*]
            weight: Embedding matrix [num_embeddings, embedding_dim]
            padding_idx: Optional padding index
            
        Returns:
            Embeddings [*, embedding_dim]
        """
        ...

    def layer_norm(
        self,
        input: Array,
        normalized_shape: Shape,
        weight: Optional[Array] = None,
        bias: Optional[Array] = None,
        eps: float = 1e-5
    ) -> Array:
        """Layer normalization.
        
        Args:
            input: Input array
            normalized_shape: Shape to normalize over
            weight: Optional scale parameter
            bias: Optional shift parameter
            eps: Epsilon for numerical stability
            
        Returns:
            Normalized array
        """
        ...

    def dropout(
        self,
        input: Array,
        p: float = 0.5,
        training: bool = True,
        seed: Optional[int] = None
    ) -> Array:
        """Dropout regularization.
        
        Args:
            input: Input array
            p: Dropout probability
            training: Whether in training mode
            seed: Random seed
            
        Returns:
            Array with dropout applied
        """
        ...

    def softmax(
        self,
        input: Array,
        dim: int = -1
    ) -> Array:
        """Softmax activation.
        
        Args:
            input: Input array
            dim: Dimension to apply softmax over
            
        Returns:
            Softmax output
        """
        ...

    def cross_entropy(
        self,
        input: Array,
        target: Array,
        reduction: str = "mean",
        ignore_index: int = -100
    ) -> Array:
        """Cross entropy loss.
        
        Args:
            input: Predictions [batch_size, num_classes]
            target: Targets [batch_size]
            reduction: Reduction method ('none', 'mean', 'sum')
            ignore_index: Index to ignore
            
        Returns:
            Loss value
        """
        ...

    def attention(
        self,
        query: Array,
        key: Array,
        value: Array,
        mask: Optional[Array] = None,
        dropout_p: float = 0.0,
        scale: Optional[float] = None,
        training: bool = True,
    ) -> tuple[Array, Optional[Array]]:
        """Scaled dot-product attention.
        
        Args:
            query: Query tensor [batch, seq_len, d_k]
            key: Key tensor [batch, seq_len, d_k]
            value: Value tensor [batch, seq_len, d_v]
            mask: Optional attention mask
            dropout_p: Dropout probability
            scale: Optional scale factor (default: 1/sqrt(d_k))
            training: Whether in training mode
            
        Returns:
            Tuple of (attention output, attention weights)
        """
        ...

    def gelu(self, input: Array, approximate: bool = False) -> Array:
        """GELU activation function.
        
        Args:
            input: Input array
            approximate: Whether to use approximate version
            
        Returns:
            GELU output
        """
        ...

    def swiglu(self, input: Array, gate: Array) -> Array:
        """SwiGLU activation function.
        
        Args:
            input: Input array
            gate: Gate array
            
        Returns:
            SwiGLU output
        """
        ...
    
    # Training operations (moved from compute port)
    
    def forward_pass(
        self,
        model: NeuralModule,
        inputs: Dict[str, Array],
        training: bool = False,
    ) -> Dict[str, Array]:
        """Perform forward pass through model.
        
        Args:
            model: Neural network model
            inputs: Input data as dictionary of arrays
            training: Whether in training mode
            
        Returns:
            Dictionary containing outputs (logits, loss, hidden_states, etc.)
        """
        ...
    
    def compute_loss(
        self,
        predictions: Array,
        targets: Array,
        loss_type: LossType = LossType.CROSS_ENTROPY,
        **kwargs: Any,
    ) -> Array:
        """Compute loss between predictions and targets.
        
        Args:
            predictions: Model predictions
            targets: Ground truth targets
            loss_type: Type of loss function
            **kwargs: Additional loss-specific parameters
            
        Returns:
            Loss value
        """
        ...
    
    def backward_pass(
        self,
        loss: Array,
        model: NeuralModule,
        retain_graph: bool = False,
    ) -> Dict[str, Array]:
        """Perform backward pass to compute gradients.
        
        Args:
            loss: Loss value to backpropagate
            model: Neural network model
            retain_graph: Whether to retain computation graph
            
        Returns:
            Dictionary of gradients
        """
        ...
    
    def optimize_step(
        self,
        model_params: Dict[str, Array],
        gradients: Dict[str, Array],
        optimizer_state: Dict[str, Any],
        learning_rate: float,
        **kwargs: Any,
    ) -> Tuple[Dict[str, Array], Dict[str, Any]]:
        """Perform optimization step.
        
        Args:
            model_params: Current model parameters
            gradients: Computed gradients
            optimizer_state: Current optimizer state
            learning_rate: Learning rate
            **kwargs: Additional optimizer parameters
            
        Returns:
            Tuple of (updated_parameters, updated_optimizer_state)
        """
        ...


# Alias for compatibility
NeuralPort = NeuralBackend
