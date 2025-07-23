"""Neural network abstraction port for framework-agnostic neural operations.

This port provides a comprehensive abstraction layer for neural network operations,
allowing models to be written in a framework-agnostic way. It supports swapping
between different ML frameworks (MLX, PyTorch, JAX) without changing model code.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Iterator, Protocol, Sequence, runtime_checkable

from typing_extensions import TypeAlias

from .compute import Array, ArrayLike, DataType, Device, DType, Shape

# Type aliases for neural network specific types
Parameter: TypeAlias = Any  # Framework-specific parameter type
Optimizer: TypeAlias = Any  # Framework-specific optimizer type
GradientDict: TypeAlias = dict[str, Array]
ParameterDict: TypeAlias = dict[str, Parameter]


class ActivationType(Enum):
    """Supported activation functions."""
    
    RELU = "relu"
    GELU = "gelu"
    SILU = "silu"  # Also known as Swish
    TANH = "tanh"
    SIGMOID = "sigmoid"
    LEAKY_RELU = "leaky_relu"
    ELU = "elu"
    SOFTMAX = "softmax"
    LOG_SOFTMAX = "log_softmax"


class NormalizationType(Enum):
    """Supported normalization types."""
    
    LAYER_NORM = "layer_norm"
    RMS_NORM = "rms_norm"
    BATCH_NORM = "batch_norm"
    GROUP_NORM = "group_norm"


class LossType(Enum):
    """Supported loss functions."""
    
    CROSS_ENTROPY = "cross_entropy"
    BINARY_CROSS_ENTROPY = "binary_cross_entropy"
    MSE = "mse"
    MAE = "mae"
    HUBER = "huber"
    SMOOTH_L1 = "smooth_l1"


@dataclass
class ModuleInfo:
    """Information about a neural network module."""
    
    name: str
    module_type: str
    trainable_params: int
    total_params: int
    input_shape: Shape | None = None
    output_shape: Shape | None = None


class Module(ABC):
    """Abstract base class for all neural network modules.
    
    This provides a framework-agnostic interface for neural network layers
    and models, similar to torch.nn.Module or mlx.nn.Module.
    """
    
    def __init__(self):
        """Initialize the module."""
        self._parameters: ParameterDict = {}
        self._modules: dict[str, "Module"] = {}
        self._training: bool = True
        self._name: str | None = None
    
    @abstractmethod
    def forward(self, *args, **kwargs) -> Any:
        """Forward pass of the module.
        
        Must be implemented by all subclasses.
        """
        pass
    
    def __call__(self, *args, **kwargs) -> Any:
        """Make the module callable."""
        return self.forward(*args, **kwargs)
    
    @property
    def training(self) -> bool:
        """Whether the module is in training mode."""
        return self._training
    
    def train(self, mode: bool = True) -> "Module":
        """Set the module in training or evaluation mode.
        
        Args:
            mode: Whether to set training mode (True) or eval mode (False)
            
        Returns:
            Self for chaining
        """
        self._training = mode
        for module in self._modules.values():
            module.train(mode)
        return self
    
    def eval(self) -> "Module":
        """Set the module in evaluation mode."""
        return self.train(False)
    
    def parameters(self) -> Iterator[Parameter]:
        """Iterator over module parameters."""
        for param in self._parameters.values():
            yield param
        for module in self._modules.values():
            yield from module.parameters()
    
    def named_parameters(self) -> Iterator[tuple[str, Parameter]]:
        """Iterator over module parameters with names."""
        for name, param in self._parameters.items():
            yield name, param
        for module_name, module in self._modules.items():
            for param_name, param in module.named_parameters():
                yield f"{module_name}.{param_name}", param
    
    def modules(self) -> Iterator["Module"]:
        """Iterator over all modules (including self)."""
        yield self
        for module in self._modules.values():
            yield from module.modules()
    
    def named_modules(self) -> Iterator[tuple[str, "Module"]]:
        """Iterator over all modules with names."""
        yield "", self
        for name, module in self._modules.items():
            yield name, module
            for subname, submodule in module.named_modules():
                if subname:
                    yield f"{name}.{subname}", submodule
    
    def add_module(self, name: str, module: "Module | None") -> None:
        """Add a child module.
        
        Args:
            name: Name of the module
            module: Module to add (can be None)
        """
        if module is not None:
            self._modules[name] = module
    
    def register_parameter(self, name: str, param: Parameter | None) -> None:
        """Register a parameter.
        
        Args:
            name: Name of the parameter
            param: Parameter to register (can be None)
        """
        if param is not None:
            self._parameters[name] = param
    
    def get_info(self) -> ModuleInfo:
        """Get information about this module."""
        trainable = sum(1 for _ in self.parameters())
        total = trainable  # In framework-agnostic setting, all params are trainable
        return ModuleInfo(
            name=self._name or self.__class__.__name__,
            module_type=self.__class__.__name__,
            trainable_params=trainable,
            total_params=total
        )


@runtime_checkable
class NeuralBackend(Protocol):
    """Port for neural network operations and layer creation.
    
    This protocol defines the interface that must be implemented by
    framework-specific adapters (MLX, PyTorch, JAX).
    """
    
    @property
    def name(self) -> str:
        """Name of the neural backend."""
        ...
    
    @property
    def supports_mixed_precision(self) -> bool:
        """Whether this backend supports mixed precision training."""
        ...
    
    # Layer creation methods
    
    def linear(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        dtype: DType | None = None
    ) -> Module:
        """Create a linear (fully connected) layer.
        
        Args:
            in_features: Size of input features
            out_features: Size of output features
            bias: Whether to include bias
            dtype: Data type for parameters
            
        Returns:
            Linear layer module
        """
        ...
    
    def embedding(
        self,
        num_embeddings: int,
        embedding_dim: int,
        padding_idx: int | None = None,
        max_norm: float | None = None,
        norm_type: float = 2.0,
        scale_grad_by_freq: bool = False,
        sparse: bool = False,
        dtype: DType | None = None
    ) -> Module:
        """Create an embedding layer.
        
        Args:
            num_embeddings: Size of the vocabulary
            embedding_dim: Dimension of embeddings
            padding_idx: If given, pads the output with zeros at this index
            max_norm: If given, renormalizes embeddings to have norm <= max_norm
            norm_type: The p of the p-norm to compute for max_norm
            scale_grad_by_freq: Scale gradients by the frequency of words
            sparse: If True, gradient w.r.t. weight will be sparse
            dtype: Data type for embeddings
            
        Returns:
            Embedding layer module
        """
        ...
    
    def layer_norm(
        self,
        normalized_shape: int | Shape,
        eps: float = 1e-5,
        elementwise_affine: bool = True,
        bias: bool = True,
        dtype: DType | None = None
    ) -> Module:
        """Create a layer normalization module.
        
        Args:
            normalized_shape: Shape to normalize over
            eps: Epsilon for numerical stability
            elementwise_affine: Whether to learn affine parameters
            bias: Whether to use bias in affine transformation
            dtype: Data type for parameters
            
        Returns:
            Layer normalization module
        """
        ...
    
    def rms_norm(
        self,
        normalized_shape: int | Shape,
        eps: float = 1e-6,
        elementwise_affine: bool = True,
        bias: bool = False,
        dtype: DType | None = None
    ) -> Module:
        """Create RMS normalization module.
        
        Args:
            normalized_shape: Shape to normalize over
            eps: Epsilon for numerical stability
            elementwise_affine: Whether to learn affine parameters
            bias: Whether to use bias (typically False for RMSNorm)
            dtype: Data type for parameters
            
        Returns:
            RMS normalization module
        """
        ...
    
    def dropout(
        self,
        p: float = 0.5,
        inplace: bool = False
    ) -> Module:
        """Create a dropout module.
        
        Args:
            p: Dropout probability
            inplace: Whether to perform operation in-place
            
        Returns:
            Dropout module
        """
        ...
    
    def multi_head_attention(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        bias: bool = True,
        add_bias_kv: bool = False,
        add_zero_attn: bool = False,
        kdim: int | None = None,
        vdim: int | None = None,
        batch_first: bool = True,
        dtype: DType | None = None
    ) -> Module:
        """Create a multi-head attention module.
        
        Args:
            embed_dim: Total dimension of the model
            num_heads: Number of attention heads
            dropout: Dropout probability
            bias: Whether to add bias to input/output projection
            add_bias_kv: Whether to add bias to key/value projection
            add_zero_attn: Whether to add a zero attention
            kdim: Dimension of keys (defaults to embed_dim)
            vdim: Dimension of values (defaults to embed_dim)
            batch_first: Whether batch dimension is first
            dtype: Data type for parameters
            
        Returns:
            Multi-head attention module
        """
        ...
    
    # Activation functions
    
    def activation(
        self,
        activation_type: ActivationType,
        **kwargs
    ) -> Module:
        """Create an activation module.
        
        Args:
            activation_type: Type of activation function
            **kwargs: Additional arguments for specific activations
            
        Returns:
            Activation module
        """
        ...
    
    def gelu(self, approximate: str = "none") -> Module:
        """Create GELU activation.
        
        Args:
            approximate: Approximation method ('none' or 'tanh')
            
        Returns:
            GELU activation module
        """
        ...
    
    def relu(self, inplace: bool = False) -> Module:
        """Create ReLU activation.
        
        Args:
            inplace: Whether to perform operation in-place
            
        Returns:
            ReLU activation module
        """
        ...
    
    def silu(self, inplace: bool = False) -> Module:
        """Create SiLU (Swish) activation.
        
        Args:
            inplace: Whether to perform operation in-place
            
        Returns:
            SiLU activation module
        """
        ...
    
    # Container modules
    
    def sequential(self, *modules: Module) -> Module:
        """Create a sequential container.
        
        Args:
            *modules: Modules to run sequentially
            
        Returns:
            Sequential container module
        """
        ...
    
    def module_list(self, modules: list[Module] | None = None) -> Module:
        """Create a module list container.
        
        Args:
            modules: Optional initial list of modules
            
        Returns:
            ModuleList container
        """
        ...
    
    def module_dict(self, modules: dict[str, Module] | None = None) -> Module:
        """Create a module dictionary container.
        
        Args:
            modules: Optional initial dictionary of modules
            
        Returns:
            ModuleDict container
        """
        ...
    
    # Loss functions
    
    def cross_entropy_loss(
        self,
        weight: Array | None = None,
        ignore_index: int = -100,
        reduction: str = "mean",
        label_smoothing: float = 0.0
    ) -> Callable[[Array, Array], Array]:
        """Create cross entropy loss function.
        
        Args:
            weight: Manual rescaling weight for each class
            ignore_index: Index to ignore in loss computation
            reduction: Reduction method ('none', 'mean', 'sum')
            label_smoothing: Label smoothing factor
            
        Returns:
            Loss function
        """
        ...
    
    def binary_cross_entropy_loss(
        self,
        weight: Array | None = None,
        reduction: str = "mean"
    ) -> Callable[[Array, Array], Array]:
        """Create binary cross entropy loss function.
        
        Args:
            weight: Manual rescaling weight
            reduction: Reduction method ('none', 'mean', 'sum')
            
        Returns:
            Loss function
        """
        ...
    
    def mse_loss(
        self,
        reduction: str = "mean"
    ) -> Callable[[Array, Array], Array]:
        """Create mean squared error loss function.
        
        Args:
            reduction: Reduction method ('none', 'mean', 'sum')
            
        Returns:
            Loss function
        """
        ...
    
    # Tensor operations
    
    def matmul(self, a: Array, b: Array) -> Array:
        """Matrix multiplication.
        
        Args:
            a: First matrix
            b: Second matrix
            
        Returns:
            Matrix product
        """
        ...
    
    def transpose(self, input: Array, dim0: int, dim1: int) -> Array:
        """Transpose two dimensions.
        
        Args:
            input: Input array
            dim0: First dimension
            dim1: Second dimension
            
        Returns:
            Transposed array
        """
        ...
    
    def reshape(self, input: Array, shape: Shape) -> Array:
        """Reshape array.
        
        Args:
            input: Input array
            shape: Target shape
            
        Returns:
            Reshaped array
        """
        ...
    
    def concat(self, arrays: Sequence[Array], dim: int = 0) -> Array:
        """Concatenate arrays along a dimension.
        
        Args:
            arrays: Arrays to concatenate
            dim: Dimension to concatenate along
            
        Returns:
            Concatenated array
        """
        ...
    
    def split(
        self,
        input: Array,
        split_size_or_sections: int | list[int],
        dim: int = 0
    ) -> list[Array]:
        """Split array along a dimension.
        
        Args:
            input: Input array
            split_size_or_sections: Size of splits or list of section sizes
            dim: Dimension to split along
            
        Returns:
            List of split arrays
        """
        ...
    
    def mean(
        self,
        input: Array,
        dim: int | tuple[int, ...] | None = None,
        keepdim: bool = False
    ) -> Array:
        """Compute mean.
        
        Args:
            input: Input array
            dim: Dimension(s) to reduce
            keepdim: Whether to keep reduced dimensions
            
        Returns:
            Mean values
        """
        ...
    
    def sum(
        self,
        input: Array,
        dim: int | tuple[int, ...] | None = None,
        keepdim: bool = False
    ) -> Array:
        """Compute sum.
        
        Args:
            input: Input array
            dim: Dimension(s) to reduce
            keepdim: Whether to keep reduced dimensions
            
        Returns:
            Sum values
        """
        ...
    
    def max(
        self,
        input: Array,
        dim: int | None = None,
        keepdim: bool = False
    ) -> Array | tuple[Array, Array]:
        """Compute maximum.
        
        Args:
            input: Input array
            dim: Dimension to reduce
            keepdim: Whether to keep reduced dimensions
            
        Returns:
            Maximum values (and indices if dim is specified)
        """
        ...
    
    def min(
        self,
        input: Array,
        dim: int | None = None,
        keepdim: bool = False
    ) -> Array | tuple[Array, Array]:
        """Compute minimum.
        
        Args:
            input: Input array
            dim: Dimension to reduce
            keepdim: Whether to keep reduced dimensions
            
        Returns:
            Minimum values (and indices if dim is specified)
        """
        ...
    
    def softmax(self, input: Array, dim: int = -1) -> Array:
        """Apply softmax.
        
        Args:
            input: Input array
            dim: Dimension to apply softmax over
            
        Returns:
            Softmax output
        """
        ...
    
    def log_softmax(self, input: Array, dim: int = -1) -> Array:
        """Apply log softmax.
        
        Args:
            input: Input array
            dim: Dimension to apply log softmax over
            
        Returns:
            Log softmax output
        """
        ...
    
    # Advanced operations for BERT models
    
    def rotary_embedding(
        self,
        dim: int,
        max_position_embeddings: int = 8192,
        base: float = 10000.0,
        scaling_factor: float = 1.0,
        dtype: DType | None = None
    ) -> Module:
        """Create rotary positional embeddings (RoPE).
        
        Args:
            dim: Dimension of embeddings
            max_position_embeddings: Maximum sequence length
            base: Base frequency
            scaling_factor: Scaling factor for frequency
            dtype: Data type
            
        Returns:
            RoPE module
        """
        ...
    
    def apply_rotary_pos_emb(
        self,
        q: Array,
        k: Array,
        cos: Array,
        sin: Array,
        position_ids: Array | None = None
    ) -> tuple[Array, Array]:
        """Apply rotary positional embeddings to query and key.
        
        Args:
            q: Query tensor
            k: Key tensor
            cos: Cosine values
            sin: Sine values
            position_ids: Optional position IDs
            
        Returns:
            Tuple of (rotated_q, rotated_k)
        """
        ...
    
    def masked_fill(
        self,
        input: Array,
        mask: Array,
        value: float
    ) -> Array:
        """Fill masked positions with value.
        
        Args:
            input: Input array
            mask: Boolean mask
            value: Value to fill
            
        Returns:
            Filled array
        """
        ...
    
    def where(
        self,
        condition: Array,
        x: Array | float,
        y: Array | float
    ) -> Array:
        """Select elements from x or y based on condition.
        
        Args:
            condition: Boolean condition
            x: Values for True
            y: Values for False
            
        Returns:
            Selected values
        """
        ...
    
    # Utility methods
    
    def parameter(
        self,
        data: ArrayLike,
        requires_grad: bool = True,
        dtype: DType | None = None
    ) -> Parameter:
        """Create a parameter from data.
        
        Args:
            data: Parameter data
            requires_grad: Whether parameter requires gradients
            dtype: Data type
            
        Returns:
            Parameter object
        """
        ...
    
    def no_grad(self) -> Any:
        """Context manager to disable gradient computation.
        
        Returns:
            Context manager
        """
        ...
    
    def enable_grad(self) -> Any:
        """Context manager to enable gradient computation.
        
        Returns:
            Context manager
        """
        ...
    
    def device_context(self, device: Device) -> Any:
        """Context manager for device placement.
        
        Args:
            device: Target device
            
        Returns:
            Context manager
        """
        ...
    
    def unsqueeze(self, input: Array, dim: int) -> Array:
        """Add a dimension at the specified position.
        
        Args:
            input: Input array
            dim: Position to add dimension
            
        Returns:
            Array with added dimension
        """
        ...
    
    def arange(
        self,
        start: int | float = 0,
        stop: int | float | None = None,
        step: int | float = 1,
        dtype: DType | None = None,
        device: Device | None = None
    ) -> Array:
        """Create array with evenly spaced values.
        
        Args:
            start: Start value (or stop if only one arg)
            stop: Stop value (exclusive)
            step: Step size
            dtype: Data type
            device: Target device
            
        Returns:
            Array of values
        """
        ...
    
    def broadcast_to(self, input: Array, shape: Shape) -> Array:
        """Broadcast array to shape.
        
        Args:
            input: Input array
            shape: Target shape
            
        Returns:
            Broadcasted array
        """
        ...
    
    def zeros_like(
        self,
        input: Array,
        dtype: DType | None = None,
        device: Device | None = None
    ) -> Array:
        """Create array of zeros with same shape as input.
        
        Args:
            input: Input array to match shape
            dtype: Data type
            device: Target device
            
        Returns:
            Array of zeros
        """
        ...
    
    def ones(
        self,
        shape: Shape,
        dtype: DType | None = None,
        device: Device | None = None
    ) -> Array:
        """Create array of ones.
        
        Args:
            shape: Array shape
            dtype: Data type
            device: Target device
            
        Returns:
            Array of ones
        """
        ...


class Identity(Module):
    """Identity module that passes input through unchanged."""
    
    def forward(self, x: Any) -> Any:
        """Pass through input unchanged."""
        return x


def create_neural_backend(backend_name: str) -> NeuralBackend:
    """Factory function to create a neural backend.
    
    Args:
        backend_name: Name of the backend ('mlx', 'pytorch', 'jax')
        
    Returns:
        Neural backend instance
        
    Raises:
        ValueError: If backend is not supported
    """
    backend_name = backend_name.lower()
    
    if backend_name == "mlx":
        from core.adapters.neural import MLXNeuralBackend
        return MLXNeuralBackend()
    elif backend_name in ["pytorch", "torch"]:
        from core.adapters.neural.pytorch_backend import PyTorchNeuralBackend
        return PyTorchNeuralBackend()
    elif backend_name == "jax":
        from core.adapters.neural.jax_backend import JAXNeuralBackend
        return JAXNeuralBackend()
    else:
        raise ValueError(f"Unsupported neural backend: {backend_name}")