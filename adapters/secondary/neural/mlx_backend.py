"""MLX adapter implementation for the neural network port.

This module provides a complete MLX implementation of the NeuralBackend protocol,
enabling framework-agnostic neural network operations using Apple's MLX framework.
"""

from contextlib import contextmanager
from typing import Any, Callable, Iterator, Sequence, Dict, Tuple

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np

from domain.protocols.compute import Array, DataType
from ports.secondary.compute import ArrayLike, Device, DType, Shape
from ports.secondary.neural import (
    ActivationType,
    AttentionConfig,
    AttentionMask,
    EmbeddingConfig,
    FeedForwardConfig,
    GradientDict,
    InitializationType,
    LossType,
    Module,
    ModuleInfo,
    NeuralBackend,
    NeuralModule,
    NormalizationType,
    Parameter,
    ParameterDict,
    PositionalEncoding,
)


class MLXModule(Module):
    """Wrapper around mlx.nn.Module to implement our Module interface."""
    
    def __init__(self, mlx_module: nn.Module | None = None):
        """Initialize MLX module wrapper.
        
        Args:
            mlx_module: Optional MLX module to wrap
        """
        super().__init__()
        self._mlx_module = mlx_module
        if mlx_module is not None:
            # Sync parameters from MLX module
            self._sync_from_mlx()
    
    def _sync_from_mlx(self):
        """Synchronize parameters from MLX module."""
        if self._mlx_module is None:
            return
            
        # Get all parameters from MLX module
        for name, param in self._mlx_module.parameters().items():
            self._parameters[name] = param
            
        # Get all submodules
        for name, module in self._mlx_module.items():
            if isinstance(module, nn.Module):
                self._modules[name] = MLXModule(module)
    
    def forward(self, *args, **kwargs) -> Any:
        """Forward pass through the wrapped MLX module."""
        if self._mlx_module is None:
            raise NotImplementedError("No MLX module to forward through")
        return self._mlx_module(*args, **kwargs)
    
    def train(self, mode: bool = True) -> "Module":
        """Set training mode."""
        super().train(mode)
        if self._mlx_module is not None and hasattr(self._mlx_module, "train"):
            self._mlx_module.train(mode)
        return self
    
    def parameters(self) -> Iterator[Parameter]:
        """Get all parameters."""
        if self._mlx_module is not None:
            # Use MLX module's parameters
            for param in self._mlx_module.parameters().values():
                yield param
        else:
            # Use our own parameter tracking
            yield from super().parameters()
    
    def named_parameters(self) -> Iterator[tuple[str, Parameter]]:
        """Get all parameters with names."""
        if self._mlx_module is not None:
            # Use MLX module's parameters
            yield from self._mlx_module.parameters().items()
        else:
            # Use our own parameter tracking
            yield from super().named_parameters()


class MLXLinear(MLXModule):
    """MLX Linear layer wrapped as our Module."""
    
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        mlx_linear = nn.Linear(in_features, out_features, bias=bias)
        super().__init__(mlx_linear)


class MLXEmbedding(MLXModule):
    """MLX Embedding layer wrapped as our Module."""
    
    def __init__(self, num_embeddings: int, embedding_dim: int):
        mlx_embedding = nn.Embedding(num_embeddings, embedding_dim)
        super().__init__(mlx_embedding)


class MLXLayerNorm(MLXModule):
    """MLX LayerNorm wrapped as our Module."""
    
    def __init__(
        self, 
        dims: int | tuple[int, ...], 
        eps: float = 1e-5,
        affine: bool = True,
        bias: bool = True
    ):
        if affine:
            mlx_norm = nn.LayerNorm(dims, eps=eps, affine=True)
        else:
            # MLX LayerNorm always has affine, so we need a custom implementation
            mlx_norm = nn.LayerNorm(dims, eps=eps, affine=False)
        super().__init__(mlx_norm)


class MLXRMSNorm(MLXModule):
    """MLX RMSNorm wrapped as our Module."""
    
    def __init__(self, dims: int, eps: float = 1e-6):
        mlx_norm = nn.RMSNorm(dims, eps=eps)
        super().__init__(mlx_norm)


class MLXDropout(MLXModule):
    """MLX Dropout wrapped as our Module."""
    
    def __init__(self, p: float = 0.5):
        mlx_dropout = nn.Dropout(p=p)
        super().__init__(mlx_dropout)


class MLXMultiHeadAttention(MLXModule):
    """MLX MultiHeadAttention wrapped as our Module."""
    
    def __init__(
        self,
        dims: int,
        num_heads: int,
        query_input_dims: int | None = None,
        key_input_dims: int | None = None,
        value_input_dims: int | None = None,
        value_dims: int | None = None,
        value_output_dims: int | None = None,
        bias: bool = False,
    ):
        mlx_mha = nn.MultiHeadAttention(
            dims=dims,
            num_heads=num_heads,
            query_input_dims=query_input_dims,
            key_input_dims=key_input_dims,
            value_input_dims=value_input_dims,
            value_dims=value_dims,
            value_output_dims=value_output_dims,
            bias=bias,
        )
        super().__init__(mlx_mha)
    
    def forward(self, x: mx.array, y: mx.array | None = None, z: mx.array | None = None) -> mx.array:
        """Forward pass handling self-attention case."""
        if y is None:
            y = x
        if z is None:
            z = x
        return self._mlx_module(x, y, z)


class MLXActivation(MLXModule):
    """Generic activation wrapper."""
    
    def __init__(self, activation_fn: Callable):
        super().__init__()
        self.activation_fn = activation_fn
    
    def forward(self, x: mx.array) -> mx.array:
        return self.activation_fn(x)


class MLXSequential(MLXModule):
    """Sequential container for MLX modules."""
    
    def __init__(self, *modules: Module):
        super().__init__()
        self.layers = []
        for i, module in enumerate(modules):
            self.add_module(str(i), module)
            self.layers.append(module)
    
    def forward(self, x: mx.array) -> mx.array:
        for layer in self.layers:
            x = layer(x)
        return x


class MLXModuleList(MLXModule):
    """Module list container."""
    
    def __init__(self, modules: list[Module] | None = None):
        super().__init__()
        self._list = []
        if modules is not None:
            for i, module in enumerate(modules):
                self.add_module(str(i), module)
                self._list.append(module)
    
    def append(self, module: Module):
        """Append a module to the list."""
        idx = len(self._list)
        self.add_module(str(idx), module)
        self._list.append(module)
    
    def __getitem__(self, idx: int) -> Module:
        return self._list[idx]
    
    def __len__(self) -> int:
        return len(self._list)
    
    def forward(self, *args, **kwargs):
        raise NotImplementedError("ModuleList has no forward method")


class MLXModuleDict(MLXModule):
    """Module dictionary container."""
    
    def __init__(self, modules: dict[str, Module] | None = None):
        super().__init__()
        if modules is not None:
            for name, module in modules.items():
                self.add_module(name, module)
    
    def __getitem__(self, key: str) -> Module:
        return self._modules[key]
    
    def forward(self, *args, **kwargs):
        raise NotImplementedError("ModuleDict has no forward method")


class MLXRotaryEmbedding(MLXModule):
    """Rotary positional embeddings (RoPE) for MLX."""
    
    def __init__(
        self,
        dims: int,
        traditional: bool = False,
        base: float = 10000,
        scale: float = 1.0
    ):
        super().__init__()
        self.dims = dims
        self.traditional = traditional
        self.base = base
        self.scale = scale
        
        # Precompute frequencies
        self._freqs = None
    
    def _compute_freqs(self, seq_len: int) -> tuple[mx.array, mx.array]:
        """Compute cosine and sine frequencies."""
        if self._freqs is not None and self._freqs[0].shape[0] >= seq_len:
            return self._freqs[0][:seq_len], self._freqs[1][:seq_len]
        
        # Compute theta values
        theta = 1.0 / (self.base ** (mx.arange(0, self.dims, 2) / self.dims))
        
        # Scale positions
        positions = mx.arange(seq_len) * self.scale
        
        # Compute angles
        angles = positions[:, None] * theta[None, :]
        
        # Compute cos and sin
        cos = mx.cos(angles)
        sin = mx.sin(angles)
        
        # Expand dimensions for broadcasting
        cos = mx.expand_dims(cos, 1)  # [seq_len, 1, dims//2]
        sin = mx.expand_dims(sin, 1)  # [seq_len, 1, dims//2]
        
        self._freqs = (cos, sin)
        return cos, sin
    
    def forward(self, x: mx.array, offset: int = 0) -> tuple[mx.array, mx.array]:
        """Forward pass returns cos and sin for positions."""
        seq_len = x.shape[1] if len(x.shape) > 1 else x.shape[0]
        cos, sin = self._compute_freqs(seq_len + offset)
        if offset > 0:
            cos = cos[offset:]
            sin = sin[offset:]
        return cos, sin


class MLXNeuralBackend(NeuralBackend):
    """MLX implementation of the NeuralBackend protocol."""
    
    @property
    def name(self) -> str:
        """Name of the neural backend."""
        return "mlx"
    
    @property
    def supports_mixed_precision(self) -> bool:
        """MLX supports mixed precision through bfloat16."""
        return True
    
    # Layer creation methods
    
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
        linear = MLXLinear(in_features, out_features, bias=bias)
        # Initialize weights according to init_type
        self.initialize_weights(linear, init_type)
        return linear
    
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
        # Note: MLX doesn't support all PyTorch embedding features
        if config.padding_idx is not None:
            print(f"Warning: MLX embeddings don't support padding_idx")
        if not config.scale_embeddings:
            print(f"Warning: MLX embeddings don't support disabling scale_embeddings")
            
        embedding = MLXEmbedding(config.vocab_size, config.embedding_dim)
        
        # Apply dropout if specified
        if config.dropout > 0:
            dropout = MLXDropout(config.dropout)
            return MLXSequential(embedding, dropout)
        
        return embedding
    
    def layer_norm(
        self,
        normalized_shape: int | Shape,
        eps: float = 1e-5,
        elementwise_affine: bool = True,
        bias: bool = True,
        dtype: DType | None = None
    ) -> Module:
        """Create layer normalization."""
        dims = normalized_shape if isinstance(normalized_shape, int) else normalized_shape[-1]
        return MLXLayerNorm(dims, eps=eps, affine=elementwise_affine, bias=bias)
    
    def rms_norm(
        self,
        normalized_shape: int | Shape,
        eps: float = 1e-6,
        elementwise_affine: bool = True,
        bias: bool = False,
        dtype: DType | None = None
    ) -> Module:
        """Create RMS normalization."""
        dims = normalized_shape if isinstance(normalized_shape, int) else normalized_shape[-1]
        if bias:
            print("Warning: MLX RMSNorm doesn't support bias")
        return MLXRMSNorm(dims, eps=eps)
    
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
        return MLXDropout(p=p)
    
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
        embed_dim = config.num_heads * config.head_dim
        
        if config.use_flash_attention:
            print("Warning: MLX doesn't have native flash attention support")
        if is_cross_attention:
            print("Warning: MLX MHA cross-attention may need special handling")
            
        return MLXMultiHeadAttention(
            dims=embed_dim,
            num_heads=config.num_heads,
            query_input_dims=embed_dim,
            key_input_dims=embed_dim,
            value_input_dims=embed_dim,
            bias=config.use_bias
        )
    
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
        if norm_type == NormalizationType.LAYER:
            return MLXLayerNorm(dim, eps=eps)
        elif norm_type == NormalizationType.RMS:
            return MLXRMSNorm(dim, eps=eps)
        elif norm_type == NormalizationType.BATCH:
            print("Warning: MLX doesn't have native batch normalization")
            # Fall back to layer norm
            return MLXLayerNorm(dim, eps=eps)
        elif norm_type == NormalizationType.GROUP:
            print("Warning: MLX doesn't have native group normalization")
            # Fall back to layer norm
            return MLXLayerNorm(dim, eps=eps)
        else:
            raise ValueError(f"Unsupported normalization type: {norm_type}")
    
    # Activation functions
    
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
        if activation_type == ActivationType.RELU:
            return MLXActivation(nn.relu)
        elif activation_type == ActivationType.GELU:
            return MLXActivation(nn.gelu)
        elif activation_type == ActivationType.SILU:
            return MLXActivation(nn.silu)
        elif activation_type == ActivationType.TANH:
            return MLXActivation(mx.tanh)
        elif activation_type == ActivationType.SIGMOID:
            return MLXActivation(mx.sigmoid)
        elif activation_type == ActivationType.SOFTMAX:
            return MLXActivation(lambda x: mx.softmax(x, axis=-1))
        elif activation_type == ActivationType.SWIGLU:
            # SwiGLU needs special handling - it takes two inputs
            return MLXActivation(lambda x: self._swiglu_activation(x))
        elif activation_type == ActivationType.GEGLU:
            # GeGLU needs special handling - it takes two inputs  
            return MLXActivation(lambda x: self._geglu_activation(x))
        else:
            raise ValueError(f"Unsupported activation type: {activation_type}")
    
    def _swiglu_activation(self, x: mx.array) -> mx.array:
        """SwiGLU activation - splits input in half and applies SwiGLU."""
        # Split input in half along last dimension
        gate, input_part = mx.split(x, 2, axis=-1)
        return nn.silu(gate) * input_part
    
    def _geglu_activation(self, x: mx.array) -> mx.array:
        """GeGLU activation - splits input in half and applies GeGLU."""
        # Split input in half along last dimension
        gate, input_part = mx.split(x, 2, axis=-1)
        return nn.gelu(gate) * input_part
    
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
        if config.use_gated:
            # For gated feed-forward (SwiGLU/GeGLU), we need to double the hidden dimension
            # because we split it in half for gating
            gate_linear = MLXLinear(config.hidden_dim, config.hidden_dim * 2, bias=config.use_bias)
            activation = self.create_activation(config.activation)
            down_linear = MLXLinear(config.hidden_dim, config.hidden_dim, bias=config.use_bias)
            
            layers = [gate_linear, activation]
            if config.dropout > 0:
                layers.append(MLXDropout(config.dropout))
            layers.append(down_linear)
            
            return MLXSequential(*layers)
        else:
            # Standard feed-forward
            up_linear = MLXLinear(config.hidden_dim, config.hidden_dim, bias=config.use_bias)
            activation = self.create_activation(config.activation)
            down_linear = MLXLinear(config.hidden_dim, config.hidden_dim, bias=config.use_bias)
            
            layers = [up_linear, activation]
            if config.dropout > 0:
                layers.append(MLXDropout(config.dropout))
            layers.append(down_linear)
            
            return MLXSequential(*layers)
    
    def gelu(self, approximate: str = "none") -> Module:
        """Create GELU activation."""
        if approximate == "tanh":
            return MLXActivation(lambda x: nn.gelu_approx(x))
        else:
            return MLXActivation(nn.gelu)
    
    def relu(self, inplace: bool = False) -> Module:
        """Create ReLU activation."""
        return MLXActivation(nn.relu)
    
    def silu(self, inplace: bool = False) -> Module:
        """Create SiLU activation."""
        return MLXActivation(nn.silu)
    
    # Container modules
    
    def sequential(self, *modules: Module) -> Module:
        """Create sequential container."""
        return MLXSequential(*modules)
    
    def module_list(self, modules: list[Module] | None = None) -> Module:
        """Create module list."""
        return MLXModuleList(modules)
    
    def module_dict(self, modules: dict[str, Module] | None = None) -> Module:
        """Create module dictionary."""
        return MLXModuleDict(modules)
    
    # Loss functions
    
    def cross_entropy_loss(
        self,
        weight: Array | None = None,
        ignore_index: int = -100,
        reduction: str = "mean",
        label_smoothing: float = 0.0
    ) -> Callable[[Array, Array], Array]:
        """Create cross entropy loss function."""
        def loss_fn(logits: mx.array, targets: mx.array) -> mx.array:
            # Handle ignore_index by masking
            if ignore_index != -100:
                mask = targets != ignore_index
                logits = logits[mask]
                targets = targets[mask]
            
            # Apply label smoothing if needed
            if label_smoothing > 0:
                n_classes = logits.shape[-1]
                smooth_targets = mx.one_hot(targets, n_classes)
                smooth_targets = smooth_targets * (1 - label_smoothing) + label_smoothing / n_classes
                loss = -mx.sum(smooth_targets * mx.log_softmax(logits, axis=-1), axis=-1)
            else:
                loss = nn.losses.cross_entropy(logits, targets, reduction="none")
            
            # Apply class weights if provided
            if weight is not None:
                loss = loss * mx.take(weight, targets)
            
            # Apply reduction
            if reduction == "mean":
                return mx.mean(loss)
            elif reduction == "sum":
                return mx.sum(loss)
            else:
                return loss
        
        return loss_fn
    
    def binary_cross_entropy_loss(
        self,
        weight: Array | None = None,
        reduction: str = "mean"
    ) -> Callable[[Array, Array], Array]:
        """Create binary cross entropy loss."""
        def loss_fn(predictions: mx.array, targets: mx.array) -> mx.array:
            loss = nn.losses.binary_cross_entropy(predictions, targets, reduction="none")
            
            if weight is not None:
                loss = loss * weight
            
            if reduction == "mean":
                return mx.mean(loss)
            elif reduction == "sum":
                return mx.sum(loss)
            else:
                return loss
        
        return loss_fn
    
    def mse_loss(
        self,
        reduction: str = "mean"
    ) -> Callable[[Array, Array], Array]:
        """Create MSE loss."""
        def loss_fn(predictions: mx.array, targets: mx.array) -> mx.array:
            loss = nn.losses.mse_loss(predictions, targets, reduction=reduction)
            return loss
        
        return loss_fn
    
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
        if loss_type == LossType.CROSS_ENTROPY:
            return self.cross_entropy_loss(
                reduction=kwargs.get("reduction", "mean"),
                ignore_index=kwargs.get("ignore_index", -100),
                label_smoothing=kwargs.get("label_smoothing", 0.0),
                weight=kwargs.get("weight", None)
            )
        elif loss_type == LossType.MSE:
            return self.mse_loss(reduction=kwargs.get("reduction", "mean"))
        elif loss_type == LossType.MAE:
            def mae_loss_fn(predictions: mx.array, targets: mx.array) -> mx.array:
                loss = mx.abs(predictions - targets)
                reduction = kwargs.get("reduction", "mean")
                if reduction == "mean":
                    return mx.mean(loss)
                elif reduction == "sum":
                    return mx.sum(loss)
                else:
                    return loss
            return mae_loss_fn
        elif loss_type == LossType.HUBER:
            def huber_loss_fn(predictions: mx.array, targets: mx.array) -> mx.array:
                delta = kwargs.get("delta", 1.0)
                diff = mx.abs(predictions - targets)
                loss = mx.where(
                    diff < delta,
                    0.5 * diff ** 2,
                    delta * (diff - 0.5 * delta)
                )
                reduction = kwargs.get("reduction", "mean")
                if reduction == "mean":
                    return mx.mean(loss)
                elif reduction == "sum":
                    return mx.sum(loss)
                else:
                    return loss
            return huber_loss_fn
        elif loss_type == LossType.COSINE_SIMILARITY:
            def cosine_loss_fn(predictions: mx.array, targets: mx.array) -> mx.array:
                dot_product = mx.sum(predictions * targets, axis=-1)
                norm_pred = mx.linalg.norm(predictions, axis=-1)
                norm_target = mx.linalg.norm(targets, axis=-1)
                similarity = dot_product / (norm_pred * norm_target + 1e-8)
                loss = 1.0 - similarity
                reduction = kwargs.get("reduction", "mean")
                if reduction == "mean":
                    return mx.mean(loss)
                elif reduction == "sum":
                    return mx.sum(loss)
                else:
                    return loss
            return cosine_loss_fn
        else:
            raise ValueError(f"Unsupported loss type: {loss_type}")
    
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
        if encoding_type == PositionalEncoding.SINUSOIDAL:
            # Create sinusoidal position encoding
            position = mx.arange(max_length)[:, None]
            div_term = mx.exp(mx.arange(0, dim, 2) * -(mx.log(mx.array(10000.0)) / dim))
            
            # Create sin and cos components
            sin_part = mx.sin(position * div_term)
            cos_part = mx.cos(position * div_term)
            
            # Interleave sin and cos
            pe_parts = []
            for i in range(dim // 2):
                pe_parts.append(sin_part[:, i:i+1])
                pe_parts.append(cos_part[:, i:i+1])
            
            # Handle odd dimensions
            if dim % 2 == 1:
                pe_parts.append(mx.zeros((max_length, 1)))
            
            pe = mx.concatenate(pe_parts, axis=1)
            return pe
        elif encoding_type == PositionalEncoding.LEARNED:
            # Create learnable position embeddings
            return MLXEmbedding(max_length, dim)
        elif encoding_type == PositionalEncoding.ROPE:
            # Create RoPE embeddings
            return MLXRotaryEmbedding(dim)
        elif encoding_type == PositionalEncoding.ALIBI:
            print("Warning: ALiBi position encoding not implemented in MLX")
            # Fall back to sinusoidal
            position = mx.arange(max_length)[:, None]
            div_term = mx.exp(mx.arange(0, dim, 2) * -(mx.log(mx.array(10000.0)) / dim))
            
            # Create sin and cos components
            sin_part = mx.sin(position * div_term)
            cos_part = mx.cos(position * div_term)
            
            # Interleave sin and cos
            pe_parts = []
            for i in range(dim // 2):
                pe_parts.append(sin_part[:, i:i+1])
                pe_parts.append(cos_part[:, i:i+1])
            
            # Handle odd dimensions
            if dim % 2 == 1:
                pe_parts.append(mx.zeros((max_length, 1)))
            
            pe = mx.concatenate(pe_parts, axis=1)
            return pe
        elif encoding_type == PositionalEncoding.NONE:
            # Return zeros
            return mx.zeros((max_length, dim))
        else:
            raise ValueError(f"Unsupported position encoding type: {encoding_type}")
    
    # Tensor operations
    
    def matmul(self, a: Array, b: Array) -> Array:
        """Matrix multiplication."""
        return mx.matmul(a, b)
    
    def transpose(self, input: Array, dim0: int, dim1: int) -> Array:
        """Transpose dimensions."""
        # MLX uses axis-based transpose
        axes = list(range(len(input.shape)))
        axes[dim0], axes[dim1] = axes[dim1], axes[dim0]
        return mx.transpose(input, axes)
    
    def reshape(self, input: Array, shape: Shape) -> Array:
        """Reshape array."""
        return mx.reshape(input, shape)
    
    def concat(self, arrays: Sequence[Array], dim: int = 0) -> Array:
        """Concatenate arrays."""
        return mx.concatenate(arrays, axis=dim)
    
    def split(
        self,
        input: Array,
        split_size_or_sections: int | list[int],
        dim: int = 0
    ) -> list[Array]:
        """Split array."""
        if isinstance(split_size_or_sections, int):
            # Split into equal chunks
            n_sections = input.shape[dim] // split_size_or_sections
            return mx.split(input, n_sections, axis=dim)
        else:
            # Split by section sizes
            indices = np.cumsum(split_size_or_sections[:-1])
            return mx.split(input, indices, axis=dim)
    
    def mean(
        self,
        input: Array,
        dim: int | tuple[int, ...] | None = None,
        keepdim: bool = False
    ) -> Array:
        """Compute mean."""
        return mx.mean(input, axis=dim, keepdims=keepdim)
    
    def sum(
        self,
        input: Array,
        dim: int | tuple[int, ...] | None = None,
        keepdim: bool = False
    ) -> Array:
        """Compute sum."""
        return mx.sum(input, axis=dim, keepdims=keepdim)
    
    def max(
        self,
        input: Array,
        dim: int | None = None,
        keepdim: bool = False
    ) -> Array | tuple[Array, Array]:
        """Compute maximum."""
        if dim is None:
            return mx.max(input)
        else:
            values = mx.max(input, axis=dim, keepdims=keepdim)
            indices = mx.argmax(input, axis=dim, keepdims=keepdim)
            return values, indices
    
    def min(
        self,
        input: Array,
        dim: int | None = None,
        keepdim: bool = False
    ) -> Array | tuple[Array, Array]:
        """Compute minimum."""
        if dim is None:
            return mx.min(input)
        else:
            values = mx.min(input, axis=dim, keepdims=keepdim)
            indices = mx.argmin(input, axis=dim, keepdims=keepdim)
            return values, indices
    
    def softmax(self, input: Array, dim: int = -1) -> Array:
        """Apply softmax."""
        return mx.softmax(input, axis=dim)
    
    def log_softmax(self, input: Array, dim: int = -1) -> Array:
        """Apply log softmax."""
        # MLX doesn't have log_softmax, so compute it manually
        return input - mx.logsumexp(input, axis=dim, keepdims=True)
    
    # Advanced operations
    
    def rotary_embedding(
        self,
        dim: int,
        max_position_embeddings: int = 8192,
        base: float = 10000.0,
        scaling_factor: float = 1.0,
        dtype: DType | None = None
    ) -> Module:
        """Create rotary embeddings module."""
        return MLXRotaryEmbedding(
            dims=dim,
            traditional=False,
            base=base,
            scale=scaling_factor
        )
    
    def apply_rotary_pos_emb(
        self,
        q: Array,
        k: Array,
        cos: Array,
        sin: Array,
        position_ids: Array | None = None
    ) -> tuple[Array, Array]:
        """Apply rotary position embeddings."""
        # Get shapes
        batch_size, num_heads, seq_len, head_dim = q.shape
        
        # Ensure cos/sin have the right shape for broadcasting
        # cos/sin shape: [seq_len, 1, head_dim//2]
        # We need to broadcast to [batch_size, num_heads, seq_len, head_dim//2]
        cos = mx.expand_dims(cos, 0)  # [1, seq_len, 1, head_dim//2]
        sin = mx.expand_dims(sin, 0)  # [1, seq_len, 1, head_dim//2]
        
        # Broadcast to match q/k shape
        cos = mx.broadcast_to(cos, (batch_size, seq_len, num_heads, head_dim // 2))
        sin = mx.broadcast_to(sin, (batch_size, seq_len, num_heads, head_dim // 2))
        
        # Transpose to match q/k layout [batch, heads, seq, dim]
        cos = mx.transpose(cos, (0, 2, 1, 3))
        sin = mx.transpose(sin, (0, 2, 1, 3))
        
        # Split last dimension in half for rotation
        q1, q2 = mx.split(q, 2, axis=-1)
        k1, k2 = mx.split(k, 2, axis=-1)
        
        # Apply rotation
        q_rotated = mx.concatenate([q1 * cos - q2 * sin, q1 * sin + q2 * cos], axis=-1)
        k_rotated = mx.concatenate([k1 * cos - k2 * sin, k1 * sin + k2 * cos], axis=-1)
        
        return q_rotated, k_rotated
    
    def masked_fill(
        self,
        input: Array,
        mask: Array,
        value: float
    ) -> Array:
        """Fill masked positions."""
        # MLX where expects condition to broadcast properly
        return mx.where(mask, mx.full(input.shape, value), input)
    
    def where(
        self,
        condition: Array,
        x: Array | float,
        y: Array | float
    ) -> Array:
        """Conditional selection."""
        return mx.where(condition, x, y)
    
    # Utility methods
    
    def parameter(
        self,
        data: ArrayLike,
        requires_grad: bool = True,
        dtype: DType | None = None
    ) -> Parameter:
        """Create parameter."""
        if isinstance(data, mx.array):
            return data
        else:
            return mx.array(data, dtype=dtype)
    
    @contextmanager
    def no_grad(self):
        """Context for no gradient computation."""
        # MLX doesn't have a direct no_grad context
        # Operations are only traced when used in grad functions
        yield
    
    @contextmanager
    def enable_grad(self):
        """Context for enabling gradients."""
        # MLX always computes gradients when requested
        yield
    
    @contextmanager  
    def device_context(self, device: Device):
        """Context for device placement."""
        # MLX automatically handles device placement
        yield
    
    def unsqueeze(self, input: Array, dim: int) -> Array:
        """Add dimension."""
        return mx.expand_dims(input, axis=dim)
    
    def arange(
        self,
        start: int | float = 0,
        stop: int | float | None = None,
        step: int | float = 1,
        dtype: DType | None = None,
        device: Device | None = None
    ) -> Array:
        """Create range array."""
        if stop is None:
            stop = start
            start = 0
        return mx.arange(start, stop, step, dtype=dtype)
    
    def broadcast_to(self, input: Array, shape: Shape) -> Array:
        """Broadcast to shape."""
        return mx.broadcast_to(input, shape)
    
    def zeros_like(
        self,
        input: Array,
        dtype: DType | None = None,
        device: Device | None = None
    ) -> Array:
        """Create zeros like input."""
        if dtype is None:
            return mx.zeros_like(input)
        else:
            return mx.zeros(input.shape, dtype=dtype)
    
    def ones(
        self,
        shape: Shape,
        dtype: DType | None = None,
        device: Device | None = None
    ) -> Array:
        """Create ones array."""
        return mx.ones(shape, dtype=dtype)
    
    # Module management methods
    
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
        if not isinstance(module, MLXModule):
            print(f"Warning: Cannot initialize weights for non-MLX module: {type(module)}")
            return
        
        # MLX modules are typically initialized automatically by the framework
        # For now, we'll provide a warning that initialization is handled automatically
        if init_type != InitializationType.XAVIER_UNIFORM:
            print(f"Warning: MLX modules use automatic initialization, ignoring {init_type}")
    
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
        if not isinstance(module, MLXModule):
            print(f"Warning: Cannot count parameters for non-MLX module: {type(module)}")
            return 0
        
        total_params = 0
        for param in module.parameters():
            if isinstance(param, mx.array):
                total_params += param.size
            else:
                # Fallback for other parameter types
                total_params += 1
        
        return total_params
    
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
        if not isinstance(module, MLXModule):
            print(f"Warning: Cannot freeze non-MLX module: {type(module)}")
            return
        
        # MLX doesn't have a direct freeze mechanism like PyTorch
        # This would need to be handled at the optimizer level
        print(f"Warning: MLX doesn't support direct parameter freezing, handle at optimizer level")
    
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
        # MLX automatically handles device placement
        return "mlx"
    
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
        # MLX automatically handles device placement
        if device != "mlx":
            print(f"Warning: MLX doesn't support moving to device '{device}', using MLX device")
        return module
    
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
        if mask.mask is not None:
            # Use provided mask
            return self.masked_fill(attention_scores, mask.mask, -1e9)
        
        # Generate mask based on type
        seq_len = attention_scores.shape[-1]
        
        if mask.mask_type.value == "causal":
            # Create causal mask (lower triangular)
            causal_mask = mx.triu(mx.ones((seq_len, seq_len)), k=1).astype(mx.bool_)
            return self.masked_fill(attention_scores, causal_mask, -1e9)
        elif mask.mask_type.value == "bidirectional":
            # No masking for bidirectional
            return attention_scores
        elif mask.mask_type.value == "prefix_lm":
            if mask.prefix_length is None:
                print("Warning: prefix_lm mask requires prefix_length")
                return attention_scores
            # Create prefix mask
            prefix_mask = mx.ones((seq_len, seq_len), dtype=mx.bool_)
            
            # Allow attention within prefix and from prefix to all (set to False)
            prefix_part = mx.zeros((mask.prefix_length, seq_len), dtype=mx.bool_)
            
            # Create causal mask for the non-prefix part
            remaining_len = seq_len - mask.prefix_length
            if remaining_len > 0:
                causal_part = mx.triu(mx.ones((remaining_len, remaining_len)), k=1).astype(mx.bool_)
                # Combine with zeros for the prefix columns
                prefix_zeros = mx.zeros((remaining_len, mask.prefix_length), dtype=mx.bool_)
                suffix_part = mx.concatenate([prefix_zeros, causal_part], axis=1)
            else:
                suffix_part = mx.zeros((0, seq_len), dtype=mx.bool_)
            
            # Combine prefix and suffix parts
            prefix_mask = mx.concatenate([prefix_part, suffix_part], axis=0)
            return self.masked_fill(attention_scores, prefix_mask, -1e9)
        else:
            # Custom or unknown mask type
            print(f"Warning: Unknown mask type {mask.mask_type}, no masking applied")
            return attention_scores
    
    # Methods moved from compute port to neural port
    
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
        # Use the existing cross_entropy_loss function
        loss_fn = self.cross_entropy_loss(reduction=reduction, ignore_index=ignore_index)
        return loss_fn(input, target)
    
    def attention(
        self,
        query: Array,
        key: Array,
        value: Array,
        mask: Array | None = None,
        dropout_p: float = 0.0,
        scale: float | None = None,
        training: bool = True,
    ) -> tuple[Array, Array | None]:
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
        batch_size, seq_len, d_k = query.shape
        
        # Compute attention scores
        if scale is None:
            scale = 1.0 / mx.sqrt(mx.array(d_k, dtype=query.dtype))
        
        scores = mx.matmul(query, mx.transpose(key, axes=[0, 2, 1])) * scale
        
        # Apply mask if provided
        if mask is not None:
            scores = self.masked_fill(scores, mask, -1e9)
        
        # Apply softmax
        attn_weights = mx.softmax(scores, axis=-1)
        
        # Apply dropout if in training mode
        if training and dropout_p > 0:
            dropout = MLXDropout(dropout_p)
            attn_weights = dropout(attn_weights)
        
        # Compute attention output
        attn_output = mx.matmul(attn_weights, value)
        
        return attn_output, attn_weights
    
    def swiglu(self, input: Array, gate: Array) -> Array:
        """SwiGLU activation function.
        
        Args:
            input: Input array
            gate: Gate array
            
        Returns:
            SwiGLU output
        """
        # SwiGLU = SiLU(gate) * input
        return mx.nn.silu(gate) * input
    
    def forward_pass(
        self,
        model: Module,
        inputs: dict[str, Array],
        training: bool = False,
    ) -> dict[str, Array]:
        """Perform forward pass through model.
        
        Args:
            model: Neural network model
            inputs: Input data as dictionary of arrays
            training: Whether in training mode
            
        Returns:
            Dictionary containing outputs (logits, loss, hidden_states, etc.)
        """
        # Set training mode
        model.train(training)
        
        # Ensure model is MLXModule
        if not isinstance(model, MLXModule):
            raise TypeError(f"Expected MLXModule, got {type(model)}")
        
        # Forward pass
        outputs = model(**inputs)
        
        # Ensure outputs is a dictionary
        if not isinstance(outputs, dict):
            outputs = {"output": outputs}
        
        return outputs
    
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
        if loss_type == LossType.CROSS_ENTROPY:
            return self.cross_entropy(predictions, targets, **kwargs)
        elif loss_type == LossType.MSE:
            loss_fn = self.mse_loss(**kwargs)
            return loss_fn(predictions, targets)
        elif loss_type == LossType.MAE:
            # MAE loss
            return mx.mean(mx.abs(predictions - targets))
        elif loss_type == LossType.HUBER:
            # Huber loss
            delta = kwargs.get("delta", 1.0)
            diff = mx.abs(predictions - targets)
            return mx.mean(
                mx.where(
                    diff < delta,
                    0.5 * diff ** 2,
                    delta * (diff - 0.5 * delta)
                )
            )
        elif loss_type == LossType.COSINE_SIMILARITY:
            # Cosine similarity loss
            dot_product = mx.sum(predictions * targets, axis=-1)
            norm_pred = mx.linalg.norm(predictions, axis=-1)
            norm_target = mx.linalg.norm(targets, axis=-1)
            return 1.0 - mx.mean(dot_product / (norm_pred * norm_target + 1e-8))
        else:
            raise ValueError(f"Unsupported loss type: {loss_type}")
    
    def backward_pass(
        self,
        loss: Array,
        model: Module,
        retain_graph: bool = False,
    ) -> dict[str, Array]:
        """Perform backward pass to compute gradients.
        
        Args:
            loss: Loss value to backpropagate
            model: Neural network model
            retain_graph: Whether to retain computation graph
            
        Returns:
            Dictionary of gradients
        """
        # In MLX, gradients are computed using mx.grad
        # This requires the forward pass to be wrapped in a function
        # For now, we'll return a placeholder
        # The actual gradient computation should be done in optimize_step
        # using mx.value_and_grad
        return {}
    
    def optimize_step(
        self,
        model_params: dict[str, Array],
        gradients: dict[str, Array],
        optimizer_state: dict[str, Any],
        learning_rate: float,
        **kwargs: Any,
    ) -> tuple[dict[str, Array], dict[str, Any]]:
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
        # Get optimizer type
        optimizer_type = optimizer_state.get("type", "adamw")
        
        # Create optimizer if not in state
        if "optimizer" not in optimizer_state:
            if optimizer_type == "adam":
                optimizer = optim.Adam(learning_rate=learning_rate)
            elif optimizer_type == "adamw":
                optimizer = optim.AdamW(learning_rate=learning_rate)
            elif optimizer_type == "sgd":
                optimizer = optim.SGD(learning_rate=learning_rate)
            else:
                raise ValueError(f"Unsupported optimizer type: {optimizer_type}")
            
            optimizer_state["optimizer"] = optimizer
        else:
            optimizer = optimizer_state["optimizer"]
            # Update learning rate
            optimizer.learning_rate = learning_rate
        
        # Apply gradients
        updated_params = optimizer.apply_gradients(gradients, model_params)
        
        return updated_params, optimizer_state