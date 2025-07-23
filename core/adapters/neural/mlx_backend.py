"""MLX adapter implementation for the neural network port.

This module provides a complete MLX implementation of the NeuralBackend protocol,
enabling framework-agnostic neural network operations using Apple's MLX framework.
"""

from contextlib import contextmanager
from typing import Any, Callable, Iterator, Sequence

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np

from core.ports.compute import Array, ArrayLike, DataType, Device, DType, Shape
from core.ports.neural import (
    ActivationType,
    GradientDict,
    LossType,
    Module,
    ModuleInfo,
    NeuralBackend,
    NormalizationType,
    Parameter,
    ParameterDict,
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
    
    def linear(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        dtype: DType | None = None
    ) -> Module:
        """Create a linear layer."""
        return MLXLinear(in_features, out_features, bias=bias)
    
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
        """Create an embedding layer."""
        # Note: MLX doesn't support all PyTorch embedding features
        if padding_idx is not None:
            print(f"Warning: MLX embeddings don't support padding_idx")
        if max_norm is not None:
            print(f"Warning: MLX embeddings don't support max_norm")
        if scale_grad_by_freq:
            print(f"Warning: MLX embeddings don't support scale_grad_by_freq")
        if sparse:
            print(f"Warning: MLX embeddings don't support sparse gradients")
            
        return MLXEmbedding(num_embeddings, embedding_dim)
    
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
    
    def dropout(
        self,
        p: float = 0.5,
        inplace: bool = False
    ) -> Module:
        """Create dropout module."""
        if inplace:
            print("Warning: MLX dropout doesn't support inplace operations")
        return MLXDropout(p=p)
    
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
        """Create multi-head attention."""
        if add_bias_kv:
            print("Warning: MLX MHA doesn't support add_bias_kv")
        if add_zero_attn:
            print("Warning: MLX MHA doesn't support add_zero_attn")
        if not batch_first:
            print("Warning: MLX MHA expects batch_first format")
            
        return MLXMultiHeadAttention(
            dims=embed_dim,
            num_heads=num_heads,
            query_input_dims=embed_dim,
            key_input_dims=kdim or embed_dim,
            value_input_dims=vdim or embed_dim,
            bias=bias
        )
    
    # Activation functions
    
    def activation(
        self,
        activation_type: ActivationType,
        **kwargs
    ) -> Module:
        """Create an activation module."""
        if activation_type == ActivationType.RELU:
            return self.relu(**kwargs)
        elif activation_type == ActivationType.GELU:
            return self.gelu(**kwargs)
        elif activation_type == ActivationType.SILU:
            return self.silu(**kwargs)
        elif activation_type == ActivationType.TANH:
            return MLXActivation(mx.tanh)
        elif activation_type == ActivationType.SIGMOID:
            return MLXActivation(mx.sigmoid)
        elif activation_type == ActivationType.LEAKY_RELU:
            negative_slope = kwargs.get("negative_slope", 0.01)
            return MLXActivation(lambda x: nn.leaky_relu(x, negative_slope))
        elif activation_type == ActivationType.ELU:
            alpha = kwargs.get("alpha", 1.0)
            return MLXActivation(lambda x: nn.elu(x, alpha))
        elif activation_type == ActivationType.SOFTMAX:
            dim = kwargs.get("dim", -1)
            return MLXActivation(lambda x: mx.softmax(x, axis=dim))
        elif activation_type == ActivationType.LOG_SOFTMAX:
            dim = kwargs.get("dim", -1)
            return MLXActivation(lambda x: mx.log_softmax(x, axis=dim))
        else:
            raise ValueError(f"Unsupported activation type: {activation_type}")
    
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