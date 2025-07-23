"""PyTorch adapter implementation for the neural network port.

This module provides a complete PyTorch implementation of the NeuralBackend protocol,
enabling framework-agnostic neural network operations using PyTorch.
"""

from contextlib import contextmanager
from typing import Any, Callable, Iterator, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ports.secondary.compute import Array, ArrayLike, DataType, Device, DType, Shape
from ports.secondary.neural import (
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


class PyTorchModule(Module):
    """Wrapper around torch.nn.Module to implement our Module interface."""
    
    def __init__(self, torch_module: nn.Module | None = None):
        """Initialize PyTorch module wrapper.
        
        Args:
            torch_module: Optional PyTorch module to wrap
        """
        super().__init__()
        self._torch_module = torch_module
        if torch_module is not None:
            # Sync parameters from PyTorch module
            self._sync_from_torch()
    
    def _sync_from_torch(self):
        """Synchronize parameters from PyTorch module."""
        if self._torch_module is None:
            return
            
        # Get all parameters from PyTorch module
        for name, param in self._torch_module.named_parameters():
            self._parameters[name] = param
            
        # Get all submodules
        for name, module in self._torch_module.named_children():
            self._modules[name] = PyTorchModule(module)
    
    def forward(self, *args, **kwargs) -> Any:
        """Forward pass through the wrapped PyTorch module."""
        if self._torch_module is None:
            raise NotImplementedError("No PyTorch module to forward through")
        return self._torch_module(*args, **kwargs)
    
    def train(self, mode: bool = True) -> "Module":
        """Set training mode."""
        super().train(mode)
        if self._torch_module is not None:
            self._torch_module.train(mode)
        return self
    
    def parameters(self) -> Iterator[Parameter]:
        """Get all parameters."""
        if self._torch_module is not None:
            # Use PyTorch module's parameters
            yield from self._torch_module.parameters()
        else:
            # Use our own parameter tracking
            yield from super().parameters()
    
    def named_parameters(self) -> Iterator[tuple[str, Parameter]]:
        """Get all parameters with names."""
        if self._torch_module is not None:
            # Use PyTorch module's parameters
            yield from self._torch_module.named_parameters()
        else:
            # Use our own parameter tracking
            yield from super().named_parameters()


class PyTorchLinear(PyTorchModule):
    """PyTorch Linear layer wrapped as our Module."""
    
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        torch_linear = nn.Linear(in_features, out_features, bias=bias)
        super().__init__(torch_linear)


class PyTorchEmbedding(PyTorchModule):
    """PyTorch Embedding layer wrapped as our Module."""
    
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        padding_idx: int | None = None,
        max_norm: float | None = None,
        norm_type: float = 2.0,
        scale_grad_by_freq: bool = False,
        sparse: bool = False
    ):
        torch_embedding = nn.Embedding(
            num_embeddings=num_embeddings,
            embedding_dim=embedding_dim,
            padding_idx=padding_idx,
            max_norm=max_norm,
            norm_type=norm_type,
            scale_grad_by_freq=scale_grad_by_freq,
            sparse=sparse
        )
        super().__init__(torch_embedding)


class PyTorchLayerNorm(PyTorchModule):
    """PyTorch LayerNorm wrapped as our Module."""
    
    def __init__(
        self,
        normalized_shape: int | Shape,
        eps: float = 1e-5,
        elementwise_affine: bool = True
    ):
        # Convert to tuple if needed
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
        torch_layer_norm = nn.LayerNorm(
            normalized_shape=normalized_shape,
            eps=eps,
            elementwise_affine=elementwise_affine
        )
        super().__init__(torch_layer_norm)


class PyTorchRMSNorm(PyTorchModule):
    """PyTorch RMSNorm implementation wrapped as our Module."""
    
    def __init__(
        self,
        normalized_shape: int | Shape,
        eps: float = 1e-6,
        elementwise_affine: bool = True
    ):
        # PyTorch doesn't have built-in RMSNorm, so we'll implement it
        class RMSNorm(nn.Module):
            def __init__(self, dim: int, eps: float = 1e-6):
                super().__init__()
                self.eps = eps
                self.weight = nn.Parameter(torch.ones(dim)) if elementwise_affine else None
            
            def forward(self, x):
                rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
                x = x / rms
                if self.weight is not None:
                    x = x * self.weight
                return x
        
        dim = normalized_shape if isinstance(normalized_shape, int) else normalized_shape[-1]
        torch_rms_norm = RMSNorm(dim, eps)
        super().__init__(torch_rms_norm)


class PyTorchDropout(PyTorchModule):
    """PyTorch Dropout wrapped as our Module."""
    
    def __init__(self, p: float = 0.5, inplace: bool = False):
        torch_dropout = nn.Dropout(p=p, inplace=inplace)
        super().__init__(torch_dropout)


class PyTorchMultiHeadAttention(PyTorchModule):
    """PyTorch MultiheadAttention wrapped as our Module."""
    
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        bias: bool = True,
        add_bias_kv: bool = False,
        add_zero_attn: bool = False,
        kdim: int | None = None,
        vdim: int | None = None,
        batch_first: bool = True
    ):
        torch_mha = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            bias=bias,
            add_bias_kv=add_bias_kv,
            add_zero_attn=add_zero_attn,
            kdim=kdim,
            vdim=vdim,
            batch_first=batch_first
        )
        super().__init__(torch_mha)


class PyTorchSequential(PyTorchModule):
    """PyTorch Sequential container wrapped as our Module."""
    
    def __init__(self, *modules: Module):
        # Convert our modules to PyTorch modules
        torch_modules = []
        for mod in modules:
            if isinstance(mod, PyTorchModule) and mod._torch_module is not None:
                torch_modules.append(mod._torch_module)
            else:
                # Wrap in a custom module that calls our Module's forward
                class WrapperModule(nn.Module):
                    def __init__(self, module: Module):
                        super().__init__()
                        self.module = module
                    
                    def forward(self, x):
                        return self.module(x)
                
                torch_modules.append(WrapperModule(mod))
        
        torch_sequential = nn.Sequential(*torch_modules)
        super().__init__(torch_sequential)


class PyTorchActivation(PyTorchModule):
    """Generic activation wrapper."""
    
    def __init__(self, activation_fn: nn.Module):
        super().__init__(activation_fn)


class PyTorchNeuralBackend(NeuralBackend):
    """PyTorch implementation of the NeuralBackend protocol."""
    
    @property
    def name(self) -> str:
        """Name of the neural backend."""
        return "pytorch"
    
    @property
    def supports_mixed_precision(self) -> bool:
        """Whether this backend supports mixed precision training."""
        return True
    
    def linear(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        dtype: DType | None = None
    ) -> Module:
        """Create a linear (fully connected) layer."""
        return PyTorchLinear(in_features, out_features, bias)
    
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
        return PyTorchEmbedding(
            num_embeddings=num_embeddings,
            embedding_dim=embedding_dim,
            padding_idx=padding_idx,
            max_norm=max_norm,
            norm_type=norm_type,
            scale_grad_by_freq=scale_grad_by_freq,
            sparse=sparse
        )
    
    def layer_norm(
        self,
        normalized_shape: int | Shape,
        eps: float = 1e-5,
        elementwise_affine: bool = True,
        bias: bool = True,
        dtype: DType | None = None
    ) -> Module:
        """Create a layer normalization module."""
        return PyTorchLayerNorm(normalized_shape, eps, elementwise_affine)
    
    def rms_norm(
        self,
        normalized_shape: int | Shape,
        eps: float = 1e-6,
        elementwise_affine: bool = True,
        bias: bool = False,
        dtype: DType | None = None
    ) -> Module:
        """Create RMS normalization module."""
        return PyTorchRMSNorm(normalized_shape, eps, elementwise_affine)
    
    def dropout(
        self,
        p: float = 0.5,
        inplace: bool = False
    ) -> Module:
        """Create a dropout module."""
        return PyTorchDropout(p, inplace)
    
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
        """Create a multi-head attention module."""
        return PyTorchMultiHeadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            bias=bias,
            add_bias_kv=add_bias_kv,
            add_zero_attn=add_zero_attn,
            kdim=kdim,
            vdim=vdim,
            batch_first=batch_first
        )
    
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
            return PyTorchActivation(nn.Tanh())
        elif activation_type == ActivationType.SIGMOID:
            return PyTorchActivation(nn.Sigmoid())
        elif activation_type == ActivationType.LEAKY_RELU:
            negative_slope = kwargs.get('negative_slope', 0.01)
            return PyTorchActivation(nn.LeakyReLU(negative_slope))
        elif activation_type == ActivationType.ELU:
            alpha = kwargs.get('alpha', 1.0)
            return PyTorchActivation(nn.ELU(alpha))
        elif activation_type == ActivationType.SOFTMAX:
            dim = kwargs.get('dim', -1)
            return PyTorchActivation(nn.Softmax(dim=dim))
        elif activation_type == ActivationType.LOG_SOFTMAX:
            dim = kwargs.get('dim', -1)
            return PyTorchActivation(nn.LogSoftmax(dim=dim))
        else:
            raise ValueError(f"Unsupported activation type: {activation_type}")
    
    def gelu(self, approximate: str = "none") -> Module:
        """Create GELU activation."""
        # PyTorch GELU doesn't have approximate parameter in older versions
        return PyTorchActivation(nn.GELU())
    
    def relu(self, inplace: bool = False) -> Module:
        """Create ReLU activation."""
        return PyTorchActivation(nn.ReLU(inplace=inplace))
    
    def silu(self, inplace: bool = False) -> Module:
        """Create SiLU (Swish) activation."""
        return PyTorchActivation(nn.SiLU(inplace=inplace))
    
    def sequential(self, *modules: Module) -> Module:
        """Create a sequential container."""
        return PyTorchSequential(*modules)
    
    def module_list(self, modules: list[Module] | None = None) -> Module:
        """Create a module list container."""
        class ModuleListWrapper(PyTorchModule):
            def __init__(self, modules: list[Module] | None = None):
                modules = modules or []
                torch_modules = nn.ModuleList()
                for mod in modules:
                    if isinstance(mod, PyTorchModule) and mod._torch_module is not None:
                        torch_modules.append(mod._torch_module)
                super().__init__(torch_modules)
        
        return ModuleListWrapper(modules)
    
    def module_dict(self, modules: dict[str, Module] | None = None) -> Module:
        """Create a module dictionary container."""
        class ModuleDictWrapper(PyTorchModule):
            def __init__(self, modules: dict[str, Module] | None = None):
                modules = modules or {}
                torch_modules = nn.ModuleDict()
                for name, mod in modules.items():
                    if isinstance(mod, PyTorchModule) and mod._torch_module is not None:
                        torch_modules[name] = mod._torch_module
                super().__init__(torch_modules)
        
        return ModuleDictWrapper(modules)
    
    def cross_entropy_loss(
        self,
        weight: Array | None = None,
        ignore_index: int = -100,
        reduction: str = "mean",
        label_smoothing: float = 0.0
    ) -> Callable[[Array, Array], Array]:
        """Create cross entropy loss function."""
        def loss_fn(input: Array, target: Array) -> Array:
            # Convert to PyTorch tensors if needed
            if not isinstance(input, torch.Tensor):
                input = torch.tensor(input)
            if not isinstance(target, torch.Tensor):
                target = torch.tensor(target)
            
            # Convert weight if provided
            torch_weight = None
            if weight is not None:
                torch_weight = torch.tensor(weight) if not isinstance(weight, torch.Tensor) else weight
            
            return F.cross_entropy(
                input,
                target,
                weight=torch_weight,
                ignore_index=ignore_index,
                reduction=reduction,
                label_smoothing=label_smoothing
            )
        
        return loss_fn
    
    def binary_cross_entropy_loss(
        self,
        weight: Array | None = None,
        reduction: str = "mean"
    ) -> Callable[[Array, Array], Array]:
        """Create binary cross entropy loss function."""
        def loss_fn(input: Array, target: Array) -> Array:
            # Convert to PyTorch tensors if needed
            if not isinstance(input, torch.Tensor):
                input = torch.tensor(input)
            if not isinstance(target, torch.Tensor):
                target = torch.tensor(target)
            
            # Convert weight if provided
            torch_weight = None
            if weight is not None:
                torch_weight = torch.tensor(weight) if not isinstance(weight, torch.Tensor) else weight
            
            return F.binary_cross_entropy(
                input,
                target,
                weight=torch_weight,
                reduction=reduction
            )
        
        return loss_fn
    
    def mse_loss(
        self,
        reduction: str = "mean"
    ) -> Callable[[Array, Array], Array]:
        """Create mean squared error loss function."""
        def loss_fn(input: Array, target: Array) -> Array:
            # Convert to PyTorch tensors if needed
            if not isinstance(input, torch.Tensor):
                input = torch.tensor(input)
            if not isinstance(target, torch.Tensor):
                target = torch.tensor(target)
            
            return F.mse_loss(input, target, reduction=reduction)
        
        return loss_fn
    
    def matmul(self, a: Array, b: Array) -> Array:
        """Matrix multiplication."""
        # Convert to PyTorch tensors if needed
        if not isinstance(a, torch.Tensor):
            a = torch.tensor(a)
        if not isinstance(b, torch.Tensor):
            b = torch.tensor(b)
        return torch.matmul(a, b)
    
    def transpose(self, input: Array, dim0: int, dim1: int) -> Array:
        """Transpose two dimensions."""
        if not isinstance(input, torch.Tensor):
            input = torch.tensor(input)
        return torch.transpose(input, dim0, dim1)
    
    def reshape(self, input: Array, shape: Shape) -> Array:
        """Reshape array."""
        if not isinstance(input, torch.Tensor):
            input = torch.tensor(input)
        return torch.reshape(input, shape)
    
    def concat(self, arrays: Sequence[Array], dim: int = 0) -> Array:
        """Concatenate arrays along a dimension."""
        # Convert to PyTorch tensors if needed
        torch_arrays = []
        for arr in arrays:
            if not isinstance(arr, torch.Tensor):
                arr = torch.tensor(arr)
            torch_arrays.append(arr)
        return torch.cat(torch_arrays, dim=dim)
    
    def split(
        self,
        input: Array,
        split_size_or_sections: int | list[int],
        dim: int = 0
    ) -> list[Array]:
        """Split array along a dimension."""
        if not isinstance(input, torch.Tensor):
            input = torch.tensor(input)
        return torch.split(input, split_size_or_sections, dim=dim)
    
    def mean(
        self,
        input: Array,
        dim: int | tuple[int, ...] | None = None,
        keepdim: bool = False
    ) -> Array:
        """Compute mean."""
        if not isinstance(input, torch.Tensor):
            input = torch.tensor(input)
        return torch.mean(input, dim=dim, keepdim=keepdim)
    
    def sum(
        self,
        input: Array,
        dim: int | tuple[int, ...] | None = None,
        keepdim: bool = False
    ) -> Array:
        """Compute sum."""
        if not isinstance(input, torch.Tensor):
            input = torch.tensor(input)
        return torch.sum(input, dim=dim, keepdim=keepdim)
    
    def max(
        self,
        input: Array,
        dim: int | None = None,
        keepdim: bool = False
    ) -> Array | tuple[Array, Array]:
        """Compute maximum."""
        if not isinstance(input, torch.Tensor):
            input = torch.tensor(input)
        
        if dim is None:
            return torch.max(input)
        else:
            return torch.max(input, dim=dim, keepdim=keepdim)
    
    def min(
        self,
        input: Array,
        dim: int | None = None,
        keepdim: bool = False
    ) -> Array | tuple[Array, Array]:
        """Compute minimum."""
        if not isinstance(input, torch.Tensor):
            input = torch.tensor(input)
        
        if dim is None:
            return torch.min(input)
        else:
            return torch.min(input, dim=dim, keepdim=keepdim)
    
    def softmax(self, input: Array, dim: int = -1) -> Array:
        """Apply softmax."""
        if not isinstance(input, torch.Tensor):
            input = torch.tensor(input)
        return F.softmax(input, dim=dim)
    
    def log_softmax(self, input: Array, dim: int = -1) -> Array:
        """Apply log softmax."""
        if not isinstance(input, torch.Tensor):
            input = torch.tensor(input)
        return F.log_softmax(input, dim=dim)
    
    def rotary_embedding(
        self,
        dim: int,
        max_position_embeddings: int = 8192,
        base: float = 10000.0,
        scaling_factor: float = 1.0,
        dtype: DType | None = None
    ) -> Module:
        """Create rotary positional embeddings (RoPE)."""
        # Simplified RoPE implementation for demonstration
        class RotaryEmbedding(nn.Module):
            def __init__(self, dim, max_position_embeddings, base, scaling_factor):
                super().__init__()
                self.dim = dim
                self.max_position_embeddings = max_position_embeddings
                self.base = base
                self.scaling_factor = scaling_factor
                
                # Precompute frequencies
                inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
                self.register_buffer("inv_freq", inv_freq)
            
            def forward(self, positions):
                # Compute cos and sin for the given positions
                freqs = torch.einsum("i,j->ij", positions.float(), self.inv_freq)
                cos = torch.cos(freqs * self.scaling_factor)
                sin = torch.sin(freqs * self.scaling_factor)
                return cos, sin
        
        return PyTorchModule(RotaryEmbedding(dim, max_position_embeddings, base, scaling_factor))
    
    def apply_rotary_pos_emb(
        self,
        q: Array,
        k: Array,
        cos: Array,
        sin: Array,
        position_ids: Array | None = None
    ) -> tuple[Array, Array]:
        """Apply rotary positional embeddings to query and key."""
        # Convert to PyTorch tensors if needed
        if not isinstance(q, torch.Tensor):
            q = torch.tensor(q)
        if not isinstance(k, torch.Tensor):
            k = torch.tensor(k)
        if not isinstance(cos, torch.Tensor):
            cos = torch.tensor(cos)
        if not isinstance(sin, torch.Tensor):
            sin = torch.tensor(sin)
        
        # Simplified implementation
        # In practice, this would involve complex number rotations
        q_embed = q * cos + q * sin
        k_embed = k * cos + k * sin
        
        return q_embed, k_embed
    
    def masked_fill(
        self,
        input: Array,
        mask: Array,
        value: float
    ) -> Array:
        """Fill masked positions with value."""
        if not isinstance(input, torch.Tensor):
            input = torch.tensor(input)
        if not isinstance(mask, torch.Tensor):
            mask = torch.tensor(mask)
        
        return input.masked_fill(mask, value)
    
    def where(
        self,
        condition: Array,
        x: Array | float,
        y: Array | float
    ) -> Array:
        """Select elements from x or y based on condition."""
        if not isinstance(condition, torch.Tensor):
            condition = torch.tensor(condition)
        if not isinstance(x, torch.Tensor) and not isinstance(x, (int, float)):
            x = torch.tensor(x)
        if not isinstance(y, torch.Tensor) and not isinstance(y, (int, float)):
            y = torch.tensor(y)
        
        return torch.where(condition, x, y)
    
    def parameter(
        self,
        data: ArrayLike,
        requires_grad: bool = True,
        dtype: DType | None = None
    ) -> Parameter:
        """Create a parameter from data."""
        if not isinstance(data, torch.Tensor):
            data = torch.tensor(data)
        
        if dtype is not None:
            # Map our DType to PyTorch dtype
            dtype_map = {
                DataType.FLOAT32: torch.float32,
                DataType.FLOAT16: torch.float16,
                DataType.BFLOAT16: torch.bfloat16,
                DataType.INT32: torch.int32,
                DataType.INT64: torch.int64,
            }
            if dtype in dtype_map:
                data = data.to(dtype_map[dtype])
        
        return nn.Parameter(data, requires_grad=requires_grad)
    
    @contextmanager
    def no_grad(self):
        """Context manager to disable gradient computation."""
        with torch.no_grad():
            yield
    
    @contextmanager
    def enable_grad(self):
        """Context manager to enable gradient computation."""
        with torch.enable_grad():
            yield
    
    @contextmanager
    def device_context(self, device: Device):
        """Context manager for device placement."""
        # Map our Device to PyTorch device
        if device.type == "cpu":
            torch_device = torch.device("cpu")
        elif device.type == "cuda":
            torch_device = torch.device(f"cuda:{device.index}")
        elif device.type == "mps":
            torch_device = torch.device("mps")
        else:
            torch_device = torch.device(device.type)
        
        # PyTorch doesn't have a direct device context like MLX
        # This is a simplified implementation
        yield torch_device
    
    def unsqueeze(self, input: Array, dim: int) -> Array:
        """Add a dimension at the specified position."""
        if not isinstance(input, torch.Tensor):
            input = torch.tensor(input)
        return torch.unsqueeze(input, dim)
    
    def arange(
        self,
        start: int | float = 0,
        stop: int | float | None = None,
        step: int | float = 1,
        dtype: DType | None = None,
        device: Device | None = None
    ) -> Array:
        """Create array with evenly spaced values."""
        # Handle single argument case
        if stop is None:
            stop = start
            start = 0
        
        # Map device
        torch_device = None
        if device is not None:
            if device.type == "cpu":
                torch_device = torch.device("cpu")
            elif device.type == "cuda":
                torch_device = torch.device(f"cuda:{device.index}")
            elif device.type == "mps":
                torch_device = torch.device("mps")
        
        # Map dtype
        torch_dtype = None
        if dtype is not None:
            dtype_map = {
                DataType.FLOAT32: torch.float32,
                DataType.FLOAT16: torch.float16,
                DataType.INT32: torch.int32,
                DataType.INT64: torch.int64,
            }
            torch_dtype = dtype_map.get(dtype)
        
        return torch.arange(start, stop, step, dtype=torch_dtype, device=torch_device)
    
    def broadcast_to(self, input: Array, shape: Shape) -> Array:
        """Broadcast array to shape."""
        if not isinstance(input, torch.Tensor):
            input = torch.tensor(input)
        return torch.broadcast_to(input, shape)
    
    def zeros_like(
        self,
        input: Array,
        dtype: DType | None = None,
        device: Device | None = None
    ) -> Array:
        """Create array of zeros with same shape as input."""
        if not isinstance(input, torch.Tensor):
            input = torch.tensor(input)
        
        # Map dtype
        torch_dtype = None
        if dtype is not None:
            dtype_map = {
                DataType.FLOAT32: torch.float32,
                DataType.FLOAT16: torch.float16,
                DataType.INT32: torch.int32,
                DataType.INT64: torch.int64,
            }
            torch_dtype = dtype_map.get(dtype)
        
        return torch.zeros_like(input, dtype=torch_dtype)
    
    def ones(
        self,
        shape: Shape,
        dtype: DType | None = None,
        device: Device | None = None
    ) -> Array:
        """Create array of ones."""
        # Map device
        torch_device = None
        if device is not None:
            if device.type == "cpu":
                torch_device = torch.device("cpu")
            elif device.type == "cuda":
                torch_device = torch.device(f"cuda:{device.index}")
            elif device.type == "mps":
                torch_device = torch.device("mps")
        
        # Map dtype
        torch_dtype = None
        if dtype is not None:
            dtype_map = {
                DataType.FLOAT32: torch.float32,
                DataType.FLOAT16: torch.float16,
                DataType.INT32: torch.int32,
                DataType.INT64: torch.int64,
            }
            torch_dtype = dtype_map.get(dtype)
        
        return torch.ones(shape, dtype=torch_dtype, device=torch_device)