"""MLX implementation of the ComputeBackend."""

from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple
from contextlib import contextmanager

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np

from infrastructure.di import adapter, Scope
from domain.protocols.compute import Array, DataType
from application.ports.secondary.compute import ComputeBackend, ArrayLike, Device, DType, Shape
from adapters.secondary.compute.base import BaseComputeAdapter
from .utils import convert_to_mlx_array, get_mlx_device_info


@adapter(ComputeBackend, scope=Scope.SINGLETON)
class MLXComputeAdapter(BaseComputeAdapter):
    """MLX implementation of the ComputeBackend for tensor and array operations.
    
    This adapter focuses on low-level tensor operations, array manipulation,
    and device management. High-level neural operations have been moved to
    the MLXNeuralAdapter following hexagonal architecture separation.
    """
    
    def __init__(self):
        """Initialize MLX compute adapter."""
        super().__init__()
        
    @property
    def name(self) -> str:
        """Name of the compute backend."""
        return "mlx"
        
    @property
    def supports_compilation(self) -> bool:
        """Whether this backend supports JIT compilation."""
        return True
    
    # Array creation methods
    
    def array(
        self,
        data: ArrayLike,
        dtype: DataType | DType | None = None,
        device: Device | None = None
    ) -> Array:
        """Create an array from data.
        
        Args:
            data: Input data
            dtype: Data type
            device: Target device (ignored in MLX as it uses unified memory)
            
        Returns:
            MLX array
        """
        return convert_to_mlx_array(data, dtype=dtype)
    
    def zeros(
        self,
        shape: Shape,
        dtype: DataType | DType | None = None,
        device: Device | None = None
    ) -> Array:
        """Create array of zeros.
        
        Args:
            shape: Array shape
            dtype: Data type
            device: Target device (ignored in MLX)
            
        Returns:
            Array of zeros
        """
        return mx.zeros(shape, dtype=dtype)
    
    def ones(
        self,
        shape: Shape,
        dtype: DataType | DType | None = None,
        device: Device | None = None
    ) -> Array:
        """Create array of ones.
        
        Args:
            shape: Array shape
            dtype: Data type
            device: Target device (ignored in MLX)
            
        Returns:
            Array of ones
        """
        return mx.ones(shape, dtype=dtype)
    
    def randn(
        self,
        shape: Shape,
        dtype: DataType | DType | None = None,
        device: Device | None = None,
        seed: int | None = None
    ) -> Array:
        """Create array with normal random values.
        
        Args:
            shape: Array shape
            dtype: Data type
            device: Target device (ignored in MLX)
            seed: Random seed
            
        Returns:
            Array with random values
        """
        if seed is not None:
            mx.random.seed(seed)
        return mx.random.normal(shape, dtype=dtype)
    
    def to_numpy(self, array: Array) -> np.ndarray:
        """Convert array to numpy.
        
        Args:
            array: Backend array
            
        Returns:
            Numpy array
        """
        return np.array(array)
    
    def from_numpy(
        self,
        array: np.ndarray,
        dtype: DataType | DType | None = None,
        device: Device | None = None
    ) -> Array:
        """Create array from numpy.
        
        Args:
            array: Numpy array
            dtype: Target data type
            device: Target device (ignored in MLX)
            
        Returns:
            Backend array
        """
        return mx.array(array, dtype=dtype)
    
    def shape(self, array: Array) -> Shape:
        """Get array shape.
        
        Args:
            array: Input array
            
        Returns:
            Array shape
        """
        return tuple(array.shape)
    
    def dtype(self, array: Array) -> DType:
        """Get array data type.
        
        Args:
            array: Input array
            
        Returns:
            Array data type
        """
        return array.dtype
    
    def device(self, array: Array) -> Device:
        """Get array device.
        
        Args:
            array: Input array
            
        Returns:
            Array device
        """
        return "mlx"  # MLX uses unified memory
    
    # Array manipulation methods
    
    def reshape(
        self,
        array: Array,
        shape: Shape,
    ) -> Array:
        """Reshape array.
        
        Args:
            array: Input array
            shape: New shape
            
        Returns:
            Reshaped array
        """
        return mx.reshape(array, shape)
    
    def transpose(
        self,
        array: Array,
        axes: Sequence[int] | None = None,
    ) -> Array:
        """Transpose array dimensions.
        
        Args:
            array: Input array
            axes: Permutation of axes (None for reverse)
            
        Returns:
            Transposed array
        """
        if axes is None:
            return mx.transpose(array)
        return mx.transpose(array, axes)
    
    def concatenate(
        self,
        arrays: Sequence[Array],
        axis: int = 0,
    ) -> Array:
        """Concatenate arrays along axis.
        
        Args:
            arrays: Arrays to concatenate
            axis: Axis to concatenate along
            
        Returns:
            Concatenated array
        """
        return mx.concatenate(arrays, axis=axis)
    
    def split(
        self,
        array: Array,
        indices_or_sections: int | Sequence[int],
        axis: int = 0,
    ) -> list[Array]:
        """Split array along axis.
        
        Args:
            array: Array to split
            indices_or_sections: Split indices or number of sections
            axis: Axis to split along
            
        Returns:
            List of split arrays
        """
        return mx.split(array, indices_or_sections, axis=axis)
    
    # Mathematical operations
    
    def matmul(
        self,
        a: Array,
        b: Array,
    ) -> Array:
        """Matrix multiplication.
        
        Args:
            a: First array
            b: Second array
            
        Returns:
            Matrix product
        """
        return mx.matmul(a, b)
    
    def sum(
        self,
        array: Array,
        axis: int | Sequence[int] | None = None,
        keepdims: bool = False,
    ) -> Array:
        """Compute sum along axis.
        
        Args:
            array: Input array
            axis: Axis or axes to reduce
            keepdims: Whether to keep reduced dimensions
            
        Returns:
            Sum array
        """
        return mx.sum(array, axis=axis, keepdims=keepdims)
    
    def mean(
        self,
        array: Array,
        axis: int | Sequence[int] | None = None,
        keepdims: bool = False,
    ) -> Array:
        """Compute mean along axis.
        
        Args:
            array: Input array
            axis: Axis or axes to reduce
            keepdims: Whether to keep reduced dimensions
            
        Returns:
            Mean array
        """
        return mx.mean(array, axis=axis, keepdims=keepdims)
    
    def cast(
        self,
        array: Array,
        dtype: DataType | DType
    ) -> Array:
        """Cast array to different data type.
        
        Args:
            array: Input array
            dtype: Target data type
            
        Returns:
            Cast array
        """
        return array.astype(dtype)
    
    # Compilation and gradient methods
    
    def compile(
        self,
        function: Callable[..., Any],
        static_argnums: Sequence[int] | None = None,
        static_argnames: Sequence[str] | None = None
    ) -> Callable[..., Any]:
        """Compile a function for faster execution.
        
        Args:
            function: Function to compile
            static_argnums: Indices of static arguments
            static_argnames: Names of static arguments
            
        Returns:
            Compiled function
        """
        # MLX compilation
        return mx.compile(function)
    
    def gradient(
        self,
        function: Callable[..., Array],
        argnums: int | Sequence[int] = 0
    ) -> Callable[..., Array | tuple[Array, ...]]:
        """Create gradient function.
        
        Args:
            function: Function to differentiate
            argnums: Argument indices to differentiate with respect to
            
        Returns:
            Gradient function
        """
        return mx.grad(function, argnums=argnums)
    
    def value_and_gradient(
        self,
        function: Callable[..., Array],
        argnums: int | Sequence[int] = 0
    ) -> Callable[..., tuple[Array, Array | tuple[Array, ...]]]:
        """Create function that returns both value and gradient.
        
        Args:
            function: Function to differentiate
            argnums: Argument indices to differentiate with respect to
            
        Returns:
            Function returning (value, gradient)
        """
        return mx.value_and_grad(function, argnums=argnums)
    
    # Array I/O methods
    
    def save_arrays(self, path: str, arrays: dict[str, Array]) -> None:
        """Save arrays to file.
        
        Args:
            path: Path to save file
            arrays: Dictionary of arrays to save
        """
        mx.savez(path, **arrays)
    
    def load_arrays(self, path: str) -> dict[str, Array]:
        """Load arrays from file.
        
        Args:
            path: Path to load file
            
        Returns:
            Dictionary of loaded arrays
        """
        return dict(mx.load(path))
    
    def load_weights(self, path: str) -> dict[str, Array]:
        """Load weights from file.
        
        Args:
            path: Path to weights file
            
        Returns:
            Dictionary of parameter names to arrays
        """
        # For MLX, weights are stored as .npz files typically
        return dict(mx.load(path))
    
    def tree_unflatten(self, items: list[tuple[str, Array]]) -> dict[str, Any]:
        """Unflatten a list of (key, value) pairs into nested dictionary.
        
        Args:
            items: List of (key, value) pairs with dot-separated keys
            
        Returns:
            Nested dictionary structure
        """
        result = {}
        for key, value in items:
            parts = key.split('.')
            current = result
            for part in parts[:-1]:
                if part not in current:
                    current[part] = {}
                current = current[part]
            current[parts[-1]] = value
        return result
    
    # Device and system operations
    
    
    def get_device_info(
        self,
    ) -> Dict[str, Any]:
        """Get information about compute device.
        
        Returns:
            Dictionary with device information
        """
        if self._device_cache is None:
            self._device_cache = get_mlx_device_info()
        return self._device_cache
    
    def synchronize(
        self,
    ) -> None:
        """Synchronize compute operations."""
        # Force evaluation of any pending operations
        mx.eval()
    
