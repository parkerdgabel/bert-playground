"""Secondary compute port - Compute backend that the application depends on.

This port defines the compute interface that the application core uses
to perform ML computations. It's a driven port implemented by adapters
for different ML frameworks (MLX, PyTorch, JAX, etc.).
"""

from typing import Any, Callable, Dict, List, Optional, Protocol, Sequence, Tuple, runtime_checkable

import numpy as np
from typing_extensions import TypeAlias

from infrastructure.di import port

# Import domain types
from domain.protocols.compute import Array, Module, DataType

# Additional type aliases specific to compute backends
ArrayLike: TypeAlias = np.ndarray | list | tuple | Any
Device: TypeAlias = str | Any
DType: TypeAlias = Any
Shape: TypeAlias = tuple[int, ...]


@port()
@runtime_checkable
class ComputeBackend(Protocol):
    """Secondary port for compute operations.
    
    This interface is implemented by adapters for specific ML frameworks.
    The application core depends on this interface for all compute operations.
    """

    @property
    def name(self) -> str:
        """Name of the compute backend (e.g., 'mlx', 'pytorch')."""
        ...

    @property
    def supports_compilation(self) -> bool:
        """Whether this backend supports JIT compilation."""
        ...

    def array(
        self,
        data: ArrayLike,
        dtype: DataType | Optional[DType] = None,
        device: Optional[Device] = None
    ) -> Array:
        """Create an array from data.
        
        Args:
            data: Input data
            dtype: Data type
            device: Target device
            
        Returns:
            Backend-specific array
        """
        ...

    def zeros(
        self,
        shape: Shape,
        dtype: DataType | Optional[DType] = None,
        device: Optional[Device] = None
    ) -> Array:
        """Create array of zeros.
        
        Args:
            shape: Array shape
            dtype: Data type
            device: Target device
            
        Returns:
            Array of zeros
        """
        ...

    def ones(
        self,
        shape: Shape,
        dtype: DataType | Optional[DType] = None,
        device: Optional[Device] = None
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

    def randn(
        self,
        shape: Shape,
        dtype: DataType | Optional[DType] = None,
        device: Optional[Device] = None,
        seed: Optional[int] = None
    ) -> Array:
        """Create array with normal random values.
        
        Args:
            shape: Array shape
            dtype: Data type
            device: Target device
            seed: Random seed
            
        Returns:
            Array with random values
        """
        ...

    def to_numpy(self, array: Array) -> np.ndarray:
        """Convert array to numpy.
        
        Args:
            array: Backend array
            
        Returns:
            Numpy array
        """
        ...

    def from_numpy(
        self,
        array: np.ndarray,
        dtype: DataType | Optional[DType] = None,
        device: Optional[Device] = None
    ) -> Array:
        """Create array from numpy.
        
        Args:
            array: Numpy array
            dtype: Target data type
            device: Target device
            
        Returns:
            Backend array
        """
        ...

    def shape(self, array: Array) -> Shape:
        """Get array shape.
        
        Args:
            array: Input array
            
        Returns:
            Array shape
        """
        ...

    def dtype(self, array: Array) -> DType:
        """Get array data type.
        
        Args:
            array: Input array
            
        Returns:
            Array data type
        """
        ...

    def device(self, array: Array) -> Device:
        """Get array device.
        
        Args:
            array: Input array
            
        Returns:
            Array device
        """
        ...

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
        ...

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
        ...

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
        ...

    def load_weights(self, path: str) -> dict[str, Array]:
        """Load weights from file.
        
        Args:
            path: Path to weights file
            
        Returns:
            Dictionary of parameter names to arrays
        """
        ...

    def tree_unflatten(self, items: list[tuple[str, Array]]) -> dict[str, Any]:
        """Unflatten a list of (key, value) pairs into nested dictionary.
        
        Args:
            items: List of (key, value) pairs with dot-separated keys
            
        Returns:
            Nested dictionary structure
        """
        ...

    def save_arrays(self, path: str, arrays: dict[str, Array]) -> None:
        """Save arrays to file.
        
        Args:
            path: Path to save file
            arrays: Dictionary of arrays to save
        """
        ...

    def load_arrays(self, path: str) -> dict[str, Array]:
        """Load arrays from file.
        
        Args:
            path: Path to load file
            
        Returns:
            Dictionary of loaded arrays
        """
        ...
    
    # Basic tensor operations
    
    def matmul(self, a: Array, b: Array) -> Array:
        """Matrix multiplication.
        
        Args:
            a: First array
            b: Second array
            
        Returns:
            Matrix product
        """
        ...
    
    def transpose(self, array: Array, axes: Sequence[int] | None = None) -> Array:
        """Transpose array dimensions.
        
        Args:
            array: Input array
            axes: Permutation of axes (None for reverse)
            
        Returns:
            Transposed array
        """
        ...
    
    def reshape(self, array: Array, shape: Shape) -> Array:
        """Reshape array.
        
        Args:
            array: Input array
            shape: New shape
            
        Returns:
            Reshaped array
        """
        ...
    
    def concatenate(
        self, arrays: Sequence[Array], axis: int = 0
    ) -> Array:
        """Concatenate arrays along axis.
        
        Args:
            arrays: Arrays to concatenate
            axis: Axis to concatenate along
            
        Returns:
            Concatenated array
        """
        ...
    
    def split(
        self, array: Array, indices_or_sections: int | Sequence[int], axis: int = 0
    ) -> list[Array]:
        """Split array along axis.
        
        Args:
            array: Array to split
            indices_or_sections: Split indices or number of sections
            axis: Axis to split along
            
        Returns:
            List of split arrays
        """
        ...
    
    def mean(self, array: Array, axis: int | Sequence[int] | None = None, keepdims: bool = False) -> Array:
        """Compute mean along axis.
        
        Args:
            array: Input array
            axis: Axis or axes to reduce
            keepdims: Whether to keep reduced dimensions
            
        Returns:
            Mean array
        """
        ...
    
    def sum(self, array: Array, axis: int | Sequence[int] | None = None, keepdims: bool = False) -> Array:
        """Compute sum along axis.
        
        Args:
            array: Input array
            axis: Axis or axes to reduce
            keepdims: Whether to keep reduced dimensions
            
        Returns:
            Sum array
        """
        ...
    
    def cast(self, array: Array, dtype: DataType | DType) -> Array:
        """Cast array to different data type.
        
        Args:
            array: Input array
            dtype: Target data type
            
        Returns:
            Cast array
        """
        ...


