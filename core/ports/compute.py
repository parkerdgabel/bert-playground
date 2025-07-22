"""Compute backend port interface for ML operations.

This port abstracts compute operations, allowing the core domain
to be independent of specific ML frameworks (MLX, PyTorch, etc.).
"""

from enum import Enum
from typing import Any, Callable, Protocol, Sequence, runtime_checkable

import numpy as np
from typing_extensions import TypeAlias

# Type aliases for framework-agnostic arrays
Array: TypeAlias = Any  # Framework-specific array type
ArrayLike: TypeAlias = np.ndarray | list | tuple | Any
Device: TypeAlias = str | Any
DType: TypeAlias = Any
Shape: TypeAlias = tuple[int, ...]


class DataType(Enum):
    """Framework-agnostic data types."""
    
    FLOAT32 = "float32"
    FLOAT16 = "float16"
    BFLOAT16 = "bfloat16"
    INT32 = "int32"
    INT64 = "int64"
    BOOL = "bool"


@runtime_checkable
class ComputeBackend(Protocol):
    """Port for compute operations."""

    @property
    def name(self) -> str:
        """Name of the compute backend."""
        ...

    @property
    def supports_compilation(self) -> bool:
        """Whether this backend supports JIT compilation."""
        ...

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
            device: Target device
            
        Returns:
            Backend-specific array
        """
        ...

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
            device: Target device
            
        Returns:
            Array of zeros
        """
        ...

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
            device: Target device
            
        Returns:
            Array of ones
        """
        ...

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
        dtype: DataType | DType | None = None,
        device: Device | None = None
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


@runtime_checkable
class NeuralOps(Protocol):
    """Neural network operations port."""

    def linear(
        self,
        input: Array,
        weight: Array,
        bias: Array | None = None
    ) -> Array:
        """Linear transformation.
        
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
        padding_idx: int | None = None
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
        weight: Array | None = None,
        bias: Array | None = None,
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
        seed: int | None = None
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