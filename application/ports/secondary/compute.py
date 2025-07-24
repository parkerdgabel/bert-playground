"""Secondary compute port - Compute backend that the application depends on.

This port defines the compute interface that the application core uses
to perform ML computations. It's a driven port implemented by adapters
for different ML frameworks (MLX, PyTorch, JAX, etc.).

This port now supports lazy evaluation and function transformation capabilities
to enable MLX/JAX optimizations.
"""

from typing import Any, Callable, Dict, List, Optional, Protocol, Sequence, Tuple, TypeVar, runtime_checkable
from typing_extensions import TypeAlias
from abc import abstractmethod
from enum import Enum

import numpy as np

from infrastructure.di import port

# Import domain types
from domain.protocols.compute import Array, Module, DataType

# Type variables
F = TypeVar('F', bound=Callable[..., Any])

# Additional type aliases specific to compute backends
ArrayLike: TypeAlias = np.ndarray | list | tuple | Any
Device: TypeAlias = str | Any
DType: TypeAlias = Any
Shape: TypeAlias = tuple[int, ...]

# Lazy evaluation types
LazyArray: TypeAlias = Any  # Framework-specific lazy array
EagerArray: TypeAlias = Any  # Framework-specific eager array
ComputationGraph: TypeAlias = Any  # Framework-specific computation graph


class BackendCapability(Enum):
    """Capabilities that a compute backend might support."""
    
    LAZY_EVALUATION = "lazy_evaluation"
    JIT_COMPILATION = "jit_compilation"
    AUTOMATIC_DIFFERENTIATION = "automatic_differentiation"
    VECTORIZATION = "vectorization"
    MIXED_PRECISION = "mixed_precision"
    DISTRIBUTED = "distributed"
    GPU_ACCELERATION = "gpu_acceleration"


@port()
@runtime_checkable
class ComputeBackend(Protocol):
    """Secondary port for compute operations.
    
    This interface is implemented by adapters for specific ML frameworks.
    The application core depends on this interface for all compute operations.
    
    Key features:
    - Lazy evaluation support for building computation graphs
    - Function transformation (JIT, grad, vmap)
    - Explicit evaluation control
    - Backend capability discovery
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Name of the compute backend (e.g., 'mlx', 'pytorch')."""
        ...

    @property
    @abstractmethod
    def supports_compilation(self) -> bool:
        """Whether this backend supports JIT compilation."""
        ...
    
    @property
    @abstractmethod
    def capabilities(self) -> set[BackendCapability]:
        """Set of capabilities supported by this backend."""
        ...
    
    # Capability checking methods
    
    def has_capability(self, capability: BackendCapability) -> bool:
        """Check if backend has a specific capability.
        
        Args:
            capability: Capability to check for
            
        Returns:
            True if capability is supported
        """
        ...
    
    # Lazy evaluation control
    
    def is_lazy(self, array: Array) -> bool:
        """Check if an array is lazy (not yet evaluated).
        
        Args:
            array: Array to check
            
        Returns:
            True if array is lazy
        """
        ...
    
    def eval(self, array: Array | LazyArray) -> EagerArray:
        """Force evaluation of a lazy array.
        
        Args:
            array: Array to evaluate
            
        Returns:
            Eager array with computed values
        """
        ...
    
    def make_lazy(self, array: EagerArray) -> LazyArray:
        """Convert an eager array to lazy.
        
        Args:
            array: Eager array
            
        Returns:
            Lazy array
        """
        ...
    
    def stop_gradient(self, array: Array) -> Array:
        """Stop gradient propagation through array.
        
        Args:
            array: Array to stop gradients for
            
        Returns:
            Array with gradients stopped
        """
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

    # Function transformation methods
    
    def jit(
        self,
        function: F,
        static_argnums: Sequence[int] | None = None,
        static_argnames: Sequence[str] | None = None,
        donate_argnums: Sequence[int] | None = None
    ) -> F:
        """JIT compile a function for faster execution.
        
        Args:
            function: Function to compile
            static_argnums: Indices of static arguments
            static_argnames: Names of static arguments
            donate_argnums: Indices of arguments that can be donated
            
        Returns:
            JIT compiled function
        """
        ...
    
    def compile(
        self,
        function: Callable[..., Any],
        static_argnums: Sequence[int] | None = None,
        static_argnames: Sequence[str] | None = None
    ) -> Callable[..., Any]:
        """Compile a function for faster execution (legacy alias for jit).
        
        Args:
            function: Function to compile
            static_argnums: Indices of static arguments
            static_argnames: Names of static arguments
            
        Returns:
            Compiled function
        """
        ...

    def grad(
        self,
        function: Callable[..., Array],
        argnums: int | Sequence[int] = 0,
        has_aux: bool = False
    ) -> Callable[..., Array | tuple[Array, ...]]:
        """Create gradient function using automatic differentiation.
        
        Args:
            function: Function to differentiate (must return scalar)
            argnums: Argument indices to differentiate with respect to
            has_aux: Whether function returns auxiliary data
            
        Returns:
            Gradient function
        """
        ...
    
    def gradient(
        self,
        function: Callable[..., Array],
        argnums: int | Sequence[int] = 0
    ) -> Callable[..., Array | tuple[Array, ...]]:
        """Create gradient function (alias for grad).
        
        Args:
            function: Function to differentiate
            argnums: Argument indices to differentiate with respect to
            
        Returns:
            Gradient function
        """
        ...

    def value_and_grad(
        self,
        function: Callable[..., Array],
        argnums: int | Sequence[int] = 0,
        has_aux: bool = False
    ) -> Callable[..., tuple[Array, Array | tuple[Array, ...]]]:
        """Create function that returns both value and gradient.
        
        Args:
            function: Function to differentiate
            argnums: Argument indices to differentiate with respect to
            has_aux: Whether function returns auxiliary data
            
        Returns:
            Function returning (value, gradient) or ((value, aux), gradient)
        """
        ...
    
    def value_and_gradient(
        self,
        function: Callable[..., Array],
        argnums: int | Sequence[int] = 0
    ) -> Callable[..., tuple[Array, Array | tuple[Array, ...]]]:
        """Create function that returns both value and gradient (alias).
        
        Args:
            function: Function to differentiate
            argnums: Argument indices to differentiate with respect to
            
        Returns:
            Function returning (value, gradient)
        """
        ...
    
    def vmap(
        self,
        function: F,
        in_axes: int | Sequence[int | None] = 0,
        out_axes: int | Sequence[int] = 0
    ) -> F:
        """Vectorize a function to operate on batches.
        
        Args:
            function: Function to vectorize
            in_axes: Axes to map over for inputs (None means don't map)
            out_axes: Axes to map over for outputs
            
        Returns:
            Vectorized function
        """
        ...
    
    def pmap(
        self,
        function: F,
        axis_name: str | None = None,
        in_axes: int | Sequence[int | None] = 0,
        out_axes: int | Sequence[int] = 0
    ) -> F:
        """Parallelize a function across devices.
        
        Args:
            function: Function to parallelize
            axis_name: Name for the mapped axis
            in_axes: Axes to map over for inputs
            out_axes: Axes to map over for outputs
            
        Returns:
            Parallelized function
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
    
    # Graph optimization methods
    
    def create_computation_graph(self, fn: Callable[..., Array]) -> ComputationGraph:
        """Create a computation graph from a function.
        
        Args:
            fn: Function that builds the computation
            
        Returns:
            Computation graph representation
        """
        ...
    
    def optimize_graph(self, graph: ComputationGraph) -> ComputationGraph:
        """Optimize a computation graph.
        
        Args:
            graph: Graph to optimize
            
        Returns:
            Optimized graph
        """
        ...
    
    def profile(
        self,
        function: Callable[..., Any],
        *args,
        **kwargs
    ) -> tuple[Any, dict[str, Any]]:
        """Profile a function execution.
        
        Args:
            function: Function to profile
            *args: Function arguments
            **kwargs: Function keyword arguments
            
        Returns:
            Tuple of (result, profile_data)
        """
        ...
    
    # Advanced array operations
    
    def where(self, condition: Array, x: Array, y: Array) -> Array:
        """Element-wise selection based on condition.
        
        Args:
            condition: Boolean array
            x: Values where condition is True
            y: Values where condition is False
            
        Returns:
            Result array
        """
        ...
    
    def einsum(self, subscripts: str, *operands: Array) -> Array:
        """Einstein summation convention.
        
        Args:
            subscripts: Einsum subscripts string
            *operands: Input arrays
            
        Returns:
            Result of einsum
        """
        ...
    
    def scan(
        self,
        function: Callable[[Any, Any], tuple[Any, Any]],
        init: Any,
        xs: Array,
        length: int | None = None
    ) -> tuple[Any, Array]:
        """Scan (fold) over leading axis of array.
        
        Args:
            function: Function to apply (carry, x) -> (new_carry, y)
            init: Initial carry value
            xs: Input array to scan over
            length: Optional length to scan
            
        Returns:
            Tuple of (final_carry, stacked_outputs)
        """
        ...


