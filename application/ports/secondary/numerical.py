"""Secondary numerical port - Numerical operations that the application depends on.

This port defines the numerical operations interface that the application core uses
for mathematical computations. It's a driven port implemented by adapters
for different numerical libraries (numpy, MLX, JAX, etc.).

This port now supports lazy evaluation and function transformation capabilities
to enable MLX/JAX optimizations like JIT compilation, automatic differentiation,
and vectorization.
"""

from typing import Any, Callable, List, Optional, Protocol, Tuple, TypeVar, runtime_checkable
from typing_extensions import TypeAlias
from abc import abstractmethod
from enum import Enum

from infrastructure.di import port

# Type variables and aliases
T = TypeVar('T')
F = TypeVar('F', bound=Callable[..., Any])

# Lazy array type that supports tracing/compilation
LazyArray: TypeAlias = Any  # Framework-specific lazy array type
EagerArray: TypeAlias = Any  # Framework-specific eager array type
ComputationGraph: TypeAlias = Any  # Framework-specific computation graph


class EvaluationMode(Enum):
    """Evaluation modes for computations."""
    
    LAZY = "lazy"  # Build computation graph without evaluation
    EAGER = "eager"  # Evaluate immediately
    JIT = "jit"  # JIT compile and evaluate


@port()
@runtime_checkable
class NumericalOperations(Protocol):
    """Secondary port for numerical operations.
    
    This interface abstracts numerical operations needed by domain services,
    particularly the ensemble service. It allows the domain layer to remain
    pure without direct dependencies on numerical libraries.
    
    Implementations might use numpy, MLX, JAX, or other numerical backends.
    
    Key features:
    - Lazy evaluation: Operations return computation graphs by default
    - Function transformation: Support for JIT, grad, vmap
    - Explicit evaluation: Methods to materialize results when needed
    - Type safety: Clear separation between lazy and eager arrays
    """
    
    @property
    @abstractmethod
    def supports_lazy_evaluation(self) -> bool:
        """Whether this backend supports lazy evaluation."""
        ...
    
    @property
    @abstractmethod
    def supports_jit(self) -> bool:
        """Whether this backend supports JIT compilation."""
        ...
    
    @property
    @abstractmethod
    def supports_autodiff(self) -> bool:
        """Whether this backend supports automatic differentiation."""
        ...
    
    @property
    @abstractmethod
    def supports_vmap(self) -> bool:
        """Whether this backend supports vectorization."""
        ...
    
    # Evaluation control methods
    
    def set_evaluation_mode(self, mode: EvaluationMode) -> None:
        """Set the evaluation mode for subsequent operations.
        
        Args:
            mode: Evaluation mode to use
        """
        ...
    
    def eval(self, computation: LazyArray | ComputationGraph) -> EagerArray:
        """Explicitly evaluate a lazy computation.
        
        Args:
            computation: Lazy array or computation graph
            
        Returns:
            Eagerly evaluated array
        """
        ...
    
    def make_lazy(self, array: EagerArray) -> LazyArray:
        """Convert an eager array to lazy evaluation.
        
        Args:
            array: Eager array
            
        Returns:
            Lazy array
        """
        ...
    
    # Function transformation methods
    
    def jit(self, fn: F, static_argnums: Optional[Tuple[int, ...]] = None) -> F:
        """JIT compile a function for faster execution.
        
        Args:
            fn: Function to compile
            static_argnums: Indices of static arguments
            
        Returns:
            JIT compiled function
        """
        ...
    
    def grad(self, fn: Callable[..., LazyArray], argnums: int | Tuple[int, ...] = 0) -> Callable[..., LazyArray | Tuple[LazyArray, ...]]:
        """Create gradient function using automatic differentiation.
        
        Args:
            fn: Function to differentiate
            argnums: Argument indices to differentiate with respect to
            
        Returns:
            Gradient function
        """
        ...
    
    def value_and_grad(self, fn: Callable[..., LazyArray], argnums: int | Tuple[int, ...] = 0) -> Callable[..., Tuple[LazyArray, LazyArray | Tuple[LazyArray, ...]]]:
        """Create function that returns both value and gradient.
        
        Args:
            fn: Function to differentiate
            argnums: Argument indices to differentiate with respect to
            
        Returns:
            Function returning (value, gradient)
        """
        ...
    
    def vmap(self, fn: F, in_axes: int | Tuple[int | None, ...] = 0, out_axes: int = 0) -> F:
        """Vectorize a function to operate on batches.
        
        Args:
            fn: Function to vectorize
            in_axes: Axes to map over for inputs
            out_axes: Axes to map over for outputs
            
        Returns:
            Vectorized function
        """
        ...
    
    def stop_gradient(self, array: LazyArray) -> LazyArray:
        """Stop gradient propagation through an array.
        
        Args:
            array: Array to stop gradients for
            
        Returns:
            Array with gradients stopped
        """
        ...

    # Original operations now return lazy arrays by default
    
    def mean(self, values: List[float] | LazyArray) -> LazyArray:
        """Calculate the arithmetic mean of values.
        
        Args:
            values: List of numerical values or lazy array
            
        Returns:
            Lazy array containing the mean (evaluate with eval())
            
        Raises:
            ValueError: If values is empty
        """
        ...

    def std(self, values: List[float] | LazyArray, ddof: int = 0) -> LazyArray:
        """Calculate the standard deviation of values.
        
        Args:
            values: List of numerical values or lazy array
            ddof: Delta degrees of freedom (default: 0 for population std)
            
        Returns:
            Lazy array containing the standard deviation
            
        Raises:
            ValueError: If values is empty
        """
        ...

    def correlation(self, x: List[float] | LazyArray, y: List[float] | LazyArray) -> LazyArray:
        """Calculate the Pearson correlation coefficient.
        
        Args:
            x: First array of values
            y: Second array of values
            
        Returns:
            Lazy array containing correlation coefficient
            
        Raises:
            ValueError: If arrays have different lengths or are empty
        """
        ...

    def clip(self, value: float | LazyArray, min_val: float, max_val: float) -> LazyArray:
        """Clip a value to be within the specified range.
        
        Args:
            value: The value or array to clip
            min_val: Minimum allowed value
            max_val: Maximum allowed value
            
        Returns:
            Lazy array containing clipped value
            
        Raises:
            ValueError: If min_val > max_val
        """
        ...

    def softmax(self, values: List[float] | LazyArray) -> LazyArray:
        """Apply softmax function to convert values to probabilities.
        
        Args:
            values: Array of values (logits)
            
        Returns:
            Lazy array of probabilities that sum to 1.0
            
        Raises:
            ValueError: If values is empty
        """
        ...

    def stack_arrays(self, arrays: List[LazyArray]) -> LazyArray:
        """Stack a list of arrays along a new axis.
        
        Args:
            arrays: List of lazy arrays with the same shape
            
        Returns:
            Lazy array with one additional dimension
            
        Raises:
            ValueError: If arrays is empty or arrays have incompatible shapes
        """
        ...

    def array_mean(self, array: LazyArray, axis: Optional[int] = None) -> LazyArray:
        """Calculate mean of an array along specified axis.
        
        Args:
            array: Input lazy array
            axis: Axis along which to compute mean. None for overall mean.
            
        Returns:
            Lazy array containing mean value(s)
            
        Raises:
            ValueError: If axis is out of bounds
        """
        ...

    def array_sum(self, array: LazyArray, axis: Optional[int] = None) -> LazyArray:
        """Calculate sum of an array along specified axis.
        
        Args:
            array: Input array
            axis: Axis along which to compute sum. None for overall sum.
            
        Returns:
            Sum value(s). Scalar if axis is None, array otherwise.
            
        Raises:
            ValueError: If axis is out of bounds
        """
        ...

    def array_std(self, array: LazyArray, axis: Optional[int] = None, ddof: int = 0) -> LazyArray:
        """Calculate standard deviation of an array along specified axis.
        
        Args:
            array: Input array
            axis: Axis along which to compute std. None for overall std.
            ddof: Delta degrees of freedom
            
        Returns:
            Standard deviation value(s). Scalar if axis is None, array otherwise.
            
        Raises:
            ValueError: If axis is out of bounds
        """
        ...

    def array_shape(self, array: LazyArray) -> Tuple[int, ...]:
        """Get the shape of an array.
        
        Args:
            array: Input array
            
        Returns:
            Tuple of integers representing array dimensions
        """
        ...

    def zeros_like(self, array: LazyArray) -> LazyArray:
        """Create an array of zeros with the same shape as input.
        
        Args:
            array: Reference array
            
        Returns:
            New array filled with zeros
        """
        ...

    def ones_like(self, array: LazyArray) -> LazyArray:
        """Create an array of ones with the same shape as input.
        
        Args:
            array: Reference array
            
        Returns:
            New array filled with ones
        """
        ...

    def array_max(self, array: LazyArray, axis: Optional[int] = None) -> LazyArray:
        """Find maximum value in array along specified axis.
        
        Args:
            array: Input array
            axis: Axis along which to find max. None for overall max.
            
        Returns:
            Maximum value(s). Scalar if axis is None, array otherwise.
        """
        ...

    def array_min(self, array: LazyArray, axis: Optional[int] = None) -> LazyArray:
        """Find minimum value in array along specified axis.
        
        Args:
            array: Input array
            axis: Axis along which to find min. None for overall min.
            
        Returns:
            Minimum value(s). Scalar if axis is None, array otherwise.
        """
        ...

    def array_clip(self, array: LazyArray, min_val: float, max_val: float) -> LazyArray:
        """Clip all values in an array to specified range.
        
        Args:
            array: Input array
            min_val: Minimum allowed value
            max_val: Maximum allowed value
            
        Returns:
            Array with clipped values
            
        Raises:
            ValueError: If min_val > max_val
        """
        ...

    def array_multiply(self, array: LazyArray, scalar: float) -> LazyArray:
        """Multiply array by scalar value.
        
        Args:
            array: Input array
            scalar: Multiplication factor
            
        Returns:
            Array with all elements multiplied by scalar
        """
        ...

    def array_add(self, array1: LazyArray, array2: LazyArray) -> LazyArray:
        """Element-wise addition of two arrays.
        
        Args:
            array1: First array
            array2: Second array
            
        Returns:
            Sum of arrays
            
        Raises:
            ValueError: If arrays have incompatible shapes
        """
        ...

    def array_log(self, array: LazyArray, epsilon: float = 1e-15) -> LazyArray:
        """Natural logarithm of array elements with numerical stability.
        
        Args:
            array: Input array
            epsilon: Small value to add for numerical stability
            
        Returns:
            Array of log values
        """
        ...

    def array_power(self, array: LazyArray, exponent: float) -> LazyArray:
        """Raise array elements to specified power.
        
        Args:
            array: Input array
            exponent: Power to raise elements to
            
        Returns:
            Array with elements raised to power
        """
        ...

    def array_argsort(self, array: LazyArray, axis: Optional[int] = -1) -> LazyArray:
        """Get indices that would sort an array.
        
        Args:
            array: Input array
            axis: Axis along which to sort. Default is last axis.
            
        Returns:
            Array of indices that would sort the input
        """
        ...

    def array_corrcoef(self, x: LazyArray, y: Optional[LazyArray] = None) -> LazyArray:
        """Calculate correlation coefficient matrix.
        
        Args:
            x: First array or set of arrays
            y: Optional second array. If None, compute correlations within x.
            
        Returns:
            Correlation coefficient matrix
        """
        ...

    def maximum(self, array: LazyArray, value: float) -> LazyArray:
        """Element-wise maximum of array and scalar.
        
        Args:
            array: Input lazy array
            value: Scalar value to compare against
            
        Returns:
            Lazy array with element-wise maximum values
        """
        ...
    
    # Additional lazy evaluation helpers
    
    def is_lazy(self, array: Any) -> bool:
        """Check if an array is lazy (not yet evaluated).
        
        Args:
            array: Array to check
            
        Returns:
            True if array is lazy, False if eager
        """
        ...
    
    def batch_eval(self, computations: List[LazyArray]) -> List[EagerArray]:
        """Evaluate multiple lazy computations efficiently.
        
        Args:
            computations: List of lazy arrays to evaluate
            
        Returns:
            List of eager arrays
        """
        ...
    
    def create_computation_graph(self, fn: Callable[..., LazyArray]) -> ComputationGraph:
        """Create a computation graph from a function.
        
        Args:
            fn: Function that returns a lazy array
            
        Returns:
            Computation graph representation
        """
        ...
    
    def optimize_graph(self, graph: ComputationGraph) -> ComputationGraph:
        """Optimize a computation graph for better performance.
        
        Args:
            graph: Computation graph to optimize
            
        Returns:
            Optimized computation graph
        """
        ...