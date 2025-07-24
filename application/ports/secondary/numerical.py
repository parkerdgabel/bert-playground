"""Secondary numerical port - Numerical operations that the application depends on.

This port defines the numerical operations interface that the application core uses
for mathematical computations. It's a driven port implemented by adapters
for different numerical libraries (numpy, MLX, JAX, etc.).
"""

from typing import Any, List, Optional, Protocol, Tuple, runtime_checkable

from infrastructure.di import port


@port()
@runtime_checkable
class NumericalOperations(Protocol):
    """Secondary port for numerical operations.
    
    This interface abstracts numerical operations needed by domain services,
    particularly the ensemble service. It allows the domain layer to remain
    pure without direct dependencies on numerical libraries.
    
    Implementations might use numpy, MLX, JAX, or other numerical backends.
    """

    def mean(self, values: List[float]) -> float:
        """Calculate the arithmetic mean of a list of values.
        
        Args:
            values: List of numerical values
            
        Returns:
            The arithmetic mean
            
        Raises:
            ValueError: If values is empty
        """
        ...

    def std(self, values: List[float], ddof: int = 0) -> float:
        """Calculate the standard deviation of a list of values.
        
        Args:
            values: List of numerical values
            ddof: Delta degrees of freedom (default: 0 for population std)
            
        Returns:
            The standard deviation
            
        Raises:
            ValueError: If values is empty
        """
        ...

    def correlation(self, x: List[float], y: List[float]) -> float:
        """Calculate the Pearson correlation coefficient between two lists.
        
        Args:
            x: First list of values
            y: Second list of values
            
        Returns:
            Correlation coefficient between -1 and 1
            
        Raises:
            ValueError: If lists have different lengths or are empty
        """
        ...

    def clip(self, value: float, min_val: float, max_val: float) -> float:
        """Clip a value to be within the specified range.
        
        Args:
            value: The value to clip
            min_val: Minimum allowed value
            max_val: Maximum allowed value
            
        Returns:
            Clipped value between min_val and max_val
            
        Raises:
            ValueError: If min_val > max_val
        """
        ...

    def softmax(self, values: List[float]) -> List[float]:
        """Apply softmax function to convert values to probabilities.
        
        Args:
            values: List of values (logits)
            
        Returns:
            List of probabilities that sum to 1.0
            
        Raises:
            ValueError: If values is empty
        """
        ...

    def stack_arrays(self, arrays: List[Any]) -> Any:
        """Stack a list of arrays along a new axis.
        
        Args:
            arrays: List of arrays with the same shape
            
        Returns:
            Stacked array with one additional dimension
            
        Raises:
            ValueError: If arrays is empty or arrays have incompatible shapes
        """
        ...

    def array_mean(self, array: Any, axis: Optional[int] = None) -> Any:
        """Calculate mean of an array along specified axis.
        
        Args:
            array: Input array
            axis: Axis along which to compute mean. None for overall mean.
            
        Returns:
            Mean value(s). Scalar if axis is None, array otherwise.
            
        Raises:
            ValueError: If axis is out of bounds
        """
        ...

    def array_sum(self, array: Any, axis: Optional[int] = None) -> Any:
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

    def array_std(self, array: Any, axis: Optional[int] = None, ddof: int = 0) -> Any:
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

    def array_shape(self, array: Any) -> Tuple[int, ...]:
        """Get the shape of an array.
        
        Args:
            array: Input array
            
        Returns:
            Tuple of integers representing array dimensions
        """
        ...

    def zeros_like(self, array: Any) -> Any:
        """Create an array of zeros with the same shape as input.
        
        Args:
            array: Reference array
            
        Returns:
            New array filled with zeros
        """
        ...

    def ones_like(self, array: Any) -> Any:
        """Create an array of ones with the same shape as input.
        
        Args:
            array: Reference array
            
        Returns:
            New array filled with ones
        """
        ...

    def array_max(self, array: Any, axis: Optional[int] = None) -> Any:
        """Find maximum value in array along specified axis.
        
        Args:
            array: Input array
            axis: Axis along which to find max. None for overall max.
            
        Returns:
            Maximum value(s). Scalar if axis is None, array otherwise.
        """
        ...

    def array_min(self, array: Any, axis: Optional[int] = None) -> Any:
        """Find minimum value in array along specified axis.
        
        Args:
            array: Input array
            axis: Axis along which to find min. None for overall min.
            
        Returns:
            Minimum value(s). Scalar if axis is None, array otherwise.
        """
        ...

    def array_clip(self, array: Any, min_val: float, max_val: float) -> Any:
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

    def array_multiply(self, array: Any, scalar: float) -> Any:
        """Multiply array by scalar value.
        
        Args:
            array: Input array
            scalar: Multiplication factor
            
        Returns:
            Array with all elements multiplied by scalar
        """
        ...

    def array_add(self, array1: Any, array2: Any) -> Any:
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

    def array_log(self, array: Any, epsilon: float = 1e-15) -> Any:
        """Natural logarithm of array elements with numerical stability.
        
        Args:
            array: Input array
            epsilon: Small value to add for numerical stability
            
        Returns:
            Array of log values
        """
        ...

    def array_power(self, array: Any, exponent: float) -> Any:
        """Raise array elements to specified power.
        
        Args:
            array: Input array
            exponent: Power to raise elements to
            
        Returns:
            Array with elements raised to power
        """
        ...

    def array_argsort(self, array: Any, axis: Optional[int] = -1) -> Any:
        """Get indices that would sort an array.
        
        Args:
            array: Input array
            axis: Axis along which to sort. Default is last axis.
            
        Returns:
            Array of indices that would sort the input
        """
        ...

    def array_corrcoef(self, x: Any, y: Optional[Any] = None) -> Any:
        """Calculate correlation coefficient matrix.
        
        Args:
            x: First array or set of arrays
            y: Optional second array. If None, compute correlations within x.
            
        Returns:
            Correlation coefficient matrix
        """
        ...

    def maximum(self, array: Any, value: float) -> Any:
        """Element-wise maximum of array and scalar.
        
        Args:
            array: Input array
            value: Scalar value to compare against
            
        Returns:
            Array with element-wise maximum values
        """
        ...