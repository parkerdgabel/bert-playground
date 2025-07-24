"""NumPy implementation of the NumericalOperations port.

This adapter provides a NumPy-based implementation of numerical operations,
supporting both eager and lazy evaluation modes (though NumPy is inherently eager).
"""

from typing import Any, Callable, List, Optional, Tuple, TypeVar, Union
import numpy as np
from functools import wraps
import warnings

from application.ports.secondary.numerical import (
    NumericalOperations,
    EvaluationMode,
    LazyArray,
    EagerArray,
    ComputationGraph
)
from infrastructure.di import adapter

# Type variables
T = TypeVar('T')
F = TypeVar('F', bound=Callable[..., Any])


class LazyNumPyArray:
    """Wrapper for NumPy arrays to simulate lazy evaluation.
    
    Since NumPy is inherently eager, this is a thin wrapper that
    stores the array and provides lazy-like interface.
    """
    
    def __init__(self, array: np.ndarray):
        self._array = array
        self._is_lazy = True
    
    @property
    def value(self) -> np.ndarray:
        """Get the underlying NumPy array."""
        return self._array
    
    def eval(self) -> np.ndarray:
        """Evaluate the lazy array (returns the stored array)."""
        return self._array


@adapter(NumericalOperations, priority=50)
class NumPyNumericalAdapter:
    """NumPy implementation of the NumericalOperations port.
    
    This adapter uses NumPy for all numerical operations. Since NumPy
    is inherently eager, lazy evaluation is simulated through wrappers.
    
    Features:
    - Full support for all numerical operations
    - Simulated lazy evaluation (operations are still eager internally)
    - No support for JIT, autodiff, or vmap (NumPy limitations)
    - Efficient array operations using NumPy's optimized routines
    """
    
    def __init__(self):
        self._evaluation_mode = EvaluationMode.EAGER
    
    @property
    def supports_lazy_evaluation(self) -> bool:
        """NumPy doesn't truly support lazy evaluation."""
        return False
    
    @property
    def supports_jit(self) -> bool:
        """NumPy doesn't support JIT compilation."""
        return False
    
    @property
    def supports_autodiff(self) -> bool:
        """NumPy doesn't support automatic differentiation."""
        return False
    
    @property
    def supports_vmap(self) -> bool:
        """NumPy doesn't have explicit vmap support."""
        return False
    
    # Evaluation control methods
    
    def set_evaluation_mode(self, mode: EvaluationMode) -> None:
        """Set the evaluation mode for subsequent operations."""
        if mode == EvaluationMode.JIT:
            warnings.warn("NumPy doesn't support JIT compilation, using EAGER mode instead")
            self._evaluation_mode = EvaluationMode.EAGER
        else:
            self._evaluation_mode = mode
    
    def eval(self, computation: Union[LazyArray, ComputationGraph]) -> EagerArray:
        """Explicitly evaluate a lazy computation."""
        if isinstance(computation, LazyNumPyArray):
            return computation.eval()
        elif isinstance(computation, np.ndarray):
            return computation
        else:
            # Assume it's already an eager array or convertible
            return np.asarray(computation)
    
    def make_lazy(self, array: EagerArray) -> LazyArray:
        """Convert an eager array to lazy evaluation."""
        if isinstance(array, LazyNumPyArray):
            return array
        return LazyNumPyArray(np.asarray(array))
    
    # Function transformation methods (limited support)
    
    def jit(self, fn: F, static_argnums: Optional[Tuple[int, ...]] = None) -> F:
        """JIT compilation not supported, returns original function."""
        warnings.warn("NumPy doesn't support JIT compilation, returning original function")
        return fn
    
    def grad(self, fn: Callable[..., LazyArray], argnums: Union[int, Tuple[int, ...]] = 0) -> Callable[..., Union[LazyArray, Tuple[LazyArray, ...]]]:
        """Gradient computation not supported."""
        raise NotImplementedError("NumPy doesn't support automatic differentiation. Consider using JAX or PyTorch backends.")
    
    def value_and_grad(self, fn: Callable[..., LazyArray], argnums: Union[int, Tuple[int, ...]] = 0) -> Callable[..., Tuple[LazyArray, Union[LazyArray, Tuple[LazyArray, ...]]]]:
        """Value and gradient computation not supported."""
        raise NotImplementedError("NumPy doesn't support automatic differentiation. Consider using JAX or PyTorch backends.")
    
    def vmap(self, fn: F, in_axes: Union[int, Tuple[Optional[int], ...]] = 0, out_axes: int = 0) -> F:
        """Vectorization through manual broadcasting."""
        warnings.warn("NumPy doesn't have explicit vmap, using vectorize instead")
        return np.vectorize(fn)  # type: ignore
    
    def stop_gradient(self, array: LazyArray) -> LazyArray:
        """No gradients in NumPy, returns array as-is."""
        return array
    
    # Helper method to wrap results based on evaluation mode
    
    def _wrap_result(self, result: np.ndarray) -> Union[LazyArray, EagerArray]:
        """Wrap result based on current evaluation mode."""
        if self._evaluation_mode == EvaluationMode.LAZY:
            return LazyNumPyArray(result)
        return result
    
    # Numerical operations
    
    def mean(self, values: Union[List[float], LazyArray]) -> LazyArray:
        """Calculate the arithmetic mean of values."""
        if isinstance(values, LazyNumPyArray):
            arr = values.value
        elif isinstance(values, np.ndarray):
            arr = values
        else:
            arr = np.asarray(values)
        
        if arr.size == 0:
            raise ValueError("Cannot calculate mean of empty array")
        
        result = np.mean(arr)
        # Ensure result is always an array
        if np.isscalar(result):
            result = np.array(result)
        return self._wrap_result(result)
    
    def std(self, values: Union[List[float], LazyArray], ddof: int = 0) -> LazyArray:
        """Calculate the standard deviation of values."""
        if isinstance(values, LazyNumPyArray):
            arr = values.value
        elif isinstance(values, np.ndarray):
            arr = values
        else:
            arr = np.asarray(values)
        
        if arr.size == 0:
            raise ValueError("Cannot calculate standard deviation of empty array")
        
        result = np.std(arr, ddof=ddof)
        return self._wrap_result(result)
    
    def correlation(self, x: Union[List[float], LazyArray], y: Union[List[float], LazyArray]) -> LazyArray:
        """Calculate the Pearson correlation coefficient."""
        if isinstance(x, LazyNumPyArray):
            x_arr = x.value
        elif isinstance(x, np.ndarray):
            x_arr = x
        else:
            x_arr = np.asarray(x)
        
        if isinstance(y, LazyNumPyArray):
            y_arr = y.value
        elif isinstance(y, np.ndarray):
            y_arr = y
        else:
            y_arr = np.asarray(y)
        
        if x_arr.size == 0 or y_arr.size == 0:
            raise ValueError("Cannot calculate correlation of empty arrays")
        
        if x_arr.shape != y_arr.shape:
            raise ValueError(f"Arrays must have the same shape, got {x_arr.shape} and {y_arr.shape}")
        
        # Calculate correlation coefficient
        corr_matrix = np.corrcoef(x_arr.flatten(), y_arr.flatten())
        result = corr_matrix[0, 1]
        
        return self._wrap_result(np.asarray(result))
    
    def clip(self, value: Union[float, LazyArray], min_val: float, max_val: float) -> LazyArray:
        """Clip a value to be within the specified range."""
        if min_val > max_val:
            raise ValueError(f"min_val ({min_val}) must be <= max_val ({max_val})")
        
        if isinstance(value, LazyNumPyArray):
            arr = value.value
        elif isinstance(value, (int, float)):
            arr = np.asarray(value)
        else:
            arr = np.asarray(value)
        
        result = np.clip(arr, min_val, max_val)
        return self._wrap_result(result)
    
    def softmax(self, values: Union[List[float], LazyArray]) -> LazyArray:
        """Apply softmax function to convert values to probabilities."""
        if isinstance(values, LazyNumPyArray):
            arr = values.value
        elif isinstance(values, np.ndarray):
            arr = values
        else:
            arr = np.asarray(values)
        
        if arr.size == 0:
            raise ValueError("Cannot apply softmax to empty array")
        
        # Numerically stable softmax
        exp_values = np.exp(arr - np.max(arr))
        result = exp_values / np.sum(exp_values)
        
        return self._wrap_result(result)
    
    def stack_arrays(self, arrays: List[LazyArray]) -> LazyArray:
        """Stack a list of arrays along a new axis."""
        if not arrays:
            raise ValueError("Cannot stack empty list of arrays")
        
        # Convert all arrays to numpy
        np_arrays = []
        for arr in arrays:
            if isinstance(arr, LazyNumPyArray):
                np_arrays.append(arr.value)
            elif isinstance(arr, np.ndarray):
                np_arrays.append(arr)
            else:
                np_arrays.append(np.asarray(arr))
        
        # Check shapes are compatible
        shapes = [arr.shape for arr in np_arrays]
        if not all(shape == shapes[0] for shape in shapes):
            raise ValueError(f"All arrays must have the same shape, got shapes: {shapes}")
        
        result = np.stack(np_arrays)
        return self._wrap_result(result)
    
    def array_mean(self, array: LazyArray, axis: Optional[int] = None) -> LazyArray:
        """Calculate mean of an array along specified axis."""
        if isinstance(array, LazyNumPyArray):
            arr = array.value
        else:
            arr = np.asarray(array)
        
        if axis is not None and (axis < -arr.ndim or axis >= arr.ndim):
            raise ValueError(f"Axis {axis} is out of bounds for array of dimension {arr.ndim}")
        
        result = np.mean(arr, axis=axis)
        return self._wrap_result(result)
    
    def array_sum(self, array: LazyArray, axis: Optional[int] = None) -> LazyArray:
        """Calculate sum of an array along specified axis."""
        if isinstance(array, LazyNumPyArray):
            arr = array.value
        else:
            arr = np.asarray(array)
        
        if axis is not None and (axis < -arr.ndim or axis >= arr.ndim):
            raise ValueError(f"Axis {axis} is out of bounds for array of dimension {arr.ndim}")
        
        result = np.sum(arr, axis=axis)
        return self._wrap_result(result)
    
    def array_std(self, array: LazyArray, axis: Optional[int] = None, ddof: int = 0) -> LazyArray:
        """Calculate standard deviation of an array along specified axis."""
        if isinstance(array, LazyNumPyArray):
            arr = array.value
        else:
            arr = np.asarray(array)
        
        if axis is not None and (axis < -arr.ndim or axis >= arr.ndim):
            raise ValueError(f"Axis {axis} is out of bounds for array of dimension {arr.ndim}")
        
        result = np.std(arr, axis=axis, ddof=ddof)
        return self._wrap_result(result)
    
    def array_shape(self, array: LazyArray) -> Tuple[int, ...]:
        """Get the shape of an array."""
        if isinstance(array, LazyNumPyArray):
            arr = array.value
        else:
            arr = np.asarray(array)
        
        return arr.shape
    
    def zeros_like(self, array: LazyArray) -> LazyArray:
        """Create an array of zeros with the same shape as input."""
        if isinstance(array, LazyNumPyArray):
            arr = array.value
        else:
            arr = np.asarray(array)
        
        result = np.zeros_like(arr)
        return self._wrap_result(result)
    
    def ones_like(self, array: LazyArray) -> LazyArray:
        """Create an array of ones with the same shape as input."""
        if isinstance(array, LazyNumPyArray):
            arr = array.value
        else:
            arr = np.asarray(array)
        
        result = np.ones_like(arr)
        return self._wrap_result(result)
    
    def array_max(self, array: LazyArray, axis: Optional[int] = None) -> LazyArray:
        """Find maximum value in array along specified axis."""
        if isinstance(array, LazyNumPyArray):
            arr = array.value
        else:
            arr = np.asarray(array)
        
        if axis is not None and (axis < -arr.ndim or axis >= arr.ndim):
            raise ValueError(f"Axis {axis} is out of bounds for array of dimension {arr.ndim}")
        
        result = np.max(arr, axis=axis)
        return self._wrap_result(result)
    
    def array_min(self, array: LazyArray, axis: Optional[int] = None) -> LazyArray:
        """Find minimum value in array along specified axis."""
        if isinstance(array, LazyNumPyArray):
            arr = array.value
        else:
            arr = np.asarray(array)
        
        if axis is not None and (axis < -arr.ndim or axis >= arr.ndim):
            raise ValueError(f"Axis {axis} is out of bounds for array of dimension {arr.ndim}")
        
        result = np.min(arr, axis=axis)
        return self._wrap_result(result)
    
    def array_clip(self, array: LazyArray, min_val: float, max_val: float) -> LazyArray:
        """Clip all values in an array to specified range."""
        if min_val > max_val:
            raise ValueError(f"min_val ({min_val}) must be <= max_val ({max_val})")
        
        if isinstance(array, LazyNumPyArray):
            arr = array.value
        else:
            arr = np.asarray(array)
        
        result = np.clip(arr, min_val, max_val)
        return self._wrap_result(result)
    
    def array_multiply(self, array: LazyArray, scalar: float) -> LazyArray:
        """Multiply array by scalar value."""
        if isinstance(array, LazyNumPyArray):
            arr = array.value
        else:
            arr = np.asarray(array)
        
        result = arr * scalar
        return self._wrap_result(result)
    
    def array_add(self, array1: LazyArray, array2: LazyArray) -> LazyArray:
        """Element-wise addition of two arrays."""
        if isinstance(array1, LazyNumPyArray):
            arr1 = array1.value
        else:
            arr1 = np.asarray(array1)
        
        if isinstance(array2, LazyNumPyArray):
            arr2 = array2.value
        else:
            arr2 = np.asarray(array2)
        
        try:
            result = arr1 + arr2
        except ValueError as e:
            raise ValueError(f"Arrays have incompatible shapes: {arr1.shape} and {arr2.shape}") from e
        
        return self._wrap_result(result)
    
    def array_log(self, array: LazyArray, epsilon: float = 1e-15) -> LazyArray:
        """Natural logarithm of array elements with numerical stability."""
        if isinstance(array, LazyNumPyArray):
            arr = array.value
        else:
            arr = np.asarray(array)
        
        # Add epsilon for numerical stability and handle negative values
        # Use maximum to ensure we never take log of negative values
        safe_arr = np.maximum(arr, epsilon)
        result = np.log(safe_arr)
        return self._wrap_result(result)
    
    def array_power(self, array: LazyArray, exponent: float) -> LazyArray:
        """Raise array elements to specified power."""
        if isinstance(array, LazyNumPyArray):
            arr = array.value
        else:
            arr = np.asarray(array)
        
        result = np.power(arr, exponent)
        return self._wrap_result(result)
    
    def array_argsort(self, array: LazyArray, axis: Optional[int] = -1) -> LazyArray:
        """Get indices that would sort an array."""
        if isinstance(array, LazyNumPyArray):
            arr = array.value
        else:
            arr = np.asarray(array)
        
        if axis is not None and (axis < -arr.ndim or axis >= arr.ndim):
            raise ValueError(f"Axis {axis} is out of bounds for array of dimension {arr.ndim}")
        
        result = np.argsort(arr, axis=axis)
        return self._wrap_result(result)
    
    def array_corrcoef(self, x: LazyArray, y: Optional[LazyArray] = None) -> LazyArray:
        """Calculate correlation coefficient matrix."""
        if isinstance(x, LazyNumPyArray):
            x_arr = x.value
        else:
            x_arr = np.asarray(x)
        
        if y is not None:
            if isinstance(y, LazyNumPyArray):
                y_arr = y.value
            else:
                y_arr = np.asarray(y)
            
            # Stack arrays for corrcoef
            result = np.corrcoef(x_arr, y_arr)
        else:
            result = np.corrcoef(x_arr)
        
        return self._wrap_result(result)
    
    def maximum(self, array: LazyArray, value: float) -> LazyArray:
        """Element-wise maximum of array and scalar."""
        if isinstance(array, LazyNumPyArray):
            arr = array.value
        else:
            arr = np.asarray(array)
        
        result = np.maximum(arr, value)
        return self._wrap_result(result)
    
    # Additional lazy evaluation helpers
    
    def is_lazy(self, array: Any) -> bool:
        """Check if an array is lazy (not yet evaluated)."""
        return isinstance(array, LazyNumPyArray)
    
    def batch_eval(self, computations: List[LazyArray]) -> List[EagerArray]:
        """Evaluate multiple lazy computations efficiently."""
        results = []
        for comp in computations:
            if isinstance(comp, LazyNumPyArray):
                results.append(comp.eval())
            elif isinstance(comp, np.ndarray):
                results.append(comp)
            else:
                results.append(np.asarray(comp))
        return results
    
    def create_computation_graph(self, fn: Callable[..., LazyArray]) -> ComputationGraph:
        """Create a computation graph from a function."""
        # NumPy doesn't support computation graphs, return the function itself
        warnings.warn("NumPy doesn't support computation graphs, returning function as-is")
        return fn
    
    def optimize_graph(self, graph: ComputationGraph) -> ComputationGraph:
        """Optimize a computation graph for better performance."""
        # No optimization possible with NumPy
        warnings.warn("NumPy doesn't support graph optimization, returning graph as-is")
        return graph