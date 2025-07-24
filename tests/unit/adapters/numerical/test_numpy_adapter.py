"""Tests for NumPy numerical operations adapter."""

import pytest
import numpy as np

from infrastructure.adapters.secondary.numerical.numpy_adapter import NumPyNumericalAdapter
from application.ports.secondary.numerical import EvaluationMode, NumericalOperations


class TestNumPyNumericalAdapter:
    """Test suite for NumPy numerical adapter."""
    
    @pytest.fixture
    def adapter(self):
        """Create adapter instance."""
        return NumPyNumericalAdapter()
    
    def test_implements_interface(self, adapter):
        """Test that adapter implements NumericalOperations interface."""
        assert isinstance(adapter, NumericalOperations)
    
    def test_capabilities(self, adapter):
        """Test adapter capabilities."""
        assert adapter.supports_lazy_evaluation is False
        assert adapter.supports_jit is False
        assert adapter.supports_autodiff is False
        assert adapter.supports_vmap is False
    
    def test_mean_eager(self, adapter):
        """Test mean calculation in eager mode."""
        adapter.set_evaluation_mode(EvaluationMode.EAGER)
        
        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        result = adapter.mean(values)
        
        assert isinstance(result, (np.ndarray, np.generic))
        assert float(result) == 3.0
    
    def test_mean_lazy(self, adapter):
        """Test mean calculation in lazy mode."""
        adapter.set_evaluation_mode(EvaluationMode.LAZY)
        
        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        lazy_result = adapter.mean(values)
        
        assert adapter.is_lazy(lazy_result)
        
        eager_result = adapter.eval(lazy_result)
        assert isinstance(eager_result, (np.ndarray, np.generic))
        assert float(eager_result) == 3.0
    
    def test_std(self, adapter):
        """Test standard deviation calculation."""
        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        
        # Population std (ddof=0)
        result = adapter.std(values, ddof=0)
        expected = np.std(values, ddof=0)
        np.testing.assert_almost_equal(adapter.eval(result), expected)
        
        # Sample std (ddof=1)
        result = adapter.std(values, ddof=1)
        expected = np.std(values, ddof=1)
        np.testing.assert_almost_equal(adapter.eval(result), expected)
    
    def test_correlation(self, adapter):
        """Test correlation calculation."""
        x = [1.0, 2.0, 3.0, 4.0, 5.0]
        y = [2.0, 4.0, 6.0, 8.0, 10.0]
        
        result = adapter.correlation(x, y)
        np.testing.assert_almost_equal(adapter.eval(result), 1.0)
    
    def test_softmax(self, adapter):
        """Test softmax calculation."""
        logits = [1.0, 2.0, 3.0]
        result = adapter.softmax(logits)
        result_eval = adapter.eval(result)
        
        # Check sum to 1
        np.testing.assert_almost_equal(np.sum(result_eval), 1.0)
        
        # Check values
        expected = np.array([0.09003057, 0.24472847, 0.66524096])
        np.testing.assert_almost_equal(result_eval, expected)
    
    def test_clip(self, adapter):
        """Test clipping values."""
        # Scalar clipping
        result = adapter.clip(5.0, 0.0, 3.0)
        assert adapter.eval(result) == 3.0
        
        result = adapter.clip(-2.0, 0.0, 3.0)
        assert adapter.eval(result) == 0.0
        
        result = adapter.clip(1.5, 0.0, 3.0)
        assert adapter.eval(result) == 1.5
    
    def test_array_operations(self, adapter):
        """Test array operations."""
        arr = np.array([[1, 2, 3], [4, 5, 6]])
        
        # Shape
        assert adapter.array_shape(arr) == (2, 3)
        
        # Sum
        total = adapter.array_sum(arr)
        assert adapter.eval(total) == 21
        
        row_sum = adapter.array_sum(arr, axis=1)
        np.testing.assert_array_equal(adapter.eval(row_sum), [6, 15])
        
        # Mean
        mean = adapter.array_mean(arr)
        assert adapter.eval(mean) == 3.5
        
        col_mean = adapter.array_mean(arr, axis=0)
        np.testing.assert_array_equal(adapter.eval(col_mean), [2.5, 3.5, 4.5])
    
    def test_stack_arrays(self, adapter):
        """Test stacking arrays."""
        arr1 = np.array([1, 2, 3])
        arr2 = np.array([4, 5, 6])
        arr3 = np.array([7, 8, 9])
        
        stacked = adapter.stack_arrays([arr1, arr2, arr3])
        result = adapter.eval(stacked)
        
        assert result.shape == (3, 3)
        np.testing.assert_array_equal(result[0], arr1)
        np.testing.assert_array_equal(result[1], arr2)
        np.testing.assert_array_equal(result[2], arr3)
    
    def test_array_arithmetic(self, adapter):
        """Test array arithmetic operations."""
        arr1 = np.array([1, 2, 3])
        arr2 = np.array([4, 5, 6])
        
        # Addition
        result = adapter.array_add(arr1, arr2)
        np.testing.assert_array_equal(adapter.eval(result), [5, 7, 9])
        
        # Multiplication by scalar
        result = adapter.array_multiply(arr1, 2.0)
        np.testing.assert_array_equal(adapter.eval(result), [2, 4, 6])
        
        # Power
        result = adapter.array_power(arr1, 2.0)
        np.testing.assert_array_equal(adapter.eval(result), [1, 4, 9])
    
    def test_array_log(self, adapter):
        """Test logarithm with numerical stability."""
        arr = np.array([1.0, 0.0, -1.0])
        
        result = adapter.array_log(arr, epsilon=1e-10)
        result_eval = adapter.eval(result)
        
        # Check that we don't get -inf for 0
        assert not np.isinf(result_eval[1])
    
    def test_zeros_ones_like(self, adapter):
        """Test zeros_like and ones_like."""
        arr = np.array([[1, 2], [3, 4]])
        
        zeros = adapter.zeros_like(arr)
        np.testing.assert_array_equal(adapter.eval(zeros), np.zeros((2, 2)))
        
        ones = adapter.ones_like(arr)
        np.testing.assert_array_equal(adapter.eval(ones), np.ones((2, 2)))
    
    def test_array_min_max(self, adapter):
        """Test min/max operations."""
        arr = np.array([[1, 5, 3], [9, 2, 7]])
        
        # Overall min/max
        assert adapter.eval(adapter.array_min(arr)) == 1
        assert adapter.eval(adapter.array_max(arr)) == 9
        
        # Along axis
        np.testing.assert_array_equal(
            adapter.eval(adapter.array_min(arr, axis=0)), [1, 2, 3]
        )
        np.testing.assert_array_equal(
            adapter.eval(adapter.array_max(arr, axis=1)), [5, 9]
        )
    
    def test_array_clip(self, adapter):
        """Test array clipping."""
        arr = np.array([[-2, 0, 2], [4, 6, 8]])
        
        result = adapter.array_clip(arr, 0, 5)
        expected = np.array([[0, 0, 2], [4, 5, 5]])
        np.testing.assert_array_equal(adapter.eval(result), expected)
    
    def test_array_argsort(self, adapter):
        """Test argsort."""
        arr = np.array([[3, 1, 4], [2, 8, 5]])
        
        result = adapter.array_argsort(arr)
        expected = np.array([[1, 0, 2], [0, 2, 1]])
        np.testing.assert_array_equal(adapter.eval(result), expected)
    
    def test_array_corrcoef(self, adapter):
        """Test correlation coefficient matrix."""
        x = np.array([1, 2, 3, 4, 5])
        y = np.array([2, 4, 6, 8, 10])
        
        result = adapter.array_corrcoef(x, y)
        result_eval = adapter.eval(result)
        
        # Perfect correlation
        np.testing.assert_almost_equal(result_eval[0, 1], 1.0)
        np.testing.assert_almost_equal(result_eval[1, 0], 1.0)
    
    def test_maximum(self, adapter):
        """Test element-wise maximum."""
        arr = np.array([1, 2, 3, 4, 5])
        
        result = adapter.maximum(arr, 3.0)
        expected = np.array([3, 3, 3, 4, 5])
        np.testing.assert_array_equal(adapter.eval(result), expected)
    
    def test_batch_eval(self, adapter):
        """Test batch evaluation of lazy arrays."""
        adapter.set_evaluation_mode(EvaluationMode.LAZY)
        
        # Create multiple lazy computations
        lazy_results = [
            adapter.mean([1, 2, 3]),
            adapter.std([4, 5, 6]),
            adapter.softmax([1, 2, 3])
        ]
        
        # Batch evaluate
        eager_results = adapter.batch_eval(lazy_results)
        
        assert len(eager_results) == 3
        assert eager_results[0] == 2.0
        assert isinstance(eager_results[1], (float, np.ndarray))
        assert isinstance(eager_results[2], np.ndarray)
    
    def test_error_handling(self, adapter):
        """Test error handling."""
        # Empty array
        with pytest.raises(ValueError, match="empty"):
            adapter.mean([])
        
        # Invalid clip range
        with pytest.raises(ValueError, match="min_val.*max_val"):
            adapter.clip(5.0, 10.0, 5.0)
        
        # Incompatible shapes for correlation
        with pytest.raises(ValueError, match="same shape"):
            adapter.correlation([1, 2, 3], [4, 5])
        
        # Incompatible shapes for addition
        arr1 = np.array([1, 2, 3])
        arr2 = np.array([[1, 2], [3, 4]])
        with pytest.raises(ValueError, match="incompatible shapes"):
            adapter.array_add(arr1, arr2)
    
    def test_autodiff_not_supported(self, adapter):
        """Test that autodiff operations raise NotImplementedError."""
        with pytest.raises(NotImplementedError, match="automatic differentiation"):
            adapter.grad(lambda x: x**2)
        
        with pytest.raises(NotImplementedError, match="automatic differentiation"):
            adapter.value_and_grad(lambda x: x**2)
    
    def test_jit_warning(self, adapter):
        """Test that JIT compilation shows warning."""
        import warnings
        
        def dummy_fn(x):
            return x * 2
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result_fn = adapter.jit(dummy_fn)
            
            assert len(w) == 1
            assert "JIT compilation" in str(w[0].message)
            assert result_fn is dummy_fn  # Returns original function