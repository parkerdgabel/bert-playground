"""
MLX dtype utilities for consistent type handling.
Provides conversion and validation functions for MLX arrays.
"""

from typing import Any, Dict, List, Optional, Union, Tuple
import numpy as np
import mlx.core as mx
from loguru import logger


# Standard dtype mappings for different data types
DTYPE_MAPPINGS = {
    "input_ids": mx.int32,
    "attention_mask": mx.float32,  # float32 for MLX compatibility
    "token_type_ids": mx.int32,
    "position_ids": mx.int32,
    "label": mx.int32,
    "labels": mx.int32,
    "features": mx.float32,
    "embeddings": mx.float32,
    "logits": mx.float32,
    "probabilities": mx.float32,
    "indices": mx.int32,
    "mask": mx.float32,
}

# Numpy to MLX dtype mappings
NP_TO_MLX_DTYPE = {
    np.float16: mx.float16,
    np.float32: mx.float32,
    np.float64: mx.float32,  # Downcast to float32
    np.int8: mx.int8,
    np.int16: mx.int16,
    np.int32: mx.int32,
    np.int64: mx.int32,  # Downcast to int32
    np.uint8: mx.uint8,
    np.uint16: mx.uint16,
    np.uint32: mx.uint32,
    np.uint64: mx.uint32,  # Downcast to uint32
    np.bool_: mx.bool_,
}


def infer_dtype(value: Any, field_name: Optional[str] = None) -> mx.Dtype:
    """
    Infer appropriate MLX dtype for a value.
    
    Args:
        value: Value to infer dtype for
        field_name: Optional field name for standard mappings
        
    Returns:
        MLX dtype
    """
    # Check standard mappings first
    if field_name and field_name in DTYPE_MAPPINGS:
        return DTYPE_MAPPINGS[field_name]
    
    # Infer from value type
    if isinstance(value, (mx.array, np.ndarray)):
        if hasattr(value, 'dtype'):
            np_dtype = np.dtype(value.dtype)
            return NP_TO_MLX_DTYPE.get(np_dtype.type, mx.float32)
    
    elif isinstance(value, bool):
        return mx.bool_
    
    elif isinstance(value, int):
        # Check range to determine appropriate int type
        if -128 <= value <= 127:
            return mx.int8
        elif -32768 <= value <= 32767:
            return mx.int16
        else:
            return mx.int32
    
    elif isinstance(value, float):
        return mx.float32
    
    elif isinstance(value, (list, tuple)):
        if value:
            # Infer from first element
            return infer_dtype(value[0], field_name)
    
    # Default to float32
    return mx.float32


def convert_to_mlx(
    data: Any,
    dtype: Optional[mx.Dtype] = None,
    field_name: Optional[str] = None
) -> mx.array:
    """
    Convert data to MLX array with appropriate dtype.
    
    Args:
        data: Data to convert
        dtype: Explicit dtype (if None, will infer)
        field_name: Field name for dtype inference
        
    Returns:
        MLX array
    """
    # Determine dtype
    if dtype is None:
        dtype = infer_dtype(data, field_name)
    
    # Handle different input types
    if isinstance(data, mx.array):
        if data.dtype == dtype:
            return data
        else:
            # Cast to desired dtype
            return data.astype(dtype)
    
    elif isinstance(data, np.ndarray):
        return mx.array(data, dtype=dtype)
    
    elif isinstance(data, (list, tuple)):
        return mx.array(data, dtype=dtype)
    
    elif isinstance(data, (int, float, bool)):
        return mx.array([data], dtype=dtype).squeeze()
    
    else:
        raise TypeError(f"Cannot convert type {type(data)} to MLX array")


def ensure_dtype_consistency(
    batch: Dict[str, Any],
    dtype_spec: Optional[Dict[str, mx.Dtype]] = None
) -> Dict[str, mx.array]:
    """
    Ensure all arrays in batch have consistent dtypes.
    
    Args:
        batch: Batch dictionary
        dtype_spec: Explicit dtype specifications
        
    Returns:
        Batch with consistent dtypes
    """
    dtype_spec = dtype_spec or {}
    result = {}
    
    for key, value in batch.items():
        # Determine target dtype
        if key in dtype_spec:
            target_dtype = dtype_spec[key]
        elif key in DTYPE_MAPPINGS:
            target_dtype = DTYPE_MAPPINGS[key]
        else:
            target_dtype = None
        
        # Convert to MLX array
        if isinstance(value, (mx.array, np.ndarray, list, tuple)):
            result[key] = convert_to_mlx(value, target_dtype, key)
        else:
            # Keep non-array values as is
            result[key] = value
    
    return result


def validate_dtypes(
    data: Dict[str, mx.array],
    expected_dtypes: Optional[Dict[str, mx.Dtype]] = None,
    strict: bool = False
) -> Tuple[bool, List[str]]:
    """
    Validate dtypes of arrays in data.
    
    Args:
        data: Data dictionary
        expected_dtypes: Expected dtypes
        strict: Whether to enforce exact dtype match
        
    Returns:
        Tuple of (is_valid, error_messages)
    """
    expected_dtypes = expected_dtypes or DTYPE_MAPPINGS
    errors = []
    
    for key, array in data.items():
        if not isinstance(array, mx.array):
            if strict:
                errors.append(f"Field '{key}' is not an MLX array")
            continue
        
        if key in expected_dtypes:
            expected = expected_dtypes[key]
            actual = array.dtype
            
            if actual != expected:
                if strict:
                    errors.append(f"Field '{key}' has dtype {actual}, expected {expected}")
                else:
                    # Check if conversion is safe
                    if not is_safe_cast(actual, expected):
                        errors.append(f"Field '{key}' has dtype {actual}, cannot safely cast to {expected}")
    
    return len(errors) == 0, errors


def is_safe_cast(from_dtype: mx.Dtype, to_dtype: mx.Dtype) -> bool:
    """
    Check if casting from one dtype to another is safe.
    
    Args:
        from_dtype: Source dtype
        to_dtype: Target dtype
        
    Returns:
        Whether cast is safe
    """
    # Same dtype is always safe
    if from_dtype == to_dtype:
        return True
    
    # Define safe casting rules
    safe_casts = {
        (mx.int8, mx.int16): True,
        (mx.int8, mx.int32): True,
        (mx.int16, mx.int32): True,
        (mx.uint8, mx.uint16): True,
        (mx.uint8, mx.uint32): True,
        (mx.uint16, mx.uint32): True,
        (mx.float16, mx.float32): True,
        (mx.int32, mx.float32): True,  # Common for attention masks
        (mx.bool_, mx.float32): True,
        (mx.bool_, mx.int32): True,
    }
    
    return safe_casts.get((from_dtype, to_dtype), False)


def optimize_dtype_for_memory(
    data: mx.array,
    field_name: Optional[str] = None,
    preserve_precision: bool = True
) -> mx.array:
    """
    Optimize array dtype for memory efficiency.
    
    Args:
        data: Array to optimize
        field_name: Field name for context
        preserve_precision: Whether to preserve numerical precision
        
    Returns:
        Array with optimized dtype
    """
    current_dtype = data.dtype
    
    # Don't optimize if field has standard dtype
    if field_name and field_name in DTYPE_MAPPINGS:
        return data
    
    # For integer types, find smallest that fits
    if current_dtype in [mx.int32, mx.int64]:
        min_val = mx.min(data).item()
        max_val = mx.max(data).item()
        
        if -128 <= min_val and max_val <= 127:
            return data.astype(mx.int8)
        elif -32768 <= min_val and max_val <= 32767:
            return data.astype(mx.int16)
    
    # For float types, consider float16 if appropriate
    elif current_dtype == mx.float32 and not preserve_precision:
        # Check if values are in float16 range
        max_abs = mx.max(mx.abs(data)).item()
        if max_abs < 65504:  # Max float16 value
            return data.astype(mx.float16)
    
    return data


def create_dtype_spec(
    sample_batch: Dict[str, Any],
    overrides: Optional[Dict[str, mx.Dtype]] = None
) -> Dict[str, mx.Dtype]:
    """
    Create dtype specification from sample batch.
    
    Args:
        sample_batch: Sample batch to analyze
        overrides: Explicit dtype overrides
        
    Returns:
        Dtype specification
    """
    dtype_spec = {}
    
    for key, value in sample_batch.items():
        if isinstance(value, (mx.array, np.ndarray, list, tuple)):
            # Use override if available
            if overrides and key in overrides:
                dtype_spec[key] = overrides[key]
            else:
                # Infer from data
                dtype_spec[key] = infer_dtype(value, key)
    
    return dtype_spec


def cast_batch_dtypes(
    batch: Dict[str, mx.array],
    dtype_spec: Dict[str, mx.Dtype]
) -> Dict[str, mx.array]:
    """
    Cast all arrays in batch to specified dtypes.
    
    Args:
        batch: Batch to cast
        dtype_spec: Target dtypes
        
    Returns:
        Batch with casted arrays
    """
    result = {}
    
    for key, value in batch.items():
        if key in dtype_spec and isinstance(value, mx.array):
            target_dtype = dtype_spec[key]
            if value.dtype != target_dtype:
                result[key] = value.astype(target_dtype)
            else:
                result[key] = value
        else:
            result[key] = value
    
    return result


def get_memory_usage(data: Union[mx.array, Dict[str, mx.array]]) -> int:
    """
    Calculate memory usage of arrays in bytes.
    
    Args:
        data: Array or dictionary of arrays
        
    Returns:
        Memory usage in bytes
    """
    if isinstance(data, mx.array):
        return data.nbytes
    
    elif isinstance(data, dict):
        total = 0
        for value in data.values():
            if isinstance(value, mx.array):
                total += value.nbytes
        return total
    
    else:
        return 0


def format_bytes(num_bytes: int) -> str:
    """
    Format bytes as human-readable string.
    
    Args:
        num_bytes: Number of bytes
        
    Returns:
        Formatted string
    """
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if abs(num_bytes) < 1024.0:
            return f"{num_bytes:.2f} {unit}"
        num_bytes /= 1024.0
    return f"{num_bytes:.2f} PB"