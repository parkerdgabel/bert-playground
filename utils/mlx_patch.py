"""
MLX Compatibility Patch

A simple monkey patch to fix astype() compatibility issues in mlx-embeddings.
"""

import mlx.core as mx
from loguru import logger
from typing import Any


def patch_mlx_astype():
    """
    Monkey patch MLX arrays to handle old astype() calls.
    
    The mlx-embeddings library uses the old astype() signature that
    expects different dtype formats. This patch converts them to the
    correct MLX dtypes.
    """
    # Store the original astype method
    original_astype = mx.array.astype
    
    def patched_astype(self, dtype, stream=None):
        """
        Patched astype method that converts various dtype formats to MLX dtypes.
        """
        # Convert various dtype formats to MLX dtypes
        if hasattr(dtype, 'name'):
            # Handle numpy dtypes
            dtype_name = dtype.name
        elif hasattr(dtype, '__name__'):
            # Handle type objects
            dtype_name = dtype.__name__
        elif isinstance(dtype, str):
            # Handle string dtypes
            dtype_name = dtype
        else:
            # Try to use as-is first
            try:
                return original_astype(self, dtype, stream)
            except Exception:
                # Convert to string and try again
                dtype_name = str(dtype)
        
        # Map common dtype names to MLX dtypes
        dtype_map = {
            'int32': mx.int32,
            'int64': mx.int64,
            'int16': mx.int16,
            'int8': mx.int8,
            'float32': mx.float32,
            'float64': mx.float64,
            'float16': mx.float16,
            'bool': mx.bool_,
            'bool_': mx.bool_,
            'uint32': mx.uint32,
            'uint64': mx.uint64,
            'uint16': mx.uint16,
            'uint8': mx.uint8,
        }
        
        # Try to map the dtype
        if dtype_name in dtype_map:
            mlx_dtype = dtype_map[dtype_name]
            return original_astype(self, mlx_dtype, stream)
        
        # If we can't map it, try the original call
        return original_astype(self, dtype, stream)
    
    # Replace the astype method
    mx.array.astype = patched_astype
    logger.info("Applied MLX astype() compatibility patch")


def apply_mlx_patches():
    """Apply all MLX compatibility patches."""
    logger.info("Applying MLX compatibility patches...")
    patch_mlx_astype()
    logger.info("MLX compatibility patches applied successfully")