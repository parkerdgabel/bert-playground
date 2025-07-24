"""MLX utility functions for compute operations."""

from typing import Any, Dict, Optional, Union, List
import numpy as np
import mlx.core as mx
import mlx.utils as mlx_utils
import psutil
import platform


def convert_to_mlx_array(
    data: Any,
    dtype: Optional[mx.Dtype] = None
) -> mx.array:
    """Convert various data types to MLX array.
    
    Args:
        data: Input data (list, numpy array, etc.)
        dtype: Target MLX dtype
        
    Returns:
        MLX array
    """
    if isinstance(data, mx.array):
        if dtype is not None and data.dtype != dtype:
            return data.astype(dtype)
        return data
    
    # Convert to numpy first if needed
    if isinstance(data, list):
        np_array = np.array(data)
    elif isinstance(data, np.ndarray):
        np_array = data
    else:
        np_array = np.array(data)
    
    # Create MLX array
    mlx_array = mx.array(np_array)
    
    # Convert dtype if specified
    if dtype is not None:
        mlx_array = mlx_array.astype(dtype)
    
    return mlx_array


def convert_from_mlx_array(
    array: mx.array,
    to_numpy: bool = True
) -> Union[np.ndarray, List]:
    """Convert MLX array to numpy or list.
    
    Args:
        array: MLX array
        to_numpy: If True, return numpy array; else return list
        
    Returns:
        Numpy array or list
    """
    # First convert to numpy
    np_array = np.array(array)
    
    if to_numpy:
        return np_array
    else:
        return np_array.tolist()


def get_mlx_dtype(dtype_str: str) -> mx.Dtype:
    """Get MLX dtype from string representation.
    
    Args:
        dtype_str: String dtype (e.g., "float32", "int32")
        
    Returns:
        MLX dtype
    """
    dtype_map = {
        "float32": mx.float32,
        "float16": mx.float16,
        "bfloat16": mx.bfloat16,
        "int32": mx.int32,
        "int64": mx.int64,
        "int8": mx.int8,
        "uint8": mx.uint8,
        "bool": mx.bool_,
    }
    
    if dtype_str not in dtype_map:
        raise ValueError(f"Unknown dtype: {dtype_str}")
    
    return dtype_map[dtype_str]


def get_mlx_device_info() -> Dict[str, Any]:
    """Get information about MLX compute device.
    
    Returns:
        Dictionary with device information
    """
    # Get system information
    system_info = platform.uname()
    
    # Get memory information
    memory = psutil.virtual_memory()
    
    # Check if we're on Apple Silicon
    is_apple_silicon = (
        system_info.system == "Darwin" and 
        system_info.machine in ["arm64", "aarch64"]
    )
    
    device_info = {
        "device_type": "gpu" if is_apple_silicon else "cpu",
        "device_name": "Apple Silicon GPU" if is_apple_silicon else "CPU",
        "memory_total": memory.total,
        "memory_available": memory.available,
        "memory_used": memory.used,
        "memory_percent": memory.percent,
        "platform": system_info.system,
        "machine": system_info.machine,
        "processor": system_info.processor,
    }
    
    # Add MLX-specific information
    device_info["mlx_version"] = getattr(mx, "__version__", "unknown")
    device_info["supports_metal"] = is_apple_silicon
    device_info["unified_memory"] = is_apple_silicon
    
    return device_info


def create_attention_mask(
    input_ids: mx.array,
    pad_token_id: int = 0
) -> mx.array:
    """Create attention mask from input IDs.
    
    Args:
        input_ids: Input token IDs
        pad_token_id: Padding token ID
        
    Returns:
        Attention mask (1 for real tokens, 0 for padding)
    """
    return (input_ids != pad_token_id).astype(mx.int32)


def create_causal_mask(
    seq_length: int,
    dtype: mx.Dtype = mx.float32
) -> mx.array:
    """Create causal attention mask.
    
    Args:
        seq_length: Sequence length
        dtype: Data type for mask
        
    Returns:
        Causal mask
    """
    # Create lower triangular matrix
    mask = mx.tril(mx.ones((seq_length, seq_length), dtype=dtype))
    return mask


def apply_rotary_embeddings(
    query: mx.array,
    key: mx.array,
    cos: mx.array,
    sin: mx.array
) -> tuple[mx.array, mx.array]:
    """Apply rotary position embeddings to query and key.
    
    Args:
        query: Query tensor [batch_size, num_heads, seq_len, head_dim]
        key: Key tensor [batch_size, num_heads, seq_len, head_dim]
        cos: Cosine values for rotation
        sin: Sine values for rotation
        
    Returns:
        Rotated query and key tensors
    """
    # Reshape for rotation
    q_shape = query.shape
    k_shape = key.shape
    
    # Split last dimension for rotation
    query_rot = query.reshape(*q_shape[:-1], -1, 2)
    key_rot = key.reshape(*k_shape[:-1], -1, 2)
    
    # Apply rotation
    q_cos = query_rot * cos.reshape(1, 1, -1, 1, 1)
    q_sin = mx.stack([-query_rot[..., 1], query_rot[..., 0]], axis=-1) * sin.reshape(1, 1, -1, 1, 1)
    query_rot = q_cos + q_sin
    
    k_cos = key_rot * cos.reshape(1, 1, -1, 1, 1)
    k_sin = mx.stack([-key_rot[..., 1], key_rot[..., 0]], axis=-1) * sin.reshape(1, 1, -1, 1, 1)
    key_rot = k_cos + k_sin
    
    # Reshape back
    query_rot = query_rot.reshape(q_shape)
    key_rot = key_rot.reshape(k_shape)
    
    return query_rot, key_rot


def chunk_array(
    array: mx.array,
    chunk_size: int,
    dim: int = 0
) -> List[mx.array]:
    """Split array into chunks along specified dimension.
    
    Args:
        array: Input array
        chunk_size: Size of each chunk
        dim: Dimension to split along
        
    Returns:
        List of array chunks
    """
    shape = array.shape
    num_chunks = (shape[dim] + chunk_size - 1) // chunk_size
    
    chunks = []
    for i in range(num_chunks):
        start = i * chunk_size
        end = min((i + 1) * chunk_size, shape[dim])
        
        # Create slice for indexing
        indices = [slice(None)] * len(shape)
        indices[dim] = slice(start, end)
        
        chunk = array[tuple(indices)]
        chunks.append(chunk)
    
    return chunks


def tree_map_with_path(
    f: callable,
    tree: Any,
    *rest: Any,
    is_leaf: Optional[callable] = None,
    path: tuple = ()
) -> Any:
    """Map function over tree structure with path information.
    
    Args:
        f: Function to apply (receives path as first argument)
        tree: Tree structure
        *rest: Additional trees
        is_leaf: Function to determine if node is leaf
        path: Current path in tree
        
    Returns:
        Mapped tree
    """
    if is_leaf is not None and is_leaf(tree):
        return f(path, tree, *rest)
    
    if isinstance(tree, dict):
        return {
            k: tree_map_with_path(
                f, v, *(r[k] for r in rest), 
                is_leaf=is_leaf, 
                path=path + (k,)
            )
            for k, v in tree.items()
        }
    elif isinstance(tree, (list, tuple)):
        mapped = [
            tree_map_with_path(
                f, v, *(r[i] for r in rest),
                is_leaf=is_leaf,
                path=path + (i,)
            )
            for i, v in enumerate(tree)
        ]
        return type(tree)(mapped)
    else:
        return f(path, tree, *rest)