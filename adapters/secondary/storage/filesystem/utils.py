"""Utility functions for filesystem storage operations."""

import pickle
from pathlib import Path
from typing import Any, Dict, Optional

try:
    import mlx.core as mx
    import mlx.nn as nn
    from safetensors.mlx import save_file, load_file
    MLX_AVAILABLE = True
except ImportError:
    MLX_AVAILABLE = False


def save_mlx_model(model: Any, path: Path) -> None:
    """Save MLX model weights to safetensors format.
    
    Args:
        model: MLX model to save
        path: Save path
    """
    if not MLX_AVAILABLE:
        raise ImportError("MLX and safetensors required for model saving")
    
    # Extract model parameters
    if hasattr(model, 'parameters'):
        weights = dict(model.parameters())
    elif hasattr(model, 'state_dict'):
        weights = model.state_dict()
    else:
        raise ValueError("Model must have parameters() or state_dict() method")
    
    # Save using safetensors
    save_file(weights, str(path))


def load_mlx_model(path: Path, config: Optional[Dict[str, Any]] = None) -> Any:
    """Load MLX model from safetensors format.
    
    Args:
        path: Model path
        config: Optional model configuration
        
    Returns:
        Loaded model (as dict of weights - actual model construction would be done by caller)
    """
    if not MLX_AVAILABLE:
        raise ImportError("MLX and safetensors required for model loading")
    
    # Load weights
    weights = load_file(str(path))
    
    # Return weights dict - actual model construction would be done by the caller
    # who knows the model class and can instantiate it with the config
    return {
        'weights': weights,
        'config': config,
    }


def save_optimizer_state(optimizer_state: Dict[str, Any], path: Path) -> None:
    """Save optimizer state to disk.
    
    Args:
        optimizer_state: Optimizer state dictionary
        path: Save path
    """
    # For MLX optimizers, we need to convert arrays to a serializable format
    serializable_state = {}
    
    for key, value in optimizer_state.items():
        if MLX_AVAILABLE and isinstance(value, mx.array):
            # Convert MLX array to numpy for serialization
            serializable_state[key] = {
                'type': 'mlx_array',
                'data': value.tolist(),
                'shape': value.shape,
                'dtype': str(value.dtype),
            }
        elif isinstance(value, dict):
            # Recursively handle nested dicts (e.g., parameter groups)
            serializable_state[key] = _serialize_nested_dict(value)
        else:
            serializable_state[key] = value
    
    with open(path, 'wb') as f:
        pickle.dump(serializable_state, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_optimizer_state(path: Path) -> Dict[str, Any]:
    """Load optimizer state from disk.
    
    Args:
        path: Load path
        
    Returns:
        Optimizer state dictionary
    """
    with open(path, 'rb') as f:
        serializable_state = pickle.load(f)
    
    # Convert back to MLX arrays if needed
    optimizer_state = {}
    
    for key, value in serializable_state.items():
        if isinstance(value, dict) and value.get('type') == 'mlx_array':
            if MLX_AVAILABLE:
                # Recreate MLX array
                optimizer_state[key] = mx.array(
                    value['data'],
                    dtype=getattr(mx, value['dtype'].split('.')[-1])
                )
            else:
                # Keep as numpy array if MLX not available
                optimizer_state[key] = value['data']
        elif isinstance(value, dict):
            # Recursively handle nested dicts
            optimizer_state[key] = _deserialize_nested_dict(value)
        else:
            optimizer_state[key] = value
    
    return optimizer_state


def _serialize_nested_dict(d: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively serialize nested dictionary with MLX arrays.
    
    Args:
        d: Dictionary to serialize
        
    Returns:
        Serializable dictionary
    """
    result = {}
    for key, value in d.items():
        if MLX_AVAILABLE and isinstance(value, mx.array):
            result[key] = {
                'type': 'mlx_array',
                'data': value.tolist(),
                'shape': value.shape,
                'dtype': str(value.dtype),
            }
        elif isinstance(value, dict):
            result[key] = _serialize_nested_dict(value)
        else:
            result[key] = value
    return result


def _deserialize_nested_dict(d: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively deserialize nested dictionary with MLX arrays.
    
    Args:
        d: Serialized dictionary
        
    Returns:
        Deserialized dictionary
    """
    result = {}
    for key, value in d.items():
        if isinstance(value, dict) and value.get('type') == 'mlx_array':
            if MLX_AVAILABLE:
                result[key] = mx.array(
                    value['data'],
                    dtype=getattr(mx, value['dtype'].split('.')[-1])
                )
            else:
                result[key] = value['data']
        elif isinstance(value, dict) and 'type' not in value:
            result[key] = _deserialize_nested_dict(value)
        else:
            result[key] = value
    return result


def atomic_save(save_func, path: Path, *args, **kwargs) -> None:
    """Perform atomic save operation.
    
    Args:
        save_func: Function to perform the save
        path: Target path
        *args: Arguments for save_func
        **kwargs: Keyword arguments for save_func
    """
    # Save to temporary file first
    temp_path = path.with_suffix(path.suffix + '.tmp')
    
    try:
        save_func(*args, str(temp_path), **kwargs)
        # Atomic rename
        temp_path.rename(path)
    except Exception:
        # Clean up temporary file on error
        if temp_path.exists():
            temp_path.unlink()
        raise


def calculate_model_size(weights: Dict[str, Any]) -> int:
    """Calculate total size of model weights in bytes.
    
    Args:
        weights: Model weights dictionary
        
    Returns:
        Total size in bytes
    """
    total_size = 0
    
    for key, value in weights.items():
        if MLX_AVAILABLE and isinstance(value, mx.array):
            # Calculate MLX array size
            total_size += value.nbytes
        elif hasattr(value, 'nbytes'):
            # NumPy array or similar
            total_size += value.nbytes
        elif isinstance(value, (list, tuple)):
            # Estimate size for lists/tuples
            import sys
            total_size += sys.getsizeof(value)
        else:
            # Estimate size for other objects
            import sys
            total_size += sys.getsizeof(value)
    
    return total_size