"""Memory pooling for MLX arrays to reduce allocations.

This module provides a simple memory pool for reusing MLX arrays
to reduce allocation overhead during training.
"""

from collections import defaultdict

import mlx.core as mx
import mlx.nn as nn
from loguru import logger


class ArrayPool:
    """Memory pool for reusing MLX arrays.

    This pool maintains a collection of pre-allocated arrays that can be
    reused instead of allocating new ones. This is particularly useful
    for temporary arrays used in the training loop.
    """

    def __init__(self, enabled: bool = True):
        """Initialize the array pool.

        Args:
            enabled: Whether pooling is enabled
        """
        self.enabled = enabled
        self._pool: dict[tuple[tuple[int, ...], mx.Dtype], list[mx.array]] = (
            defaultdict(list)
        )
        self._stats = {
            "hits": 0,
            "misses": 0,
            "allocations": 0,
            "returns": 0,
        }

    def get(self, shape: tuple[int, ...], dtype: mx.Dtype = mx.float32) -> mx.array:
        """Get an array from the pool or allocate a new one.

        Args:
            shape: Shape of the array
            dtype: Data type of the array

        Returns:
            MLX array with the requested shape and dtype
        """
        if not self.enabled:
            return mx.zeros(shape, dtype=dtype)

        key = (shape, dtype)
        if self._pool[key]:
            self._stats["hits"] += 1
            return self._pool[key].pop()
        else:
            self._stats["misses"] += 1
            self._stats["allocations"] += 1
            return mx.zeros(shape, dtype=dtype)

    def return_array(self, array: mx.array) -> None:
        """Return an array to the pool for reuse.

        Args:
            array: Array to return to the pool
        """
        if not self.enabled:
            return

        key = (array.shape, array.dtype)
        self._pool[key].append(array)
        self._stats["returns"] += 1

    def clear(self) -> None:
        """Clear all arrays from the pool."""
        self._pool.clear()
        logger.debug(f"Array pool cleared. Stats: {self._stats}")

    def get_stats(self) -> dict[str, int]:
        """Get pool statistics."""
        return self._stats.copy()

    def reset_stats(self) -> None:
        """Reset pool statistics."""
        self._stats = {
            "hits": 0,
            "misses": 0,
            "allocations": 0,
            "returns": 0,
        }


class GradientPool:
    """Specialized pool for gradient arrays.

    This pool is optimized for gradient accumulation where we need
    arrays with the same shapes as model parameters.
    """

    def __init__(self, model_shapes: dict[str, tuple[int, ...]], enabled: bool = True):
        """Initialize gradient pool.

        Args:
            model_shapes: Dictionary mapping parameter names to shapes
            enabled: Whether pooling is enabled
        """
        self.enabled = enabled
        self.model_shapes = model_shapes
        self._pool: dict[str, list[mx.array]] = defaultdict(list)
        self._preallocated: dict[str, mx.array] = {}

        if enabled:
            # Pre-allocate gradient arrays
            self._preallocate()

    def _preallocate(self) -> None:
        """Pre-allocate gradient arrays for all parameters."""
        for name, shape in self.model_shapes.items():
            # Pre-allocate 2 arrays per parameter (for double buffering)
            self._pool[name].extend(
                [mx.zeros(shape, dtype=mx.float32), mx.zeros(shape, dtype=mx.float32)]
            )
        logger.debug(
            f"Pre-allocated gradient arrays for {len(self.model_shapes)} parameters"
        )

    def get_gradient_dict(self) -> dict[str, mx.array]:
        """Get a dictionary of gradient arrays.

        Returns:
            Dictionary mapping parameter names to arrays
        """
        if not self.enabled:
            return {
                name: mx.zeros(shape, dtype=mx.float32)
                for name, shape in self.model_shapes.items()
            }

        result = {}
        for name, shape in self.model_shapes.items():
            if self._pool[name]:
                result[name] = self._pool[name].pop()
            else:
                # Allocate new array if pool is empty
                result[name] = mx.zeros(shape, dtype=mx.float32)

        return result

    def return_gradient_dict(self, gradients: dict[str, mx.array]) -> None:
        """Return gradient arrays to the pool.

        Args:
            gradients: Dictionary of gradient arrays to return
        """
        if not self.enabled:
            return

        for name, array in gradients.items():
            if name in self.model_shapes:
                self._pool[name].append(array)


def create_memory_pools(
    model: nn.Module, config: dict | None = None
) -> tuple[ArrayPool, GradientPool]:
    """Create memory pools for training.

    Args:
        model: Model to get parameter shapes from
        config: Optional configuration dictionary

    Returns:
        Tuple of (array_pool, gradient_pool)
    """
    config = config or {}
    enabled = config.get("enable_memory_pooling", True)

    # Get model parameter shapes
    model_shapes = {}
    params = model.parameters()

    # Flatten nested parameter dict
    def flatten_params(params, prefix=""):
        shapes = {}
        for key, value in params.items():
            if isinstance(value, dict):
                shapes.update(flatten_params(value, prefix + key + "."))
            elif isinstance(value, list):
                # Handle lists of parameters (e.g., encoder layers)
                for i, item in enumerate(value):
                    if isinstance(item, dict):
                        shapes.update(flatten_params(item, f"{prefix}{key}.{i}."))
                    elif hasattr(item, "shape"):
                        shapes[f"{prefix}{key}.{i}"] = item.shape
            elif hasattr(value, "shape"):
                shapes[prefix + key] = value.shape
        return shapes

    model_shapes = flatten_params(params)

    # Create pools
    array_pool = ArrayPool(enabled=enabled)
    gradient_pool = GradientPool(model_shapes, enabled=enabled)

    logger.info(
        f"Created memory pools (enabled={enabled}) for {len(model_shapes)} parameters"
    )

    return array_pool, gradient_pool
