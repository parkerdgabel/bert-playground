"""MLX compute backend adapter implementation."""

from typing import Any, Callable, Sequence

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from core.ports.compute import Array, ArrayLike, ComputeBackend, DataType, Device, DType, NeuralOps, Shape


class MLXComputeAdapter:
    """MLX implementation of the ComputeBackend port."""

    @property
    def name(self) -> str:
        """Name of the compute backend."""
        return "mlx"

    @property
    def supports_compilation(self) -> bool:
        """MLX supports JIT compilation."""
        return True

    def _convert_dtype(self, dtype: DataType | DType | None) -> mx.Dtype | None:
        """Convert generic dtype to MLX dtype."""
        if dtype is None:
            return None
        if isinstance(dtype, DataType):
            dtype_map = {
                DataType.FLOAT32: mx.float32,
                DataType.FLOAT16: mx.float16,
                DataType.BFLOAT16: mx.bfloat16,
                DataType.INT32: mx.int32,
                DataType.INT64: mx.int64,
                DataType.BOOL: mx.bool_,
            }
            return dtype_map.get(dtype, mx.float32)
        return dtype  # Assume it's already an MLX dtype

    def array(
        self,
        data: ArrayLike,
        dtype: DataType | DType | None = None,
        device: Device | None = None
    ) -> mx.array:
        """Create an MLX array from data."""
        mlx_dtype = self._convert_dtype(dtype)
        return mx.array(data, dtype=mlx_dtype)

    def zeros(
        self,
        shape: Shape,
        dtype: DataType | DType | None = None,
        device: Device | None = None
    ) -> mx.array:
        """Create array of zeros."""
        mlx_dtype = self._convert_dtype(dtype) or mx.float32
        return mx.zeros(shape, dtype=mlx_dtype)

    def ones(
        self,
        shape: Shape,
        dtype: DataType | DType | None = None,
        device: Device | None = None
    ) -> mx.array:
        """Create array of ones."""
        mlx_dtype = self._convert_dtype(dtype) or mx.float32
        return mx.ones(shape, dtype=mlx_dtype)

    def randn(
        self,
        shape: Shape,
        dtype: DataType | DType | None = None,
        device: Device | None = None,
        seed: int | None = None
    ) -> mx.array:
        """Create array with normal random values."""
        if seed is not None:
            mx.random.seed(seed)
        mlx_dtype = self._convert_dtype(dtype) or mx.float32
        return mx.random.normal(shape, dtype=mlx_dtype)

    def to_numpy(self, array: mx.array) -> np.ndarray:
        """Convert MLX array to numpy."""
        return np.array(array)

    def from_numpy(
        self,
        array: np.ndarray,
        dtype: DataType | DType | None = None,
        device: Device | None = None
    ) -> mx.array:
        """Create MLX array from numpy."""
        mlx_dtype = self._convert_dtype(dtype)
        return mx.array(array, dtype=mlx_dtype)

    def shape(self, array: mx.array) -> Shape:
        """Get array shape."""
        return array.shape

    def dtype(self, array: mx.array) -> mx.Dtype:
        """Get array data type."""
        return array.dtype

    def device(self, array: mx.array) -> str:
        """Get array device."""
        # MLX uses unified memory, so device is always "gpu"
        return "gpu"

    def compile(
        self,
        function: Callable[..., Any],
        static_argnums: Sequence[int] | None = None,
        static_argnames: Sequence[str] | None = None
    ) -> Callable[..., Any]:
        """Compile a function using MLX compilation."""
        # MLX uses mx.compile for JIT compilation
        return mx.compile(function)

    def gradient(
        self,
        function: Callable[..., mx.array],
        argnums: int | Sequence[int] = 0
    ) -> Callable[..., mx.array | tuple[mx.array, ...]]:
        """Create gradient function using MLX."""
        return mx.grad(function, argnums=argnums)

    def value_and_gradient(
        self,
        function: Callable[..., mx.array],
        argnums: int | Sequence[int] = 0
    ) -> Callable[..., tuple[mx.array, mx.array | tuple[mx.array, ...]]]:
        """Create function that returns both value and gradient."""
        return mx.value_and_grad(function, argnums=argnums)


class MLXNeuralOpsAdapter:
    """MLX implementation of the NeuralOps port."""

    def __init__(self, compute_backend: ComputeBackend | None = None):
        """Initialize with optional compute backend."""
        self.backend = compute_backend or MLXComputeAdapter()

    def linear(
        self,
        input: mx.array,
        weight: mx.array,
        bias: mx.array | None = None
    ) -> mx.array:
        """Linear transformation using MLX."""
        output = input @ weight.T
        if bias is not None:
            output = output + bias
        return output

    def embedding(
        self,
        input: mx.array,
        weight: mx.array,
        padding_idx: int | None = None
    ) -> mx.array:
        """Embedding lookup using MLX."""
        # MLX doesn't have a direct embedding function, so we use array indexing
        return weight[input]

    def layer_norm(
        self,
        input: mx.array,
        normalized_shape: Shape,
        weight: mx.array | None = None,
        bias: mx.array | None = None,
        eps: float = 1e-5
    ) -> mx.array:
        """Layer normalization using MLX."""
        # Use MLX's layer norm function
        return nn.LayerNorm(normalized_shape[-1], eps=eps, affine=weight is not None)(input)

    def dropout(
        self,
        input: mx.array,
        p: float = 0.5,
        training: bool = True,
        seed: int | None = None
    ) -> mx.array:
        """Dropout using MLX."""
        if not training or p == 0:
            return input
        if seed is not None:
            mx.random.seed(seed)
        return nn.Dropout(p)(input)

    def softmax(
        self,
        input: mx.array,
        dim: int = -1
    ) -> mx.array:
        """Softmax activation using MLX."""
        return mx.softmax(input, axis=dim)

    def cross_entropy(
        self,
        input: mx.array,
        target: mx.array,
        reduction: str = "mean",
        ignore_index: int = -100
    ) -> mx.array:
        """Cross entropy loss using MLX."""
        # MLX cross entropy expects logits and integer targets
        loss = nn.losses.cross_entropy(input, target, reduction="none")
        
        # Handle ignore_index
        if ignore_index != -100:
            mask = target != ignore_index
            loss = loss * mask
            
        # Apply reduction
        if reduction == "mean":
            if ignore_index != -100:
                return mx.sum(loss) / mx.sum(mask)
            return mx.mean(loss)
        elif reduction == "sum":
            return mx.sum(loss)
        else:  # reduction == "none"
            return loss

    def load_weights(self, path: str) -> dict[str, mx.array]:
        """Load weights from file using MLX."""
        return mx.load(path)

    def tree_unflatten(self, items: list[tuple[str, mx.array]]) -> dict[str, Any]:
        """Unflatten using MLX tree utilities."""
        from mlx.utils import tree_unflatten
        return tree_unflatten(items)

    def save_arrays(self, path: str, arrays: dict[str, mx.array]) -> None:
        """Save arrays using MLX safetensors."""
        mx.save_safetensors(path, arrays)

    def load_arrays(self, path: str) -> dict[str, mx.array]:
        """Load arrays using MLX."""
        return mx.load(path)