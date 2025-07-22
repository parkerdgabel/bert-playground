"""Base framework adapter protocol.

This module defines the interface that framework-specific adapters must implement
to abstract away framework details from the training logic.
"""

from abc import ABC, abstractmethod
from typing import Any, Callable, Protocol, runtime_checkable

from training.commands.base import CommandContext


@runtime_checkable
class TensorLike(Protocol):
    """Protocol for tensor-like objects."""
    
    @property
    def shape(self) -> tuple[int, ...]:
        """Tensor shape."""
        ...
    
    @property
    def dtype(self) -> Any:
        """Tensor data type."""
        ...
    
    def item(self) -> float:
        """Convert to Python scalar."""
        ...
    
    def numpy(self) -> Any:
        """Convert to numpy array."""
        ...


@runtime_checkable
class FrameworkAdapter(Protocol):
    """Protocol for framework adapters."""
    
    @property
    def name(self) -> str:
        """Framework name."""
        ...
    
    @property
    def available(self) -> bool:
        """Whether framework is available."""
        ...
    
    # Tensor operations
    def to_tensor(self, data: Any) -> TensorLike:
        """Convert data to framework tensor."""
        ...
    
    def to_python(self, tensor: TensorLike) -> float | int | list:
        """Convert tensor to Python types."""
        ...
    
    def tensor_add(self, a: TensorLike, b: TensorLike) -> TensorLike:
        """Add two tensors."""
        ...
    
    def tensor_multiply(self, a: TensorLike, b: float | TensorLike) -> TensorLike:
        """Multiply tensor by scalar or tensor."""
        ...
    
    def tensor_norm(self, tensor: TensorLike) -> float:
        """Compute tensor norm."""
        ...
    
    def tensor_clip(self, tensor: TensorLike, min_val: float, max_val: float) -> TensorLike:
        """Clip tensor values."""
        ...
    
    def evaluate_tensors(self, *tensors: TensorLike) -> None:
        """Force evaluation of lazy tensors (if applicable)."""
        ...
    
    # Gradient operations
    def create_value_and_grad_fn(
        self,
        model: Any,
        loss_fn: Callable
    ) -> Callable:
        """Create value and gradient function."""
        ...
    
    def compute_gradient_norm(self, gradients: dict[str, Any]) -> float:
        """Compute norm of gradients."""
        ...
    
    def clip_gradients_by_norm(
        self,
        gradients: dict[str, Any],
        max_norm: float
    ) -> tuple[dict[str, Any], float]:
        """Clip gradients by norm."""
        ...
    
    def clip_gradients_by_value(
        self,
        gradients: dict[str, Any],
        max_value: float
    ) -> dict[str, Any]:
        """Clip gradients by value."""
        ...
    
    def scale_gradients(
        self,
        gradients: dict[str, Any],
        scale: float
    ) -> dict[str, Any]:
        """Scale gradients by factor."""
        ...
    
    def accumulate_gradients(
        self,
        accumulated: dict[str, Any],
        current: dict[str, Any]
    ) -> dict[str, Any]:
        """Accumulate gradients."""
        ...
    
    # Model operations
    def get_model_parameters(self, model: Any) -> dict[str, Any]:
        """Get model parameters."""
        ...
    
    def update_model_parameters(
        self,
        model: Any,
        optimizer: Any,
        gradients: dict[str, Any]
    ) -> None:
        """Update model parameters with optimizer."""
        ...
    
    def get_learning_rate(self, optimizer: Any) -> float:
        """Get current learning rate from optimizer."""
        ...
    
    # Mixed precision operations
    def apply_mixed_precision(self, inputs: dict[str, Any]) -> dict[str, Any]:
        """Apply mixed precision to inputs."""
        ...
    
    def create_loss_scaler(self, initial_scale: float = 1.0) -> Any:
        """Create loss scaler for mixed precision."""
        ...
    
    # Compilation and optimization
    def compile_function(self, fn: Callable, **kwargs) -> Callable:
        """Compile function for optimization (if supported)."""
        ...
    
    def set_random_seed(self, seed: int) -> None:
        """Set random seed for reproducibility."""
        ...


class BaseFrameworkAdapter(ABC):
    """Base implementation of FrameworkAdapter with common functionality."""
    
    def __init__(self, name: str):
        """Initialize adapter.
        
        Args:
            name: Framework name
        """
        self._name = name
    
    @property
    def name(self) -> str:
        """Framework name."""
        return self._name
    
    @property
    @abstractmethod
    def available(self) -> bool:
        """Whether framework is available."""
        pass
    
    def to_python(self, tensor: TensorLike) -> float | int | list:
        """Default tensor to python conversion."""
        if hasattr(tensor, 'item') and callable(tensor.item):
            return tensor.item()
        elif hasattr(tensor, 'numpy') and callable(tensor.numpy):
            array = tensor.numpy()
            if array.ndim == 0:
                return array.item()
            return array.tolist()
        else:
            return float(tensor)
    
    def evaluate_tensors(self, *tensors: TensorLike) -> None:
        """Default implementation - no lazy evaluation."""
        pass
    
    def compile_function(self, fn: Callable, **kwargs) -> Callable:
        """Default implementation - no compilation."""
        return fn
    
    def create_loss_scaler(self, initial_scale: float = 1.0) -> Any:
        """Default implementation - no loss scaling."""
        return None
    
    def apply_mixed_precision(self, inputs: dict[str, Any]) -> dict[str, Any]:
        """Default implementation - no mixed precision."""
        return inputs