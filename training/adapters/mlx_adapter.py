"""MLX framework adapter implementation.

This module provides the MLX-specific implementation of the framework adapter,
abstracting MLX operations behind the common interface.
"""

from functools import partial
from typing import Any, Callable

from .base import BaseFrameworkAdapter, TensorLike


class MLXFrameworkAdapter(BaseFrameworkAdapter):
    """MLX-specific implementation of FrameworkAdapter."""
    
    def __init__(self):
        """Initialize MLX adapter."""
        super().__init__("MLX")
        self._mlx = None
        self._nn = None
        self._init_mlx()
    
    def _init_mlx(self) -> None:
        """Initialize MLX modules."""
        try:
            import mlx.core as mx
            import mlx.nn as nn
            self._mlx = mx
            self._nn = nn
        except ImportError:
            self._mlx = None
            self._nn = None
    
    @property
    def available(self) -> bool:
        """Whether MLX is available."""
        return self._mlx is not None
    
    def to_tensor(self, data: Any) -> TensorLike:
        """Convert data to MLX array."""
        if not self.available:
            raise RuntimeError("MLX not available")
        
        if isinstance(data, self._mlx.array):
            return data
        return self._mlx.array(data)
    
    def tensor_add(self, a: TensorLike, b: TensorLike) -> TensorLike:
        """Add two MLX tensors."""
        return a + b
    
    def tensor_multiply(self, a: TensorLike, b: float | TensorLike) -> TensorLike:
        """Multiply MLX tensor."""
        return a * b
    
    def tensor_norm(self, tensor: TensorLike) -> float:
        """Compute MLX tensor norm."""
        if hasattr(tensor, 'shape') and len(tensor.shape) > 0:
            flat_tensor = self._mlx.reshape(tensor, [-1])
            norm = self._mlx.sqrt(self._mlx.sum(self._mlx.square(flat_tensor)))
        else:
            norm = self._mlx.abs(tensor)
        
        return float(norm.item())
    
    def tensor_clip(self, tensor: TensorLike, min_val: float, max_val: float) -> TensorLike:
        """Clip MLX tensor values."""
        return self._mlx.clip(tensor, min_val, max_val)
    
    def evaluate_tensors(self, *tensors: TensorLike) -> None:
        """Force evaluation of MLX tensors."""
        if self._mlx is not None:
            self._mlx.eval(tensors)
    
    def create_value_and_grad_fn(
        self,
        model: Any,
        loss_fn: Callable
    ) -> Callable:
        """Create MLX value and gradient function."""
        if not self.available:
            raise RuntimeError("MLX not available")
        
        return self._mlx.value_and_grad(loss_fn)
    
    def compute_gradient_norm(self, gradients: dict[str, Any]) -> float:
        """Compute norm of MLX gradients."""
        if not gradients:
            return 0.0
        
        # Flatten all gradients
        grad_values = []
        for grad in gradients.values():
            if isinstance(grad, dict):
                # Recursive case for nested gradients
                for v in grad.values():
                    if v is not None:
                        grad_values.append(self._mlx.reshape(v, [-1]))
            elif grad is not None:
                grad_values.append(self._mlx.reshape(grad, [-1]))
        
        if not grad_values:
            return 0.0
        
        # Concatenate all gradients and compute norm
        all_grads = self._mlx.concatenate(grad_values)
        grad_norm = self._mlx.sqrt(self._mlx.sum(self._mlx.square(all_grads)))
        
        return float(grad_norm.item())
    
    def clip_gradients_by_norm(
        self,
        gradients: dict[str, Any],
        max_norm: float
    ) -> tuple[dict[str, Any], float]:
        """Clip MLX gradients by norm."""
        grad_norm = self.compute_gradient_norm(gradients)
        
        if grad_norm <= max_norm:
            return gradients, grad_norm
        
        # Scale gradients
        scale = max_norm / grad_norm
        clipped_grads = self.scale_gradients(gradients, scale)
        
        return clipped_grads, max_norm
    
    def clip_gradients_by_value(
        self,
        gradients: dict[str, Any],
        max_value: float
    ) -> dict[str, Any]:
        """Clip MLX gradients by value."""
        def clip_grad(grad):
            if isinstance(grad, dict):
                return {k: clip_grad(v) for k, v in grad.items()}
            elif grad is not None:
                return self._mlx.clip(grad, -max_value, max_value)
            return grad
        
        return clip_grad(gradients)
    
    def scale_gradients(
        self,
        gradients: dict[str, Any],
        scale: float
    ) -> dict[str, Any]:
        """Scale MLX gradients."""
        def scale_grad(grad):
            if isinstance(grad, dict):
                return {k: scale_grad(v) for k, v in grad.items()}
            elif grad is not None:
                return grad * scale
            return grad
        
        return scale_grad(gradients)
    
    def accumulate_gradients(
        self,
        accumulated: dict[str, Any],
        current: dict[str, Any]
    ) -> dict[str, Any]:
        """Accumulate MLX gradients."""
        result = {}
        
        # Handle all keys from both dictionaries
        all_keys = set(accumulated.keys()) | set(current.keys())
        
        for key in all_keys:
            acc_value = accumulated.get(key)
            curr_value = current.get(key)
            
            if isinstance(acc_value, dict) and isinstance(curr_value, dict):
                # Recursive case
                result[key] = self.accumulate_gradients(acc_value, curr_value)
            elif acc_value is not None and curr_value is not None:
                # Add tensors
                result[key] = acc_value + curr_value
            elif acc_value is not None:
                result[key] = acc_value
            elif curr_value is not None:
                result[key] = curr_value
            else:
                result[key] = None
        
        return result
    
    def get_model_parameters(self, model: Any) -> dict[str, Any]:
        """Get MLX model parameters."""
        if hasattr(model, 'parameters'):
            return model.parameters()
        elif hasattr(model, 'state'):
            return model.state
        else:
            return {}
    
    def update_model_parameters(
        self,
        model: Any,
        optimizer: Any,
        gradients: dict[str, Any]
    ) -> None:
        """Update MLX model parameters."""
        # MLX optimizers use the update method
        optimizer.update(model, gradients)
        
        # Ensure updates are evaluated
        if hasattr(model, 'parameters'):
            self._mlx.eval(model.parameters())
    
    def get_learning_rate(self, optimizer: Any) -> float:
        """Get learning rate from MLX optimizer."""
        if hasattr(optimizer, 'learning_rate'):
            lr = optimizer.learning_rate
            # Handle MLX array type
            if hasattr(lr, 'item'):
                return lr.item()
            return float(lr)
        return 0.0
    
    def apply_mixed_precision(self, inputs: dict[str, Any]) -> dict[str, Any]:
        """Apply MLX mixed precision (bfloat16)."""
        def convert_precision(tensor):
            if isinstance(tensor, dict):
                return {k: convert_precision(v) for k, v in tensor.items()}
            elif hasattr(tensor, 'dtype') and hasattr(tensor, 'astype'):
                # Convert float32 to bfloat16
                if tensor.dtype == self._mlx.float32:
                    return tensor.astype(self._mlx.bfloat16)
            return tensor
        
        return convert_precision(inputs)
    
    def compile_function(self, fn: Callable, **kwargs) -> Callable:
        """Compile function using MLX compilation."""
        if not self.available:
            return fn
        
        # Extract MLX compile options
        inputs = kwargs.get('inputs', [])
        outputs = kwargs.get('outputs', [])
        
        if inputs or outputs:
            return partial(self._mlx.compile, inputs=inputs, outputs=outputs)(fn)
        else:
            return self._mlx.compile(fn)
    
    def set_random_seed(self, seed: int) -> None:
        """Set MLX random seed."""
        if self.available:
            self._mlx.random.seed(seed)
    
    def create_loss_fn_with_grad(
        self,
        model: Any,
        forward_fn: Callable | None = None
    ) -> Callable:
        """Create MLX loss function with automatic gradient computation."""
        if forward_fn is None:
            # Default forward function
            def default_forward_fn(model, batch):
                # Remove metadata if present
                model_inputs = {
                    k: v for k, v in batch.items()
                    if k not in ["metadata"] and v is not None
                }
                
                try:
                    outputs = model(**model_inputs)
                except TypeError:
                    outputs = model(batch)
                
                # Extract loss
                loss = outputs.get("loss")
                if loss is None:
                    raise ValueError("Model must return a dictionary with 'loss' key")
                
                return loss, outputs
            
            forward_fn = default_forward_fn
        
        return self._mlx.value_and_grad(forward_fn)
    
    def create_gradient_accumulator(self, accumulation_steps: int = 1):
        """Create MLX-specific gradient accumulator."""
        from training.commands.gradient_accumulation import MLXGradientAccumulationCommand
        return MLXGradientAccumulationCommand(
            accumulation_steps=accumulation_steps,
            normalize_accumulated_gradients=True,
        )