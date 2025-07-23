"""MLX optimization utilities."""

from typing import Any, Dict, Optional, List, Tuple
from dataclasses import dataclass

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim


@dataclass
class MLXOptimizerState:
    """State for MLX optimizer."""
    type: str
    learning_rate: float
    weight_decay: float = 0.0
    beta1: float = 0.9
    beta2: float = 0.999
    eps: float = 1e-8
    momentum: float = 0.0
    dampening: float = 0.0
    nesterov: bool = False
    step: int = 0


class MLXOptimizer:
    """Wrapper for MLX optimizers with gradient clipping and state management."""
    
    def __init__(
        self,
        model_adapter: "MLXModelAdapter",
        optimizer_type: str = "adamw",
        learning_rate: float = 1e-3,
        weight_decay: float = 0.0,
        **kwargs: Any
    ):
        """Initialize MLX optimizer.
        
        Args:
            model_adapter: The MLX model adapter
            optimizer_type: Type of optimizer
            learning_rate: Learning rate
            weight_decay: Weight decay
            **kwargs: Additional optimizer parameters
        """
        self.model_adapter = model_adapter
        self.optimizer_type = optimizer_type
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.kwargs = kwargs
        
        # Create MLX optimizer
        self._optimizer = self._create_optimizer()
        self._step = 0
        
        # For gradient computation
        self._loss_and_grad_fn = None
        self._last_loss = None
    
    def update(
        self,
        model_adapter: "MLXModelAdapter",
        max_grad_norm: Optional[float] = None,
    ) -> float:
        """Update model weights.
        
        Args:
            model_adapter: The model adapter
            max_grad_norm: Maximum gradient norm for clipping
            
        Returns:
            Gradient norm
        """
        # Get model
        model = model_adapter.get_mlx_model()
        
        # Get gradients from last forward pass
        # In MLX, gradients are computed via mx.grad or mx.value_and_grad
        # This assumes gradients were computed externally
        grads = model.gradients()
        
        # Clip gradients if needed
        grad_norm = 0.0
        if max_grad_norm is not None and max_grad_norm > 0:
            grad_norm = self._clip_gradients(grads, max_grad_norm)
        else:
            grad_norm = self._compute_grad_norm(grads)
        
        # Update weights
        self._optimizer.update(model, grads)
        
        # Increment step
        self._step += 1
        
        return grad_norm
    
    def compute_gradients(
        self,
        loss_fn: callable,
        model: nn.Module,
        *args,
        **kwargs
    ) -> Tuple[Any, Dict[str, mx.array]]:
        """Compute gradients using MLX autodiff.
        
        Args:
            loss_fn: Loss function
            model: MLX model
            *args: Positional arguments for loss function
            **kwargs: Keyword arguments for loss function
            
        Returns:
            Tuple of (loss_value, gradients)
        """
        # Create value and gradient function
        loss_and_grad_fn = mx.value_and_grad(loss_fn)
        
        # Compute loss and gradients
        loss_value, grads = loss_and_grad_fn(model, *args, **kwargs)
        
        return loss_value, grads
    
    def zero_grad(self) -> None:
        """Zero gradients (not needed in MLX but provided for compatibility)."""
        # MLX doesn't accumulate gradients, so this is a no-op
        pass
    
    def get_state(self) -> Dict[str, Any]:
        """Get optimizer state."""
        return {
            "type": self.optimizer_type,
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "step": self._step,
            **self.kwargs
        }
    
    def set_state(self, state: Dict[str, Any]) -> None:
        """Set optimizer state."""
        self.optimizer_type = state.get("type", self.optimizer_type)
        self.learning_rate = state.get("learning_rate", self.learning_rate)
        self.weight_decay = state.get("weight_decay", self.weight_decay)
        self._step = state.get("step", 0)
        
        # Update optimizer learning rate
        if hasattr(self._optimizer, "learning_rate"):
            self._optimizer.learning_rate = self.learning_rate
    
    # Private methods
    
    def _create_optimizer(self) -> optim.Optimizer:
        """Create MLX optimizer based on type."""
        # Get model parameters
        model = self.model_adapter.get_mlx_model()
        
        if self.optimizer_type == "adam":
            return optim.Adam(
                learning_rate=self.learning_rate,
                betas=(self.kwargs.get("beta1", 0.9), self.kwargs.get("beta2", 0.999)),
                eps=self.kwargs.get("eps", 1e-8),
            )
        elif self.optimizer_type == "adamw":
            return optim.AdamW(
                learning_rate=self.learning_rate,
                betas=(self.kwargs.get("beta1", 0.9), self.kwargs.get("beta2", 0.999)),
                eps=self.kwargs.get("eps", 1e-8),
                weight_decay=self.weight_decay,
            )
        elif self.optimizer_type == "sgd":
            return optim.SGD(
                learning_rate=self.learning_rate,
                momentum=self.kwargs.get("momentum", 0.0),
                weight_decay=self.weight_decay,
                nesterov=self.kwargs.get("nesterov", False),
            )
        elif self.optimizer_type == "rmsprop":
            return optim.RMSprop(
                learning_rate=self.learning_rate,
                alpha=self.kwargs.get("alpha", 0.99),
                eps=self.kwargs.get("eps", 1e-8),
                weight_decay=self.weight_decay,
                momentum=self.kwargs.get("momentum", 0.0),
            )
        elif self.optimizer_type == "lion":
            return optim.Lion(
                learning_rate=self.learning_rate,
                betas=(self.kwargs.get("beta1", 0.9), self.kwargs.get("beta2", 0.99)),
                weight_decay=self.weight_decay,
            )
        else:
            raise ValueError(f"Unknown optimizer type: {self.optimizer_type}")
    
    def _clip_gradients(
        self,
        grads: Dict[str, mx.array],
        max_norm: float
    ) -> float:
        """Clip gradients by global norm.
        
        Args:
            grads: Dictionary of gradients
            max_norm: Maximum gradient norm
            
        Returns:
            Gradient norm before clipping
        """
        # Compute global gradient norm
        grad_norm = self._compute_grad_norm(grads)
        
        # Clip if needed
        if grad_norm > max_norm:
            scale = max_norm / grad_norm
            for key in grads:
                grads[key] = grads[key] * scale
        
        return grad_norm
    
    def _compute_grad_norm(self, grads: Dict[str, mx.array]) -> float:
        """Compute global gradient norm.
        
        Args:
            grads: Dictionary of gradients
            
        Returns:
            Global gradient norm
        """
        total_norm = 0.0
        for grad in grads.values():
            if grad is not None:
                grad_norm = mx.sum(grad * grad)
                total_norm += grad_norm
        
        return mx.sqrt(total_norm).item()