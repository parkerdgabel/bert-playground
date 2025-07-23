"""MLX implementation of Optimizer port."""

from typing import Any, Callable, Optional

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

from ports.secondary.optimization import Optimizer, OptimizerConfig
from ports.secondary.compute import Module, Array


class MLXOptimizerAdapter:
    """MLX implementation of the Optimizer port."""
    
    def __init__(self, config: OptimizerConfig, optimizer_type: str = "adamw"):
        """Initialize MLX optimizer adapter.
        
        Args:
            config: Optimizer configuration
            optimizer_type: Type of optimizer to use
        """
        self.config = config
        self.optimizer_type = optimizer_type
        self._mlx_optimizer: Optional[optim.Optimizer] = None
        self._param_groups: list[dict[str, Any]] = []
        self._step_count = 0
        self._state: dict[str, Any] = {}
        
    def _ensure_optimizer(self) -> optim.Optimizer:
        """Ensure optimizer is created."""
        if self._mlx_optimizer is None:
            self._mlx_optimizer = self._create_optimizer()
        return self._mlx_optimizer
    
    def _create_optimizer(self) -> optim.Optimizer:
        """Create MLX optimizer based on type."""
        if self.optimizer_type == "adam":
            return optim.Adam(
                learning_rate=self.config.learning_rate,
                betas=(self.config.beta1, self.config.beta2),
                eps=self.config.epsilon,
            )
        elif self.optimizer_type == "adamw":
            return optim.AdamW(
                learning_rate=self.config.learning_rate,
                betas=(self.config.beta1, self.config.beta2),
                eps=self.config.epsilon,
                weight_decay=self.config.weight_decay,
            )
        elif self.optimizer_type == "sgd":
            return optim.SGD(
                learning_rate=self.config.learning_rate,
                momentum=self.config.momentum,
                weight_decay=self.config.weight_decay,
                nesterov=self.config.nesterov,
            )
        elif self.optimizer_type == "rmsprop":
            return optim.RMSprop(
                learning_rate=self.config.learning_rate,
                alpha=0.99,  # RMSprop specific
                eps=self.config.epsilon,
                weight_decay=self.config.weight_decay,
                momentum=self.config.momentum,
                centered=self.config.centered,
            )
        elif self.optimizer_type == "lion":
            return optim.Lion(
                learning_rate=self.config.learning_rate,
                betas=(self.config.beta1, self.config.beta2),
                weight_decay=self.config.weight_decay,
            )
        else:
            raise ValueError(f"Unknown optimizer type: {self.optimizer_type}")
    
    def update(self, model: Module, gradients: dict[str, Array]) -> None:
        """Update model parameters with gradients.
        
        Args:
            model: Model to update
            gradients: Gradients for each parameter
        """
        optimizer = self._ensure_optimizer()
        
        # Convert to MLX types if needed
        mlx_model = self._ensure_mlx_module(model)
        mlx_grads = self._ensure_mlx_arrays(gradients)
        
        # Update using MLX optimizer
        optimizer.update(mlx_model, mlx_grads)
        
        self._step_count += 1
    
    @property
    def learning_rate(self) -> float:
        """Current learning rate."""
        optimizer = self._ensure_optimizer()
        if hasattr(optimizer, 'learning_rate'):
            if callable(optimizer.learning_rate):
                return optimizer.learning_rate()
            return optimizer.learning_rate
        return self.config.learning_rate
    
    @learning_rate.setter
    def learning_rate(self, value: float) -> None:
        """Set learning rate."""
        self.config.learning_rate = value
        optimizer = self._ensure_optimizer()
        if hasattr(optimizer, 'learning_rate'):
            optimizer.learning_rate = value
    
    @property
    def state(self) -> dict[str, Any]:
        """Optimizer state (momentum buffers, etc.)."""
        return {
            "step_count": self._step_count,
            "config": self.config.to_dict(),
            "optimizer_type": self.optimizer_type,
            "param_groups": self._param_groups,
            **self._state
        }
    
    def state_dict(self) -> dict[str, Any]:
        """Get optimizer state dictionary.
        
        Returns:
            State dictionary for serialization
        """
        state = {
            "state": self.state,
            "param_groups": self._param_groups,
        }
        
        # Add MLX optimizer state if available
        if self._mlx_optimizer is not None:
            if hasattr(self._mlx_optimizer, 'state'):
                state["mlx_state"] = self._mlx_optimizer.state
        
        return state
    
    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Load optimizer state.
        
        Args:
            state_dict: State dictionary to load
        """
        if "state" in state_dict:
            saved_state = state_dict["state"]
            self._step_count = saved_state.get("step_count", 0)
            
            # Update config
            if "config" in saved_state:
                config_dict = saved_state["config"]
                self.config = OptimizerConfig(**config_dict)
            
            # Update internal state
            self._state.update({
                k: v for k, v in saved_state.items()
                if k not in ["step_count", "config", "optimizer_type", "param_groups"]
            })
        
        if "param_groups" in state_dict:
            self._param_groups = state_dict["param_groups"]
        
        # Recreate optimizer with loaded state
        if "mlx_state" in state_dict and self._mlx_optimizer is not None:
            # MLX optimizers don't have a direct load_state_dict method
            # We would need to recreate with the saved state
            pass
    
    def zero_grad(self) -> None:
        """Zero out gradients."""
        # MLX doesn't accumulate gradients, so this is a no-op
        # Gradients are computed fresh each time
        pass
    
    def step(self, closure: Optional[Callable[[], float]] = None) -> Optional[float]:
        """Perform optimization step.
        
        Args:
            closure: Optional closure for computing loss
            
        Returns:
            Loss value if closure provided
        """
        loss = None
        if closure is not None:
            # Enable gradient computation for closure
            with mx.grad_mode():
                loss = closure()
        
        # Note: Actual parameter updates should be done via update() method
        # This is here for compatibility with PyTorch-style API
        
        return loss
    
    def add_param_group(self, param_group: dict[str, Any]) -> None:
        """Add a parameter group with its own settings.
        
        Args:
            param_group: Parameter group configuration
        """
        self._param_groups.append(param_group)
    
    def get_param_groups(self) -> list[dict[str, Any]]:
        """Get all parameter groups.
        
        Returns:
            List of parameter groups
        """
        return self._param_groups.copy()
    
    # Helper methods
    
    def _ensure_mlx_module(self, model: Module) -> nn.Module:
        """Ensure model is an MLX module."""
        if isinstance(model, nn.Module):
            return model
        
        # If it's a wrapped model, try to get the underlying MLX model
        if hasattr(model, 'get_mlx_model'):
            return model.get_mlx_model()
        elif hasattr(model, '_model') and isinstance(model._model, nn.Module):
            return model._model
        else:
            raise TypeError(f"Cannot convert {type(model)} to MLX module")
    
    def _ensure_mlx_arrays(self, arrays: dict[str, Array]) -> dict[str, mx.array]:
        """Ensure arrays are MLX arrays."""
        mlx_arrays = {}
        for key, value in arrays.items():
            if isinstance(value, mx.array):
                mlx_arrays[key] = value
            elif hasattr(value, 'to_mlx'):
                mlx_arrays[key] = value.to_mlx()
            else:
                # Try to convert to MLX array
                mlx_arrays[key] = mx.array(value)
        return mlx_arrays
    
    def compute_gradients(
        self,
        loss_fn: Callable,
        model: Module,
        *args,
        **kwargs
    ) -> tuple[Any, dict[str, mx.array]]:
        """Compute gradients using MLX autodiff.
        
        Args:
            loss_fn: Loss function
            model: Model
            *args: Positional arguments for loss function
            **kwargs: Keyword arguments for loss function
            
        Returns:
            Tuple of (loss_value, gradients)
        """
        mlx_model = self._ensure_mlx_module(model)
        
        # Create value and gradient function
        loss_and_grad_fn = mx.value_and_grad(loss_fn)
        
        # Compute loss and gradients
        loss_value, grads = loss_and_grad_fn(mlx_model, *args, **kwargs)
        
        return loss_value, grads
    
    def clip_gradients(
        self,
        gradients: dict[str, mx.array],
        max_norm: float
    ) -> float:
        """Clip gradients by global norm.
        
        Args:
            gradients: Dictionary of gradients
            max_norm: Maximum gradient norm
            
        Returns:
            Gradient norm before clipping
        """
        # Compute global gradient norm
        total_norm = 0.0
        for grad in gradients.values():
            if grad is not None:
                grad_norm = mx.sum(grad * grad)
                total_norm += grad_norm
        
        grad_norm = mx.sqrt(total_norm).item()
        
        # Clip if needed
        if grad_norm > max_norm:
            scale = max_norm / grad_norm
            for key in gradients:
                gradients[key] = gradients[key] * scale
        
        return grad_norm