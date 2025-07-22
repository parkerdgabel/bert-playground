"""Backward pass command implementation.

This command handles gradient computation, abstracting away MLX-specific
gradient operations behind a clean interface.
"""

from typing import Any, Callable

from .base import BaseCommand, CommandContext, CommandResult


class BackwardCommand(BaseCommand):
    """Command for computing gradients through backward pass."""
    
    def __init__(
        self,
        grad_clip_norm: float = 0.0,
        grad_clip_value: float = 0.0,
        compute_grad_norm: bool = True,
        loss_scale: float = 1.0,
    ):
        """Initialize backward command.
        
        Args:
            grad_clip_norm: Maximum gradient norm for clipping (0 = no clipping)
            grad_clip_value: Maximum gradient value for clipping (0 = no clipping)
            compute_grad_norm: Whether to compute gradient norm
            loss_scale: Loss scaling factor for mixed precision
        """
        super().__init__("BackwardPass")
        self.grad_clip_norm = grad_clip_norm
        self.grad_clip_value = grad_clip_value
        self.compute_grad_norm = compute_grad_norm
        self.loss_scale = loss_scale
        self._requires_grad = True
    
    def can_execute(self, context: CommandContext) -> bool:
        """Check if backward pass can be executed."""
        return (
            context.model is not None
            and context.loss is not None
            and context.is_training
        )
    
    def execute(self, context: CommandContext) -> CommandResult:
        """Execute backward pass to compute gradients."""
        try:
            # Get loss value
            loss = context.loss
            if loss is None:
                return CommandResult(
                    success=False,
                    error=ValueError("No loss available for backward pass")
                )
            
            # Apply loss scaling if needed
            if self.loss_scale != 1.0:
                loss = loss * self.loss_scale
            
            # Compute gradients using framework-agnostic interface
            gradients = self._compute_gradients(
                model=context.model,
                loss=loss,
                inputs=context.batch,
                outputs=context.outputs
            )
            
            # Apply gradient clipping if configured
            grad_norm = None
            if self.grad_clip_norm > 0 or self.grad_clip_value > 0:
                gradients, grad_norm = self._clip_gradients(
                    gradients,
                    max_norm=self.grad_clip_norm,
                    max_value=self.grad_clip_value
                )
            elif self.compute_grad_norm:
                grad_norm = self._compute_grad_norm(gradients)
            
            # Unscale gradients if loss scaling was applied
            if self.loss_scale != 1.0:
                gradients = self._unscale_gradients(gradients, self.loss_scale)
            
            # Update context
            context.gradients = gradients
            if grad_norm is not None:
                context.metrics["grad_norm"] = grad_norm
            
            # Build result
            result = CommandResult(
                success=True,
                outputs={"gradients": gradients},
                metrics={"grad_norm": grad_norm} if grad_norm is not None else {}
            )
            
            return result
            
        except Exception as e:
            return CommandResult(
                success=False,
                error=e,
                should_continue=False
            )
    
    def _compute_gradients(
        self,
        model: Any,
        loss: float,
        inputs: dict[str, Any],
        outputs: dict[str, Any]
    ) -> dict[str, Any]:
        """Compute gradients using framework-agnostic interface.
        
        This method should be overridden by framework-specific implementations.
        For MLX, this would use mx.value_and_grad. For PyTorch, loss.backward().
        """
        # This is a placeholder - actual implementation would be framework-specific
        # For now, return empty gradients
        return {}
    
    def _clip_gradients(
        self,
        gradients: dict[str, Any],
        max_norm: float = 0.0,
        max_value: float = 0.0
    ) -> tuple[dict[str, Any], float]:
        """Clip gradients by norm and/or value.
        
        Args:
            gradients: Dictionary of gradients
            max_norm: Maximum gradient norm (0 = no clipping)
            max_value: Maximum gradient value (0 = no clipping)
            
        Returns:
            Tuple of (clipped_gradients, grad_norm)
        """
        # Compute gradient norm first
        grad_norm = self._compute_grad_norm(gradients)
        
        # Apply norm clipping if needed
        if max_norm > 0 and grad_norm > max_norm:
            scale = max_norm / grad_norm
            gradients = self._scale_gradients(gradients, scale)
            grad_norm = max_norm
        
        # Apply value clipping if needed
        if max_value > 0:
            gradients = self._clip_gradient_values(gradients, max_value)
        
        return gradients, grad_norm
    
    def _compute_grad_norm(self, gradients: dict[str, Any]) -> float:
        """Compute the norm of all gradients.
        
        This is a placeholder - actual implementation would be framework-specific.
        """
        return 0.0
    
    def _scale_gradients(self, gradients: dict[str, Any], scale: float) -> dict[str, Any]:
        """Scale all gradients by a factor."""
        # Placeholder - framework-specific implementation needed
        return gradients
    
    def _clip_gradient_values(self, gradients: dict[str, Any], max_value: float) -> dict[str, Any]:
        """Clip gradient values to [-max_value, max_value]."""
        # Placeholder - framework-specific implementation needed
        return gradients
    
    def _unscale_gradients(self, gradients: dict[str, Any], scale: float) -> dict[str, Any]:
        """Unscale gradients after loss scaling."""
        return self._scale_gradients(gradients, 1.0 / scale)


class MLXBackwardCommand(BackwardCommand):
    """MLX-specific implementation of backward command."""
    
    def __init__(self, value_and_grad_fn: Callable | None = None, **kwargs):
        """Initialize MLX backward command.
        
        Args:
            value_and_grad_fn: Optional pre-compiled value_and_grad function
            **kwargs: Arguments passed to parent class
        """
        super().__init__(**kwargs)
        self.value_and_grad_fn = value_and_grad_fn
    
    def _compute_gradients(
        self,
        model: Any,
        loss: float,
        inputs: dict[str, Any],
        outputs: dict[str, Any]
    ) -> dict[str, Any]:
        """Compute gradients using MLX."""
        import mlx.core as mx
        
        if self.value_and_grad_fn is not None:
            # Use pre-compiled function
            _, grads = self.value_and_grad_fn(model, inputs)
        else:
            # Create value_and_grad function on the fly
            def loss_fn(model, batch):
                # Re-run forward pass to compute gradients
                model_inputs = {
                    k: v for k, v in batch.items()
                    if k not in ["metadata"] and v is not None
                }
                outputs = model(**model_inputs)
                return outputs.get("loss", mx.array(0.0)), outputs
            
            value_and_grad_fn = mx.value_and_grad(loss_fn)
            _, grads = value_and_grad_fn(model, inputs)
        
        # Force evaluation to prevent lazy computation buildup
        mx.eval(grads)
        
        return grads
    
    def _compute_grad_norm(self, gradients: dict[str, Any]) -> float:
        """Compute gradient norm using MLX."""
        import mlx.core as mx
        
        # Flatten all gradients and compute L2 norm
        grad_values = []
        for grad in gradients.values():
            if isinstance(grad, dict):
                # Recursive case for nested gradients
                for v in grad.values():
                    if v is not None:
                        grad_values.append(mx.reshape(v, [-1]))
            elif grad is not None:
                grad_values.append(mx.reshape(grad, [-1]))
        
        if not grad_values:
            return 0.0
        
        all_grads = mx.concatenate(grad_values)
        grad_norm = mx.sqrt(mx.sum(mx.square(all_grads)))
        
        # Convert to Python float
        return float(grad_norm.item())
    
    def _scale_gradients(self, gradients: dict[str, Any], scale: float) -> dict[str, Any]:
        """Scale gradients using MLX."""
        import mlx.core as mx
        
        def scale_grad(grad):
            if isinstance(grad, dict):
                return {k: scale_grad(v) for k, v in grad.items()}
            elif grad is not None:
                return grad * scale
            return grad
        
        return scale_grad(gradients)
    
    def _clip_gradient_values(self, gradients: dict[str, Any], max_value: float) -> dict[str, Any]:
        """Clip gradient values using MLX."""
        import mlx.core as mx
        
        def clip_grad(grad):
            if isinstance(grad, dict):
                return {k: clip_grad(v) for k, v in grad.items()}
            elif grad is not None:
                return mx.clip(grad, -max_value, max_value)
            return grad
        
        return clip_grad(gradients)