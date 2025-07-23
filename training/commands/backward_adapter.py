"""Framework-agnostic backward pass command implementation.

This command handles gradient computation using the FrameworkAdapter,
abstracting away framework-specific gradient operations.
"""

from typing import Any, Callable

from loguru import logger

from training.adapters.framework_adapter import FrameworkAdapter
from training.core.optimization import clip_gradients, compute_gradient_stats
from .base import BaseCommand, CommandContext, CommandResult


class BackwardCommand(BaseCommand):
    """Framework-agnostic command for computing gradients through backward pass."""
    
    def __init__(
        self,
        framework: FrameworkAdapter,
        grad_clip_norm: float = 0.0,
        grad_clip_value: float = 0.0,
        compute_grad_norm: bool = True,
        loss_scale: float = 1.0,
        compute_detailed_stats: bool = False,
    ):
        """Initialize backward command.
        
        Args:
            framework: Framework adapter for gradient operations
            grad_clip_norm: Maximum gradient norm for clipping (0 = no clipping)
            grad_clip_value: Maximum gradient value for clipping (0 = no clipping)
            compute_grad_norm: Whether to compute gradient norm
            loss_scale: Loss scaling factor for mixed precision
            compute_detailed_stats: Whether to compute detailed gradient statistics
        """
        super().__init__("BackwardPass")
        self.framework = framework
        self.grad_clip_norm = grad_clip_norm
        self.grad_clip_value = grad_clip_value
        self.compute_grad_norm = compute_grad_norm
        self.loss_scale = loss_scale
        self.compute_detailed_stats = compute_detailed_stats
        self._requires_grad = True
        
        # Create value and grad function if supported
        self._value_and_grad_fn = None
    
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
                loss = self.framework.tensor_multiply(loss, self.loss_scale)
            
            # Compute gradients using framework adapter
            gradients = self._compute_gradients(
                model=context.model,
                loss=loss,
                inputs=context.batch,
                outputs=context.outputs
            )
            
            # Apply gradient clipping if configured
            grad_norm = None
            if self.grad_clip_norm > 0:
                gradients, grad_norm = clip_gradients(
                    gradients,
                    max_norm=self.grad_clip_norm,
                    framework=self.framework
                )
            elif self.compute_grad_norm:
                grad_norm = self.framework.compute_gradient_norm(gradients)
            
            # Apply value clipping if needed
            if self.grad_clip_value > 0:
                gradients = self._clip_gradient_values(gradients, self.grad_clip_value)
            
            # Unscale gradients if loss scaling was applied
            if self.loss_scale != 1.0:
                gradients = self.framework.scale_gradients(gradients, 1.0 / self.loss_scale)
            
            # Compute gradient statistics if requested
            grad_stats = {}
            if self.compute_detailed_stats:
                grad_stats = compute_gradient_stats(
                    gradients,
                    framework=self.framework,
                    detailed=True
                )
            
            # Update context
            context.gradients = gradients
            if grad_norm is not None:
                context.metrics["grad_norm"] = grad_norm
            if grad_stats:
                context.metrics.update(grad_stats)
            
            # Build result
            result = CommandResult(
                success=True,
                outputs={"gradients": gradients},
                metrics={"grad_norm": grad_norm} if grad_norm is not None else {}
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Backward pass failed: {e}")
            return CommandResult(
                success=False,
                error=e,
                should_continue=False
            )
    
    def _compute_gradients(
        self,
        model: Any,
        loss: Any,
        inputs: dict[str, Any],
        outputs: dict[str, Any]
    ) -> dict[str, Any]:
        """Compute gradients using framework adapter.
        
        Args:
            model: Model instance
            loss: Loss value
            inputs: Model inputs
            outputs: Model outputs
            
        Returns:
            Dictionary of gradients
        """
        # Get or create value and grad function
        if self._value_and_grad_fn is None:
            def loss_fn(model, batch):
                # Remove metadata if present
                model_inputs = {
                    k: v for k, v in batch.items()
                    if k not in ["metadata"] and v is not None
                }
                
                # Run model
                try:
                    outputs = model(**model_inputs)
                except TypeError:
                    # Fallback for models that expect batch directly
                    outputs = model(batch)
                
                # Extract loss
                loss = outputs.get("loss")
                if loss is None:
                    raise ValueError("Model must return a dictionary with 'loss' key")
                
                return loss, outputs
            
            self._value_and_grad_fn = self.framework.create_value_and_grad_fn(
                model, loss_fn
            )
        
        # Compute gradients
        _, grads = self._value_and_grad_fn(model, inputs)
        
        # Force evaluation to prevent lazy computation buildup
        self.framework.evaluate_tensors(grads)
        
        return grads
    
    def _clip_gradient_values(self, gradients: dict[str, Any], max_value: float) -> dict[str, Any]:
        """Clip gradient values to [-max_value, max_value].
        
        Args:
            gradients: Dictionary of gradients
            max_value: Maximum absolute value
            
        Returns:
            Clipped gradients
        """
        # For now, we'll need to add this to the framework adapter
        # or implement it here using the adapter's tensor operations
        if hasattr(self.framework._adapter, 'clip_gradients_by_value'):
            return self.framework._adapter.clip_gradients_by_value(gradients, max_value)
        
        # Fallback: just return gradients unmodified
        logger.warning("Gradient value clipping not implemented for this framework")
        return gradients


class AdaptiveBackwardCommand(BackwardCommand):
    """Backward command with adaptive gradient clipping and scaling."""
    
    def __init__(
        self,
        framework: FrameworkAdapter,
        initial_clip_norm: float = 1.0,
        clip_percentile: float = 0.95,
        history_size: int = 100,
        **kwargs
    ):
        """Initialize adaptive backward command.
        
        Args:
            framework: Framework adapter
            initial_clip_norm: Initial gradient clipping norm
            clip_percentile: Percentile for adaptive clipping
            history_size: Number of gradient norms to track
            **kwargs: Additional arguments for parent class
        """
        super().__init__(framework, grad_clip_norm=initial_clip_norm, **kwargs)
        self.clip_percentile = clip_percentile
        self.history_size = history_size
        self.grad_norm_history = []
    
    def execute(self, context: CommandContext) -> CommandResult:
        """Execute backward pass with adaptive clipping."""
        # Run standard backward pass
        result = super().execute(context)
        
        if result.success and "grad_norm" in result.metrics:
            grad_norm = result.metrics["grad_norm"]
            
            # Update history
            self.grad_norm_history.append(grad_norm)
            if len(self.grad_norm_history) > self.history_size:
                self.grad_norm_history.pop(0)
            
            # Adapt clipping threshold
            if len(self.grad_norm_history) >= 10:  # Need some history
                import numpy as np
                percentile_value = np.percentile(
                    self.grad_norm_history, 
                    self.clip_percentile * 100
                )
                self.grad_clip_norm = float(percentile_value)
                
                # Log adaptation
                if context.state.global_step % 100 == 0:
                    logger.info(
                        f"Adaptive gradient clipping: norm={self.grad_clip_norm:.4f}, "
                        f"current={grad_norm:.4f}"
                    )
        
        return result