"""Optimizer step command implementation.

This command handles weight updates through the optimizer, abstracting away
framework-specific optimizer operations.
"""

from typing import Any

from .base import BaseCommand, CommandContext, CommandResult


class OptimizerStepCommand(BaseCommand):
    """Command for executing optimizer weight updates."""
    
    def __init__(
        self,
        scale_learning_rate: bool = True,
        update_lr_scheduler: bool = True,
        zero_grad_after_step: bool = True,
    ):
        """Initialize optimizer step command.
        
        Args:
            scale_learning_rate: Whether to scale LR by gradient accumulation
            update_lr_scheduler: Whether to update LR scheduler after step
            zero_grad_after_step: Whether to zero gradients after update
        """
        super().__init__("OptimizerStep")
        self.scale_learning_rate = scale_learning_rate
        self.update_lr_scheduler = update_lr_scheduler
        self.zero_grad_after_step = zero_grad_after_step
    
    def can_execute(self, context: CommandContext) -> bool:
        """Check if optimizer step can be executed."""
        return (
            context.optimizer is not None
            and context.gradients
            and context.should_update_weights
            and context.is_training
        )
    
    def execute(self, context: CommandContext) -> CommandResult:
        """Execute optimizer step to update model weights."""
        try:
            # Check if we have gradients
            if not context.gradients:
                return CommandResult(
                    success=False,
                    error=ValueError("No gradients available for optimizer step")
                )
            
            # Get current learning rate
            current_lr = self._get_learning_rate(context.optimizer)
            
            # Scale learning rate if using gradient accumulation
            effective_lr = current_lr
            if self.scale_learning_rate and context.config.get("gradient_accumulation_steps", 1) > 1:
                effective_lr = current_lr / context.config["gradient_accumulation_steps"]
            
            # Update model weights
            self._update_weights(
                optimizer=context.optimizer,
                model=context.model,
                gradients=context.gradients,
                learning_rate=effective_lr
            )
            
            # Update learning rate scheduler if configured
            new_lr = current_lr
            if self.update_lr_scheduler and context.lr_scheduler is not None:
                new_lr = self._update_lr_scheduler(
                    scheduler=context.lr_scheduler,
                    metrics=context.metrics
                )
            
            # Zero gradients if configured
            if self.zero_grad_after_step:
                context.gradients = {}
            
            # Update global step
            context.state.global_step += 1
            
            # Build result
            result = CommandResult(
                success=True,
                outputs={
                    "learning_rate": new_lr,
                    "effective_learning_rate": effective_lr,
                },
                metrics={
                    "learning_rate": new_lr,
                }
            )
            
            return result
            
        except Exception as e:
            return CommandResult(
                success=False,
                error=e,
                should_continue=False
            )
    
    def _get_learning_rate(self, optimizer: Any) -> float:
        """Get current learning rate from optimizer."""
        # Use protocol method
        if hasattr(optimizer, "learning_rate"):
            lr = optimizer.learning_rate
            # Handle property vs method
            if callable(lr):
                return lr()
            return lr
        
        # Fallback for common optimizer attributes
        for attr in ["lr", "_lr", "param_groups"]:
            if hasattr(optimizer, attr):
                value = getattr(optimizer, attr)
                if isinstance(value, (int, float)):
                    return float(value)
                elif isinstance(value, list) and value:
                    # PyTorch-style param groups
                    return value[0].get("lr", 0.0)
        
        return 0.0
    
    def _update_weights(
        self,
        optimizer: Any,
        model: Any,
        gradients: dict[str, Any],
        learning_rate: float | None = None
    ) -> None:
        """Update model weights using optimizer."""
        # Use protocol method
        if hasattr(optimizer, "update"):
            optimizer.update(model, gradients)
        else:
            # Fallback for other optimizer interfaces
            if hasattr(optimizer, "step"):
                optimizer.step()
            else:
                raise ValueError(f"Optimizer {type(optimizer)} has no update or step method")
    
    def _update_lr_scheduler(
        self,
        scheduler: Any,
        metrics: dict[str, float] | None = None
    ) -> float:
        """Update learning rate scheduler and return new LR."""
        # Use protocol method
        if hasattr(scheduler, "step"):
            new_lr = scheduler.step(metrics)
            if new_lr is not None:
                return new_lr
        
        # Get current LR after update
        if hasattr(scheduler, "current_lr"):
            return scheduler.current_lr
        elif hasattr(scheduler, "get_last_lr"):
            # PyTorch-style
            lrs = scheduler.get_last_lr()
            return lrs[0] if lrs else 0.0
        
        return 0.0


class MLXOptimizerStepCommand(OptimizerStepCommand):
    """MLX-specific implementation of optimizer step command."""
    
    def _update_weights(
        self,
        optimizer: Any,
        model: Any,
        gradients: dict[str, Any],
        learning_rate: float | None = None
    ) -> None:
        """Update weights using MLX optimizer."""
        import mlx.core as mx
        
        # MLX optimizers use the update method
        optimizer.update(model, gradients)
        
        # Ensure updates are evaluated immediately
        mx.eval(model.parameters())
    
    def _get_learning_rate(self, optimizer: Any) -> float:
        """Get learning rate from MLX optimizer."""
        # MLX optimizers have learning_rate property
        lr = optimizer.learning_rate
        
        # Handle MLX array type
        if hasattr(lr, "item"):
            return lr.item()
        
        return float(lr)