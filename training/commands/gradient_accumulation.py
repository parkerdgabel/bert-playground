"""Gradient accumulation command implementation.

This command handles gradient accumulation across multiple batches,
enabling effective larger batch sizes without increased memory usage.
"""

from typing import Any

from .base import BaseCommand, CommandContext, CommandResult


class GradientAccumulationCommand(BaseCommand):
    """Command for managing gradient accumulation."""
    
    def __init__(
        self,
        accumulation_steps: int = 1,
        normalize_accumulated_gradients: bool = True,
    ):
        """Initialize gradient accumulation command.
        
        Args:
            accumulation_steps: Number of steps to accumulate gradients
            normalize_accumulated_gradients: Whether to normalize by accumulation steps
        """
        super().__init__("GradientAccumulation")
        self.accumulation_steps = accumulation_steps
        self.normalize_accumulated_gradients = normalize_accumulated_gradients
        self._accumulated_gradients: dict[str, Any] = {}
        self._accumulation_count = 0
    
    def can_execute(self, context: CommandContext) -> bool:
        """Check if gradient accumulation can be executed."""
        return (
            context.gradients is not None
            and len(context.gradients) > 0
            and context.is_training
        )
    
    def execute(self, context: CommandContext) -> CommandResult:
        """Execute gradient accumulation logic."""
        try:
            # Get current gradients
            current_gradients = context.gradients
            if not current_gradients:
                return CommandResult(
                    success=False,
                    error=ValueError("No gradients to accumulate")
                )
            
            # Accumulate gradients
            if self._accumulation_count == 0:
                # First accumulation - just store
                self._accumulated_gradients = self._deep_copy_gradients(current_gradients)
            else:
                # Add to existing accumulated gradients
                self._accumulated_gradients = self._add_gradients(
                    self._accumulated_gradients,
                    current_gradients
                )
            
            self._accumulation_count += 1
            
            # Check if we should update weights
            should_update = self._accumulation_count >= self.accumulation_steps
            
            # Prepare gradients for optimizer
            if should_update:
                # Normalize accumulated gradients if configured
                if self.normalize_accumulated_gradients and self.accumulation_steps > 1:
                    final_gradients = self._scale_gradients(
                        self._accumulated_gradients,
                        1.0 / self.accumulation_steps
                    )
                else:
                    final_gradients = self._accumulated_gradients
                
                # Update context with final gradients
                context.gradients = final_gradients
                context.should_update_weights = True
                
                # Reset accumulation
                self._accumulated_gradients = {}
                self._accumulation_count = 0
            else:
                # Don't update weights yet
                context.should_update_weights = False
                # Keep accumulated gradients in context for next iteration
                context.gradients = self._accumulated_gradients
            
            # Build result
            result = CommandResult(
                success=True,
                outputs={
                    "accumulation_count": self._accumulation_count,
                    "should_update": should_update,
                },
                metrics={
                    "gradient_accumulation_progress": self._accumulation_count / self.accumulation_steps,
                }
            )
            
            return result
            
        except Exception as e:
            return CommandResult(
                success=False,
                error=e,
                should_continue=False
            )
    
    def rollback(self, context: CommandContext) -> None:
        """Reset gradient accumulation state."""
        self._accumulated_gradients = {}
        self._accumulation_count = 0
        context.should_update_weights = True
    
    def _deep_copy_gradients(self, gradients: dict[str, Any]) -> dict[str, Any]:
        """Create a deep copy of gradients."""
        # This is a simplified version - actual implementation would handle
        # framework-specific tensor copying
        result = {}
        for key, value in gradients.items():
            if isinstance(value, dict):
                result[key] = self._deep_copy_gradients(value)
            elif value is not None:
                # Framework-specific tensor copy would go here
                result[key] = value
            else:
                result[key] = None
        return result
    
    def _add_gradients(
        self,
        accumulated: dict[str, Any],
        current: dict[str, Any]
    ) -> dict[str, Any]:
        """Add current gradients to accumulated gradients."""
        result = {}
        
        # Handle all keys from both dictionaries
        all_keys = set(accumulated.keys()) | set(current.keys())
        
        for key in all_keys:
            acc_value = accumulated.get(key)
            curr_value = current.get(key)
            
            if isinstance(acc_value, dict) and isinstance(curr_value, dict):
                # Recursive case for nested gradients
                result[key] = self._add_gradients(acc_value, curr_value)
            elif acc_value is not None and curr_value is not None:
                # Add tensors - framework-specific implementation needed
                result[key] = self._add_tensors(acc_value, curr_value)
            elif acc_value is not None:
                result[key] = acc_value
            elif curr_value is not None:
                result[key] = curr_value
            else:
                result[key] = None
        
        return result
    
    def _add_tensors(self, tensor1: Any, tensor2: Any) -> Any:
        """Add two tensors (framework-specific)."""
        # Placeholder - actual implementation would be framework-specific
        return tensor1
    
    def _scale_gradients(self, gradients: dict[str, Any], scale: float) -> dict[str, Any]:
        """Scale gradients by a factor."""
        result = {}
        for key, value in gradients.items():
            if isinstance(value, dict):
                result[key] = self._scale_gradients(value, scale)
            elif value is not None:
                # Framework-specific scaling would go here
                result[key] = value
            else:
                result[key] = None
        return result


class MLXGradientAccumulationCommand(GradientAccumulationCommand):
    """MLX-specific implementation of gradient accumulation."""
    
    def _deep_copy_gradients(self, gradients: dict[str, Any]) -> dict[str, Any]:
        """Deep copy gradients using MLX."""
        import mlx.core as mx
        
        def copy_grad(grad):
            if isinstance(grad, dict):
                return {k: copy_grad(v) for k, v in grad.items()}
            elif grad is not None and hasattr(grad, "copy"):
                return grad.copy()
            elif grad is not None:
                # For MLX arrays, creating a new array with same data
                return mx.array(grad)
            return grad
        
        return copy_grad(gradients)
    
    def _add_tensors(self, tensor1: Any, tensor2: Any) -> Any:
        """Add two MLX tensors."""
        import mlx.core as mx
        
        # Ensure both are MLX arrays
        if not isinstance(tensor1, mx.array):
            tensor1 = mx.array(tensor1)
        if not isinstance(tensor2, mx.array):
            tensor2 = mx.array(tensor2)
        
        return tensor1 + tensor2
    
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