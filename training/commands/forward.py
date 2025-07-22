"""Forward pass command implementation.

This command handles the forward pass through the model, abstracting away
MLX-specific details and providing a clean interface for model execution.
"""

from typing import Any

from .base import BaseCommand, CommandContext, CommandResult


class ForwardCommand(BaseCommand):
    """Command for executing forward pass through the model."""
    
    def __init__(
        self,
        compute_loss: bool = True,
        return_outputs: bool = True,
        mixed_precision: bool = False,
        label_smoothing: float = 0.0,
    ):
        """Initialize forward command.
        
        Args:
            compute_loss: Whether to compute loss
            return_outputs: Whether to return model outputs
            mixed_precision: Whether to use mixed precision
            label_smoothing: Label smoothing factor
        """
        super().__init__("ForwardPass")
        self.compute_loss = compute_loss
        self.return_outputs = return_outputs
        self.mixed_precision = mixed_precision
        self.label_smoothing = label_smoothing
        self._requires_grad = False  # Forward pass itself doesn't need grad
    
    def can_execute(self, context: CommandContext) -> bool:
        """Check if forward pass can be executed."""
        return (
            context.model is not None
            and context.batch is not None
            and len(context.batch) > 0
        )
    
    def execute(self, context: CommandContext) -> CommandResult:
        """Execute forward pass.
        
        This method abstracts away MLX-specific array operations and provides
        a clean interface for model forward pass.
        """
        try:
            # Get model inputs from batch, filtering out metadata
            model_inputs = self._prepare_inputs(context.batch)
            
            # Apply mixed precision if enabled (framework-agnostic)
            if self.mixed_precision:
                model_inputs = self._apply_mixed_precision(model_inputs)
            
            # Execute forward pass
            outputs = self._forward_pass(context.model, model_inputs, context.batch)
            
            # Process outputs
            loss = None
            if self.compute_loss:
                loss = self._extract_loss(outputs, model_inputs)
                if loss is not None and self.label_smoothing > 0:
                    loss = self._apply_label_smoothing(loss, self.label_smoothing)
            
            # Update context
            context.outputs = outputs if self.return_outputs else {}
            context.loss = loss
            
            # Build result
            result = CommandResult(
                success=True,
                outputs={"loss": loss, "model_outputs": outputs} if self.return_outputs else {"loss": loss},
            )
            
            return result
            
        except Exception as e:
            return CommandResult(
                success=False,
                error=e,
                should_continue=False
            )
    
    def _prepare_inputs(self, batch: dict[str, Any]) -> dict[str, Any]:
        """Prepare model inputs from batch data."""
        # Filter out non-model inputs
        excluded_keys = {"metadata", "original_idx", "augmentation_info"}
        return {
            k: v for k, v in batch.items()
            if k not in excluded_keys and v is not None
        }
    
    def _apply_mixed_precision(self, inputs: dict[str, Any]) -> dict[str, Any]:
        """Apply mixed precision to inputs (framework-agnostic)."""
        # This is a placeholder - actual implementation would use
        # framework-specific precision conversion
        return inputs
    
    def _forward_pass(
        self,
        model: Any,
        model_inputs: dict[str, Any],
        original_batch: dict[str, Any]
    ) -> dict[str, Any]:
        """Execute model forward pass with multiple calling conventions."""
        try:
            # Try unpacked arguments first (common for transformer models)
            outputs = model(**model_inputs)
        except TypeError:
            try:
                # Fall back to batch dictionary
                outputs = model(original_batch)
            except Exception:
                # Last resort - pass inputs as positional argument
                outputs = model(model_inputs)
        
        # Ensure outputs is a dictionary
        if not isinstance(outputs, dict):
            if hasattr(outputs, "__dict__"):
                outputs = outputs.__dict__
            else:
                outputs = {"output": outputs}
        
        return outputs
    
    def _extract_loss(
        self,
        outputs: dict[str, Any],
        inputs: dict[str, Any]
    ) -> float | None:
        """Extract loss from model outputs."""
        # Direct loss in outputs
        if "loss" in outputs and outputs["loss"] is not None:
            return self._to_float(outputs["loss"])
        
        # Try to compute loss from logits and labels
        if "logits" in outputs and "labels" in inputs:
            # This would use framework-specific loss computation
            # For now, we return None to indicate loss computation is needed
            return None
        
        # Check for other common loss keys
        for loss_key in ["total_loss", "combined_loss", "training_loss"]:
            if loss_key in outputs and outputs[loss_key] is not None:
                return self._to_float(outputs[loss_key])
        
        return None
    
    def _apply_label_smoothing(self, loss: float, smoothing: float) -> float:
        """Apply label smoothing to loss."""
        return loss * (1.0 - smoothing)
    
    def _to_float(self, value: Any) -> float:
        """Convert value to float (handles framework-specific tensors)."""
        if isinstance(value, (int, float)):
            return float(value)
        # Handle framework-specific tensor types
        if hasattr(value, "item"):
            return value.item()
        if hasattr(value, "numpy"):
            return float(value.numpy())
        return float(value)