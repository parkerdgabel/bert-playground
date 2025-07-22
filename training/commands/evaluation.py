"""Evaluation command implementation.

This command handles model evaluation on validation data.
"""

from typing import Any

from .base import BaseCommand, CommandContext, CommandResult


class EvaluationCommand(BaseCommand):
    """Command for evaluating model on a dataset."""
    
    def __init__(
        self,
        compute_metrics: bool = True,
        update_best_metrics: bool = True,
        save_predictions: bool = False,
    ):
        """Initialize evaluation command.
        
        Args:
            compute_metrics: Whether to compute metrics
            update_best_metrics: Whether to update best metric tracking
            save_predictions: Whether to save predictions
        """
        super().__init__("Evaluation")
        self.compute_metrics = compute_metrics
        self.update_best_metrics = update_best_metrics
        self.save_predictions = save_predictions
    
    def can_execute(self, context: CommandContext) -> bool:
        """Check if evaluation can be executed."""
        return (
            context.model is not None
            and context.val_dataloader is not None
            and not context.is_training  # Only run in eval mode
        )
    
    def execute(self, context: CommandContext) -> CommandResult:
        """Execute model evaluation."""
        try:
            # Ensure model is in evaluation mode
            original_training_state = context.is_training
            context.is_training = False
            
            # Initialize metrics
            total_loss = 0.0
            total_samples = 0
            all_predictions = []
            all_targets = []
            metrics_accumulator = {}
            
            # Iterate through validation data
            for batch_idx, batch in enumerate(context.val_dataloader):
                # Forward pass (no gradients needed)
                outputs = self._evaluate_batch(context.model, batch)
                
                # Accumulate loss
                if "loss" in outputs:
                    batch_size = self._get_batch_size(batch)
                    total_loss += float(outputs["loss"]) * batch_size
                    total_samples += batch_size
                
                # Collect predictions and targets if needed
                if self.compute_metrics or self.save_predictions:
                    if "logits" in outputs:
                        all_predictions.append(outputs["logits"])
                    if "labels" in batch:
                        all_targets.append(batch["labels"])
                
                # Accumulate other metrics
                for key, value in outputs.items():
                    if key not in ["loss", "logits", "hidden_states"]:
                        if key not in metrics_accumulator:
                            metrics_accumulator[key] = []
                        metrics_accumulator[key].append(float(value))
            
            # Compute final metrics
            avg_loss = total_loss / total_samples if total_samples > 0 else 0.0
            metrics = {"val_loss": avg_loss}
            
            # Compute additional metrics
            if self.compute_metrics and all_predictions and all_targets:
                additional_metrics = self._compute_metrics(all_predictions, all_targets)
                metrics.update(additional_metrics)
            
            # Average accumulated metrics
            for key, values in metrics_accumulator.items():
                metrics[f"val_{key}"] = sum(values) / len(values)
            
            # Update best metrics if configured
            if self.update_best_metrics:
                self._update_best_metrics(context, metrics)
            
            # Save predictions if configured
            predictions_path = None
            if self.save_predictions and all_predictions:
                predictions_path = self._save_predictions(
                    all_predictions,
                    all_targets,
                    context.config.get("output_dir", ".")
                )
            
            # Restore training state
            context.is_training = original_training_state
            
            # Update context metrics
            context.metrics.update(metrics)
            context.state.val_loss = avg_loss
            
            # Build result
            result = CommandResult(
                success=True,
                outputs={
                    "predictions_path": predictions_path,
                    "num_samples": total_samples,
                },
                metrics=metrics
            )
            
            return result
            
        except Exception as e:
            return CommandResult(
                success=False,
                error=e,
                should_continue=False
            )
    
    def _evaluate_batch(self, model: Any, batch: dict[str, Any]) -> dict[str, Any]:
        """Evaluate a single batch."""
        # Remove metadata
        model_inputs = {
            k: v for k, v in batch.items()
            if k not in ["metadata"] and v is not None
        }
        
        # Forward pass without gradients
        try:
            outputs = model(**model_inputs)
        except TypeError:
            outputs = model(batch)
        
        # Ensure outputs is a dictionary
        if not isinstance(outputs, dict):
            outputs = {"output": outputs}
        
        return outputs
    
    def _get_batch_size(self, batch: dict[str, Any]) -> int:
        """Get batch size from batch data."""
        # Look for common batch size indicators
        for key in ["input_ids", "inputs", "features", "x", "labels"]:
            if key in batch and hasattr(batch[key], "shape"):
                return batch[key].shape[0]
        
        # Fallback to first tensor-like value
        for value in batch.values():
            if hasattr(value, "shape") and len(value.shape) > 0:
                return value.shape[0]
        
        return 1
    
    def _compute_metrics(
        self,
        predictions: list[Any],
        targets: list[Any]
    ) -> dict[str, float]:
        """Compute evaluation metrics."""
        # This is a placeholder - actual implementation would compute
        # task-specific metrics (accuracy, F1, etc.)
        return {}
    
    def _update_best_metrics(
        self,
        context: CommandContext,
        metrics: dict[str, float]
    ) -> None:
        """Update best metric tracking in context."""
        # Update best validation loss
        if "val_loss" in metrics:
            if metrics["val_loss"] < context.state.best_val_loss:
                context.state.best_val_loss = metrics["val_loss"]
                context.state.no_improvement_count = 0
            else:
                context.state.no_improvement_count += 1
        
        # Update best validation metric (e.g., accuracy)
        primary_metric = context.config.get("primary_metric", "val_accuracy")
        if primary_metric in metrics:
            current_value = metrics[primary_metric]
            # Assume higher is better for non-loss metrics
            if "loss" not in primary_metric:
                if current_value > context.state.best_val_metric:
                    context.state.best_val_metric = current_value
                    context.state.improvement_streak += 1
                else:
                    context.state.improvement_streak = 0
            else:
                # Lower is better for loss metrics
                if current_value < context.state.best_val_metric:
                    context.state.best_val_metric = current_value
                    context.state.improvement_streak += 1
                else:
                    context.state.improvement_streak = 0
    
    def _save_predictions(
        self,
        predictions: list[Any],
        targets: list[Any],
        output_dir: str
    ) -> str:
        """Save predictions to file."""
        # Placeholder - actual implementation would save predictions
        import os
        predictions_path = os.path.join(output_dir, "predictions.json")
        return predictions_path