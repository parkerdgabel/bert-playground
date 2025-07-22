"""Logging command implementation.

This command handles metrics logging and progress reporting.
"""

import time
from typing import Any

from .base import BaseCommand, CommandContext, CommandResult


class LoggingCommand(BaseCommand):
    """Command for logging training progress and metrics."""
    
    def __init__(
        self,
        log_interval: int = 10,
        log_to_file: bool = True,
        log_to_console: bool = True,
        log_to_mlflow: bool = True,
        verbose: bool = False,
    ):
        """Initialize logging command.
        
        Args:
            log_interval: Log every N steps
            log_to_file: Whether to log to file
            log_to_console: Whether to log to console
            log_to_mlflow: Whether to log to MLflow
            verbose: Whether to include detailed metrics
        """
        super().__init__("Logging")
        self.log_interval = log_interval
        self.log_to_file = log_to_file
        self.log_to_console = log_to_console
        self.log_to_mlflow = log_to_mlflow
        self.verbose = verbose
        self._last_log_step = 0
        self._start_time = time.time()
    
    def can_execute(self, context: CommandContext) -> bool:
        """Check if logging should be executed."""
        # Log at specified intervals or if forced
        should_log = (
            context.state.global_step % self.log_interval == 0
            or context.state.global_step == 1  # Always log first step
            or context.outputs.get("force_log", False)
        )
        
        return should_log and context.state.global_step > self._last_log_step
    
    def execute(self, context: CommandContext) -> CommandResult:
        """Execute logging logic."""
        try:
            # Update last log step
            self._last_log_step = context.state.global_step
            
            # Prepare metrics
            metrics = self._prepare_metrics(context)
            
            # Calculate timing information
            elapsed_time = time.time() - self._start_time
            steps_per_second = context.state.global_step / elapsed_time if elapsed_time > 0 else 0
            metrics["steps_per_second"] = steps_per_second
            metrics["elapsed_time"] = elapsed_time
            
            # Log to console
            if self.log_to_console:
                self._log_to_console(context, metrics)
            
            # Log to file
            if self.log_to_file and context.metrics_collector is not None:
                context.metrics_collector.add_metrics(metrics, context.state.global_step)
            
            # Log to MLflow
            if self.log_to_mlflow and "mlflow" in context.outputs:
                self._log_to_mlflow(context, metrics)
            
            # Update training history
            if context.is_training:
                context.state.train_history.append({
                    "step": context.state.global_step,
                    "epoch": context.state.epoch,
                    **metrics
                })
            else:
                context.state.val_history.append({
                    "step": context.state.global_step,
                    "epoch": context.state.epoch,
                    **metrics
                })
            
            # Build result
            result = CommandResult(
                success=True,
                outputs={
                    "logged_metrics": metrics,
                    "log_step": context.state.global_step,
                }
            )
            
            return result
            
        except Exception as e:
            # Logging failures shouldn't stop training
            return CommandResult(
                success=False,
                error=e,
                should_continue=True
            )
    
    def _prepare_metrics(self, context: CommandContext) -> dict[str, float]:
        """Prepare metrics for logging."""
        metrics = {}
        
        # Add basic metrics
        if context.loss is not None:
            prefix = "train" if context.is_training else "val"
            metrics[f"{prefix}_loss"] = float(context.loss)
        
        # Add context metrics
        metrics.update(context.metrics)
        
        # Add state information if verbose
        if self.verbose:
            metrics.update({
                "epoch": context.state.epoch,
                "global_step": context.state.global_step,
                "samples_seen": context.state.samples_seen,
            })
            
            # Add gradient norm if available
            if context.state.grad_norm > 0:
                metrics["grad_norm"] = context.state.grad_norm
            
            # Add learning rate if available
            if "learning_rate" in context.outputs:
                metrics["learning_rate"] = context.outputs["learning_rate"]
        
        return metrics
    
    def _log_to_console(self, context: CommandContext, metrics: dict[str, float]) -> None:
        """Log metrics to console."""
        # Build log message
        mode = "Train" if context.is_training else "Val"
        
        # Basic info
        msg_parts = [
            f"[{mode}]",
            f"Epoch {context.state.epoch}",
            f"Step {context.state.global_step}",
        ]
        
        # Add loss if available
        if "train_loss" in metrics:
            msg_parts.append(f"Loss: {metrics['train_loss']:.4f}")
        elif "val_loss" in metrics:
            msg_parts.append(f"Loss: {metrics['val_loss']:.4f}")
        
        # Add other key metrics
        for key in ["accuracy", "f1_score", "learning_rate", "grad_norm"]:
            if key in metrics:
                value = metrics[key]
                if isinstance(value, float):
                    msg_parts.append(f"{key}: {value:.4f}")
        
        # Add performance metrics
        if "steps_per_second" in metrics:
            msg_parts.append(f"Speed: {metrics['steps_per_second']:.1f} steps/s")
        
        # Log the message
        print(" | ".join(msg_parts))
    
    def _log_to_mlflow(self, context: CommandContext, metrics: dict[str, float]) -> None:
        """Log metrics to MLflow."""
        # This would integrate with MLflow tracking
        # For now, it's a placeholder
        pass