"""Checkpoint command implementation.

This command handles model checkpointing during training.
"""

from pathlib import Path
from typing import Any

from .base import BaseCommand, CommandContext, CommandResult


class CheckpointCommand(BaseCommand):
    """Command for saving and loading model checkpoints."""
    
    def __init__(
        self,
        save_checkpoint: bool = True,
        save_best_only: bool = False,
        save_on_epoch_end: bool = True,
        save_optimizer_state: bool = True,
        keep_last_n: int = 3,
    ):
        """Initialize checkpoint command.
        
        Args:
            save_checkpoint: Whether to save checkpoints
            save_best_only: Only save when best metric improves
            save_on_epoch_end: Save at epoch end (vs every N steps)
            save_optimizer_state: Include optimizer state in checkpoint
            keep_last_n: Number of recent checkpoints to keep
        """
        super().__init__("Checkpoint")
        self.save_checkpoint = save_checkpoint
        self.save_best_only = save_best_only
        self.save_on_epoch_end = save_on_epoch_end
        self.save_optimizer_state = save_optimizer_state
        self.keep_last_n = keep_last_n
    
    def can_execute(self, context: CommandContext) -> bool:
        """Check if checkpointing can be executed."""
        return (
            context.model is not None
            and context.checkpoint_manager is not None
            and self.save_checkpoint
        )
    
    def execute(self, context: CommandContext) -> CommandResult:
        """Execute checkpointing logic."""
        try:
            # Determine if we should save
            should_save = self._should_save_checkpoint(context)
            
            if not should_save:
                return CommandResult(
                    success=True,
                    outputs={"checkpoint_saved": False}
                )
            
            # Determine if this is the best checkpoint
            is_best = self._is_best_checkpoint(context)
            
            # Save checkpoint
            checkpoint_path = context.checkpoint_manager.save_checkpoint(
                model=context.model,
                optimizer=context.optimizer if self.save_optimizer_state else None,
                state=context.state,
                metrics=context.metrics,
                is_best=is_best
            )
            
            # Clean up old checkpoints
            if self.keep_last_n > 0:
                context.checkpoint_manager.cleanup_old_checkpoints(
                    keep_best=1,
                    keep_last=self.keep_last_n
                )
            
            # Build result
            result = CommandResult(
                success=True,
                outputs={
                    "checkpoint_saved": True,
                    "checkpoint_path": str(checkpoint_path),
                    "is_best": is_best,
                },
                metrics={
                    "checkpoint_step": context.state.global_step,
                }
            )
            
            return result
            
        except Exception as e:
            return CommandResult(
                success=False,
                error=e,
                should_continue=True  # Don't stop training on checkpoint failure
            )
    
    def _should_save_checkpoint(self, context: CommandContext) -> bool:
        """Determine if checkpoint should be saved."""
        if not self.save_checkpoint:
            return False
        
        # Check if we're only saving best checkpoints
        if self.save_best_only:
            return self._is_best_checkpoint(context)
        
        # Check if we're saving on epoch end
        if self.save_on_epoch_end:
            # This would typically check if we're at epoch boundary
            # For now, we'll use a simple step-based check
            save_steps = context.config.get("save_steps", 1000)
            return context.state.global_step % save_steps == 0
        
        return True
    
    def _is_best_checkpoint(self, context: CommandContext) -> bool:
        """Determine if current checkpoint is the best so far."""
        primary_metric = context.config.get("primary_metric", "val_loss")
        
        if primary_metric not in context.metrics:
            return False
        
        current_value = context.metrics[primary_metric]
        
        # For loss metrics, lower is better
        if "loss" in primary_metric:
            best_value = context.state.best_val_loss
            return current_value < best_value
        else:
            # For other metrics (accuracy, etc.), higher is better
            best_value = context.state.best_val_metric
            return current_value > best_value


class LoadCheckpointCommand(BaseCommand):
    """Command for loading model checkpoints."""
    
    def __init__(self, checkpoint_path: Path | str):
        """Initialize load checkpoint command.
        
        Args:
            checkpoint_path: Path to checkpoint to load
        """
        super().__init__("LoadCheckpoint")
        self.checkpoint_path = Path(checkpoint_path)
    
    def can_execute(self, context: CommandContext) -> bool:
        """Check if checkpoint loading can be executed."""
        return (
            context.model is not None
            and context.checkpoint_manager is not None
            and self.checkpoint_path.exists()
        )
    
    def execute(self, context: CommandContext) -> CommandResult:
        """Execute checkpoint loading."""
        try:
            # Load checkpoint
            loaded_state = context.checkpoint_manager.load_checkpoint(
                path=self.checkpoint_path,
                model=context.model,
                optimizer=context.optimizer
            )
            
            # Update context state
            context.state = loaded_state
            
            # Build result
            result = CommandResult(
                success=True,
                outputs={
                    "checkpoint_loaded": True,
                    "loaded_from": str(self.checkpoint_path),
                    "resumed_step": loaded_state.global_step,
                    "resumed_epoch": loaded_state.epoch,
                }
            )
            
            return result
            
        except Exception as e:
            return CommandResult(
                success=False,
                error=e,
                should_continue=False  # Can't continue if checkpoint load fails
            )