"""
Learning rate scheduler callback for dynamic learning rate adjustment.
"""

from typing import Optional, Dict
from loguru import logger

from .base import Callback
from ..core.protocols import Trainer, TrainingState


class LearningRateScheduler(Callback):
    """
    Callback for learning rate scheduling.
    
    This callback handles learning rate updates based on the configured scheduler.
    It can work with step-based or epoch-based scheduling.
    
    Args:
        update_freq: When to update LR - 'step', 'epoch', or int for every N steps
        verbose: Whether to log LR changes
        warmup_steps: Override warmup steps from config
    """
    
    def __init__(
        self,
        update_freq: str | int = "step",
        verbose: bool = True,
        warmup_steps: Optional[int] = None,
    ):
        super().__init__()
        self.update_freq = update_freq
        self.verbose = verbose
        self.warmup_steps = warmup_steps
        
        self.last_lr = None
        self.scheduler = None
    
    @property
    def priority(self) -> int:
        """LR scheduling should happen early in the pipeline."""
        return 30
    
    def on_train_begin(self, trainer: Trainer, state: TrainingState) -> None:
        """Initialize scheduler."""
        # Get scheduler from trainer
        if hasattr(trainer, 'lr_scheduler'):
            self.scheduler = trainer.lr_scheduler
            
            # Override warmup steps if provided
            if self.warmup_steps is not None and hasattr(self.scheduler, 'warmup_steps'):
                self.scheduler.warmup_steps = self.warmup_steps
            
            # Get initial LR
            self.last_lr = trainer.optimizer.learning_rate
            
            if self.verbose:
                logger.info(f"LearningRateScheduler initialized with {self.scheduler.__class__.__name__}")
    
    def on_batch_end(self, trainer: Trainer, state: TrainingState, loss: float) -> None:
        """Update LR after batch if configured."""
        if self.scheduler is None:
            return
        
        should_update = False
        
        if self.update_freq == "step":
            should_update = True
        elif isinstance(self.update_freq, int) and state.global_step % self.update_freq == 0:
            should_update = True
        
        if should_update:
            self._update_lr(trainer, state)
    
    def on_epoch_end(self, trainer: Trainer, state: TrainingState) -> None:
        """Update LR after epoch if configured."""
        if self.scheduler is None:
            return
        
        if self.update_freq == "epoch":
            self._update_lr(trainer, state)
    
    def on_evaluate_end(self, trainer: Trainer, state: TrainingState, metrics: Dict[str, float]) -> None:
        """Update LR based on metrics for ReduceLROnPlateau scheduler."""
        if self.scheduler is None:
            return
        
        # Check if scheduler needs metrics
        if hasattr(self.scheduler, 'step') and 'metrics' in self.scheduler.step.__code__.co_varnames:
            current_lr = self.scheduler.step(metrics)
            
            if self.verbose and current_lr != self.last_lr:
                logger.info(
                    f"Learning rate changed: {self.last_lr:.2e} → {current_lr:.2e} "
                    f"(based on metrics)"
                )
                self.last_lr = current_lr
    
    def _update_lr(self, trainer: Trainer, state: TrainingState) -> None:
        """Update learning rate."""
        if self.scheduler is None:
            return
        
        # Update LR
        current_lr = self.scheduler.step()
        
        # Log if changed
        if self.verbose and current_lr != self.last_lr:
            # Determine warmup status
            in_warmup = False
            if hasattr(self.scheduler, 'warmup_steps') and state.global_step <= self.scheduler.warmup_steps:
                in_warmup = True
            
            status = "warmup" if in_warmup else "schedule"
            logger.info(
                f"Learning rate changed: {self.last_lr:.2e} → {current_lr:.2e} "
                f"(step {state.global_step}, {status})"
            )
            
            self.last_lr = current_lr
        
        # Update state
        if hasattr(state, 'learning_rate'):
            # Ensure learning rate is converted to Python float
            state.learning_rate = float(current_lr)