"""
Progress bar callback for training visualization.
"""

from typing import Optional, Dict
from tqdm.auto import tqdm
from loguru import logger

from .base import Callback
from ..core.protocols import Trainer, TrainingState


class ProgressBar(Callback):
    """
    Progress bar callback using tqdm.
    
    This callback provides visual progress tracking for:
    - Training epochs
    - Training batches
    - Evaluation batches
    
    Args:
        show_epoch_progress: Show progress bar for epochs
        show_batch_progress: Show progress bar for batches
        show_eval_progress: Show progress bar for evaluation
        leave_epoch: Leave epoch progress bar after completion
        leave_batch: Leave batch progress bar after completion
        update_freq: Update frequency for batch progress
    """
    
    def __init__(
        self,
        show_epoch_progress: bool = True,
        show_batch_progress: bool = True,
        show_eval_progress: bool = True,
        leave_epoch: bool = True,
        leave_batch: bool = False,
        update_freq: int = 1,
    ):
        super().__init__()
        self.show_epoch_progress = show_epoch_progress
        self.show_batch_progress = show_batch_progress
        self.show_eval_progress = show_eval_progress
        self.leave_epoch = leave_epoch
        self.leave_batch = leave_batch
        self.update_freq = update_freq
        
        # Progress bars
        self.epoch_pbar: Optional[tqdm] = None
        self.batch_pbar: Optional[tqdm] = None
        self.eval_pbar: Optional[tqdm] = None
        
        # Tracking
        self.total_epochs = 0
        self.total_batches = 0
        self.batch_count = 0
    
    @property
    def priority(self) -> int:
        """Progress bars should update after metrics are computed."""
        return 80
    
    def on_train_begin(self, trainer: Trainer, state: TrainingState) -> None:
        """Initialize epoch progress bar."""
        self.total_epochs = trainer.config.training.num_epochs
        
        if self.show_epoch_progress:
            self.epoch_pbar = tqdm(
                total=self.total_epochs,
                desc="Training",
                unit="epoch",
                leave=self.leave_epoch,
                position=0,
            )
            self.epoch_pbar.update(state.epoch)
    
    def on_epoch_begin(self, trainer: Trainer, state: TrainingState) -> None:
        """Initialize batch progress bar."""
        # Get number of batches from trainer
        if hasattr(trainer, '_train_dataloader'):
            self.total_batches = len(trainer._train_dataloader)
        else:
            self.total_batches = None
        
        self.batch_count = 0
        
        if self.show_batch_progress and self.total_batches:
            self.batch_pbar = tqdm(
                total=self.total_batches,
                desc=f"Epoch {state.epoch + 1}/{self.total_epochs}",
                unit="batch",
                leave=self.leave_batch,
                position=1,
            )
    
    def on_batch_end(self, trainer: Trainer, state: TrainingState, loss: float) -> None:
        """Update batch progress bar."""
        self.batch_count += 1
        
        if self.batch_pbar and self.batch_count % self.update_freq == 0:
            # Update progress
            self.batch_pbar.update(self.update_freq)
            
            # Update postfix with metrics
            metrics = {
                "loss": f"{loss:.4f}",
                "lr": f"{trainer.optimizer.learning_rate:.2e}",
            }
            
            # Add gradient norm if available
            if hasattr(state, 'grad_norm'):
                metrics["grad_norm"] = f"{state.grad_norm:.2f}"
            
            self.batch_pbar.set_postfix(metrics)
    
    def on_epoch_end(self, trainer: Trainer, state: TrainingState) -> None:
        """Close batch progress bar and update epoch progress."""
        # Close batch progress
        if self.batch_pbar:
            self.batch_pbar.close()
            self.batch_pbar = None
        
        # Update epoch progress
        if self.epoch_pbar:
            self.epoch_pbar.update(1)
            
            # Update postfix with epoch metrics
            metrics = {
                "train_loss": f"{state.train_loss:.4f}",
            }
            
            if state.val_loss > 0:
                metrics["val_loss"] = f"{state.val_loss:.4f}"
            
            if state.best_val_metric > 0:
                metrics["best"] = f"{state.best_val_metric:.4f}"
            
            self.epoch_pbar.set_postfix(metrics)
    
    def on_evaluate_begin(self, trainer: Trainer, state: TrainingState) -> None:
        """Initialize evaluation progress bar."""
        if not self.show_eval_progress:
            return
        
        # Get number of eval batches
        if hasattr(trainer, '_val_dataloader'):
            total_eval_batches = len(trainer._val_dataloader)
        else:
            total_eval_batches = None
        
        if total_eval_batches:
            self.eval_pbar = tqdm(
                total=total_eval_batches,
                desc="Evaluating",
                unit="batch",
                leave=False,
                position=2,
            )
    
    def on_evaluate_end(self, trainer: Trainer, state: TrainingState, metrics: Dict[str, float]) -> None:
        """Close evaluation progress bar."""
        if self.eval_pbar:
            self.eval_pbar.close()
            self.eval_pbar = None
    
    def on_train_end(self, trainer: Trainer, state: TrainingState, result) -> None:
        """Close all progress bars."""
        if self.epoch_pbar:
            self.epoch_pbar.close()
            self.epoch_pbar = None
        
        if self.batch_pbar:
            self.batch_pbar.close()
            self.batch_pbar = None
        
        if self.eval_pbar:
            self.eval_pbar.close()
            self.eval_pbar = None