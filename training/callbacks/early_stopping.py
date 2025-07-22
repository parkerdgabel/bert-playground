"""
Early stopping callback to prevent overfitting.
"""

from loguru import logger

from core.protocols import Trainer, TrainingState
from .base import Callback


class EarlyStopping(Callback):
    """
    Early stopping callback that monitors a metric and stops training when it stops improving.

    Args:
        monitor: Metric to monitor (e.g., 'eval_loss', 'eval_accuracy')
        patience: Number of epochs with no improvement to wait
        min_delta: Minimum change to qualify as an improvement
        mode: 'min' or 'max' - whether lower or higher is better
        baseline: Baseline value for the monitored metric
        restore_best: Whether to restore model to best weights when stopped
    """

    def __init__(
        self,
        monitor: str = "eval_loss",
        patience: int = 3,
        min_delta: float = 0.0001,
        mode: str = "min",
        baseline: float | None = None,
        restore_best: bool = True,
    ):
        super().__init__()
        self.monitor = monitor
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.baseline = baseline
        self.restore_best = restore_best

        # Internal state
        self.best_value = None
        self.best_epoch = 0
        self.best_checkpoint_path = None
        self.wait_count = 0
        self.stopped_epoch = 0

        # Set comparison function
        if mode == "min":
            self.monitor_op = lambda a, b: a < b - min_delta
            self.best_value = float("inf")
        else:
            self.monitor_op = lambda a, b: a > b + min_delta
            self.best_value = float("-inf")

        if baseline is not None:
            self.best_value = baseline

    @property
    def priority(self) -> int:
        """Early stopping should run early to prevent unnecessary work."""
        return 20

    def on_train_begin(self, trainer: Trainer, state: TrainingState) -> None:
        """Reset early stopping state."""
        self.wait_count = 0
        self.stopped_epoch = 0
        self.best_epoch = state.epoch

    def on_evaluate_end(
        self, trainer: Trainer, state: TrainingState, metrics: dict
    ) -> None:
        """Check if we should stop training."""
        # Get current value
        current_value = metrics.get(self.monitor)

        if current_value is None:
            logger.warning(
                f"Early stopping monitor '{self.monitor}' not found in metrics. "
                f"Available metrics: {list(metrics.keys())}"
            )
            return

        # Check if improved
        if self.monitor_op(current_value, self.best_value):
            self.best_value = current_value
            self.best_epoch = state.epoch
            self.wait_count = 0

            # Save best checkpoint path if trainer supports it
            if self.restore_best and hasattr(trainer, "checkpoint_manager"):
                self.best_checkpoint_path = (
                    trainer.checkpoint_manager.get_best_checkpoint()
                )

            logger.info(
                f"Early stopping: new best {self.monitor}={current_value:.4f} "
                f"(improved by {abs(current_value - self.best_value):.4f})"
            )
        else:
            self.wait_count += 1
            logger.info(
                f"Early stopping: no improvement in {self.monitor} for {self.wait_count} epochs "
                f"(best={self.best_value:.4f}, current={current_value:.4f})"
            )

            if self.wait_count >= self.patience:
                state.should_stop = True
                self.stopped_epoch = state.epoch
                logger.info(
                    f"Early stopping triggered! No improvement for {self.patience} epochs. "
                    f"Best {self.monitor}={self.best_value:.4f} at epoch {self.best_epoch}"
                )

    def on_train_end(self, trainer: Trainer, state: TrainingState, result) -> None:
        """Restore best weights if requested and training was stopped early."""
        if self.stopped_epoch > 0:
            logger.info(f"Training stopped early at epoch {self.stopped_epoch}")

            if self.restore_best and self.best_checkpoint_path:
                try:
                    trainer.load_checkpoint(self.best_checkpoint_path)
                    logger.info(f"Restored best model from epoch {self.best_epoch}")
                except Exception as e:
                    logger.error(f"Failed to restore best model: {e}")

        # Update result with early stopping info
        if hasattr(result, "early_stopped"):
            result.early_stopped = self.stopped_epoch > 0
        if hasattr(result, "best_epoch"):
            result.best_epoch = self.best_epoch
