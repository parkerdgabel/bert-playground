"""
Model checkpoint callback for saving models during training.
"""

from pathlib import Path

from loguru import logger

from core.protocols import Trainer, TrainingState
from .base import Callback


class ModelCheckpoint(Callback):
    """
    Callback to save model checkpoints during training.

    Args:
        monitor: Metric to monitor for best model
        save_dir: Directory to save checkpoints
        filename_prefix: Prefix for checkpoint filenames
        save_best_only: Only save when monitored metric improves
        save_freq: Save frequency - 'epoch' or int for steps
        mode: 'min' or 'max' - whether lower or higher is better
        verbose: Whether to print messages
        save_weights_only: Only save model weights (not full trainer state)
    """

    def __init__(
        self,
        monitor: str = "eval_loss",
        save_dir: Path | None = None,
        filename_prefix: str = "checkpoint",
        save_best_only: bool = False,
        save_freq: str | int = "epoch",
        mode: str = "min",
        verbose: bool = True,
        save_weights_only: bool = False,
    ):
        super().__init__()
        self.monitor = monitor
        self.save_dir = save_dir
        self.filename_prefix = filename_prefix
        self.save_best_only = save_best_only
        self.save_freq = save_freq
        self.mode = mode
        self.verbose = verbose
        self.save_weights_only = save_weights_only

        # Internal state
        self.best_value = None
        self.last_saved_step = 0

        # Set comparison function
        if mode == "min":
            self.monitor_op = lambda a, b: a < b
            self.best_value = float("inf")
        else:
            self.monitor_op = lambda a, b: a > b
            self.best_value = float("-inf")

    @property
    def priority(self) -> int:
        """Checkpoint saving should happen after metrics are computed."""
        return 60

    def on_train_begin(self, trainer: Trainer, state: TrainingState) -> None:
        """Initialize checkpoint directory."""
        if self.save_dir is None:
            self.save_dir = trainer.config.environment.output_dir / "checkpoints"

        self.save_dir = Path(self.save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

    def on_batch_end(self, trainer: Trainer, state: TrainingState, loss: float) -> None:
        """Check if we should save based on step frequency."""
        if isinstance(self.save_freq, int) and state.global_step % self.save_freq == 0:
            if state.global_step > self.last_saved_step:
                self._save_checkpoint(trainer, state, is_scheduled=True)
                self.last_saved_step = state.global_step

    def on_epoch_end(self, trainer: Trainer, state: TrainingState) -> None:
        """Check if we should save based on epoch frequency."""
        if self.save_freq == "epoch":
            self._save_checkpoint(trainer, state, is_scheduled=True)

    def on_evaluate_end(
        self, trainer: Trainer, state: TrainingState, metrics: dict
    ) -> None:
        """Check if we should save based on metric improvement."""
        if self.save_best_only:
            current_value = metrics.get(self.monitor)

            if current_value is None:
                if self.verbose:
                    logger.warning(
                        f"ModelCheckpoint monitor '{self.monitor}' not found in metrics. "
                        f"Available metrics: {list(metrics.keys())}"
                    )
                return

            # Check if improved
            if self.monitor_op(current_value, self.best_value):
                self.best_value = current_value
                self._save_checkpoint(trainer, state, is_best=True)

                if self.verbose:
                    logger.info(
                        f"ModelCheckpoint: new best {self.monitor}={current_value:.4f}, "
                        f"saving checkpoint"
                    )

    def _save_checkpoint(
        self,
        trainer: Trainer,
        state: TrainingState,
        is_best: bool = False,
        is_scheduled: bool = False,
    ) -> None:
        """Save a checkpoint."""
        # Skip if save_best_only and not best
        if self.save_best_only and not is_best:
            return

        # Determine checkpoint name
        if is_best:
            name = f"{self.filename_prefix}_best"
        else:
            name = (
                f"{self.filename_prefix}_epoch_{state.epoch}_step_{state.global_step}"
            )

        # Save checkpoint
        if self.save_weights_only:
            # Only save model weights
            checkpoint_path = self.save_dir / name
            checkpoint_path.mkdir(exist_ok=True)

            model_path = checkpoint_path / "model.safetensors"
            trainer.model.save_pretrained(checkpoint_path)

            if self.verbose:
                logger.info(f"Saved model weights to {checkpoint_path}")
        else:
            # Save full checkpoint through trainer
            checkpoint_path = trainer.save_checkpoint(self.save_dir / name)

            if self.verbose:
                logger.info(f"Saved checkpoint to {checkpoint_path}")

        # Call trainer hook
        trainer._call_hooks("on_checkpoint_save", state, str(checkpoint_path))
