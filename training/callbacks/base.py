"""
Base callback implementation and callback list manager.
"""

from abc import ABC
from typing import Any

import mlx.core as mx
from loguru import logger

from core.protocols import Trainer, TrainingResult, TrainingState


class Callback(ABC):
    """
    Base class for training callbacks.

    Callbacks allow customization of training behavior at various points.
    All methods are optional - implement only the ones you need.
    """

    def __init__(self):
        """Initialize callback."""
        self.trainer: Trainer | None = None

    def set_trainer(self, trainer: Trainer) -> None:
        """Set the trainer instance."""
        self.trainer = trainer

    def on_train_begin(self, trainer: Trainer, state: TrainingState) -> None:
        """Called at the beginning of training."""
        pass

    def on_train_end(
        self, trainer: Trainer, state: TrainingState, result: TrainingResult
    ) -> None:
        """Called at the end of training."""
        pass

    def on_epoch_begin(self, trainer: Trainer, state: TrainingState) -> None:
        """Called at the beginning of each epoch."""
        pass

    def on_epoch_end(self, trainer: Trainer, state: TrainingState) -> None:
        """Called at the end of each epoch."""
        pass

    def on_batch_begin(
        self, trainer: Trainer, state: TrainingState, batch: dict[str, mx.array]
    ) -> None:
        """Called before processing each batch."""
        pass

    def on_batch_end(self, trainer: Trainer, state: TrainingState, loss: float) -> None:
        """Called after processing each batch."""
        pass

    def on_evaluate_begin(self, trainer: Trainer, state: TrainingState) -> None:
        """Called before evaluation."""
        pass

    def on_evaluate_end(
        self, trainer: Trainer, state: TrainingState, metrics: dict[str, float]
    ) -> None:
        """Called after evaluation."""
        pass

    def on_checkpoint_save(
        self, trainer: Trainer, state: TrainingState, checkpoint_path: str
    ) -> None:
        """Called after saving a checkpoint."""
        pass

    def on_checkpoint_load(
        self, trainer: Trainer, state: TrainingState, checkpoint_path: str
    ) -> None:
        """Called after loading a checkpoint."""
        pass

    def on_log(
        self, trainer: Trainer, state: TrainingState, logs: dict[str, Any]
    ) -> None:
        """Called when logging metrics."""
        pass

    @property
    def priority(self) -> int:
        """
        Priority for callback execution order.
        Lower values are executed first.
        """
        return 50


class CallbackList:
    """
    Manager for multiple callbacks.

    Handles callback registration, ordering, and execution.
    """

    def __init__(self, callbacks: list[Callback] | None = None):
        """
        Initialize callback list.

        Args:
            callbacks: Optional list of callbacks
        """
        self.callbacks = callbacks or []
        self._trainer: Trainer | None = None

        # Sort by priority
        self._sort_callbacks()

    def _sort_callbacks(self) -> None:
        """Sort callbacks by priority."""
        self.callbacks.sort(key=lambda cb: cb.priority)

    def append(self, callback: Callback) -> None:
        """Add a callback to the list."""
        self.callbacks.append(callback)
        if self._trainer is not None:
            callback.set_trainer(self._trainer)
        self._sort_callbacks()

    def extend(self, callbacks: list[Callback]) -> None:
        """Add multiple callbacks to the list."""
        for callback in callbacks:
            if self._trainer is not None:
                callback.set_trainer(self._trainer)
        self.callbacks.extend(callbacks)
        self._sort_callbacks()

    def remove(self, callback: Callback) -> None:
        """Remove a callback from the list."""
        self.callbacks.remove(callback)

    def set_trainer(self, trainer: Trainer) -> None:
        """Set trainer for all callbacks."""
        self._trainer = trainer
        for callback in self.callbacks:
            callback.set_trainer(trainer)

    def on_train_begin(self, trainer: Trainer, state: TrainingState) -> None:
        """Called at the beginning of training."""
        for callback in self.callbacks:
            try:
                callback.on_train_begin(trainer, state)
            except Exception as e:
                logger.error(
                    f"Error in {callback.__class__.__name__}.on_train_begin: {e}"
                )

    def on_train_end(
        self, trainer: Trainer, state: TrainingState, result: TrainingResult
    ) -> None:
        """Called at the end of training."""
        for callback in self.callbacks:
            try:
                callback.on_train_end(trainer, state, result)
            except Exception as e:
                logger.error(
                    f"Error in {callback.__class__.__name__}.on_train_end: {e}"
                )

    def on_epoch_begin(self, trainer: Trainer, state: TrainingState) -> None:
        """Called at the beginning of each epoch."""
        for callback in self.callbacks:
            try:
                callback.on_epoch_begin(trainer, state)
            except Exception as e:
                logger.error(
                    f"Error in {callback.__class__.__name__}.on_epoch_begin: {e}"
                )

    def on_epoch_end(self, trainer: Trainer, state: TrainingState) -> None:
        """Called at the end of each epoch."""
        for callback in self.callbacks:
            try:
                callback.on_epoch_end(trainer, state)
            except Exception as e:
                logger.error(
                    f"Error in {callback.__class__.__name__}.on_epoch_end: {e}"
                )

    def on_batch_begin(
        self, trainer: Trainer, state: TrainingState, batch: dict[str, mx.array]
    ) -> None:
        """Called before processing each batch."""
        for callback in self.callbacks:
            try:
                callback.on_batch_begin(trainer, state, batch)
            except Exception as e:
                logger.error(
                    f"Error in {callback.__class__.__name__}.on_batch_begin: {e}"
                )

    def on_batch_end(self, trainer: Trainer, state: TrainingState, loss: float) -> None:
        """Called after processing each batch."""
        for callback in self.callbacks:
            try:
                callback.on_batch_end(trainer, state, loss)
            except Exception as e:
                logger.error(
                    f"Error in {callback.__class__.__name__}.on_batch_end: {e}"
                )

    def on_evaluate_begin(self, trainer: Trainer, state: TrainingState) -> None:
        """Called before evaluation."""
        for callback in self.callbacks:
            try:
                callback.on_evaluate_begin(trainer, state)
            except Exception as e:
                logger.error(
                    f"Error in {callback.__class__.__name__}.on_evaluate_begin: {e}"
                )

    def on_evaluate_end(
        self, trainer: Trainer, state: TrainingState, metrics: dict[str, float]
    ) -> None:
        """Called after evaluation."""
        for callback in self.callbacks:
            try:
                callback.on_evaluate_end(trainer, state, metrics)
            except Exception as e:
                logger.error(
                    f"Error in {callback.__class__.__name__}.on_evaluate_end: {e}"
                )

    def on_checkpoint_save(
        self, trainer: Trainer, state: TrainingState, checkpoint_path: str
    ) -> None:
        """Called after saving a checkpoint."""
        for callback in self.callbacks:
            try:
                callback.on_checkpoint_save(trainer, state, checkpoint_path)
            except Exception as e:
                logger.error(
                    f"Error in {callback.__class__.__name__}.on_checkpoint_save: {e}"
                )

    def on_checkpoint_load(
        self, trainer: Trainer, state: TrainingState, checkpoint_path: str
    ) -> None:
        """Called after loading a checkpoint."""
        for callback in self.callbacks:
            try:
                callback.on_checkpoint_load(trainer, state, checkpoint_path)
            except Exception as e:
                logger.error(
                    f"Error in {callback.__class__.__name__}.on_checkpoint_load: {e}"
                )

    def on_log(
        self, trainer: Trainer, state: TrainingState, logs: dict[str, Any]
    ) -> None:
        """Called when logging metrics."""
        for callback in self.callbacks:
            try:
                callback.on_log(trainer, state, logs)
            except Exception as e:
                logger.error(f"Error in {callback.__class__.__name__}.on_log: {e}")

    def __len__(self) -> int:
        """Get number of callbacks."""
        return len(self.callbacks)

    def __getitem__(self, index: int) -> Callback:
        """Get callback by index."""
        return self.callbacks[index]

    def __iter__(self):
        """Iterate over callbacks."""
        return iter(self.callbacks)
