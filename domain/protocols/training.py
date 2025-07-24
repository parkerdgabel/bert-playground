"""Domain training protocols - Core training abstractions.

These protocols define the fundamental training contracts used throughout
the system. They are independent of any specific implementation.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Protocol

from .compute import Array, Module
from .data import DataLoader
from .models import Model


@dataclass
class TrainingState:
    """Represents the current state of training."""

    # Progress tracking
    epoch: int = 0
    global_step: int = 0
    samples_seen: int = 0

    # Performance metrics
    train_loss: float = 0.0
    val_loss: float = 0.0
    best_val_loss: float = float("inf")
    best_val_metric: float = 0.0
    metrics: dict[str, float] = None

    # Training history
    train_history: list[dict[str, float]] = None
    val_history: list[dict[str, float]] = None

    # State flags
    should_stop: bool = False
    improvement_streak: int = 0
    no_improvement_count: int = 0

    # Timing
    epoch_start_time: float = 0.0
    training_start_time: float = 0.0

    # Gradient norm (store as float, not MLX array)
    grad_norm: float = 0.0

    def __post_init__(self):
        if self.metrics is None:
            self.metrics = {}
        if self.train_history is None:
            self.train_history = []
        if self.val_history is None:
            self.val_history = []

    def to_dict(self) -> dict[str, Any]:
        """Convert state to dictionary for serialization."""
        return {
            "epoch": self.epoch,
            "global_step": self.global_step,
            "samples_seen": self.samples_seen,
            "train_loss": self.train_loss,
            "val_loss": self.val_loss,
            "best_val_loss": self.best_val_loss,
            "best_val_metric": self.best_val_metric,
            "metrics": self.metrics,
            "train_history": self.train_history,
            "val_history": self.val_history,
            "should_stop": self.should_stop,
            "improvement_streak": self.improvement_streak,
            "no_improvement_count": self.no_improvement_count,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TrainingState":
        """Create state from dictionary."""
        return cls(**data)


class Trainer(Protocol):
    """Protocol for trainer implementations."""

    def train(
        self,
        train_dataloader: DataLoader,
        val_dataloader: DataLoader | None = None,
        resume_from: Path | None = None,
    ) -> dict[str, Any]:
        """Run the training loop.

        Args:
            train_dataloader: Training data loader
            val_dataloader: Optional validation data loader
            resume_from: Optional checkpoint path to resume from

        Returns:
            Training result with final metrics and paths
        """
        ...

    def evaluate(self, dataloader: DataLoader) -> dict[str, float]:
        """Evaluate the model on a dataset.

        Args:
            dataloader: Data loader for evaluation

        Returns:
            Dictionary of metrics
        """
        ...

    def predict(self, dataloader: DataLoader) -> Array:
        """Generate predictions for a dataset.

        Args:
            dataloader: Data loader for prediction

        Returns:
            Predictions as array
        """
        ...

    @property
    def model(self) -> Model:
        """Get the model being trained."""
        ...

    @property
    def state(self) -> TrainingState:
        """Get current training state."""
        ...

    def save_checkpoint(self, path: Path) -> None:
        """Save training checkpoint."""
        ...

    def load_checkpoint(self, path: Path) -> None:
        """Load training checkpoint."""
        ...


class Optimizer(Protocol):
    """Protocol for optimizers."""

    def update(self, model: Module, gradients: dict[str, Array]) -> None:
        """Update model parameters with gradients.
        
        Args:
            model: Model to update
            gradients: Gradients for each parameter
        """
        ...

    @property
    def learning_rate(self) -> float:
        """Current learning rate."""
        ...

    @property
    def state(self) -> dict[str, Any]:
        """Optimizer state."""
        ...


class Scheduler(Protocol):
    """Protocol for learning rate schedulers."""

    def step(self, metrics: dict[str, float] | None = None) -> float:
        """Update learning rate and return new value.
        
        Args:
            metrics: Optional metrics for schedulers that use them
            
        Returns:
            New learning rate
        """
        ...

    @property
    def current_lr(self) -> float:
        """Current learning rate."""
        ...

    def state_dict(self) -> dict[str, Any]:
        """Get scheduler state.
        
        Returns:
            State dictionary
        """
        ...

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Load scheduler state.
        
        Args:
            state: State dictionary to load
        """
        ...


class Callback(Protocol):
    """Protocol for training callbacks."""

    def on_train_begin(self, trainer: Trainer, state: TrainingState) -> None:
        """Called at the beginning of training."""
        ...

    def on_train_end(
        self, trainer: Trainer, state: TrainingState, result: dict[str, Any]
    ) -> None:
        """Called at the end of training."""
        ...

    def on_epoch_begin(self, trainer: Trainer, state: TrainingState) -> None:
        """Called at the beginning of each epoch."""
        ...

    def on_epoch_end(self, trainer: Trainer, state: TrainingState) -> None:
        """Called at the end of each epoch."""
        ...

    def on_batch_begin(
        self, trainer: Trainer, state: TrainingState, batch: dict[str, Array]
    ) -> None:
        """Called before processing each batch."""
        ...

    def on_batch_end(self, trainer: Trainer, state: TrainingState, loss: float) -> None:
        """Called after processing each batch."""
        ...

    def on_evaluate_begin(self, trainer: Trainer, state: TrainingState) -> None:
        """Called before evaluation."""
        ...

    def on_evaluate_end(
        self, trainer: Trainer, state: TrainingState, metrics: dict[str, float]
    ) -> None:
        """Called after evaluation."""
        ...


class Metric(Protocol):
    """Protocol for metrics computation."""

    @property
    def name(self) -> str:
        """Name of the metric."""
        ...

    def update(self, predictions: Array, targets: Array) -> None:
        """Update metric with batch results.
        
        Args:
            predictions: Model predictions
            targets: Ground truth targets
        """
        ...

    def compute(self) -> float:
        """Compute final metric value.
        
        Returns:
            Metric value
        """
        ...

    def reset(self) -> None:
        """Reset metric state."""
        ...