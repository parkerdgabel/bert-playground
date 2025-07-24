"""Primary training port - Training API exposed to external actors.

This port defines the training interface that external actors (CLI, web UI, etc.)
use to train models. It's a driving port in hexagonal architecture.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, runtime_checkable, Optional

from infrastructure.di import port
from domain.protocols.data import DataLoader
from domain.protocols.models import Model


@dataclass
class TrainingState:
    """Represents the current state of training.
    
    This is a data structure that external actors can inspect to understand
    the current training progress.
    """

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

    # Gradient norm
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


@dataclass
class TrainingResult:
    """Result of a training run that external actors receive."""

    # Final metrics
    final_train_loss: float
    final_val_loss: float
    best_val_loss: float
    best_val_metric: float
    final_metrics: dict[str, float]

    # Training history
    train_history: list[dict[str, float]]
    val_history: list[dict[str, float]]

    # Model paths
    final_model_path: Optional[Path] = None
    best_model_path: Optional[Path] = None

    # Metadata
    total_epochs: int = 0
    total_steps: int = 0
    total_time: float = 0.0
    early_stopped: bool = False
    stop_reason: Optional[str] = None

    # Tracking
    run_id: Optional[str] = None
    experiment_id: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        """Convert result to dictionary."""
        return {
            "final_train_loss": self.final_train_loss,
            "final_val_loss": self.final_val_loss,
            "best_val_loss": self.best_val_loss,
            "best_val_metric": self.best_val_metric,
            "final_metrics": self.final_metrics,
            "train_history": self.train_history,
            "val_history": self.val_history,
            "final_model_path": str(self.final_model_path) if self.final_model_path else None,
            "best_model_path": str(self.best_model_path) if self.best_model_path else None,
            "total_epochs": self.total_epochs,
            "total_steps": self.total_steps,
            "total_time": self.total_time,
            "early_stopped": self.early_stopped,
            "stop_reason": self.stop_reason,
            "run_id": self.run_id,
            "experiment_id": self.experiment_id,
        }


@port()
@runtime_checkable
class TrainerConfig(Protocol):
    """Configuration protocol for trainers - what external actors provide."""

    @property
    def num_epochs(self) -> int:
        """Number of training epochs."""
        ...

    @property
    def learning_rate(self) -> float:
        """Initial learning rate."""
        ...

    @property
    def batch_size(self) -> int:
        """Training batch size."""
        ...

    @property
    def gradient_accumulation_steps(self) -> int:
        """Number of gradient accumulation steps."""
        ...

    @property
    def eval_steps(self) -> int:
        """Evaluate every N steps."""
        ...

    @property
    def save_steps(self) -> int:
        """Save checkpoint every N steps."""
        ...

    @property
    def output_dir(self) -> Path:
        """Output directory for checkpoints and logs."""
        ...

    def to_dict(self) -> dict[str, Any]:
        """Convert config to dictionary."""
        ...

    def validate(self) -> list[str]:
        """Validate configuration and return list of errors."""
        ...


@port()
@runtime_checkable
class Trainer(Protocol):
    """Primary training port - the main training API exposed to external actors.
    
    This is what the CLI, web UI, or other external systems use to train models.
    It's a driving port that the application core must implement.
    """

    def train(
        self,
        train_dataloader: DataLoader,
        val_dataloader: Optional[DataLoader] = None,
        resume_from: Optional[Path] = None,
    ) -> TrainingResult:
        """Run the training loop.

        This is the main entry point that external actors call to train a model.

        Args:
            train_dataloader: Training data loader
            val_dataloader: Optional validation data loader
            resume_from: Optional checkpoint path to resume from

        Returns:
            TrainingResult with final metrics and paths
        """
        ...

    def evaluate(self, dataloader: DataLoader) -> dict[str, float]:
        """Evaluate the model on a dataset.

        External actors use this to evaluate a trained model.

        Args:
            dataloader: Data loader for evaluation

        Returns:
            Dictionary of metrics
        """
        ...

    def predict(self, dataloader: DataLoader) -> Any:
        """Generate predictions for a dataset.

        External actors use this to get predictions from a trained model.

        Args:
            dataloader: Data loader for prediction

        Returns:
            Predictions (framework-agnostic)
        """
        ...

    @property
    def model(self) -> Model:
        """Get the model being trained."""
        ...

    @property
    def config(self) -> TrainerConfig:
        """Get trainer configuration."""
        ...

    @property
    def state(self) -> TrainingState:
        """Get current training state."""
        ...

    def save_checkpoint(self, path: Path) -> None:
        """Save training checkpoint.
        
        External actors can call this to manually save checkpoints.
        """
        ...

    def load_checkpoint(self, path: Path) -> None:
        """Load training checkpoint.
        
        External actors can call this to load a specific checkpoint.
        """
        ...


@port()
@runtime_checkable
class TrainingStrategy(Protocol):
    """Strategy pattern for different training approaches.
    
    External actors can select different training strategies.
    """

    @property
    def name(self) -> str:
        """Strategy name."""
        ...

    @property
    def description(self) -> str:
        """Strategy description."""
        ...

    def create_trainer(self, model: Model, config: TrainerConfig) -> Trainer:
        """Create a trainer using this strategy.
        
        Args:
            model: Model to train
            config: Training configuration
            
        Returns:
            Configured trainer
        """
        ...

    def get_default_config(self) -> dict[str, Any]:
        """Get default configuration for this strategy."""
        ...


# Convenience functions that external actors can use
def train_model(
    model: Model,
    train_data: DataLoader,
    config: TrainerConfig,
    val_data: Optional[DataLoader] = None,
    strategy: Optional[TrainingStrategy] = None,
) -> TrainingResult:
    """High-level function to train a model.
    
    This is a convenience function that external actors can use
    instead of directly working with Trainer instances.
    
    Args:
        model: Model to train
        train_data: Training data
        config: Training configuration
        val_data: Optional validation data
        strategy: Optional training strategy
        
    Returns:
        Training result
    """
    # This would be implemented by the application core
    raise NotImplementedError("To be implemented by application core")


def evaluate_model(
    model: Model,
    data: DataLoader,
    metrics: Optional[list[str]] = None,
) -> dict[str, float]:
    """Evaluate a model on a dataset.
    
    Args:
        model: Model to evaluate
        data: Evaluation data
        metrics: Optional list of metrics to compute
        
    Returns:
        Dictionary of metric values
    """
    # This would be implemented by the application core
    raise NotImplementedError("To be implemented by application core")


def predict_with_model(
    model: Model,
    data: DataLoader,
    batch_size: Optional[int] = None,
) -> Any:
    """Generate predictions with a model.
    
    Args:
        model: Model to use
        data: Input data
        batch_size: Optional batch size override
        
    Returns:
        Predictions
    """
    # This would be implemented by the application core
    raise NotImplementedError("To be implemented by application core")