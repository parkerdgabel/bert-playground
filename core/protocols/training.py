"""Training-related protocols for k-bert.

These protocols define the contracts for training components including
trainers, optimizers, schedulers, callbacks, and metrics.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Protocol

from ports.secondary.compute import Array, Module

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


@dataclass
class TrainingResult:
    """Result of a training run."""

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
    final_model_path: Path | None = None
    best_model_path: Path | None = None

    # Metadata
    total_epochs: int = 0
    total_steps: int = 0
    total_time: float = 0.0
    early_stopped: bool = False
    stop_reason: str | None = None

    # MLflow tracking
    mlflow_run_id: str | None = None
    mlflow_experiment_id: str | None = None

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
            "final_model_path": str(self.final_model_path)
            if self.final_model_path
            else None,
            "best_model_path": str(self.best_model_path)
            if self.best_model_path
            else None,
            "total_epochs": self.total_epochs,
            "total_steps": self.total_steps,
            "total_time": self.total_time,
            "early_stopped": self.early_stopped,
            "stop_reason": self.stop_reason,
            "mlflow_run_id": self.mlflow_run_id,
            "mlflow_experiment_id": self.mlflow_experiment_id,
        }


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


class LRScheduler(Protocol):
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


class TrainerConfig(Protocol):
    """Protocol for trainer configuration."""

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

    @classmethod
    def from_dict(cls, config: dict[str, Any]) -> "TrainerConfig":
        """Create config from dictionary."""
        ...

    def validate(self) -> list[str]:
        """Validate configuration and return list of errors."""
        ...


class Trainer(Protocol):
    """Protocol for trainer implementations."""

    def train(
        self,
        train_dataloader: DataLoader,
        val_dataloader: DataLoader | None = None,
        resume_from: Path | None = None,
    ) -> TrainingResult:
        """Run the training loop.

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
            Predictions as MLX array
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
        """Save training checkpoint."""
        ...

    def load_checkpoint(self, path: Path) -> None:
        """Load training checkpoint."""
        ...


class TrainingHook(Protocol):
    """Protocol for training hooks/callbacks."""

    def on_train_begin(self, trainer: Trainer, state: TrainingState) -> None:
        """Called at the beginning of training."""
        ...

    def on_train_end(
        self, trainer: Trainer, state: TrainingState, result: TrainingResult
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


class Callback(Protocol):
    """Protocol for training callbacks (alternative to hooks)."""

    def set_trainer(self, trainer: Trainer) -> None:
        """Set the trainer instance."""
        ...

    @property
    def priority(self) -> int:
        """Priority for callback execution order (lower = earlier)."""
        ...

    # Same methods as TrainingHook
    def on_train_begin(self, trainer: Trainer, state: TrainingState) -> None:
        """Called at the beginning of training."""
        ...

    def on_train_end(
        self, trainer: Trainer, state: TrainingState, result: TrainingResult
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


class MetricsCollector(Protocol):
    """Protocol for metrics collection."""

    def add_metric(self, name: str, value: float, step: int | None = None) -> None:
        """Add a metric value."""
        ...

    def add_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
        """Add multiple metrics."""
        ...

    def get_metric(self, name: str) -> list[tuple[int, float]]:
        """Get metric history."""
        ...

    def get_latest_metrics(self) -> dict[str, float]:
        """Get latest metric values."""
        ...

    def clear(self) -> None:
        """Clear all metrics."""
        ...

    def save(self, path: Path) -> None:
        """Save metrics to file."""
        ...

    def load(self, path: Path) -> None:
        """Load metrics from file."""
        ...


class CheckpointManager(Protocol):
    """Protocol for checkpoint management."""

    def save_checkpoint(
        self,
        model: Model,
        optimizer: Optimizer,
        state: TrainingState,
        metrics: dict[str, float],
        is_best: bool = False,
    ) -> Path:
        """Save a training checkpoint."""
        ...

    def load_checkpoint(
        self,
        path: Path,
        model: Model,
        optimizer: Optimizer,
    ) -> TrainingState:
        """Load a training checkpoint."""
        ...

    def get_best_checkpoint(self) -> Path | None:
        """Get path to best checkpoint."""
        ...

    def get_latest_checkpoint(self) -> Path | None:
        """Get path to latest checkpoint."""
        ...

    def cleanup_old_checkpoints(self, keep_best: int = 1, keep_last: int = 1) -> None:
        """Remove old checkpoints."""
        ...

    @property
    def checkpoint_dir(self) -> Path:
        """Get checkpoint directory."""
        ...


class Command(Protocol):
    """Protocol for training commands (Command pattern)."""
    
    @property
    def name(self) -> str:
        """Command name for logging and debugging."""
        ...
    
    @property
    def requires_grad(self) -> bool:
        """Whether this command requires gradient computation."""
        ...
    
    def can_execute(self, context: "CommandContext") -> bool:
        """Check if command can be executed given current context."""
        ...
    
    def execute(self, context: "CommandContext") -> "CommandResult":
        """Execute the command."""
        ...
    
    def rollback(self, context: "CommandContext") -> None:
        """Rollback command effects if needed."""
        ...


class CommandContext(Protocol):
    """Protocol for command execution context."""
    
    # Core components
    model: Model
    optimizer: Optimizer
    state: TrainingState
    
    # Optional components
    train_dataloader: Optional["DataLoader"]
    val_dataloader: Optional["DataLoader"]
    lr_scheduler: Optional[LRScheduler]
    metrics_collector: Optional[MetricsCollector]
    checkpoint_manager: Optional[CheckpointManager]
    
    # Current execution state
    batch: Optional[Dict[str, Any]]
    batch_idx: int
    outputs: Dict[str, Any]
    gradients: Dict[str, Any]
    loss: Optional[float]
    metrics: Dict[str, float]
    
    # Control flags
    should_accumulate_gradients: bool
    should_update_weights: bool
    is_training: bool
    
    # Configuration
    config: dict[str, Any]


class CommandResult(Protocol):
    """Protocol for command execution results."""
    
    success: bool
    outputs: dict[str, Any]
    error: Exception | None
    metrics: dict[str, float]
    should_continue: bool
    should_skip_remaining: bool


class Pipeline(Protocol):
    """Protocol for training pipelines."""
    
    commands: list[Command]
    middleware: list["Middleware"]
    name: str
    stop_on_error: bool
    
    def execute(self, context: CommandContext) -> CommandResult:
        """Execute the pipeline with all middleware."""
        ...


class Middleware(Protocol):
    """Protocol for pipeline middleware."""
    
    @property
    def name(self) -> str:
        """Middleware name for debugging."""
        ...
    
    @property
    def enabled(self) -> bool:
        """Whether middleware is enabled."""
        ...
    
    def before_pipeline(self, context: CommandContext) -> CommandContext:
        """Called before pipeline execution starts."""
        ...
    
    def after_pipeline(self, context: CommandContext, result: CommandResult) -> CommandResult:
        """Called after pipeline execution completes."""
        ...
    
    def before_command(self, command: Command, context: CommandContext) -> tuple[Command, CommandContext]:
        """Called before each command execution."""
        ...
    
    def after_command(self, command: Command, context: CommandContext, result: CommandResult) -> CommandResult:
        """Called after each command execution."""
        ...
    
    def on_error(self, command: Command, context: CommandContext, error: Exception) -> CommandResult | None:
        """Called when a command raises an error."""
        ...


class TrainingStrategy(Protocol):
    """Protocol for training strategies (Strategy pattern)."""
    
    @property
    def name(self) -> str:
        """Strategy name."""
        ...
    
    @property
    def description(self) -> str:
        """Strategy description."""
        ...
    
    def create_pipeline(self, context: CommandContext) -> Pipeline:
        """Create training pipeline for this strategy."""
        ...
    
    def configure_context(self, context: CommandContext) -> CommandContext:
        """Configure context for this strategy."""
        ...
    
    def validate_requirements(self, context: CommandContext) -> list[str]:
        """Validate that context meets strategy requirements."""
        ...
    
    def get_default_config(self) -> dict[str, Any]:
        """Get default configuration for this strategy."""
        ...


class FrameworkAdapter(Protocol):
    """Protocol for framework-specific adapters."""
    
    @property
    def name(self) -> str:
        """Framework name."""
        ...
    
    @property
    def available(self) -> bool:
        """Whether framework is available."""
        ...
    
    # Tensor operations
    def to_tensor(self, data: Any) -> Any:
        """Convert data to framework tensor."""
        ...
    
    def to_python(self, tensor: Any) -> float | int | list:
        """Convert tensor to Python types."""
        ...
    
    def compute_gradient_norm(self, gradients: dict[str, Any]) -> float:
        """Compute norm of gradients."""
        ...
    
    def clip_gradients_by_norm(self, gradients: dict[str, Any], max_norm: float) -> tuple[dict[str, Any], float]:
        """Clip gradients by norm."""
        ...
    
    def scale_gradients(self, gradients: dict[str, Any], scale: float) -> dict[str, Any]:
        """Scale gradients by factor."""
        ...
    
    def update_model_parameters(self, model: Model, optimizer: Optimizer, gradients: dict[str, Any]) -> None:
        """Update model parameters with optimizer."""
        ...
    
    def get_learning_rate(self, optimizer: Optimizer) -> float:
        """Get current learning rate from optimizer."""
        ...