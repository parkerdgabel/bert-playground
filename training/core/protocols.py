"""
Core protocols for the training module following the same pattern as data/core/interfaces.py.
These protocols define the contracts that all training components must follow.
"""

from typing import Protocol, Dict, Any, List, Optional, Iterator, Tuple, Callable
from dataclasses import dataclass
from abc import ABC, abstractmethod
from pathlib import Path
import mlx.core as mx
import mlx.nn as nn


class Model(Protocol):
    """Protocol for models compatible with the training system."""
    
    def __call__(self, inputs: Dict[str, mx.array]) -> Dict[str, mx.array]:
        """Forward pass of the model."""
        ...
    
    def parameters(self) -> Dict[str, mx.array]:
        """Get model parameters."""
        ...
    
    def save_pretrained(self, path: Path) -> None:
        """Save model to disk."""
        ...
    
    @classmethod
    def load_pretrained(cls, path: Path) -> "Model":
        """Load model from disk."""
        ...
    
    @property
    def config(self) -> Optional[Any]:
        """Model configuration."""
        ...


class DataLoader(Protocol):
    """Protocol for data loaders compatible with the training system."""
    
    def __iter__(self) -> Iterator[Dict[str, mx.array]]:
        """Iterate over batches."""
        ...
    
    def __len__(self) -> int:
        """Number of batches."""
        ...
    
    @property
    def batch_size(self) -> int:
        """Batch size."""
        ...
    
    @property
    def num_samples(self) -> int:
        """Total number of samples."""
        ...


class Optimizer(Protocol):
    """Protocol for optimizers."""
    
    def update(self, model: nn.Module, gradients: Dict[str, mx.array]) -> None:
        """Update model parameters with gradients."""
        ...
    
    @property
    def learning_rate(self) -> float:
        """Current learning rate."""
        ...
    
    @property
    def state(self) -> Dict[str, Any]:
        """Optimizer state."""
        ...


class LRScheduler(Protocol):
    """Protocol for learning rate schedulers."""
    
    def step(self, metrics: Optional[Dict[str, float]] = None) -> float:
        """Update learning rate and return new value."""
        ...
    
    @property
    def current_lr(self) -> float:
        """Current learning rate."""
        ...
    
    def state_dict(self) -> Dict[str, Any]:
        """Get scheduler state."""
        ...
    
    def load_state_dict(self, state: Dict[str, Any]) -> None:
        """Load scheduler state."""
        ...


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
    best_val_loss: float = float('inf')
    best_val_metric: float = 0.0
    metrics: Dict[str, float] = None
    
    # Training history
    train_history: List[Dict[str, float]] = None
    val_history: List[Dict[str, float]] = None
    
    # State flags
    should_stop: bool = False
    improvement_streak: int = 0
    no_improvement_count: int = 0
    
    # Timing
    epoch_start_time: float = 0.0
    training_start_time: float = 0.0
    
    def __post_init__(self):
        if self.metrics is None:
            self.metrics = {}
        if self.train_history is None:
            self.train_history = []
        if self.val_history is None:
            self.val_history = []
    
    def to_dict(self) -> Dict[str, Any]:
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
    def from_dict(cls, data: Dict[str, Any]) -> "TrainingState":
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
    final_metrics: Dict[str, float]
    
    # Training history
    train_history: List[Dict[str, float]]
    val_history: List[Dict[str, float]]
    
    # Model paths
    final_model_path: Optional[Path] = None
    best_model_path: Optional[Path] = None
    
    # Metadata
    total_epochs: int = 0
    total_steps: int = 0
    total_time: float = 0.0
    early_stopped: bool = False
    stop_reason: Optional[str] = None
    
    # MLflow tracking
    mlflow_run_id: Optional[str] = None
    mlflow_experiment_id: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
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
            "mlflow_run_id": self.mlflow_run_id,
            "mlflow_experiment_id": self.mlflow_experiment_id,
        }


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
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        ...
    
    @classmethod
    def from_dict(cls, config: Dict[str, Any]) -> "TrainerConfig":
        """Create config from dictionary."""
        ...
    
    def validate(self) -> List[str]:
        """Validate configuration and return list of errors."""
        ...


class Trainer(Protocol):
    """Protocol for trainer implementations."""
    
    def train(
        self,
        train_dataloader: DataLoader,
        val_dataloader: Optional[DataLoader] = None,
        resume_from: Optional[Path] = None,
    ) -> TrainingResult:
        """
        Run the training loop.
        
        Args:
            train_dataloader: Training data loader
            val_dataloader: Optional validation data loader
            resume_from: Optional checkpoint path to resume from
            
        Returns:
            TrainingResult with final metrics and paths
        """
        ...
    
    def evaluate(self, dataloader: DataLoader) -> Dict[str, float]:
        """
        Evaluate the model on a dataset.
        
        Args:
            dataloader: Data loader for evaluation
            
        Returns:
            Dictionary of metrics
        """
        ...
    
    def predict(self, dataloader: DataLoader) -> mx.array:
        """
        Generate predictions for a dataset.
        
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
    
    def on_train_end(self, trainer: Trainer, state: TrainingState, result: TrainingResult) -> None:
        """Called at the end of training."""
        ...
    
    def on_epoch_begin(self, trainer: Trainer, state: TrainingState) -> None:
        """Called at the beginning of each epoch."""
        ...
    
    def on_epoch_end(self, trainer: Trainer, state: TrainingState) -> None:
        """Called at the end of each epoch."""
        ...
    
    def on_batch_begin(self, trainer: Trainer, state: TrainingState, batch: Dict[str, mx.array]) -> None:
        """Called before processing each batch."""
        ...
    
    def on_batch_end(self, trainer: Trainer, state: TrainingState, loss: float) -> None:
        """Called after processing each batch."""
        ...
    
    def on_evaluate_begin(self, trainer: Trainer, state: TrainingState) -> None:
        """Called before evaluation."""
        ...
    
    def on_evaluate_end(self, trainer: Trainer, state: TrainingState, metrics: Dict[str, float]) -> None:
        """Called after evaluation."""
        ...


class MetricsCollector(Protocol):
    """Protocol for metrics collection."""
    
    def add_metric(self, name: str, value: float, step: Optional[int] = None) -> None:
        """Add a metric value."""
        ...
    
    def add_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        """Add multiple metrics."""
        ...
    
    def get_metric(self, name: str) -> List[Tuple[int, float]]:
        """Get metric history."""
        ...
    
    def get_latest_metrics(self) -> Dict[str, float]:
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
        metrics: Dict[str, float],
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
    
    def get_best_checkpoint(self) -> Optional[Path]:
        """Get path to best checkpoint."""
        ...
    
    def get_latest_checkpoint(self) -> Optional[Path]:
        """Get path to latest checkpoint."""
        ...
    
    def cleanup_old_checkpoints(self, keep_best: int = 1, keep_last: int = 1) -> None:
        """Remove old checkpoints."""
        ...
    
    @property
    def checkpoint_dir(self) -> Path:
        """Get checkpoint directory."""
        ...