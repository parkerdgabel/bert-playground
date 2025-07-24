"""Training port for model training operations.

This port defines the interface for training execution, separating
the domain's training logic from infrastructure concerns.
"""

from typing import Protocol, Dict, Any, Optional, Iterator, Tuple
from dataclasses import dataclass
from datetime import datetime

from domain.entities.model import BertModel
from domain.entities.dataset import Dataset
from domain.entities.training import TrainingState, TrainingSession
from domain.value_objects.hyperparameters import Hyperparameters


@dataclass
class TrainingBatch:
    """Represents a batch of training data."""
    input_ids: Any
    attention_mask: Any
    labels: Any
    batch_idx: int
    total_batches: int
    
    @property
    def progress(self) -> float:
        """Get batch progress as percentage."""
        return (self.batch_idx + 1) / self.total_batches * 100


@dataclass
class TrainingStepResult:
    """Result from a single training step."""
    loss: float
    gradients_norm: Optional[float] = None
    learning_rate: float = 0.0
    throughput_samples_per_sec: Optional[float] = None
    memory_used_mb: Optional[float] = None
    additional_metrics: Dict[str, float] = None
    
    def __post_init__(self):
        if self.additional_metrics is None:
            self.additional_metrics = {}


@dataclass
class EvaluationResult:
    """Result from model evaluation."""
    loss: float
    metrics: Dict[str, float]
    predictions: Optional[Any] = None
    num_samples: int = 0
    duration_seconds: float = 0.0


class TrainingExecutor(Protocol):
    """Port for executing model training.
    
    This port abstracts the actual training execution, allowing the domain
    to remain pure while infrastructure handles framework-specific details.
    """
    
    def initialize_training(
        self,
        model: BertModel,
        hyperparameters: Hyperparameters,
        device: Optional[str] = None
    ) -> Tuple[Any, Any, Any]:
        """Initialize training components.
        
        Args:
            model: Domain model to train
            hyperparameters: Training hyperparameters
            device: Optional device specification
            
        Returns:
            Tuple of (compiled_model, optimizer, scheduler)
        """
        ...
    
    def create_data_iterator(
        self,
        dataset: Dataset,
        batch_size: int,
        shuffle: bool = True,
        num_workers: int = 0
    ) -> Iterator[TrainingBatch]:
        """Create iterator for training data.
        
        Args:
            dataset: Domain dataset
            batch_size: Batch size
            shuffle: Whether to shuffle data
            num_workers: Number of data loading workers
            
        Returns:
            Iterator of training batches
        """
        ...
    
    def training_step(
        self,
        model: Any,
        batch: TrainingBatch,
        optimizer: Any,
        accumulation_steps: int = 1,
        clip_grad_norm: Optional[float] = None
    ) -> TrainingStepResult:
        """Execute a single training step.
        
        Args:
            model: Compiled model
            batch: Training batch
            optimizer: Optimizer instance
            accumulation_steps: Gradient accumulation steps
            clip_grad_norm: Optional gradient clipping
            
        Returns:
            Training step results
        """
        ...
    
    def evaluation_step(
        self,
        model: Any,
        batch: TrainingBatch
    ) -> Tuple[float, Dict[str, float]]:
        """Execute a single evaluation step.
        
        Args:
            model: Compiled model
            batch: Evaluation batch
            
        Returns:
            Tuple of (loss, metrics)
        """
        ...
    
    def update_learning_rate(
        self,
        scheduler: Any,
        step: int
    ) -> float:
        """Update and return current learning rate.
        
        Args:
            scheduler: Learning rate scheduler
            step: Current training step
            
        Returns:
            Updated learning rate
        """
        ...
    
    def save_checkpoint(
        self,
        model: Any,
        optimizer: Any,
        scheduler: Any,
        training_state: TrainingState,
        path: str
    ) -> None:
        """Save training checkpoint.
        
        Args:
            model: Model to save
            optimizer: Optimizer state
            scheduler: Scheduler state
            training_state: Training state
            path: Save path
        """
        ...
    
    def load_checkpoint(
        self,
        path: str
    ) -> Tuple[Any, Any, Any, TrainingState]:
        """Load training checkpoint.
        
        Args:
            path: Checkpoint path
            
        Returns:
            Tuple of (model, optimizer, scheduler, training_state)
        """
        ...
    
    def compile_model(
        self,
        model: Any,
        compile_options: Optional[Dict[str, Any]] = None
    ) -> Any:
        """Compile model for optimized execution.
        
        Args:
            model: Model to compile
            compile_options: Optional compilation options
            
        Returns:
            Compiled model
        """
        ...
    
    def get_memory_stats(self) -> Dict[str, float]:
        """Get current memory statistics.
        
        Returns:
            Dictionary with memory stats
        """
        ...
    
    def cleanup(self) -> None:
        """Clean up training resources."""
        ...


class TrainingMonitor(Protocol):
    """Port for monitoring training progress."""
    
    def on_training_start(
        self,
        session: TrainingSession
    ) -> None:
        """Called when training starts."""
        ...
    
    def on_epoch_start(
        self,
        epoch: int,
        total_epochs: int
    ) -> None:
        """Called at the start of each epoch."""
        ...
    
    def on_batch_end(
        self,
        batch_idx: int,
        result: TrainingStepResult,
        moving_avg_loss: float
    ) -> None:
        """Called after each batch."""
        ...
    
    def on_evaluation_end(
        self,
        epoch: int,
        result: EvaluationResult
    ) -> None:
        """Called after evaluation."""
        ...
    
    def on_checkpoint_saved(
        self,
        path: str,
        is_best: bool
    ) -> None:
        """Called when checkpoint is saved."""
        ...
    
    def on_training_end(
        self,
        final_metrics: Dict[str, float]
    ) -> None:
        """Called when training completes."""
        ...