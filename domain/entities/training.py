"""Training-related domain entities."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from enum import Enum
from datetime import datetime


class TrainingPhase(Enum):
    """Training phases."""
    WARMUP = "warmup"
    TRAINING = "training"
    VALIDATION = "validation"
    EVALUATION = "evaluation"


class OptimizerType(Enum):
    """Supported optimizer types."""
    ADAM = "adam"
    ADAMW = "adamw"
    SGD = "sgd"
    LION = "lion"


class SchedulerType(Enum):
    """Learning rate scheduler types."""
    CONSTANT = "constant"
    LINEAR = "linear"
    COSINE = "cosine"
    POLYNOMIAL = "polynomial"
    EXPONENTIAL = "exponential"


@dataclass
class TrainingConfig:
    """Configuration for model training."""
    # Basic settings
    num_epochs: int
    batch_size: int
    learning_rate: float
    max_grad_norm: float = 1.0
    
    # Optimizer settings
    optimizer_type: OptimizerType = OptimizerType.ADAMW
    weight_decay: float = 0.01
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_epsilon: float = 1e-8
    
    # Scheduler settings
    scheduler_type: SchedulerType = SchedulerType.LINEAR
    warmup_steps: int = 0
    warmup_ratio: float = 0.0
    
    # Training behavior
    gradient_accumulation_steps: int = 1
    mixed_precision: bool = False
    compile_model: bool = True
    seed: int = 42
    
    # Evaluation
    eval_steps: int = 500
    save_steps: int = 500
    logging_steps: int = 10
    
    # Early stopping
    early_stopping_patience: Optional[int] = None
    early_stopping_threshold: float = 0.0
    
    @property
    def effective_batch_size(self) -> int:
        """Calculate effective batch size including gradient accumulation."""
        return self.batch_size * self.gradient_accumulation_steps
    
    @property
    def has_warmup(self) -> bool:
        """Check if warmup is configured."""
        return self.warmup_steps > 0 or self.warmup_ratio > 0


@dataclass
class TrainingState:
    """Mutable state during training."""
    epoch: int = 0
    global_step: int = 0
    total_loss: float = 0.0
    best_metric: Optional[float] = None
    best_metric_epoch: Optional[int] = None
    current_learning_rate: float = 0.0
    phase: TrainingPhase = TrainingPhase.TRAINING
    
    # Tracking
    train_loss_history: List[float] = field(default_factory=list)
    eval_loss_history: List[float] = field(default_factory=list)
    learning_rate_history: List[float] = field(default_factory=list)
    
    # Timing
    start_time: Optional[datetime] = None
    last_checkpoint_time: Optional[datetime] = None
    
    def update_step(self, loss: float, learning_rate: float) -> None:
        """Update state after a training step."""
        self.global_step += 1
        self.total_loss += loss
        self.current_learning_rate = learning_rate
        self.train_loss_history.append(loss)
        self.learning_rate_history.append(learning_rate)
    
    def complete_epoch(self) -> None:
        """Mark epoch as complete."""
        self.epoch += 1
        self.total_loss = 0.0
    
    def update_best_metric(self, metric: float) -> bool:
        """Update best metric if improved. Returns True if improved."""
        if self.best_metric is None or metric > self.best_metric:
            self.best_metric = metric
            self.best_metric_epoch = self.epoch
            return True
        return False


@dataclass
class TrainingSession:
    """Represents a complete training session."""
    session_id: str
    config: TrainingConfig
    state: TrainingState = field(default_factory=TrainingState)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Results
    final_metrics: Optional[Dict[str, float]] = None
    checkpoint_paths: List[str] = field(default_factory=list)
    
    @property
    def is_completed(self) -> bool:
        """Check if training has completed."""
        return self.state.epoch >= self.config.num_epochs
    
    @property
    def should_stop_early(self) -> bool:
        """Check if early stopping criteria met."""
        if self.config.early_stopping_patience is None:
            return False
        
        if self.state.best_metric_epoch is None:
            return False
            
        epochs_without_improvement = self.state.epoch - self.state.best_metric_epoch
        return epochs_without_improvement >= self.config.early_stopping_patience
    
    def add_checkpoint(self, path: str) -> None:
        """Record a checkpoint path."""
        self.checkpoint_paths.append(path)
        self.state.last_checkpoint_time = datetime.now()