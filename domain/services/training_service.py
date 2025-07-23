"""Domain service for training logic.

This module contains the pure business logic for training BERT models,
free from any framework dependencies.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, TypeVar, Generic, Callable
from enum import Enum
from datetime import datetime
import math


TArray = TypeVar('TArray')
TOptimizer = TypeVar('TOptimizer')
TScheduler = TypeVar('TScheduler')


class TrainingPhase(Enum):
    """Phases of model training."""
    WARMUP = "warmup"
    TRAINING = "training"
    VALIDATION = "validation"
    EVALUATION = "evaluation"
    COOLDOWN = "cooldown"


class OptimizerType(Enum):
    """Types of optimizers."""
    ADAM = "adam"
    ADAMW = "adamw"
    SGD = "sgd"
    LAMB = "lamb"
    LION = "lion"


class SchedulerType(Enum):
    """Types of learning rate schedulers."""
    CONSTANT = "constant"
    LINEAR = "linear"
    COSINE = "cosine"
    POLYNOMIAL = "polynomial"
    EXPONENTIAL = "exponential"
    WARMUP_CONSTANT = "warmup_constant"
    WARMUP_LINEAR = "warmup_linear"
    WARMUP_COSINE = "warmup_cosine"


@dataclass
class TrainingConfig:
    """Configuration for training process."""
    
    # Basic settings
    num_epochs: int = 3
    batch_size: int = 32
    learning_rate: float = 5e-5
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    
    # Optimizer
    optimizer_type: OptimizerType = OptimizerType.ADAMW
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_epsilon: float = 1e-8
    
    # Scheduler
    scheduler_type: SchedulerType = SchedulerType.WARMUP_LINEAR
    warmup_steps: Optional[int] = None
    warmup_ratio: float = 0.1
    
    # Evaluation
    eval_steps: Optional[int] = None
    eval_strategy: str = "epoch"  # "epoch", "steps", "no"
    save_steps: Optional[int] = None
    save_strategy: str = "epoch"  # "epoch", "steps", "no"
    
    # Early stopping
    early_stopping_patience: Optional[int] = None
    early_stopping_threshold: float = 0.0
    
    # Gradient accumulation
    gradient_accumulation_steps: int = 1
    
    # Mixed precision
    use_mixed_precision: bool = False
    
    # Checkpointing
    save_total_limit: Optional[int] = None
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "eval_loss"
    greater_is_better: bool = False
    
    # Logging
    logging_steps: int = 100
    logging_first_step: bool = True
    
    # Advanced
    label_smoothing_factor: float = 0.0
    gradient_checkpointing: bool = False
    
    def __post_init__(self):
        """Validate and compute derived values."""
        if self.warmup_steps is None and self.warmup_ratio > 0:
            # Will be computed based on total steps
            pass
        elif self.warmup_steps is not None and self.warmup_steps < 0:
            raise ValueError("Warmup steps must be non-negative")
            
        if self.eval_strategy == "steps" and self.eval_steps is None:
            raise ValueError("eval_steps must be specified when eval_strategy='steps'")
            
        if self.save_strategy == "steps" and self.save_steps is None:
            raise ValueError("save_steps must be specified when save_strategy='steps'")
    
    @property
    def effective_batch_size(self) -> int:
        """Compute effective batch size with gradient accumulation."""
        return self.batch_size * self.gradient_accumulation_steps
    
    def compute_total_steps(self, num_training_samples: int) -> int:
        """Compute total training steps."""
        steps_per_epoch = math.ceil(
            num_training_samples / self.effective_batch_size
        )
        return steps_per_epoch * self.num_epochs
    
    def compute_warmup_steps(self, total_steps: int) -> int:
        """Compute warmup steps from ratio if needed."""
        if self.warmup_steps is not None:
            return self.warmup_steps
        return int(total_steps * self.warmup_ratio)


@dataclass
class TrainingState:
    """State of the training process."""
    
    # Progress tracking
    epoch: int = 0
    global_step: int = 0
    total_steps: int = 0
    
    # Performance tracking
    train_loss: float = 0.0
    eval_loss: Optional[float] = None
    best_metric: Optional[float] = None
    best_model_step: Optional[int] = None
    
    # Learning rate
    learning_rate: float = 0.0
    
    # Timing
    start_time: Optional[datetime] = None
    epoch_start_time: Optional[datetime] = None
    
    # Early stopping
    early_stopping_counter: int = 0
    should_stop: bool = False
    
    # Metrics history
    train_history: List[Dict[str, float]] = field(default_factory=list)
    eval_history: List[Dict[str, float]] = field(default_factory=list)
    
    @property
    def current_phase(self) -> TrainingPhase:
        """Determine current training phase."""
        if self.global_step == 0:
            return TrainingPhase.WARMUP
        elif self.should_stop:
            return TrainingPhase.COOLDOWN
        else:
            return TrainingPhase.TRAINING
    
    def update_metrics(self, metrics: Dict[str, float], is_eval: bool = False):
        """Update training metrics."""
        if is_eval:
            self.eval_history.append({
                "step": self.global_step,
                "epoch": self.epoch,
                **metrics
            })
            if "loss" in metrics:
                self.eval_loss = metrics["loss"]
        else:
            self.train_history.append({
                "step": self.global_step,
                "epoch": self.epoch,
                **metrics
            })
            if "loss" in metrics:
                self.train_loss = metrics["loss"]
    
    def check_improvement(
        self, 
        metric_value: float,
        metric_name: str,
        greater_is_better: bool = False
    ) -> bool:
        """Check if metric has improved."""
        if self.best_metric is None:
            self.best_metric = metric_value
            self.best_model_step = self.global_step
            return True
            
        if greater_is_better:
            improved = metric_value > self.best_metric
        else:
            improved = metric_value < self.best_metric
            
        if improved:
            self.best_metric = metric_value
            self.best_model_step = self.global_step
            self.early_stopping_counter = 0
            return True
        else:
            self.early_stopping_counter += 1
            return False


class LearningRateSchedule(ABC):
    """Abstract learning rate schedule."""
    
    def __init__(self, 
                 base_lr: float,
                 warmup_steps: int = 0,
                 total_steps: Optional[int] = None):
        self.base_lr = base_lr
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
    
    @abstractmethod
    def get_lr(self, step: int) -> float:
        """Get learning rate for given step."""
        pass
    
    def get_warmup_lr(self, step: int) -> float:
        """Get learning rate during warmup."""
        if step >= self.warmup_steps:
            return self.base_lr
        return self.base_lr * (step / self.warmup_steps)


class LinearSchedule(LearningRateSchedule):
    """Linear learning rate schedule with warmup."""
    
    def get_lr(self, step: int) -> float:
        if step < self.warmup_steps:
            return self.get_warmup_lr(step)
            
        if self.total_steps is None:
            return self.base_lr
            
        progress = (step - self.warmup_steps) / (self.total_steps - self.warmup_steps)
        return self.base_lr * (1 - progress)


class CosineSchedule(LearningRateSchedule):
    """Cosine learning rate schedule with warmup."""
    
    def get_lr(self, step: int) -> float:
        if step < self.warmup_steps:
            return self.get_warmup_lr(step)
            
        if self.total_steps is None:
            return self.base_lr
            
        progress = (step - self.warmup_steps) / (self.total_steps - self.warmup_steps)
        return self.base_lr * (1 + math.cos(math.pi * progress)) / 2


class TrainingStrategy(ABC):
    """Abstract training strategy."""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
    
    @abstractmethod
    def should_evaluate(self, state: TrainingState) -> bool:
        """Determine if evaluation should run."""
        pass
    
    @abstractmethod
    def should_save(self, state: TrainingState) -> bool:
        """Determine if model should be saved."""
        pass
    
    @abstractmethod
    def should_log(self, state: TrainingState) -> bool:
        """Determine if metrics should be logged."""
        pass
    
    def should_stop_early(self, state: TrainingState) -> bool:
        """Check early stopping criteria."""
        if self.config.early_stopping_patience is None:
            return False
            
        return state.early_stopping_counter >= self.config.early_stopping_patience


class EpochStrategy(TrainingStrategy):
    """Strategy for epoch-based training."""
    
    def should_evaluate(self, state: TrainingState) -> bool:
        return self.config.eval_strategy == "epoch" and state.global_step > 0
    
    def should_save(self, state: TrainingState) -> bool:
        return self.config.save_strategy == "epoch" and state.global_step > 0
    
    def should_log(self, state: TrainingState) -> bool:
        return (state.global_step % self.config.logging_steps == 0 or
                (self.config.logging_first_step and state.global_step == 1))


class StepStrategy(TrainingStrategy):
    """Strategy for step-based training."""
    
    def should_evaluate(self, state: TrainingState) -> bool:
        if self.config.eval_strategy != "steps":
            return False
        return state.global_step % self.config.eval_steps == 0
    
    def should_save(self, state: TrainingState) -> bool:
        if self.config.save_strategy != "steps":
            return False
        return state.global_step % self.config.save_steps == 0
    
    def should_log(self, state: TrainingState) -> bool:
        return (state.global_step % self.config.logging_steps == 0 or
                (self.config.logging_first_step and state.global_step == 1))


@dataclass
class TrainingMetrics:
    """Metrics collected during training."""
    
    # Loss metrics
    loss: float
    gradient_norm: Optional[float] = None
    
    # Performance metrics
    learning_rate: float = 0.0
    epoch: float = 0.0
    
    # Throughput metrics
    samples_per_second: Optional[float] = None
    steps_per_second: Optional[float] = None
    
    # Resource metrics
    memory_used_gb: Optional[float] = None
    
    # Task-specific metrics
    task_metrics: Dict[str, float] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, float]:
        """Convert metrics to dictionary."""
        result = {
            "loss": self.loss,
            "learning_rate": self.learning_rate,
            "epoch": self.epoch,
        }
        
        if self.gradient_norm is not None:
            result["gradient_norm"] = self.gradient_norm
        if self.samples_per_second is not None:
            result["samples_per_second"] = self.samples_per_second
        if self.steps_per_second is not None:
            result["steps_per_second"] = self.steps_per_second
        if self.memory_used_gb is not None:
            result["memory_used_gb"] = self.memory_used_gb
            
        result.update(self.task_metrics)
        return result


class TrainingService(ABC, Generic[TArray, TOptimizer, TScheduler]):
    """Abstract training service defining training logic."""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.state = TrainingState()
        self.strategy = self._create_strategy()
    
    def _create_strategy(self) -> TrainingStrategy:
        """Create appropriate training strategy."""
        if self.config.eval_strategy == "steps" or self.config.save_strategy == "steps":
            return StepStrategy(self.config)
        return EpochStrategy(self.config)
    
    @abstractmethod
    def create_optimizer(self, parameters: Any) -> TOptimizer:
        """Create optimizer instance."""
        pass
    
    @abstractmethod
    def create_scheduler(self, optimizer: TOptimizer) -> TScheduler:
        """Create learning rate scheduler."""
        pass
    
    @abstractmethod
    def training_step(
        self,
        model: Any,
        batch: Dict[str, TArray],
        optimizer: TOptimizer
    ) -> TrainingMetrics:
        """Execute single training step."""
        pass
    
    @abstractmethod
    def evaluation_step(
        self,
        model: Any,
        batch: Dict[str, TArray]
    ) -> Dict[str, float]:
        """Execute single evaluation step."""
        pass
    
    def should_evaluate(self) -> bool:
        """Check if evaluation should run."""
        return self.strategy.should_evaluate(self.state)
    
    def should_save(self) -> bool:
        """Check if model should be saved."""
        return self.strategy.should_save(self.state)
    
    def should_log(self) -> bool:
        """Check if metrics should be logged."""
        return self.strategy.should_log(self.state)
    
    def should_stop(self) -> bool:
        """Check if training should stop."""
        if self.state.should_stop:
            return True
        return self.strategy.should_stop_early(self.state)
    
    def update_state(self, metrics: TrainingMetrics, is_eval: bool = False):
        """Update training state with metrics."""
        self.state.update_metrics(metrics.to_dict(), is_eval)
        
        if not is_eval:
            self.state.global_step += 1
            self.state.learning_rate = metrics.learning_rate