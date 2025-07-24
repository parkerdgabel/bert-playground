"""Training orchestrator service for managing the training process.

This service contains the business logic for orchestrating model training,
independent of any specific ML framework.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum
from datetime import datetime

from domain.entities.model import BertModel
from domain.entities.dataset import Dataset, DataBatch
from domain.entities.training import TrainingSession, TrainingState, TrainingConfig
from domain.entities.metrics import TrainingMetrics, EvaluationMetrics
from infrastructure.di import service


class TrainingPhase(Enum):
    """Phases of the training process."""
    INITIALIZATION = "initialization"
    WARMUP = "warmup"
    TRAINING = "training"
    VALIDATION = "validation"
    CHECKPOINT = "checkpoint"
    COMPLETION = "completion"


class StopReason(Enum):
    """Reasons for stopping training."""
    COMPLETED = "completed"
    EARLY_STOPPING = "early_stopping"
    MAX_STEPS = "max_steps"
    USER_STOPPED = "user_stopped"
    ERROR = "error"


@dataclass
class TrainingPlan:
    """Plan for executing training."""
    total_epochs: int
    steps_per_epoch: int
    total_steps: int
    warmup_steps: int
    eval_frequency: int
    checkpoint_frequency: int
    gradient_accumulation_steps: int
    effective_batch_size: int
    
    @property
    def total_optimization_steps(self) -> int:
        """Total number of optimization steps."""
        return self.total_steps // self.gradient_accumulation_steps


@dataclass
class TrainingProgress:
    """Tracks training progress."""
    current_epoch: int = 0
    current_step: int = 0
    global_step: int = 0
    samples_seen: int = 0
    phase: TrainingPhase = TrainingPhase.INITIALIZATION
    
    # Performance tracking
    best_metric: float = float('-inf')
    best_metric_epoch: int = 0
    best_metric_step: int = 0
    no_improvement_count: int = 0
    
    # Timing
    start_time: datetime = field(default_factory=datetime.now)
    epoch_start_time: Optional[datetime] = None
    
    def update_step(self, batch_size: int) -> None:
        """Update progress after a step."""
        self.current_step += 1
        self.global_step += 1
        self.samples_seen += batch_size
    
    def start_epoch(self, epoch: int) -> None:
        """Start a new epoch."""
        self.current_epoch = epoch
        self.current_step = 0
        self.epoch_start_time = datetime.now()
        self.phase = TrainingPhase.TRAINING
    
    def update_best_metric(self, metric: float, greater_is_better: bool = True) -> bool:
        """Update best metric and return if improved."""
        is_better = (
            metric > self.best_metric if greater_is_better 
            else metric < self.best_metric
        )
        
        if is_better:
            self.best_metric = metric
            self.best_metric_epoch = self.current_epoch
            self.best_metric_step = self.global_step
            self.no_improvement_count = 0
            return True
        else:
            self.no_improvement_count += 1
            return False


@service
class TrainingOrchestrator:
    """Orchestrates the training process.
    
    This service manages the high-level training workflow and
    business logic, delegating framework-specific operations
    to ports and adapters.
    """
    
    def create_training_plan(
        self,
        dataset: Dataset,
        config: TrainingConfig
    ) -> TrainingPlan:
        """Create a training plan based on dataset and config.
        
        Args:
            dataset: Training dataset
            config: Training configuration
            
        Returns:
            TrainingPlan with calculated values
        """
        # Calculate steps
        steps_per_epoch = dataset.size // config.batch_size
        if dataset.size % config.batch_size != 0:
            steps_per_epoch += 1
        
        total_steps = steps_per_epoch * config.num_epochs
        
        # Calculate warmup steps
        if config.warmup_ratio > 0:
            warmup_steps = int(total_steps * config.warmup_ratio)
        else:
            warmup_steps = config.warmup_steps or 0
        
        # Calculate frequencies
        eval_frequency = config.eval_steps or steps_per_epoch  # Default: eval each epoch
        checkpoint_frequency = config.save_steps or steps_per_epoch
        
        # Calculate effective batch size
        effective_batch_size = (
            config.batch_size * config.gradient_accumulation_steps
        )
        
        return TrainingPlan(
            total_epochs=config.num_epochs,
            steps_per_epoch=steps_per_epoch,
            total_steps=total_steps,
            warmup_steps=warmup_steps,
            eval_frequency=eval_frequency,
            checkpoint_frequency=checkpoint_frequency,
            gradient_accumulation_steps=config.gradient_accumulation_steps,
            effective_batch_size=effective_batch_size
        )
    
    def should_evaluate(
        self,
        progress: TrainingProgress,
        plan: TrainingPlan,
        force: bool = False
    ) -> bool:
        """Determine if evaluation should run.
        
        Args:
            progress: Current training progress
            plan: Training plan
            force: Force evaluation regardless of schedule
            
        Returns:
            True if should evaluate
        """
        if force:
            return True
        
        # Evaluate at end of epoch
        if progress.current_step == plan.steps_per_epoch:
            return True
        
        # Evaluate based on frequency
        if progress.global_step % plan.eval_frequency == 0:
            return True
        
        return False
    
    def should_checkpoint(
        self,
        progress: TrainingProgress,
        plan: TrainingPlan,
        force: bool = False
    ) -> bool:
        """Determine if checkpoint should be saved.
        
        Args:
            progress: Current training progress
            plan: Training plan
            force: Force checkpoint regardless of schedule
            
        Returns:
            True if should checkpoint
        """
        if force:
            return True
        
        # Checkpoint at end of epoch
        if progress.current_step == plan.steps_per_epoch:
            return True
        
        # Checkpoint based on frequency
        if progress.global_step % plan.checkpoint_frequency == 0:
            return True
        
        return False
    
    def should_stop_early(
        self,
        progress: TrainingProgress,
        config: TrainingConfig
    ) -> Tuple[bool, Optional[StopReason]]:
        """Determine if training should stop early.
        
        Args:
            progress: Current training progress
            config: Training configuration
            
        Returns:
            Tuple of (should_stop, reason)
        """
        # Check early stopping patience
        if config.early_stopping_patience is not None:
            if progress.no_improvement_count >= config.early_stopping_patience:
                return True, StopReason.EARLY_STOPPING
        
        # Check max steps
        if config.max_steps is not None:
            if progress.global_step >= config.max_steps:
                return True, StopReason.MAX_STEPS
        
        return False, None
    
    def calculate_learning_rate(
        self,
        step: int,
        config: TrainingConfig,
        plan: TrainingPlan
    ) -> float:
        """Calculate learning rate for current step.
        
        Args:
            step: Current global step
            config: Training configuration
            plan: Training plan
            
        Returns:
            Learning rate for this step
        """
        base_lr = config.learning_rate
        
        # Warmup phase
        if step < plan.warmup_steps:
            return base_lr * (step / plan.warmup_steps)
        
        # Post-warmup scheduling
        if config.scheduler_type == "linear":
            progress = (step - plan.warmup_steps) / (plan.total_steps - plan.warmup_steps)
            return base_lr * (1.0 - progress)
        elif config.scheduler_type == "cosine":
            import math
            progress = (step - plan.warmup_steps) / (plan.total_steps - plan.warmup_steps)
            return base_lr * 0.5 * (1.0 + math.cos(math.pi * progress))
        else:  # constant
            return base_lr
    
    def calculate_metrics_summary(
        self,
        metrics_history: List[TrainingMetrics]
    ) -> Dict[str, float]:
        """Calculate summary statistics from metrics history.
        
        Args:
            metrics_history: List of training metrics
            
        Returns:
            Summary statistics
        """
        if not metrics_history:
            return {}
        
        # Calculate averages
        avg_loss = sum(m.loss for m in metrics_history) / len(metrics_history)
        
        # Calculate rates
        total_samples = sum(m.samples_per_second or 0 for m in metrics_history)
        avg_samples_per_sec = total_samples / len(metrics_history) if total_samples > 0 else 0
        
        # Get latest values
        latest = metrics_history[-1]
        
        return {
            "avg_loss": avg_loss,
            "final_loss": latest.loss,
            "avg_samples_per_second": avg_samples_per_sec,
            "total_steps": len(metrics_history),
            "final_learning_rate": latest.learning_rate,
        }
    
    def prepare_training_state(
        self,
        model: BertModel,
        progress: TrainingProgress,
        plan: TrainingPlan,
        metrics: Optional[TrainingMetrics] = None
    ) -> TrainingState:
        """Prepare training state for persistence.
        
        Args:
            model: Current model
            progress: Training progress
            plan: Training plan
            metrics: Latest metrics
            
        Returns:
            TrainingState for checkpointing
        """
        state = TrainingState(
            epoch=progress.current_epoch,
            global_step=progress.global_step,
            samples_seen=progress.samples_seen,
            best_metric=progress.best_metric,
            best_metric_epoch=progress.best_metric_epoch,
            no_improvement_count=progress.no_improvement_count,
        )
        
        if metrics:
            state.train_loss = metrics.loss
            state.learning_rate = metrics.learning_rate
            state.grad_norm = metrics.gradient_norm
        
        # Add metadata
        state.metadata = {
            "model_id": model.id,
            "model_name": model.name,
            "total_epochs": plan.total_epochs,
            "total_steps": plan.total_steps,
            "phase": progress.phase.value,
            "timestamp": datetime.now().isoformat(),
        }
        
        return state
    
    def validate_training_setup(
        self,
        model: BertModel,
        dataset: Dataset,
        config: TrainingConfig
    ) -> List[str]:
        """Validate training setup and return any errors.
        
        Args:
            model: Model to train
            dataset: Training dataset  
            config: Training configuration
            
        Returns:
            List of validation errors (empty if valid)
        """
        errors = []
        
        # Validate model
        if not model.architecture:
            errors.append("Model must have architecture defined")
        
        if model.task_head and dataset.num_classes:
            if model.task_head.num_labels != dataset.num_classes:
                errors.append(
                    f"Model labels ({model.task_head.num_labels}) "
                    f"don't match dataset classes ({dataset.num_classes})"
                )
        
        # Validate dataset
        if dataset.is_empty:
            errors.append("Dataset cannot be empty")
        
        if not dataset.is_labeled:
            errors.append("Dataset must have labels for training")
        
        # Validate config
        if config.batch_size > dataset.size:
            errors.append("Batch size cannot exceed dataset size")
        
        if config.warmup_ratio and config.warmup_steps:
            errors.append("Cannot specify both warmup_ratio and warmup_steps")
        
        if config.warmup_ratio and not (0 <= config.warmup_ratio <= 1):
            errors.append("Warmup ratio must be between 0 and 1")
        
        return errors