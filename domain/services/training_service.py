"""Pure domain training service.

This service contains only business logic for training decisions,
completely free from infrastructure concerns.
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from datetime import datetime, timedelta
from enum import Enum

from domain.entities.model import BertModel
from domain.entities.dataset import Dataset
from domain.entities.training import TrainingSession, TrainingState
from domain.value_objects.hyperparameters import Hyperparameters, LearningRateSchedule
from domain.registry import domain_service, ServiceScope


class TrainingDecision(Enum):
    """Decisions the training service can make."""
    CONTINUE = "continue"
    CHECKPOINT = "checkpoint"
    EVALUATE = "evaluate"
    STOP_EARLY = "stop_early"
    COMPLETE = "complete"


@dataclass
class TrainingProgress:
    """Represents current training progress."""
    current_epoch: int
    total_epochs: int
    current_step: int
    total_steps: int
    current_loss: float
    best_loss: float
    improvement_rate: float
    time_elapsed: timedelta
    estimated_time_remaining: timedelta
    
    @property
    def percentage_complete(self) -> float:
        """Get completion percentage."""
        return (self.current_step / self.total_steps) * 100 if self.total_steps > 0 else 0
    
    @property
    def is_improving(self) -> bool:
        """Check if model is still improving."""
        return self.improvement_rate > 0.001  # 0.1% improvement threshold


@dataclass
class LearningStrategy:
    """Encapsulates learning strategy decisions."""
    should_reduce_lr: bool
    new_learning_rate: Optional[float]
    should_increase_batch_size: bool
    suggested_batch_size: Optional[int]
    reason: str


@dataclass
class CheckpointStrategy:
    """Strategy for checkpointing during training."""
    checkpoint_frequency: str  # "epoch", "steps", "best"
    checkpoint_steps: Optional[int] = None
    keep_best_only: bool = False
    keep_last_n: int = 3
    metric_for_best: str = "loss"
    minimize_metric: bool = True


@domain_service(scope=ServiceScope.SINGLETON)
class TrainingService:
    """Pure domain service for training logic.
    
    This service makes training decisions based on business rules,
    without any knowledge of how training is actually executed.
    """
    
    def __init__(self):
        """Initialize the training service."""
        self.training_history: List[TrainingSession] = []
    
    def create_training_plan(
        self,
        model: BertModel,
        dataset: Dataset,
        hyperparameters: Hyperparameters
    ) -> TrainingSession:
        """Create a training plan based on model and data characteristics.
        
        This method analyzes the model complexity and dataset size to
        create an optimal training plan.
        """
        # Calculate total steps
        steps_per_epoch = self._calculate_steps_per_epoch(
            dataset.size,
            hyperparameters.batch_size,
            hyperparameters.gradient_accumulation_steps
        )
        total_steps = steps_per_epoch * hyperparameters.num_epochs
        
        # Determine checkpoint strategy
        checkpoint_strategy = self._determine_checkpoint_strategy(
            dataset.size,
            hyperparameters.num_epochs,
            steps_per_epoch
        )
        
        # Create training session
        session = TrainingSession(
            id=f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            model_id=model.id,
            dataset_id=dataset.id,
            hyperparameters=hyperparameters,
            checkpoint_strategy=checkpoint_strategy,
            total_steps=total_steps,
            steps_per_epoch=steps_per_epoch,
            created_at=datetime.now()
        )
        
        self.training_history.append(session)
        return session
    
    def make_training_decision(
        self,
        session: TrainingSession,
        current_metrics: Dict[str, float]
    ) -> TrainingDecision:
        """Make a decision about what to do next in training.
        
        This is the core business logic that decides whether to:
        - Continue training
        - Save a checkpoint
        - Run evaluation
        - Stop early
        - Complete training
        """
        state = session.state
        
        # Update metrics
        state.update_metrics(current_metrics)
        
        # Check if training is complete
        if state.current_epoch >= session.hyperparameters.num_epochs:
            return TrainingDecision.COMPLETE
        
        # Check early stopping
        if self._should_stop_early(session):
            return TrainingDecision.STOP_EARLY
        
        # Check if evaluation is needed
        if self._should_evaluate(session):
            return TrainingDecision.EVALUATE
        
        # Check if checkpoint is needed
        if self._should_checkpoint(session):
            return TrainingDecision.CHECKPOINT
        
        return TrainingDecision.CONTINUE
    
    def analyze_training_progress(
        self,
        session: TrainingSession
    ) -> TrainingProgress:
        """Analyze current training progress.
        
        Provides insights into training progress without
        making decisions about what to do next.
        """
        state = session.state
        
        # Calculate improvement rate
        improvement_rate = 0.0
        if len(state.loss_history) > 10:
            recent_losses = state.loss_history[-10:]
            older_losses = state.loss_history[-20:-10]
            avg_recent = sum(recent_losses) / len(recent_losses)
            avg_older = sum(older_losses) / len(older_losses)
            improvement_rate = (avg_older - avg_recent) / avg_older
        
        # Estimate time remaining
        elapsed = datetime.now() - session.created_at
        if state.current_step > 0:
            time_per_step = elapsed / state.current_step
            remaining_steps = session.total_steps - state.current_step
            estimated_remaining = time_per_step * remaining_steps
        else:
            estimated_remaining = timedelta(0)
        
        return TrainingProgress(
            current_epoch=state.current_epoch,
            total_epochs=session.hyperparameters.num_epochs,
            current_step=state.current_step,
            total_steps=session.total_steps,
            current_loss=state.current_loss,
            best_loss=state.best_loss,
            improvement_rate=improvement_rate,
            time_elapsed=elapsed,
            estimated_time_remaining=estimated_remaining
        )
    
    def suggest_learning_adjustment(
        self,
        session: TrainingSession,
        validation_metrics: Optional[Dict[str, float]] = None
    ) -> LearningStrategy:
        """Suggest adjustments to learning strategy.
        
        Based on training dynamics, suggest whether to:
        - Adjust learning rate
        - Change batch size
        - Modify other hyperparameters
        """
        state = session.state
        
        # Check if learning has plateaued
        if self._has_plateaued(state):
            current_lr = self._get_current_learning_rate(session)
            return LearningStrategy(
                should_reduce_lr=True,
                new_learning_rate=current_lr * 0.5,
                should_increase_batch_size=False,
                suggested_batch_size=None,
                reason="Learning has plateaued, reducing learning rate"
            )
        
        # Check if training is unstable
        if self._is_training_unstable(state):
            return LearningStrategy(
                should_reduce_lr=True,
                new_learning_rate=self._get_current_learning_rate(session) * 0.1,
                should_increase_batch_size=True,
                suggested_batch_size=session.hyperparameters.batch_size * 2,
                reason="Training is unstable, reducing LR and increasing batch size"
            )
        
        return LearningStrategy(
            should_reduce_lr=False,
            new_learning_rate=None,
            should_increase_batch_size=False,
            suggested_batch_size=None,
            reason="Training is progressing normally"
        )
    
    def _calculate_steps_per_epoch(
        self,
        dataset_size: int,
        batch_size: int,
        gradient_accumulation_steps: int
    ) -> int:
        """Calculate steps per epoch."""
        effective_batch_size = batch_size * gradient_accumulation_steps
        return (dataset_size + effective_batch_size - 1) // effective_batch_size
    
    def _determine_checkpoint_strategy(
        self,
        dataset_size: int,
        num_epochs: int,
        steps_per_epoch: int
    ) -> CheckpointStrategy:
        """Determine optimal checkpoint strategy."""
        # For small datasets, checkpoint every epoch
        if dataset_size < 10000:
            return CheckpointStrategy(
                checkpoint_frequency="epoch",
                keep_best_only=False,
                keep_last_n=3
            )
        
        # For large datasets with few epochs, checkpoint by steps
        if num_epochs < 5:
            checkpoint_steps = max(100, steps_per_epoch // 4)
            return CheckpointStrategy(
                checkpoint_frequency="steps",
                checkpoint_steps=checkpoint_steps,
                keep_best_only=True,
                keep_last_n=2
            )
        
        # Default: checkpoint on best performance
        return CheckpointStrategy(
            checkpoint_frequency="best",
            keep_best_only=True,
            keep_last_n=1
        )
    
    def _should_stop_early(self, session: TrainingSession) -> bool:
        """Determine if training should stop early."""
        state = session.state
        patience = session.hyperparameters.early_stopping_patience
        
        if patience is None or patience <= 0:
            return False
        
        # Check if we've seen enough epochs
        if state.current_epoch < 5:
            return False
        
        # Check improvement
        return state.epochs_without_improvement >= patience
    
    def _should_evaluate(self, session: TrainingSession) -> bool:
        """Determine if evaluation should run."""
        eval_frequency = session.hyperparameters.evaluation_strategy
        state = session.state
        
        if eval_frequency == "epoch":
            return state.current_step % session.steps_per_epoch == 0
        elif eval_frequency == "steps":
            eval_steps = session.hyperparameters.eval_steps or 500
            return state.current_step % eval_steps == 0
        else:
            return False
    
    def _should_checkpoint(self, session: TrainingSession) -> bool:
        """Determine if checkpoint should be saved."""
        strategy = session.checkpoint_strategy
        state = session.state
        
        if strategy.checkpoint_frequency == "epoch":
            return state.current_step % session.steps_per_epoch == 0
        elif strategy.checkpoint_frequency == "steps":
            return state.current_step % strategy.checkpoint_steps == 0
        elif strategy.checkpoint_frequency == "best":
            return state.is_best_model
        else:
            return False
    
    def _has_plateaued(self, state: TrainingState) -> bool:
        """Check if learning has plateaued."""
        if len(state.loss_history) < 20:
            return False
        
        recent_losses = state.loss_history[-10:]
        avg_loss = sum(recent_losses) / len(recent_losses)
        std_loss = (sum((x - avg_loss) ** 2 for x in recent_losses) / len(recent_losses)) ** 0.5
        
        # Plateaued if standard deviation is very small
        return std_loss < 0.001
    
    def _is_training_unstable(self, state: TrainingState) -> bool:
        """Check if training is unstable."""
        if len(state.loss_history) < 10:
            return False
        
        recent_losses = state.loss_history[-10:]
        
        # Check for NaN or infinite values
        if any(not (0 < loss < float('inf')) for loss in recent_losses):
            return True
        
        # Check for high variance
        avg_loss = sum(recent_losses) / len(recent_losses)
        variance = sum((x - avg_loss) ** 2 for x in recent_losses) / len(recent_losses)
        
        return variance > avg_loss * 0.5  # High variance relative to mean
    
    def _get_current_learning_rate(self, session: TrainingSession) -> float:
        """Calculate current learning rate based on schedule."""
        schedule = session.hyperparameters.lr_schedule
        step = session.state.current_step
        total_steps = session.total_steps
        base_lr = session.hyperparameters.learning_rate
        
        if schedule == LearningRateSchedule.CONSTANT:
            return base_lr
        elif schedule == LearningRateSchedule.LINEAR:
            return base_lr * (1 - step / total_steps)
        elif schedule == LearningRateSchedule.COSINE:
            import math
            return base_lr * (1 + math.cos(math.pi * step / total_steps)) / 2
        else:
            return base_lr