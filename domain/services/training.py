"""Model training service - pure business logic.

This service contains only the business logic for training,
without any dependencies on external systems or frameworks.
"""

from typing import Optional, Dict, Any, List, Callable
from dataclasses import dataclass
from domain.entities.model import BertModel
from domain.entities.training import TrainingSession, TrainingState, TrainingConfig
from domain.entities.dataset import Dataset, DataBatch
from domain.entities.metrics import TrainingMetrics, EvaluationMetrics
from domain.exceptions import TrainingError, ModelNotInitializedError
from infrastructure.di import service


@service
class ModelTrainingService:
    """Service for orchestrating model training logic.
    
    This service contains pure business logic for training
    without any framework-specific implementation details.
    It defines WHAT should happen during training, not HOW.
    """
    
    def validate_training_setup(
        self,
        model: BertModel,
        config: TrainingConfig,
        dataset: Dataset,
    ) -> List[str]:
        """Validate that training can proceed.
        
        Args:
            model: Model to train
            config: Training configuration
            dataset: Training dataset
            
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
        
        if not dataset.is_labeled and model.task_head:
            errors.append("Dataset must have labels for supervised training")
        
        # Validate config
        if config.batch_size > dataset.size:
            errors.append("Batch size cannot exceed dataset size")
        
        if config.num_epochs <= 0:
            errors.append("Number of epochs must be positive")
        
        return errors
    
    def create_training_session(
        self,
        model: BertModel,
        config: TrainingConfig,
        session_id: Optional[str] = None,
    ) -> TrainingSession:
        """Create a new training session.
        
        Args:
            model: Model to train
            config: Training configuration
            session_id: Optional session identifier
            
        Returns:
            New training session
        """
        import uuid
        from datetime import datetime
        
        if session_id is None:
            session_id = str(uuid.uuid4())
        
        state = TrainingState(
            epoch=0,
            global_step=0,
            best_metric=float('inf') if config.lower_is_better else float('-inf'),
            metadata={
                "model_id": model.id,
                "model_name": model.name,
                "start_time": datetime.now().isoformat(),
            }
        )
        
        return TrainingSession(
            session_id=session_id,
            config=config,
            state=state,
            metadata={
                "model_architecture": model.architecture.model_type.value,
                "task_type": model.task_type.value if model.task_type else None,
            }
        )
    
    def should_stop_early(
        self,
        state: TrainingState,
        config: TrainingConfig,
    ) -> bool:
        """Determine if training should stop early.
        
        Args:
            state: Current training state
            config: Training configuration
            
        Returns:
            True if training should stop
        """
        # Check if already flagged to stop
        if state.should_stop:
            return True
        
        # Check early stopping patience
        if config.early_stopping_patience:
            if state.no_improvement_count >= config.early_stopping_patience:
                return True
        
        # Check for NaN loss
        if state.train_loss and (
            state.train_loss != state.train_loss or  # NaN check
            state.train_loss == float('inf')
        ):
            return True
        
        return False
    
    def update_training_state(
        self,
        state: TrainingState,
        metrics: TrainingMetrics,
        is_best: bool = False,
    ) -> TrainingState:
        """Update training state with new metrics.
        
        Args:
            state: Current state
            metrics: New metrics
            is_best: Whether this is the best result so far
            
        Returns:
            Updated state
        """
        # Update basic metrics
        state.epoch = metrics.epoch
        state.global_step = metrics.step
        state.train_loss = metrics.loss
        state.learning_rate = metrics.learning_rate
        
        if metrics.gradient_norm is not None:
            state.grad_norm = metrics.gradient_norm
        
        # Update history
        state.train_history.append({
            "epoch": metrics.epoch,
            "step": metrics.step,
            "loss": metrics.loss,
            "learning_rate": metrics.learning_rate,
            **metrics.additional,
        })
        
        # Update best metric tracking
        if is_best:
            state.best_val_loss = state.val_loss
            state.best_val_metric = state.metrics.get("primary_metric", state.val_loss)
            state.no_improvement_count = 0
        else:
            state.no_improvement_count += 1
        
        return state
    
    def calculate_learning_rate(
        self,
        base_lr: float,
        step: int,
        total_steps: int,
        warmup_steps: int,
        schedule_type: str = "linear",
    ) -> float:
        """Calculate learning rate for current step.
        
        Args:
            base_lr: Base learning rate
            step: Current step
            total_steps: Total training steps
            warmup_steps: Warmup steps
            schedule_type: Type of schedule
            
        Returns:
            Learning rate for this step
        """
        # Warmup phase
        if step < warmup_steps:
            return base_lr * (step / warmup_steps)
        
        # Post-warmup scheduling
        progress = (step - warmup_steps) / (total_steps - warmup_steps)
        
        if schedule_type == "linear":
            return base_lr * (1.0 - progress)
        elif schedule_type == "cosine":
            import math
            return base_lr * 0.5 * (1.0 + math.cos(math.pi * progress))
        elif schedule_type == "constant":
            return base_lr
        else:
            return base_lr
    
    def prepare_checkpoint_data(
        self,
        model: BertModel,
        session: TrainingSession,
        metrics: Optional[Dict[str, float]] = None,
    ) -> Dict[str, Any]:
        """Prepare data for checkpointing.
        
        Args:
            model: Current model
            session: Training session
            metrics: Optional current metrics
            
        Returns:
            Checkpoint data dictionary
        """
        from datetime import datetime
        
        checkpoint_data = {
            "model_id": model.id,
            "model_name": model.name,
            "model_config": model.get_config(),
            "session_id": session.session_id,
            "training_state": {
                "epoch": session.state.epoch,
                "global_step": session.state.global_step,
                "best_metric": session.state.best_val_metric,
                "no_improvement_count": session.state.no_improvement_count,
            },
            "training_config": session.config.to_dict(),
            "timestamp": datetime.now().isoformat(),
        }
        
        if metrics:
            checkpoint_data["metrics"] = metrics
        
        return checkpoint_data
    
    def calculate_gradient_accumulation_steps(
        self,
        desired_batch_size: int,
        max_batch_size: int,
    ) -> int:
        """Calculate gradient accumulation steps needed.
        
        Args:
            desired_batch_size: Desired effective batch size
            max_batch_size: Maximum batch size that fits in memory
            
        Returns:
            Number of gradient accumulation steps
        """
        if desired_batch_size <= max_batch_size:
            return 1
        
        steps = desired_batch_size // max_batch_size
        if desired_batch_size % max_batch_size != 0:
            steps += 1
        
        return steps