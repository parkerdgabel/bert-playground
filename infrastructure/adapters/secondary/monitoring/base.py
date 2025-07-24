"""Base monitoring adapter with common functionality."""

from typing import Dict, Any, Optional, List
from abc import ABC
from application.ports.secondary.monitoring import MonitoringService
from domain.entities.metrics import TrainingMetrics, EvaluationMetrics
from domain.entities.training import TrainingSession


class BaseMonitoringAdapter(MonitoringService, ABC):
    """Base class for monitoring adapters with common functionality."""
    
    def __init__(self):
        """Initialize base monitoring adapter."""
        self._active_run_id: Optional[str] = None
        self._run_metrics: Dict[str, List[Dict[str, Any]]] = {}
        self._run_params: Dict[str, Any] = {}
        self._run_artifacts: List[str] = []
    
    def log_training_metrics(self, metrics: TrainingMetrics) -> None:
        """Log training metrics.
        
        Args:
            metrics: Training metrics object
        """
        self.log_metrics(
            metrics.to_dict(),
            step=metrics.step,
            epoch=metrics.epoch,
        )
    
    def log_evaluation_metrics(self, metrics: EvaluationMetrics) -> None:
        """Log evaluation metrics.
        
        Args:
            metrics: Evaluation metrics object
        """
        metric_dict = metrics.to_dict()
        
        # Log main metrics
        self.log_metrics({
            f"eval/{k}": v 
            for k, v in metric_dict.items() 
            if isinstance(v, (int, float))
        })
        
        # Log per-class metrics separately if available
        if metrics.per_class_precision:
            for class_name, value in metrics.per_class_precision.items():
                self.log_metrics({f"eval/precision_{class_name}": value})
        
        if metrics.per_class_recall:
            for class_name, value in metrics.per_class_recall.items():
                self.log_metrics({f"eval/recall_{class_name}": value})
        
        if metrics.per_class_f1:
            for class_name, value in metrics.per_class_f1.items():
                self.log_metrics({f"eval/f1_{class_name}": value})
    
    def log_training_session(self, session: TrainingSession) -> None:
        """Log complete training session information.
        
        Args:
            session: Training session object
        """
        # Log hyperparameters
        config_dict = {
            "num_epochs": session.config.num_epochs,
            "batch_size": session.config.batch_size,
            "learning_rate": session.config.learning_rate,
            "optimizer": session.config.optimizer_type.value,
            "scheduler": session.config.scheduler_type.value,
            "max_grad_norm": session.config.max_grad_norm,
            "weight_decay": session.config.weight_decay,
            "warmup_steps": session.config.warmup_steps,
            "seed": session.config.seed,
        }
        self.log_hyperparameters(config_dict)
        
        # Log session metadata
        self.log_message(
            f"Training session {session.session_id} completed",
            context={
                "session_id": session.session_id,
                "total_epochs": session.state.epoch,
                "total_steps": session.state.global_step,
                "best_metric": session.state.best_metric,
                "best_metric_epoch": session.state.best_metric_epoch,
                **session.metadata
            }
        )
        
        # Log final metrics if available
        if session.final_metrics:
            self.log_metrics(session.final_metrics)
        
        # Log checkpoint paths as artifacts metadata
        for checkpoint_path in session.checkpoint_paths:
            self.log_message(
                f"Checkpoint saved: {checkpoint_path}",
                level="INFO",
                context={"checkpoint_path": checkpoint_path}
            )


class BaseProgressBar(object):
    """Base progress bar implementation."""
    
    def __init__(self, total: int, description: str):
        """Initialize progress bar.
        
        Args:
            total: Total number of items
            description: Progress bar description
        """
        self.total = total
        self.description = description
        self.current = 0
        self.postfix = {}
    
    def update(self, n: int = 1) -> None:
        """Update progress.
        
        Args:
            n: Number of items completed
        """
        self.current = min(self.current + n, self.total)
    
    def set_description(self, description: str) -> None:
        """Update description.
        
        Args:
            description: New description
        """
        self.description = description
    
    def set_postfix(self, **kwargs: Any) -> None:
        """Set postfix values.
        
        Args:
            **kwargs: Key-value pairs to display
        """
        self.postfix.update(kwargs)
    
    def close(self) -> None:
        """Close progress bar."""
        pass