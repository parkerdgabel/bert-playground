"""Base monitoring adapter with common functionality."""

from typing import Dict, Any, Optional, List
from abc import ABC
from application.ports.secondary.monitoring import MonitoringService


class BaseMonitoringAdapter(MonitoringService, ABC):
    """Base class for monitoring adapters with common functionality."""
    
    def __init__(self):
        """Initialize base monitoring adapter."""
        self._active_run_id: Optional[str] = None
        self._run_metrics: Dict[str, List[Dict[str, Any]]] = {}
        self._run_params: Dict[str, Any] = {}
        self._run_artifacts: List[str] = []
    
    def log_training_metrics(self, metrics: Dict[str, Any]) -> None:
        """Log training metrics.
        
        Args:
            metrics: Training metrics as dictionary
        """
        # Extract step and epoch if present
        step = metrics.get('step')
        epoch = metrics.get('epoch')
        
        # Create metrics dict without step/epoch for logging
        metric_dict = {k: v for k, v in metrics.items() 
                      if k not in ['step', 'epoch'] and isinstance(v, (int, float))}
        
        self.log_metrics(metric_dict, step=step, epoch=epoch)
    
    def log_evaluation_metrics(self, metrics: Dict[str, Any]) -> None:
        """Log evaluation metrics.
        
        Args:
            metrics: Evaluation metrics as dictionary
        """
        # Log main metrics
        main_metrics = {
            f"eval/{k}": v 
            for k, v in metrics.items() 
            if isinstance(v, (int, float)) and not k.startswith('per_class_')
        }
        if main_metrics:
            self.log_metrics(main_metrics)
        
        # Log per-class metrics separately if available
        per_class_precision = metrics.get('per_class_precision', {})
        if per_class_precision:
            for class_name, value in per_class_precision.items():
                self.log_metrics({f"eval/precision_{class_name}": value})
        
        per_class_recall = metrics.get('per_class_recall', {})
        if per_class_recall:
            for class_name, value in per_class_recall.items():
                self.log_metrics({f"eval/recall_{class_name}": value})
        
        per_class_f1 = metrics.get('per_class_f1', {})
        if per_class_f1:
            for class_name, value in per_class_f1.items():
                self.log_metrics({f"eval/f1_{class_name}": value})
    
    def log_training_session(self, session: Dict[str, Any]) -> None:
        """Log complete training session information.
        
        Args:
            session: Training session data as dictionary
        """
        # Extract config and create hyperparameters dict
        config = session.get('config', {})
        
        # Handle optimizer_type and scheduler_type which might be enums or dicts
        optimizer_type = config.get('optimizer_type', 'unknown')
        if isinstance(optimizer_type, dict) and 'value' in optimizer_type:
            optimizer_type = optimizer_type['value']
        
        scheduler_type = config.get('scheduler_type', 'unknown')
        if isinstance(scheduler_type, dict) and 'value' in scheduler_type:
            scheduler_type = scheduler_type['value']
        
        config_dict = {
            "num_epochs": config.get('num_epochs'),
            "batch_size": config.get('batch_size'),
            "learning_rate": config.get('learning_rate'),
            "optimizer": optimizer_type,
            "scheduler": scheduler_type,
            "max_grad_norm": config.get('max_grad_norm'),
            "weight_decay": config.get('weight_decay'),
            "warmup_steps": config.get('warmup_steps'),
            "seed": config.get('seed'),
        }
        # Filter out None values
        config_dict = {k: v for k, v in config_dict.items() if v is not None}
        self.log_hyperparameters(config_dict)
        
        # Extract state info
        state = session.get('state', {})
        session_id = session.get('session_id', 'unknown')
        metadata = session.get('metadata', {})
        
        # Log session metadata
        context = {
            "session_id": session_id,
            "total_epochs": state.get('epoch'),
            "total_steps": state.get('global_step'),
            "best_metric": state.get('best_metric'),
            "best_metric_epoch": state.get('best_metric_epoch'),
        }
        # Add metadata
        context.update(metadata)
        # Filter out None values
        context = {k: v for k, v in context.items() if v is not None}
        
        self.log_message(
            f"Training session {session_id} completed",
            context=context
        )
        
        # Log final metrics if available
        final_metrics = session.get('final_metrics')
        if final_metrics:
            self.log_metrics(final_metrics)
        
        # Log checkpoint paths as artifacts metadata
        checkpoint_paths = session.get('checkpoint_paths', [])
        for checkpoint_path in checkpoint_paths:
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