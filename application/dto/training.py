"""Training-related Data Transfer Objects."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Dict, Any, List
from datetime import datetime


@dataclass
class TrainingRequestDTO:
    """Request DTO for training a model.
    
    This is what external actors (CLI, API) provide to initiate training.
    """
    
    # Model configuration
    model_type: str
    model_config: Dict[str, Any]
    
    # Data configuration
    train_data_path: Path
    val_data_path: Optional[Path] = None
    test_data_path: Optional[Path] = None
    
    # Training configuration
    num_epochs: int = 3
    batch_size: int = 32
    learning_rate: float = 5e-5
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    
    # Optimizer configuration
    optimizer_type: str = "adamw"
    optimizer_params: Dict[str, Any] = field(default_factory=dict)
    
    # Scheduler configuration
    scheduler_type: str = "warmup_linear"
    warmup_steps: Optional[int] = None
    warmup_ratio: float = 0.1
    
    # Evaluation configuration
    eval_strategy: str = "epoch"  # "epoch", "steps", "no"
    eval_steps: Optional[int] = None
    save_strategy: str = "epoch"  # "epoch", "steps", "no"
    save_steps: Optional[int] = None
    
    # Early stopping
    early_stopping_patience: Optional[int] = None
    early_stopping_threshold: float = 0.0
    metric_for_best_model: str = "eval_loss"
    greater_is_better: bool = False
    
    # Advanced options
    gradient_accumulation_steps: int = 1
    use_mixed_precision: bool = False
    gradient_checkpointing: bool = False
    label_smoothing_factor: float = 0.0
    
    # Output configuration
    output_dir: Path = Path("output")
    run_name: Optional[str] = None
    experiment_name: Optional[str] = None
    
    # Logging configuration
    logging_steps: int = 100
    logging_first_step: bool = True
    log_level: str = "info"
    
    # Checkpointing
    save_total_limit: Optional[int] = None
    load_best_model_at_end: bool = True
    resume_from_checkpoint: Optional[Path] = None
    
    # Data loading options
    num_workers: int = 0
    prefetch_factor: Optional[int] = None
    persistent_workers: bool = False
    
    # Tracking configuration
    use_mlflow: bool = True
    mlflow_tracking_uri: Optional[str] = None
    tags: Dict[str, str] = field(default_factory=dict)
    
    def validate(self) -> List[str]:
        """Validate the request and return list of errors."""
        errors = []
        
        if self.num_epochs <= 0:
            errors.append("num_epochs must be positive")
            
        if self.batch_size <= 0:
            errors.append("batch_size must be positive")
            
        if self.learning_rate <= 0:
            errors.append("learning_rate must be positive")
            
        if self.eval_strategy == "steps" and self.eval_steps is None:
            errors.append("eval_steps must be specified when eval_strategy='steps'")
            
        if self.save_strategy == "steps" and self.save_steps is None:
            errors.append("save_steps must be specified when save_strategy='steps'")
            
        if not self.train_data_path.exists():
            errors.append(f"Training data not found: {self.train_data_path}")
            
        if self.val_data_path and not self.val_data_path.exists():
            errors.append(f"Validation data not found: {self.val_data_path}")
            
        return errors


@dataclass
class TrainingResponseDTO:
    """Response DTO after training completes.
    
    This is what external actors receive after training finishes.
    """
    
    # Status
    success: bool
    error_message: Optional[str] = None
    
    # Final metrics
    final_train_loss: float = 0.0
    final_val_loss: Optional[float] = None
    best_val_loss: Optional[float] = None
    best_val_metric: Optional[float] = None
    final_metrics: Dict[str, float] = field(default_factory=dict)
    
    # Training history
    train_history: List[Dict[str, float]] = field(default_factory=list)
    val_history: List[Dict[str, float]] = field(default_factory=list)
    
    # Model artifacts
    final_model_path: Optional[Path] = None
    best_model_path: Optional[Path] = None
    checkpoint_paths: List[Path] = field(default_factory=list)
    
    # Training metadata
    total_epochs: int = 0
    total_steps: int = 0
    total_time_seconds: float = 0.0
    samples_seen: int = 0
    
    # Early stopping info
    early_stopped: bool = False
    stop_reason: Optional[str] = None
    stopped_at_epoch: Optional[int] = None
    stopped_at_step: Optional[int] = None
    
    # Tracking info
    run_id: Optional[str] = None
    experiment_id: Optional[str] = None
    mlflow_run_url: Optional[str] = None
    
    # System info
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    gpu_model: Optional[str] = None
    peak_memory_gb: Optional[float] = None
    
    # Configuration used
    config_used: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "success": self.success,
            "error_message": self.error_message,
            "final_train_loss": self.final_train_loss,
            "final_val_loss": self.final_val_loss,
            "best_val_loss": self.best_val_loss,
            "best_val_metric": self.best_val_metric,
            "final_metrics": self.final_metrics,
            "train_history": self.train_history,
            "val_history": self.val_history,
            "final_model_path": str(self.final_model_path) if self.final_model_path else None,
            "best_model_path": str(self.best_model_path) if self.best_model_path else None,
            "checkpoint_paths": [str(p) for p in self.checkpoint_paths],
            "total_epochs": self.total_epochs,
            "total_steps": self.total_steps,
            "total_time_seconds": self.total_time_seconds,
            "samples_seen": self.samples_seen,
            "early_stopped": self.early_stopped,
            "stop_reason": self.stop_reason,
            "stopped_at_epoch": self.stopped_at_epoch,
            "stopped_at_step": self.stopped_at_step,
            "run_id": self.run_id,
            "experiment_id": self.experiment_id,
            "mlflow_run_url": self.mlflow_run_url,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "gpu_model": self.gpu_model,
            "peak_memory_gb": self.peak_memory_gb,
            "config_used": self.config_used,
        }
    
    @classmethod
    def from_error(cls, error: Exception) -> "TrainingResponseDTO":
        """Create error response from exception."""
        return cls(
            success=False,
            error_message=str(error)
        )