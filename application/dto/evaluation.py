"""Evaluation-related Data Transfer Objects."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Dict, Any, List
from datetime import datetime


@dataclass
class EvaluationRequestDTO:
    """Request DTO for evaluating a model.
    
    This is what external actors provide to evaluate a trained model.
    """
    
    # Model specification
    model_path: Path
    model_type: Optional[str] = None
    
    # Data specification
    data_path: Path
    data_split: str = "test"  # "train", "val", "test", or custom
    
    # Evaluation configuration
    batch_size: int = 32
    metrics: List[str] = field(default_factory=lambda: ["accuracy", "loss"])
    
    # Output configuration
    output_dir: Optional[Path] = None
    save_predictions: bool = False
    save_confusion_matrix: bool = False
    save_per_class_metrics: bool = False
    
    # Computation options
    use_gpu: bool = True
    num_workers: int = 0
    prefetch_factor: Optional[int] = None
    
    # Analysis options
    error_analysis: bool = False
    confidence_analysis: bool = False
    feature_importance: bool = False
    
    # Tracking
    use_mlflow: bool = True
    experiment_name: Optional[str] = None
    run_name: Optional[str] = None
    
    def validate(self) -> List[str]:
        """Validate the request and return list of errors."""
        errors = []
        
        if not self.model_path.exists():
            errors.append(f"Model not found: {self.model_path}")
            
        if not self.data_path.exists():
            errors.append(f"Data not found: {self.data_path}")
            
        if self.batch_size <= 0:
            errors.append("batch_size must be positive")
            
        if not self.metrics:
            errors.append("At least one metric must be specified")
            
        return errors


@dataclass
class EvaluationResponseDTO:
    """Response DTO after evaluation completes.
    
    This is what external actors receive after evaluation finishes.
    """
    
    # Status
    success: bool
    error_message: Optional[str] = None
    
    # Core metrics
    metrics: Dict[str, float] = field(default_factory=dict)
    
    # Per-class metrics (for classification)
    per_class_metrics: Optional[Dict[str, Dict[str, float]]] = None
    
    # Confusion matrix (for classification)
    confusion_matrix: Optional[List[List[int]]] = None
    class_names: Optional[List[str]] = None
    
    # Predictions
    predictions_path: Optional[Path] = None
    num_samples_evaluated: int = 0
    
    # Error analysis
    error_samples: Optional[List[Dict[str, Any]]] = None
    error_distribution: Optional[Dict[str, int]] = None
    
    # Confidence analysis
    confidence_distribution: Optional[Dict[str, float]] = None
    calibration_metrics: Optional[Dict[str, float]] = None
    
    # Performance stats
    evaluation_time_seconds: float = 0.0
    samples_per_second: float = 0.0
    
    # System info
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    gpu_model: Optional[str] = None
    peak_memory_gb: Optional[float] = None
    
    # Model info
    model_path: Optional[Path] = None
    model_config: Optional[Dict[str, Any]] = None
    
    # Data info
    data_path: Optional[Path] = None
    data_split: Optional[str] = None
    
    # Tracking info
    run_id: Optional[str] = None
    experiment_id: Optional[str] = None
    mlflow_run_url: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "success": self.success,
            "error_message": self.error_message,
            "metrics": self.metrics,
            "per_class_metrics": self.per_class_metrics,
            "confusion_matrix": self.confusion_matrix,
            "class_names": self.class_names,
            "predictions_path": str(self.predictions_path) if self.predictions_path else None,
            "num_samples_evaluated": self.num_samples_evaluated,
            "error_samples": self.error_samples,
            "error_distribution": self.error_distribution,
            "confidence_distribution": self.confidence_distribution,
            "calibration_metrics": self.calibration_metrics,
            "evaluation_time_seconds": self.evaluation_time_seconds,
            "samples_per_second": self.samples_per_second,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "gpu_model": self.gpu_model,
            "peak_memory_gb": self.peak_memory_gb,
            "model_path": str(self.model_path) if self.model_path else None,
            "model_config": self.model_config,
            "data_path": str(self.data_path) if self.data_path else None,
            "data_split": self.data_split,
            "run_id": self.run_id,
            "experiment_id": self.experiment_id,
            "mlflow_run_url": self.mlflow_run_url,
        }
    
    @classmethod
    def from_error(cls, error: Exception) -> "EvaluationResponseDTO":
        """Create error response from exception."""
        return cls(
            success=False,
            error_message=str(error)
        )
    
    def get_summary(self) -> str:
        """Get a human-readable summary of evaluation results."""
        if not self.success:
            return f"Evaluation failed: {self.error_message}"
        
        lines = ["Evaluation Results:"]
        lines.append(f"  Samples evaluated: {self.num_samples_evaluated}")
        lines.append(f"  Time taken: {self.evaluation_time_seconds:.2f}s")
        lines.append(f"  Speed: {self.samples_per_second:.2f} samples/s")
        
        lines.append("\nMetrics:")
        for metric, value in self.metrics.items():
            lines.append(f"  {metric}: {value:.4f}")
        
        if self.per_class_metrics:
            lines.append("\nPer-class metrics available")
            
        if self.confusion_matrix:
            lines.append("Confusion matrix available")
            
        return "\n".join(lines)