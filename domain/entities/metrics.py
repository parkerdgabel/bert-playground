"""Metrics entities for training and evaluation."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta


@dataclass
class TrainingMetrics:
    """Metrics collected during training."""
    epoch: int
    step: int
    loss: float
    learning_rate: float
    gradient_norm: Optional[float] = None
    
    # Performance metrics
    samples_per_second: Optional[float] = None
    tokens_per_second: Optional[float] = None
    
    # Memory metrics
    memory_used_gb: Optional[float] = None
    memory_peak_gb: Optional[float] = None
    
    # Time metrics
    step_time: Optional[timedelta] = None
    accumulated_time: Optional[timedelta] = None
    
    # Additional metrics
    additional: Dict[str, float] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging."""
        result = {
            "epoch": self.epoch,
            "step": self.step,
            "loss": self.loss,
            "learning_rate": self.learning_rate,
        }
        
        if self.gradient_norm is not None:
            result["gradient_norm"] = self.gradient_norm
        if self.samples_per_second is not None:
            result["samples_per_second"] = self.samples_per_second
        if self.tokens_per_second is not None:
            result["tokens_per_second"] = self.tokens_per_second
        if self.memory_used_gb is not None:
            result["memory_used_gb"] = self.memory_used_gb
        if self.memory_peak_gb is not None:
            result["memory_peak_gb"] = self.memory_peak_gb
        if self.step_time is not None:
            result["step_time_seconds"] = self.step_time.total_seconds()
        if self.accumulated_time is not None:
            result["accumulated_time_seconds"] = self.accumulated_time.total_seconds()
        
        result.update(self.additional)
        return result


@dataclass
class EvaluationMetrics:
    """Metrics from model evaluation."""
    dataset_name: str
    split: str
    
    # Core metrics
    loss: float
    accuracy: Optional[float] = None
    
    # Classification metrics
    precision: Optional[float] = None
    recall: Optional[float] = None
    f1_score: Optional[float] = None
    auc_roc: Optional[float] = None
    auc_pr: Optional[float] = None
    
    # Regression metrics
    mse: Optional[float] = None
    mae: Optional[float] = None
    r2_score: Optional[float] = None
    
    # Per-class metrics (for classification)
    per_class_precision: Optional[Dict[str, float]] = None
    per_class_recall: Optional[Dict[str, float]] = None
    per_class_f1: Optional[Dict[str, float]] = None
    confusion_matrix: Optional[List[List[int]]] = None
    
    # Performance metrics
    inference_time_per_sample: Optional[float] = None
    total_samples: Optional[int] = None
    
    # Metadata
    timestamp: datetime = field(default_factory=datetime.now)
    additional: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def is_classification(self) -> bool:
        """Check if metrics are for classification task."""
        return self.accuracy is not None
    
    @property
    def is_regression(self) -> bool:
        """Check if metrics are for regression task."""
        return self.mse is not None or self.mae is not None
    
    def get_primary_metric(self) -> Tuple[str, float]:
        """Get the primary metric for this evaluation."""
        if self.is_classification:
            if self.f1_score is not None:
                return "f1_score", self.f1_score
            elif self.accuracy is not None:
                return "accuracy", self.accuracy
        elif self.is_regression:
            if self.r2_score is not None:
                return "r2_score", self.r2_score
            elif self.mse is not None:
                return "mse", self.mse
        
        return "loss", self.loss
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging."""
        result = {
            "dataset_name": self.dataset_name,
            "split": self.split,
            "loss": self.loss,
            "timestamp": self.timestamp.isoformat(),
        }
        
        # Add all non-None metrics
        for field_name in [
            "accuracy", "precision", "recall", "f1_score", 
            "auc_roc", "auc_pr", "mse", "mae", "r2_score",
            "inference_time_per_sample", "total_samples"
        ]:
            value = getattr(self, field_name)
            if value is not None:
                result[field_name] = value
        
        # Add complex metrics
        if self.per_class_precision:
            result["per_class_precision"] = self.per_class_precision
        if self.per_class_recall:
            result["per_class_recall"] = self.per_class_recall
        if self.per_class_f1:
            result["per_class_f1"] = self.per_class_f1
        if self.confusion_matrix:
            result["confusion_matrix"] = self.confusion_matrix
        
        result.update(self.additional)
        return result
    
    def summary(self) -> str:
        """Get a summary string of key metrics."""
        metric_name, metric_value = self.get_primary_metric()
        summary = f"{self.dataset_name}/{self.split}: {metric_name}={metric_value:.4f}"
        
        if self.is_classification and self.accuracy is not None:
            summary += f", accuracy={self.accuracy:.4f}"
        
        summary += f", loss={self.loss:.4f}"
        return summary