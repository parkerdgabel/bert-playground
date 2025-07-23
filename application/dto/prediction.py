"""Prediction-related Data Transfer Objects."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Dict, Any, List, Union
from datetime import datetime
from enum import Enum


class PredictionFormat(Enum):
    """Supported prediction output formats."""
    CSV = "csv"
    JSON = "json"
    PARQUET = "parquet"
    NUMPY = "numpy"


@dataclass
class PredictionRequestDTO:
    """Request DTO for generating predictions.
    
    This is what external actors provide to generate predictions.
    """
    
    # Model specification
    model_path: Path
    model_type: Optional[str] = None
    
    # Data specification
    data_path: Path
    data_format: Optional[str] = None  # Auto-detect if not specified
    
    # Prediction configuration
    batch_size: int = 32
    output_format: PredictionFormat = PredictionFormat.CSV
    output_path: Optional[Path] = None
    
    # What to include in output
    include_probabilities: bool = True
    include_embeddings: bool = False
    include_attention_weights: bool = False
    include_input_ids: bool = False
    
    # Computation options
    use_gpu: bool = True
    num_workers: int = 0
    prefetch_factor: Optional[int] = None
    
    # Post-processing
    probability_threshold: Optional[float] = None
    top_k_predictions: Optional[int] = None
    
    # For ensemble predictions
    ensemble_method: Optional[str] = None  # "voting", "averaging", etc.
    model_weights: Optional[List[float]] = None
    
    # Tracking
    track_predictions: bool = False
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
            
        if self.probability_threshold is not None:
            if not 0 <= self.probability_threshold <= 1:
                errors.append("probability_threshold must be between 0 and 1")
                
        if self.top_k_predictions is not None and self.top_k_predictions <= 0:
            errors.append("top_k_predictions must be positive")
            
        return errors


@dataclass
class PredictionResponseDTO:
    """Response DTO after predictions are generated.
    
    This is what external actors receive after prediction finishes.
    """
    
    # Status
    success: bool
    error_message: Optional[str] = None
    
    # Output info
    output_path: Optional[Path] = None
    output_format: Optional[PredictionFormat] = None
    num_predictions: int = 0
    
    # Prediction statistics
    prediction_distribution: Optional[Dict[str, int]] = None
    confidence_stats: Optional[Dict[str, float]] = None
    
    # For classification
    class_names: Optional[List[str]] = None
    label_mapping: Optional[Dict[str, int]] = None
    
    # Performance stats
    prediction_time_seconds: float = 0.0
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
    data_format: Optional[str] = None
    
    # Sample predictions (for preview)
    sample_predictions: Optional[List[Dict[str, Any]]] = None
    
    # Tracking info
    run_id: Optional[str] = None
    experiment_id: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "success": self.success,
            "error_message": self.error_message,
            "output_path": str(self.output_path) if self.output_path else None,
            "output_format": self.output_format.value if self.output_format else None,
            "num_predictions": self.num_predictions,
            "prediction_distribution": self.prediction_distribution,
            "confidence_stats": self.confidence_stats,
            "class_names": self.class_names,
            "label_mapping": self.label_mapping,
            "prediction_time_seconds": self.prediction_time_seconds,
            "samples_per_second": self.samples_per_second,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "gpu_model": self.gpu_model,
            "peak_memory_gb": self.peak_memory_gb,
            "model_path": str(self.model_path) if self.model_path else None,
            "model_config": self.model_config,
            "data_path": str(self.data_path) if self.data_path else None,
            "data_format": self.data_format,
            "sample_predictions": self.sample_predictions,
            "run_id": self.run_id,
            "experiment_id": self.experiment_id,
        }
    
    @classmethod
    def from_error(cls, error: Exception) -> "PredictionResponseDTO":
        """Create error response from exception."""
        return cls(
            success=False,
            error_message=str(error)
        )
    
    def get_summary(self) -> str:
        """Get a human-readable summary of prediction results."""
        if not self.success:
            return f"Prediction failed: {self.error_message}"
        
        lines = ["Prediction Results:"]
        lines.append(f"  Total predictions: {self.num_predictions}")
        lines.append(f"  Output saved to: {self.output_path}")
        lines.append(f"  Format: {self.output_format.value if self.output_format else 'unknown'}")
        lines.append(f"  Time taken: {self.prediction_time_seconds:.2f}s")
        lines.append(f"  Speed: {self.samples_per_second:.2f} samples/s")
        
        if self.prediction_distribution:
            lines.append("\nPrediction distribution:")
            for label, count in self.prediction_distribution.items():
                percentage = (count / self.num_predictions) * 100
                lines.append(f"  {label}: {count} ({percentage:.1f}%)")
        
        if self.confidence_stats:
            lines.append("\nConfidence statistics:")
            for stat, value in self.confidence_stats.items():
                lines.append(f"  {stat}: {value:.4f}")
        
        return "\n".join(lines)