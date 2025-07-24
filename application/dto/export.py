"""Export-related Data Transfer Objects."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Dict, Any, List


@dataclass
class ExportRequestDTO:
    """Request DTO for exporting a model.
    
    Supports various export formats and optimization options.
    """
    
    # Required fields
    model_path: Path
    export_format: str  # "mlx", "onnx", "coreml", "tensorflow", "torchscript"
    output_dir: Path
    
    # Optimization options
    optimize: bool = False
    quantize: bool = False
    prune: bool = False
    optimize_for_inference: bool = False
    
    # Optimization parameters
    quantization_bits: Optional[int] = None  # 8, 4, etc.
    pruning_ratio: Optional[float] = None  # 0.0-1.0
    
    # Export configuration
    export_batch_size: Optional[int] = None
    export_sequence_length: Optional[int] = None
    dynamic_axes: Optional[Dict[str, Dict[int, str]]] = None
    
    # Format-specific options
    onnx_opset_version: Optional[int] = None
    coreml_minimum_deployment_target: Optional[str] = None
    
    # Deployment package options
    create_deployment_package: bool = False
    include_dockerfile: bool = False
    include_serving_code: bool = True
    
    # Validation options
    test_inference: bool = True
    test_data_path: Optional[Path] = None
    
    def validate(self) -> List[str]:
        """Validate the export request."""
        errors = []
        
        # Check model exists
        if not self.model_path.exists():
            errors.append(f"Model not found: {self.model_path}")
        
        # Check valid export format
        valid_formats = ["mlx", "onnx", "coreml", "tensorflow", "torchscript"]
        if self.export_format not in valid_formats:
            errors.append(f"Invalid export format: {self.export_format}. Must be one of {valid_formats}")
        
        # Check quantization bits
        if self.quantize and self.quantization_bits:
            if self.quantization_bits not in [4, 8, 16]:
                errors.append(f"Invalid quantization bits: {self.quantization_bits}. Must be 4, 8, or 16")
        
        # Check pruning ratio
        if self.prune and self.pruning_ratio:
            if not 0.0 <= self.pruning_ratio <= 1.0:
                errors.append(f"Invalid pruning ratio: {self.pruning_ratio}. Must be between 0.0 and 1.0")
        
        # Format-specific validation
        if self.export_format == "onnx" and self.onnx_opset_version:
            if self.onnx_opset_version < 9:
                errors.append(f"ONNX opset version must be >= 9, got {self.onnx_opset_version}")
        
        return errors


@dataclass
class ExportResponseDTO:
    """Response DTO after model export."""
    
    # Status
    success: bool
    error_message: Optional[str] = None
    
    # Export results
    export_paths: Dict[str, Path] = field(default_factory=dict)
    export_format: Optional[str] = None
    
    # Model info
    model_size_mb: float = 0.0
    original_size_mb: Optional[float] = None
    compression_ratio: Optional[float] = None
    
    # Optimization results
    optimization_applied: bool = False
    quantization_applied: bool = False
    pruning_applied: bool = False
    
    # Validation results
    validation_results: Dict[str, Any] = field(default_factory=dict)
    inference_test_passed: bool = False
    
    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    export_time_seconds: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "success": self.success,
            "error_message": self.error_message,
            "export_paths": {k: str(v) for k, v in self.export_paths.items()},
            "export_format": self.export_format,
            "model_size_mb": self.model_size_mb,
            "original_size_mb": self.original_size_mb,
            "compression_ratio": self.compression_ratio,
            "optimization_applied": self.optimization_applied,
            "quantization_applied": self.quantization_applied,
            "pruning_applied": self.pruning_applied,
            "validation_results": self.validation_results,
            "inference_test_passed": self.inference_test_passed,
            "metadata": self.metadata,
            "export_time_seconds": self.export_time_seconds
        }
    
    @classmethod
    def from_error(cls, error: Exception) -> "ExportResponseDTO":
        """Create error response from exception."""
        return cls(
            success=False,
            error_message=str(error)
        )