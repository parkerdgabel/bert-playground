"""Primary model management port - Model management API for external actors.

This port defines the model management interface that external actors use
to save, load, and manage models. It's a driving port in hexagonal architecture.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, runtime_checkable, Optional

from infrastructure.di import port
from domain.protocols.models import Model


@dataclass
class ModelInfo:
    """Information about a saved model that external actors can inspect."""
    
    name: str
    path: Path
    version: Optional[str]
    created_at: str
    size_bytes: int
    metrics: dict[str, float]
    metadata: dict[str, Any]
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "path": str(self.path),
            "version": self.version,
            "created_at": self.created_at,
            "size_bytes": self.size_bytes,
            "metrics": self.metrics,
            "metadata": self.metadata,
        }


@port()
@runtime_checkable
class ModelManager(Protocol):
    """Primary port for model management operations.
    
    External actors use this to manage trained models.
    """
    
    def save_model(
        self,
        model: Model,
        name: str,
        version: Optional[str] = None,
        metadata: dict[str, Any] | None = None,
        metrics: dict[str, float] | None = None,
    ) -> ModelInfo:
        """Save a model.
        
        Args:
            model: Model to save
            name: Model name
            version: Optional version string
            metadata: Optional metadata
            metrics: Optional metrics
            
        Returns:
            Model information
        """
        ...
    
    def load_model(
        self,
        name: str,
        version: Optional[str] = None,
        device: Optional[str] = None,
    ) -> Model:
        """Load a model.
        
        Args:
            name: Model name
            version: Optional version (latest if not specified)
            device: Optional device to load to
            
        Returns:
            Loaded model
        """
        ...
    
    def list_models(
        self,
        name_pattern: Optional[str] = None,
        include_versions: bool = False,
    ) -> list[ModelInfo]:
        """List available models.
        
        Args:
            name_pattern: Optional name pattern to filter
            include_versions: Whether to include all versions
            
        Returns:
            List of model information
        """
        ...
    
    def delete_model(
        self,
        name: str,
        version: Optional[str] = None,
        force: bool = False,
    ) -> bool:
        """Delete a model.
        
        Args:
            name: Model name
            version: Optional version (all versions if not specified)
            force: Force deletion without confirmation
            
        Returns:
            True if deleted
        """
        ...
    
    def get_model_info(
        self,
        name: str,
        version: Optional[str] = None,
    ) -> Optional[ModelInfo]:
        """Get information about a model.
        
        Args:
            name: Model name
            version: Optional version
            
        Returns:
            Model information or None if not found
        """
        ...
    
    def export_model(
        self,
        name: str,
        export_path: Path,
        format: str = "onnx",
        version: Optional[str] = None,
    ) -> Path:
        """Export a model to a different format.
        
        Args:
            name: Model name
            export_path: Path to export to
            format: Export format (onnx, coreml, etc.)
            version: Optional version
            
        Returns:
            Path to exported model
        """
        ...
    
    def compare_models(
        self,
        model1_name: str,
        model2_name: str,
        metrics: list[str] | None = None,
    ) -> dict[str, Any]:
        """Compare two models.
        
        Args:
            model1_name: First model name
            model2_name: Second model name
            metrics: Optional metrics to compare
            
        Returns:
            Comparison results
        """
        ...


# Convenience functions for external actors
def save_model(
    model: Model,
    path: Path,
    metadata: dict[str, Any] | None = None,
) -> None:
    """Save a model to disk.
    
    This is a simple convenience function for external actors
    who just want to save a model without full management.
    
    Args:
        model: Model to save
        path: Save path
        metadata: Optional metadata
    """
    # This would be implemented by the application core
    raise NotImplementedError("To be implemented by application core")


def load_model(
    path: Path,
    device: Optional[str] = None,
) -> Model:
    """Load a model from disk.
    
    This is a simple convenience function for external actors
    who just want to load a model without full management.
    
    Args:
        path: Model path
        device: Optional device to load to
        
    Returns:
        Loaded model
    """
    # This would be implemented by the application core
    raise NotImplementedError("To be implemented by application core")


def list_models(
    directory: Optional[Path] = None,
) -> list[str]:
    """List available models in a directory.
    
    Args:
        directory: Directory to search (default: current)
        
    Returns:
        List of model names
    """
    # This would be implemented by the application core
    raise NotImplementedError("To be implemented by application core")


def delete_model(
    path: Path,
    force: bool = False,
) -> bool:
    """Delete a model from disk.
    
    Args:
        path: Model path
        force: Force deletion
        
    Returns:
        True if deleted
    """
    # This would be implemented by the application core
    raise NotImplementedError("To be implemented by application core")