"""Storage and checkpoint ports for persistence operations."""

from typing import Protocol, Dict, Any, Optional, List
from domain.entities.model import BertModel
from domain.entities.training import TrainingSession, TrainingState


class StoragePort(Protocol):
    """Port for general storage operations."""
    
    def save(
        self,
        data: Any,
        path: str,
        format: Optional[str] = None,
    ) -> None:
        """Save data to storage.
        
        Args:
            data: Data to save
            path: Storage path
            format: Optional format specification
        """
        ...
    
    def load(
        self,
        path: str,
        format: Optional[str] = None,
    ) -> Any:
        """Load data from storage.
        
        Args:
            path: Storage path
            format: Optional format specification
            
        Returns:
            Loaded data
        """
        ...
    
    def exists(
        self,
        path: str,
    ) -> bool:
        """Check if path exists.
        
        Args:
            path: Path to check
            
        Returns:
            True if exists
        """
        ...
    
    def delete(
        self,
        path: str,
    ) -> None:
        """Delete from storage.
        
        Args:
            path: Path to delete
        """
        ...
    
    def list(
        self,
        path: str,
        pattern: Optional[str] = None,
    ) -> List[str]:
        """List items in storage.
        
        Args:
            path: Directory path
            pattern: Optional filter pattern
            
        Returns:
            List of paths
        """
        ...
    
    def get_size(
        self,
        path: str,
    ) -> int:
        """Get size of stored item.
        
        Args:
            path: Path to item
            
        Returns:
            Size in bytes
        """
        ...
    
    def copy(
        self,
        source: str,
        destination: str,
    ) -> None:
        """Copy item in storage.
        
        Args:
            source: Source path
            destination: Destination path
        """
        ...


class CheckpointPort(Protocol):
    """Port for checkpoint operations."""
    
    def save_checkpoint(
        self,
        model: BertModel,
        training_state: TrainingState,
        optimizer_state: Dict[str, Any],
        path: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Save training checkpoint.
        
        Args:
            model: Model to save
            training_state: Current training state
            optimizer_state: Optimizer state
            path: Checkpoint path
            metadata: Optional metadata
        """
        ...
    
    def load_checkpoint(
        self,
        path: str,
    ) -> Dict[str, Any]:
        """Load training checkpoint.
        
        Args:
            path: Checkpoint path
            
        Returns:
            Dictionary containing:
            - 'model': Loaded model
            - 'training_state': Training state
            - 'optimizer_state': Optimizer state
            - 'metadata': Optional metadata
        """
        ...
    
    def save_model(
        self,
        model: BertModel,
        path: str,
        save_config: bool = True,
        save_tokenizer: bool = False,
    ) -> None:
        """Save model for inference.
        
        Args:
            model: Model to save
            path: Save path
            save_config: Whether to save configuration
            save_tokenizer: Whether to save tokenizer
        """
        ...
    
    def load_model(
        self,
        path: str,
        load_config: bool = True,
    ) -> BertModel:
        """Load model for inference.
        
        Args:
            path: Model path
            load_config: Whether to load configuration
            
        Returns:
            Loaded model
        """
        ...
    
    def list_checkpoints(
        self,
        directory: str,
    ) -> List[Dict[str, Any]]:
        """List available checkpoints.
        
        Args:
            directory: Directory to search
            
        Returns:
            List of checkpoint info dictionaries
        """
        ...
    
    def get_best_checkpoint(
        self,
        directory: str,
        metric: str = "loss",
        mode: str = "min",
    ) -> Optional[str]:
        """Get best checkpoint based on metric.
        
        Args:
            directory: Directory to search
            metric: Metric to compare
            mode: 'min' or 'max'
            
        Returns:
            Path to best checkpoint or None
        """
        ...
    
    def cleanup_checkpoints(
        self,
        directory: str,
        keep_last: int = 3,
        keep_best: bool = True,
    ) -> List[str]:
        """Clean up old checkpoints.
        
        Args:
            directory: Directory to clean
            keep_last: Number of recent checkpoints to keep
            keep_best: Whether to keep best checkpoint
            
        Returns:
            List of deleted checkpoint paths
        """
        ...
    
    def export_model(
        self,
        model: BertModel,
        path: str,
        format: str = "onnx",
        optimize: bool = True,
    ) -> None:
        """Export model to different format.
        
        Args:
            model: Model to export
            path: Export path
            format: Target format (onnx, coreml, etc.)
            optimize: Whether to optimize exported model
        """
        ...