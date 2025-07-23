"""Base storage adapter with common functionality."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
import json
import pickle
from pathlib import Path
from datetime import datetime

from domain.ports.storage import StoragePort, CheckpointPort
from domain.entities.model import BertModel
from domain.entities.training import TrainingState


class BaseStorageAdapter(StoragePort, ABC):
    """Base storage adapter with common functionality."""
    
    def __init__(self, base_path: Optional[str] = None):
        """Initialize base storage adapter.
        
        Args:
            base_path: Base path for storage operations
        """
        self.base_path = Path(base_path) if base_path else Path.cwd()
        self._ensure_base_path()
    
    def _ensure_base_path(self) -> None:
        """Ensure base path exists."""
        self.base_path.mkdir(parents=True, exist_ok=True)
    
    def _resolve_path(self, path: str) -> Path:
        """Resolve path relative to base path.
        
        Args:
            path: Path to resolve
            
        Returns:
            Resolved absolute path
        """
        p = Path(path)
        if not p.is_absolute():
            p = self.base_path / p
        return p
    
    def _get_format(self, path: str, format: Optional[str] = None) -> str:
        """Determine format from path or explicit format.
        
        Args:
            path: File path
            format: Explicit format
            
        Returns:
            Format string
        """
        if format:
            return format
        
        suffix = Path(path).suffix.lower()
        format_map = {
            '.json': 'json',
            '.pkl': 'pickle',
            '.pickle': 'pickle',
            '.txt': 'text',
            '.safetensors': 'safetensors',
            '.pt': 'pytorch',
            '.npz': 'numpy',
            '.yaml': 'yaml',
            '.yml': 'yaml',
        }
        
        return format_map.get(suffix, 'pickle')
    
    def _serialize(self, data: Any, format: str) -> bytes:
        """Serialize data based on format.
        
        Args:
            data: Data to serialize
            format: Serialization format
            
        Returns:
            Serialized bytes
        """
        if format == 'json':
            return json.dumps(data, indent=2).encode('utf-8')
        elif format == 'pickle':
            return pickle.dumps(data)
        elif format == 'text':
            return str(data).encode('utf-8')
        else:
            # Default to pickle for unknown formats
            return pickle.dumps(data)
    
    def _deserialize(self, data: bytes, format: str) -> Any:
        """Deserialize data based on format.
        
        Args:
            data: Serialized data
            format: Serialization format
            
        Returns:
            Deserialized data
        """
        if format == 'json':
            return json.loads(data.decode('utf-8'))
        elif format == 'pickle':
            return pickle.loads(data)
        elif format == 'text':
            return data.decode('utf-8')
        else:
            # Try pickle as default
            return pickle.loads(data)


class BaseCheckpointAdapter(CheckpointPort, ABC):
    """Base checkpoint adapter with common functionality."""
    
    def __init__(self, storage: StoragePort):
        """Initialize checkpoint adapter.
        
        Args:
            storage: Storage port for persistence
        """
        self.storage = storage
    
    def _create_checkpoint_metadata(
        self,
        model: BertModel,
        training_state: TrainingState,
        optimizer_state: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Create checkpoint metadata.
        
        Args:
            model: Model being checkpointed
            training_state: Current training state
            optimizer_state: Optimizer state
            metadata: Additional metadata
            
        Returns:
            Complete checkpoint metadata
        """
        checkpoint_metadata = {
            'timestamp': datetime.now().isoformat(),
            'epoch': training_state.current_epoch,
            'step': training_state.global_step,
            'loss': training_state.current_loss,
            'metrics': training_state.metrics,
            'model_name': model.name,
            'model_type': model.model_type.value if hasattr(model, 'model_type') else 'unknown',
        }
        
        if metadata:
            checkpoint_metadata.update(metadata)
        
        return checkpoint_metadata
    
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
        checkpoints = self.list_checkpoints(directory)
        if not checkpoints:
            return None
        
        best_path = None
        best_value = None
        
        for checkpoint_info in checkpoints:
            path = checkpoint_info.get('path', '')
            metrics = checkpoint_info.get('metrics', {})
            
            if metric in metrics:
                value = metrics[metric]
                
                if best_value is None:
                    best_value = value
                    best_path = path
                elif mode == 'min' and value < best_value:
                    best_value = value
                    best_path = path
                elif mode == 'max' and value > best_value:
                    best_value = value
                    best_path = path
        
        return best_path
    
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
        checkpoints = self.list_checkpoints(directory)
        if len(checkpoints) <= keep_last:
            return []
        
        # Sort by timestamp (newest first)
        checkpoints.sort(
            key=lambda x: x.get('timestamp', ''),
            reverse=True
        )
        
        # Identify checkpoints to keep
        keep_paths = set()
        
        # Keep most recent checkpoints
        for checkpoint in checkpoints[:keep_last]:
            keep_paths.add(checkpoint['path'])
        
        # Keep best checkpoint if requested
        if keep_best:
            best_checkpoint = self.get_best_checkpoint(directory)
            if best_checkpoint:
                keep_paths.add(best_checkpoint)
        
        # Delete old checkpoints
        deleted = []
        for checkpoint in checkpoints:
            if checkpoint['path'] not in keep_paths:
                try:
                    self.storage.delete(checkpoint['path'])
                    deleted.append(checkpoint['path'])
                except Exception:
                    # Continue cleaning even if one deletion fails
                    pass
        
        return deleted