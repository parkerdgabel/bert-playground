"""Filesystem implementation of CheckpointPort."""

import json
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime
import re

from domain.entities.model import BertModel
from domain.entities.training import TrainingState
from adapters.secondary.storage.base import BaseCheckpointAdapter
from adapters.secondary.storage.filesystem.storage_adapter import FilesystemStorageAdapter
from adapters.secondary.storage.filesystem.utils import (
    save_mlx_model,
    load_mlx_model,
    save_optimizer_state,
    load_optimizer_state,
)


class FilesystemCheckpointAdapter(BaseCheckpointAdapter):
    """Filesystem implementation of the CheckpointPort."""
    
    def __init__(self, base_path: Optional[str] = None):
        """Initialize filesystem checkpoint adapter.
        
        Args:
            base_path: Base path for checkpoint storage
        """
        storage = FilesystemStorageAdapter(base_path)
        super().__init__(storage)
        self.base_path = Path(base_path) if base_path else Path.cwd()
    
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
        checkpoint_dir = self.base_path / path
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Save model weights
        model_path = checkpoint_dir / "model.safetensors"
        save_mlx_model(model, model_path)
        
        # Save model config
        if hasattr(model, 'config'):
            config_path = checkpoint_dir / "config.json"
            config_data = model.config.to_dict() if hasattr(model.config, 'to_dict') else model.config
            self.storage.save(config_data, str(config_path), format='json')
        
        # Save optimizer state
        optimizer_path = checkpoint_dir / "optimizer.pkl"
        save_optimizer_state(optimizer_state, optimizer_path)
        
        # Save training state
        training_state_data = {
            'current_epoch': training_state.current_epoch,
            'global_step': training_state.global_step,
            'current_loss': float(training_state.current_loss) if training_state.current_loss is not None else None,
            'metrics': {k: float(v) if isinstance(v, (int, float)) else v for k, v in training_state.metrics.items()},
            'best_metric': float(training_state.best_metric) if training_state.best_metric is not None else None,
            'best_metric_name': training_state.best_metric_name,
            'epochs_since_improvement': training_state.epochs_since_improvement,
        }
        training_state_path = checkpoint_dir / "training_state.json"
        self.storage.save(training_state_data, str(training_state_path), format='json')
        
        # Save checkpoint metadata
        checkpoint_metadata = self._create_checkpoint_metadata(
            model, training_state, optimizer_state, metadata
        )
        metadata_path = checkpoint_dir / "metadata.json"
        self.storage.save(checkpoint_metadata, str(metadata_path), format='json')
    
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
        checkpoint_dir = self.base_path / path
        
        if not checkpoint_dir.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")
        
        # Load model
        model_path = checkpoint_dir / "model.safetensors"
        config_path = checkpoint_dir / "config.json"
        
        config = None
        if config_path.exists():
            config = self.storage.load(str(config_path), format='json')
        
        model = load_mlx_model(model_path, config)
        
        # Load optimizer state
        optimizer_path = checkpoint_dir / "optimizer.pkl"
        optimizer_state = None
        if optimizer_path.exists():
            optimizer_state = load_optimizer_state(optimizer_path)
        
        # Load training state
        training_state_path = checkpoint_dir / "training_state.json"
        training_state_data = self.storage.load(str(training_state_path), format='json')
        
        # Reconstruct TrainingState
        training_state = TrainingState(
            current_epoch=training_state_data['current_epoch'],
            global_step=training_state_data['global_step'],
            current_loss=training_state_data.get('current_loss'),
            metrics=training_state_data.get('metrics', {}),
            best_metric=training_state_data.get('best_metric'),
            best_metric_name=training_state_data.get('best_metric_name', 'loss'),
            epochs_since_improvement=training_state_data.get('epochs_since_improvement', 0),
        )
        
        # Load metadata
        metadata_path = checkpoint_dir / "metadata.json"
        metadata = None
        if metadata_path.exists():
            metadata = self.storage.load(str(metadata_path), format='json')
        
        return {
            'model': model,
            'training_state': training_state,
            'optimizer_state': optimizer_state,
            'metadata': metadata,
        }
    
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
        model_dir = self.base_path / path
        model_dir.mkdir(parents=True, exist_ok=True)
        
        # Save model weights
        weights_path = model_dir / "model.safetensors"
        save_mlx_model(model, weights_path)
        
        # Save config if requested
        if save_config and hasattr(model, 'config'):
            config_path = model_dir / "config.json"
            config_data = model.config.to_dict() if hasattr(model.config, 'to_dict') else model.config
            self.storage.save(config_data, str(config_path), format='json')
        
        # Save tokenizer if requested
        if save_tokenizer and hasattr(model, 'tokenizer'):
            tokenizer_dir = model_dir / "tokenizer"
            tokenizer_dir.mkdir(exist_ok=True)
            if hasattr(model.tokenizer, 'save'):
                model.tokenizer.save(str(tokenizer_dir))
        
        # Save model metadata
        metadata = {
            'saved_at': datetime.now().isoformat(),
            'model_name': model.name,
            'model_type': model.model_type.value if hasattr(model, 'model_type') else 'unknown',
            'framework': 'mlx',
        }
        metadata_path = model_dir / "model_metadata.json"
        self.storage.save(metadata, str(metadata_path), format='json')
    
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
        model_dir = self.base_path / path
        
        if not model_dir.exists():
            raise FileNotFoundError(f"Model not found: {path}")
        
        # Load config if requested
        config = None
        if load_config:
            config_path = model_dir / "config.json"
            if config_path.exists():
                config = self.storage.load(str(config_path), format='json')
        
        # Load model weights
        weights_path = model_dir / "model.safetensors"
        model = load_mlx_model(weights_path, config)
        
        # Load tokenizer if available
        tokenizer_dir = model_dir / "tokenizer"
        if tokenizer_dir.exists():
            # This would need to be implemented based on tokenizer type
            pass
        
        return model
    
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
        dir_path = self.base_path / directory
        
        if not dir_path.exists():
            return []
        
        checkpoints = []
        
        # Find all checkpoint directories
        for item in dir_path.iterdir():
            if item.is_dir():
                metadata_path = item / "metadata.json"
                if metadata_path.exists():
                    try:
                        metadata = self.storage.load(str(metadata_path), format='json')
                        checkpoint_info = {
                            'path': str(item.relative_to(self.base_path)),
                            'timestamp': metadata.get('timestamp'),
                            'epoch': metadata.get('epoch'),
                            'step': metadata.get('step'),
                            'metrics': metadata.get('metrics', {}),
                        }
                        checkpoints.append(checkpoint_info)
                    except Exception:
                        # Skip invalid checkpoints
                        continue
        
        # Sort by timestamp
        checkpoints.sort(key=lambda x: x.get('timestamp', ''), reverse=True)
        
        return checkpoints
    
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
        
        # Filter checkpoints that have the metric
        valid_checkpoints = [
            cp for cp in checkpoints
            if metric in cp.get('metrics', {})
        ]
        
        if not valid_checkpoints:
            return None
        
        # Find best checkpoint
        if mode == 'min':
            best = min(valid_checkpoints, key=lambda x: x['metrics'][metric])
        else:
            best = max(valid_checkpoints, key=lambda x: x['metrics'][metric])
        
        return best['path']
    
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
        export_path = self.base_path / path
        export_path.parent.mkdir(parents=True, exist_ok=True)
        
        if format == "onnx":
            # Would need ONNX export implementation
            raise NotImplementedError("ONNX export not yet implemented")
        elif format == "coreml":
            # Would need CoreML export implementation
            raise NotImplementedError("CoreML export not yet implemented")
        else:
            raise ValueError(f"Unsupported export format: {format}")
        
        # Save export metadata
        metadata = {
            'exported_at': datetime.now().isoformat(),
            'format': format,
            'optimized': optimize,
            'source_model': model.name,
        }
        metadata_path = export_path.with_suffix('.meta.json')
        self.storage.save(metadata, str(metadata_path), format='json')