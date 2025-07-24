"""Google Cloud Storage adapter implementation."""

import io
import json
import pickle
from typing import Any, List, Optional, Dict
from pathlib import Path
import tempfile
from datetime import datetime

from infrastructure.di import adapter, Scope
from application.ports.secondary.storage import StorageService
from application.ports.secondary.checkpointing import CheckpointManager
from domain.entities.model import BertModel
from domain.entities.training import TrainingState
from adapters.secondary.storage.base import BaseStorageAdapter, BaseCheckpointAdapter


@adapter(StorageService, scope=Scope.SINGLETON)
class GCSStorageAdapter(BaseStorageAdapter):
    """Google Cloud Storage implementation of the StoragePort."""
    
    def __init__(
        self,
        bucket_name: str,
        prefix: Optional[str] = None,
        project: Optional[str] = None,
        credentials_path: Optional[str] = None,
    ):
        """Initialize GCS storage adapter.
        
        Args:
            bucket_name: GCS bucket name
            prefix: Optional key prefix
            project: GCP project ID
            credentials_path: Path to service account credentials JSON
        """
        super().__init__(prefix)
        self.bucket_name = bucket_name
        self.prefix = prefix or ""
        
        try:
            from google.cloud import storage
        except ImportError:
            raise ImportError(
                "google-cloud-storage library required for GCS storage. "
                "Install with: pip install google-cloud-storage"
            )
        
        # Initialize GCS client
        if credentials_path:
            self.client = storage.Client.from_service_account_json(
                credentials_path,
                project=project
            )
        else:
            self.client = storage.Client(project=project)
        
        # Get bucket
        try:
            self.bucket = self.client.bucket(self.bucket_name)
            # Verify bucket exists
            self.bucket.reload()
        except Exception as e:
            raise ValueError(f"Cannot access GCS bucket '{self.bucket_name}': {e}")
    
    def _get_blob_name(self, path: str) -> str:
        """Get GCS blob name from path.
        
        Args:
            path: Storage path
            
        Returns:
            GCS blob name
        """
        if self.prefix:
            return f"{self.prefix}/{path}".lstrip('/')
        return path.lstrip('/')
    
    def save(
        self,
        data: Any,
        path: str,
        format: Optional[str] = None,
    ) -> None:
        """Save data to GCS.
        
        Args:
            data: Data to save
            path: Storage path
            format: Optional format specification
        """
        blob_name = self._get_blob_name(path)
        format = self._get_format(path, format)
        
        # Serialize data
        serialized = self._serialize(data, format)
        
        # Upload to GCS
        blob = self.bucket.blob(blob_name)
        blob.upload_from_string(
            serialized,
            content_type=self._get_content_type(format)
        )
    
    def load(
        self,
        path: str,
        format: Optional[str] = None,
    ) -> Any:
        """Load data from GCS.
        
        Args:
            path: Storage path
            format: Optional format specification
            
        Returns:
            Loaded data
        """
        blob_name = self._get_blob_name(path)
        format = self._get_format(path, format)
        
        blob = self.bucket.blob(blob_name)
        
        if not blob.exists():
            raise FileNotFoundError(f"GCS blob not found: {blob_name}")
        
        # Download from GCS
        data = blob.download_as_bytes()
        
        # Deserialize data
        return self._deserialize(data, format)
    
    def exists(self, path: str) -> bool:
        """Check if path exists in GCS.
        
        Args:
            path: Path to check
            
        Returns:
            True if exists
        """
        blob_name = self._get_blob_name(path)
        blob = self.bucket.blob(blob_name)
        return blob.exists()
    
    def delete(self, path: str) -> None:
        """Delete from GCS.
        
        Args:
            path: Path to delete
        """
        blob_name = self._get_blob_name(path)
        
        # Check if it's a single blob
        blob = self.bucket.blob(blob_name)
        if blob.exists():
            blob.delete()
            return
        
        # Try as prefix (directory)
        prefix = blob_name
        if not prefix.endswith('/'):
            prefix += '/'
        
        blobs = list(self.bucket.list_blobs(prefix=prefix))
        
        if not blobs:
            raise FileNotFoundError(f"GCS path not found: {path}")
        
        # Delete all blobs with this prefix
        for blob in blobs:
            blob.delete()
    
    def list(
        self,
        path: str,
        pattern: Optional[str] = None,
    ) -> List[str]:
        """List items in GCS.
        
        Args:
            path: Directory path
            pattern: Optional filter pattern (supports basic wildcards)
            
        Returns:
            List of paths
        """
        prefix = self._get_blob_name(path)
        if prefix and not prefix.endswith('/'):
            prefix += '/'
        
        # List blobs with prefix
        blobs = self.bucket.list_blobs(prefix=prefix)
        
        results = []
        for blob in blobs:
            # Remove prefix to get relative path
            rel_path = blob.name[len(self.prefix):].lstrip('/') if self.prefix else blob.name
            
            # Apply pattern matching if specified
            if pattern:
                import fnmatch
                if fnmatch.fnmatch(rel_path, pattern):
                    results.append(rel_path)
            else:
                results.append(rel_path)
        
        return sorted(results)
    
    def get_size(self, path: str) -> int:
        """Get size of stored item.
        
        Args:
            path: Path to item
            
        Returns:
            Size in bytes
        """
        blob_name = self._get_blob_name(path)
        
        # Check if it's a single blob
        blob = self.bucket.blob(blob_name)
        if blob.exists():
            blob.reload()
            return blob.size
        
        # Check if it's a "directory"
        prefix = blob_name
        if not prefix.endswith('/'):
            prefix += '/'
        
        total_size = 0
        blobs = self.bucket.list_blobs(prefix=prefix)
        
        found = False
        for blob in blobs:
            total_size += blob.size
            found = True
        
        if not found:
            raise FileNotFoundError(f"GCS path not found: {path}")
        
        return total_size
    
    def copy(
        self,
        source: str,
        destination: str,
    ) -> None:
        """Copy item in GCS.
        
        Args:
            source: Source path
            destination: Destination path
        """
        source_blob_name = self._get_blob_name(source)
        dest_blob_name = self._get_blob_name(destination)
        
        # Check if source is a single blob
        source_blob = self.bucket.blob(source_blob_name)
        if source_blob.exists():
            # Single blob copy
            dest_blob = self.bucket.blob(dest_blob_name)
            dest_blob.upload_from_string(
                source_blob.download_as_bytes(),
                content_type=source_blob.content_type
            )
            return
        
        # Try as prefix (directory)
        prefix = source_blob_name
        if not prefix.endswith('/'):
            prefix += '/'
        
        blobs = list(self.bucket.list_blobs(prefix=prefix))
        
        if not blobs:
            raise FileNotFoundError(f"GCS source not found: {source}")
        
        # Copy all blobs
        for source_blob in blobs:
            # Calculate destination blob name
            rel_path = source_blob.name[len(prefix):]
            dest_name = dest_blob_name
            if not dest_name.endswith('/'):
                dest_name += '/'
            dest_name += rel_path
            
            dest_blob = self.bucket.blob(dest_name)
            dest_blob.upload_from_string(
                source_blob.download_as_bytes(),
                content_type=source_blob.content_type
            )
    
    def _get_content_type(self, format: str) -> str:
        """Get content type for format.
        
        Args:
            format: Data format
            
        Returns:
            Content type string
        """
        content_types = {
            'json': 'application/json',
            'text': 'text/plain',
            'pickle': 'application/octet-stream',
            'safetensors': 'application/octet-stream',
            'numpy': 'application/octet-stream',
            'yaml': 'application/x-yaml',
        }
        return content_types.get(format, 'application/octet-stream')


@adapter(CheckpointManager, scope=Scope.SINGLETON)
class GCSCheckpointAdapter(BaseCheckpointAdapter):
    """GCS implementation of the CheckpointPort."""
    
    def __init__(
        self,
        bucket_name: str,
        prefix: Optional[str] = None,
        **kwargs
    ):
        """Initialize GCS checkpoint adapter.
        
        Args:
            bucket_name: GCS bucket name
            prefix: Optional key prefix
            **kwargs: Additional GCS configuration
        """
        storage = GCSStorageAdapter(bucket_name, prefix, **kwargs)
        super().__init__(storage)
        self.bucket_name = bucket_name
        self.prefix = prefix or ""
    
    def save_checkpoint(
        self,
        model: BertModel,
        training_state: TrainingState,
        optimizer_state: Dict[str, Any],
        path: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Save training checkpoint to GCS.
        
        Args:
            model: Model to save
            training_state: Current training state
            optimizer_state: Optimizer state
            path: Checkpoint path
            metadata: Optional metadata
        """
        # Create checkpoint data
        checkpoint_data = {
            'model_state': dict(model.parameters()) if hasattr(model, 'parameters') else model.state_dict(),
            'training_state': {
                'current_epoch': training_state.current_epoch,
                'global_step': training_state.global_step,
                'current_loss': float(training_state.current_loss) if training_state.current_loss is not None else None,
                'metrics': training_state.metrics,
                'best_metric': float(training_state.best_metric) if training_state.best_metric is not None else None,
                'best_metric_name': training_state.best_metric_name,
                'epochs_since_improvement': training_state.epochs_since_improvement,
            },
            'optimizer_state': optimizer_state,
            'metadata': self._create_checkpoint_metadata(
                model, training_state, optimizer_state, metadata
            ),
        }
        
        # Save checkpoint as single file
        checkpoint_path = f"{path}/checkpoint.pkl"
        self.storage.save(checkpoint_data, checkpoint_path, format='pickle')
        
        # Also save metadata separately for easy access
        metadata_path = f"{path}/metadata.json"
        self.storage.save(checkpoint_data['metadata'], metadata_path, format='json')
    
    def load_checkpoint(
        self,
        path: str,
    ) -> Dict[str, Any]:
        """Load training checkpoint from GCS.
        
        Args:
            path: Checkpoint path
            
        Returns:
            Dictionary containing checkpoint data
        """
        checkpoint_path = f"{path}/checkpoint.pkl"
        
        try:
            checkpoint_data = self.storage.load(checkpoint_path, format='pickle')
        except FileNotFoundError:
            raise FileNotFoundError(f"Checkpoint not found: {path}")
        
        # Reconstruct TrainingState
        ts_data = checkpoint_data['training_state']
        training_state = TrainingState(
            current_epoch=ts_data['current_epoch'],
            global_step=ts_data['global_step'],
            current_loss=ts_data.get('current_loss'),
            metrics=ts_data.get('metrics', {}),
            best_metric=ts_data.get('best_metric'),
            best_metric_name=ts_data.get('best_metric_name', 'loss'),
            epochs_since_improvement=ts_data.get('epochs_since_improvement', 0),
        )
        
        return {
            'model': checkpoint_data['model_state'],  # Note: returns state dict, not model instance
            'training_state': training_state,
            'optimizer_state': checkpoint_data.get('optimizer_state'),
            'metadata': checkpoint_data.get('metadata'),
        }
    
    def save_model(
        self,
        model: BertModel,
        path: str,
        save_config: bool = True,
        save_tokenizer: bool = False,
    ) -> None:
        """Save model for inference to GCS.
        
        Args:
            model: Model to save
            path: Save path
            save_config: Whether to save configuration
            save_tokenizer: Whether to save tokenizer
        """
        # Save model weights
        model_state = dict(model.parameters()) if hasattr(model, 'parameters') else model.state_dict()
        weights_path = f"{path}/model.safetensors"
        
        # For GCS, we need to use a temporary file for safetensors
        with tempfile.NamedTemporaryFile(suffix='.safetensors') as tmp:
            try:
                from safetensors.mlx import save_file
                save_file(model_state, tmp.name)
                tmp.seek(0)
                # Upload to GCS
                blob_name = self.storage._get_blob_name(weights_path)
                blob = self.storage.bucket.blob(blob_name)
                blob.upload_from_file(tmp, content_type='application/octet-stream')
            except ImportError:
                # Fallback to pickle if safetensors not available
                weights_path = f"{path}/model.pkl"
                self.storage.save(model_state, weights_path, format='pickle')
        
        # Save config if requested
        if save_config and hasattr(model, 'config'):
            config_data = model.config.to_dict() if hasattr(model.config, 'to_dict') else model.config
            config_path = f"{path}/config.json"
            self.storage.save(config_data, config_path, format='json')
        
        # Save model metadata
        metadata = {
            'saved_at': datetime.now().isoformat(),
            'model_name': model.name,
            'model_type': model.model_type.value if hasattr(model, 'model_type') else 'unknown',
            'framework': 'mlx',
        }
        metadata_path = f"{path}/model_metadata.json"
        self.storage.save(metadata, metadata_path, format='json')
    
    def load_model(
        self,
        path: str,
        load_config: bool = True,
    ) -> Dict[str, Any]:
        """Load model for inference from GCS.
        
        Note: This returns a dict with weights and config, not a model instance.
        The caller should instantiate the model with the config and load the weights.
        
        Args:
            path: Model path
            load_config: Whether to load configuration
            
        Returns:
            Dict with 'weights' and optional 'config'
        """
        # Load config if requested
        config = None
        if load_config:
            config_path = f"{path}/config.json"
            try:
                config = self.storage.load(config_path, format='json')
            except FileNotFoundError:
                pass
        
        # Try to load safetensors first
        weights_path = f"{path}/model.safetensors"
        try:
            # Download to temporary file
            with tempfile.NamedTemporaryFile(suffix='.safetensors') as tmp:
                blob_name = self.storage._get_blob_name(weights_path)
                blob = self.storage.bucket.blob(blob_name)
                blob.download_to_file(tmp)
                tmp.seek(0)
                
                from safetensors.mlx import load_file
                weights = load_file(tmp.name)
        except (FileNotFoundError, ImportError):
            # Fallback to pickle
            weights_path = f"{path}/model.pkl"
            weights = self.storage.load(weights_path, format='pickle')
        
        return {
            'weights': weights,
            'config': config,
        }
    
    def list_checkpoints(
        self,
        directory: str,
    ) -> List[Dict[str, Any]]:
        """List available checkpoints in GCS.
        
        Args:
            directory: Directory to search
            
        Returns:
            List of checkpoint info dictionaries
        """
        checkpoints = []
        
        # List all subdirectories that contain metadata.json
        items = self.storage.list(directory, pattern='*/metadata.json')
        
        for item in items:
            # Extract checkpoint directory from metadata path
            checkpoint_dir = str(Path(item).parent)
            
            try:
                metadata = self.storage.load(item, format='json')
                checkpoint_info = {
                    'path': checkpoint_dir,
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
            format: Target format
            optimize: Whether to optimize
        """
        # Export would need to be done locally first
        raise NotImplementedError(
            "Model export for GCS storage requires local export first. "
            "Use FilesystemCheckpointAdapter for export operations."
        )