"""AWS S3 storage adapter implementation."""

import io
import json
import pickle
from typing import Any, List, Optional, Dict
from pathlib import Path
import tempfile

from domain.entities.model import BertModel
from domain.entities.training import TrainingState
from adapters.secondary.storage.base import BaseStorageAdapter, BaseCheckpointAdapter


class S3StorageAdapter(BaseStorageAdapter):
    """AWS S3 implementation of the StoragePort."""
    
    def __init__(
        self,
        bucket: str,
        prefix: Optional[str] = None,
        aws_access_key_id: Optional[str] = None,
        aws_secret_access_key: Optional[str] = None,
        region_name: Optional[str] = None,
    ):
        """Initialize S3 storage adapter.
        
        Args:
            bucket: S3 bucket name
            prefix: Optional key prefix
            aws_access_key_id: AWS access key ID
            aws_secret_access_key: AWS secret access key
            region_name: AWS region name
        """
        super().__init__(prefix)
        self.bucket = bucket
        self.prefix = prefix or ""
        
        try:
            import boto3
        except ImportError:
            raise ImportError("boto3 library required for S3 storage. Install with: pip install boto3")
        
        # Initialize S3 client
        self.s3 = boto3.client(
            's3',
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
            region_name=region_name,
        )
        
        # Verify bucket exists
        try:
            self.s3.head_bucket(Bucket=self.bucket)
        except Exception as e:
            raise ValueError(f"Cannot access S3 bucket '{self.bucket}': {e}")
    
    def _get_key(self, path: str) -> str:
        """Get S3 key from path.
        
        Args:
            path: Storage path
            
        Returns:
            S3 key
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
        """Save data to S3.
        
        Args:
            data: Data to save
            path: Storage path
            format: Optional format specification
        """
        key = self._get_key(path)
        format = self._get_format(path, format)
        
        # Serialize data
        serialized = self._serialize(data, format)
        
        # Upload to S3
        self.s3.put_object(
            Bucket=self.bucket,
            Key=key,
            Body=serialized,
            ContentType=self._get_content_type(format),
        )
    
    def load(
        self,
        path: str,
        format: Optional[str] = None,
    ) -> Any:
        """Load data from S3.
        
        Args:
            path: Storage path
            format: Optional format specification
            
        Returns:
            Loaded data
        """
        key = self._get_key(path)
        format = self._get_format(path, format)
        
        try:
            # Download from S3
            response = self.s3.get_object(Bucket=self.bucket, Key=key)
            data = response['Body'].read()
            
            # Deserialize data
            return self._deserialize(data, format)
        except self.s3.exceptions.NoSuchKey:
            raise FileNotFoundError(f"S3 key not found: {key}")
    
    def exists(self, path: str) -> bool:
        """Check if path exists in S3.
        
        Args:
            path: Path to check
            
        Returns:
            True if exists
        """
        key = self._get_key(path)
        
        try:
            self.s3.head_object(Bucket=self.bucket, Key=key)
            return True
        except self.s3.exceptions.NoSuchKey:
            return False
    
    def delete(self, path: str) -> None:
        """Delete from S3.
        
        Args:
            path: Path to delete
        """
        key = self._get_key(path)
        
        # Check if it's a "directory" (prefix)
        response = self.s3.list_objects_v2(
            Bucket=self.bucket,
            Prefix=key,
        )
        
        if 'Contents' not in response:
            raise FileNotFoundError(f"S3 key not found: {key}")
        
        # Delete all objects with this prefix
        objects = [{'Key': obj['Key']} for obj in response['Contents']]
        
        if objects:
            self.s3.delete_objects(
                Bucket=self.bucket,
                Delete={'Objects': objects}
            )
    
    def list(
        self,
        path: str,
        pattern: Optional[str] = None,
    ) -> List[str]:
        """List items in S3.
        
        Args:
            path: Directory path
            pattern: Optional filter pattern (supports basic wildcards)
            
        Returns:
            List of paths
        """
        prefix = self._get_key(path)
        if not prefix.endswith('/'):
            prefix += '/'
        
        # List objects with prefix
        paginator = self.s3.get_paginator('list_objects_v2')
        pages = paginator.paginate(Bucket=self.bucket, Prefix=prefix)
        
        results = []
        for page in pages:
            if 'Contents' in page:
                for obj in page['Contents']:
                    key = obj['Key']
                    # Remove prefix to get relative path
                    rel_path = key[len(self.prefix):].lstrip('/') if self.prefix else key
                    
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
        key = self._get_key(path)
        
        try:
            response = self.s3.head_object(Bucket=self.bucket, Key=key)
            return response['ContentLength']
        except self.s3.exceptions.NoSuchKey:
            # Check if it's a "directory"
            prefix = key
            if not prefix.endswith('/'):
                prefix += '/'
            
            total_size = 0
            paginator = self.s3.get_paginator('list_objects_v2')
            pages = paginator.paginate(Bucket=self.bucket, Prefix=prefix)
            
            for page in pages:
                if 'Contents' in page:
                    for obj in page['Contents']:
                        total_size += obj['Size']
            
            if total_size == 0:
                raise FileNotFoundError(f"S3 key not found: {key}")
            
            return total_size
    
    def copy(
        self,
        source: str,
        destination: str,
    ) -> None:
        """Copy item in S3.
        
        Args:
            source: Source path
            destination: Destination path
        """
        source_key = self._get_key(source)
        dest_key = self._get_key(destination)
        
        # Check if source is a single object
        try:
            self.s3.head_object(Bucket=self.bucket, Key=source_key)
            # Single object copy
            copy_source = {'Bucket': self.bucket, 'Key': source_key}
            self.s3.copy_object(
                CopySource=copy_source,
                Bucket=self.bucket,
                Key=dest_key
            )
        except self.s3.exceptions.NoSuchKey:
            # Try as prefix (directory)
            prefix = source_key
            if not prefix.endswith('/'):
                prefix += '/'
            
            # List and copy all objects
            paginator = self.s3.get_paginator('list_objects_v2')
            pages = paginator.paginate(Bucket=self.bucket, Prefix=prefix)
            
            copied = False
            for page in pages:
                if 'Contents' in page:
                    for obj in page['Contents']:
                        source_obj_key = obj['Key']
                        # Calculate destination key
                        rel_path = source_obj_key[len(prefix):]
                        dest_obj_key = dest_key
                        if not dest_obj_key.endswith('/'):
                            dest_obj_key += '/'
                        dest_obj_key += rel_path
                        
                        copy_source = {'Bucket': self.bucket, 'Key': source_obj_key}
                        self.s3.copy_object(
                            CopySource=copy_source,
                            Bucket=self.bucket,
                            Key=dest_obj_key
                        )
                        copied = True
            
            if not copied:
                raise FileNotFoundError(f"S3 source not found: {source}")
    
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
        }
        return content_types.get(format, 'application/octet-stream')


class S3CheckpointAdapter(BaseCheckpointAdapter):
    """S3 implementation of the CheckpointPort."""
    
    def __init__(
        self,
        bucket: str,
        prefix: Optional[str] = None,
        **kwargs
    ):
        """Initialize S3 checkpoint adapter.
        
        Args:
            bucket: S3 bucket name
            prefix: Optional key prefix
            **kwargs: Additional S3 configuration
        """
        storage = S3StorageAdapter(bucket, prefix, **kwargs)
        super().__init__(storage)
        self.bucket = bucket
        self.prefix = prefix or ""
    
    def save_checkpoint(
        self,
        model: BertModel,
        training_state: TrainingState,
        optimizer_state: Dict[str, Any],
        path: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Save training checkpoint to S3.
        
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
        """Load training checkpoint from S3.
        
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
        """Save model for inference to S3.
        
        Args:
            model: Model to save
            path: Save path
            save_config: Whether to save configuration
            save_tokenizer: Whether to save tokenizer
        """
        # Save model weights
        model_state = dict(model.parameters()) if hasattr(model, 'parameters') else model.state_dict()
        weights_path = f"{path}/model.safetensors"
        
        # For S3, we need to use a temporary file for safetensors
        with tempfile.NamedTemporaryFile(suffix='.safetensors') as tmp:
            try:
                from safetensors.mlx import save_file
                save_file(model_state, tmp.name)
                tmp.seek(0)
                # Upload to S3
                key = self.storage._get_key(weights_path)
                self.storage.s3.upload_fileobj(tmp, self.storage.bucket, key)
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
    ) -> BertModel:
        """Load model for inference from S3.
        
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
                key = self.storage._get_key(weights_path)
                self.storage.s3.download_fileobj(self.storage.bucket, key, tmp)
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
        """List available checkpoints in S3.
        
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
            "Model export for S3 storage requires local export first. "
            "Use FilesystemCheckpointAdapter for export operations."
        )