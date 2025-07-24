"""Filesystem implementation of StoragePort."""

import os
import shutil
import json
import pickle
from pathlib import Path
from typing import Any, List, Optional
import glob
import tempfile
from contextlib import contextmanager

from infrastructure.di import adapter, Scope
from ports.secondary.storage import StorageService
from adapters.secondary.storage.base import BaseStorageAdapter


@adapter(StorageService, scope=Scope.SINGLETON)
class FilesystemStorageAdapter(BaseStorageAdapter):
    """Filesystem implementation of the StoragePort."""
    
    def save(
        self,
        data: Any,
        path: str,
        format: Optional[str] = None,
    ) -> None:
        """Save data to filesystem.
        
        Args:
            data: Data to save
            path: Storage path
            format: Optional format specification
        """
        resolved_path = self._resolve_path(path)
        resolved_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Use atomic write for safety
        with self._atomic_write(resolved_path) as temp_path:
            format = self._get_format(path, format)
            
            if format == 'json':
                with open(temp_path, 'w', encoding='utf-8') as f:
                    json.dump(data, f, indent=2, ensure_ascii=False)
            elif format == 'text':
                with open(temp_path, 'w', encoding='utf-8') as f:
                    f.write(str(data))
            elif format == 'safetensors':
                try:
                    from safetensors import safe_open
                    from safetensors.mlx import save_file
                    save_file(data, temp_path)
                except ImportError:
                    raise ImportError("safetensors library required for safetensors format")
            elif format == 'numpy':
                try:
                    import numpy as np
                    np.savez_compressed(temp_path, **data if isinstance(data, dict) else {'data': data})
                except ImportError:
                    raise ImportError("numpy library required for numpy format")
            else:
                # Default to pickle
                with open(temp_path, 'wb') as f:
                    pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    def load(
        self,
        path: str,
        format: Optional[str] = None,
    ) -> Any:
        """Load data from filesystem.
        
        Args:
            path: Storage path
            format: Optional format specification
            
        Returns:
            Loaded data
        """
        resolved_path = self._resolve_path(path)
        
        if not resolved_path.exists():
            raise FileNotFoundError(f"Path not found: {path}")
        
        format = self._get_format(path, format)
        
        if format == 'json':
            with open(resolved_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        elif format == 'text':
            with open(resolved_path, 'r', encoding='utf-8') as f:
                return f.read()
        elif format == 'safetensors':
            try:
                from safetensors import safe_open
                tensors = {}
                with safe_open(resolved_path, framework="mlx") as f:
                    for key in f.keys():
                        tensors[key] = f.get_tensor(key)
                return tensors
            except ImportError:
                raise ImportError("safetensors library required for safetensors format")
        elif format == 'numpy':
            try:
                import numpy as np
                data = np.load(resolved_path, allow_pickle=False)
                if len(data.files) == 1 and 'data' in data.files:
                    return data['data']
                return dict(data)
            except ImportError:
                raise ImportError("numpy library required for numpy format")
        else:
            # Default to pickle
            with open(resolved_path, 'rb') as f:
                return pickle.load(f)
    
    def exists(self, path: str) -> bool:
        """Check if path exists.
        
        Args:
            path: Path to check
            
        Returns:
            True if exists
        """
        resolved_path = self._resolve_path(path)
        return resolved_path.exists()
    
    def delete(self, path: str) -> None:
        """Delete from filesystem.
        
        Args:
            path: Path to delete
        """
        resolved_path = self._resolve_path(path)
        
        if not resolved_path.exists():
            raise FileNotFoundError(f"Path not found: {path}")
        
        if resolved_path.is_file():
            resolved_path.unlink()
        elif resolved_path.is_dir():
            shutil.rmtree(resolved_path)
        else:
            raise ValueError(f"Cannot delete {path}: not a file or directory")
    
    def list(
        self,
        path: str,
        pattern: Optional[str] = None,
    ) -> List[str]:
        """List items in filesystem.
        
        Args:
            path: Directory path
            pattern: Optional filter pattern (glob syntax)
            
        Returns:
            List of paths relative to base_path
        """
        resolved_path = self._resolve_path(path)
        
        if not resolved_path.exists():
            return []
        
        if resolved_path.is_file():
            # If path is a file, return it if it matches the pattern
            if pattern:
                if resolved_path.match(pattern):
                    return [path]
                return []
            return [path]
        
        # List directory contents
        if pattern:
            # Use glob for pattern matching
            matches = list(resolved_path.glob(pattern))
            # Also check recursive glob
            if '**' in pattern:
                matches.extend(resolved_path.rglob(pattern.replace('**/', '')))
        else:
            # List all files recursively
            matches = [p for p in resolved_path.rglob('*') if p.is_file()]
        
        # Convert to relative paths
        result = []
        for match in matches:
            try:
                # Try to make relative to base_path
                rel_path = match.relative_to(self.base_path)
                result.append(str(rel_path))
            except ValueError:
                # If not relative to base_path, use the original path
                result.append(str(match))
        
        return sorted(set(result))  # Remove duplicates and sort
    
    def get_size(self, path: str) -> int:
        """Get size of stored item.
        
        Args:
            path: Path to item
            
        Returns:
            Size in bytes
        """
        resolved_path = self._resolve_path(path)
        
        if not resolved_path.exists():
            raise FileNotFoundError(f"Path not found: {path}")
        
        if resolved_path.is_file():
            return resolved_path.stat().st_size
        elif resolved_path.is_dir():
            total_size = 0
            for dirpath, dirnames, filenames in os.walk(resolved_path):
                for filename in filenames:
                    filepath = os.path.join(dirpath, filename)
                    total_size += os.path.getsize(filepath)
            return total_size
        else:
            return 0
    
    def copy(
        self,
        source: str,
        destination: str,
    ) -> None:
        """Copy item in filesystem.
        
        Args:
            source: Source path
            destination: Destination path
        """
        source_path = self._resolve_path(source)
        dest_path = self._resolve_path(destination)
        
        if not source_path.exists():
            raise FileNotFoundError(f"Source not found: {source}")
        
        # Ensure destination directory exists
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        
        if source_path.is_file():
            shutil.copy2(source_path, dest_path)
        elif source_path.is_dir():
            shutil.copytree(source_path, dest_path, dirs_exist_ok=True)
        else:
            raise ValueError(f"Cannot copy {source}: not a file or directory")
    
    @contextmanager
    def _atomic_write(self, path: Path):
        """Context manager for atomic file writes.
        
        Args:
            path: Target path
            
        Yields:
            Temporary file path
        """
        # Create temporary file in same directory for atomic rename
        temp_fd, temp_path = tempfile.mkstemp(
            dir=path.parent,
            prefix=f'.{path.name}.',
            suffix='.tmp'
        )
        
        try:
            os.close(temp_fd)  # Close the file descriptor
            yield temp_path
            # Atomic rename
            os.replace(temp_path, path)
        except Exception:
            # Clean up temporary file on error
            try:
                os.unlink(temp_path)
            except OSError:
                pass
            raise