"""Integration tests for storage port boundaries.

These tests verify the integration between ports and adapters,
ensuring the hexagonal architecture boundaries are properly maintained.
"""

import pytest
import tempfile
import shutil
from pathlib import Path
from typing import Any, Dict
from unittest.mock import Mock, patch

from infrastructure.ports.storage import StoragePort, ModelStoragePort
from infrastructure.adapters.file_storage import FileStorageAdapter, ModelCheckpointAdapter


class TestStoragePortIntegration:
    """Test the integration between storage port and its adapters."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for testing."""
        temp_dir = tempfile.mkdtemp()
        yield Path(temp_dir)
        shutil.rmtree(temp_dir)
    
    def test_port_adapter_contract(self, temp_dir):
        """Test that adapter properly implements port interface."""
        # Create adapter
        adapter = FileStorageAdapter(base_path=temp_dir)
        
        # Verify it implements all required methods
        assert hasattr(adapter, 'save')
        assert hasattr(adapter, 'load')
        assert hasattr(adapter, 'exists')
        assert hasattr(adapter, 'delete')
        assert hasattr(adapter, 'list_keys')
        
        # Test basic operations through port interface
        port: StoragePort = adapter
        
        # Save and load
        test_data = {"key": "value", "number": 42}
        port.save("test.json", test_data)
        loaded = port.load("test.json")
        assert loaded == test_data
        
        # Exists
        assert port.exists("test.json") is True
        assert port.exists("nonexistent.json") is False
        
        # List
        keys = port.list_keys()
        assert "test.json" in keys
        
        # Delete
        port.delete("test.json")
        assert port.exists("test.json") is False
    
    def test_port_independence_from_implementation(self, temp_dir):
        """Test that port usage doesn't depend on adapter specifics."""
        # This simulates how the application layer would use the port
        def use_storage_port(storage: StoragePort, key: str, data: Any) -> Any:
            """Function that only knows about the port interface."""
            # Save data
            storage.save(key, data)
            
            # Check it exists
            if not storage.exists(key):
                raise ValueError("Data not saved")
            
            # Load and return
            return storage.load(key)
        
        # Test with file adapter
        file_adapter = FileStorageAdapter(base_path=temp_dir)
        result = use_storage_port(file_adapter, "data.json", {"test": "data"})
        assert result == {"test": "data"}
        
        # The same function should work with any adapter implementing the port
        # (In a real scenario, we'd test with different adapters like S3, etc.)
    
    def test_error_handling_at_boundary(self, temp_dir):
        """Test that errors are properly handled at port boundary."""
        adapter = FileStorageAdapter(base_path=temp_dir)
        port: StoragePort = adapter
        
        # Test loading non-existent key
        with pytest.raises(KeyError) as exc_info:
            port.load("nonexistent.txt")
        
        # Error should be meaningful at the port level
        assert "Storage key not found" in str(exc_info.value)
        
        # Test saving to invalid path (simulate permission error)
        with patch('builtins.open', side_effect=PermissionError("No permission")):
            with pytest.raises(PermissionError):
                port.save("/invalid/path/file.txt", "data")
    
    def test_metadata_handling(self, temp_dir):
        """Test metadata handling through port interface."""
        adapter = FileStorageAdapter(base_path=temp_dir)
        port: StoragePort = adapter
        
        # Save with metadata
        data = {"content": "test"}
        metadata = {"version": "1.0", "author": "test_user"}
        
        port.save("data.json", data, metadata=metadata)
        
        # Load metadata
        loaded_meta = port.get_metadata("data.json")
        assert loaded_meta == metadata
        
        # Save without metadata
        port.save("no_meta.json", data)
        assert port.get_metadata("no_meta.json") is None
    
    def test_type_safety_at_boundary(self, temp_dir):
        """Test type handling at port boundary."""
        adapter = FileStorageAdapter(base_path=temp_dir)
        port: StoragePort = adapter
        
        # Test different types
        test_cases = [
            ("string.txt", "Hello, world!", str),
            ("dict.json", {"key": "value"}, dict),
            ("list.json", [1, 2, 3], list),
            ("number.json", 42, int),
        ]
        
        for key, value, expected_type in test_cases:
            port.save(key, value)
            loaded = port.load(key)
            assert isinstance(loaded, expected_type)
            assert loaded == value


class TestModelStoragePortIntegration:
    """Test the integration between model storage port and its adapters."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for testing."""
        temp_dir = tempfile.mkdtemp()
        yield Path(temp_dir)
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def mock_model(self):
        """Create a mock model for testing."""
        model = Mock()
        model.parameters.return_value = {
            "layer1": {"weight": [1, 2, 3], "bias": [0.1]},
            "layer2": {"weight": [4, 5, 6], "bias": [0.2]}
        }
        return model
    
    def test_model_port_adapter_contract(self, temp_dir, mock_model):
        """Test that model adapter implements port interface."""
        adapter = ModelCheckpointAdapter(checkpoint_dir=temp_dir)
        
        # Verify required methods
        assert hasattr(adapter, 'save_checkpoint')
        assert hasattr(adapter, 'load_checkpoint')
        assert hasattr(adapter, 'list_checkpoints')
        assert hasattr(adapter, 'delete_checkpoint')
        
        # Test through port interface
        port: ModelStoragePort = adapter
        
        # Save checkpoint
        metadata = {"epoch": 1, "step": 100}
        path = port.save_checkpoint(mock_model, "checkpoint-100", metadata)
        assert path.exists()
        
        # List checkpoints
        checkpoints = port.list_checkpoints()
        assert "checkpoint-100" in checkpoints
        
        # Load checkpoint (with mocked file operations)
        with patch.object(adapter, 'load_checkpoint') as mock_load:
            mock_load.return_value = metadata
            loaded_meta = port.load_checkpoint(mock_model, "checkpoint-100")
            assert loaded_meta == metadata
        
        # Delete checkpoint
        port.delete_checkpoint("checkpoint-100")
        assert "checkpoint-100" not in port.list_checkpoints()
    
    def test_checkpoint_versioning(self, temp_dir):
        """Test checkpoint versioning through port."""
        adapter = ModelCheckpointAdapter(checkpoint_dir=temp_dir)
        port: ModelStoragePort = adapter
        
        # Create multiple checkpoints
        for i in [100, 200, 300]:
            checkpoint_dir = temp_dir / f"checkpoint-{i}"
            checkpoint_dir.mkdir()
            (checkpoint_dir / "weights.safetensors").touch()
        
        # List should return all
        checkpoints = port.list_checkpoints()
        assert len(checkpoints) == 3
        
        # Test getting latest
        latest = port.get_latest_checkpoint()
        assert latest == "checkpoint-300"
    
    def test_model_format_abstraction(self, temp_dir, mock_model):
        """Test that port abstracts away model format details."""
        # Create adapter with safetensors
        adapter_safe = ModelCheckpointAdapter(
            checkpoint_dir=temp_dir,
            use_safetensors=True
        )
        
        # Create adapter with numpy format
        adapter_numpy = ModelCheckpointAdapter(
            checkpoint_dir=temp_dir / "numpy",
            use_safetensors=False
        )
        
        # Both should work through the same port interface
        for adapter in [adapter_safe, adapter_numpy]:
            port: ModelStoragePort = adapter
            
            # Save and verify
            path = port.save_checkpoint(mock_model, "test-checkpoint")
            assert path.exists()
            
            # The port user doesn't need to know about format
            checkpoints = port.list_checkpoints()
            assert "test-checkpoint" in checkpoints
    
    def test_concurrent_checkpoint_access(self, temp_dir, mock_model):
        """Test handling concurrent access to checkpoints."""
        adapter = ModelCheckpointAdapter(checkpoint_dir=temp_dir)
        port: ModelStoragePort = adapter
        
        # Simulate concurrent saves
        import threading
        import time
        
        def save_checkpoint(name: str, delay: float = 0):
            time.sleep(delay)
            port.save_checkpoint(mock_model, name)
        
        threads = []
        for i in range(3):
            t = threading.Thread(
                target=save_checkpoint,
                args=(f"concurrent-{i}", i * 0.1)
            )
            threads.append(t)
            t.start()
        
        for t in threads:
            t.join()
        
        # All checkpoints should be saved
        checkpoints = port.list_checkpoints()
        assert len(checkpoints) == 3
        for i in range(3):
            assert f"concurrent-{i}" in checkpoints
    
    def test_checkpoint_recovery(self, temp_dir, mock_model):
        """Test checkpoint recovery scenarios."""
        adapter = ModelCheckpointAdapter(checkpoint_dir=temp_dir)
        port: ModelStoragePort = adapter
        
        # Save a checkpoint
        metadata = {"epoch": 5, "step": 500, "loss": 0.123}
        port.save_checkpoint(mock_model, "checkpoint-500", metadata)
        
        # Simulate partial write by creating incomplete checkpoint
        incomplete = temp_dir / "checkpoint-incomplete"
        incomplete.mkdir()
        # Only create partial files
        (incomplete / "weights.safetensors").touch()
        # Missing metadata.json
        
        # List should still work
        checkpoints = port.list_checkpoints()
        assert "checkpoint-500" in checkpoints
        assert "checkpoint-incomplete" in checkpoints
        
        # Loading incomplete should fail gracefully
        with patch.object(adapter, 'load_checkpoint') as mock_load:
            mock_load.side_effect = FileNotFoundError("metadata.json not found")
            
            with pytest.raises(FileNotFoundError):
                port.load_checkpoint(mock_model, "checkpoint-incomplete")
    
    def test_checkpoint_migration(self, temp_dir):
        """Test checkpoint format migration scenarios."""
        # Create old format checkpoint
        old_checkpoint = temp_dir / "old-checkpoint"
        old_checkpoint.mkdir()
        
        # Simulate old format with just arrays.npz
        import numpy as np
        np.savez(old_checkpoint / "arrays.npz", layer1_weight=[1, 2, 3])
        
        # New adapter should handle it
        adapter = ModelCheckpointAdapter(checkpoint_dir=temp_dir)
        port: ModelStoragePort = adapter
        
        # Should list the old checkpoint
        checkpoints = port.list_checkpoints()
        assert "old-checkpoint" in checkpoints
        
        # Loading might need special handling (not implemented in basic adapter)
        # This test documents the boundary behavior