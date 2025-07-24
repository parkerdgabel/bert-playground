"""Unit tests for file storage adapter.

These tests verify the file storage adapter implementation
in isolation from external dependencies.
"""

import pytest
import json
import pickle
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch, mock_open

from adapters.secondary.storage.file_storage import FileStorageAdapter, ModelFileStorageAdapter
from application.ports.secondary.storage import StorageKey, StorageValue, Metadata


class TestFileStorageAdapter:
    """Test file storage adapter implementation."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for testing."""
        temp_dir = tempfile.mkdtemp()
        yield Path(temp_dir)
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def storage(self, temp_dir):
        """Create storage adapter with temporary directory."""
        return FileStorageAdapter(base_path=temp_dir)
    
    def test_initialization_default(self):
        """Test default initialization."""
        storage = FileStorageAdapter()
        assert storage.base_path == Path.cwd()
    
    def test_initialization_with_path(self, temp_dir):
        """Test initialization with custom path."""
        storage = FileStorageAdapter(base_path=temp_dir)
        assert storage.base_path == temp_dir
        assert temp_dir.exists()
    
    def test_resolve_path_absolute(self, storage, temp_dir):
        """Test resolving absolute paths."""
        abs_path = temp_dir / "test.txt"
        resolved = storage._resolve_path(abs_path)
        assert resolved == abs_path
    
    def test_resolve_path_relative(self, storage, temp_dir):
        """Test resolving relative paths."""
        rel_path = "subdir/test.txt"
        resolved = storage._resolve_path(rel_path)
        assert resolved == temp_dir / "subdir/test.txt"
    
    def test_resolve_path_string_key(self, storage, temp_dir):
        """Test resolving string keys."""
        key = "data/model.pkl"
        resolved = storage._resolve_path(key)
        assert resolved == temp_dir / "data/model.pkl"
    
    def test_save_json(self, storage, temp_dir):
        """Test saving JSON data."""
        data = {"key": "value", "number": 42}
        key = "test.json"
        
        storage.save(key, data)
        
        saved_path = temp_dir / key
        assert saved_path.exists()
        
        with open(saved_path, "r") as f:
            loaded = json.load(f)
        assert loaded == data
    
    def test_save_pickle(self, storage, temp_dir):
        """Test saving pickle data."""
        data = {"key": "value", "list": [1, 2, 3]}
        key = "test.pkl"
        
        storage.save(key, data)
        
        saved_path = temp_dir / key
        assert saved_path.exists()
        
        with open(saved_path, "rb") as f:
            loaded = pickle.load(f)
        assert loaded == data
    
    def test_save_text(self, storage, temp_dir):
        """Test saving text data."""
        text = "Hello, world!"
        key = "test.txt"
        
        storage.save(key, text)
        
        saved_path = temp_dir / key
        assert saved_path.exists()
        
        with open(saved_path, "r") as f:
            loaded = f.read()
        assert loaded == text
    
    def test_save_with_metadata(self, storage, temp_dir):
        """Test saving with metadata."""
        data = {"key": "value"}
        metadata = {"version": "1.0", "created_by": "test"}
        key = "test.json"
        
        storage.save(key, data, metadata=metadata)
        
        # Check data file
        assert (temp_dir / key).exists()
        
        # Check metadata file
        meta_path = temp_dir / "test.json.meta"
        assert meta_path.exists()
        
        with open(meta_path, "r") as f:
            loaded_meta = json.load(f)
        assert loaded_meta == metadata
    
    def test_save_creates_directories(self, storage, temp_dir):
        """Test that save creates necessary directories."""
        data = {"test": "data"}
        key = "deep/nested/path/test.json"
        
        storage.save(key, data)
        
        saved_path = temp_dir / key
        assert saved_path.exists()
        assert saved_path.parent.exists()
    
    def test_load_json(self, storage, temp_dir):
        """Test loading JSON data."""
        data = {"key": "value", "number": 42}
        key = "test.json"
        
        # Save data first
        with open(temp_dir / key, "w") as f:
            json.dump(data, f)
        
        loaded = storage.load(key)
        assert loaded == data
    
    def test_load_pickle(self, storage, temp_dir):
        """Test loading pickle data."""
        data = {"key": "value", "list": [1, 2, 3]}
        key = "test.pkl"
        
        # Save data first
        with open(temp_dir / key, "wb") as f:
            pickle.dump(data, f)
        
        loaded = storage.load(key)
        assert loaded == data
    
    def test_load_text(self, storage, temp_dir):
        """Test loading text data."""
        text = "Hello, world!"
        key = "test.txt"
        
        # Save data first
        with open(temp_dir / key, "w") as f:
            f.write(text)
        
        loaded = storage.load(key)
        assert loaded == text
    
    def test_load_nonexistent_key(self, storage):
        """Test loading non-existent key raises KeyError."""
        with pytest.raises(KeyError, match="Storage key not found"):
            storage.load("nonexistent.txt")
    
    def test_exists_true(self, storage, temp_dir):
        """Test exists returns True for existing files."""
        key = "test.txt"
        (temp_dir / key).touch()
        
        assert storage.exists(key) is True
    
    def test_exists_false(self, storage):
        """Test exists returns False for non-existing files."""
        assert storage.exists("nonexistent.txt") is False
    
    def test_delete_existing(self, storage, temp_dir):
        """Test deleting existing file."""
        key = "test.txt"
        file_path = temp_dir / key
        file_path.touch()
        
        storage.delete(key)
        
        assert not file_path.exists()
    
    def test_delete_nonexistent(self, storage):
        """Test deleting non-existent file doesn't raise error."""
        # Should not raise any exception
        storage.delete("nonexistent.txt")
    
    def test_list_keys_flat(self, storage, temp_dir):
        """Test listing keys in flat directory."""
        # Create test files
        (temp_dir / "file1.txt").touch()
        (temp_dir / "file2.json").touch()
        (temp_dir / "file3.pkl").touch()
        
        keys = storage.list_keys()
        
        assert len(keys) == 3
        assert "file1.txt" in keys
        assert "file2.json" in keys
        assert "file3.pkl" in keys
    
    def test_list_keys_with_prefix(self, storage, temp_dir):
        """Test listing keys with prefix."""
        # Create test files
        (temp_dir / "data").mkdir()
        (temp_dir / "data/file1.txt").touch()
        (temp_dir / "data/file2.txt").touch()
        (temp_dir / "other/file3.txt").touch()
        
        keys = storage.list_keys(prefix="data/")
        
        assert len(keys) == 2
        assert "data/file1.txt" in keys
        assert "data/file2.txt" in keys
    
    def test_get_metadata_existing(self, storage, temp_dir):
        """Test getting metadata for existing file."""
        key = "test.json"
        metadata = {"version": "1.0", "author": "test"}
        
        # Save metadata
        meta_path = temp_dir / "test.json.meta"
        with open(meta_path, "w") as f:
            json.dump(metadata, f)
        
        loaded_meta = storage.get_metadata(key)
        assert loaded_meta == metadata
    
    def test_get_metadata_nonexistent(self, storage):
        """Test getting metadata for file without metadata."""
        result = storage.get_metadata("test.json")
        assert result is None
    
    def test_copy(self, storage, temp_dir):
        """Test copying files."""
        # Create source file
        source_key = "source.txt"
        dest_key = "dest.txt"
        content = "test content"
        
        (temp_dir / source_key).write_text(content)
        
        storage.copy(source_key, dest_key)
        
        # Both files should exist
        assert (temp_dir / source_key).exists()
        assert (temp_dir / dest_key).exists()
        assert (temp_dir / dest_key).read_text() == content
    
    def test_move(self, storage, temp_dir):
        """Test moving files."""
        # Create source file
        source_key = "source.txt"
        dest_key = "dest.txt"
        content = "test content"
        
        (temp_dir / source_key).write_text(content)
        
        storage.move(source_key, dest_key)
        
        # Only destination should exist
        assert not (temp_dir / source_key).exists()
        assert (temp_dir / dest_key).exists()
        assert (temp_dir / dest_key).read_text() == content
    
    def test_get_size(self, storage, temp_dir):
        """Test getting file size."""
        key = "test.txt"
        content = "Hello, world!"
        
        (temp_dir / key).write_text(content)
        
        size = storage.get_size(key)
        assert size == len(content.encode())
    
    def test_get_last_modified(self, storage, temp_dir):
        """Test getting last modified time."""
        key = "test.txt"
        file_path = temp_dir / key
        file_path.touch()
        
        # Get last modified time
        mtime = storage.get_last_modified(key)
        
        # Should be close to current time
        import time
        current_time = time.time()
        assert abs(mtime - current_time) < 2  # Within 2 seconds


class MockMLXModule:
    """Mock MLX module for testing."""
    
    def __init__(self):
        self.state = {"layer1": {"weight": [1, 2, 3]}}
    
    def parameters(self):
        """Mock parameters method."""
        return self.state
    
    def update(self, params):
        """Mock update method."""
        self.state = params


class TestModelFileStorageAdapter:
    """Test model checkpoint adapter."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for testing."""
        temp_dir = tempfile.mkdtemp()
        yield Path(temp_dir)
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def checkpoint_adapter(self, temp_dir):
        """Create checkpoint adapter with temporary directory."""
        return ModelFileStorageAdapter(checkpoint_dir=temp_dir)
    
    def test_initialization(self, temp_dir):
        """Test checkpoint adapter initialization."""
        adapter = ModelFileStorageAdapter(checkpoint_dir=temp_dir)
        assert adapter.checkpoint_dir == temp_dir
        assert adapter.use_safetensors is True
    
    @patch('core.adapters.file_storage.save_file')
    def test_save_checkpoint_safetensors(self, mock_save_file, checkpoint_adapter, temp_dir):
        """Test saving checkpoint in safetensors format."""
        model = MockMLXModule()
        checkpoint_name = "checkpoint-100"
        metadata = {"step": 100, "epoch": 1}
        
        path = checkpoint_adapter.save_checkpoint(
            model, checkpoint_name, metadata
        )
        
        assert path == temp_dir / checkpoint_name
        assert path.exists()
        
        # Verify safetensors save was called
        mock_save_file.assert_called_once()
        
        # Verify metadata was saved
        meta_path = path / "metadata.json"
        assert meta_path.exists()
    
    def test_save_checkpoint_numpy(self, checkpoint_adapter, temp_dir):
        """Test saving checkpoint in numpy format."""
        checkpoint_adapter.use_safetensors = False
        model = MockMLXModule()
        checkpoint_name = "checkpoint-100"
        
        path = checkpoint_adapter.save_checkpoint(model, checkpoint_name)
        
        assert path == temp_dir / checkpoint_name
        assert path.exists()
        assert (path / "arrays.npz").exists()
    
    @patch('core.adapters.file_storage.safe_open')
    @patch('core.adapters.file_storage.mx')
    def test_load_checkpoint_safetensors(
        self, mock_mx, mock_safe_open, checkpoint_adapter, temp_dir
    ):
        """Test loading checkpoint from safetensors format."""
        # Create mock checkpoint directory
        checkpoint_name = "checkpoint-100"
        checkpoint_path = temp_dir / checkpoint_name
        checkpoint_path.mkdir()
        
        # Create weights.safetensors file
        (checkpoint_path / "weights.safetensors").touch()
        
        # Create metadata
        metadata = {"step": 100, "epoch": 1}
        with open(checkpoint_path / "metadata.json", "w") as f:
            json.dump(metadata, f)
        
        # Mock safetensors file
        mock_file = Mock()
        mock_file.keys.return_value = ["layer1.weight"]
        mock_file.get_tensor.return_value = [1, 2, 3]
        mock_safe_open.return_value.__enter__.return_value = mock_file
        
        # Mock mx.array
        mock_mx.array.return_value = [1, 2, 3]
        
        model = MockMLXModule()
        loaded_metadata = checkpoint_adapter.load_checkpoint(model, checkpoint_name)
        
        assert loaded_metadata == metadata
        mock_safe_open.assert_called_once()
    
    def test_load_checkpoint_not_found(self, checkpoint_adapter):
        """Test loading non-existent checkpoint."""
        model = MockMLXModule()
        
        with pytest.raises(ValueError, match="Checkpoint not found"):
            checkpoint_adapter.load_checkpoint(model, "nonexistent")
    
    def test_list_checkpoints(self, checkpoint_adapter, temp_dir):
        """Test listing available checkpoints."""
        # Create some checkpoint directories
        (temp_dir / "checkpoint-100").mkdir()
        (temp_dir / "checkpoint-200").mkdir()
        (temp_dir / "checkpoint-best").mkdir()
        (temp_dir / "not-a-checkpoint").touch()  # File, not directory
        
        checkpoints = checkpoint_adapter.list_checkpoints()
        
        assert len(checkpoints) == 3
        assert "checkpoint-100" in checkpoints
        assert "checkpoint-200" in checkpoints
        assert "checkpoint-best" in checkpoints
        assert "not-a-checkpoint" not in checkpoints
    
    def test_delete_checkpoint(self, checkpoint_adapter, temp_dir):
        """Test deleting a checkpoint."""
        # Create checkpoint directory
        checkpoint_name = "checkpoint-100"
        checkpoint_path = temp_dir / checkpoint_name
        checkpoint_path.mkdir()
        (checkpoint_path / "weights.safetensors").touch()
        
        checkpoint_adapter.delete_checkpoint(checkpoint_name)
        
        assert not checkpoint_path.exists()
    
    def test_get_latest_checkpoint(self, checkpoint_adapter, temp_dir):
        """Test getting latest checkpoint."""
        import time
        
        # Create checkpoints with different timestamps
        checkpoint1 = temp_dir / "checkpoint-100"
        checkpoint1.mkdir()
        
        time.sleep(0.1)  # Ensure different timestamps
        
        checkpoint2 = temp_dir / "checkpoint-200"
        checkpoint2.mkdir()
        
        latest = checkpoint_adapter.get_latest_checkpoint()
        assert latest == "checkpoint-200"
    
    def test_get_latest_checkpoint_none(self, checkpoint_adapter, temp_dir):
        """Test getting latest checkpoint when none exist."""
        latest = checkpoint_adapter.get_latest_checkpoint()
        assert latest is None