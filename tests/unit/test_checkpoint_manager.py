"""Tests for checkpoint management system."""

import json
import shutil
import tempfile
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest

from training.checkpoint_manager import (
    AutoRecoveryManager,
    CheckpointConfig,
    CheckpointManager,
    CheckpointMetadata,
)


class DummyModel(nn.Module):
    """Dummy model for testing."""
    
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 10)
    
    def __call__(self, x):
        return self.linear(x)
    
    def save_pretrained(self, path: str):
        """Mock save method."""
        save_path = Path(path)
        save_path.mkdir(parents=True, exist_ok=True)
        # Create dummy model file
        model_file = save_path / "model.safetensors"
        model_file.write_text("dummy model weights")
    
    def load_pretrained(self, path: str):
        """Mock load method."""
        model_file = Path(path) / "model.safetensors"
        if not model_file.exists():
            raise FileNotFoundError(f"Model file not found: {model_file}")


class TestCheckpointMetadata:
    """Test CheckpointMetadata class."""
    
    def test_metadata_creation(self):
        """Test creating checkpoint metadata."""
        metadata = CheckpointMetadata(
            checkpoint_id="test_checkpoint",
            global_step=1000,
            epoch=5,
            timestamp=time.time(),
            metrics={"loss": 0.5, "accuracy": 0.9},
            is_best=True,
            validation_metric=0.5,
            training_time_seconds=3600.0,
            model_hash="abc123",
        )
        
        assert metadata.checkpoint_id == "test_checkpoint"
        assert metadata.global_step == 1000
        assert metadata.epoch == 5
        assert metadata.metrics["loss"] == 0.5
        assert metadata.is_best is True
    
    def test_metadata_serialization(self):
        """Test metadata to_dict and from_dict."""
        metadata = CheckpointMetadata(
            checkpoint_id="test",
            global_step=100,
            epoch=1,
            timestamp=1234567890.0,
            metrics={"loss": 0.3},
        )
        
        # Convert to dict
        data = metadata.to_dict()
        assert isinstance(data, dict)
        assert data["checkpoint_id"] == "test"
        assert data["global_step"] == 100
        
        # Convert back from dict
        metadata2 = CheckpointMetadata.from_dict(data)
        assert metadata2.checkpoint_id == metadata.checkpoint_id
        assert metadata2.global_step == metadata.global_step
        assert metadata2.metrics == metadata.metrics


class TestCheckpointConfig:
    """Test CheckpointConfig class."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = CheckpointConfig()
        
        assert config.checkpoint_dir == "./checkpoints"
        assert config.save_frequency == 1000
        assert config.keep_last_n == 5
        assert config.keep_best_n == 3
        assert config.save_optimizer_state is True
        assert config.auto_recover is True
    
    def test_custom_config(self):
        """Test custom configuration."""
        config = CheckpointConfig(
            checkpoint_dir="/tmp/checkpoints",
            save_frequency=500,
            keep_last_n=10,
            async_save=True,
        )
        
        assert config.checkpoint_dir == "/tmp/checkpoints"
        assert config.save_frequency == 500
        assert config.keep_last_n == 10
        assert config.async_save is True


class TestCheckpointManager:
    """Test CheckpointManager class."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for tests."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir
    
    @pytest.fixture
    def checkpoint_manager(self, temp_dir):
        """Create checkpoint manager for tests."""
        config = CheckpointConfig(
            checkpoint_dir=temp_dir,
            save_frequency=100,
            keep_last_n=3,
            keep_best_n=2,
        )
        return CheckpointManager(config)
    
    @pytest.fixture
    def dummy_model(self):
        """Create dummy model for tests."""
        return DummyModel()
    
    @pytest.fixture
    def dummy_optimizer(self):
        """Create dummy optimizer for tests."""
        optimizer = MagicMock()
        optimizer.state = {"step": mx.array([100])}
        return optimizer
    
    def test_initialization(self, checkpoint_manager, temp_dir):
        """Test checkpoint manager initialization."""
        assert checkpoint_manager.checkpoint_dir == Path(temp_dir)
        assert checkpoint_manager.config.save_frequency == 100
        assert len(checkpoint_manager.checkpoints) == 0
        assert len(checkpoint_manager.best_checkpoints) == 0
    
    def test_should_save_checkpoint(self, checkpoint_manager):
        """Test checkpoint save frequency logic."""
        # First checkpoint should save
        assert checkpoint_manager.should_save_checkpoint(100) is True
        
        # After saving, update last checkpoint step
        checkpoint_manager.last_checkpoint_step = 100
        
        # Should not save before frequency
        assert checkpoint_manager.should_save_checkpoint(150) is False
        
        # Should save at frequency
        assert checkpoint_manager.should_save_checkpoint(200) is True
    
    def test_save_checkpoint(self, checkpoint_manager, dummy_model, dummy_optimizer):
        """Test saving a checkpoint."""
        success, path = checkpoint_manager.save_checkpoint(
            model=dummy_model,
            optimizer=dummy_optimizer,
            global_step=100,
            epoch=1,
            metrics={"loss": 0.5, "accuracy": 0.85},
            training_state={"total_time": 1200.0},
            is_best=False,
        )
        
        assert success is True
        assert Path(path).exists()
        
        # Check checkpoint files
        checkpoint_path = Path(path)
        assert (checkpoint_path / "model.safetensors").exists()
        assert (checkpoint_path / "trainer_state.json").exists()
        
        # Check metadata
        assert len(checkpoint_manager.checkpoints) == 1
        assert checkpoint_manager.checkpoints[0].global_step == 100
        assert checkpoint_manager.checkpoints[0].metrics["loss"] == 0.5
    
    def test_save_best_checkpoint(self, checkpoint_manager, dummy_model, dummy_optimizer):
        """Test saving a best checkpoint."""
        success, path = checkpoint_manager.save_checkpoint(
            model=dummy_model,
            optimizer=dummy_optimizer,
            global_step=200,
            epoch=2,
            metrics={"loss": 0.3, "val_loss": 0.35},
            training_state={},
            is_best=True,
        )
        
        assert success is True
        assert len(checkpoint_manager.best_checkpoints) == 1
        assert checkpoint_manager.best_checkpoints[0].is_best is True
    
    def test_load_checkpoint(self, checkpoint_manager, dummy_model, dummy_optimizer):
        """Test loading a checkpoint."""
        # First save a checkpoint
        success, save_path = checkpoint_manager.save_checkpoint(
            model=dummy_model,
            optimizer=dummy_optimizer,
            global_step=300,
            epoch=3,
            metrics={"loss": 0.2},
            training_state={"custom_state": "test"},
        )
        
        assert success is True
        
        # Create new model and optimizer
        new_model = DummyModel()
        new_optimizer = MagicMock()
        
        # Load checkpoint
        success, state = checkpoint_manager.load_checkpoint(
            checkpoint_path=save_path,
            model=new_model,
            optimizer=new_optimizer,
        )
        
        assert success is True
        assert state["global_step"] == 300
        assert state["epoch"] == 3
        assert state["metrics"]["loss"] == 0.2
        assert state["training_state"]["custom_state"] == "test"
    
    def test_get_latest_checkpoint(self, checkpoint_manager, dummy_model, dummy_optimizer):
        """Test getting latest checkpoint."""
        # Save multiple checkpoints
        for step in [100, 200, 300]:
            checkpoint_manager.save_checkpoint(
                model=dummy_model,
                optimizer=dummy_optimizer,
                global_step=step,
                epoch=step // 100,
                metrics={"loss": 1.0 / step},
                training_state={},
            )
        
        latest = checkpoint_manager.get_latest_checkpoint()
        assert latest is not None
        assert "checkpoint_step_00000300" in latest
    
    def test_get_best_checkpoint(self, checkpoint_manager, dummy_model, dummy_optimizer):
        """Test getting best checkpoint."""
        # Save regular checkpoint
        checkpoint_manager.save_checkpoint(
            model=dummy_model,
            optimizer=dummy_optimizer,
            global_step=100,
            epoch=1,
            metrics={"loss": 0.5},
            training_state={},
            is_best=False,
        )
        
        # Save best checkpoint
        checkpoint_manager.save_checkpoint(
            model=dummy_model,
            optimizer=dummy_optimizer,
            global_step=200,
            epoch=2,
            metrics={"loss": 0.2},
            training_state={},
            is_best=True,
        )
        
        best = checkpoint_manager.get_best_checkpoint()
        assert best is not None
        assert "checkpoint_step_00000200" in best
    
    def test_cleanup_old_checkpoints(self, checkpoint_manager, dummy_model, dummy_optimizer):
        """Test checkpoint cleanup."""
        # Save more checkpoints than keep_last_n (3)
        for step in range(100, 600, 100):
            checkpoint_manager.save_checkpoint(
                model=dummy_model,
                optimizer=dummy_optimizer,
                global_step=step,
                epoch=step // 100,
                metrics={"loss": 1.0 / step},
                training_state={},
            )
        
        # Should only keep last 3
        assert len(checkpoint_manager.checkpoints) == 3
        
        # Check that oldest were removed
        remaining_steps = [cp.global_step for cp in checkpoint_manager.checkpoints]
        assert 300 in remaining_steps
        assert 400 in remaining_steps
        assert 500 in remaining_steps
        assert 100 not in remaining_steps
        assert 200 not in remaining_steps
    
    def test_atomic_saves(self, temp_dir, dummy_model, dummy_optimizer):
        """Test atomic checkpoint saves."""
        config = CheckpointConfig(
            checkpoint_dir=temp_dir,
            atomic_saves=True,
        )
        manager = CheckpointManager(config)
        
        # Save checkpoint
        success, path = manager.save_checkpoint(
            model=dummy_model,
            optimizer=dummy_optimizer,
            global_step=100,
            epoch=1,
            metrics={"loss": 0.5},
            training_state={},
        )
        
        assert success is True
        assert Path(path).exists()
        # Temporary directory should not exist
        assert not any(p.name.startswith(".tmp_") for p in Path(temp_dir).iterdir())
    
    def test_save_with_random_state(self, temp_dir, dummy_model, dummy_optimizer):
        """Test saving random state."""
        config = CheckpointConfig(
            checkpoint_dir=temp_dir,
            save_random_state=True,
        )
        manager = CheckpointManager(config)
        
        # Set random state
        np.random.seed(42)
        
        success, path = manager.save_checkpoint(
            model=dummy_model,
            optimizer=dummy_optimizer,
            global_step=100,
            epoch=1,
            metrics={},
            training_state={},
        )
        
        assert success is True
        
        # Load state file and check random state
        state_file = Path(path) / "trainer_state.json"
        with open(state_file) as f:
            state_data = json.load(f)
        
        assert "random_state" in state_data
        assert "numpy_state" in state_data["random_state"]
    
    def test_list_checkpoints(self, checkpoint_manager, dummy_model, dummy_optimizer):
        """Test listing checkpoints."""
        # Save checkpoints
        for step in [100, 200]:
            checkpoint_manager.save_checkpoint(
                model=dummy_model,
                optimizer=dummy_optimizer,
                global_step=step,
                epoch=step // 100,
                metrics={"loss": 1.0 / step},
                training_state={},
            )
        
        checkpoints = checkpoint_manager.list_checkpoints()
        assert len(checkpoints) == 2
        
        # Check checkpoint info
        for cp in checkpoints:
            assert "path" in cp
            assert "size_mb" in cp
            assert "global_step" in cp
            assert cp["size_mb"] > 0


class TestAutoRecoveryManager:
    """Test AutoRecoveryManager class."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for tests."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir
    
    @pytest.fixture
    def checkpoint_manager(self, temp_dir):
        """Create checkpoint manager for tests."""
        config = CheckpointConfig(checkpoint_dir=temp_dir)
        return CheckpointManager(config)
    
    @pytest.fixture
    def recovery_manager(self, checkpoint_manager):
        """Create recovery manager for tests."""
        return AutoRecoveryManager(
            checkpoint_manager=checkpoint_manager,
            max_attempts=3,
            recovery_window_seconds=300.0,
        )
    
    @pytest.fixture
    def dummy_model(self):
        """Create dummy model for tests."""
        return DummyModel()
    
    @pytest.fixture
    def dummy_optimizer(self):
        """Create dummy optimizer for tests."""
        return MagicMock()
    
    def test_initialization(self, recovery_manager):
        """Test recovery manager initialization."""
        assert recovery_manager.max_attempts == 3
        assert recovery_manager.recovery_window_seconds == 300.0
        assert recovery_manager.recovery_attempts == 0
        assert len(recovery_manager.recovery_history) == 0
    
    def test_check_and_recover_no_checkpoint(
        self, recovery_manager, dummy_model, dummy_optimizer
    ):
        """Test recovery when no checkpoint exists."""
        recovered, state = recovery_manager.check_and_recover(
            model=dummy_model,
            optimizer=dummy_optimizer,
            current_step=1000,
        )
        
        assert recovered is False
        assert state == {}
    
    def test_check_and_recover_with_checkpoint(
        self, checkpoint_manager, recovery_manager, dummy_model, dummy_optimizer
    ):
        """Test successful recovery from checkpoint."""
        # Save a checkpoint
        checkpoint_manager.save_checkpoint(
            model=dummy_model,
            optimizer=dummy_optimizer,
            global_step=500,
            epoch=5,
            metrics={"loss": 0.3},
            training_state={"custom": "data"},
        )
        
        # Attempt recovery
        recovered, state = recovery_manager.check_and_recover(
            model=dummy_model,
            optimizer=dummy_optimizer,
            current_step=1000,
        )
        
        assert recovered is True
        assert state["global_step"] == 500
        assert state["epoch"] == 5
        assert recovery_manager.recovery_attempts == 1
        assert len(recovery_manager.recovery_history) == 1
    
    def test_max_recovery_attempts(
        self, checkpoint_manager, recovery_manager, dummy_model, dummy_optimizer
    ):
        """Test max recovery attempts limit."""
        # Save a checkpoint
        checkpoint_manager.save_checkpoint(
            model=dummy_model,
            optimizer=dummy_optimizer,
            global_step=100,
            epoch=1,
            metrics={},
            training_state={},
        )
        
        # Exhaust recovery attempts
        recovery_manager.recovery_attempts = 3
        
        recovered, state = recovery_manager.check_and_recover(
            model=dummy_model,
            optimizer=dummy_optimizer,
            current_step=200,
        )
        
        assert recovered is False
        assert state == {}
    
    def test_recovery_window_reset(
        self, checkpoint_manager, recovery_manager, dummy_model, dummy_optimizer
    ):
        """Test recovery attempts reset after window expires."""
        # Set past recovery
        recovery_manager.recovery_attempts = 2
        recovery_manager.last_recovery_time = time.time() - 400  # Outside window
        
        # Save checkpoint
        checkpoint_manager.save_checkpoint(
            model=dummy_model,
            optimizer=dummy_optimizer,
            global_step=100,
            epoch=1,
            metrics={},
            training_state={},
        )
        
        # Attempt recovery
        recovered, state = recovery_manager.check_and_recover(
            model=dummy_model,
            optimizer=dummy_optimizer,
            current_step=200,
        )
        
        assert recovered is True
        assert recovery_manager.recovery_attempts == 1  # Reset and incremented
    
    def test_get_recovery_stats(self, recovery_manager):
        """Test getting recovery statistics."""
        # Add some recovery history
        recovery_manager.recovery_history = [
            {
                "timestamp": time.time(),
                "checkpoint": "/path/to/checkpoint",
                "recovered_step": 100,
                "current_step": 200,
                "attempt": 1,
            }
        ]
        recovery_manager.recovery_attempts = 1
        
        stats = recovery_manager.get_recovery_stats()
        
        assert stats["total_recoveries"] == 1
        assert stats["recovery_attempts"] == 1
        assert len(stats["recovery_history"]) == 1
        assert "last_recovery_time" in stats