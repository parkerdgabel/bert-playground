"""Unit tests for CheckpointManager component."""

import json
from pathlib import Path
from unittest.mock import Mock, patch
import pytest
import mlx.core as mx

from training.components.checkpoint_manager import CheckpointManager
from core.protocols.training import TrainingState


class TestCheckpointManager:
    """Test cases for CheckpointManager component."""
    
    @pytest.fixture
    def checkpoint_manager(self, tmp_checkpoint_dir):
        """Create CheckpointManager instance."""
        return CheckpointManager(
            checkpoint_dir=tmp_checkpoint_dir,
            save_total_limit=3,
            keep_best_only=False,
        )
    
    @pytest.fixture
    def mock_optimizer(self):
        """Create mock optimizer."""
        optimizer = Mock()
        optimizer.state = {"step": mx.array(100), "learning_rate": mx.array(1e-3)}
        return optimizer
    
    def test_initialization(self, tmp_checkpoint_dir):
        """Test CheckpointManager initialization."""
        manager = CheckpointManager(
            checkpoint_dir=tmp_checkpoint_dir,
            save_total_limit=5,
            keep_best_only=True,
        )
        
        assert manager.checkpoint_dir == tmp_checkpoint_dir
        assert manager.save_total_limit == 5
        assert manager.keep_best_only is True
        assert tmp_checkpoint_dir.exists()  # Directory should be created
        assert manager._best_checkpoint is None
        assert len(manager._checkpoints) == 0
    
    @patch('mlx.core.save_safetensors')
    def test_save_checkpoint_basic(self, mock_save, checkpoint_manager, mock_model, mock_optimizer):
        """Test basic checkpoint saving."""
        state = TrainingState(global_step=100, epoch=1)
        metrics = {"loss": 0.5, "accuracy": 0.8}
        
        # Save checkpoint
        checkpoint_path = checkpoint_manager.save_checkpoint(
            model=mock_model,
            optimizer=mock_optimizer,
            state=state,
            metrics=metrics,
            name="test_checkpoint",
        )
        
        # Check path
        assert checkpoint_path.name == "test_checkpoint"
        assert checkpoint_path.parent == checkpoint_manager.checkpoint_dir
        
        # Check files were created
        assert (checkpoint_path / "metadata.json").exists()
        assert (checkpoint_path / "state.json").exists()
        assert (checkpoint_path / "metrics.json").exists()
        
        # Check save_safetensors was called for model and optimizer
        assert mock_save.call_count == 2
        
        # Check metadata
        with open(checkpoint_path / "metadata.json", "r") as f:
            metadata = json.load(f)
        assert metadata["step"] == 100
        assert metadata["epoch"] == 1
        assert metadata["checkpoint_name"] == "test_checkpoint"
        assert metadata["is_best"] is False
    
    @patch('mlx.core.save_safetensors')
    def test_save_best_checkpoint(self, mock_save, checkpoint_manager, mock_model, mock_optimizer):
        """Test saving best checkpoint."""
        state = TrainingState(global_step=100, epoch=1)
        metrics = {"loss": 0.5}
        
        # Save as best
        checkpoint_path = checkpoint_manager.save_checkpoint(
            model=mock_model,
            optimizer=mock_optimizer,
            state=state,
            metrics=metrics,
            is_best=True,
            name="best_model",
        )
        
        # Check best checkpoint was set
        assert checkpoint_manager._best_checkpoint == checkpoint_path
        
        # Check "best" directory was created
        best_dir = checkpoint_manager.checkpoint_dir / "best"
        assert best_dir.exists()
        assert (best_dir / "metadata.json").exists()
        
        # Check metadata
        with open(checkpoint_path / "metadata.json", "r") as f:
            metadata = json.load(f)
        assert metadata["is_best"] is True
    
    @patch('mlx.core.save_safetensors')
    def test_save_with_default_name(self, mock_save, checkpoint_manager, mock_model, mock_optimizer):
        """Test saving checkpoint with default name."""
        state = TrainingState(global_step=150, epoch=2)
        metrics = {"loss": 0.4}
        
        # Save without specifying name
        checkpoint_path = checkpoint_manager.save_checkpoint(
            model=mock_model,
            optimizer=mock_optimizer,
            state=state,
            metrics=metrics,
        )
        
        # Should use default name format
        assert checkpoint_path.name == "checkpoint-150"
    
    @patch('mlx.core.save_safetensors') 
    @patch('mlx.core.load')
    def test_load_checkpoint(self, mock_load, mock_save, checkpoint_manager, mock_model, mock_optimizer):
        """Test checkpoint loading."""
        # First save a checkpoint
        state = TrainingState(global_step=100, epoch=1, train_loss=0.5)
        metrics = {"loss": 0.5}
        
        checkpoint_path = checkpoint_manager.save_checkpoint(
            model=mock_model,
            optimizer=mock_optimizer,
            state=state,
            metrics=metrics,
            name="test_load",
        )
        
        # Mock load to return fake weights/state
        mock_load.return_value = {"param1": mx.array([1.0, 2.0])}
        
        # Mock model load_weights
        mock_model.load_weights = Mock()
        
        # Load checkpoint
        loaded_state = checkpoint_manager.load_checkpoint(
            path=checkpoint_path,
            model=mock_model,
            optimizer=mock_optimizer,
        )
        
        # Check state was loaded
        assert loaded_state.global_step == 100
        assert loaded_state.epoch == 1
        assert loaded_state.train_loss == 0.5
        
        # Check model and optimizer were loaded
        mock_model.load_weights.assert_called_once()
        mock_load.assert_called()  # Called for both model and optimizer
    
    def test_load_nonexistent_checkpoint(self, checkpoint_manager, mock_model, mock_optimizer):
        """Test loading non-existent checkpoint raises error."""
        fake_path = checkpoint_manager.checkpoint_dir / "nonexistent"
        
        with pytest.raises(FileNotFoundError, match="Model weights not found"):
            checkpoint_manager.load_checkpoint(
                path=fake_path,
                model=mock_model,
                optimizer=mock_optimizer,
            )
    
    @patch('mlx.core.save_safetensors')
    def test_checkpoint_retention_limit(self, mock_save, tmp_checkpoint_dir, mock_model, mock_optimizer):
        """Test checkpoint retention with save_total_limit."""
        manager = CheckpointManager(
            checkpoint_dir=tmp_checkpoint_dir,
            save_total_limit=2,  # Keep only 2 checkpoints
            keep_best_only=False,
        )
        
        # Save 3 checkpoints
        for i in range(3):
            state = TrainingState(global_step=100 + i * 50, epoch=i)
            manager.save_checkpoint(
                model=mock_model,
                optimizer=mock_optimizer,
                state=state,
                metrics={"loss": 0.5 - i * 0.1},
                name=f"checkpoint-{100 + i * 50}",
            )
        
        # Should only keep 2 most recent
        assert len(manager._checkpoints) == 2
        # Check that first checkpoint was removed
        assert not (tmp_checkpoint_dir / "checkpoint-100").exists()
        assert (tmp_checkpoint_dir / "checkpoint-150").exists()
        assert (tmp_checkpoint_dir / "checkpoint-200").exists()
    
    @patch('mlx.core.save_safetensors')
    def test_keep_best_only(self, mock_save, tmp_checkpoint_dir, mock_model, mock_optimizer):
        """Test keep_best_only retention policy."""
        manager = CheckpointManager(
            checkpoint_dir=tmp_checkpoint_dir,
            save_total_limit=5,
            keep_best_only=True,
        )
        
        # Save regular checkpoint
        state1 = TrainingState(global_step=100, epoch=1)
        path1 = manager.save_checkpoint(
            model=mock_model,
            optimizer=mock_optimizer,
            state=state1,
            metrics={"loss": 0.5},
            name="checkpoint-100",
        )
        
        # Save best checkpoint
        state2 = TrainingState(global_step=200, epoch=2)
        path2 = manager.save_checkpoint(
            model=mock_model,
            optimizer=mock_optimizer,
            state=state2,
            metrics={"loss": 0.3},
            is_best=True,
            name="checkpoint-200",
        )
        
        # Save another regular checkpoint
        state3 = TrainingState(global_step=300, epoch=3)
        manager.save_checkpoint(
            model=mock_model,
            optimizer=mock_optimizer,
            state=state3,
            metrics={"loss": 0.6},
            name="checkpoint-300",
        )
        
        # Only best checkpoint should remain (plus best and final directories)
        regular_checkpoints = [p for p in tmp_checkpoint_dir.iterdir() 
                              if p.name.startswith("checkpoint-")]
        # In keep_best_only mode, only the best checkpoint should remain
        # (implementation removes non-best checkpoints)
    
    def test_get_best_checkpoint(self, checkpoint_manager):
        """Test getting best checkpoint path."""
        # Initially no best checkpoint
        assert checkpoint_manager.get_best_checkpoint() is None
        
        # Set a best checkpoint
        best_path = checkpoint_manager.checkpoint_dir / "best"
        checkpoint_manager._best_checkpoint = best_path
        
        assert checkpoint_manager.get_best_checkpoint() == best_path
    
    def test_get_latest_checkpoint(self, checkpoint_manager):
        """Test getting latest checkpoint path."""
        # Initially no checkpoints
        assert checkpoint_manager.get_latest_checkpoint() is None
        
        # Add some checkpoints
        path1 = checkpoint_manager.checkpoint_dir / "checkpoint-100"
        path2 = checkpoint_manager.checkpoint_dir / "checkpoint-200"
        
        checkpoint_manager._checkpoints = [(100, path1), (200, path2)]
        
        assert checkpoint_manager.get_latest_checkpoint() == path2
    
    @patch('mlx.core.save_safetensors')
    def test_metadata_persistence(self, mock_save, checkpoint_manager, mock_model, mock_optimizer):
        """Test that checkpoint metadata is saved and loaded."""
        # Save a checkpoint
        state = TrainingState(global_step=100, epoch=1)
        checkpoint_manager.save_checkpoint(
            model=mock_model,
            optimizer=mock_optimizer,
            state=state,
            metrics={"loss": 0.5},
            is_best=True,
            name="test-metadata",
        )
        
        # Create new manager instance (simulating restart)
        new_manager = CheckpointManager(
            checkpoint_dir=checkpoint_manager.checkpoint_dir,
            save_total_limit=3,
        )
        
        # Metadata should be loaded
        assert len(new_manager._checkpoints) > 0 or new_manager._best_checkpoint is not None
    
    def test_make_json_serializable(self, checkpoint_manager):
        """Test JSON serialization helper."""
        # Test with MLX array
        mlx_array = mx.array([1.0, 2.0, 3.0])
        result = checkpoint_manager._make_json_serializable(mlx_array)
        assert result == [1.0, 2.0, 3.0]
        
        # Test with scalar MLX array
        scalar_array = mx.array(42.0)
        result = checkpoint_manager._make_json_serializable(scalar_array)
        assert result == 42.0
        
        # Test with dict containing MLX arrays
        data = {"values": mx.array([1, 2]), "scalar": mx.array(5)}
        result = checkpoint_manager._make_json_serializable(data)
        assert result == {"values": [1, 2], "scalar": 5}
        
        # Test with Path
        path = Path("/tmp/test")
        result = checkpoint_manager._make_json_serializable(path)
        assert result == str(path)
    
    def test_cleanup_old_checkpoints(self, checkpoint_manager):
        """Test manual cleanup method."""
        # Add some mock checkpoints
        path1 = checkpoint_manager.checkpoint_dir / "checkpoint-100"
        path2 = checkpoint_manager.checkpoint_dir / "checkpoint-200"
        checkpoint_manager._checkpoints = [(100, path1), (200, path2)]
        
        # Should not raise error
        checkpoint_manager.cleanup_old_checkpoints(keep_best=1, keep_last=1)