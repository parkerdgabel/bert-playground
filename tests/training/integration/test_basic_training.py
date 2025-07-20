"""Basic integration test for training functionality."""

import pytest
import tempfile
from pathlib import Path
import mlx.core as mx
import mlx.nn as nn

from training.core.config import BaseTrainerConfig, get_quick_test_config
from training.core.base import BaseTrainer
from training.callbacks.base import CallbackList
from training.callbacks.progress import ProgressBar
from tests.training.fixtures.models import SimpleBinaryClassifier
from tests.training.fixtures.datasets import SyntheticDataLoader


class TestBasicTraining:
    """Test basic training functionality."""
    
    def test_simple_training_run(self):
        """Test a simple training run."""
        # Create temporary directory for outputs
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create configuration
            config = get_quick_test_config()
            config.environment.output_dir = Path(tmpdir)
            config.training.num_epochs = 2
            config.training.logging_steps = 5
            
            # Create model and data
            model = SimpleBinaryClassifier()
            train_loader = SyntheticDataLoader(
                num_samples=100,
                batch_size=config.data.batch_size,
                task_type="classification"
            )
            val_loader = SyntheticDataLoader(
                num_samples=50,
                batch_size=config.data.batch_size,
                task_type="classification"
            )
            
            # Create trainer
            trainer = BaseTrainer(
                model=model,
                config=config,
                callbacks=[ProgressBar()]
            )
            
            # Run training
            result = trainer.train(
                train_dataloader=train_loader,
                val_dataloader=val_loader
            )
            
            # Check results
            assert result.total_epochs == 2
            assert result.final_train_loss > 0
            assert result.final_val_loss > 0
            assert len(result.train_history) > 0
            assert len(result.val_history) > 0
            
    def test_training_with_early_stopping(self):
        """Test training with early stopping."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create configuration with early stopping
            config = get_quick_test_config()
            config.environment.output_dir = Path(tmpdir)
            config.training.num_epochs = 10
            config.training.early_stopping = True
            config.training.early_stopping_patience = 2
            config.training.eval_strategy = "epoch"
            
            # Create model that doesn't improve
            model = SimpleBinaryClassifier()
            train_loader = SyntheticDataLoader(
                num_samples=100,
                batch_size=config.data.batch_size,
            )
            val_loader = SyntheticDataLoader(
                num_samples=50,
                batch_size=config.data.batch_size,
            )
            
            # Create trainer
            trainer = BaseTrainer(model=model, config=config)
            
            # Run training
            result = trainer.train(
                train_dataloader=train_loader,
                val_dataloader=val_loader
            )
            
            # Should stop early
            assert result.early_stopped
            assert result.total_epochs < 10
            assert "early_stopping" in result.stop_reason.lower()
            
    def test_checkpoint_saving(self):
        """Test checkpoint saving during training."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Configure for checkpoint saving
            config = get_quick_test_config()
            config.environment.output_dir = Path(tmpdir)
            config.training.save_strategy = "epoch"
            config.training.num_epochs = 3
            
            # Create model and data
            model = SimpleBinaryClassifier()
            train_loader = SyntheticDataLoader(num_samples=50, batch_size=8)
            
            # Train
            trainer = BaseTrainer(model=model, config=config)
            result = trainer.train(train_dataloader=train_loader)
            
            # Check checkpoints exist
            checkpoint_dir = config.environment.output_dir / "checkpoints"
            assert checkpoint_dir.exists()
            checkpoints = list(checkpoint_dir.glob("checkpoint-*"))
            assert len(checkpoints) > 0
            
            # Check final model
            if result.final_model_path:
                assert result.final_model_path.exists()
                
    def test_training_with_gradient_accumulation(self):
        """Test training with gradient accumulation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = get_quick_test_config()
            config.environment.output_dir = Path(tmpdir)
            config.training.gradient_accumulation_steps = 4
            config.data.batch_size = 4  # Small batch size
            
            model = SimpleBinaryClassifier()
            train_loader = SyntheticDataLoader(
                num_samples=64,
                batch_size=config.data.batch_size
            )
            
            trainer = BaseTrainer(model=model, config=config)
            result = trainer.train(train_dataloader=train_loader)
            
            # Training should complete successfully
            assert result.total_epochs == config.training.num_epochs
            assert result.final_train_loss > 0
            
            # Effective batch size should be 16 (4 * 4)
            assert config.effective_batch_size == 16