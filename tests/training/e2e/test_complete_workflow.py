"""End-to-end test for complete training workflow."""

import pytest
import tempfile
from pathlib import Path
import json
import mlx.core as mx
import mlx.nn as nn

from training.factory import create_trainer
from training.core.config import get_kaggle_competition_config, get_quick_test_config
from tests.training.fixtures.models import SimpleBinaryClassifier
from tests.training.fixtures.datasets import SyntheticDataLoader


class TestCompleteWorkflow:
    """Test complete training workflow from config to predictions."""
    
    def test_kaggle_workflow(self):
        """Test complete Kaggle competition workflow."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # 1. Setup configuration
            config = get_kaggle_competition_config()
            config.environment.output_dir = Path(tmpdir)
            config.training.num_epochs = 2  # Quick test
            config.training.eval_steps = 10
            config.training.logging_steps = 10
            config.training.save_best_only = True
            
            # Save config for reproducibility
            config_path = Path(tmpdir) / "config.json"
            config.save(config_path)
            
            # 2. Create model and data
            model = SimpleBinaryClassifier()
            train_loader = SyntheticDataLoader(
                num_samples=200,
                batch_size=config.data.batch_size,
                task_type="classification"
            )
            val_loader = SyntheticDataLoader(
                num_samples=100,
                batch_size=config.data.batch_size,
                task_type="classification"
            )
            test_loader = SyntheticDataLoader(
                num_samples=50,
                batch_size=config.data.batch_size,
                task_type="classification"
            )
            
            # 3. Create trainer using factory
            trainer = create_trainer(
                model=model,
                config=config,
                trainer_type="kaggle"
            )
            
            # 4. Train model
            result = trainer.train(
                train_dataloader=train_loader,
                val_dataloader=val_loader
            )
            
            # 5. Verify training completed
            assert result.total_epochs > 0
            assert result.best_model_path is not None
            assert result.best_model_path.exists()
            
            # 6. Load best model for predictions
            # (In real scenario, would load from checkpoint)
            predictions = trainer.predict(test_loader)
            
            assert predictions is not None
            assert len(predictions) == len(test_loader.dataset)
            
            # 7. Verify all outputs exist
            output_dir = config.environment.output_dir
            assert (output_dir / "checkpoints").exists()
            assert config_path.exists()
            
            # 8. Check metrics were tracked
            assert len(result.train_history) > 0
            assert len(result.val_history) > 0
            assert result.best_val_loss < float('inf')
            
    def test_resume_training(self):
        """Test resuming training from checkpoint."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = get_kaggle_competition_config()
            config.environment.output_dir = Path(tmpdir)
            config.training.num_epochs = 2
            config.training.save_strategy = "epoch"
            
            model = SimpleBinaryClassifier()
            train_loader = SyntheticDataLoader(num_samples=100, batch_size=32)
            val_loader = SyntheticDataLoader(num_samples=50, batch_size=32)
            
            # Initial training
            trainer = create_trainer(
                model=model,
                config=config,
                trainer_type="base"
            )
            
            result1 = trainer.train(
                train_dataloader=train_loader,
                val_dataloader=val_loader
            )
            
            # Get last checkpoint
            checkpoint_dir = config.environment.output_dir / "checkpoints"
            checkpoints = sorted(checkpoint_dir.glob("checkpoint-*"))
            assert len(checkpoints) > 0
            last_checkpoint = checkpoints[-1]
            
            # Resume training with more epochs
            config.training.num_epochs = 4
            new_model = SimpleBinaryClassifier()
            
            trainer2 = create_trainer(
                model=new_model,
                config=config,
                trainer_type="base"
            )
            
            result2 = trainer2.train(
                train_dataloader=train_loader,
                val_dataloader=val_loader,
                resume_from=last_checkpoint
            )
            
            # Should have trained additional epochs
            assert result2.total_epochs > result1.total_epochs
            assert len(result2.train_history) > len(result1.train_history)
            
    def test_different_trainer_types(self):
        """Test different trainer types from factory."""
        trainer_types = ["base", "kaggle"]
        
        for trainer_type in trainer_types:
            with tempfile.TemporaryDirectory() as tmpdir:
                config = get_quick_test_config()
                config.environment.output_dir = Path(tmpdir) / trainer_type
                
                model = SimpleBinaryClassifier()
                
                # Create trainer
                trainer = create_trainer(
                    model=model,
                    config=config,
                    trainer_type=trainer_type
                )
                
                # Verify correct type
                if trainer_type == "kaggle":
                    from training.kaggle.trainer import KaggleTrainer
                    assert isinstance(trainer, KaggleTrainer)
                else:
                    from training.core.base import BaseTrainer
                    assert isinstance(trainer, BaseTrainer)
                    
                # Quick training test
                train_loader = SyntheticDataLoader(num_samples=32, batch_size=8)
                result = trainer.train(train_dataloader=train_loader)
                
                assert result.total_epochs == config.training.num_epochs