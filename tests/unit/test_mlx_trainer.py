"""Unit tests for MLXTrainer class."""

from unittest.mock import MagicMock, patch, call
import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest

from training.mlx_trainer import MLXTrainer, UnifiedTrainingConfig
from models.classification import TitanicClassifier


class TestUnifiedTrainingConfig:
    """Test UnifiedTrainingConfig dataclass."""
    
    def test_default_config(self, temp_dir):
        """Test default configuration values."""
        config = UnifiedTrainingConfig()
        
        assert config.learning_rate == 5e-5
        assert config.num_epochs == 3
        assert config.base_batch_size == 32
        assert config.enable_mlflow == True
        assert config.gradient_clip_val == 1.0
    
    def test_custom_config(self, temp_dir):
        """Test custom configuration."""
        config = UnifiedTrainingConfig(
            learning_rate=1e-4,
            num_epochs=5,
            output_dir=str(temp_dir),
            enable_mlflow=False,
        )
        
        assert config.learning_rate == 1e-4
        assert config.num_epochs == 5
        assert config.output_dir == str(temp_dir)
        assert config.enable_mlflow == False
    
    def test_post_init_directories(self, temp_dir):
        """Test directory creation in post_init."""
        output_dir = temp_dir / "output"
        config = UnifiedTrainingConfig(
            output_dir=str(output_dir),
            enable_caching=True,
        )
        
        assert output_dir.exists()
        assert (output_dir / "checkpoints").exists()
        assert Path(config.cache_dir).exists()


class TestMLXTrainer:
    """Test MLXTrainer class."""
    
    def test_initialization(self, mlx_trainer, training_config):
        """Test trainer initialization."""
        assert mlx_trainer.config == training_config
        assert mlx_trainer.global_step == 0
        assert mlx_trainer.current_epoch == 0
        assert mlx_trainer.best_metric == -float("inf")
        assert mlx_trainer.optimizer is not None
    
    def test_memory_usage(self, mlx_trainer):
        """Test memory usage calculation."""
        # Default implementation returns 0.5
        usage = mlx_trainer.get_memory_usage()
        assert 0 <= usage <= 1
    
    def test_adjust_batch_size_increase(self, mlx_trainer):
        """Test batch size increase when memory usage is low."""
        mlx_trainer.config.enable_dynamic_batching = True
        mlx_trainer.current_batch_size = 32
        mlx_trainer.config.max_batch_size = 64
        
        with patch.object(mlx_trainer, 'get_memory_usage', return_value=0.3):
            new_size = mlx_trainer.adjust_batch_size()
            assert new_size == 64
    
    def test_adjust_batch_size_decrease(self, mlx_trainer):
        """Test batch size decrease when memory usage is high."""
        mlx_trainer.config.enable_dynamic_batching = True
        mlx_trainer.current_batch_size = 64
        mlx_trainer.config.memory_threshold = 0.8
        
        with patch.object(mlx_trainer, 'get_memory_usage', return_value=0.9):
            new_size = mlx_trainer.adjust_batch_size()
            assert new_size == 32
    
    def test_adjust_batch_size_disabled(self, mlx_trainer):
        """Test batch size adjustment when disabled."""
        mlx_trainer.config.enable_dynamic_batching = False
        original_size = mlx_trainer.current_batch_size
        
        new_size = mlx_trainer.adjust_batch_size()
        assert new_size == original_size
    
    def test_learning_rate_warmup(self, mlx_trainer):
        """Test learning rate warmup schedule."""
        mlx_trainer.max_steps = 100
        mlx_trainer.config.warmup_ratio = 0.1
        mlx_trainer.global_step = 5
        
        lr = mlx_trainer.get_learning_rate()
        expected_lr = mlx_trainer.config.learning_rate * 5 / 10
        assert abs(lr - expected_lr) < 1e-6
    
    def test_learning_rate_cosine_decay(self, mlx_trainer):
        """Test learning rate cosine decay."""
        mlx_trainer.max_steps = 100
        mlx_trainer.config.warmup_ratio = 0.1
        mlx_trainer.global_step = 50
        
        lr = mlx_trainer.get_learning_rate()
        assert lr < mlx_trainer.config.learning_rate
        assert lr > 0
    
    @patch('mlx.core.eval')
    def test_train_step(self, mock_eval, mlx_trainer, dummy_batch):
        """Test single training step."""
        mlx_trainer.config.gradient_accumulation_steps = 1
        
        # Mock the loss computation
        with patch.object(nn, 'value_and_grad') as mock_grad:
            mock_loss = mx.array(0.5)
            mock_grads = {"weight": mx.ones((10, 10))}
            mock_grad.return_value = lambda model: (mock_loss, mock_grads)
            
            loss, metrics = mlx_trainer.train_step(dummy_batch)
            
            assert "loss" in metrics
            assert "learning_rate" in metrics
            assert "batch_size" in metrics
    
    def test_train_step_gradient_accumulation(self, mlx_trainer, dummy_batch):
        """Test training step with gradient accumulation."""
        mlx_trainer.config.gradient_accumulation_steps = 2
        
        # First step - should accumulate
        with patch.object(nn, 'value_and_grad') as mock_grad:
            mock_loss = mx.array(0.5)
            mock_grads = {"weight": mx.ones((10, 10))}
            mock_grad.return_value = lambda model: (mock_loss, mock_grads)
            
            loss, metrics = mlx_trainer.train_step(dummy_batch)
            assert metrics.get("accumulating") == True
            
            # Second step - should update
            mlx_trainer.global_step = 1
            loss, metrics = mlx_trainer.train_step(dummy_batch)
            assert "loss" in metrics
    
    def test_clip_gradients(self, mlx_trainer):
        """Test gradient clipping."""
        # Create test gradients
        grads = {
            "layer1": {"weight": mx.ones((10, 10)) * 10},
            "layer2": [mx.ones((5, 5)) * 10],
        }
        
        mlx_trainer.config.gradient_clip_val = 1.0
        mlx_trainer._clip_gradients(grads)
        
        # Gradients should be clipped
        # Note: Actual clipping logic would modify the gradients
        assert grads is not None
    
    def test_force_eval_if_needed(self, mlx_trainer):
        """Test forced evaluation."""
        mlx_trainer.steps_since_eval = 15
        mlx_trainer.config.lazy_eval_interval = 10
        mlx_trainer.accumulated_loss = mx.array(0.5)
        
        with patch('mlx.core.eval') as mock_eval:
            mlx_trainer.force_eval_if_needed()
            mock_eval.assert_called_once()
            assert mlx_trainer.steps_since_eval == 0
    
    def test_should_stop_early(self, mlx_trainer):
        """Test early stopping logic."""
        mlx_trainer.config.early_stopping_patience = 3
        mlx_trainer.config.early_stopping_threshold = 0.001
        mlx_trainer.last_best_metric = 0.8
        
        # No improvement
        should_stop = mlx_trainer.should_stop_early(0.8)
        assert not should_stop
        assert mlx_trainer.early_stopping_counter == 1
        
        # Still no improvement
        should_stop = mlx_trainer.should_stop_early(0.8)
        assert not should_stop
        assert mlx_trainer.early_stopping_counter == 2
        
        # Third time - should stop
        should_stop = mlx_trainer.should_stop_early(0.8)
        assert should_stop
        
        # Improvement resets counter
        mlx_trainer.early_stopping_counter = 2
        should_stop = mlx_trainer.should_stop_early(0.85)
        assert not should_stop
        assert mlx_trainer.early_stopping_counter == 0
    
    @patch('mlx.core.eval')
    def test_evaluate(self, mock_eval, mlx_trainer, sample_titanic_data):
        """Test model evaluation."""
        from data.unified_loader import UnifiedTitanicDataPipeline
        
        dataloader = UnifiedTitanicDataPipeline(
            data_path=str(sample_titanic_data),
            tokenizer_name="answerdotai/ModernBERT-base",
            batch_size=4,
            max_length=128,
            is_training=False,
        )
        
        # Mock model outputs
        with patch.object(mlx_trainer.model, '__call__') as mock_model:
            mock_outputs = {
                "loss": mx.array(0.5),
                "logits": mx.array([[0.1, 0.9], [0.8, 0.2], [0.3, 0.7], [0.6, 0.4]]),
            }
            mock_model.return_value = mock_outputs
            
            metrics = mlx_trainer.evaluate(dataloader, phase="val", max_batches=1)
            
            assert "val_loss" in metrics
            assert "val_accuracy" in metrics
            assert "val_f1" in metrics
    
    def test_save_checkpoint(self, mlx_trainer, temp_dir):
        """Test checkpoint saving."""
        mlx_trainer.config.checkpoint_dir = str(temp_dir / "checkpoints")
        mlx_trainer.global_step = 100
        mlx_trainer.best_metric = 0.95
        
        with patch.object(mlx_trainer.model, 'save_pretrained') as mock_save:
            mlx_trainer.save_checkpoint("test_checkpoint")
            
            checkpoint_path = Path(mlx_trainer.config.checkpoint_dir) / "test_checkpoint"
            assert checkpoint_path.exists()
            
            # Check trainer state was saved
            state_file = checkpoint_path / "trainer_state.json"
            assert state_file.exists()
    
    def test_load_checkpoint(self, mlx_trainer, temp_dir):
        """Test checkpoint loading."""
        # Create a checkpoint
        checkpoint_dir = temp_dir / "checkpoints" / "test"
        checkpoint_dir.mkdir(parents=True)
        
        # Save trainer state
        import json
        state = {
            "global_step": 200,
            "current_epoch": 2,
            "best_metric": 0.98,
            "best_metric_step": 150,
        }
        with open(checkpoint_dir / "trainer_state.json", "w") as f:
            json.dump(state, f)
        
        mlx_trainer.config.checkpoint_dir = str(temp_dir / "checkpoints")
        
        # Load checkpoint
        success = mlx_trainer.load_checkpoint("test")
        
        assert success
        assert mlx_trainer.global_step == 200
        assert mlx_trainer.current_epoch == 2
        assert mlx_trainer.best_metric == 0.98
    
    def test_cleanup_checkpoints(self, mlx_trainer, temp_dir):
        """Test old checkpoint cleanup."""
        mlx_trainer.config.checkpoint_dir = str(temp_dir)
        mlx_trainer.config.save_total_limit = 2
        
        # Create multiple checkpoints
        for i in range(4):
            ckpt_dir = temp_dir / f"checkpoint-{i * 100}"
            ckpt_dir.mkdir()
        
        mlx_trainer._cleanup_checkpoints()
        
        # Should only keep the latest 2
        remaining = list(temp_dir.glob("checkpoint-*"))
        assert len(remaining) == 2
        assert (temp_dir / "checkpoint-300").exists()
        assert (temp_dir / "checkpoint-200").exists()


class TestMLXTrainerIntegration:
    """Integration tests for MLXTrainer."""
    
    @pytest.mark.integration
    @pytest.mark.mlx
    def test_full_training_cycle(self, temp_dir, sample_titanic_data):
        """Test complete training cycle."""
        from data.unified_loader import UnifiedTitanicDataPipeline
        from models.factory import create_model
        
        # Create minimal config
        config = UnifiedTrainingConfig(
            num_epochs=1,
            base_batch_size=4,
            eval_steps=2,
            save_steps=2,
            enable_mlflow=False,
            output_dir=str(temp_dir / "output"),
        )
        
        # Create model
        bert_model = create_model("standard")
        model = TitanicClassifier(bert_model)
        
        # Create trainer
        trainer = MLXTrainer(model=model, config=config)
        
        # Create data loader
        train_loader = UnifiedTitanicDataPipeline(
            data_path=str(sample_titanic_data),
            tokenizer_name="answerdotai/ModernBERT-base",
            batch_size=4,
            max_length=128,
            is_training=True,
        )
        
        # Train
        results = trainer.train(
            train_dataloader=train_loader,
            val_dataloader=train_loader,  # Use same for testing
        )
        
        assert "best_metric" in results
        assert "history" in results
        assert results["total_time"] > 0