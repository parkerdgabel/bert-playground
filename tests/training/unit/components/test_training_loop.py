"""Unit tests for TrainingLoop component."""

import pytest
import mlx.core as mx
from unittest.mock import Mock, MagicMock

from training.components.training_loop import TrainingLoop
from training.core.config import TrainingConfig
from training.core.optimization import GradientAccumulator
from core.protocols.training import TrainingState


class TestTrainingLoop:
    """Test cases for TrainingLoop component."""
    
    @pytest.fixture
    def training_config(self):
        """Create a training configuration."""
        return TrainingConfig(
            gradient_accumulation_steps=1,
            mixed_precision=False,
            label_smoothing=0.0,
        )
    
    @pytest.fixture
    def mock_optimizer(self):
        """Create mock optimizer."""
        optimizer = Mock()
        optimizer.learning_rate = 1e-3
        optimizer.update = Mock()
        optimizer.state = {}
        return optimizer
    
    @pytest.fixture
    def gradient_accumulator(self, training_config):
        """Create gradient accumulator."""
        return GradientAccumulator(training_config.gradient_accumulation_steps)
    
    @pytest.fixture
    def training_loop(self, mock_model, mock_optimizer, training_config, gradient_accumulator):
        """Create TrainingLoop instance."""
        return TrainingLoop(
            model=mock_model,
            optimizer=mock_optimizer,
            config=training_config,
            gradient_accumulator=gradient_accumulator,
        )
    
    def test_initialization(self, training_loop, mock_model, mock_optimizer, training_config):
        """Test TrainingLoop initialization."""
        assert training_loop.model is mock_model
        assert training_loop.optimizer is mock_optimizer
        assert training_loop.config is training_config
        assert training_loop.gradient_accumulator is not None
        assert not training_loop._use_compiled
    
    def test_train_batch_basic(self, training_loop):
        """Test basic batch training."""
        # Create test batch
        batch = {
            "input": mx.random.normal((4, 10)),
            "labels": mx.random.randint(0, 2, (4,))
        }
        
        # Train batch
        loss, metrics = training_loop.train_batch(batch)
        
        # Check results
        assert isinstance(loss, mx.array)
        assert loss.size == 1  # Scalar loss
        assert "learning_rate" in metrics
        assert "loss" in metrics
        assert metrics["learning_rate"] == training_loop.optimizer.learning_rate
    
    def test_train_batch_with_metadata(self, training_loop):
        """Test batch training ignores metadata."""
        batch = {
            "input": mx.random.normal((4, 10)),
            "labels": mx.random.randint(0, 2, (4,)),
            "metadata": {"some": "data"}
        }
        
        # Should not raise error even with metadata
        loss, metrics = training_loop.train_batch(batch)
        assert isinstance(loss, mx.array)
        assert "learning_rate" in metrics
    
    def test_train_batch_with_gradient_clipping(self, mock_model, mock_optimizer, gradient_accumulator):
        """Test batch training with gradient clipping."""
        config = TrainingConfig(
            gradient_accumulation_steps=1,
            mixed_precision=False,
        )
        
        training_loop = TrainingLoop(
            model=mock_model,
            optimizer=mock_optimizer,
            config=config,
            gradient_accumulator=gradient_accumulator,
            max_grad_norm=0.5,  # Enable clipping
        )
        
        batch = {
            "input": mx.random.normal((4, 10)),
            "labels": mx.random.randint(0, 2, (4,))
        }
        
        loss, metrics = training_loop.train_batch(batch)
        assert "grad_norm" in metrics
        
    def test_train_batch_with_mixed_precision(self, mock_model, mock_optimizer, gradient_accumulator):
        """Test batch training with mixed precision."""
        config = TrainingConfig(
            gradient_accumulation_steps=1,
            mixed_precision=True,
        )
        
        training_loop = TrainingLoop(
            model=mock_model,
            optimizer=mock_optimizer,
            config=config,
            gradient_accumulator=gradient_accumulator,
        )
        
        # Use float32 inputs that should be converted to bfloat16
        batch = {
            "input": mx.random.normal((4, 10), dtype=mx.float32),
            "labels": mx.random.randint(0, 2, (4,))
        }
        
        loss, metrics = training_loop.train_batch(batch)
        assert isinstance(loss, mx.array)
    
    def test_train_batch_with_label_smoothing(self, mock_model, mock_optimizer, gradient_accumulator):
        """Test batch training with label smoothing."""
        config = TrainingConfig(
            gradient_accumulation_steps=1,
            mixed_precision=False,
            label_smoothing=0.1,
        )
        
        training_loop = TrainingLoop(
            model=mock_model,
            optimizer=mock_optimizer,
            config=config,
            gradient_accumulator=gradient_accumulator,
        )
        
        batch = {
            "input": mx.random.normal((4, 10)),
            "labels": mx.random.randint(0, 2, (4,))
        }
        
        loss, metrics = training_loop.train_batch(batch)
        assert isinstance(loss, mx.array)
    
    def test_train_epoch(self, training_loop, mock_train_loader):
        """Test training for one epoch."""
        state = TrainingState()
        
        # Train epoch
        metrics = training_loop.train_epoch(mock_train_loader, state)
        
        # Check results
        assert "loss" in metrics
        assert isinstance(metrics["loss"], float)
        assert state.global_step > 0
        assert state.samples_seen > 0
    
    def test_train_epoch_with_callbacks(self, training_loop, mock_train_loader):
        """Test training epoch with callbacks."""
        state = TrainingState()
        
        # Create mock callback
        callback = Mock()
        callback.on_batch_begin = Mock()
        callback.on_batch_end = Mock()
        
        # Train epoch with callback
        metrics = training_loop.train_epoch(mock_train_loader, state, callbacks=[callback])
        
        # Check callback was called
        assert callback.on_batch_begin.call_count > 0
        assert callback.on_batch_end.call_count > 0
        
        # Check results
        assert "loss" in metrics
        assert isinstance(metrics["loss"], float)
    
    def test_set_compiled_step(self, training_loop):
        """Test setting compiled training step."""
        # Create mock compiled step
        compiled_step = Mock(return_value=(mx.array(1.0), {"loss": mx.array(1.0)}))
        
        training_loop.set_compiled_step(compiled_step)
        
        assert training_loop._use_compiled
        assert training_loop._compiled_train_step is compiled_step
        
        # Test that compiled step is used
        batch = {"input": mx.random.normal((4, 10)), "labels": mx.random.randint(0, 2, (4,))}
        training_loop.train_batch(batch)
        
        compiled_step.assert_called_once_with(batch)
    
    def test_reset_gradients(self, training_loop):
        """Test resetting gradient accumulator."""
        # No exception should be raised
        training_loop.reset_gradients()
        
        # Verify accumulator was reset
        assert training_loop.gradient_accumulator.step_count == 0
    
    def test_gradient_accumulation(self, mock_model, mock_optimizer):
        """Test gradient accumulation over multiple steps."""
        config = TrainingConfig(
            gradient_accumulation_steps=2,  # Accumulate over 2 steps
            mixed_precision=False,
        )
        
        gradient_accumulator = GradientAccumulator(config.gradient_accumulation_steps)
        training_loop = TrainingLoop(
            model=mock_model,
            optimizer=mock_optimizer,
            config=config,
            gradient_accumulator=gradient_accumulator,
        )
        
        batch = {
            "input": mx.random.normal((4, 10)),
            "labels": mx.random.randint(0, 2, (4,))
        }
        
        # First step - should not update optimizer
        training_loop.train_batch(batch)
        assert mock_optimizer.update.call_count == 0
        
        # Second step - should update optimizer
        training_loop.train_batch(batch)
        assert mock_optimizer.update.call_count == 1
    
    def test_model_without_loss_key_raises_error(self, mock_optimizer, training_config, gradient_accumulator):
        """Test that model without loss key raises error."""
        # Create model that doesn't return loss
        def mock_model_call(**kwargs):
            return {"logits": mx.random.normal((4, 2))}
        
        mock_model = Mock(side_effect=mock_model_call)
        # Mock the parameters method
        mock_model.parameters = Mock(return_value={"weight": mx.random.normal((10, 2))})
        
        training_loop = TrainingLoop(
            model=mock_model,
            optimizer=mock_optimizer,
            config=training_config,
            gradient_accumulator=gradient_accumulator,
        )
        
        batch = {"input": mx.random.normal((4, 10)), "labels": mx.random.randint(0, 2, (4,))}
        
        with pytest.raises(ValueError, match="Model must return a dictionary with 'loss' key"):
            training_loop.train_batch(batch)