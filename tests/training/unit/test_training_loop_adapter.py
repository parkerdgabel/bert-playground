"""Tests for the refactored training loop with framework adapters."""

import pytest
from unittest.mock import Mock, MagicMock, patch
import numpy as np

from training.components.training_loop import TrainingLoop
from training.core.config import TrainingConfig
from core.protocols.training import TrainingState


class TestTrainingLoopWithAdapter:
    """Test training loop with framework adapter."""
    
    @pytest.fixture
    def mock_model(self):
        """Create mock model."""
        model = Mock()
        model.parameters = Mock(return_value={})
        return model
    
    @pytest.fixture
    def mock_optimizer(self):
        """Create mock optimizer."""
        optimizer = Mock()
        optimizer.learning_rate = 0.001
        optimizer.state = {}
        optimizer.update = Mock()
        return optimizer
    
    @pytest.fixture
    def mock_config(self):
        """Create mock training config."""
        config = Mock(spec=TrainingConfig)
        config.gradient_accumulation_steps = 1
        config.mixed_precision = False
        config.label_smoothing = 0.0
        return config
    
    @pytest.fixture
    def mock_adapter(self):
        """Create mock framework adapter."""
        adapter = Mock()
        adapter.name = "mock"
        adapter.available = True
        adapter.supports_compilation = False
        
        # Mock tensor operations
        adapter.to_python = lambda x: float(x) if hasattr(x, '__float__') else x
        adapter.to_tensor = lambda x: x
        adapter.tensor_multiply = lambda a, b: a * b
        adapter.evaluate_tensors = Mock()
        
        # Mock gradient operations
        adapter.compute_gradient_norm = Mock(return_value=1.0)
        adapter.clip_gradients_by_norm = Mock(return_value=({}, 1.0))
        adapter.scale_gradients = lambda grads, scale: {k: v * scale for k, v in grads.items()}
        adapter.accumulate_gradients = lambda a, b: {k: a.get(k, 0) + b.get(k, 0) for k in set(a) | set(b)}
        
        # Mock model operations
        adapter.get_model_parameters = Mock(return_value={})
        adapter.update_model_parameters = Mock()
        adapter.get_learning_rate = Mock(return_value=0.001)
        
        # Mock mixed precision
        adapter.apply_mixed_precision = lambda x: x
        
        # Mock compilation
        adapter.compile_function = lambda fn, **kwargs: fn
        
        return adapter
    
    @patch('training.components.training_loop.get_framework_adapter')
    def test_initialization(self, mock_get_adapter, mock_model, mock_optimizer, mock_config, mock_adapter):
        """Test training loop initialization with adapter."""
        mock_get_adapter.return_value = mock_adapter
        
        loop = TrainingLoop(
            model=mock_model,
            optimizer=mock_optimizer,
            config=mock_config,
            framework="mock"
        )
        
        assert loop.model == mock_model
        assert loop.optimizer == mock_optimizer
        assert loop.config == mock_config
        assert loop.adapter == mock_adapter
        mock_get_adapter.assert_called_once_with("mock")
    
    @patch('training.components.training_loop.get_framework_adapter')
    def test_train_step_creation(self, mock_get_adapter, mock_model, mock_optimizer, mock_config, mock_adapter):
        """Test training step function creation."""
        mock_get_adapter.return_value = mock_adapter
        
        # Mock value_and_grad function
        def mock_value_and_grad_fn(model, loss_fn):
            def wrapped(model, batch):
                loss, outputs = loss_fn(model, batch)
                grads = {"param1": 0.1, "param2": 0.2}
                return (loss, outputs), grads
            return wrapped
        
        mock_adapter.create_value_and_grad_fn = mock_value_and_grad_fn
        
        loop = TrainingLoop(
            model=mock_model,
            optimizer=mock_optimizer,
            config=mock_config,
            framework="mock"
        )
        
        # Test that train step was created
        assert loop._train_step is not None
        assert callable(loop._train_step)
    
    @patch('training.components.training_loop.get_framework_adapter')
    def test_train_batch(self, mock_get_adapter, mock_model, mock_optimizer, mock_config, mock_adapter):
        """Test single batch training."""
        mock_get_adapter.return_value = mock_adapter
        
        # Setup model to return loss
        mock_model.return_value = {"loss": 2.5, "accuracy": 0.85}
        
        # Mock value_and_grad function
        def mock_value_and_grad_fn(model, loss_fn):
            def wrapped(model, batch):
                outputs = model(**batch)
                loss = outputs["loss"]
                grads = {"param1": 0.1, "param2": 0.2}
                return (loss, outputs), grads
            return wrapped
        
        mock_adapter.create_value_and_grad_fn = mock_value_and_grad_fn
        
        loop = TrainingLoop(
            model=mock_model,
            optimizer=mock_optimizer,
            config=mock_config,
            framework="mock"
        )
        
        # Create test batch
        batch = {
            "input_ids": np.random.randint(0, 1000, size=(8, 32)),
            "labels": np.random.randint(0, 2, size=(8,))
        }
        
        # Train on batch
        loss, metrics = loop.train_batch(batch)
        
        # Verify results
        assert loss == 2.5
        assert "loss" in metrics
        assert "accuracy" in metrics
        assert metrics["learning_rate"] == 0.001
        
        # Verify adapter methods were called
        mock_adapter.evaluate_tensors.assert_called()
        mock_adapter.update_model_parameters.assert_called_once()
    
    @patch('training.components.training_loop.get_framework_adapter')
    def test_gradient_clipping(self, mock_get_adapter, mock_model, mock_optimizer, mock_config, mock_adapter):
        """Test gradient clipping with adapter."""
        mock_get_adapter.return_value = mock_adapter
        
        # Setup model
        mock_model.return_value = {"loss": 2.5}
        
        # Mock gradient clipping
        clipped_grads = {"param1": 0.05, "param2": 0.1}
        mock_adapter.clip_gradients_by_norm = Mock(return_value=(clipped_grads, 0.5))
        
        # Mock value_and_grad function
        def mock_value_and_grad_fn(model, loss_fn):
            def wrapped(model, batch):
                outputs = model(**batch)
                loss = outputs["loss"]
                grads = {"param1": 0.5, "param2": 1.0}  # Large gradients
                return (loss, outputs), grads
            return wrapped
        
        mock_adapter.create_value_and_grad_fn = mock_value_and_grad_fn
        
        loop = TrainingLoop(
            model=mock_model,
            optimizer=mock_optimizer,
            config=mock_config,
            framework="mock",
            max_grad_norm=1.0
        )
        
        # Train on batch
        batch = {"input_ids": np.random.randint(0, 1000, size=(8, 32))}
        loss, metrics = loop.train_batch(batch)
        
        # Verify gradient clipping was applied
        mock_adapter.clip_gradients_by_norm.assert_called_once()
        assert "grad_norm" in metrics
        assert metrics["grad_norm"] == 0.5
    
    @patch('training.components.training_loop.get_framework_adapter')
    def test_mixed_precision(self, mock_get_adapter, mock_model, mock_optimizer, mock_config, mock_adapter):
        """Test mixed precision training."""
        mock_get_adapter.return_value = mock_adapter
        
        # Enable mixed precision
        mock_config.mixed_precision = True
        
        # Track mixed precision calls
        mixed_precision_applied = False
        def apply_mixed_precision(inputs):
            nonlocal mixed_precision_applied
            mixed_precision_applied = True
            return inputs
        
        mock_adapter.apply_mixed_precision = apply_mixed_precision
        
        # Setup model
        mock_model.return_value = {"loss": 2.5}
        
        # Mock value_and_grad function
        def mock_value_and_grad_fn(model, loss_fn):
            def wrapped(model, batch):
                loss, outputs = loss_fn(model, batch)
                grads = {"param1": 0.1}
                return (loss, outputs), grads
            return wrapped
        
        mock_adapter.create_value_and_grad_fn = mock_value_and_grad_fn
        
        loop = TrainingLoop(
            model=mock_model,
            optimizer=mock_optimizer,
            config=mock_config,
            framework="mock"
        )
        
        # Train on batch
        batch = {"input_ids": np.random.randint(0, 1000, size=(8, 32))}
        loss, metrics = loop.train_batch(batch)
        
        # Verify mixed precision was applied
        assert mixed_precision_applied
    
    @patch('training.components.training_loop.get_framework_adapter')
    def test_train_epoch(self, mock_get_adapter, mock_model, mock_optimizer, mock_config, mock_adapter):
        """Test full epoch training."""
        mock_get_adapter.return_value = mock_adapter
        
        # Setup model
        mock_model.return_value = {"loss": 2.5, "accuracy": 0.85}
        
        # Mock value_and_grad function
        def mock_value_and_grad_fn(model, loss_fn):
            def wrapped(model, batch):
                outputs = model(**batch)
                loss = outputs["loss"]
                grads = {"param1": 0.1}
                return (loss, outputs), grads
            return wrapped
        
        mock_adapter.create_value_and_grad_fn = mock_value_and_grad_fn
        
        loop = TrainingLoop(
            model=mock_model,
            optimizer=mock_optimizer,
            config=mock_config,
            framework="mock"
        )
        
        # Create mock dataloader
        dataloader = [
            {"input_ids": np.random.randint(0, 1000, size=(8, 32))}
            for _ in range(10)
        ]
        
        # Create training state
        state = TrainingState()
        
        # Train epoch
        metrics = loop.train_epoch(dataloader, state)
        
        # Verify metrics
        assert "loss" in metrics
        assert "accuracy" in metrics
        assert metrics["loss"] == 2.5  # Average of all batches
        assert metrics["accuracy"] == 0.85
        
        # Verify state was updated
        assert state.global_step == 10
        assert state.samples_seen == 10 * loop.batch_size