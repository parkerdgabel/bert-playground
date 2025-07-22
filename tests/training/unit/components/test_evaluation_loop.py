"""Unit tests for EvaluationLoop component."""

import pytest
import mlx.core as mx
from unittest.mock import Mock, MagicMock

from training.components.evaluation_loop import EvaluationLoop
from core.protocols.training import TrainingState


class TestEvaluationLoop:
    """Test cases for EvaluationLoop component."""
    
    @pytest.fixture
    def evaluation_loop(self, mock_model):
        """Create EvaluationLoop instance."""
        return EvaluationLoop(mock_model)
    
    def test_initialization(self, evaluation_loop, mock_model):
        """Test EvaluationLoop initialization."""
        assert evaluation_loop.model is mock_model
        assert not evaluation_loop._use_compiled
        assert evaluation_loop._compiled_eval_step is None
    
    def test_evaluate_batch_basic(self, evaluation_loop):
        """Test basic batch evaluation."""
        # Create test batch
        batch = {
            "input": mx.random.normal((4, 10)),
            "labels": mx.random.randint(0, 2, (4,))
        }
        
        # Evaluate batch
        loss, metrics = evaluation_loop.evaluate_batch(batch)
        
        # Check results
        assert isinstance(loss, mx.array)
        assert loss.size == 1  # Scalar loss
        assert isinstance(metrics, dict)
        # Model returns logits, so metrics should contain logits
        assert "logits" in metrics
    
    def test_evaluate_batch_with_metadata(self, evaluation_loop):
        """Test batch evaluation ignores metadata."""
        batch = {
            "input": mx.random.normal((4, 10)),
            "labels": mx.random.randint(0, 2, (4,)),
            "metadata": {"some": "data"}
        }
        
        # Should not raise error even with metadata
        loss, metrics = evaluation_loop.evaluate_batch(batch)
        assert isinstance(loss, mx.array)
        assert isinstance(metrics, dict)
    
    def test_evaluate_batch_model_with_loss(self, mock_model):
        """Test evaluation with model that returns loss."""
        evaluation_loop = EvaluationLoop(mock_model)
        
        batch = {
            "input": mx.random.normal((4, 10)),
            "labels": mx.random.randint(0, 2, (4,))
        }
        
        # Mock model already returns loss and logits
        loss, metrics = evaluation_loop.evaluate_batch(batch)
        
        assert isinstance(loss, mx.array)
        assert "logits" in metrics
    
    def test_evaluate_batch_model_with_only_logits(self):
        """Test evaluation with model that only returns logits."""
        # Create mock model that returns only logits
        mock_model = Mock()
        mock_model.eval = Mock()
        mock_model.train = Mock()
        
        # Model returns only logits, no loss
        mock_model.return_value = {
            "logits": mx.random.normal((4, 2))
        }
        
        evaluation_loop = EvaluationLoop(mock_model)
        
        batch = {
            "input": mx.random.normal((4, 10)),
            "labels": mx.random.randint(0, 2, (4,))
        }
        
        loss, metrics = evaluation_loop.evaluate_batch(batch)
        
        # Should compute cross-entropy loss from logits and labels
        assert isinstance(loss, mx.array)
        assert loss.size == 1
    
    def test_evaluate_batch_model_without_labels(self):
        """Test evaluation without labels."""
        mock_model = Mock()
        mock_model.eval = Mock()
        mock_model.train = Mock()
        
        # Model returns only logits
        mock_model.return_value = {
            "logits": mx.random.normal((4, 2))
        }
        
        evaluation_loop = EvaluationLoop(mock_model)
        
        batch = {
            "input": mx.random.normal((4, 10)),
            # No labels
        }
        
        loss, metrics = evaluation_loop.evaluate_batch(batch)
        
        # Should use dummy loss when no labels available
        assert isinstance(loss, mx.array)
        assert float(loss.item()) == 0.0
    
    def test_evaluate_dataset(self, evaluation_loop, mock_val_loader):
        """Test evaluation on a dataset."""
        state = TrainingState()
        
        # Evaluate dataset
        metrics = evaluation_loop.evaluate(mock_val_loader, state)
        
        # Check results
        assert "eval_loss" in metrics
        assert isinstance(metrics["eval_loss"], float)
        assert metrics["eval_loss"] >= 0.0  # Loss should be non-negative
    
    def test_evaluate_dataset_with_callbacks(self, evaluation_loop, mock_val_loader):
        """Test evaluation with callbacks."""
        state = TrainingState()
        
        # Create mock callback
        callback = Mock()
        callback.on_evaluate_begin = Mock()
        callback.on_evaluate_end = Mock()
        
        # Evaluate with callback
        metrics = evaluation_loop.evaluate(mock_val_loader, state, callbacks=[callback])
        
        # Check callback was called
        callback.on_evaluate_begin.assert_called_once_with(state)
        callback.on_evaluate_end.assert_called_once_with(state, metrics)
        
        # Check results
        assert "eval_loss" in metrics
    
    def test_evaluate_without_state_and_callbacks(self, evaluation_loop, mock_val_loader):
        """Test evaluation without state and callbacks."""
        # Should work without state and callbacks
        metrics = evaluation_loop.evaluate(mock_val_loader)
        
        assert "eval_loss" in metrics
        assert isinstance(metrics["eval_loss"], float)
    
    def test_predict(self, evaluation_loop, mock_val_loader):
        """Test prediction generation."""
        predictions = evaluation_loop.predict(mock_val_loader)
        
        # Check results
        assert isinstance(predictions, mx.array)
        assert predictions.shape[0] == mock_val_loader.num_batches * mock_val_loader.batch_size
        assert predictions.shape[1] == 2  # Output dimension from mock model
    
    def test_predict_model_with_predictions_key(self):
        """Test prediction with model that returns predictions key."""
        # Create mock model that returns predictions instead of logits
        mock_model = Mock()
        mock_model.eval = Mock()
        mock_model.train = Mock()
        mock_model.return_value = {
            "predictions": mx.random.normal((4, 2))
        }
        
        evaluation_loop = EvaluationLoop(mock_model)
        
        # Create small data loader for testing
        class SmallDataLoader:
            def __init__(self):
                self.batch_size = 4
                self.batches = [
                    {"input": mx.random.normal((4, 10))},
                    {"input": mx.random.normal((4, 10))}
                ]
                self._index = 0
            
            def __iter__(self):
                self._index = 0
                return self
            
            def __next__(self):
                if self._index >= len(self.batches):
                    raise StopIteration
                batch = self.batches[self._index]
                self._index += 1
                return batch
        
        loader = SmallDataLoader()
        predictions = evaluation_loop.predict(loader)
        
        assert isinstance(predictions, mx.array)
        assert predictions.shape[0] == 8  # 2 batches * 4 samples
        assert predictions.shape[1] == 2
    
    def test_predict_model_without_logits_or_predictions(self):
        """Test prediction with model that doesn't return logits or predictions."""
        mock_model = Mock()
        mock_model.eval = Mock()
        mock_model.train = Mock()
        mock_model.return_value = {
            "other_output": mx.random.normal((4, 2))
        }
        
        evaluation_loop = EvaluationLoop(mock_model)
        
        batch = {"input": mx.random.normal((4, 10))}
        
        class SingleBatchLoader:
            def __iter__(self):
                return iter([batch])
        
        loader = SingleBatchLoader()
        
        with pytest.raises(ValueError, match="Model must return 'logits' or 'predictions'"):
            evaluation_loop.predict(loader)
    
    def test_set_compiled_step(self, evaluation_loop):
        """Test setting compiled evaluation step."""
        # Create mock compiled step
        compiled_step = Mock(return_value=(mx.array(1.0), {"logits": mx.array([[1.0, 2.0]])}))
        
        evaluation_loop.set_compiled_step(compiled_step)
        
        assert evaluation_loop._use_compiled
        assert evaluation_loop._compiled_eval_step is compiled_step
        
        # Test that compiled step is used
        batch = {"input": mx.random.normal((4, 10)), "labels": mx.random.randint(0, 2, (4,))}
        evaluation_loop.evaluate_batch(batch)
        
        compiled_step.assert_called_once_with(batch)
    
    def test_model_mode_changes(self, mock_model):
        """Test that model is set to eval mode during evaluation."""
        evaluation_loop = EvaluationLoop(mock_model)
        
        # Create simple data loader
        class SimpleLoader:
            def __iter__(self):
                return iter([{"input": mx.random.normal((4, 10))}])
        
        loader = SimpleLoader()
        
        # Evaluate
        evaluation_loop.evaluate(loader)
        
        # Check that eval() and train() were called
        mock_model.eval.assert_called()
        mock_model.train.assert_called()
    
    def test_model_fallback_calling_convention(self):
        """Test fallback to batch dictionary calling convention."""
        # Create mock model that doesn't accept **kwargs
        mock_model = Mock()
        mock_model.eval = Mock()
        mock_model.train = Mock()
        
        def model_call(batch):
            # Model that only accepts batch dict
            return {
                "loss": mx.array(1.0),
                "logits": mx.random.normal((4, 2))
            }
        
        # First call with **kwargs should raise TypeError
        # Second call with batch dict should work
        mock_model.side_effect = [TypeError(), model_call(None)]
        
        evaluation_loop = EvaluationLoop(mock_model)
        
        batch = {"input": mx.random.normal((4, 10)), "labels": mx.random.randint(0, 2, (4,))}
        
        # Should not raise error due to fallback
        loss, metrics = evaluation_loop.evaluate_batch(batch)
        
        assert isinstance(loss, mx.array)
        assert "logits" in metrics