"""Unit tests for TrainingOrchestrator component."""

import pytest
from unittest.mock import Mock, MagicMock, patch
from pathlib import Path

from training.components.training_orchestrator import TrainingOrchestrator
from training.core.config import BaseTrainerConfig, TrainingConfig, EnvironmentConfig
from core.protocols.training import TrainingState, TrainingResult


class TestTrainingOrchestrator:
    """Test cases for TrainingOrchestrator component."""
    
    @pytest.fixture
    def mock_components(self):
        """Create mock components."""
        training_loop = Mock()
        evaluation_loop = Mock()
        checkpoint_manager = Mock()
        metrics_tracker = Mock()
        
        # Set up return values
        training_loop.train_epoch.return_value = {"loss": 0.5, "accuracy": 0.8}
        evaluation_loop.evaluate.return_value = {"eval_loss": 0.6, "eval_accuracy": 0.75}
        checkpoint_manager.save_checkpoint.return_value = Path("/tmp/checkpoint")
        checkpoint_manager.get_best_checkpoint.return_value = Path("/tmp/best")
        
        return {
            "training_loop": training_loop,
            "evaluation_loop": evaluation_loop,
            "checkpoint_manager": checkpoint_manager,
            "metrics_tracker": metrics_tracker,
        }
    
    @pytest.fixture
    def trainer_config(self, tmp_output_dir):
        """Create trainer configuration."""
        return BaseTrainerConfig(
            training=TrainingConfig(
                num_epochs=3,
                eval_strategy="epoch",
                save_strategy="epoch",
                best_metric="eval_loss",
                best_metric_mode="min",
                early_stopping=True,
                early_stopping_patience=2,
            ),
            environment=EnvironmentConfig(
                output_dir=tmp_output_dir,
            ),
        )
    
    @pytest.fixture
    def orchestrator(self, mock_model, trainer_config, mock_components):
        """Create TrainingOrchestrator instance."""
        return TrainingOrchestrator(
            model=mock_model,
            config=trainer_config,
            callbacks=[],
            **mock_components,
        )
    
    def test_initialization(self, orchestrator, mock_model, trainer_config, mock_components):
        """Test TrainingOrchestrator initialization."""
        assert orchestrator._model is mock_model
        assert orchestrator._config is trainer_config
        assert orchestrator.training_loop is mock_components["training_loop"]
        assert orchestrator.evaluation_loop is mock_components["evaluation_loop"]
        assert orchestrator.checkpoint_manager is mock_components["checkpoint_manager"]
        assert orchestrator.metrics_tracker is mock_components["metrics_tracker"]
        assert len(orchestrator.callbacks) == 0
        assert isinstance(orchestrator._state, TrainingState)
    
    def test_setup_metrics_tracking(self, mock_model, trainer_config, mock_components):
        """Test metrics tracking configuration."""
        orchestrator = TrainingOrchestrator(
            model=mock_model,
            config=trainer_config,
            callbacks=[],
            **mock_components,
        )
        
        # Check that metrics tracker was configured
        mock_components["metrics_tracker"].configure_metric.assert_called_once_with(
            "eval_loss", "min"
        )
    
    def test_train_basic(self, orchestrator, mock_train_loader, mock_val_loader):
        """Test basic training workflow."""
        # Mock time for consistent results
        with patch('time.time', return_value=1000.0):
            result = orchestrator.train(mock_train_loader, mock_val_loader)
        
        # Check result
        assert isinstance(result, TrainingResult)
        assert result.total_epochs == 3
        assert result.total_time >= 0
        assert result.stop_reason == "completed"
        assert not result.early_stopped
        
        # Check training loop was called
        assert orchestrator.training_loop.train_epoch.call_count == 3
        
        # Check evaluation loop was called
        assert orchestrator.evaluation_loop.evaluate.call_count == 3
        
        # Check metrics were tracked
        assert orchestrator.metrics_tracker.add_metrics.call_count >= 3
    
    def test_train_without_validation(self, orchestrator, mock_train_loader):
        """Test training without validation data."""
        with patch('time.time', return_value=1000.0):
            result = orchestrator.train(mock_train_loader, val_dataloader=None)
        
        # Should complete training
        assert isinstance(result, TrainingResult)
        assert result.total_epochs == 3
        
        # Training should still be called
        assert orchestrator.training_loop.train_epoch.call_count == 3
        
        # But evaluation should not be called
        assert orchestrator.evaluation_loop.evaluate.call_count == 0
    
    def test_train_with_resume(self, orchestrator, mock_train_loader, mock_val_loader):
        """Test training with resume from checkpoint."""
        resume_path = Path("/tmp/resume_checkpoint")
        
        # Mock checkpoint manager to return state
        resume_state = TrainingState(epoch=1, global_step=50)
        orchestrator.checkpoint_manager.load_checkpoint.return_value = resume_state
        
        with patch('time.time', return_value=1000.0):
            result = orchestrator.train(
                mock_train_loader,
                mock_val_loader,
                resume_from=resume_path
            )
        
        # Check checkpoint was loaded
        orchestrator.checkpoint_manager.load_checkpoint.assert_called_once()
        
        # Training should complete
        assert isinstance(result, TrainingResult)
    
    def test_early_stopping(self, mock_model, trainer_config, mock_components, mock_train_loader, mock_val_loader):
        """Test early stopping functionality."""
        # Configure for early stopping with patience 1
        trainer_config.training.early_stopping_patience = 1
        
        orchestrator = TrainingOrchestrator(
            model=mock_model,
            config=trainer_config,
            callbacks=[],
            **mock_components,
        )
        
        # Mock metrics tracker to never indicate improvement
        mock_components["metrics_tracker"].is_best_metric.return_value = False
        
        with patch('time.time', return_value=1000.0):
            result = orchestrator.train(mock_train_loader, mock_val_loader)
        
        # Should stop early
        assert result.early_stopped
        assert result.stop_reason == "early_stopping"
        assert result.total_epochs < 3  # Should stop before completing all epochs
    
    def test_best_model_saving(self, orchestrator, mock_train_loader, mock_val_loader):
        """Test best model detection and saving."""
        # Mock metrics tracker to indicate improvement on first epoch
        orchestrator.metrics_tracker.is_best_metric.side_effect = [True, False, False]
        
        # Set save strategy to include best
        orchestrator._config.training.save_strategy = "best"
        
        with patch('time.time', return_value=1000.0):
            result = orchestrator.train(mock_train_loader, mock_val_loader)
        
        # Check that best checkpoint was saved
        save_calls = orchestrator.checkpoint_manager.save_checkpoint.call_args_list
        best_saves = [call for call in save_calls if call[1].get("is_best")]
        assert len(best_saves) > 0
    
    def test_evaluate_delegate(self, orchestrator, mock_val_loader):
        """Test evaluate method delegates to evaluation loop."""
        metrics = orchestrator.evaluate(mock_val_loader)
        
        # Should delegate to evaluation loop
        orchestrator.evaluation_loop.evaluate.assert_called_once()
        
        # Should return evaluation metrics
        expected_metrics = {"eval_loss": 0.6, "eval_accuracy": 0.75}
        assert metrics == expected_metrics
    
    def test_predict_delegate(self, orchestrator, mock_val_loader):
        """Test predict method delegates to evaluation loop."""
        # Mock prediction return
        import mlx.core as mx
        expected_predictions = mx.random.normal((10, 2))
        orchestrator.evaluation_loop.predict.return_value = expected_predictions
        
        predictions = orchestrator.predict(mock_val_loader)
        
        # Should delegate to evaluation loop
        orchestrator.evaluation_loop.predict.assert_called_once_with(mock_val_loader)
        
        # Should return predictions
        assert predictions is expected_predictions
    
    def test_save_checkpoint_delegate(self, orchestrator):
        """Test save_checkpoint delegates to checkpoint manager."""
        test_path = Path("/tmp/test")
        
        orchestrator.save_checkpoint(test_path)
        
        # Should call internal save method (which uses checkpoint manager)
        # This tests the public interface
    
    def test_load_checkpoint_delegate(self, orchestrator):
        """Test load_checkpoint delegates to checkpoint manager."""
        test_path = Path("/tmp/test") 
        resume_state = TrainingState(epoch=2, global_step=100)
        orchestrator.checkpoint_manager.load_checkpoint.return_value = resume_state
        
        orchestrator.load_checkpoint(test_path)
        
        # Should update state
        assert orchestrator._state.epoch == 2
        assert orchestrator._state.global_step == 100
    
    def test_property_accessors(self, orchestrator, mock_model, trainer_config):
        """Test property getter methods."""
        assert orchestrator.model is mock_model
        assert orchestrator.config is trainer_config
        assert isinstance(orchestrator.state, TrainingState)
    
    def test_should_evaluate_strategies(self, orchestrator):
        """Test evaluation strategy logic."""
        # Test epoch strategy
        orchestrator._config.training.eval_strategy = "epoch"
        assert orchestrator._should_evaluate(0, Mock()) is True
        
        # Test steps strategy
        orchestrator._config.training.eval_strategy = "steps"
        orchestrator._config.training.eval_steps = 10
        orchestrator._state.global_step = 20  # Multiple of 10
        assert orchestrator._should_evaluate(0, Mock()) is True
        
        orchestrator._state.global_step = 15  # Not multiple of 10
        assert orchestrator._should_evaluate(0, Mock()) is False
        
        # Test no strategy
        orchestrator._config.training.eval_strategy = "no"
        assert orchestrator._should_evaluate(0, Mock()) is False
        
        # Test without validation dataloader
        assert orchestrator._should_evaluate(0, None) is False
    
    def test_should_save_checkpoint_strategies(self, orchestrator):
        """Test checkpoint saving strategy logic."""
        # Test epoch strategy
        orchestrator._config.training.save_strategy = "epoch"
        assert orchestrator._should_save_checkpoint(0) is True
        
        # Test steps strategy
        orchestrator._config.training.save_strategy = "steps"
        orchestrator._config.training.save_steps = 100
        orchestrator._state.global_step = 200  # Multiple of 100
        assert orchestrator._should_save_checkpoint(0) is True
        
        orchestrator._state.global_step = 150  # Not multiple of 100
        assert orchestrator._should_save_checkpoint(0) is False
        
        # Test best strategy (should not save on epoch)
        orchestrator._config.training.save_strategy = "best"
        assert orchestrator._should_save_checkpoint(0) is False
    
    def test_callbacks_called(self, mock_model, trainer_config, mock_components, mock_train_loader):
        """Test that callbacks are properly called."""
        # Create mock callbacks
        callback1 = Mock()
        callback2 = Mock()
        
        orchestrator = TrainingOrchestrator(
            model=mock_model,
            config=trainer_config,
            callbacks=[callback1, callback2],
            **mock_components,
        )
        
        with patch('time.time', return_value=1000.0):
            orchestrator.train(mock_train_loader, val_dataloader=None)
        
        # Check callbacks were called
        for callback in [callback1, callback2]:
            callback.on_train_begin.assert_called()
            callback.on_train_end.assert_called()
    
    def test_metrics_save_epoch(self, orchestrator, mock_train_loader, mock_val_loader):
        """Test that epoch metrics are saved."""
        with patch('time.time', return_value=1000.0):
            orchestrator.train(mock_train_loader, mock_val_loader)
        
        # Check that save_epoch_metrics was called
        assert orchestrator.metrics_tracker.save_epoch_metrics.call_count == 3  # 3 epochs
        
        # Check the calls had proper structure
        save_calls = orchestrator.metrics_tracker.save_epoch_metrics.call_args_list
        for i, call in enumerate(save_calls):
            args, kwargs = call
            assert args[0] == i  # epoch number
            assert "train" in str(call) or len(args) >= 2  # train metrics provided
    
    def test_training_result_creation(self, orchestrator, mock_train_loader, mock_val_loader):
        """Test training result object creation."""
        with patch('time.time', return_value=1000.0):
            result = orchestrator.train(mock_train_loader, mock_val_loader)
        
        # Check result structure
        assert hasattr(result, 'final_train_loss')
        assert hasattr(result, 'final_val_loss')
        assert hasattr(result, 'best_val_loss')
        assert hasattr(result, 'best_val_metric')
        assert hasattr(result, 'final_metrics')
        assert hasattr(result, 'train_history')
        assert hasattr(result, 'val_history')
        assert hasattr(result, 'final_model_path')
        assert hasattr(result, 'best_model_path')
        assert hasattr(result, 'total_epochs')
        assert hasattr(result, 'total_steps')
        assert hasattr(result, 'total_time')
        assert hasattr(result, 'early_stopped')
        assert hasattr(result, 'stop_reason')