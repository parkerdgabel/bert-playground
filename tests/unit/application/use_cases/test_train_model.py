"""Unit tests for train model use case.

These tests verify the application layer logic for training models,
focusing on orchestration and coordination without external dependencies.
"""

import pytest
from unittest.mock import Mock, AsyncMock, MagicMock
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List

from application.use_cases.train_model import TrainModelUseCase
from application.dto.training import TrainingRequestDTO, TrainingResponseDTO
from domain.services.training_service import (
    TrainingConfig, TrainingService, TrainingState, TrainingMetrics,
    OptimizerType, SchedulerType
)
from ports.primary.training import TrainingResult


class TestTrainModelUseCase:
    """Test the train model use case."""
    
    @pytest.fixture
    def mock_dependencies(self):
        """Create mock dependencies for the use case."""
        return {
            'training_service': Mock(spec=TrainingService),
            'storage_port': AsyncMock(),
            'monitoring_port': AsyncMock(),
            'config_port': AsyncMock(),
            'checkpoint_port': AsyncMock(),
            'metrics_port': AsyncMock()
        }
    
    @pytest.fixture
    def use_case(self, mock_dependencies):
        """Create use case instance with mocked dependencies."""
        return TrainModelUseCase(**mock_dependencies)
    
    @pytest.fixture 
    def di_use_case(self, use_case_container):
        """Create use case instance using DI container."""
        return use_case_container.resolve(TrainModelUseCase)
    
    @pytest.fixture
    def valid_request(self):
        """Create a valid training request."""
        return TrainingRequestDTO(
            model_type="bert",
            model_config={"hidden_size": 768},
            train_data_path=Path("/data/train.csv"),
            val_data_path=Path("/data/val.csv"),
            output_dir=Path("/output"),
            num_epochs=3,
            batch_size=32,
            learning_rate=5e-5,
            optimizer_type="adamw",
            scheduler_type="linear",
            experiment_name="test_experiment",
            run_name="test_run",
            tags=["test", "unit"]
        )
    
    async def test_execute_successful_training(self, use_case, mock_dependencies, valid_request):
        """Test successful training execution."""
        # Setup mocks
        mock_dependencies['training_service'].should_evaluate.return_value = True
        mock_dependencies['training_service'].should_save.return_value = True
        mock_dependencies['training_service'].should_stop.return_value = False
        mock_dependencies['training_service'].create_optimizer.return_value = Mock()
        mock_dependencies['training_service'].create_scheduler.return_value = Mock()
        
        # Mock model and data loaders
        mock_model = Mock()
        mock_model.parameters.return_value = []
        mock_model.state_dict.return_value = {}
        
        mock_train_loader = Mock()
        mock_train_loader.dataset = Mock()
        mock_train_loader.dataset.__len__ = Mock(return_value=1000)
        
        # Override methods that would fail
        use_case._prepare_model = AsyncMock(return_value=mock_model)
        use_case._create_data_loader = AsyncMock(return_value=mock_train_loader)
        use_case._train_epoch = AsyncMock(return_value=TrainingMetrics(
            loss=0.5,
            learning_rate=5e-5,
            epoch=1.0
        ))
        use_case._validate = AsyncMock(return_value={"loss": 0.3, "accuracy": 0.9})
        
        # Execute
        response = await use_case.execute(valid_request)
        
        # Verify
        assert response.success is True
        assert response.run_id == "test_run"
        assert response.error_message is None
        
        # Verify monitoring was set up
        mock_dependencies['monitoring_port'].start_run.assert_called_once()
        mock_dependencies['monitoring_port'].log_params.assert_called_once()
        
        # Verify storage directory was created
        mock_dependencies['storage_port'].create_directory.assert_called()
    
    async def test_execute_with_validation_errors(self, use_case, valid_request):
        """Test execution with validation errors."""
        # Create invalid request
        valid_request.num_epochs = -1  # Invalid
        valid_request.validate = Mock(return_value=["num_epochs must be positive"])
        
        response = await use_case.execute(valid_request)
        
        assert response.success is False
        assert "Validation errors" in response.error_message
        assert "num_epochs must be positive" in response.error_message
    
    async def test_execute_with_exception(self, use_case, mock_dependencies, valid_request):
        """Test execution with exception during training."""
        # Setup to raise exception
        mock_dependencies['monitoring_port'].start_run.side_effect = Exception("Test error")
        
        response = await use_case.execute(valid_request)
        
        assert response.success is False
        assert response.error_message is not None
        assert "Test error" in response.error_message
        
        # Verify error was logged
        mock_dependencies['monitoring_port'].log_error.assert_called()
    
    def test_create_training_config(self, use_case, valid_request):
        """Test conversion from request DTO to domain config."""
        config = use_case._create_training_config(valid_request)
        
        assert isinstance(config, TrainingConfig)
        assert config.num_epochs == valid_request.num_epochs
        assert config.batch_size == valid_request.batch_size
        assert config.learning_rate == valid_request.learning_rate
        assert config.optimizer_type == OptimizerType.ADAMW
        assert config.scheduler_type == SchedulerType.LINEAR
    
    async def test_setup_monitoring(self, use_case, mock_dependencies, valid_request):
        """Test monitoring setup."""
        run_id = await use_case._setup_monitoring(valid_request)
        
        assert run_id == "test_run"
        
        # Verify directory creation
        expected_dir = valid_request.output_dir / "test_run"
        mock_dependencies['storage_port'].create_directory.assert_called_with(expected_dir)
        
        # Verify monitoring initialization
        mock_dependencies['monitoring_port'].start_run.assert_called_with(
            name="test_run",
            experiment="test_experiment",
            tags=["test", "unit"]
        )
        
        # Verify parameter logging
        logged_params = mock_dependencies['monitoring_port'].log_params.call_args[0][0]
        assert logged_params["model_type"] == "bert"
        assert logged_params["num_epochs"] == 3
        assert logged_params["batch_size"] == 32
    
    async def test_resume_from_checkpoint(self, use_case, mock_dependencies):
        """Test resuming from checkpoint."""
        mock_model = Mock()
        checkpoint_path = Path("/checkpoints/checkpoint-1000")
        
        # Mock checkpoint data
        checkpoint_data = {
            "model_state": {"layer1": "weights"},
            "optimizer_state": {"lr": 1e-5},
            "training_state": {"epoch": 2, "global_step": 1000}
        }
        mock_dependencies['checkpoint_port'].load_checkpoint.return_value = checkpoint_data
        
        await use_case._resume_from_checkpoint(mock_model, checkpoint_path)
        
        # Verify checkpoint was loaded
        mock_dependencies['checkpoint_port'].load_checkpoint.assert_called_with(checkpoint_path)
    
    async def test_save_checkpoint(self, use_case, mock_dependencies):
        """Test checkpoint saving."""
        mock_model = Mock()
        mock_model.state_dict.return_value = {"layer1": "weights"}
        
        mock_optimizer = Mock()
        mock_optimizer.state_dict.return_value = {"lr": 1e-5}
        
        mock_scheduler = Mock()
        mock_scheduler.state_dict.return_value = {"last_epoch": 2}
        
        state = TrainingState(epoch=2, global_step=1000)
        state.to_dict = Mock(return_value={"epoch": 2, "global_step": 1000})
        
        checkpoint_path = Path("/output/checkpoint-1000")
        
        await use_case._save_checkpoint(
            mock_model, mock_optimizer, mock_scheduler, state, checkpoint_path
        )
        
        # Verify checkpoint was saved
        saved_data = mock_dependencies['checkpoint_port'].save_checkpoint.call_args[0][0]
        assert saved_data["model_state"] == {"layer1": "weights"}
        assert saved_data["optimizer_state"] == {"lr": 1e-5}
        assert saved_data["scheduler_state"] == {"last_epoch": 2}
        assert saved_data["training_state"] == {"epoch": 2, "global_step": 1000}
        assert "timestamp" in saved_data
    
    async def test_save_final_artifacts(self, use_case, mock_dependencies):
        """Test saving final artifacts."""
        mock_model = Mock()
        result = TrainingResult(
            final_train_loss=0.2,
            final_val_loss=0.15,
            best_val_loss=0.12,
            best_val_metric=0.95,
            final_metrics={"accuracy": 0.92, "f1": 0.90},
            train_history=[{"loss": 0.5}, {"loss": 0.3}, {"loss": 0.2}],
            val_history=[{"loss": 0.4}, {"loss": 0.2}, {"loss": 0.15}],
            total_epochs=3,
            total_steps=1000
        )
        output_dir = Path("/output")
        run_id = "test_run"
        
        paths = await use_case._save_final_artifacts(mock_model, result, output_dir, run_id)
        
        # Verify model was saved
        mock_dependencies['storage_port'].save_model.assert_called_with(
            mock_model, output_dir / "model_final"
        )
        
        # Verify history was saved
        history_data = mock_dependencies['storage_port'].save_json.call_args_list[0][0][0]
        assert "train_history" in history_data
        assert "val_history" in history_data
        
        # Verify metrics were saved
        metrics_data = mock_dependencies['storage_port'].save_json.call_args_list[1][0][0]
        assert metrics_data == {"accuracy": 0.92, "f1": 0.90}
        
        # Verify paths were returned
        assert paths["final_model"] == output_dir / "model_final"
        assert paths["history"] == output_dir / "training_history.json"
        assert paths["metrics"] == output_dir / "final_metrics.json"
    
    def test_create_response(self, use_case, valid_request):
        """Test response creation."""
        result = TrainingResult(
            final_train_loss=0.2,
            final_val_loss=0.15,
            best_val_loss=0.12,
            best_val_metric=0.95,
            final_metrics={"accuracy": 0.92},
            train_history=[{"loss": 0.5}],
            val_history=[{"loss": 0.4}],
            best_model_path=Path("/output/checkpoint-best"),
            total_epochs=3,
            total_steps=1000,
            early_stopped=False
        )
        
        paths = {
            "final_model": Path("/output/model_final"),
            "history": Path("/output/training_history.json"),
            "metrics": Path("/output/final_metrics.json")
        }
        
        start_time = datetime(2024, 1, 1, 10, 0, 0)
        end_time = datetime(2024, 1, 1, 11, 30, 0)
        run_id = "test_run"
        
        response = use_case._create_response(
            result, paths, start_time, end_time, run_id, valid_request
        )
        
        assert response.success is True
        assert response.final_train_loss == 0.2
        assert response.final_val_loss == 0.15
        assert response.best_val_loss == 0.12
        assert response.best_val_metric == 0.95
        assert response.final_metrics == {"accuracy": 0.92}
        assert response.final_model_path == Path("/output/model_final")
        assert response.best_model_path == Path("/output/checkpoint-best")
        assert response.total_epochs == 3
        assert response.total_steps == 1000
        assert response.total_time_seconds == 5400.0  # 1.5 hours
        assert response.early_stopped is False
        assert response.run_id == "test_run"
        assert response.start_time == start_time
        assert response.end_time == end_time
    
    async def test_run_training_with_early_stopping(self, use_case, mock_dependencies):
        """Test training with early stopping."""
        # Setup mocks
        mock_dependencies['training_service'].should_evaluate.return_value = True
        mock_dependencies['training_service'].should_save.return_value = False
        mock_dependencies['training_service'].should_stop.side_effect = [False, False, True]  # Stop at epoch 3
        mock_dependencies['training_service'].create_optimizer.return_value = Mock()
        mock_dependencies['training_service'].create_scheduler.return_value = Mock()
        
        # Mock model and loaders
        mock_model = Mock()
        mock_model.parameters.return_value = []
        mock_model.state_dict.return_value = {}
        
        mock_train_loader = Mock()
        mock_train_loader.dataset = Mock()
        mock_train_loader.dataset.__len__ = Mock(return_value=1000)
        
        mock_val_loader = Mock()
        
        # Setup training config
        config = TrainingConfig(
            num_epochs=10,  # Would train for 10, but will stop early
            batch_size=32,
            metric_for_best_model="loss",
            greater_is_better=False
        )
        
        # Override methods
        use_case._train_epoch = AsyncMock(return_value=TrainingMetrics(
            loss=0.5,
            learning_rate=5e-5,
            epoch=1.0
        ))
        use_case._validate = AsyncMock(side_effect=[
            {"loss": 0.4},  # Epoch 1
            {"loss": 0.45},  # Epoch 2 - worse
            {"loss": 0.5}    # Epoch 3 - even worse
        ])
        use_case._save_checkpoint = AsyncMock()
        
        output_dir = Path("/output")
        run_id = "test_run"
        
        result = await use_case._run_training(
            mock_model, mock_train_loader, mock_val_loader,
            config, output_dir, run_id
        )
        
        # Verify early stopping
        assert result.early_stopped is True
        assert result.stop_reason == "Early stopping"
        assert result.total_epochs == 3  # Stopped at epoch 3
        
        # Verify best model was saved (at epoch 1)
        assert result.best_val_loss == 0.4
    
    async def test_run_training_without_validation(self, use_case, mock_dependencies):
        """Test training without validation data."""
        # Setup mocks
        mock_dependencies['training_service'].should_evaluate.return_value = False
        mock_dependencies['training_service'].should_save.return_value = True
        mock_dependencies['training_service'].should_stop.return_value = False
        mock_dependencies['training_service'].create_optimizer.return_value = Mock()
        mock_dependencies['training_service'].create_scheduler.return_value = Mock()
        
        # Mock model and loader
        mock_model = Mock()
        mock_model.parameters.return_value = []
        mock_model.state_dict.return_value = {}
        
        mock_train_loader = Mock()
        mock_train_loader.dataset = Mock()
        mock_train_loader.dataset.__len__ = Mock(return_value=1000)
        
        config = TrainingConfig(num_epochs=2, batch_size=32)
        
        # Override training epoch
        use_case._train_epoch = AsyncMock(return_value=TrainingMetrics(
            loss=0.5,
            learning_rate=5e-5,
            epoch=1.0
        ))
        use_case._save_checkpoint = AsyncMock()
        
        output_dir = Path("/output")
        run_id = "test_run"
        
        result = await use_case._run_training(
            mock_model, mock_train_loader, None,  # No validation loader
            config, output_dir, run_id
        )
        
        # Verify no validation was attempted
        assert result.final_val_loss == 0.0
        assert result.best_val_loss == 0.0
        assert len(result.val_history) == 0
        
        # Verify checkpoints were saved
        assert use_case._save_checkpoint.call_count > 0
    
    async def test_di_container_integration(self, di_use_case, valid_request):
        """Test that the use case works with DI container dependency injection."""
        # This test verifies that all dependencies are properly injected
        assert di_use_case is not None
        assert hasattr(di_use_case, 'training_service')
        assert hasattr(di_use_case, 'storage_port')
        assert hasattr(di_use_case, 'monitoring_port')
        assert hasattr(di_use_case, 'config_port')
        assert hasattr(di_use_case, 'checkpoint_port')
        assert hasattr(di_use_case, 'metrics_port')
        
        # Verify the dependencies are properly injected and of correct types
        from domain.services.training import ModelTrainingService
        from ports.secondary.storage import StorageService
        from ports.secondary.monitoring import MonitoringService
        
        assert isinstance(di_use_case.training_service, ModelTrainingService)
        # Note: These will be mock instances due to our test configuration
        # In integration tests, they would be real implementations