"""End-to-end tests for complete training workflow.

These tests verify the entire system working together,
from user request to final trained model.
"""

import pytest
import tempfile
import shutil
import asyncio
from pathlib import Path
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime

from application.use_cases.train_model import TrainModelUseCase
from application.dto.training import TrainingRequestDTO
from core.adapters.file_storage import FileStorageAdapter, ModelCheckpointAdapter
from core.adapters.loguru_monitoring import LoguruMonitoringAdapter
from core.adapters.yaml_config import YamlConfigurationAdapter
from domain.services.training_service import TrainingService, TrainingConfig


class TestTrainingWorkflowE2E:
    """Test complete training workflow end-to-end."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for test artifacts."""
        temp_dir = tempfile.mkdtemp()
        yield Path(temp_dir)
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def mock_training_service(self):
        """Create mock training service."""
        service = Mock(spec=TrainingService)
        service.should_evaluate.return_value = True
        service.should_save.return_value = True
        service.should_stop.return_value = False
        service.create_optimizer.return_value = Mock()
        service.create_scheduler.return_value = Mock()
        return service
    
    @pytest.fixture
    def mock_metrics_port(self):
        """Create mock metrics port."""
        port = AsyncMock()
        port.calculate_metrics.return_value = {"accuracy": 0.95, "f1": 0.93}
        return port
    
    @pytest.fixture
    async def training_use_case(self, temp_dir, mock_training_service, mock_metrics_port):
        """Create training use case with real adapters where possible."""
        # Use real adapters where safe
        storage_adapter = FileStorageAdapter(base_path=temp_dir)
        checkpoint_adapter = ModelCheckpointAdapter(checkpoint_dir=temp_dir / "checkpoints")
        monitoring_adapter = LoguruMonitoringAdapter(log_file=temp_dir / "training.log")
        config_adapter = YamlConfigurationAdapter(config_dir=temp_dir / "configs")
        
        use_case = TrainModelUseCase(
            training_service=mock_training_service,
            storage_port=storage_adapter,
            monitoring_port=monitoring_adapter,
            config_port=config_adapter,
            checkpoint_port=checkpoint_adapter,
            metrics_port=mock_metrics_port
        )
        
        return use_case
    
    @pytest.fixture
    def training_request(self, temp_dir):
        """Create a training request."""
        # Create dummy data files
        train_data = temp_dir / "train.csv"
        val_data = temp_dir / "val.csv"
        train_data.write_text("text,label\nSample 1,0\nSample 2,1")
        val_data.write_text("text,label\nVal 1,0\nVal 2,1")
        
        return TrainingRequestDTO(
            model_type="bert",
            model_config={
                "hidden_size": 768,
                "num_hidden_layers": 12,
                "num_attention_heads": 12
            },
            train_data_path=train_data,
            val_data_path=val_data,
            output_dir=temp_dir / "output",
            num_epochs=2,
            batch_size=16,
            learning_rate=5e-5,
            optimizer_type="adamw",
            scheduler_type="linear",
            warmup_steps=100,
            eval_strategy="epoch",
            save_strategy="epoch",
            logging_steps=10,
            experiment_name="e2e_test",
            run_name=f"test_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            tags=["e2e", "test"]
        )
    
    @pytest.mark.asyncio
    async def test_successful_training_workflow(
        self, training_use_case, training_request, temp_dir, mock_training_service
    ):
        """Test successful end-to-end training workflow."""
        # Mock model and data loading
        mock_model = Mock()
        mock_model.parameters.return_value = []
        mock_model.state_dict.return_value = {"layer1": {"weight": [1, 2, 3]}}
        
        mock_train_loader = Mock()
        mock_train_loader.dataset = Mock()
        mock_train_loader.dataset.__len__ = Mock(return_value=100)
        
        mock_val_loader = Mock()
        
        # Patch the methods that interact with external systems
        training_use_case._prepare_model = AsyncMock(return_value=mock_model)
        training_use_case._create_data_loader = AsyncMock(
            side_effect=[mock_train_loader, mock_val_loader]
        )
        training_use_case._train_epoch = AsyncMock(return_value=Mock(
            to_dict=lambda: {"loss": 0.5, "learning_rate": 5e-5}
        ))
        training_use_case._validate = AsyncMock(return_value={
            "loss": 0.3, "accuracy": 0.85
        })
        
        # Execute training
        response = await training_use_case.execute(training_request)
        
        # Verify successful completion
        assert response.success is True
        assert response.error_message is None
        assert response.total_epochs == 2
        
        # Verify output directory structure
        output_dir = temp_dir / "output" / response.run_id
        assert output_dir.exists()
        
        # Verify logs were created
        log_file = temp_dir / "training.log"
        assert log_file.exists()
        log_content = log_file.read_text()
        assert "Training completed successfully" in log_content
        
        # Verify checkpoints directory
        checkpoint_dir = temp_dir / "checkpoints"
        assert checkpoint_dir.exists()
        
        # Verify metrics were logged
        assert training_use_case.monitoring.log_metrics.called
        logged_metrics = training_use_case.monitoring.log_metrics.call_args_list
        assert len(logged_metrics) > 0
    
    @pytest.mark.asyncio
    async def test_training_with_early_stopping(
        self, training_use_case, training_request, mock_training_service
    ):
        """Test training with early stopping."""
        # Configure for early stopping
        training_request.early_stopping_patience = 2
        training_request.metric_for_best_model = "val_loss"
        training_request.num_epochs = 10  # Set high, expect early stop
        
        # Mock deteriorating validation metrics
        val_losses = [0.3, 0.35, 0.4, 0.45]  # Getting worse
        
        # Setup mocks
        mock_model = Mock()
        mock_model.parameters.return_value = []
        mock_model.state_dict.return_value = {}
        
        training_use_case._prepare_model = AsyncMock(return_value=mock_model)
        training_use_case._create_data_loader = AsyncMock(return_value=Mock(
            dataset=Mock(__len__=Mock(return_value=100))
        ))
        training_use_case._train_epoch = AsyncMock(return_value=Mock(
            to_dict=lambda: {"loss": 0.5}
        ))
        
        # Mock validation to return worsening metrics
        val_counter = 0
        async def mock_validate(*args):
            nonlocal val_counter
            loss = val_losses[min(val_counter, len(val_losses) - 1)]
            val_counter += 1
            return {"loss": loss}
        
        training_use_case._validate = mock_validate
        
        # Mock training service to stop after patience exceeded
        stop_counter = 0
        def should_stop():
            nonlocal stop_counter
            stop_counter += 1
            return stop_counter > 4  # Stop after 4 epochs
        
        mock_training_service.should_stop.side_effect = should_stop
        
        # Execute
        response = await training_use_case.execute(training_request)
        
        # Verify early stopping
        assert response.success is True
        assert response.early_stopped is True
        assert response.stop_reason == "Early stopping"
        assert response.total_epochs < 10  # Stopped early
    
    @pytest.mark.asyncio
    async def test_training_with_checkpoint_resume(
        self, training_use_case, training_request, temp_dir
    ):
        """Test resuming training from checkpoint."""
        # Create a checkpoint to resume from
        checkpoint_dir = temp_dir / "checkpoints" / "checkpoint-500"
        checkpoint_dir.mkdir(parents=True)
        
        # Save checkpoint metadata
        import json
        metadata = {
            "epoch": 1,
            "global_step": 500,
            "train_loss": 0.4,
            "best_metric": 0.35
        }
        with open(checkpoint_dir / "metadata.json", "w") as f:
            json.dump(metadata, f)
        
        # Mock checkpoint data
        (checkpoint_dir / "weights.safetensors").touch()
        
        # Set resume path
        training_request.resume_from_checkpoint = checkpoint_dir
        
        # Setup mocks
        mock_model = Mock()
        mock_model.parameters.return_value = []
        mock_model.state_dict.return_value = {}
        
        training_use_case._prepare_model = AsyncMock(return_value=mock_model)
        training_use_case._create_data_loader = AsyncMock(return_value=Mock(
            dataset=Mock(__len__=Mock(return_value=100))
        ))
        training_use_case._train_epoch = AsyncMock(return_value=Mock(
            to_dict=lambda: {"loss": 0.3}
        ))
        training_use_case._validate = AsyncMock(return_value={"loss": 0.25})
        
        # Execute
        response = await training_use_case.execute(training_request)
        
        # Verify successful resume
        assert response.success is True
        
        # Verify checkpoint was loaded
        checkpoint_port = training_use_case.checkpoints
        assert checkpoint_port.load_checkpoint.called
    
    @pytest.mark.asyncio
    async def test_training_with_configuration_override(
        self, training_use_case, training_request, temp_dir
    ):
        """Test training with configuration file override."""
        # Create config file
        config_file = temp_dir / "training_config.yaml"
        config_content = """
model:
  hidden_size: 512
  num_hidden_layers: 8
  
training:
  learning_rate: 1e-4
  warmup_ratio: 0.1
  weight_decay: 0.01
"""
        config_file.write_text(config_content)
        
        # Update request to use config file
        training_request.config_file = config_file
        
        # Mock the config loading
        async def mock_load_config(path):
            import yaml
            with open(path) as f:
                return yaml.safe_load(f)
        
        training_use_case.config.load_configuration = mock_load_config
        
        # Setup other mocks
        mock_model = Mock()
        training_use_case._prepare_model = AsyncMock(return_value=mock_model)
        training_use_case._create_data_loader = AsyncMock(return_value=Mock(
            dataset=Mock(__len__=Mock(return_value=100))
        ))
        training_use_case._train_epoch = AsyncMock(return_value=Mock(
            to_dict=lambda: {"loss": 0.5}
        ))
        
        # Execute
        response = await training_use_case.execute(training_request)
        
        assert response.success is True
        
        # Verify config was loaded and used
        # This would be reflected in the logged parameters
        logged_params = training_use_case.monitoring.log_params.call_args[0][0]
        assert logged_params["learning_rate"] == 5e-5  # From request, not config
    
    @pytest.mark.asyncio  
    async def test_training_error_recovery(
        self, training_use_case, training_request
    ):
        """Test error handling and recovery during training."""
        # Simulate error during training
        training_use_case._prepare_model = AsyncMock(
            side_effect=Exception("Model preparation failed")
        )
        
        # Execute
        response = await training_use_case.execute(training_request)
        
        # Verify error handling
        assert response.success is False
        assert response.error_message is not None
        assert "Model preparation failed" in response.error_message
        
        # Verify error was logged
        assert training_use_case.monitoring.log_error.called
        error_log = training_use_case.monitoring.log_error.call_args[0][0]
        assert "Training failed" in error_log
    
    @pytest.mark.asyncio
    async def test_complete_workflow_with_all_features(
        self, training_use_case, training_request, temp_dir
    ):
        """Test complete workflow with all features enabled."""
        # Enable all features
        training_request.use_mixed_precision = True
        training_request.gradient_checkpointing = True
        training_request.save_total_limit = 3
        training_request.load_best_model_at_end = True
        training_request.compute_metrics = ["accuracy", "f1", "precision", "recall"]
        
        # Setup comprehensive mocks
        mock_model = Mock()
        mock_model.parameters.return_value = []
        mock_model.state_dict.return_value = {"encoder": {"layer_1": "weights"}}
        
        training_use_case._prepare_model = AsyncMock(return_value=mock_model)
        training_use_case._create_data_loader = AsyncMock(return_value=Mock(
            dataset=Mock(__len__=Mock(return_value=1000))
        ))
        
        # Mock progressive improvement
        train_losses = [0.7, 0.5, 0.4, 0.35]
        val_metrics = [
            {"loss": 0.6, "accuracy": 0.7, "f1": 0.65},
            {"loss": 0.4, "accuracy": 0.8, "f1": 0.75},
            {"loss": 0.3, "accuracy": 0.85, "f1": 0.82},
            {"loss": 0.25, "accuracy": 0.9, "f1": 0.88}
        ]
        
        epoch_counter = 0
        async def mock_train_epoch(*args):
            nonlocal epoch_counter
            loss = train_losses[min(epoch_counter, len(train_losses) - 1)]
            epoch_counter += 1
            return Mock(to_dict=lambda: {"loss": loss})
        
        val_counter = 0
        async def mock_validate(*args):
            nonlocal val_counter
            metrics = val_metrics[min(val_counter, len(val_metrics) - 1)]
            val_counter += 1
            return metrics
        
        training_use_case._train_epoch = mock_train_epoch
        training_use_case._validate = mock_validate
        
        # Execute
        response = await training_use_case.execute(training_request)
        
        # Comprehensive verification
        assert response.success is True
        assert response.total_epochs == 2
        assert response.best_val_metric > 0.8
        
        # Verify all features were used
        assert response.config_used["use_mixed_precision"] is True
        assert response.config_used["gradient_checkpointing"] is True
        
        # Verify artifacts
        output_dir = temp_dir / "output" / response.run_id
        assert output_dir.exists()
        
        # Verify final model was saved
        assert response.final_model_path is not None
        assert response.final_model_path.exists()
        
        # Verify metrics history
        assert len(response.train_history) > 0
        assert len(response.val_history) > 0
        
        # Verify best model tracking
        assert response.best_model_path is not None