"""Integration tests for hexagonal architecture patterns.

This tests the integration of hexagonal architecture (ports and adapters)
with existing modules to ensure clean separation of concerns and testability.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol
from unittest.mock import MagicMock, create_autospec

import mlx.core as mx
import pytest

from training.core.base import BaseTrainer
from training.core.config import BaseTrainerConfig
from data.core.base import BaseDataset
from models.factory import create_model


# Domain Models (Core Business Logic)
@dataclass
class TrainingRequest:
    """Domain model for a training request."""
    model_config: Dict[str, Any]
    data_config: Dict[str, Any]
    training_config: Dict[str, Any]
    output_path: Path


@dataclass
class TrainingResult:
    """Domain model for training results."""
    model_path: Path
    metrics: Dict[str, float]
    duration_seconds: float
    success: bool
    error_message: Optional[str] = None


# Ports (Interfaces)
class ModelRepositoryPort(Protocol):
    """Port for model storage operations."""
    
    def save_model(self, model: Any, path: Path) -> None:
        """Save a model to storage."""
        ...
    
    def load_model(self, path: Path) -> Any:
        """Load a model from storage."""
        ...
    
    def exists(self, path: Path) -> bool:
        """Check if a model exists."""
        ...


class DataRepositoryPort(Protocol):
    """Port for data access operations."""
    
    def load_dataset(self, path: Path, config: Dict[str, Any]) -> BaseDataset:
        """Load a dataset."""
        ...
    
    def save_predictions(self, predictions: mx.array, path: Path) -> None:
        """Save predictions."""
        ...


class MetricsRepositoryPort(Protocol):
    """Port for metrics storage."""
    
    def save_metrics(self, metrics: Dict[str, float], run_id: str) -> None:
        """Save training metrics."""
        ...
    
    def load_metrics(self, run_id: str) -> Dict[str, float]:
        """Load training metrics."""
        ...


class TrainingPort(Protocol):
    """Port for training operations."""
    
    def train(self, request: TrainingRequest) -> TrainingResult:
        """Execute training."""
        ...
    
    def evaluate(self, model_path: Path, data_path: Path) -> Dict[str, float]:
        """Evaluate a model."""
        ...


# Adapters (Implementations)
class MLXModelAdapter:
    """Adapter for MLX model operations."""
    
    def __init__(self):
        self.storage = {}  # In-memory storage for testing
    
    def save_model(self, model: Any, path: Path) -> None:
        """Save model using MLX format."""
        path.parent.mkdir(parents=True, exist_ok=True)
        # Simulate saving
        self.storage[str(path)] = model
        
    def load_model(self, path: Path) -> Any:
        """Load model from MLX format."""
        if str(path) not in self.storage:
            raise FileNotFoundError(f"Model not found: {path}")
        return self.storage[str(path)]
    
    def exists(self, path: Path) -> bool:
        """Check if model exists."""
        return str(path) in self.storage


class CSVDataAdapter:
    """Adapter for CSV data operations."""
    
    def load_dataset(self, path: Path, config: Dict[str, Any]) -> BaseDataset:
        """Load dataset from CSV."""
        # Mock implementation
        mock_dataset = MagicMock(spec=BaseDataset)
        mock_dataset.__len__.return_value = 100
        mock_dataset.__getitem__.return_value = {
            "text": "sample text",
            "label": 0
        }
        return mock_dataset
    
    def save_predictions(self, predictions: mx.array, path: Path) -> None:
        """Save predictions to CSV."""
        path.parent.mkdir(parents=True, exist_ok=True)
        # Simulate saving
        path.write_text("id,prediction\n0,0\n1,1")


class MLFlowMetricsAdapter:
    """Adapter for MLflow metrics storage."""
    
    def __init__(self):
        self.metrics_store = {}
    
    def save_metrics(self, metrics: Dict[str, float], run_id: str) -> None:
        """Save metrics to MLflow."""
        self.metrics_store[run_id] = metrics
    
    def load_metrics(self, run_id: str) -> Dict[str, float]:
        """Load metrics from MLflow."""
        return self.metrics_store.get(run_id, {})


class TrainingAdapter:
    """Adapter for training operations using existing trainer."""
    
    def __init__(self, model_repo: ModelRepositoryPort, data_repo: DataRepositoryPort):
        self.model_repo = model_repo
        self.data_repo = data_repo
    
    def train(self, request: TrainingRequest) -> TrainingResult:
        """Execute training using BaseTrainer."""
        import time
        start_time = time.time()
        
        try:
            # Create model
            model = create_model(request.model_config)
            
            # Load data
            train_data = self.data_repo.load_dataset(
                Path(request.data_config["train_path"]),
                request.data_config
            )
            
            # Create trainer config
            config = BaseTrainerConfig()
            config.environment.output_dir = request.output_path
            for key, value in request.training_config.items():
                if hasattr(config.training, key):
                    setattr(config.training, key, value)
            
            # Mock training for testing
            final_metrics = {"loss": 0.5, "accuracy": 0.85}
            model_path = request.output_path / "model.safetensors"
            self.model_repo.save_model(model, model_path)
            
            duration = time.time() - start_time
            
            return TrainingResult(
                model_path=model_path,
                metrics=final_metrics,
                duration_seconds=duration,
                success=True
            )
            
        except Exception as e:
            duration = time.time() - start_time
            return TrainingResult(
                model_path=Path(),
                metrics={},
                duration_seconds=duration,
                success=False,
                error_message=str(e)
            )
    
    def evaluate(self, model_path: Path, data_path: Path) -> Dict[str, float]:
        """Evaluate a model."""
        model = self.model_repo.load_model(model_path)
        data = self.data_repo.load_dataset(data_path, {})
        
        # Mock evaluation
        return {"eval_loss": 0.45, "eval_accuracy": 0.88}


# Use Cases (Application Services)
class TrainModelUseCase:
    """Use case for training a model."""
    
    def __init__(
        self,
        training_port: TrainingPort,
        metrics_repo: MetricsRepositoryPort,
        model_repo: ModelRepositoryPort
    ):
        self.training_port = training_port
        self.metrics_repo = metrics_repo
        self.model_repo = model_repo
    
    def execute(self, request: TrainingRequest, run_id: str) -> TrainingResult:
        """Execute the training use case."""
        # Validate request
        if not request.model_config:
            raise ValueError("Model configuration is required")
        
        if not request.data_config.get("train_path"):
            raise ValueError("Training data path is required")
        
        # Execute training
        result = self.training_port.train(request)
        
        # Save metrics if successful
        if result.success:
            self.metrics_repo.save_metrics(result.metrics, run_id)
        
        return result


class EvaluateModelUseCase:
    """Use case for evaluating a model."""
    
    def __init__(
        self,
        training_port: TrainingPort,
        model_repo: ModelRepositoryPort,
        metrics_repo: MetricsRepositoryPort
    ):
        self.training_port = training_port
        self.model_repo = model_repo
        self.metrics_repo = metrics_repo
    
    def execute(self, model_path: Path, data_path: Path, run_id: str) -> Dict[str, float]:
        """Execute the evaluation use case."""
        # Check if model exists
        if not self.model_repo.exists(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        # Evaluate
        metrics = self.training_port.evaluate(model_path, data_path)
        
        # Save evaluation metrics
        self.metrics_repo.save_metrics(metrics, f"{run_id}_eval")
        
        return metrics


@pytest.fixture
def model_adapter():
    """Create model adapter."""
    return MLXModelAdapter()


@pytest.fixture
def data_adapter():
    """Create data adapter."""
    return CSVDataAdapter()


@pytest.fixture
def metrics_adapter():
    """Create metrics adapter."""
    return MLFlowMetricsAdapter()


@pytest.fixture
def training_adapter(model_adapter, data_adapter):
    """Create training adapter."""
    return TrainingAdapter(model_adapter, data_adapter)


class TestHexagonalArchitecture:
    """Test hexagonal architecture integration."""
    
    def test_train_model_use_case(
        self,
        training_adapter,
        metrics_adapter,
        model_adapter,
        tmp_path
    ):
        """Test the train model use case with all adapters."""
        # Create use case
        use_case = TrainModelUseCase(
            training_port=training_adapter,
            metrics_repo=metrics_adapter,
            model_repo=model_adapter
        )
        
        # Create request
        request = TrainingRequest(
            model_config={
                "model_type": "bert_classifier",
                "num_labels": 2
            },
            data_config={
                "train_path": "data/train.csv",
                "batch_size": 32
            },
            training_config={
                "num_epochs": 3,
                "learning_rate": 0.001
            },
            output_path=tmp_path / "output"
        )
        
        # Execute
        result = use_case.execute(request, run_id="test_run_001")
        
        # Verify result
        assert result.success
        assert result.model_path.exists() or model_adapter.exists(result.model_path)
        assert "loss" in result.metrics
        assert result.duration_seconds > 0
        
        # Verify metrics were saved
        saved_metrics = metrics_adapter.load_metrics("test_run_001")
        assert saved_metrics == result.metrics
    
    def test_evaluate_model_use_case(
        self,
        training_adapter,
        metrics_adapter,
        model_adapter,
        tmp_path
    ):
        """Test the evaluate model use case."""
        # First train a model
        model = MagicMock()
        model_path = tmp_path / "model.safetensors"
        model_adapter.save_model(model, model_path)
        
        # Create use case
        use_case = EvaluateModelUseCase(
            training_port=training_adapter,
            model_repo=model_adapter,
            metrics_repo=metrics_adapter
        )
        
        # Execute evaluation
        metrics = use_case.execute(
            model_path=model_path,
            data_path=Path("data/test.csv"),
            run_id="test_run_001"
        )
        
        # Verify metrics
        assert "eval_loss" in metrics
        assert "eval_accuracy" in metrics
        
        # Verify metrics were saved
        saved_metrics = metrics_adapter.load_metrics("test_run_001_eval")
        assert saved_metrics == metrics
    
    def test_port_substitution(self, tmp_path):
        """Test that ports can be substituted with different implementations."""
        # Create mock implementations
        mock_training = create_autospec(TrainingPort)
        mock_metrics = create_autospec(MetricsRepositoryPort)
        mock_model = create_autospec(ModelRepositoryPort)
        
        # Configure mocks
        mock_result = TrainingResult(
            model_path=tmp_path / "model.safetensors",
            metrics={"loss": 0.3},
            duration_seconds=10.0,
            success=True
        )
        mock_training.train.return_value = mock_result
        
        # Create use case with mocks
        use_case = TrainModelUseCase(
            training_port=mock_training,
            metrics_repo=mock_metrics,
            model_repo=mock_model
        )
        
        # Execute
        request = TrainingRequest(
            model_config={"type": "test"},
            data_config={"train_path": "test.csv"},
            training_config={},
            output_path=tmp_path
        )
        
        result = use_case.execute(request, "test_run")
        
        # Verify mocks were called
        mock_training.train.assert_called_once_with(request)
        mock_metrics.save_metrics.assert_called_once_with(
            result.metrics,
            "test_run"
        )
    
    def test_error_handling(self, training_adapter, metrics_adapter, model_adapter, tmp_path):
        """Test error handling in hexagonal architecture."""
        # Create use case
        use_case = TrainModelUseCase(
            training_port=training_adapter,
            metrics_repo=metrics_adapter,
            model_repo=model_adapter
        )
        
        # Test with invalid request
        invalid_request = TrainingRequest(
            model_config={},  # Empty config
            data_config={},   # Missing train_path
            training_config={},
            output_path=tmp_path
        )
        
        # Should raise validation error
        with pytest.raises(ValueError, match="Model configuration is required"):
            use_case.execute(invalid_request, "test_run")
    
    def test_adapter_isolation(self):
        """Test that adapters are properly isolated from each other."""
        # Create independent adapters
        model_adapter1 = MLXModelAdapter()
        model_adapter2 = MLXModelAdapter()
        
        # Save to first adapter
        model = MagicMock()
        path = Path("test_model.safetensors")
        model_adapter1.save_model(model, path)
        
        # Should exist in first adapter
        assert model_adapter1.exists(path)
        
        # Should not exist in second adapter
        assert not model_adapter2.exists(path)
    
    def test_dependency_injection(self, tmp_path):
        """Test that dependencies can be injected at runtime."""
        # Create a custom metrics adapter
        class CustomMetricsAdapter:
            def __init__(self):
                self.calls = []
            
            def save_metrics(self, metrics, run_id):
                self.calls.append(("save", metrics, run_id))
            
            def load_metrics(self, run_id):
                self.calls.append(("load", run_id))
                return {}
        
        # Inject custom adapter
        custom_metrics = CustomMetricsAdapter()
        training_adapter = TrainingAdapter(
            MLXModelAdapter(),
            CSVDataAdapter()
        )
        
        use_case = TrainModelUseCase(
            training_port=training_adapter,
            metrics_repo=custom_metrics,
            model_repo=MLXModelAdapter()
        )
        
        # Execute
        request = TrainingRequest(
            model_config={"model_type": "test"},
            data_config={"train_path": "test.csv"},
            training_config={},
            output_path=tmp_path
        )
        
        use_case.execute(request, "custom_run")
        
        # Verify custom adapter was used
        assert len(custom_metrics.calls) > 0
        assert custom_metrics.calls[0][0] == "save"
        assert custom_metrics.calls[0][2] == "custom_run"
    
    def test_use_case_composition(
        self,
        training_adapter,
        metrics_adapter,
        model_adapter,
        tmp_path
    ):
        """Test composing multiple use cases."""
        # Create use cases
        train_use_case = TrainModelUseCase(
            training_adapter,
            metrics_adapter,
            model_adapter
        )
        
        eval_use_case = EvaluateModelUseCase(
            training_adapter,
            model_adapter,
            metrics_adapter
        )
        
        # Execute training
        train_request = TrainingRequest(
            model_config={"model_type": "bert_classifier", "num_labels": 2},
            data_config={"train_path": "train.csv"},
            training_config={"num_epochs": 1},
            output_path=tmp_path
        )
        
        train_result = train_use_case.execute(train_request, "composite_run")
        
        # Execute evaluation on trained model
        eval_metrics = eval_use_case.execute(
            train_result.model_path,
            Path("test.csv"),
            "composite_run"
        )
        
        # Both should succeed
        assert train_result.success
        assert eval_metrics is not None
        
        # Metrics should be saved for both
        train_metrics = metrics_adapter.load_metrics("composite_run")
        eval_metrics_saved = metrics_adapter.load_metrics("composite_run_eval")
        
        assert train_metrics == train_result.metrics
        assert eval_metrics_saved == eval_metrics