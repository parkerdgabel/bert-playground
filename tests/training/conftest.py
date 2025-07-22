"""
Pytest configuration and shared fixtures for training module tests.
"""

# Add project root to path
import sys
import tempfile
from pathlib import Path
from typing import Any

import mlx.core as mx
import mlx.nn as nn
import pytest

sys.path.append(str(Path(__file__).parent.parent.parent))

from training.core.config import BaseTrainerConfig
from training.core.state import TrainingState


# Test configuration
@pytest.fixture(scope="session")
def test_config():
    """Shared test configuration."""
    return {
        "seed": 42,
        "batch_size": 4,
        "num_epochs": 2,
        "learning_rate": 1e-3,
    }


# Temporary directories
@pytest.fixture
def tmp_output_dir():
    """Create temporary output directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def tmp_checkpoint_dir():
    """Create temporary checkpoint directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir) / "checkpoints"


# Mock implementations
class MockModel(nn.Module):
    """Mock model for testing."""

    def __init__(self, input_dim: int = 10, output_dim: int = 2):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        self.input_dim = input_dim
        self.output_dim = output_dim

    def __call__(self, batch: dict[str, mx.array]) -> dict[str, mx.array]:
        """Forward pass returns dict with loss and logits."""
        x = batch.get("input", batch.get("x"))
        y = batch.get("labels", batch.get("y"))

        logits = self.linear(x)
        
        # Compute appropriate loss based on output dimensions
        if self.output_dim > 1 and y is not None:
            # Classification case - use cross entropy
            import mlx.nn as nn
            loss = nn.losses.cross_entropy(logits, y, reduction="mean")
        elif y is not None:
            # Regression case - use MSE
            loss = mx.mean((logits - y) ** 2)
        else:
            # No labels - dummy loss
            loss = mx.array(0.0)

        return {
            "loss": loss,
            "logits": logits,
        }

    def loss(self, batch: dict[str, mx.array]) -> mx.array:
        """Compute loss for batch."""
        outputs = self(batch)
        return outputs["loss"]


class MockDataLoader:
    """Mock data loader for testing."""

    def __init__(
        self,
        num_samples: int = 100,
        batch_size: int = 4,
        input_dim: int = 10,
        output_dim: int = 2,
        num_batches: int | None = None,
    ):
        self.num_samples = num_samples
        self.batch_size = batch_size
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_batches = num_batches or (num_samples // batch_size)
        self._index = 0

    def __iter__(self):
        self._index = 0
        return self

    def __next__(self):
        if self._index >= self.num_batches:
            raise StopIteration

        # Generate random batch
        x = mx.random.normal((self.batch_size, self.input_dim))
        y = mx.random.randint(0, self.output_dim, (self.batch_size,))

        self._index += 1
        return {"input": x, "labels": y}

    def __len__(self):
        return self.num_batches


# Model fixtures
@pytest.fixture
def mock_model():
    """Create mock model."""
    return MockModel()


@pytest.fixture
def mock_binary_model():
    """Create mock binary classification model."""
    return MockModel(input_dim=10, output_dim=2)


@pytest.fixture
def mock_multiclass_model():
    """Create mock multiclass model."""
    return MockModel(input_dim=10, output_dim=5)


@pytest.fixture
def mock_regression_model():
    """Create mock regression model."""
    return MockModel(input_dim=10, output_dim=1)


# DataLoader fixtures
@pytest.fixture
def mock_train_loader():
    """Create mock training data loader."""
    return MockDataLoader(num_samples=100, batch_size=4)


@pytest.fixture
def mock_val_loader():
    """Create mock validation data loader."""
    return MockDataLoader(num_samples=20, batch_size=4)


@pytest.fixture
def mock_test_loader():
    """Create mock test data loader."""
    return MockDataLoader(num_samples=40, batch_size=4)


# Configuration fixtures
@pytest.fixture
def base_config(tmp_output_dir):
    """Create base trainer configuration."""
    return BaseTrainerConfig(
        optimizer={"type": "adam", "learning_rate": 1e-3},
        training={
            "num_epochs": 2,
            "eval_strategy": "epoch",
            "save_best_only": True,
        },
        environment={
            "output_dir": tmp_output_dir,
            "seed": 42,
        },
    )


@pytest.fixture
def quick_test_config(tmp_output_dir):
    """Create quick test configuration."""
    return BaseTrainerConfig(
        optimizer={"type": "adam", "learning_rate": 1e-3},
        training={"num_epochs": 1, "eval_strategy": "steps", "eval_steps": 2},
        data={"batch_size": 2},
        environment={"output_dir": tmp_output_dir, "seed": 42},
    )


# Utility fixtures
@pytest.fixture
def assert_metrics_close():
    """Utility for asserting metrics are close."""

    def _assert(actual: float, expected: float, rtol: float = 1e-5):
        assert abs(actual - expected) < rtol, f"Expected {expected}, got {actual}"

    return _assert


@pytest.fixture
def create_dummy_checkpoint(tmp_checkpoint_dir):
    """Create dummy checkpoint for testing."""

    def _create(epoch: int = 1, step: int = 10):
        checkpoint_dir = tmp_checkpoint_dir / f"checkpoint-{step}"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Create dummy model file
        (checkpoint_dir / "model.safetensors").touch()

        # Create state file
        state = TrainingState(
            epoch=epoch,
            global_step=step,
            best_metric=0.9,
            metrics_history={"loss": [0.5, 0.4, 0.3]},
        )
        import json

        with open(checkpoint_dir / "training_state.json", "w") as f:
            json.dump(state.__dict__, f)

        return checkpoint_dir

    return _create


# MLflow fixtures
@pytest.fixture
def mock_mlflow(monkeypatch):
    """Mock MLflow for testing."""
    import mlflow

    class MockMlflow:
        def __init__(self):
            self.logged_metrics = {}
            self.logged_params = {}
            self.logged_artifacts = []
            self.active_run = None

        def start_run(self, **kwargs):
            self.active_run = "mock-run-id"
            return self

        def end_run(self):
            self.active_run = None

        def log_metric(self, key: str, value: float, step: int | None = None):
            if key not in self.logged_metrics:
                self.logged_metrics[key] = []
            self.logged_metrics[key].append((value, step))

        def log_metrics(self, metrics: dict[str, float], step: int | None = None):
            for key, value in metrics.items():
                self.log_metric(key, value, step)

        def log_param(self, key: str, value: Any):
            self.logged_params[key] = value

        def log_params(self, params: dict[str, Any]):
            self.logged_params.update(params)

        def log_artifact(self, local_path: str, artifact_path: str | None = None):
            self.logged_artifacts.append((local_path, artifact_path))

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

    mock = MockMlflow()
    monkeypatch.setattr(mlflow, "start_run", mock.start_run)
    monkeypatch.setattr(mlflow, "end_run", mock.end_run)
    monkeypatch.setattr(mlflow, "log_metric", mock.log_metric)
    monkeypatch.setattr(mlflow, "log_metrics", mock.log_metrics)
    monkeypatch.setattr(mlflow, "log_param", mock.log_param)
    monkeypatch.setattr(mlflow, "log_params", mock.log_params)
    monkeypatch.setattr(mlflow, "log_artifact", mock.log_artifact)

    return mock


# Kaggle fixtures
@pytest.fixture
def mock_kaggle_api(monkeypatch):
    """Mock Kaggle API for testing."""

    class MockKaggleApi:
        def __init__(self):
            self.submissions = []

        def competition_submit(self, file_path: str, message: str, competition: str):
            self.submissions.append(
                {
                    "file": file_path,
                    "message": message,
                    "competition": competition,
                }
            )

    mock = MockKaggleApi()
    return mock


# Test markers
def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line("markers", "unit: Unit tests")
    config.addinivalue_line("markers", "integration: Integration tests")
    config.addinivalue_line("markers", "e2e: End-to-end tests")
    config.addinivalue_line("markers", "slow: Slow tests")
    config.addinivalue_line("markers", "mlx: MLX-specific tests")
    config.addinivalue_line("markers", "mlflow: MLflow integration tests")
    config.addinivalue_line("markers", "kaggle: Kaggle functionality tests")


# Pytest plugins
def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers based on test location."""
    for item in items:
        # Add markers based on test file location
        if "unit" in str(item.fspath):
            item.add_marker(pytest.mark.unit)
        elif "integration" in str(item.fspath):
            item.add_marker(pytest.mark.integration)
        elif "e2e" in str(item.fspath):
            item.add_marker(pytest.mark.e2e)

        # Add MLX marker to all tests
        item.add_marker(pytest.mark.mlx)
