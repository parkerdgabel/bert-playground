"""pytest configuration and fixtures for CLI tests."""

import json
import tempfile
from collections.abc import Generator
from pathlib import Path
from typing import Any
from unittest.mock import Mock, patch

import pandas as pd
import pytest
import yaml
from typer.testing import CliRunner

# ==============================================================================
# Core Testing Fixtures
# ==============================================================================


@pytest.fixture
def runner() -> CliRunner:
    """Create a CLI runner for testing commands."""
    return CliRunner()


@pytest.fixture
def isolated_runner() -> CliRunner:
    """Create an isolated CLI runner that doesn't affect the system."""
    return CliRunner(env={"HOME": "/tmp"})


# ==============================================================================
# Temporary Directory and Project Fixtures
# ==============================================================================


@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """Create a temporary directory that's cleaned up after the test."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def temp_project(temp_dir: Path) -> Generator[Path, None, None]:
    """Create a temporary BERT project structure."""
    # Create project directories
    dirs = ["data/titanic", "configs", "output", "models", "mlruns"]
    for dir_path in dirs:
        (temp_dir / dir_path).mkdir(parents=True, exist_ok=True)

    # Create sample configuration
    config_path = temp_dir / "configs" / "test.yaml"
    config = {
        "model": {"type": "bert", "hidden_size": 128, "num_layers": 2, "num_heads": 2},
        "training": {"epochs": 1, "batch_size": 4, "learning_rate": 1e-4},
    }
    with open(config_path, "w") as f:
        yaml.dump(config, f)

    # Create sample data files
    sample_data = pd.DataFrame(
        {
            "PassengerId": [1, 2, 3],
            "Pclass": [1, 2, 3],
            "Sex": ["male", "female", "male"],
            "Age": [22, 38, 26],
            "Survived": [0, 1, 0],
        }
    )

    train_path = temp_dir / "data" / "titanic" / "train.csv"
    val_path = temp_dir / "data" / "titanic" / "val.csv"
    test_path = temp_dir / "data" / "titanic" / "test.csv"

    sample_data.to_csv(train_path, index=False)
    sample_data.to_csv(val_path, index=False)
    sample_data.drop("Survived", axis=1).to_csv(test_path, index=False)

    # Change to temp directory
    original_cwd = Path.cwd()
    import os

    os.chdir(temp_dir)

    try:
        yield temp_dir
    finally:
        os.chdir(original_cwd)


# ==============================================================================
# Configuration Fixtures
# ==============================================================================


@pytest.fixture
def mock_config() -> dict[str, Any]:
    """Create a mock configuration for testing."""
    return {
        "model": {
            "type": "bert",
            "hidden_size": 128,
            "num_layers": 2,
            "num_heads": 2,
            "max_length": 64,
        },
        "training": {
            "epochs": 2,
            "batch_size": 8,
            "learning_rate": 2e-5,
            "gradient_accumulation_steps": 1,
            "warmup_steps": 100,
            "weight_decay": 0.01,
        },
        "data": {"num_workers": 2, "prefetch_size": 2, "shuffle": True},
        "logging": {"level": "INFO", "log_every_n_steps": 10},
    }


@pytest.fixture
def quick_config() -> dict[str, Any]:
    """Create a quick test configuration."""
    return {
        "model": {"type": "bert", "hidden_size": 64, "num_layers": 1, "num_heads": 1},
        "training": {"epochs": 1, "batch_size": 2, "learning_rate": 1e-4},
    }


@pytest.fixture
def config_file(temp_dir: Path, mock_config: dict[str, Any]) -> Path:
    """Create a temporary configuration file."""
    config_path = temp_dir / "config.yaml"
    with open(config_path, "w") as f:
        yaml.dump(mock_config, f)
    return config_path


@pytest.fixture
def json_config_file(temp_dir: Path, mock_config: dict[str, Any]) -> Path:
    """Create a temporary JSON configuration file."""
    config_path = temp_dir / "config.json"
    with open(config_path, "w") as f:
        json.dump(mock_config, f)
    return config_path


# ==============================================================================
# External Service Mocks
# ==============================================================================


@pytest.fixture
def mock_kaggle_api():
    """Mock Kaggle API for testing."""
    with patch("kaggle.api") as mock_api:
        # Mock competition data
        mock_api.competitions_list.return_value = [
            Mock(ref="titanic", title="Titanic - Machine Learning from Disaster"),
            Mock(ref="house-prices", title="House Prices - Advanced Regression"),
        ]

        # Mock dataset data
        mock_api.dataset_list.return_value = [
            Mock(ref="titanic/train", title="Titanic Training Data"),
            Mock(ref="titanic/test", title="Titanic Test Data"),
        ]

        # Mock leaderboard data
        mock_api.competition_leaderboard_view.return_value = {
            "submissions": [
                {"teamName": "Team1", "score": 0.85},
                {"teamName": "Team2", "score": 0.83},
            ]
        }

        # Mock submission history
        mock_api.competition_submissions.return_value = [
            Mock(date="2024-01-01", score=0.80, status="complete"),
            Mock(date="2024-01-02", score=0.82, status="complete"),
        ]

        yield mock_api


@pytest.fixture
def mock_mlflow():
    """Mock MLflow for testing."""
    with patch("mlflow") as mock_mlflow:
        # Mock experiment functions
        mock_mlflow.create_experiment.return_value = "exp-123"
        mock_mlflow.get_experiment_by_name.return_value = Mock(
            experiment_id="exp-123", name="test_experiment"
        )

        # Mock run functions
        mock_mlflow.start_run.return_value.__enter__.return_value = Mock(
            info=Mock(run_id="run-123")
        )

        # Mock logging functions
        mock_mlflow.log_param.return_value = None
        mock_mlflow.log_metric.return_value = None
        mock_mlflow.log_artifacts.return_value = None

        yield mock_mlflow


@pytest.fixture
def mock_mlflow_client():
    """Mock MLflow client for testing."""
    with patch("mlflow.tracking.MlflowClient") as mock_client_class:
        mock_client = Mock()
        mock_client_class.return_value = mock_client

        # Mock search functions
        mock_client.search_experiments.return_value = [
            Mock(experiment_id="1", name="exp1"),
            Mock(experiment_id="2", name="exp2"),
        ]

        mock_client.search_runs.return_value = [
            Mock(
                info=Mock(run_id="run1", status="FINISHED"),
                data=Mock(metrics={"accuracy": 0.85}),
            )
        ]

        yield mock_client


@pytest.fixture
def mock_model_server():
    """Mock model server for testing."""
    with patch("uvicorn.run") as mock_run:
        yield mock_run


# ==============================================================================
# Model and Training Mocks
# ==============================================================================


@pytest.fixture
def mock_model():
    """Create a mock model for testing."""
    model = Mock()
    model.save.return_value = None
    model.load.return_value = model
    model.parameters.return_value = [Mock(shape=(10, 10))]
    return model


@pytest.fixture
def mock_trainer():
    """Create a mock trainer for testing."""
    with patch("training.core.trainer.BertTrainer") as mock_trainer_class:
        trainer = Mock()
        mock_trainer_class.return_value = trainer

        # Mock training methods
        trainer.train.return_value = {
            "train_loss": 0.5,
            "val_loss": 0.4,
            "val_accuracy": 0.85,
        }

        trainer.predict.return_value = pd.DataFrame(
            {"PassengerId": [1, 2, 3], "Survived": [0, 1, 0]}
        )

        yield trainer


@pytest.fixture
def mock_checkpoint(temp_dir: Path) -> Path:
    """Create a mock checkpoint directory."""
    checkpoint_dir = temp_dir / "checkpoint"
    checkpoint_dir.mkdir(parents=True)

    # Create mock checkpoint files
    (checkpoint_dir / "model.safetensors").touch()
    (checkpoint_dir / "config.json").write_text('{"model_type": "bert"}')
    (checkpoint_dir / "training_state.json").write_text('{"epoch": 5}')

    return checkpoint_dir


# ==============================================================================
# Data Fixtures
# ==============================================================================


@pytest.fixture
def sample_csv_data(temp_dir: Path) -> Path:
    """Create sample CSV data for testing."""
    data = pd.DataFrame(
        {
            "feature1": [1, 2, 3, 4, 5],
            "feature2": ["a", "b", "c", "d", "e"],
            "label": [0, 1, 0, 1, 0],
        }
    )

    file_path = temp_dir / "sample_data.csv"
    data.to_csv(file_path, index=False)
    return file_path


@pytest.fixture
def titanic_data(temp_dir: Path) -> dict[str, Path]:
    """Create sample Titanic dataset for testing."""
    train_data = pd.DataFrame(
        {
            "PassengerId": list(range(1, 101)),
            "Pclass": [1, 2, 3] * 33 + [1],
            "Sex": ["male", "female"] * 50,
            "Age": [22 + i % 50 for i in range(100)],
            "SibSp": [0, 1, 2] * 33 + [0],
            "Parch": [0, 1] * 50,
            "Fare": [10 + i * 2 for i in range(100)],
            "Embarked": ["S", "C", "Q"] * 33 + ["S"],
            "Survived": [0, 1] * 50,
        }
    )

    test_data = train_data.drop("Survived", axis=1).iloc[80:100]
    val_data = train_data.iloc[60:80]
    train_data = train_data.iloc[:60]

    paths = {}
    for name, data in [("train", train_data), ("val", val_data), ("test", test_data)]:
        path = temp_dir / f"{name}.csv"
        data.to_csv(path, index=False)
        paths[name] = path

    return paths


# ==============================================================================
# Utility Fixtures
# ==============================================================================


@pytest.fixture
def capture_output():
    """Capture stdout and stderr for testing console output."""
    import io
    import sys

    class OutputCapture:
        def __init__(self):
            self.stdout = io.StringIO()
            self.stderr = io.StringIO()
            self._original_stdout = sys.stdout
            self._original_stderr = sys.stderr

        def __enter__(self):
            sys.stdout = self.stdout
            sys.stderr = self.stderr
            return self

        def __exit__(self, *args):
            sys.stdout = self._original_stdout
            sys.stderr = self._original_stderr

    return OutputCapture()


@pytest.fixture
def env_vars():
    """Temporarily set environment variables."""
    import os

    original_env = os.environ.copy()

    def _set_env(**kwargs):
        os.environ.update(kwargs)

    yield _set_env

    # Restore original environment
    os.environ.clear()
    os.environ.update(original_env)


@pytest.fixture
def mock_subprocess():
    """Mock subprocess calls for testing external commands."""
    with patch("subprocess.run") as mock_run:
        mock_run.return_value = Mock(returncode=0, stdout="Success", stderr="")
        yield mock_run


# ==============================================================================
# Validation Fixtures
# ==============================================================================


@pytest.fixture
def invalid_paths() -> list[str]:
    """List of invalid file paths for testing validation."""
    return [
        "/nonexistent/file.txt",
        "relative/path/without/root",
        "/etc/passwd",  # System file
        "file_without_extension",
        "file.invalid_extension",
    ]


@pytest.fixture
def valid_paths(temp_dir: Path) -> list[Path]:
    """Create and return valid file paths for testing."""
    paths = []
    for name in ["file1.csv", "file2.json", "file3.yaml"]:
        path = temp_dir / name
        path.touch()
        paths.append(path)
    return paths


# ==============================================================================
# Performance Testing Fixtures
# ==============================================================================


@pytest.fixture
def benchmark_timer():
    """Simple timer for benchmarking operations."""
    import time

    class Timer:
        def __init__(self):
            self.start_time = None
            self.elapsed = None

        def __enter__(self):
            self.start_time = time.time()
            return self

        def __exit__(self, *args):
            self.elapsed = time.time() - self.start_time

    return Timer()


@pytest.fixture
def memory_tracker():
    """Track memory usage for testing."""
    import os

    import psutil

    class MemoryTracker:
        def __init__(self):
            self.process = psutil.Process(os.getpid())
            self.start_memory = None
            self.peak_memory = None

        def __enter__(self):
            self.start_memory = self.process.memory_info().rss / 1024 / 1024  # MB
            return self

        def __exit__(self, *args):
            self.peak_memory = self.process.memory_info().rss / 1024 / 1024  # MB

        @property
        def memory_used(self):
            return self.peak_memory - self.start_memory if self.peak_memory else 0

    return MemoryTracker()
