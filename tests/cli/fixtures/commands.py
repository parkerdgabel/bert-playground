"""Command-specific fixtures for CLI testing."""

from pathlib import Path
from unittest.mock import Mock, patch

import pytest
from typer import Context

from cli.commands.core import CoreCommands
from cli.commands.kaggle import KaggleCommands
from cli.commands.mlflow import MLflowCommands
from cli.commands.model import ModelCommands

# ==============================================================================
# Command Instance Fixtures
# ==============================================================================


@pytest.fixture
def core_commands() -> CoreCommands:
    """Create CoreCommands instance for testing."""
    return CoreCommands()


@pytest.fixture
def kaggle_commands() -> KaggleCommands:
    """Create KaggleCommands instance for testing."""
    return KaggleCommands()


@pytest.fixture
def mlflow_commands() -> MLflowCommands:
    """Create MLflowCommands instance for testing."""
    return MLflowCommands()


@pytest.fixture
def model_commands() -> ModelCommands:
    """Create ModelCommands instance for testing."""
    return ModelCommands()


# ==============================================================================
# Command Context Fixtures
# ==============================================================================


@pytest.fixture
def cli_context() -> Context:
    """Create a Typer context for testing."""
    ctx = Context(command=Mock())
    ctx.obj = {"config": {}, "verbose": False, "quiet": False}
    return ctx


@pytest.fixture
def verbose_context() -> Context:
    """Create a verbose Typer context."""
    ctx = Context(command=Mock())
    ctx.obj = {"config": {}, "verbose": True, "quiet": False}
    return ctx


# ==============================================================================
# Command Argument Fixtures
# ==============================================================================


@pytest.fixture
def train_args() -> dict:
    """Common training command arguments."""
    return {
        "train_path": "data/train.csv",
        "val_path": "data/val.csv",
        "model_type": "bert",
        "batch_size": 32,
        "epochs": 5,
        "learning_rate": 2e-5,
        "max_length": 128,
        "num_workers": 4,
        "output_dir": "output",
        "experiment_name": "test_experiment",
        "config_path": None,
        "resume_from": None,
    }


@pytest.fixture
def predict_args() -> dict:
    """Common prediction command arguments."""
    return {
        "test_path": "data/test.csv",
        "checkpoint_path": "output/checkpoint",
        "output_path": "predictions.csv",
        "batch_size": 64,
        "num_workers": 4,
        "tta_rounds": 0,
    }


@pytest.fixture
def benchmark_args() -> dict:
    """Common benchmark command arguments."""
    return {
        "model_type": "bert",
        "batch_size": 32,
        "seq_length": 128,
        "steps": 10,
        "warmup_steps": 3,
        "profile": False,
    }


@pytest.fixture
def kaggle_competition_args() -> dict:
    """Common Kaggle competition arguments."""
    return {
        "competition": "titanic",
        "path": "data/titanic",
        "unzip": True,
        "force": False,
    }


@pytest.fixture
def kaggle_submit_args() -> dict:
    """Common Kaggle submission arguments."""
    return {
        "competition": "titanic",
        "file_path": "submission.csv",
        "message": "Test submission",
        "quiet": False,
    }


@pytest.fixture
def mlflow_server_args() -> dict:
    """Common MLflow server arguments."""
    return {
        "backend_store_uri": "./mlruns",
        "default_artifact_root": "./mlartifacts",
        "host": "127.0.0.1",
        "port": 5000,
        "workers": 4,
        "env": "production",
    }


@pytest.fixture
def model_serve_args() -> dict:
    """Common model serving arguments."""
    return {
        "model_path": "output/model",
        "host": "0.0.0.0",
        "port": 8080,
        "workers": 1,
        "reload": False,
    }


# ==============================================================================
# Command Result Fixtures
# ==============================================================================


@pytest.fixture
def successful_train_result() -> dict:
    """Mock successful training result."""
    return {
        "status": "success",
        "final_epoch": 5,
        "best_epoch": 3,
        "best_metric": 0.85,
        "train_loss": 0.15,
        "val_loss": 0.18,
        "output_dir": "output/run_001",
    }


@pytest.fixture
def successful_predict_result() -> dict:
    """Mock successful prediction result."""
    return {
        "status": "success",
        "num_predictions": 100,
        "output_file": "predictions.csv",
        "inference_time": 2.5,
    }


@pytest.fixture
def kaggle_competitions_result() -> list[dict]:
    """Mock Kaggle competitions list."""
    return [
        {
            "ref": "titanic",
            "title": "Titanic - Machine Learning from Disaster",
            "deadline": "2030-01-01",
            "reward": "$0",
            "teamCount": 15000,
        },
        {
            "ref": "house-prices",
            "title": "House Prices - Advanced Regression",
            "deadline": "2030-01-01",
            "reward": "$0",
            "teamCount": 5000,
        },
    ]


@pytest.fixture
def mlflow_experiments_result() -> list[dict]:
    """Mock MLflow experiments list."""
    return [
        {
            "experiment_id": "1",
            "name": "Default",
            "artifact_location": "./mlruns/1",
            "lifecycle_stage": "active",
        },
        {
            "experiment_id": "2",
            "name": "titanic_bert",
            "artifact_location": "./mlruns/2",
            "lifecycle_stage": "active",
        },
    ]


# ==============================================================================
# Command Mock Fixtures
# ==============================================================================


@pytest.fixture
def mock_train_command():
    """Mock the train command execution."""
    with patch("cli.commands.core.core_commands.train_model") as mock_train:
        mock_train.return_value = {
            "status": "success",
            "output_dir": "output/run_001",
            "best_metric": 0.85,
        }
        yield mock_train


@pytest.fixture
def mock_predict_command():
    """Mock the predict command execution."""
    with patch("cli.commands.core.core_commands.generate_predictions") as mock_predict:
        mock_predict.return_value = {
            "status": "success",
            "predictions_file": "predictions.csv",
        }
        yield mock_predict


@pytest.fixture
def mock_kaggle_download():
    """Mock Kaggle download functionality."""
    with patch("cli.commands.kaggle.kaggle_commands.download_competition") as mock_dl:
        mock_dl.return_value = {"status": "success", "files": ["train.csv", "test.csv"]}
        yield mock_dl


@pytest.fixture
def mock_mlflow_server():
    """Mock MLflow server operations."""
    with patch("cli.commands.mlflow.mlflow_commands.MLflowServer") as mock_server:
        server_instance = Mock()
        mock_server.return_value = server_instance
        server_instance.start.return_value = True
        server_instance.stop.return_value = True
        server_instance.is_running.return_value = True
        yield server_instance


# ==============================================================================
# Command Validation Fixtures
# ==============================================================================


@pytest.fixture
def invalid_train_args() -> list[dict]:
    """Invalid training argument combinations for testing."""
    return [
        {"batch_size": -1},  # Negative batch size
        {"learning_rate": 2.0},  # Learning rate too high
        {"epochs": 0},  # Zero epochs
        {"max_length": 100000},  # Sequence too long
        {"train_path": "nonexistent.csv"},  # Missing file
    ]


@pytest.fixture
def invalid_model_types() -> list[str]:
    """Invalid model type values."""
    return ["gpt", "t5", "invalid_model", ""]


@pytest.fixture
def invalid_ports() -> list[int]:
    """Invalid port numbers."""
    return [-1, 0, 70000, 80, 443]  # Including privileged ports


# ==============================================================================
# Command Chain Fixtures
# ==============================================================================


@pytest.fixture
def train_predict_chain():
    """Mock a training followed by prediction workflow."""

    def _chain(train_args: dict, predict_args: dict):
        # Simulate training
        checkpoint_path = Path("output/checkpoint")
        checkpoint_path.mkdir(parents=True, exist_ok=True)

        # Update predict args with checkpoint
        predict_args["checkpoint_path"] = str(checkpoint_path)

        return {
            "train_result": {"status": "success", "checkpoint": str(checkpoint_path)},
            "predict_result": {"status": "success", "predictions": "submission.csv"},
        }

    return _chain


@pytest.fixture
def full_kaggle_workflow():
    """Mock a complete Kaggle competition workflow."""

    def _workflow(competition: str):
        return {
            "download": {"status": "success", "files": ["train.csv", "test.csv"]},
            "train": {"status": "success", "model": "output/model"},
            "predict": {"status": "success", "file": "submission.csv"},
            "submit": {"status": "success", "submission_id": "12345"},
        }

    return _workflow
