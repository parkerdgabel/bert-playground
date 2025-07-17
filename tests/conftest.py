"""Shared pytest fixtures and configuration."""

import os
import shutil
import tempfile
from pathlib import Path
from typing import Generator

import mlx.core as mx
import pandas as pd
import pytest
from loguru import logger

from models.factory import create_model
from models.classification import TitanicClassifier
from training.mlx_trainer import MLXTrainer
from training.config import TrainingConfig
from utils.mlflow_central import MLflowCentral


@pytest.fixture(autouse=True)
def setup_test_logging():
    """Configure logging for tests."""
    logger.remove()
    logger.add(
        lambda msg: None,  # Suppress logs during tests
        level="ERROR",
    )
    yield
    logger.remove()


@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """Create a temporary directory for test files."""
    temp_path = Path(tempfile.mkdtemp())
    yield temp_path
    shutil.rmtree(temp_path)


@pytest.fixture
def sample_titanic_data(temp_dir: Path) -> Path:
    """Create sample Titanic dataset for testing."""
    data = {
        "PassengerId": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        "Survived": [0, 1, 1, 1, 0, 0, 0, 0, 1, 1],
        "Pclass": [3, 1, 3, 1, 3, 3, 1, 3, 3, 2],
        "Name": ["Braund, Mr. Owen Harris", "Cumings, Mrs. John Bradley", "Heikkinen, Miss. Laina",
                 "Futrelle, Mrs. Jacques Heath", "Allen, Mr. William Henry", "Moran, Mr. James",
                 "McCarthy, Mr. Timothy J", "Palsson, Master. Gosta Leonard", "Johnson, Mrs. Oscar W",
                 "Nasser, Mrs. Nicholas"],
        "Sex": ["male", "female", "female", "female", "male", "male", "male", "male", "female", "female"],
        "Age": [22.0, 38.0, 26.0, 35.0, 35.0, None, 54.0, 2.0, 27.0, 14.0],
        "SibSp": [1, 1, 0, 1, 0, 0, 0, 3, 0, 1],
        "Parch": [0, 0, 0, 0, 0, 0, 0, 1, 2, 0],
        "Fare": [7.25, 71.28, 7.92, 53.10, 8.05, 8.46, 51.86, 21.08, 11.13, 30.07],
        "Embarked": ["S", "C", "S", "S", "S", "Q", "S", "S", "S", "C"],
        "Cabin": [None, "C85", None, "C123", None, None, "E46", None, None, None],
    }
    
    df = pd.DataFrame(data)
    csv_path = temp_dir / "train.csv"
    # Convert all columns to strings and fill NA to ensure MLX CSV reader compatibility
    for col in df.columns:
        df[col] = df[col].astype(str).replace('nan', '')
    df.to_csv(csv_path, index=False)
    
    return csv_path


@pytest.fixture
def sample_test_data(temp_dir: Path) -> Path:
    """Create sample test dataset without labels."""
    data = {
        "PassengerId": [892, 893, 894, 895, 896],
        "Pclass": [3, 3, 2, 3, 3],
        "Name": ["Kelly, Mr. James", "Wilkes, Mrs. James", "Myles, Mr. Thomas Francis",
                 "Wirz, Mr. Albert", "Hirvonen, Mrs. Alexander"],
        "Sex": ["male", "female", "male", "male", "female"],
        "Age": [34.5, 47.0, 62.0, 27.0, 22.0],
        "SibSp": [0, 1, 0, 0, 1],
        "Parch": [0, 0, 0, 0, 1],
        "Fare": [7.83, 7.00, 9.69, 8.66, 12.29],
        "Embarked": ["Q", "S", "Q", "S", "S"],
        "Cabin": [None, None, None, None, None],
    }
    
    df = pd.DataFrame(data)
    csv_path = temp_dir / "test.csv"
    # Convert all columns to strings and fill NA to ensure MLX CSV reader compatibility
    for col in df.columns:
        df[col] = df[col].astype(str).replace('nan', '')
    df.to_csv(csv_path, index=False)
    
    return csv_path


@pytest.fixture
def base_model():
    """Create a base ModernBERT model for testing."""
    return create_model("standard")


@pytest.fixture
def titanic_classifier(base_model):
    """Create a TitanicClassifier for testing."""
    return TitanicClassifier(base_model)


@pytest.fixture
def training_config(temp_dir: Path) -> TrainingConfig:
    """Create a minimal training configuration for testing."""
    return TrainingConfig(
        learning_rate=2e-5,
        epochs=1,
        batch_size=4,
        output_dir=str(temp_dir / "output")
    )


@pytest.fixture
def mlx_trainer(titanic_classifier, training_config) -> MLXTrainer:
    """Create an MLXTrainer instance for testing."""
    return MLXTrainer(
        model=titanic_classifier,
        config=training_config,
    )


@pytest.fixture
def mock_mlflow_central(temp_dir: Path, monkeypatch):
    """Mock MLflow central configuration."""
    tracking_uri = f"sqlite:///{temp_dir}/test_mlflow.db"
    artifact_root = str(temp_dir / "artifacts")
    
    # Patch the class attributes
    monkeypatch.setattr(MLflowCentral, "TRACKING_URI", tracking_uri)
    monkeypatch.setattr(MLflowCentral, "ARTIFACT_ROOT", artifact_root)
    
    return MLflowCentral()


@pytest.fixture
def cleanup_mlx():
    """Ensure MLX memory is cleaned up after tests."""
    yield
    mx.eval({})  # Force evaluation of any pending operations
    mx.synchronize()  # Ensure all operations are complete


# Test data fixtures
@pytest.fixture
def dummy_batch():
    """Create a dummy batch for testing."""
    batch_size = 4
    seq_length = 128
    
    return {
        "input_ids": mx.ones((batch_size, seq_length), dtype=mx.int32),
        "attention_mask": mx.ones((batch_size, seq_length), dtype=mx.int32),
        "labels": mx.array([0, 1, 1, 0], dtype=mx.int32),
    }


@pytest.fixture
def tokenizer_name():
    """Default tokenizer name for tests."""
    return "answerdotai/ModernBERT-base"


# Environment setup
@pytest.fixture(autouse=True)
def setup_test_env(monkeypatch):
    """Set up test environment variables."""
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "sqlite:///test_mlflow.db")
    monkeypatch.setenv("LOGURU_LEVEL", "ERROR")
    monkeypatch.setenv("MLX_METAL_DEBUG", "0")