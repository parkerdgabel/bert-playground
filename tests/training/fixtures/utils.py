"""Utility functions for testing."""

import mlx.core as mx
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import json
import time
from contextlib import contextmanager


def assert_array_equal(actual: mx.array, expected: mx.array, rtol: float = 1e-5):
    """Assert two MLX arrays are equal within tolerance."""
    assert actual.shape == expected.shape, f"Shape mismatch: {actual.shape} vs {expected.shape}"
    assert mx.allclose(actual, expected, rtol=rtol), "Arrays are not close"


def assert_metrics_equal(
    actual: Dict[str, float],
    expected: Dict[str, float],
    rtol: float = 1e-5,
):
    """Assert two metrics dictionaries are equal."""
    assert set(actual.keys()) == set(expected.keys()), f"Keys mismatch: {actual.keys()} vs {expected.keys()}"
    
    for key in expected:
        assert abs(actual[key] - expected[key]) < rtol, f"Metric {key}: {actual[key]} != {expected[key]}"


def create_dummy_checkpoint(
    checkpoint_dir: Path,
    epoch: int = 1,
    step: int = 100,
    metrics: Optional[Dict[str, float]] = None,
) -> Path:
    """Create a dummy checkpoint for testing."""
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # Create dummy model file
    model_path = checkpoint_dir / "model.safetensors"
    model_path.touch()
    
    # Create optimizer state
    optimizer_path = checkpoint_dir / "optimizer.safetensors"
    optimizer_path.touch()
    
    # Create training state
    if metrics is None:
        metrics = {"loss": 0.5, "accuracy": 0.8}
    
    state = {
        "epoch": epoch,
        "global_step": step,
        "best_metric": max(metrics.values()),
        "metrics_history": {k: [v] for k, v in metrics.items()},
    }
    
    with open(checkpoint_dir / "training_state.json", "w") as f:
        json.dump(state, f)
    
    # Create config
    config = {
        "optimizer": {"type": "adam", "learning_rate": 1e-3},
        "training": {"num_epochs": 10},
    }
    
    with open(checkpoint_dir / "config.json", "w") as f:
        json.dump(config, f)
    
    return checkpoint_dir


def verify_checkpoint_structure(checkpoint_dir: Path):
    """Verify checkpoint has correct structure."""
    assert checkpoint_dir.exists(), f"Checkpoint directory {checkpoint_dir} does not exist"
    
    # Check required files
    required_files = [
        "model.safetensors",
        "training_state.json",
        "config.json",
    ]
    
    for file_name in required_files:
        file_path = checkpoint_dir / file_name
        assert file_path.exists(), f"Missing required file: {file_name}"
    
    # Verify JSON files are valid
    with open(checkpoint_dir / "training_state.json") as f:
        state = json.load(f)
        assert "epoch" in state
        assert "global_step" in state
    
    with open(checkpoint_dir / "config.json") as f:
        config = json.load(f)
        assert "optimizer" in config or "training" in config


@contextmanager
def timer(name: str = "Operation"):
    """Context manager for timing operations."""
    start = time.time()
    yield
    elapsed = time.time() - start
    print(f"{name} took {elapsed:.3f} seconds")


def generate_predictions(
    num_samples: int,
    num_classes: int = 2,
    correct_ratio: float = 0.8,
    seed: int = 42,
) -> Tuple[mx.array, mx.array]:
    """Generate synthetic predictions and labels for testing metrics."""
    np.random.seed(seed)
    
    labels = mx.array(np.random.randint(0, num_classes, size=num_samples))
    
    # Generate predictions with specified accuracy
    predictions = mx.zeros((num_samples, num_classes))
    
    for i in range(num_samples):
        if np.random.random() < correct_ratio:
            # Correct prediction
            predictions[i, labels[i]] = 1.0
        else:
            # Incorrect prediction
            wrong_class = (labels[i] + 1) % num_classes
            predictions[i, wrong_class] = 1.0
    
    # Add some noise
    noise = mx.random.normal(predictions.shape) * 0.1
    predictions = predictions + noise
    
    return predictions, labels


def create_mock_mlflow_run(
    run_id: str = "test-run-123",
    experiment_id: str = "0",
    metrics: Optional[Dict[str, float]] = None,
    params: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Create mock MLflow run data."""
    if metrics is None:
        metrics = {"loss": 0.5, "accuracy": 0.8}
    
    if params is None:
        params = {"learning_rate": "0.001", "batch_size": "32"}
    
    return {
        "info": {
            "run_id": run_id,
            "experiment_id": experiment_id,
            "status": "FINISHED",
            "start_time": time.time() * 1000,
            "end_time": (time.time() + 100) * 1000,
        },
        "data": {
            "metrics": metrics,
            "params": params,
            "tags": {},
        },
    }


def compare_model_outputs(
    model1_output: mx.array,
    model2_output: mx.array,
    rtol: float = 1e-5,
) -> bool:
    """Compare outputs from two models."""
    if model1_output.shape != model2_output.shape:
        return False
    
    return mx.allclose(model1_output, model2_output, rtol=rtol)


def create_sample_batch(
    batch_size: int = 4,
    input_dim: int = 10,
    task_type: str = "classification",
    num_classes: int = 2,
) -> Dict[str, mx.array]:
    """Create a sample batch for testing."""
    batch = {
        "input": mx.random.normal((batch_size, input_dim)),
    }
    
    if task_type == "classification":
        batch["labels"] = mx.random.randint(0, num_classes, (batch_size,))
    else:  # regression
        batch["targets"] = mx.random.normal((batch_size,))
    
    return batch


def simulate_training_metrics(
    num_steps: int,
    initial_loss: float = 1.0,
    final_loss: float = 0.1,
    noise_level: float = 0.05,
) -> List[Dict[str, float]]:
    """Simulate training metrics over time."""
    metrics_history = []
    
    for step in range(num_steps):
        # Exponential decay with noise
        progress = step / max(num_steps - 1, 1)
        loss = initial_loss * np.exp(-3 * progress) + final_loss
        loss += np.random.normal(0, noise_level)
        loss = max(0.01, loss)  # Ensure positive
        
        # Accuracy increases with training
        accuracy = 0.5 + 0.5 * (1 - np.exp(-3 * progress))
        accuracy += np.random.normal(0, noise_level / 2)
        accuracy = np.clip(accuracy, 0.0, 1.0)
        
        metrics_history.append({
            "loss": float(loss),
            "accuracy": float(accuracy),
            "learning_rate": 1e-3 * (1 - progress * 0.5),  # Linear decay
        })
    
    return metrics_history


class MockCallback:
    """Mock callback for testing callback system."""
    
    def __init__(self, name: str = "mock"):
        self.name = name
        self.events = []
    
    def on_train_begin(self, trainer, state):
        self.events.append("train_begin")
    
    def on_train_end(self, trainer, state):
        self.events.append("train_end")
    
    def on_epoch_begin(self, trainer, state):
        self.events.append(f"epoch_begin_{state.epoch}")
    
    def on_epoch_end(self, trainer, state):
        self.events.append(f"epoch_end_{state.epoch}")
    
    def on_step_begin(self, trainer, state):
        self.events.append(f"step_begin_{state.global_step}")
    
    def on_step_end(self, trainer, state):
        self.events.append(f"step_end_{state.global_step}")
    
    def reset(self):
        self.events = []


def verify_training_result(result: Any):
    """Verify training result has expected structure."""
    assert hasattr(result, "metrics"), "Result missing metrics"
    assert hasattr(result, "best_checkpoint"), "Result missing best_checkpoint"
    assert hasattr(result, "final_checkpoint"), "Result missing final_checkpoint"
    assert hasattr(result, "history"), "Result missing history"
    
    # Verify metrics
    assert isinstance(result.metrics, dict), "Metrics should be a dict"
    assert "loss" in result.metrics, "Metrics missing loss"
    
    # Verify paths
    if result.best_checkpoint:
        assert isinstance(result.best_checkpoint, Path), "best_checkpoint should be Path"
    
    if result.final_checkpoint:
        assert isinstance(result.final_checkpoint, Path), "final_checkpoint should be Path"