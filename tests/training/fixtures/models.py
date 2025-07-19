"""Model fixtures for testing."""

import mlx.core as mx
import mlx.nn as nn
from typing import Dict, Optional, Tuple


class SimpleBinaryClassifier(nn.Module):
    """Simple binary classifier for testing."""
    
    def __init__(self, input_dim: int = 10, hidden_dim: int = 32):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 2)
        self.dropout = nn.Dropout(0.1)
    
    def __call__(self, batch: Dict[str, mx.array]) -> Dict[str, mx.array]:
        """Forward pass returns dict with loss and logits."""
        x = batch["input"]
        y = batch["labels"]
        
        # Forward through network
        x = self.fc1(x)
        x = nn.relu(x)
        x = self.dropout(x)
        logits = self.fc2(x)
        
        # Compute loss
        loss = nn.losses.cross_entropy(logits, y, reduction="mean")
        
        return {
            "loss": loss,
            "logits": logits,
        }
    
    def loss(self, batch: Dict[str, mx.array]) -> mx.array:
        """Compute loss for batch."""
        outputs = self(batch)
        return outputs["loss"]


class SimpleMulticlassClassifier(nn.Module):
    """Simple multiclass classifier for testing."""
    
    def __init__(self, input_dim: int = 10, hidden_dim: int = 32, num_classes: int = 5):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_classes)
        self.num_classes = num_classes
    
    def __call__(self, batch: Dict[str, mx.array]) -> Dict[str, mx.array]:
        """Forward pass returns dict with loss and logits."""
        x = batch["input"]
        y = batch["labels"]
        
        # Forward through network
        x = self.fc1(x)
        x = nn.relu(x)
        logits = self.fc2(x)
        
        # Compute loss
        loss = nn.losses.cross_entropy(logits, y, reduction="mean")
        
        return {
            "loss": loss,
            "logits": logits,
        }
    
    def loss(self, batch: Dict[str, mx.array]) -> mx.array:
        """Compute loss for batch."""
        outputs = self(batch)
        return outputs["loss"]


class SimpleRegressor(nn.Module):
    """Simple regression model for testing."""
    
    def __init__(self, input_dim: int = 10, hidden_dim: int = 32):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)
    
    def __call__(self, batch: Dict[str, mx.array]) -> Dict[str, mx.array]:
        """Forward pass returns dict with loss and predictions."""
        x = batch["input"]
        y = batch.get("targets", batch.get("labels"))  # Support both keys
        
        # Forward through network
        x = self.fc1(x)
        x = nn.relu(x)
        pred = self.fc2(x).squeeze(-1)
        
        # Compute MSE loss
        loss = mx.mean((pred - y) ** 2)
        
        return {
            "loss": loss,
            "predictions": pred,
        }
    
    def loss(self, batch: Dict[str, mx.array]) -> mx.array:
        """Compute MSE loss for batch."""
        outputs = self(batch)
        return outputs["loss"]


class BrokenModel(nn.Module):
    """Model that raises errors for testing error handling."""
    
    def __init__(self, error_on: str = "forward"):
        super().__init__()
        self.error_on = error_on
        self.linear = nn.Linear(10, 2)
    
    def __call__(self, batch: Dict[str, mx.array]) -> Dict[str, mx.array]:
        if self.error_on == "forward":
            raise RuntimeError("Model forward pass failed")
        
        x = batch["input"]
        y = batch["labels"]
        logits = self.linear(x)
        
        if self.error_on == "loss":
            raise RuntimeError("Loss computation failed")
            
        loss = nn.losses.cross_entropy(logits, y, reduction="mean")
        
        return {
            "loss": loss,
            "logits": logits,
        }
    
    def loss(self, batch: Dict[str, mx.array]) -> mx.array:
        outputs = self(batch)
        return outputs["loss"]


class NaNModel(nn.Module):
    """Model that produces NaN values for testing."""
    
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 2)
    
    def __call__(self, batch: Dict[str, mx.array]) -> Dict[str, mx.array]:
        x = batch["input"]
        y = batch["labels"]
        
        # Force NaN by dividing by zero
        logits = self.linear(x) / 0.0
        loss = nn.losses.cross_entropy(logits, y, reduction="mean")
        
        return {
            "loss": loss,
            "logits": logits,
        }
    
    def loss(self, batch: Dict[str, mx.array]) -> mx.array:
        outputs = self(batch)
        return outputs["loss"]


class MemoryIntensiveModel(nn.Module):
    """Model with large memory footprint for testing."""
    
    def __init__(self, hidden_dim: int = 10000):
        super().__init__()
        self.fc1 = nn.Linear(10, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 2)
    
    def __call__(self, batch: Dict[str, mx.array]) -> Dict[str, mx.array]:
        x = batch["input"]
        y = batch["labels"]
        
        # Forward through network
        x = self.fc1(x)
        x = nn.relu(x)
        x = self.fc2(x)
        x = nn.relu(x)
        logits = self.fc3(x)
        
        # Compute loss
        loss = nn.losses.cross_entropy(logits, y, reduction="mean")
        
        return {
            "loss": loss,
            "logits": logits,
        }
    
    def loss(self, batch: Dict[str, mx.array]) -> mx.array:
        outputs = self(batch)
        return outputs["loss"]


# Factory functions
def create_test_model(model_type: str = "binary", **kwargs) -> nn.Module:
    """Create test model by type."""
    models = {
        "binary": SimpleBinaryClassifier,
        "multiclass": SimpleMulticlassClassifier,
        "regression": SimpleRegressor,
        "broken": BrokenModel,
        "nan": NaNModel,
        "memory_intensive": MemoryIntensiveModel,
    }
    
    if model_type not in models:
        raise ValueError(f"Unknown model type: {model_type}")
    
    return models[model_type](**kwargs)


def assert_model_params_equal(model1: nn.Module, model2: nn.Module, rtol: float = 1e-5):
    """Assert two models have equal parameters."""
    params1 = model1.parameters()
    params2 = model2.parameters()
    
    # Get all parameter paths
    paths1 = set(params1.keys())
    paths2 = set(params2.keys())
    
    assert paths1 == paths2, f"Model parameter paths differ: {paths1} vs {paths2}"
    
    # Compare each parameter
    for path in paths1:
        p1 = params1[path]
        p2 = params2[path]
        assert mx.allclose(p1, p2, rtol=rtol), f"Parameter {path} differs"