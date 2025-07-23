#!/usr/bin/env python
"""
Simple demonstration of hexagonal architecture concepts.

This demo shows the key architectural patterns without complex dependencies.
"""

from typing import Protocol, Dict, Any, List
from dataclasses import dataclass
from abc import ABC, abstractmethod


# 1. PORTS (Interfaces/Contracts)

class ConfigurationPort(Protocol):
    """Port for configuration management."""
    def load_config(self, path: str) -> Dict[str, Any]:
        """Load configuration from a source."""
        ...
    
    def validate_config(self, config: Dict[str, Any]) -> bool:
        """Validate configuration."""
        ...


class StoragePort(Protocol):
    """Port for storage operations."""
    def save(self, path: str, data: Any) -> None:
        """Save data to storage."""
        ...
    
    def load(self, path: str) -> Any:
        """Load data from storage."""
        ...


class ComputePort(Protocol):
    """Port for compute operations."""
    def train_model(self, data: Any, config: Dict[str, Any]) -> Any:
        """Train a model."""
        ...
    
    def predict(self, model: Any, data: Any) -> Any:
        """Make predictions."""
        ...


# 2. DOMAIN (Business Logic)

@dataclass
class TrainingConfig:
    """Domain model for training configuration."""
    epochs: int
    batch_size: int
    learning_rate: float


class TrainingService:
    """Domain service for training operations."""
    
    def __init__(self, compute: ComputePort, storage: StoragePort):
        self.compute = compute
        self.storage = storage
    
    def train(self, config: TrainingConfig, data_path: str) -> str:
        """Train a model and save it."""
        print(f"Training with config: epochs={config.epochs}, batch_size={config.batch_size}")
        
        # Load data
        data = self.storage.load(data_path)
        
        # Train model
        model = self.compute.train_model(
            data, 
            {"epochs": config.epochs, "batch_size": config.batch_size}
        )
        
        # Save model
        model_path = "model.bin"
        self.storage.save(model_path, model)
        
        return model_path


# 3. ADAPTERS (Implementations)

class YAMLConfigAdapter:
    """Adapter for YAML configuration."""
    
    def load_config(self, path: str) -> Dict[str, Any]:
        print(f"Loading config from {path}")
        # Simplified for demo
        return {
            "training": {
                "epochs": 10,
                "batch_size": 32,
                "learning_rate": 0.001
            }
        }
    
    def validate_config(self, config: Dict[str, Any]) -> bool:
        print("Validating config")
        return "training" in config


class FileStorageAdapter:
    """Adapter for file system storage."""
    
    def save(self, path: str, data: Any) -> None:
        print(f"Saving data to {path}")
        # Simplified for demo
    
    def load(self, path: str) -> Any:
        print(f"Loading data from {path}")
        # Simplified for demo
        return {"features": [[1, 2, 3]], "labels": [0]}


class MLXComputeAdapter:
    """Adapter for MLX compute backend."""
    
    def train_model(self, data: Any, config: Dict[str, Any]) -> Any:
        print(f"Training model with MLX: {config}")
        # Simplified for demo
        return {"weights": [0.1, 0.2, 0.3]}
    
    def predict(self, model: Any, data: Any) -> Any:
        print("Making predictions with MLX")
        return [0.95]


# 4. DEPENDENCY INJECTION

class Container:
    """Simple DI container."""
    
    def __init__(self):
        self._services = {}
        self._instances = {}
    
    def register(self, interface: type, implementation: Any, singleton: bool = False):
        """Register a service."""
        self._services[interface] = (implementation, singleton)
    
    def resolve(self, interface: type) -> Any:
        """Resolve a service."""
        if interface not in self._services:
            raise ValueError(f"No registration for {interface}")
        
        implementation, singleton = self._services[interface]
        
        if singleton:
            if interface not in self._instances:
                self._instances[interface] = implementation()
            return self._instances[interface]
        
        return implementation()


# 5. APPLICATION BOOTSTRAP

def bootstrap_application() -> Container:
    """Bootstrap the application with all dependencies."""
    container = Container()
    
    # Register adapters
    container.register(ConfigurationPort, YAMLConfigAdapter, singleton=True)
    container.register(StoragePort, FileStorageAdapter, singleton=True)
    container.register(ComputePort, MLXComputeAdapter, singleton=True)
    
    print("✓ Application bootstrapped")
    return container


# 6. COMMAND PATTERN

class TrainCommand:
    """Command for training operations."""
    
    def __init__(self, container: Container):
        self.container = container
    
    def execute(self, config_path: str, data_path: str) -> None:
        """Execute training command."""
        print("\n=== Executing Train Command ===")
        
        # Get services from container
        config_port = self.container.resolve(ConfigurationPort)
        compute_port = self.container.resolve(ComputePort)
        storage_port = self.container.resolve(StoragePort)
        
        # Load and validate config
        config_dict = config_port.load_config(config_path)
        if not config_port.validate_config(config_dict):
            raise ValueError("Invalid configuration")
        
        # Create domain model
        training_config = TrainingConfig(
            epochs=config_dict["training"]["epochs"],
            batch_size=config_dict["training"]["batch_size"],
            learning_rate=config_dict["training"]["learning_rate"]
        )
        
        # Create domain service
        training_service = TrainingService(compute_port, storage_port)
        
        # Execute training
        model_path = training_service.train(training_config, data_path)
        print(f"✓ Model saved to: {model_path}")


# 7. DEMONSTRATION

def main():
    """Demonstrate the hexagonal architecture."""
    print("K-BERT Hexagonal Architecture Demo")
    print("=" * 50)
    
    # Bootstrap application
    container = bootstrap_application()
    
    # Create and execute command
    train_command = TrainCommand(container)
    train_command.execute("config.yaml", "data.csv")
    
    print("\n✅ Demo completed successfully!")
    
    # Show benefits
    print("\n=== Benefits Demonstrated ===")
    print("1. Clear separation between domain logic and infrastructure")
    print("2. Easy to test (can mock any port)")
    print("3. Easy to swap implementations (e.g., S3 instead of file storage)")
    print("4. Domain logic has no dependencies on external libraries")
    print("5. Dependency injection provides flexibility")


if __name__ == "__main__":
    main()