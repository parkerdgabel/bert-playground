"""Integration example showing how to use DI in k-bert.

This demonstrates how the DI system can be used to wire up the various
components of k-bert in a clean, testable way.
"""

from typing import Protocol, Optional, Dict, Any
from pathlib import Path
from core.di import injectable, singleton, provider, get_container


# Define protocols for core k-bert components

class ModelFactoryProtocol(Protocol):
    """Protocol for model creation."""
    
    def create_model(self, config: Dict[str, Any]) -> Any:
        """Create a model from configuration."""
        ...


class DataLoaderProtocol(Protocol):
    """Protocol for data loading."""
    
    def load_data(self, path: Path) -> Any:
        """Load data from path."""
        ...


class TrainerProtocol(Protocol):
    """Protocol for model training."""
    
    def train(self, model: Any, data: Any) -> None:
        """Train the model."""
        ...


class ConfigManagerProtocol(Protocol):
    """Protocol for configuration management."""
    
    def get_config(self, key: str) -> Any:
        """Get configuration value."""
        ...
        
    def merge_configs(self, *configs: Dict[str, Any]) -> Dict[str, Any]:
        """Merge multiple configurations."""
        ...


# Implementations

@singleton(bind_to=ConfigManagerProtocol)
class ConfigManager:
    """Manages configuration for k-bert."""
    
    def __init__(self):
        self._config = {
            "model": {
                "type": "modernbert",
                "hidden_size": 768,
                "num_layers": 12,
            },
            "training": {
                "batch_size": 32,
                "learning_rate": 1e-4,
                "epochs": 10,
            },
            "data": {
                "max_length": 512,
                "shuffle": True,
            }
        }
        print("ConfigManager initialized (singleton)")
        
    def get_config(self, key: str) -> Any:
        """Get configuration value by dot-separated key."""
        parts = key.split(".")
        config = self._config
        for part in parts:
            config = config.get(part, {})
        return config
        
    def merge_configs(self, *configs: Dict[str, Any]) -> Dict[str, Any]:
        """Merge multiple configurations."""
        result = self._config.copy()
        for config in configs:
            result.update(config)
        return result


@injectable(bind_to=ModelFactoryProtocol)
class MLXModelFactory:
    """Factory for creating MLX models."""
    
    def __init__(self, config_manager: ConfigManagerProtocol):
        self.config_manager = config_manager
        print("MLXModelFactory initialized with ConfigManager")
        
    def create_model(self, config: Optional[Dict[str, Any]] = None) -> Any:
        """Create a model from configuration."""
        model_config = self.config_manager.get_config("model")
        if config:
            model_config = self.config_manager.merge_configs(model_config, config)
            
        print(f"Creating model: {model_config['type']} with config: {model_config}")
        # In real implementation, this would create actual MLX model
        return f"MockModel({model_config['type']})"


@injectable(bind_to=DataLoaderProtocol)
class MLXDataLoader:
    """MLX-optimized data loader."""
    
    def __init__(self, config_manager: ConfigManagerProtocol):
        self.config_manager = config_manager
        self.data_config = config_manager.get_config("data")
        print(f"MLXDataLoader initialized with config: {self.data_config}")
        
    def load_data(self, path: Path) -> Any:
        """Load and preprocess data."""
        print(f"Loading data from {path}")
        print(f"Max length: {self.data_config.get('max_length')}")
        print(f"Shuffle: {self.data_config.get('shuffle')}")
        # In real implementation, this would load actual data
        return {"train": "mock_train_data", "val": "mock_val_data"}


@injectable(bind_to=TrainerProtocol)
class MLXTrainer:
    """Trainer for MLX models."""
    
    def __init__(
        self,
        config_manager: ConfigManagerProtocol,
        model_factory: ModelFactoryProtocol,
    ):
        self.config_manager = config_manager
        self.model_factory = model_factory
        self.training_config = config_manager.get_config("training")
        print(f"MLXTrainer initialized with config: {self.training_config}")
        
    def train(self, model: Any, data: Any) -> None:
        """Train the model."""
        print(f"Training model: {model}")
        print(f"Batch size: {self.training_config.get('batch_size')}")
        print(f"Learning rate: {self.training_config.get('learning_rate')}")
        print(f"Epochs: {self.training_config.get('epochs')}")
        print(f"Training data: {data}")


# Providers for complex objects

class MLFlowTracker:
    """Mock MLFlow tracker."""
    
    def __init__(self):
        self.experiment_id = "exp_001"
        
    def log_metric(self, key: str, value: float):
        print(f"[MLFlow] Logged metric: {key}={value}")


@provider
def create_mlflow_tracker() -> MLFlowTracker:
    """Factory for creating MLFlow tracker."""
    print("Creating MLFlow tracker via provider")
    return MLFlowTracker()


# Main application class that uses DI

@injectable
class KBertApplication:
    """Main application class demonstrating DI usage."""
    
    def __init__(
        self,
        config_manager: ConfigManagerProtocol,
        model_factory: ModelFactoryProtocol,
        data_loader: DataLoaderProtocol,
        trainer: TrainerProtocol,
        mlflow_tracker: MLFlowTracker,
    ):
        self.config_manager = config_manager
        self.model_factory = model_factory
        self.data_loader = data_loader
        self.trainer = trainer
        self.mlflow_tracker = mlflow_tracker
        print("\nKBertApplication initialized with all dependencies")
        
    def run(self, data_path: Path):
        """Run the complete training pipeline."""
        print("\n=== Running k-bert training pipeline ===\n")
        
        # Load data
        data = self.data_loader.load_data(data_path)
        
        # Create model
        model = self.model_factory.create_model()
        
        # Train model
        self.trainer.train(model, data)
        
        # Log metrics
        self.mlflow_tracker.log_metric("final_loss", 0.123)
        self.mlflow_tracker.log_metric("accuracy", 0.95)
        
        print("\n=== Training complete ===")


def demo_di_integration():
    """Demonstrate the DI system in action."""
    print("=== k-bert DI Integration Demo ===\n")
    
    # Get container (all decorators have already registered services)
    container = get_container()
    
    # Resolve the main application
    app = container.resolve(KBertApplication)
    
    # Run the application
    app.run(Path("data/train.csv"))
    
    print("\n=== Demonstrating singleton behavior ===\n")
    
    # Get config manager again - should be same instance
    config1 = container.resolve(ConfigManagerProtocol)
    config2 = container.resolve(ConfigManagerProtocol)
    print(f"Config managers are same instance: {config1 is config2}")
    
    # Get data loader again - should be new instance (transient)
    loader1 = container.resolve(DataLoaderProtocol)
    loader2 = container.resolve(DataLoaderProtocol)
    print(f"Data loaders are same instance: {loader1 is loader2}")


def demo_testing_with_mocks():
    """Demonstrate how DI makes testing easier."""
    print("\n\n=== Testing with Mock Implementations ===\n")
    
    from core.di import Container
    
    # Create a test container
    test_container = Container()
    
    # Register mock implementations
    class MockConfigManager:
        def get_config(self, key: str) -> Any:
            return {"test": "config"}
            
        def merge_configs(self, *configs) -> Dict[str, Any]:
            return {"merged": "config"}
    
    class MockModelFactory:
        def __init__(self, config_manager: ConfigManagerProtocol):
            pass
            
        def create_model(self, config=None):
            return "MockTestModel"
    
    # Register mocks
    test_container.register(ConfigManagerProtocol, MockConfigManager, singleton=True)
    test_container.register(ModelFactoryProtocol, MockModelFactory)
    
    # Create model factory with mocked dependencies
    factory = test_container.resolve(ModelFactoryProtocol)
    model = factory.create_model()
    print(f"Created test model: {model}")
    
    print("\n=== Testing complete - easy to swap implementations! ===")


if __name__ == "__main__":
    demo_di_integration()
    demo_testing_with_mocks()
    
    # Clean up
    from core.di import reset_container
    reset_container()