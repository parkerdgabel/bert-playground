#!/usr/bin/env python
"""
Demonstration of the new hexagonal architecture.

This script shows how to use the new architecture with dependency injection
for various tasks.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from core.bootstrap import initialize_application, get_service
from core.ports.compute import ComputeBackend
from core.ports.storage import StorageService
from core.ports.config import ConfigurationProvider
from core.ports.monitoring import MonitoringService
from core.events.bus import get_event_bus
from core.events.events import ApplicationEvent
from models.factory_facade import ModelFactory
from training.commands.train import TrainCommand


def demonstrate_di_container():
    """Demonstrate dependency injection container usage."""
    print("\n=== Dependency Injection Demo ===")
    
    # Initialize the application
    container = initialize_application()
    print("✓ Application initialized")
    
    # Get services from container
    monitoring = get_service(MonitoringService)
    monitoring.log_info("Got monitoring service from container")
    
    compute = get_service(ComputeBackend)
    print(f"✓ Compute backend: {type(compute).__name__}")
    
    storage = get_service(StorageService)
    print(f"✓ Storage service: {type(storage).__name__}")
    
    return container


def demonstrate_configuration():
    """Demonstrate configuration management."""
    print("\n=== Configuration Management Demo ===")
    
    config_provider = get_service(ConfigurationProvider)
    
    # Create test configurations
    base_config = {
        "model": {"hidden_size": 768},
        "training": {"epochs": 10, "batch_size": 32}
    }
    
    override_config = {
        "training": {"epochs": 5}
    }
    
    # Merge configurations
    merged = config_provider.merge_configs(base_config, override_config)
    print(f"✓ Merged config: {merged}")
    
    # Validate configuration
    is_valid = config_provider.validate_config(merged)
    print(f"✓ Config valid: {is_valid}")


def demonstrate_event_system():
    """Demonstrate event-driven communication."""
    print("\n=== Event System Demo ===")
    
    event_bus = get_event_bus()
    
    # Define custom event
    class DemoEvent(ApplicationEvent):
        event_type = "demo_event"
    
    # Create event handler
    events_received = []
    def on_demo_event(event):
        events_received.append(event)
        print(f"✓ Received event: {event.event_type} with data: {event.data}")
    
    # Subscribe to event
    event_bus.subscribe(DemoEvent, on_demo_event)
    
    # Publish event
    demo_event = DemoEvent(data={"message": "Hello from hexagonal architecture!"})
    event_bus.publish(demo_event)
    
    print(f"✓ Total events processed: {len(events_received)}")


def demonstrate_model_creation():
    """Demonstrate model creation through factory."""
    print("\n=== Model Creation Demo ===")
    
    model_factory = get_service(ModelFactory)
    
    # Create model configuration
    model_config = {
        "type": "modernbert_with_head",
        "hidden_size": 128,
        "num_hidden_layers": 2,
        "num_attention_heads": 2,
        "intermediate_size": 256,
        "vocab_size": 30522,
        "head": {
            "type": "classification",
            "num_labels": 2
        }
    }
    
    # Create model
    model = model_factory.create_model(model_config)
    print(f"✓ Created model: {type(model).__name__}")
    print(f"✓ Model config: hidden_size={model_config['hidden_size']}, layers={model_config['num_hidden_layers']}")


def demonstrate_training_command():
    """Demonstrate training command pattern."""
    print("\n=== Training Command Demo ===")
    
    # Get container to resolve command
    container = get_service(Container)
    
    # Create training command
    train_command = container.resolve(TrainCommand)
    print(f"✓ Created training command: {type(train_command).__name__}")
    
    # Command has all dependencies injected
    print("✓ Dependencies injected:")
    print(f"  - model_factory: {hasattr(train_command, 'model_factory')}")
    print(f"  - dataset_factory: {hasattr(train_command, 'dataset_factory')}")
    print(f"  - training_orchestrator: {hasattr(train_command, 'training_orchestrator')}")
    print(f"  - storage: {hasattr(train_command, 'storage')}")


def demonstrate_monitoring():
    """Demonstrate monitoring capabilities."""
    print("\n=== Monitoring Demo ===")
    
    monitoring = get_service(MonitoringService)
    
    # Log at different levels
    monitoring.log_debug("Debug message")
    monitoring.log_info("Info message")
    monitoring.log_warning("Warning message")
    print("✓ Logged messages at different levels")
    
    # Record metrics
    monitoring.record_metric("demo_metric", 42.0, {"tag": "demo"})
    monitoring.record_metric("accuracy", 0.95, {"epoch": 1, "split": "validation"})
    print("✓ Recorded metrics")
    
    # Create span for timing
    with monitoring.create_span("demo_operation") as span:
        # Simulate work
        import time
        time.sleep(0.1)
        span.set_attribute("status", "success")
    print("✓ Created timing span")


def demonstrate_storage():
    """Demonstrate storage capabilities."""
    print("\n=== Storage Demo ===")
    
    storage = get_service(StorageService)
    
    # Test storage operations
    test_data = {"key": "value", "number": 42}
    test_path = Path("/tmp/demo_test.json")
    
    # Save data
    storage.save_json(test_data, test_path)
    print(f"✓ Saved data to {test_path}")
    
    # Load data
    loaded_data = storage.load_json(test_path)
    print(f"✓ Loaded data: {loaded_data}")
    
    # Check existence
    exists = storage.exists(test_path)
    print(f"✓ File exists: {exists}")
    
    # Clean up
    test_path.unlink()


def main():
    """Run all demonstrations."""
    print("K-BERT Hexagonal Architecture Demonstration")
    print("=" * 50)
    
    try:
        # Initialize and demonstrate DI
        container = demonstrate_di_container()
        
        # Run demonstrations
        demonstrate_configuration()
        demonstrate_event_system()
        demonstrate_model_creation()
        demonstrate_training_command()
        demonstrate_monitoring()
        demonstrate_storage()
        
        print("\n✅ All demonstrations completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Error during demonstration: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()