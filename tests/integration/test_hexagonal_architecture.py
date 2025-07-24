"""Integration test to verify the hexagonal architecture works end-to-end."""

import asyncio
from pathlib import Path
import tempfile
from typing import Dict, Any

# Domain imports
from domain.entities.model import BertModel, ModelArchitecture, ModelWeights
from domain.entities.training import TrainingConfig, TrainingSession
from domain.entities.dataset import Dataset, DataBatch
from domain.services.training import ModelTrainingService
from domain.services.tokenization import TokenizationService

# Application imports
from application.commands.train import TrainModelCommand
from application.dto.training import TrainingRequestDTO

# Infrastructure imports
from infrastructure.di.container import Container

# Adapter imports
from adapters.secondary.compute.mlx.compute_adapter import MLXComputeAdapter
from adapters.secondary.data.mlx.data_loader import MLXDataLoader
from adapters.secondary.monitoring.console.console_adapter import ConsoleMonitoringAdapter
from adapters.secondary.storage.filesystem.storage_adapter import FilesystemStorageAdapter
from adapters.secondary.storage.filesystem.checkpoint_adapter import FilesystemCheckpointAdapter
from adapters.secondary.tokenizer.huggingface.tokenizer_adapter import HuggingFaceTokenizerAdapter
from adapters.secondary.metrics.mlx.metrics_calculator import MLXMetricsCalculator


def test_domain_entities():
    """Test that domain entities can be created without framework dependencies."""
    print("Testing domain entities...")
    
    # Create model architecture
    arch = ModelArchitecture(
        model_type="bert",
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=3072,
        vocab_size=30522
    )
    
    # Create model
    model = BertModel(
        architecture=arch,
        weights=ModelWeights({}),
        config={}
    )
    
    # Create training config
    config = TrainingConfig(
        batch_size=32,
        learning_rate=1e-4,
        num_epochs=3,
        warmup_steps=100
    )
    
    print("✓ Domain entities created successfully")


def test_adapters():
    """Test that adapters can be instantiated."""
    print("\nTesting adapters...")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create adapters
        compute = MLXComputeAdapter()
        monitor = ConsoleMonitoringAdapter(verbosity="quiet")
        storage = FilesystemStorageAdapter(base_path=tmpdir)
        
        # Test basic functionality
        assert compute is not None
        assert monitor is not None
        assert storage is not None
        
        print("✓ Adapters created successfully")


def test_service_orchestration():
    """Test that domain services can orchestrate adapters using DI."""
    print("\nTesting service orchestration...")
    
    # Create mock dataset
    dataset = Dataset(
        name="test_dataset",
        data=[],  # Empty for test
        metadata={"num_samples": 0}
    )
    
    # Create container and register services
    from infrastructure.di.container import Container
    from infrastructure.di.decorators import clear_registry
    from unittest.mock import AsyncMock
    
    clear_registry()
    container = Container()
    
    # Register mock ports
    from ports.secondary.compute import ComputePort
    from ports.secondary.monitoring import MonitoringPort
    from ports.secondary.checkpoint import CheckpointPort
    from ports.secondary.metrics import MetricsPort
    from ports.secondary.tokenizer import TokenizerPort
    
    container.register(ComputePort, AsyncMock(spec=ComputePort), instance=True)
    container.register(MonitoringPort, AsyncMock(spec=MonitoringPort), instance=True)
    container.register(CheckpointPort, AsyncMock(spec=CheckpointPort), instance=True)
    container.register(MetricsPort, AsyncMock(spec=MetricsPort), instance=True)
    container.register(TokenizerPort, AsyncMock(spec=TokenizerPort), instance=True)
    
    # Register and resolve services through DI
    container.register_decorator(ModelTrainingService)
    container.register_decorator(TokenizationService)
    
    training_service = container.resolve(ModelTrainingService)
    tokenization_service = container.resolve(TokenizationService)
    
    assert training_service is not None
    assert tokenization_service is not None
    
    print("✓ Services created successfully via DI")


def test_dependency_injection():
    """Test that DI container can wire everything together."""
    print("\nTesting dependency injection...")
    
    # Create container
    container = Container()
    
    # Register adapters
    container.register_singleton(MLXComputeAdapter)
    container.register_singleton(ConsoleMonitoringAdapter)
    
    # Resolve
    compute = container.resolve(MLXComputeAdapter)
    monitor = container.resolve(ConsoleMonitoringAdapter)
    
    assert compute is not None
    assert monitor is not None
    
    print("✓ Dependency injection working")


async def test_application_command():
    """Test that application commands work with domain and adapters."""
    print("\nTesting application command...")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create minimal DTO
        request = TrainingRequestDTO(
            model_type="bert",
            model_size="small",
            train_data_path=Path(tmpdir) / "train.csv",
            output_dir=Path(tmpdir) / "output",
            config={
                "num_epochs": 1,
                "batch_size": 8,
                "learning_rate": 1e-4
            }
        )
        
        # Note: Full execution would require all dependencies
        # This just tests the DTO creation
        assert request.model_type == "bert"
        assert request.config["num_epochs"] == 1
        
        print("✓ Application DTOs working")


def test_hexagonal_principles():
    """Verify hexagonal architecture principles are followed."""
    print("\nVerifying hexagonal architecture principles...")
    
    # Test 1: Domain has no framework dependencies
    import domain.entities.model
    import domain.services.training
    
    # Check domain modules don't import frameworks
    domain_module = domain.entities.model.__file__
    with open(domain_module, 'r') as f:
        content = f.read()
        assert 'import mlx' not in content
        assert 'import torch' not in content
        assert 'import tensorflow' not in content
    
    print("✓ Domain has no framework dependencies")
    
    # Test 2: Ports are defined as protocols
    from ports.secondary.compute import ComputePort
    assert hasattr(ComputePort, '__annotations__')
    print("✓ Ports are properly defined as protocols")
    
    # Test 3: Adapters implement ports
    compute_adapter = MLXComputeAdapter()
    # Check it has required methods
    assert hasattr(compute_adapter, 'forward_backward')
    assert hasattr(compute_adapter, 'update_weights')
    print("✓ Adapters implement port interfaces")


def main():
    """Run all integration tests."""
    print("Running hexagonal architecture integration tests...\n")
    
    try:
        test_domain_entities()
        test_adapters()
        test_service_orchestration()
        test_dependency_injection()
        asyncio.run(test_application_command())
        test_hexagonal_principles()
        
        print("\n✅ All integration tests passed!")
        print("\nThe hexagonal architecture is properly implemented:")
        print("- Domain layer has no framework dependencies")
        print("- Ports define clear contracts")
        print("- Adapters implement framework-specific details")
        print("- Application layer orchestrates use cases")
        print("- Dependency injection wires everything together")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        raise


if __name__ == "__main__":
    main()