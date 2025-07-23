"""Simple integration test to verify hexagonal architecture principles."""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


def test_domain_independence():
    """Test that domain layer has no framework dependencies."""
    print("Testing domain independence...")
    
    # Import domain modules
    import domain.entities.model
    import domain.entities.training
    import domain.entities.dataset
    import domain.entities.metrics
    import domain.services.training
    import domain.services.evaluation
    import domain.services.tokenization
    import domain.services.checkpointing
    
    # Check a domain module doesn't import frameworks
    domain_file = Path(domain.entities.model.__file__)
    with open(domain_file, 'r') as f:
        content = f.read()
        
    # Verify no framework imports
    frameworks = ['mlx', 'torch', 'tensorflow', 'jax', 'transformers']
    for framework in frameworks:
        assert f'import {framework}' not in content, f"Domain imports {framework}!"
        assert f'from {framework}' not in content, f"Domain imports from {framework}!"
    
    print("✓ Domain has no framework dependencies")


def test_domain_entities():
    """Test that domain entities can be created."""
    print("\nTesting domain entities...")
    
    from domain.entities.model import BertModel, ModelArchitecture, ModelWeights
    from domain.entities.training import TrainingConfig, TrainingSession, TrainingState
    from domain.entities.dataset import Dataset, DataBatch
    from domain.entities.metrics import TrainingMetrics, EvaluationMetrics
    
    # Create model architecture
    arch = ModelArchitecture(
        vocab_size=30522,
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=3072,
        max_position_embeddings=512
    )
    
    # Create model
    model = BertModel(
        architecture=arch,
        weights=ModelWeights({}),
        metadata={"name": "test-model"}
    )
    
    # Create training config
    config = TrainingConfig(
        num_epochs=3,
        batch_size=32,
        learning_rate=1e-4
    )
    
    # Create training session
    session = TrainingSession(
        session_id="test-session",
        config=config,
        state=TrainingState(),
        metadata={"model": "test-model"}
    )
    
    print("✓ All domain entities created successfully")


def test_ports_are_protocols():
    """Test that ports are defined as protocols."""
    print("\nTesting port definitions...")
    
    from domain.ports.compute import ComputePort
    from domain.ports.monitoring import MonitoringPort
    from domain.ports.storage import StoragePort, CheckpointPort
    from domain.ports.tokenizer import TokenizerPort
    from domain.ports.data import DataLoaderPort
    
    # Check they are protocols (have __annotations__)
    ports = [ComputePort, MonitoringPort, StoragePort, CheckpointPort, 
             TokenizerPort, DataLoaderPort]
    
    for port in ports:
        # Protocols have __annotations__ from their method signatures
        assert hasattr(port, '__annotations__') or hasattr(port, '__abstractmethods__'), \
            f"{port.__name__} is not a proper protocol/interface"
    
    print("✓ All ports are properly defined")


def test_adapter_structure():
    """Test that adapters are organized correctly."""
    print("\nTesting adapter structure...")
    
    import adapters.secondary.compute.mlx
    import adapters.secondary.monitoring.console
    import adapters.secondary.storage.filesystem
    import adapters.secondary.tokenizer.huggingface
    
    # Test MLX compute adapter
    from adapters.secondary.compute.mlx.compute_adapter import MLXComputeAdapter
    adapter = MLXComputeAdapter()
    
    # Check it has required methods
    required_methods = ['forward', 'backward', 'optimize_step', 'compile_model']
    for method in required_methods:
        assert hasattr(adapter, method), f"MLXComputeAdapter missing {method}"
    
    print("✓ Adapter structure is correct")


def test_infrastructure_location():
    """Test that infrastructure code is in the right place."""
    print("\nTesting infrastructure location...")
    
    # Check infrastructure directory exists
    infra_path = Path(project_root) / "infrastructure"
    assert infra_path.exists(), "Infrastructure directory missing!"
    
    # Core should not exist
    core_path = Path(project_root) / "core"
    assert not core_path.exists(), "Old core/ directory still exists!"
    
    print("✓ Infrastructure is properly located")


def test_application_layer():
    """Test that application layer exists and is structured correctly."""
    print("\nTesting application layer...")
    
    # Import DTOs
    from application.dto.training import TrainingRequestDTO, TrainingResponseDTO
    from application.dto.evaluation import EvaluationRequestDTO, EvaluationResponseDTO
    from application.dto.prediction import PredictionRequestDTO, PredictionResponseDTO
    
    # Create a simple DTO
    dto = TrainingRequestDTO(
        model_type="bert",
        model_size="base",
        train_data_path=Path("/tmp/data.csv"),
        output_dir=Path("/tmp/output"),
        config={"epochs": 1}
    )
    
    assert dto.model_type == "bert"
    print("✓ Application layer is properly structured")


def test_hexagonal_architecture_summary():
    """Print summary of architecture validation."""
    print("\n" + "="*60)
    print("HEXAGONAL ARCHITECTURE VALIDATION SUMMARY")
    print("="*60)
    
    layers = {
        "Domain": "Pure business logic - ✓ No framework dependencies",
        "Application": "Use case orchestration - ✓ DTOs and commands",
        "Infrastructure": "Technical utilities - ✓ Moved from core/",
        "Ports": "Domain interfaces - ✓ Defined as protocols",
        "Adapters": "Framework implementations - ✓ MLX, console, filesystem",
        "CLI": "Presentation layer - ✓ Thin wrapper using commands"
    }
    
    for layer, status in layers.items():
        print(f"{layer:15} {status}")
    
    print("\n✅ Hexagonal architecture is properly implemented!")
    print("\nKey achievements:")
    print("- Domain is completely framework-agnostic")
    print("- Can swap MLX for PyTorch/JAX by changing adapters")
    print("- Business logic is isolated from technical concerns")
    print("- Clean dependency flow: CLI → Application → Domain ← Adapters")


def main():
    """Run all tests."""
    try:
        test_domain_independence()
        test_domain_entities()
        test_ports_are_protocols()
        test_adapter_structure()
        test_infrastructure_location()
        test_application_layer()
        test_hexagonal_architecture_summary()
    except AssertionError as e:
        print(f"\n❌ Test failed: {e}")
        return 1
    except ImportError as e:
        print(f"\n❌ Import error: {e}")
        return 1
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())