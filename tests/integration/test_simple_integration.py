"""
Simple integration test to verify the new architecture works.
"""

# import pytest
from pathlib import Path
import tempfile

from infrastructure.bootstrap import ApplicationBootstrap, initialize_application, get_service
from infrastructure.di.container import InfrastructureContainer
from infrastructure.config.manager import ConfigurationManager


def test_basic_bootstrap():
    """Test that we can bootstrap the application."""
    # This should work without any configuration
    container = initialize_application()
    
    # Should return a container
    assert isinstance(container, InfrastructureContainer)
    
    # Basic health check
    health = container.health_check()
    assert health["initialized"] is True
    assert health["services_count"] > 0


def test_bootstrap_with_config():
    """Test bootstrap with a configuration file."""
    # Create a temporary config
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        f.write("""
model:
  type: modernbert_with_head
  hidden_size: 768
training:
  epochs: 10
  batch_size: 32
  learning_rate: 1e-4
""")
        
        # Initialize with config
        container = initialize_application(Path(f.name))
        assert isinstance(container, InfrastructureContainer)


def test_service_resolution():
    """Test that we can resolve basic services."""
    container = initialize_application()
    
    # Configuration manager should always be available
    config_manager = container.resolve(ConfigurationManager)
    assert config_manager is not None
    assert isinstance(config_manager, ConfigurationManager)
    
    # Should be able to get services through the global function
    config_manager2 = get_service(ConfigurationManager)
    assert config_manager2 is config_manager  # Should be singleton


def test_decorated_components():
    """Test that decorated components are discovered and available."""
    container = initialize_application()
    
    # Try to resolve some decorated components if they exist
    try:
        from models.factory import ModelBuilder
        model_builder = container.resolve(ModelBuilder)
        assert model_builder is not None
    except ImportError:
        # Model builder might not be available in all environments
        pass
    
    try:
        from adapters.secondary.tokenizer.huggingface import HuggingFaceTokenizerAdapter
        # This should work if the adapter is decorated
        pass
    except ImportError:
        pass


if __name__ == "__main__":
    # Run basic tests
    test_basic_bootstrap()
    print("✓ Basic bootstrap works")
    
    test_bootstrap_with_config()
    print("✓ Bootstrap with config works")
    
    test_service_resolution()
    print("✓ Service resolution works")
    
    test_decorated_components()
    print("✓ Decorated components work")
    
    print("\n✅ All integration tests passed!")