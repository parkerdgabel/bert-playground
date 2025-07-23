"""
Simple integration test to verify the new architecture works.
"""

import pytest
from pathlib import Path
import tempfile

from infrastructure.bootstrap import ApplicationBootstrap, initialize_application, get_service
from infrastructure.di.container import Container
from ports.secondary.monitoring import MonitoringService


def test_basic_bootstrap():
    """Test that we can bootstrap the application."""
    # This should work without any configuration
    container = initialize_application()
    
    # Should return a container
    assert isinstance(container, Container)
    
    # Should be able to get monitoring service
    monitoring = get_service(MonitoringService)
    assert monitoring is not None
    
    # Should be able to log
    monitoring.log("INFO", "Test message")


def test_bootstrap_with_config():
    """Test bootstrap with a configuration file."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml') as f:
        f.write("""
model:
  type: test
training:
  epochs: 1
""")
        f.flush()
        
        # Initialize with config
        container = initialize_application(Path(f.name))
        assert isinstance(container, Container)


def test_service_resolution():
    """Test that we can resolve basic services."""
    container = initialize_application()
    
    # Import services to check
    from ports.secondary.compute import ComputeBackend
    from ports.secondary.storage import StorageService
    from infrastructure.events.bus import EventBus
    
    # Should be able to resolve
    compute = container.resolve(ComputeBackend)
    assert compute is not None
    
    storage = container.resolve(StorageService)
    assert storage is not None
    
    event_bus = container.resolve(EventBus)
    assert event_bus is not None


if __name__ == "__main__":
    # Run basic tests
    test_basic_bootstrap()
    print("✓ Basic bootstrap works")
    
    test_bootstrap_with_config()
    print("✓ Bootstrap with config works")
    
    test_service_resolution()
    print("✓ Service resolution works")
    
    print("\n✅ All integration tests passed!")