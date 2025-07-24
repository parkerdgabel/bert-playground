"""
Integration tests for the application bootstrap process.

These tests verify that the bootstrap correctly initializes all components
and that they can work together.
"""

import pytest
from pathlib import Path
import tempfile
from unittest.mock import Mock, patch

from infrastructure.bootstrap import ApplicationBootstrap, get_bootstrap, initialize_application, get_service
from infrastructure.di.container import InfrastructureContainer
from infrastructure.config.manager import ConfigurationManager
from infrastructure.plugins.registry import PluginRegistry
from infrastructure.plugins.loader import PluginLoader


class TestApplicationBootstrap:
    """Test the application bootstrap process."""
    
    @pytest.fixture
    def config_file(self):
        """Create a temporary config file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write("""
model:
  type: modernbert_with_head
  hidden_size: 128
training:
  epochs: 1
  batch_size: 16
""")
            yield Path(f.name)
        Path(f.name).unlink()
    
    def test_bootstrap_initialization(self, config_file):
        """Test that bootstrap initializes correctly."""
        bootstrap = ApplicationBootstrap(config_path=config_file)
        container = bootstrap.initialize()
        
        # Should return a container
        assert isinstance(container, InfrastructureContainer)
        
        # Should be marked as initialized
        assert bootstrap._initialized
        
        # Should not re-initialize
        container2 = bootstrap.initialize()
        assert container2 is container
    
    def test_configuration_manager_available(self):
        """Test that configuration manager is registered."""
        bootstrap = ApplicationBootstrap()
        container = bootstrap.initialize()
        
        # Configuration manager should be available
        config_manager = container.resolve(ConfigurationManager)
        assert config_manager is not None
        assert isinstance(config_manager, ConfigurationManager)
    
    def test_get_service_before_init_fails(self):
        """Test that getting service before init raises error."""
        bootstrap = ApplicationBootstrap()
        
        with pytest.raises(RuntimeError, match="Application not initialized"):
            bootstrap.get_service(ConfigurationManager)
    
    def test_get_status(self):
        """Test status reporting."""
        bootstrap = ApplicationBootstrap()
        
        # Before initialization
        status = bootstrap.get_status()
        assert status["initialized"] is False
        
        # After initialization
        container = bootstrap.initialize()
        status = bootstrap.get_status()
        assert status["initialized"] is True
        assert "container_initialized" in status
        assert "services_count" in status
    
    def test_shutdown(self):
        """Test graceful shutdown."""
        bootstrap = ApplicationBootstrap()
        container = bootstrap.initialize()
        
        # Should be initialized
        assert bootstrap._initialized
        
        # Shutdown
        bootstrap.shutdown()
        
        # Should no longer be initialized
        assert not bootstrap._initialized
    
    def test_global_functions(self, config_file):
        """Test global convenience functions."""
        # Initialize application
        container = initialize_application(config_path=config_file)
        assert isinstance(container, InfrastructureContainer)
        
        # Get service
        config_manager = get_service(ConfigurationManager)
        assert isinstance(config_manager, ConfigurationManager)
        
        # Get bootstrap
        bootstrap = get_bootstrap()
        assert isinstance(bootstrap, ApplicationBootstrap)
        assert bootstrap._initialized
    
    def test_auto_discovery(self):
        """Test that auto-discovery finds components."""
        bootstrap = ApplicationBootstrap()
        container = bootstrap.initialize()
        
        # Should have discovered some components
        health = container.health_check()
        assert health["services_count"] > 0
    
    def test_config_loading(self, config_file):
        """Test configuration loading through bootstrap."""
        bootstrap = ApplicationBootstrap(config_path=config_file)
        container = bootstrap.initialize()
        
        # Config should be accessible
        config = bootstrap.get_config("model.type")
        assert config == "modernbert_with_head"
        
        config = bootstrap.get_config("training.epochs")
        assert config == 1
        
        # Default value
        config = bootstrap.get_config("missing.key", "default")
        assert config == "default"


class TestBootstrapWithMocks:
    """Test bootstrap with mocked components."""
    
    @patch('infrastructure.bootstrap.ConfigurationManager')
    def test_config_manager_initialization(self, mock_config_cls):
        """Test that config manager is initialized with correct paths."""
        mock_config = Mock()
        mock_config_cls.return_value = mock_config
        mock_config.load_configuration.return_value = {}
        
        user_path = Path("/user/config.yaml")
        project_path = Path("/project/k-bert.yaml")
        command_path = Path("/cmd/config.yaml")
        
        bootstrap = ApplicationBootstrap(
            config_path=command_path,
            user_config_path=user_path,
            project_config_path=project_path
        )
        
        # Config manager not created until initialize
        mock_config_cls.assert_not_called()
        
        # Initialize
        container = bootstrap.initialize()
        
        # Config manager should be created with correct paths
        mock_config_cls.assert_called_once_with(
            user_config_path=user_path,
            project_config_path=project_path,
            command_config_path=command_path
        )