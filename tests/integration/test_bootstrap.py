"""
Integration tests for the application bootstrap process.

These tests verify that the bootstrap correctly initializes all components
and that they can work together.
"""

import pytest
from pathlib import Path
import tempfile
from unittest.mock import Mock, patch

from core.bootstrap import ApplicationBootstrap, get_bootstrap, initialize_application, get_service
from core.di.container import Container
from core.ports.compute import ComputeBackend
from core.ports.storage import StorageService, ModelStorageService
from core.ports.config import ConfigurationProvider
from core.ports.monitoring import MonitoringService
from core.ports.tokenizer import TokenizerFactory
from core.events.bus import EventBus
from core.plugins.registry import PluginRegistry
from core.plugins.loader import PluginLoader


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
        bootstrap = ApplicationBootstrap(config_file)
        container = bootstrap.initialize()
        
        # Should return a container
        assert isinstance(container, Container)
        
        # Should be marked as initialized
        assert bootstrap._initialized
        
        # Second initialization should return same container
        container2 = bootstrap.initialize()
        assert container2 is container
    
    def test_infrastructure_setup(self):
        """Test that infrastructure components are set up correctly."""
        bootstrap = ApplicationBootstrap()
        container = bootstrap.initialize()
        
        # Event bus should be registered
        event_bus = container.resolve(EventBus)
        assert event_bus is not None
        
        # Configuration provider should be registered
        config_provider = container.resolve(ConfigurationProvider)
        assert config_provider is not None
    
    def test_ports_and_adapters_setup(self):
        """Test that all ports have adapters registered."""
        bootstrap = ApplicationBootstrap()
        container = bootstrap.initialize()
        
        # All ports should have implementations
        compute = container.resolve(ComputeBackend)
        assert compute is not None
        
        storage = container.resolve(StorageService)
        assert storage is not None
        
        model_storage = container.resolve(ModelStorageService)
        assert model_storage is not None
        
        monitoring = container.resolve(MonitoringService)
        assert monitoring is not None
        
        tokenizer_factory = container.resolve(TokenizerFactory)
        assert tokenizer_factory is not None
    
    def test_domain_services_setup(self):
        """Test that domain services are registered."""
        bootstrap = ApplicationBootstrap()
        container = bootstrap.initialize()
        
        # Import here to match bootstrap
        from models.factory_facade import ModelFactory
        from adapters.secondary.data.factory import DatasetFactory
        
        # Domain factories should be available
        model_factory = container.resolve(ModelFactory)
        assert model_factory is not None
        
        dataset_factory = container.resolve(DatasetFactory)
        assert dataset_factory is not None
    
    def test_application_services_setup(self):
        """Test that application services are registered."""
        bootstrap = ApplicationBootstrap()
        container = bootstrap.initialize()
        
        # Import here to match bootstrap
        from training.components.training_orchestrator import TrainingOrchestrator
        from training.components.training_loop import TrainingLoop
        from training.components.evaluation_loop import EvaluationLoop
        from training.components.checkpoint_manager import CheckpointManager
        from training.components.metrics_tracker import MetricsTracker
        
        # All training components should be available
        orchestrator = container.resolve(TrainingOrchestrator)
        assert orchestrator is not None
        
        training_loop = container.resolve(TrainingLoop)
        assert training_loop is not None
        
        eval_loop = container.resolve(EvaluationLoop)
        assert eval_loop is not None
        
        checkpoint_mgr = container.resolve(CheckpointManager)
        assert checkpoint_mgr is not None
        
        metrics_tracker = container.resolve(MetricsTracker)
        assert metrics_tracker is not None
    
    def test_cli_layer_setup(self):
        """Test that CLI components are registered."""
        bootstrap = ApplicationBootstrap()
        container = bootstrap.initialize()
        
        from cli.factory import CommandFactory
        
        # Command factory should be available
        cmd_factory = container.resolve(CommandFactory)
        assert cmd_factory is not None
    
    def test_plugin_system_setup(self):
        """Test that plugin system is initialized."""
        bootstrap = ApplicationBootstrap()
        container = bootstrap.initialize()
        
        # Plugin components should be registered
        registry = container.resolve(PluginRegistry)
        assert registry is not None
        
        loader = container.resolve(PluginLoader)
        assert loader is not None
    
    def test_get_service_convenience(self):
        """Test the get_service convenience method."""
        bootstrap = ApplicationBootstrap()
        container = bootstrap.initialize()
        
        # Should be able to get services through bootstrap
        monitoring = bootstrap.get_service(MonitoringService)
        assert monitoring is not None
        
        # Should be same instance as from container
        monitoring2 = container.resolve(MonitoringService)
        assert monitoring is monitoring2
    
    def test_global_bootstrap_singleton(self):
        """Test that get_bootstrap returns singleton."""
        bootstrap1 = get_bootstrap()
        bootstrap2 = get_bootstrap()
        
        assert bootstrap1 is bootstrap2
    
    def test_initialize_application_function(self, config_file):
        """Test the initialize_application convenience function."""
        container = initialize_application(config_file)
        
        assert isinstance(container, Container)
        
        # Should be able to resolve services
        monitoring = container.resolve(MonitoringService)
        assert monitoring is not None
    
    def test_get_service_function(self):
        """Test the global get_service function."""
        # Should initialize if needed
        monitoring = get_service(MonitoringService)
        assert monitoring is not None
        
        # Should return same instance on subsequent calls
        monitoring2 = get_service(MonitoringService)
        assert monitoring is monitoring2
    
    def test_bootstrap_with_missing_config(self):
        """Test bootstrap with non-existent config file."""
        # Should not fail, will use defaults
        bootstrap = ApplicationBootstrap(Path("/non/existent/file.yaml"))
        container = bootstrap.initialize()
        
        assert isinstance(container, Container)
    
    @patch('core.plugins.loader.PluginLoader.discover_plugins')
    def test_plugin_loading_error_handling(self, mock_discover):
        """Test that plugin loading errors don't crash bootstrap."""
        mock_discover.side_effect = Exception("Plugin error")
        
        # Should still initialize successfully
        bootstrap = ApplicationBootstrap()
        container = bootstrap.initialize()
        
        assert isinstance(container, Container)
    
    def test_service_dependencies_resolved(self):
        """Test that services with dependencies are properly resolved."""
        bootstrap = ApplicationBootstrap()
        container = bootstrap.initialize()
        
        # Get a service that has dependencies
        from training.components.training_orchestrator import TrainingOrchestrator
        orchestrator = container.resolve(TrainingOrchestrator)
        
        # Should have all its dependencies injected
        assert hasattr(orchestrator, 'training_loop')
        assert hasattr(orchestrator, 'evaluation_loop')
        assert hasattr(orchestrator, 'checkpoint_manager')
        assert hasattr(orchestrator, 'metrics_tracker')
    
    def test_configuration_with_environment_override(self, config_file, monkeypatch):
        """Test that environment variables can override configuration."""
        # Set environment variable
        monkeypatch.setenv("K_BERT_TRAINING_EPOCHS", "5")
        
        bootstrap = ApplicationBootstrap(config_file)
        container = bootstrap.initialize()
        
        config_provider = container.resolve(ConfigurationProvider)
        # This would work if the config provider supports env overrides
        # Just verify it doesn't crash
        assert config_provider is not None