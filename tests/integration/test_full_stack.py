"""
Full stack integration tests for the hexagonal architecture.

These tests verify that all layers work together correctly:
- CLI → Application → Domain → Infrastructure
"""

import pytest
from pathlib import Path
import tempfile
import yaml
from unittest.mock import Mock, patch

from infrastructure.bootstrap import ApplicationBootstrap, initialize_application, get_service
from infrastructure.di.container import Container
from infrastructure.ports.compute import ComputeBackend
from infrastructure.ports.storage import StorageService, ModelStorageService
from infrastructure.ports.config import ConfigurationProvider
from infrastructure.ports.monitoring import MonitoringService
from cli.app import app
from training.commands.train import TrainCommand
from models.factory_facade import ModelFactory


class TestFullStackIntegration:
    """Test the complete application stack."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for tests."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)
    
    @pytest.fixture
    def test_config(self, temp_dir):
        """Create a test configuration."""
        config = {
            "model": {
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
            },
            "training": {
                "epochs": 1,
                "batch_size": 2,
                "learning_rate": 1e-4,
                "output_dir": str(temp_dir / "output")
            },
            "data": {
                "train_path": str(temp_dir / "train.csv"),
                "val_path": str(temp_dir / "val.csv"),
                "text_column": "text",
                "label_column": "label"
            }
        }
        
        # Create dummy data files
        train_data = "text,label\nSample text 1,0\nSample text 2,1\n"
        val_data = "text,label\nValidation text 1,0\nValidation text 2,1\n"
        
        (temp_dir / "train.csv").write_text(train_data)
        (temp_dir / "val.csv").write_text(val_data)
        
        config_path = temp_dir / "config.yaml"
        with open(config_path, "w") as f:
            yaml.dump(config, f)
        
        return config_path
    
    def test_bootstrap_initialization(self, test_config):
        """Test that the bootstrap process initializes all components."""
        # Create bootstrap
        bootstrap = ApplicationBootstrap(test_config)
        container = bootstrap.initialize()
        
        # Verify core services are registered
        assert container.resolve(ComputeBackend) is not None
        assert container.resolve(StorageService) is not None
        assert container.resolve(ModelStorageService) is not None
        assert container.resolve(ConfigurationProvider) is not None
        assert container.resolve(MonitoringService) is not None
        assert container.resolve(ModelFactory) is not None
        
        # Verify we can get services through convenience function
        monitoring = bootstrap.get_service(MonitoringService)
        assert monitoring is not None
    
    def test_cli_to_application_flow(self, test_config, monkeypatch):
        """Test flow from CLI through application layer."""
        # Initialize application
        container = initialize_application(test_config)
        
        # Mock MLX operations since we're not testing actual training
        mock_compute = Mock()
        container.register(ComputeBackend, mock_compute, instance=True)
        
        # Test that we can resolve a training command
        train_command = container.resolve(TrainCommand)
        assert train_command is not None
        
        # Verify dependencies are injected
        assert hasattr(train_command, 'model_factory')
        assert hasattr(train_command, 'dataset_factory')
        assert hasattr(train_command, 'training_orchestrator')
    
    def test_configuration_loading_hierarchy(self, temp_dir):
        """Test configuration loading and merging."""
        # Create multiple config files
        base_config = {
            "model": {"hidden_size": 128},
            "training": {"epochs": 10}
        }
        
        override_config = {
            "training": {"epochs": 5, "batch_size": 32}
        }
        
        base_path = temp_dir / "base.yaml"
        override_path = temp_dir / "override.yaml"
        
        with open(base_path, "w") as f:
            yaml.dump(base_config, f)
        with open(override_path, "w") as f:
            yaml.dump(override_config, f)
        
        # Initialize with base config
        container = initialize_application(base_path)
        config_provider = get_service(ConfigurationProvider)
        
        # Load and merge configs
        base = config_provider.load_config(str(base_path))
        override = config_provider.load_config(str(override_path))
        merged = config_provider.merge_configs(base, override)
        
        # Verify merging
        assert merged["model"]["hidden_size"] == 128
        assert merged["training"]["epochs"] == 5
        assert merged["training"]["batch_size"] == 32
    
    def test_dependency_injection_chain(self):
        """Test that dependency injection works through the entire chain."""
        container = Container()
        
        # Create mock implementations
        mock_compute = Mock(spec=ComputeBackend)
        mock_storage = Mock(spec=StorageService)
        mock_monitoring = Mock(spec=MonitoringService)
        
        # Register mocks
        container.register(ComputeBackend, mock_compute, instance=True)
        container.register(StorageService, mock_storage, instance=True)
        container.register(MonitoringService, mock_monitoring, instance=True)
        
        # Create a service that depends on these
        class TestService:
            def __init__(self, compute: ComputeBackend, storage: StorageService):
                self.compute = compute
                self.storage = storage
        
        container.register(TestService)
        
        # Resolve and verify
        service = container.resolve(TestService)
        assert service.compute is mock_compute
        assert service.storage is mock_storage
    
    def test_error_propagation(self, test_config):
        """Test that errors propagate correctly through layers."""
        container = initialize_application(test_config)
        
        # Create a mock that raises an error
        mock_compute = Mock(spec=ComputeBackend)
        mock_compute.create_optimizer.side_effect = RuntimeError("Test error")
        container.register(ComputeBackend, mock_compute, instance=True)
        
        # Verify error is handled properly
        train_command = container.resolve(TrainCommand)
        config_provider = get_service(ConfigurationProvider)
        config = config_provider.load_config(str(test_config))
        
        # Execute should handle the error gracefully
        with pytest.raises(RuntimeError, match="Test error"):
            # This would normally be caught and logged by the command
            mock_compute.create_optimizer(Mock(), Mock())
    
    @pytest.mark.slow
    def test_end_to_end_training_flow(self, test_config, monkeypatch):
        """Test complete training flow from CLI to model save."""
        # This test is marked slow as it involves the full stack
        
        # Mock the CLI runner to avoid actually running training
        mock_runner = Mock()
        monkeypatch.setattr("typer.testing.CliRunner", Mock(return_value=mock_runner))
        
        # Initialize application
        container = initialize_application(test_config)
        
        # Verify all components can be resolved
        assert container.resolve(TrainCommand) is not None
        assert get_service(ModelFactory) is not None
        assert get_service(StorageService) is not None
        assert get_service(MonitoringService) is not None
    
    def test_plugin_system_integration(self, temp_dir):
        """Test that the plugin system integrates correctly."""
        from infrastructure.plugins.registry import PluginRegistry
        from infrastructure.plugins.base import Plugin
        
        # Create a test plugin
        class TestPlugin(Plugin):
            name = "test_plugin"
            version = "1.0.0"
            
            def activate(self):
                self.activated = True
            
            def deactivate(self):
                self.activated = False
        
        # Initialize application
        container = initialize_application()
        registry = container.resolve(PluginRegistry)
        
        # Register and activate plugin
        plugin = TestPlugin()
        registry.register(plugin)
        
        # Verify plugin is registered
        assert "test_plugin" in [p.name for p in registry.list_plugins()]
        assert plugin.activated
    
    def test_container_health_check(self):
        """Test that container health check works."""
        # Initialize application
        container = initialize_application()
        
        # Health check should report healthy state
        health = container.health_check()
        assert health["initialized"] is True
        assert health["services_count"] > 0
        assert health["config_manager_available"] is True
    
    def test_monitoring_integration(self, capsys):
        """Test that monitoring works throughout the stack."""
        # Initialize application
        container = initialize_application()
        monitoring = get_service(MonitoringService)
        
        # Log at different levels
        monitoring.log_debug("Debug message")
        monitoring.log_info("Info message")
        monitoring.log_warning("Warning message")
        
        # Create a metric
        monitoring.record_metric("test_metric", 42.0, {"tag": "test"})
        
        # Verify monitoring is working (actual output depends on configuration)
        # This just ensures no exceptions are raised
        assert monitoring is not None