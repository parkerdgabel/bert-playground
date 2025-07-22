"""
Complete integration test for Phase 2 components.

Tests the interaction between event system, plugin system, hexagonal architecture,
training components, and data enhancements working together.
"""

import tempfile
from pathlib import Path
from typing import Any, Dict
from unittest.mock import MagicMock, patch

import mlx.core as mx
import pytest

from core.events.bus import EventBus, GlobalEventBus
from core.events.types import Event, EventContext, EventType, EventPriority
from core.plugins.loader import PluginLoader
from core.plugins.registry import PluginRegistry
from core.ports.compute import ComputeBackend
from core.adapters.mlx_adapter import MLXComputeAdapter
from training.components.training_orchestrator import TrainingOrchestrator
from training.components.training_loop import TrainingLoop
from training.components.evaluation_loop import EvaluationLoop
from training.components.checkpoint_manager import CheckpointManager
from training.components.metrics_tracker import MetricsTracker
from training.strategies.standard import StandardTrainingStrategy
from data.templates.registry import TemplateRegistry
from data.validation.schema import DataValidator
from training.core.base import BaseTrainer
from training.core.config import BaseTrainerConfig


class TestEventSystem:
    """Event system integration tests."""
    
    def test_event_bus_creation(self):
        """Test basic event bus functionality."""
        bus = EventBus("test")
        events_received = []
        
        def handler(event: Event):
            events_received.append(event)
        
        # Subscribe to training events
        reg_id = bus.subscribe(
            handler, 
            EventType.TRAINING,
            priority=EventPriority.HIGH
        )
        
        # Emit event
        bus.emit(
            EventType.TRAINING,
            "training_started",
            source="test",
            data={"epochs": 5}
        )
        
        assert len(events_received) == 1
        assert events_received[0].type == EventType.TRAINING
        assert events_received[0].name == "training_started"
        assert events_received[0].data["epochs"] == 5
        
        # Unsubscribe
        assert bus.unsubscribe(reg_id)
        bus.emit(EventType.TRAINING, "another_event", source="test")
        assert len(events_received) == 1  # No new events


class TestHexagonalArchitecture:
    """Hexagonal architecture integration tests."""
    
    def test_compute_adapter_integration(self):
        """Test compute adapter with MLX backend."""
        adapter = MLXComputeAdapter()
        
        # Test basic operations
        x = adapter.array([1, 2, 3])
        assert adapter.shape(x) == (3,)
        
        y = adapter.zeros((2, 2))
        assert adapter.shape(y) == (2, 2)
        
        # Test evaluation
        adapter.eval(x, y)
        assert True  # Should not raise


class TestTrainingComponents:
    """Training component integration tests."""
    
    @pytest.fixture
    def mock_model(self):
        """Mock model for testing."""
        model = MagicMock()
        model.parameters.return_value = {}
        
        def forward(**kwargs):
            batch_size = kwargs.get("input_ids", mx.zeros((1, 1))).shape[0]
            return {
                "loss": mx.array(0.5),
                "logits": mx.zeros((batch_size, 2))
            }
        
        model.__call__ = MagicMock(side_effect=forward)
        return model
    
    @pytest.fixture
    def mock_dataloader(self):
        """Mock dataloader."""
        batches = []
        for i in range(2):
            batch = {
                "input_ids": mx.zeros((2, 10)),
                "labels": mx.zeros((2,), dtype=mx.int32)
            }
            batches.append(batch)
        
        loader = MagicMock()
        loader.__iter__.return_value = iter(batches)
        loader.__len__.return_value = len(batches)
        return loader
    
    @pytest.fixture  
    def config(self, tmp_path):
        """Training configuration."""
        config = BaseTrainerConfig()
        config.environment.output_dir = tmp_path
        config.training.num_epochs = 1
        config.training.logging_steps = 1
        config.training.save_strategy = "no"
        config.optimizer.learning_rate = 0.001
        return config
    
    def test_training_components_integration(self, mock_model, mock_dataloader, config):
        """Test that training components work together."""
        # Create components
        training_loop = TrainingLoop(config.training)
        eval_loop = EvaluationLoop(config.training)
        checkpoint_mgr = CheckpointManager(config.checkpoint, config.environment.output_dir)
        metrics_tracker = MetricsTracker()
        
        orchestrator = TrainingOrchestrator(
            training_loop=training_loop,
            eval_loop=eval_loop, 
            checkpoint_manager=checkpoint_mgr,
            metrics_tracker=metrics_tracker
        )
        
        # Test orchestrator can run
        assert orchestrator.training_loop == training_loop
        assert orchestrator.eval_loop == eval_loop
        
    def test_strategy_integration(self, mock_model, config):
        """Test training strategy integration."""
        strategy = StandardTrainingStrategy()
        
        # Strategy should create pipeline
        pipeline = strategy.create_pipeline({
            "model": mock_model,
            "config": config,
            "optimizer": MagicMock()
        })
        
        assert pipeline is not None


class TestDataEnhancements:
    """Data enhancement integration tests."""
    
    def test_template_registry(self):
        """Test template system integration."""
        registry = TemplateRegistry()
        
        # Should have built-in templates
        templates = registry.list_templates()
        assert len(templates) > 0
        assert "basic" in templates
        
        # Get template
        template = registry.get_template("basic")
        assert template is not None
    
    def test_data_validation(self):
        """Test data validation framework."""
        validator = DataValidator()
        
        # Test with valid data
        valid_data = {
            "input_ids": [1, 2, 3],
            "labels": [0]
        }
        
        result = validator.validate(valid_data)
        assert result.is_valid


class TestPluginSystem:
    """Plugin system integration tests."""
    
    def test_plugin_loader_basic(self):
        """Test basic plugin loading."""
        loader = PluginLoader()
        registry = PluginRegistry()
        
        # Should be able to load plugins (even if none exist)
        plugins = loader.discover_plugins()
        assert isinstance(plugins, list)
        
        # Registry should work
        assert registry.get_plugins_by_category("model") == []


class TestEndToEndIntegration:
    """End-to-end integration tests."""
    
    @pytest.fixture
    def integration_setup(self, tmp_path):
        """Setup for integration tests."""
        # Create config
        config = BaseTrainerConfig()
        config.environment.output_dir = tmp_path
        config.training.num_epochs = 1
        config.training.logging_steps = 1
        config.training.save_strategy = "no"
        
        # Create mock components
        model = MagicMock()
        model.parameters.return_value = {}
        
        def forward(**kwargs):
            return {
                "loss": mx.array(0.5),
                "logits": mx.zeros((2, 2))
            }
        model.__call__ = MagicMock(side_effect=forward)
        
        dataloader = MagicMock()
        batch = {
            "input_ids": mx.zeros((2, 10)),
            "labels": mx.zeros((2,), dtype=mx.int32)
        }
        dataloader.__iter__.return_value = iter([batch])
        dataloader.__len__.return_value = 1
        
        return {
            "config": config,
            "model": model,
            "dataloader": dataloader
        }
    
    def test_event_driven_training(self, integration_setup):
        """Test training with event system integration."""
        setup = integration_setup
        
        # Create event bus
        bus = EventBus("integration_test")
        events_captured = []
        
        def event_handler(event: Event):
            events_captured.append(event.name)
        
        # Subscribe to all events
        bus.subscribe(event_handler, EventType.TRAINING)
        
        # Emit some training events
        bus.emit(EventType.TRAINING, "training_started", "trainer", {"epochs": 1})
        bus.emit(EventType.TRAINING, "epoch_started", "trainer", {"epoch": 0})
        bus.emit(EventType.TRAINING, "training_completed", "trainer", {"success": True})
        
        # Verify events were captured
        assert len(events_captured) >= 3
        assert "training_started" in events_captured
        assert "epoch_started" in events_captured 
        assert "training_completed" in events_captured
    
    def test_component_interaction(self, integration_setup):
        """Test interaction between different Phase 2 components."""
        setup = integration_setup
        
        # Create event bus for communication
        bus = EventBus("component_test")
        component_interactions = []
        
        def interaction_handler(event: Event):
            component_interactions.append({
                "component": event.context.source,
                "event": event.name,
                "data": event.data
            })
        
        bus.subscribe(interaction_handler, EventType.TRAINING)
        
        # Simulate component interactions
        bus.emit(EventType.TRAINING, "model_loaded", "model_factory", {"model_type": "bert"})
        bus.emit(EventType.TRAINING, "data_prepared", "data_loader", {"samples": 100})
        bus.emit(EventType.TRAINING, "training_step", "trainer", {"step": 1, "loss": 0.5})
        
        # Verify interactions were captured
        assert len(component_interactions) == 3
        
        sources = [i["component"] for i in component_interactions]
        assert "model_factory" in sources
        assert "data_loader" in sources
        assert "trainer" in sources
    
    def test_error_handling_integration(self, integration_setup):
        """Test error handling across components."""
        setup = integration_setup
        
        # Create event bus with error handler
        errors_handled = []
        
        def error_handler(event: Event, error: Exception):
            errors_handled.append({
                "event": event.name,
                "error": str(error)
            })
        
        bus = EventBus("error_test", error_handler=error_handler)
        
        def failing_handler(event: Event):
            raise ValueError("Test error")
        
        # Subscribe failing handler
        bus.subscribe(failing_handler, EventType.TRAINING)
        
        # Emit event - should handle error gracefully
        try:
            bus.emit(EventType.TRAINING, "test_event", "test", {})
        except ValueError:
            # Error should be caught by error handler
            pass
        
        # Verify error was handled
        assert len(errors_handled) == 1
        assert "Test error" in errors_handled[0]["error"]
    
    def test_performance_monitoring_integration(self, integration_setup):
        """Test performance monitoring across components."""
        setup = integration_setup
        
        # Create event bus for performance events
        bus = EventBus("perf_test")
        performance_events = []
        
        def perf_handler(event: Event):
            if "time" in event.data or "memory" in event.data:
                performance_events.append(event)
        
        bus.subscribe(perf_handler, EventType.TRAINING)
        
        # Simulate performance events
        bus.emit(EventType.TRAINING, "batch_processed", "trainer", {
            "time": 0.05,
            "memory": 1024,
            "batch_size": 32
        })
        
        bus.emit(EventType.TRAINING, "checkpoint_saved", "checkpoint_manager", {
            "time": 0.2,
            "size_mb": 50
        })
        
        # Verify performance events were captured
        assert len(performance_events) == 2
        
        # Check event data
        batch_event = performance_events[0]
        assert batch_event.data["time"] == 0.05
        assert batch_event.data["batch_size"] == 32
        
        checkpoint_event = performance_events[1] 
        assert checkpoint_event.data["time"] == 0.2
        assert checkpoint_event.data["size_mb"] == 50


if __name__ == "__main__":
    pytest.main([__file__, "-v"])