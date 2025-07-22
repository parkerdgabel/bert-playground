"""Integration tests for the event system with training components.

This tests the integration of a new event-driven architecture with the existing
training infrastructure to ensure smooth operation and backward compatibility.
"""

import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch

import mlx.core as mx
import pytest

from training.core.base import BaseTrainer
from training.core.config import BaseTrainerConfig
from training.core.protocols import TrainingHook, TrainingState


# Mock event system components that would be implemented in Phase 2
class EventBus:
    """Simple event bus implementation for testing."""
    
    def __init__(self):
        self.listeners = {}
        self.event_history = []
    
    def emit(self, event_name: str, data: Dict[str, Any]):
        """Emit an event to all registered listeners."""
        self.event_history.append((event_name, data))
        if event_name in self.listeners:
            for listener in self.listeners[event_name]:
                listener(data)
    
    def on(self, event_name: str, callback):
        """Register a listener for an event."""
        if event_name not in self.listeners:
            self.listeners[event_name] = []
        self.listeners[event_name].append(callback)
    
    def off(self, event_name: str, callback):
        """Remove a listener."""
        if event_name in self.listeners:
            self.listeners[event_name].remove(callback)


class EventDrivenTrainingHook(TrainingHook):
    """Adapter to connect training hooks to the event system."""
    
    def __init__(self, event_bus: EventBus):
        self.event_bus = event_bus
        self.events_captured = []
    
    def on_train_begin(self, trainer: BaseTrainer, state: TrainingState):
        """Called when training begins."""
        self.event_bus.emit("training.started", {
            "trainer": trainer,
            "state": state,
            "total_epochs": trainer.config.training.num_epochs
        })
    
    def on_epoch_begin(self, trainer: BaseTrainer, state: TrainingState):
        """Called at the beginning of each epoch."""
        self.event_bus.emit("epoch.started", {
            "epoch": state.epoch,
            "global_step": state.global_step
        })
    
    def on_batch_begin(self, trainer: BaseTrainer, state: TrainingState, batch: dict):
        """Called before processing each batch."""
        self.event_bus.emit("batch.started", {
            "batch_size": len(batch.get("input_ids", [])),
            "global_step": state.global_step
        })
    
    def on_batch_end(self, trainer: BaseTrainer, state: TrainingState, loss: mx.array):
        """Called after processing each batch."""
        loss_value = float(loss.item()) if hasattr(loss, "item") else float(loss)
        self.event_bus.emit("batch.completed", {
            "loss": loss_value,
            "global_step": state.global_step,
            "learning_rate": trainer.optimizer.learning_rate if trainer.optimizer else 0.0
        })
    
    def on_epoch_end(self, trainer: BaseTrainer, state: TrainingState):
        """Called at the end of each epoch."""
        self.event_bus.emit("epoch.completed", {
            "epoch": state.epoch,
            "train_loss": state.train_loss,
            "val_loss": state.val_loss
        })
    
    def on_evaluate_begin(self, trainer: BaseTrainer, state: TrainingState):
        """Called before evaluation."""
        self.event_bus.emit("evaluation.started", {"epoch": state.epoch})
    
    def on_evaluate_end(self, trainer: BaseTrainer, state: TrainingState, metrics: dict):
        """Called after evaluation."""
        self.event_bus.emit("evaluation.completed", {
            "epoch": state.epoch,
            "metrics": metrics
        })
    
    def on_train_end(self, trainer: BaseTrainer, state: TrainingState, result):
        """Called when training completes."""
        self.event_bus.emit("training.completed", {
            "total_epochs": state.epoch + 1,
            "total_steps": state.global_step,
            "final_loss": state.train_loss,
            "early_stopped": result.early_stopped if hasattr(result, "early_stopped") else False
        })


class EventTrainingMonitor:
    """Example of a component that listens to training events."""
    
    def __init__(self, event_bus: EventBus):
        self.event_bus = event_bus
        self.metrics_history = []
        self.checkpoints = []
        
        # Register event listeners
        event_bus.on("batch.completed", self.on_batch_completed)
        event_bus.on("epoch.completed", self.on_epoch_completed)
        event_bus.on("training.completed", self.on_training_completed)
    
    def on_batch_completed(self, data: Dict[str, Any]):
        """Track batch-level metrics."""
        self.metrics_history.append({
            "step": data["global_step"],
            "loss": data["loss"],
            "lr": data.get("learning_rate", 0.0)
        })
    
    def on_epoch_completed(self, data: Dict[str, Any]):
        """Track epoch-level metrics and trigger checkpointing."""
        self.checkpoints.append({
            "epoch": data["epoch"],
            "train_loss": data["train_loss"],
            "val_loss": data["val_loss"]
        })
    
    def on_training_completed(self, data: Dict[str, Any]):
        """Finalize training metrics."""
        print(f"Training completed: {data['total_epochs']} epochs, {data['total_steps']} steps")


@pytest.fixture
def event_bus():
    """Create a fresh event bus for testing."""
    return EventBus()


@pytest.fixture
def mock_model():
    """Create a mock model for testing."""
    model = MagicMock()
    model.parameters.return_value = {}
    model.eval.return_value = None
    model.train.return_value = None
    
    # Mock forward pass
    def forward(**kwargs):
        batch_size = kwargs.get("input_ids", mx.zeros((1, 1))).shape[0]
        return {
            "loss": mx.array(0.5),
            "logits": mx.zeros((batch_size, 2))
        }
    
    model.__call__ = MagicMock(side_effect=forward)
    return model


@pytest.fixture
def mock_dataloader():
    """Create a mock dataloader."""
    # Create 3 batches
    batches = []
    for i in range(3):
        batch = {
            "input_ids": mx.zeros((4, 10)),  # batch_size=4, seq_len=10
            "labels": mx.zeros((4,), dtype=mx.int32)
        }
        batches.append(batch)
    
    loader = MagicMock()
    loader.__iter__.return_value = iter(batches)
    loader.__len__.return_value = len(batches)
    return loader


@pytest.fixture
def trainer_config(tmp_path):
    """Create a basic trainer configuration."""
    config = BaseTrainerConfig()
    config.environment.output_dir = tmp_path
    config.training.num_epochs = 2
    config.training.logging_steps = 1
    config.training.save_strategy = "no"
    config.optimizer.learning_rate = 0.001
    return config


class TestEventSystemIntegration:
    """Test integration of event system with training components."""
    
    def test_event_driven_training_hook(self, event_bus, mock_model, mock_dataloader, trainer_config):
        """Test that training hooks properly emit events."""
        # Create event-driven hook
        event_hook = EventDrivenTrainingHook(event_bus)
        
        # Create trainer with the hook
        trainer = BaseTrainer(
            model=mock_model,
            config=trainer_config,
            callbacks=[event_hook]
        )
        
        # Train for a short run
        trainer.train(mock_dataloader)
        
        # Verify events were emitted
        event_names = [event[0] for event in event_bus.event_history]
        
        assert "training.started" in event_names
        assert "epoch.started" in event_names
        assert "batch.started" in event_names
        assert "batch.completed" in event_names
        assert "epoch.completed" in event_names
        assert "training.completed" in event_names
        
        # Verify event order
        assert event_names[0] == "training.started"
        assert event_names[-1] == "training.completed"
    
    def test_event_monitoring_integration(self, event_bus, mock_model, mock_dataloader, trainer_config):
        """Test that external components can monitor training via events."""
        # Create monitor
        monitor = EventTrainingMonitor(event_bus)
        
        # Create event hook
        event_hook = EventDrivenTrainingHook(event_bus)
        
        # Create trainer
        trainer = BaseTrainer(
            model=mock_model,
            config=trainer_config,
            callbacks=[event_hook]
        )
        
        # Train
        trainer.train(mock_dataloader)
        
        # Verify monitor captured metrics
        assert len(monitor.metrics_history) > 0
        assert len(monitor.checkpoints) == trainer_config.training.num_epochs
        
        # Verify metrics structure
        first_metric = monitor.metrics_history[0]
        assert "step" in first_metric
        assert "loss" in first_metric
        assert "lr" in first_metric
    
    def test_multiple_event_listeners(self, event_bus, mock_model, mock_dataloader, trainer_config):
        """Test that multiple listeners can subscribe to the same events."""
        batch_counters = {"listener1": 0, "listener2": 0}
        
        def listener1(data):
            batch_counters["listener1"] += 1
        
        def listener2(data):
            batch_counters["listener2"] += 1
        
        event_bus.on("batch.completed", listener1)
        event_bus.on("batch.completed", listener2)
        
        # Create and run training
        event_hook = EventDrivenTrainingHook(event_bus)
        trainer = BaseTrainer(
            model=mock_model,
            config=trainer_config,
            callbacks=[event_hook]
        )
        
        trainer.train(mock_dataloader)
        
        # Both listeners should have been called equally
        assert batch_counters["listener1"] > 0
        assert batch_counters["listener1"] == batch_counters["listener2"]
    
    def test_event_data_integrity(self, event_bus, mock_model, mock_dataloader, trainer_config):
        """Test that event data is passed correctly."""
        captured_events = {}
        
        def capture_event(event_name):
            def handler(data):
                captured_events[event_name] = data
            return handler
        
        # Register handlers for key events
        event_bus.on("training.started", capture_event("training.started"))
        event_bus.on("epoch.completed", capture_event("epoch.completed"))
        event_bus.on("training.completed", capture_event("training.completed"))
        
        # Run training
        event_hook = EventDrivenTrainingHook(event_bus)
        trainer = BaseTrainer(
            model=mock_model,
            config=trainer_config,
            callbacks=[event_hook]
        )
        
        result = trainer.train(mock_dataloader)
        
        # Verify training.started event
        assert "training.started" in captured_events
        start_data = captured_events["training.started"]
        assert start_data["total_epochs"] == trainer_config.training.num_epochs
        
        # Verify epoch.completed event
        assert "epoch.completed" in captured_events
        epoch_data = captured_events["epoch.completed"]
        assert "epoch" in epoch_data
        assert "train_loss" in epoch_data
        
        # Verify training.completed event
        assert "training.completed" in captured_events
        complete_data = captured_events["training.completed"]
        assert complete_data["total_epochs"] == trainer_config.training.num_epochs
        assert "early_stopped" in complete_data
    
    def test_event_error_handling(self, event_bus, mock_model, mock_dataloader, trainer_config):
        """Test that errors in event handlers don't crash training."""
        def failing_handler(data):
            raise RuntimeError("Handler error")
        
        event_bus.on("batch.completed", failing_handler)
        
        # Wrap emit to catch and log errors
        original_emit = event_bus.emit
        errors_caught = []
        
        def safe_emit(event_name, data):
            try:
                original_emit(event_name, data)
            except Exception as e:
                errors_caught.append((event_name, str(e)))
        
        event_bus.emit = safe_emit
        
        # Run training - should complete despite handler errors
        event_hook = EventDrivenTrainingHook(event_bus)
        trainer = BaseTrainer(
            model=mock_model,
            config=trainer_config,
            callbacks=[event_hook]
        )
        
        # Training should complete successfully
        result = trainer.train(mock_dataloader)
        assert result is not None
        
        # But errors should have been caught
        assert len(errors_caught) > 0
        assert any("batch.completed" in error[0] for error in errors_caught)
    
    def test_backward_compatibility(self, mock_model, mock_dataloader, trainer_config):
        """Test that trainer works without event system (backward compatibility)."""
        # Create trainer without event hooks
        trainer = BaseTrainer(
            model=mock_model,
            config=trainer_config,
            callbacks=[]
        )
        
        # Should train normally
        result = trainer.train(mock_dataloader)
        assert result is not None
        assert result.total_epochs == trainer_config.training.num_epochs
    
    def test_event_filtering(self, event_bus):
        """Test that events can be filtered by criteria."""
        high_loss_events = []
        
        def high_loss_handler(data):
            if data.get("loss", 0) > 0.7:
                high_loss_events.append(data)
        
        event_bus.on("batch.completed", high_loss_handler)
        
        # Emit various events
        event_bus.emit("batch.completed", {"loss": 0.5, "step": 1})
        event_bus.emit("batch.completed", {"loss": 0.8, "step": 2})
        event_bus.emit("batch.completed", {"loss": 0.6, "step": 3})
        event_bus.emit("batch.completed", {"loss": 0.9, "step": 4})
        
        # Only high loss events should be captured
        assert len(high_loss_events) == 2
        assert all(event["loss"] > 0.7 for event in high_loss_events)