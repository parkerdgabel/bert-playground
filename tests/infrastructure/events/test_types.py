"""Tests for event types and data structures."""

import pytest
from datetime import datetime

from infrastructure.events.types import (
    Event,
    EventContext,
    EventPriority,
    EventType,
    TrainingEvent,
)
from domain.protocols.training import TrainingState


class TestEventPriority:
    """Test EventPriority enum."""

    def test_priority_ordering(self):
        """Test that priorities are correctly ordered."""
        assert EventPriority.CRITICAL < EventPriority.HIGH
        assert EventPriority.HIGH < EventPriority.NORMAL
        assert EventPriority.NORMAL < EventPriority.LOW
        assert EventPriority.LOW < EventPriority.DEFERRED

    def test_priority_values(self):
        """Test priority values."""
        assert EventPriority.CRITICAL.value == 0
        assert EventPriority.HIGH.value == 10
        assert EventPriority.NORMAL.value == 50
        assert EventPriority.LOW.value == 100
        assert EventPriority.DEFERRED.value == 1000


class TestEventContext:
    """Test EventContext functionality."""

    def test_context_creation(self):
        """Test creating event context."""
        context = EventContext(
            source="test_source",
            source_type=str
        )
        
        assert context.source == "test_source"
        assert context.source_type == str
        assert isinstance(context.timestamp, datetime)
        assert context.sequence_number == 0
        assert context.metadata == {}
        assert context.tags == set()
        assert context.error is None
        assert context.error_handled is False

    def test_context_tags(self):
        """Test tag operations."""
        context = EventContext(source="test", source_type=str)
        
        # Add tags
        context.add_tag("important")
        context.add_tag("test")
        
        assert context.has_tag("important")
        assert context.has_tag("test")
        assert not context.has_tag("missing")
        assert len(context.tags) == 2

    def test_context_metadata(self):
        """Test metadata operations."""
        context = EventContext(source="test", source_type=str)
        
        # Set metadata
        context.set_metadata("key1", "value1")
        context.set_metadata("key2", 42)
        
        assert context.get_metadata("key1") == "value1"
        assert context.get_metadata("key2") == 42
        assert context.get_metadata("missing") is None
        assert context.get_metadata("missing", "default") == "default"


class TestEvent:
    """Test Event functionality."""

    def test_event_creation(self):
        """Test creating an event."""
        context = EventContext(source="test", source_type=str)
        event = Event(
            type=EventType.TRAINING_STARTED,
            name="test.event",
            context=context
        )
        
        assert event.type == EventType.TRAINING_STARTED
        assert event.name == "test.event"
        assert event.context == context
        assert event.data == {}
        assert event.propagate is True
        assert event.handled is False

    def test_event_data_operations(self):
        """Test event data operations."""
        context = EventContext(source="test", source_type=str)
        event = Event(
            type=EventType.TRAINING_STARTED,
            name="test.event",
            context=context
        )
        
        # Set and get data
        event.set_data("loss", 0.5)
        event.set_data("epoch", 1)
        
        assert event.get_data("loss") == 0.5
        assert event.get_data("epoch") == 1
        assert event.get_data("missing") is None
        assert event.get_data("missing", "default") == "default"

    def test_event_control(self):
        """Test event control methods."""
        context = EventContext(source="test", source_type=str)
        event = Event(
            type=EventType.TRAINING_STARTED,
            name="test.event",
            context=context
        )
        
        # Test propagation control
        assert event.propagate is True
        event.stop_propagation()
        assert event.propagate is False
        
        # Test handled flag
        assert event.handled is False
        event.mark_handled()
        assert event.handled is True

    def test_event_error_detection(self):
        """Test error event detection."""
        context = EventContext(source="test", source_type=str)
        event = Event(
            type=EventType.TRAINING_FAILED,
            name="error.event",
            context=context
        )
        
        assert not event.is_error_event
        
        # Add error to context
        context.error = ValueError("Test error")
        assert event.is_error_event

    def test_event_string_representation(self):
        """Test event string representation."""
        context = EventContext(source="test_source", source_type=str)
        event = Event(
            type=EventType.TRAINING_STARTED,
            name="test.event",
            context=context
        )
        
        str_repr = str(event)
        assert "TRAINING_STARTED" in str_repr
        assert "test.event" in str_repr
        assert "test_source" in str_repr


class TestTrainingEvent:
    """Test TrainingEvent functionality."""

    def test_training_event_creation(self):
        """Test creating a training event."""
        # Create mock training state
        state = TrainingState(
            epoch=5,
            global_step=100,
            num_epochs=10,
            steps_per_epoch=20,
            batch_idx=10,
            train_loss=0.5,
            val_loss=0.4,
            best_val_loss=0.35,
            metrics={"accuracy": 0.9},
            learning_rate=0.001,
            should_stop=False
        )
        
        event = TrainingEvent.create(
            event_type=EventType.EPOCH_COMPLETED,
            name="epoch.completed",
            source="trainer",
            source_type=type("trainer"),
            training_state=state,
            data={"loss": 0.5, "metrics": {"accuracy": 0.9}}
        )
        
        assert event.type == EventType.EPOCH_COMPLETED
        assert event.name == "epoch.completed"
        assert event.training_state == state
        assert event.get_data("loss") == 0.5

    def test_training_event_properties(self):
        """Test training event convenience properties."""
        state = TrainingState(
            epoch=5,
            global_step=100,
            num_epochs=10,
            steps_per_epoch=20,
            batch_idx=10,
            train_loss=0.5,
            val_loss=0.4,
            best_val_loss=0.35,
            metrics={"accuracy": 0.9},
            learning_rate=0.001,
            should_stop=False
        )
        
        event = TrainingEvent.create(
            event_type=EventType.BATCH_COMPLETED,
            name="batch.completed",
            source="trainer",
            source_type=type("trainer"),
            training_state=state,
            data={
                "loss": 0.45,
                "metrics": {"accuracy": 0.92, "f1": 0.88}
            }
        )
        
        # Test convenience properties
        assert event.epoch == 5
        assert event.global_step == 100
        assert event.loss == 0.45
        assert event.metrics == {"accuracy": 0.92, "f1": 0.88}

    def test_training_event_without_state(self):
        """Test training event without training state."""
        event = TrainingEvent.create(
            event_type=EventType.MODEL_LOADED,
            name="model.loaded",
            source="model_loader",
            source_type=type("loader"),
            training_state=None,
            data={"model_path": "/path/to/model"}
        )
        
        assert event.training_state is None
        assert event.epoch is None
        assert event.global_step is None
        assert event.loss is None
        assert event.metrics is None