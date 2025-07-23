"""Tests for event handlers."""

import asyncio
from unittest.mock import MagicMock, Mock

import pytest

from infrastructure.events.handlers import (
    AsyncFunctionEventHandler,
    CompositeEventHandler,
    ErrorHandlingWrapper,
    FunctionEventHandler,
    HandlerRegistration,
    SimpleEventFilter,
)
from infrastructure.events.types import Event, EventContext, EventPriority, EventType


class TestHandlerRegistration:
    """Test HandlerRegistration functionality."""

    def test_registration_creation(self):
        """Test creating handler registration."""
        handler = MagicMock()
        handler.priority = EventPriority.NORMAL
        
        registration = HandlerRegistration(
            handler=handler,
            event_types={EventType.TRAINING_STARTED, EventType.TRAINING_COMPLETED},
            priority=EventPriority.HIGH,
            name="test_handler"
        )
        
        assert registration.handler == handler
        assert EventType.TRAINING_STARTED in registration.event_types
        assert EventType.TRAINING_COMPLETED in registration.event_types
        assert registration.priority == EventPriority.HIGH
        assert registration.name == "test_handler"
        assert registration.enabled is True

    def test_registration_auto_naming(self):
        """Test automatic naming of registration."""
        handler = MagicMock()
        handler.__class__.__name__ = "MockHandler"
        
        registration = HandlerRegistration(
            handler=handler,
            event_types={EventType.TRAINING_STARTED}
        )
        
        assert registration.name == "MockHandler"

    def test_should_handle_event_type(self):
        """Test event type filtering."""
        handler = MagicMock()
        registration = HandlerRegistration(
            handler=handler,
            event_types={EventType.TRAINING_STARTED, EventType.EPOCH_STARTED}
        )
        
        # Create test events
        context = EventContext(source="test", source_type=str)
        
        training_event = Event(
            type=EventType.TRAINING_STARTED,
            name="test",
            context=context
        )
        assert registration.should_handle(training_event)
        
        epoch_event = Event(
            type=EventType.EPOCH_STARTED,
            name="test",
            context=context
        )
        assert registration.should_handle(epoch_event)
        
        batch_event = Event(
            type=EventType.BATCH_STARTED,
            name="test",
            context=context
        )
        assert not registration.should_handle(batch_event)

    def test_should_handle_disabled(self):
        """Test disabled handler."""
        handler = MagicMock()
        registration = HandlerRegistration(
            handler=handler,
            event_types={EventType.TRAINING_STARTED},
            enabled=False
        )
        
        context = EventContext(source="test", source_type=str)
        event = Event(
            type=EventType.TRAINING_STARTED,
            name="test",
            context=context
        )
        
        assert not registration.should_handle(event)

    def test_should_handle_with_filter(self):
        """Test handler with filter."""
        handler = MagicMock()
        filter_mock = Mock()
        filter_mock.should_handle.return_value = False
        
        registration = HandlerRegistration(
            handler=handler,
            event_types={EventType.TRAINING_STARTED},
            filter=filter_mock
        )
        
        context = EventContext(source="test", source_type=str)
        event = Event(
            type=EventType.TRAINING_STARTED,
            name="test",
            context=context
        )
        
        assert not registration.should_handle(event)
        filter_mock.should_handle.assert_called_once_with(event)

    def test_is_async_detection(self):
        """Test async handler detection."""
        # Sync handler
        sync_handler = MagicMock()
        sync_registration = HandlerRegistration(
            handler=sync_handler,
            event_types={EventType.TRAINING_STARTED}
        )
        assert not sync_registration.is_async

        # Async handler
        async def async_handle(event):
            pass
        
        async_handler = MagicMock()
        async_handler.handle = async_handle
        
        async_registration = HandlerRegistration(
            handler=async_handler,
            event_types={EventType.TRAINING_STARTED}
        )
        assert async_registration.is_async


class TestSimpleEventFilter:
    """Test SimpleEventFilter functionality."""

    def test_filter_by_event_type(self):
        """Test filtering by event type."""
        filter = SimpleEventFilter(
            event_types={EventType.TRAINING_STARTED, EventType.TRAINING_COMPLETED}
        )
        
        context = EventContext(source="test", source_type=str)
        
        # Should pass
        event1 = Event(
            type=EventType.TRAINING_STARTED,
            name="test",
            context=context
        )
        assert filter.should_handle(event1)
        
        # Should not pass
        event2 = Event(
            type=EventType.EPOCH_STARTED,
            name="test",
            context=context
        )
        assert not filter.should_handle(event2)

    def test_filter_by_tags(self):
        """Test filtering by tags."""
        filter = SimpleEventFilter(tags={"important", "critical"})
        
        context1 = EventContext(source="test", source_type=str)
        context1.add_tag("important")
        event1 = Event(
            type=EventType.TRAINING_STARTED,
            name="test",
            context=context1
        )
        assert filter.should_handle(event1)
        
        context2 = EventContext(source="test", source_type=str)
        context2.add_tag("normal")
        event2 = Event(
            type=EventType.TRAINING_STARTED,
            name="test",
            context=context2
        )
        assert not filter.should_handle(event2)

    def test_filter_by_source(self):
        """Test filtering by source."""
        filter = SimpleEventFilter(sources={"trainer", "evaluator"})
        
        # Should pass
        context1 = EventContext(source="trainer", source_type=str)
        event1 = Event(
            type=EventType.TRAINING_STARTED,
            name="test",
            context=context1
        )
        assert filter.should_handle(event1)
        
        # Should not pass
        context2 = EventContext(source="loader", source_type=str)
        event2 = Event(
            type=EventType.TRAINING_STARTED,
            name="test",
            context=context2
        )
        assert not filter.should_handle(event2)

    def test_custom_filter(self):
        """Test custom filter function."""
        def custom_filter(event: Event) -> bool:
            return event.get_data("priority") == "high"
        
        filter = SimpleEventFilter(custom_filter=custom_filter)
        
        context = EventContext(source="test", source_type=str)
        
        # Should pass
        event1 = Event(
            type=EventType.TRAINING_STARTED,
            name="test",
            context=context,
            data={"priority": "high"}
        )
        assert filter.should_handle(event1)
        
        # Should not pass
        event2 = Event(
            type=EventType.TRAINING_STARTED,
            name="test",
            context=context,
            data={"priority": "low"}
        )
        assert not filter.should_handle(event2)


class TestFunctionEventHandler:
    """Test FunctionEventHandler wrapper."""

    def test_function_handler(self):
        """Test wrapping a function as event handler."""
        called_events = []
        
        def handle_event(event: Event):
            called_events.append(event)
        
        handler = FunctionEventHandler(
            handle_event,
            priority=EventPriority.HIGH
        )
        
        assert handler.priority == EventPriority.HIGH
        
        # Test handling
        context = EventContext(source="test", source_type=str)
        event = Event(
            type=EventType.TRAINING_STARTED,
            name="test",
            context=context
        )
        
        handler.handle(event)
        assert len(called_events) == 1
        assert called_events[0] == event


class TestAsyncFunctionEventHandler:
    """Test AsyncFunctionEventHandler wrapper."""

    @pytest.mark.asyncio
    async def test_async_function_handler(self):
        """Test wrapping an async function as event handler."""
        called_events = []
        
        async def handle_event(event: Event):
            called_events.append(event)
            await asyncio.sleep(0.01)  # Simulate async work
        
        handler = AsyncFunctionEventHandler(
            handle_event,
            priority=EventPriority.LOW
        )
        
        assert handler.priority == EventPriority.LOW
        
        # Test handling
        context = EventContext(source="test", source_type=str)
        event = Event(
            type=EventType.TRAINING_STARTED,
            name="test",
            context=context
        )
        
        await handler.handle(event)
        assert len(called_events) == 1
        assert called_events[0] == event


class TestCompositeEventHandler:
    """Test CompositeEventHandler functionality."""

    def test_composite_handler_ordering(self):
        """Test that composite handler respects priority ordering."""
        called_order = []
        
        # Create handlers with different priorities
        handler1 = MagicMock()
        handler1.priority = EventPriority.LOW
        handler1.handle.side_effect = lambda e: called_order.append(1)
        
        handler2 = MagicMock()
        handler2.priority = EventPriority.HIGH
        handler2.handle.side_effect = lambda e: called_order.append(2)
        
        handler3 = MagicMock()
        handler3.priority = EventPriority.NORMAL
        handler3.handle.side_effect = lambda e: called_order.append(3)
        
        composite = CompositeEventHandler([handler1, handler2, handler3])
        
        # Should have highest priority
        assert composite.priority == EventPriority.HIGH
        
        # Test handling
        context = EventContext(source="test", source_type=str)
        event = Event(
            type=EventType.TRAINING_STARTED,
            name="test",
            context=context
        )
        
        composite.handle(event)
        
        # Should be called in priority order: HIGH, NORMAL, LOW
        assert called_order == [2, 3, 1]

    def test_composite_handler_propagation(self):
        """Test that composite handler respects propagation control."""
        called = []
        
        def handler1(event):
            called.append(1)
            event.stop_propagation()
        
        def handler2(event):
            called.append(2)
        
        handler1_wrapper = FunctionEventHandler(handler1)
        handler2_wrapper = FunctionEventHandler(handler2)
        
        composite = CompositeEventHandler([handler1_wrapper, handler2_wrapper])
        
        context = EventContext(source="test", source_type=str)
        event = Event(
            type=EventType.TRAINING_STARTED,
            name="test",
            context=context
        )
        
        composite.handle(event)
        
        # Only first handler should be called
        assert called == [1]


class TestErrorHandlingWrapper:
    """Test ErrorHandlingWrapper functionality."""

    def test_error_handling_wrapper_success(self):
        """Test wrapper when handler succeeds."""
        handler = MagicMock()
        handler.priority = EventPriority.NORMAL
        handler.handle.return_value = None
        
        wrapper = ErrorHandlingWrapper(handler)
        
        context = EventContext(source="test", source_type=str)
        event = Event(
            type=EventType.TRAINING_STARTED,
            name="test",
            context=context
        )
        
        # Should not raise
        wrapper.handle(event)
        handler.handle.assert_called_once_with(event)
        assert event.context.error is None

    def test_error_handling_wrapper_with_error(self):
        """Test wrapper when handler raises error."""
        handler = MagicMock()
        handler.priority = EventPriority.NORMAL
        handler.handle.side_effect = ValueError("Test error")
        
        error_handler = Mock()
        wrapper = ErrorHandlingWrapper(
            handler,
            error_handler=error_handler,
            reraise=False
        )
        
        context = EventContext(source="test", source_type=str)
        event = Event(
            type=EventType.TRAINING_STARTED,
            name="test",
            context=context
        )
        
        # Should not raise
        wrapper.handle(event)
        
        # Error should be recorded
        assert isinstance(event.context.error, ValueError)
        assert event.context.error_handled is True
        
        # Error handler should be called
        error_handler.assert_called_once()
        args = error_handler.call_args[0]
        assert args[0] == event
        assert isinstance(args[1], ValueError)

    def test_error_handling_wrapper_reraise(self):
        """Test wrapper with reraise option."""
        handler = MagicMock()
        handler.priority = EventPriority.NORMAL
        handler.handle.side_effect = ValueError("Test error")
        
        wrapper = ErrorHandlingWrapper(handler, reraise=True)
        
        context = EventContext(source="test", source_type=str)
        event = Event(
            type=EventType.TRAINING_STARTED,
            name="test",
            context=context
        )
        
        # Should raise
        with pytest.raises(ValueError, match="Test error"):
            wrapper.handle(event)
        
        # Error should be recorded but not handled
        assert isinstance(event.context.error, ValueError)
        assert event.context.error_handled is False