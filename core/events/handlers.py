"""
Event handler interfaces and implementations.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Optional, Protocol, Union

from .types import Event, EventPriority, EventType


class EventFilter(Protocol):
    """Protocol for event filters."""

    def should_handle(self, event: Event) -> bool:
        """
        Determine if an event should be handled.
        
        Args:
            event: The event to check
            
        Returns:
            True if the event should be handled, False otherwise
        """
        ...


class EventHandler(Protocol):
    """Protocol for synchronous event handlers."""

    @property
    def priority(self) -> EventPriority:
        """Handler priority (lower value = higher priority)."""
        return EventPriority.NORMAL

    def handle(self, event: Event) -> None:
        """
        Handle an event synchronously.
        
        Args:
            event: The event to handle
        """
        ...


class AsyncEventHandler(Protocol):
    """Protocol for asynchronous event handlers."""

    @property
    def priority(self) -> EventPriority:
        """Handler priority (lower value = higher priority)."""
        return EventPriority.NORMAL

    async def handle(self, event: Event) -> None:
        """
        Handle an event asynchronously.
        
        Args:
            event: The event to handle
        """
        ...


@dataclass
class HandlerRegistration:
    """Registration information for an event handler."""

    handler: Union[EventHandler, AsyncEventHandler]
    event_types: set[EventType]
    filter: Optional[EventFilter] = None
    priority: EventPriority = EventPriority.NORMAL
    name: Optional[str] = None
    enabled: bool = True
    tags: set[str] = field(default_factory=set)

    def __post_init__(self):
        """Set handler name if not provided."""
        if self.name is None:
            self.name = self.handler.__class__.__name__

    def should_handle(self, event: Event) -> bool:
        """Check if this handler should handle the event."""
        if not self.enabled:
            return False
            
        # Check event type
        if event.type not in self.event_types and EventType.CUSTOM not in self.event_types:
            return False
            
        # Check filter if present
        if self.filter and not self.filter.should_handle(event):
            return False
            
        return True

    @property
    def is_async(self) -> bool:
        """Check if handler is asynchronous."""
        return hasattr(self.handler.handle, "__await__")


class SimpleEventFilter:
    """Simple event filter implementation."""

    def __init__(
        self,
        event_types: Optional[set[EventType]] = None,
        tags: Optional[set[str]] = None,
        sources: Optional[set[str]] = None,
        custom_filter: Optional[Callable[[Event], bool]] = None
    ):
        """
        Initialize filter.
        
        Args:
            event_types: Set of event types to allow
            tags: Set of tags that events must have
            sources: Set of allowed event sources
            custom_filter: Custom filter function
        """
        self.event_types = event_types
        self.tags = tags
        self.sources = sources
        self.custom_filter = custom_filter

    def should_handle(self, event: Event) -> bool:
        """Check if event passes all filters."""
        # Check event type
        if self.event_types and event.type not in self.event_types:
            return False
            
        # Check tags
        if self.tags and not any(event.context.has_tag(tag) for tag in self.tags):
            return False
            
        # Check source
        if self.sources and event.context.source not in self.sources:
            return False
            
        # Check custom filter
        if self.custom_filter and not self.custom_filter(event):
            return False
            
        return True


class FunctionEventHandler:
    """Wrapper to convert a function to an EventHandler."""

    def __init__(
        self,
        func: Callable[[Event], None],
        priority: EventPriority = EventPriority.NORMAL
    ):
        """
        Initialize function handler.
        
        Args:
            func: Function to call for events
            priority: Handler priority
        """
        self.func = func
        self._priority = priority

    @property
    def priority(self) -> EventPriority:
        """Handler priority."""
        return self._priority

    def handle(self, event: Event) -> None:
        """Handle event by calling the function."""
        self.func(event)


class AsyncFunctionEventHandler:
    """Wrapper to convert an async function to an AsyncEventHandler."""

    def __init__(
        self,
        func: Callable[[Event], Awaitable[None]],
        priority: EventPriority = EventPriority.NORMAL
    ):
        """
        Initialize async function handler.
        
        Args:
            func: Async function to call for events
            priority: Handler priority
        """
        self.func = func
        self._priority = priority

    @property
    def priority(self) -> EventPriority:
        """Handler priority."""
        return self._priority

    async def handle(self, event: Event) -> None:
        """Handle event by calling the async function."""
        await self.func(event)


class CompositeEventHandler(EventHandler):
    """Handler that delegates to multiple sub-handlers."""

    def __init__(self, handlers: list[EventHandler]):
        """
        Initialize composite handler.
        
        Args:
            handlers: List of handlers to delegate to
        """
        self.handlers = sorted(handlers, key=lambda h: h.priority.value)

    @property
    def priority(self) -> EventPriority:
        """Return highest priority among sub-handlers."""
        return self.handlers[0].priority if self.handlers else EventPriority.NORMAL

    def handle(self, event: Event) -> None:
        """Handle event by delegating to all sub-handlers."""
        for handler in self.handlers:
            if not event.propagate:
                break
            handler.handle(event)


class ErrorHandlingWrapper(EventHandler):
    """Wrapper that adds error handling to a handler."""

    def __init__(
        self,
        handler: EventHandler,
        error_handler: Optional[Callable[[Event, Exception], None]] = None,
        reraise: bool = False
    ):
        """
        Initialize error handling wrapper.
        
        Args:
            handler: Handler to wrap
            error_handler: Optional error handler function
            reraise: Whether to re-raise exceptions after handling
        """
        self.handler = handler
        self.error_handler = error_handler
        self.reraise = reraise

    @property
    def priority(self) -> EventPriority:
        """Handler priority."""
        return self.handler.priority

    def handle(self, event: Event) -> None:
        """Handle event with error handling."""
        try:
            self.handler.handle(event)
        except Exception as e:
            if self.error_handler:
                self.error_handler(event, e)
            
            # Add error to event context
            event.context.error = e
            event.context.error_handled = not self.reraise
            
            if self.reraise:
                raise