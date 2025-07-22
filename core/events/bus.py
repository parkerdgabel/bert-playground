"""
Event bus implementation for publish-subscribe pattern.
"""

import asyncio
import threading
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from typing import Any, Callable, Optional, Union

from loguru import logger

from .handlers import (
    AsyncEventHandler,
    EventHandler,
    HandlerRegistration,
    SimpleEventFilter,
)
from .middleware import EventMiddleware, MiddlewareChain
from .types import Event, EventPriority, EventType


class EventBus:
    """
    Synchronous event bus implementation.
    
    Provides a publish-subscribe mechanism for decoupled event handling.
    """

    def __init__(
        self,
        name: str = "default",
        middleware: Optional[list[EventMiddleware]] = None,
        error_handler: Optional[Callable[[Event, Exception], None]] = None
    ):
        """
        Initialize event bus.
        
        Args:
            name: Name of the event bus
            middleware: List of middleware to apply to all events
            error_handler: Global error handler for failed event processing
        """
        self.name = name
        self.middleware = MiddlewareChain(middleware or [])
        self.error_handler = error_handler
        
        # Handler registrations by event type
        self._handlers: dict[EventType, list[HandlerRegistration]] = defaultdict(list)
        
        # All registrations for management
        self._registrations: dict[str, HandlerRegistration] = {}
        
        # Thread-safe lock
        self._lock = threading.RLock()
        
        # Event processing state
        self._processing = False
        self._event_queue: list[Event] = []
        
        logger.debug(f"EventBus '{name}' initialized")

    def subscribe(
        self,
        handler: Union[EventHandler, Callable[[Event], None]],
        event_types: Union[EventType, list[EventType]],
        priority: EventPriority = EventPriority.NORMAL,
        filter: Optional[Any] = None,
        name: Optional[str] = None,
        tags: Optional[set[str]] = None
    ) -> str:
        """
        Subscribe a handler to one or more event types.
        
        Args:
            handler: Event handler or callable
            event_types: Event type(s) to subscribe to
            priority: Handler priority
            filter: Optional event filter
            name: Optional handler name
            tags: Optional tags for the registration
            
        Returns:
            Registration ID for unsubscribing
        """
        from .handlers import FunctionEventHandler
        
        # Convert callable to handler
        if callable(handler) and not hasattr(handler, "handle"):
            handler = FunctionEventHandler(handler, priority)
        
        # Normalize event types
        if isinstance(event_types, EventType):
            event_types = [event_types]
        event_types_set = set(event_types)
        
        # Create registration
        registration = HandlerRegistration(
            handler=handler,
            event_types=event_types_set,
            filter=filter,
            priority=priority,
            name=name,
            tags=tags or set()
        )
        
        # Register handler
        with self._lock:
            reg_id = f"{self.name}_{len(self._registrations)}"
            self._registrations[reg_id] = registration
            
            for event_type in event_types_set:
                self._handlers[event_type].append(registration)
                # Sort by priority
                self._handlers[event_type].sort(key=lambda r: r.priority.value)
            
        logger.debug(
            f"Handler '{registration.name}' subscribed to {len(event_types_set)} "
            f"event types with priority {priority.name}"
        )
        
        return reg_id

    def unsubscribe(self, registration_id: str) -> bool:
        """
        Unsubscribe a handler.
        
        Args:
            registration_id: Registration ID returned by subscribe
            
        Returns:
            True if unsubscribed, False if not found
        """
        with self._lock:
            if registration_id not in self._registrations:
                return False
            
            registration = self._registrations.pop(registration_id)
            
            # Remove from event type mappings
            for event_type in registration.event_types:
                if event_type in self._handlers:
                    self._handlers[event_type] = [
                        r for r in self._handlers[event_type]
                        if r != registration
                    ]
                    
        logger.debug(f"Handler '{registration.name}' unsubscribed")
        return True

    def publish(self, event: Event) -> None:
        """
        Publish an event to all subscribed handlers.
        
        Args:
            event: Event to publish
        """
        # Add sequence number
        with self._lock:
            event.context.sequence_number = len(self._event_queue)
            self._event_queue.append(event)
        
        # Process through middleware
        self.middleware.process(event, self._dispatch_event)

    def _dispatch_event(self, event: Event) -> None:
        """
        Dispatch event to handlers.
        
        Args:
            event: Event to dispatch
        """
        # Get handlers for this event type
        with self._lock:
            handlers = self._handlers.get(event.type, []).copy()
            
            # Also check for handlers subscribed to all events
            if EventType.CUSTOM in self._handlers:
                handlers.extend(self._handlers[EventType.CUSTOM])
                handlers.sort(key=lambda r: r.priority.value)
        
        # Process handlers
        for registration in handlers:
            if not event.propagate:
                logger.debug(f"Event propagation stopped for {event}")
                break
                
            if not registration.should_handle(event):
                continue
                
            try:
                registration.handler.handle(event)
            except Exception as e:
                logger.error(
                    f"Error in handler '{registration.name}' for event {event}: {e}"
                )
                
                if self.error_handler:
                    self.error_handler(event, e)
                else:
                    # Re-raise if no error handler
                    raise

    def emit(
        self,
        event_type: EventType,
        name: str,
        source: Any,
        data: Optional[dict[str, Any]] = None,
        **kwargs
    ) -> None:
        """
        Convenience method to create and publish an event.
        
        Args:
            event_type: Type of event
            name: Event name
            source: Event source
            data: Event data
            **kwargs: Additional context fields
        """
        from .types import EventContext
        
        context = EventContext(
            source=str(source),
            source_type=type(source),
            **kwargs
        )
        
        event = Event(
            type=event_type,
            name=name,
            context=context,
            data=data or {}
        )
        
        self.publish(event)

    def enable_handler(self, registration_id: str) -> bool:
        """Enable a disabled handler."""
        with self._lock:
            if registration_id in self._registrations:
                self._registrations[registration_id].enabled = True
                return True
        return False

    def disable_handler(self, registration_id: str) -> bool:
        """Disable a handler without unsubscribing."""
        with self._lock:
            if registration_id in self._registrations:
                self._registrations[registration_id].enabled = False
                return True
        return False

    def get_handlers(
        self, event_type: Optional[EventType] = None
    ) -> list[HandlerRegistration]:
        """Get all handlers or handlers for a specific event type."""
        with self._lock:
            if event_type:
                return self._handlers.get(event_type, []).copy()
            else:
                return list(self._registrations.values())

    def clear(self) -> None:
        """Remove all handlers."""
        with self._lock:
            self._handlers.clear()
            self._registrations.clear()
            self._event_queue.clear()

    @contextmanager
    def batch_publish(self):
        """Context manager for batching event publication."""
        batch_events = []
        original_publish = self.publish
        
        def batch_publish_impl(event: Event) -> None:
            batch_events.append(event)
        
        try:
            self.publish = batch_publish_impl
            yield
        finally:
            self.publish = original_publish
            # Publish all batched events
            for event in batch_events:
                self.publish(event)


class AsyncEventBus(EventBus):
    """
    Asynchronous event bus implementation.
    
    Supports both synchronous and asynchronous event handlers.
    """

    def __init__(
        self,
        name: str = "async_default",
        middleware: Optional[list[EventMiddleware]] = None,
        error_handler: Optional[Callable[[Event, Exception], None]] = None,
        executor: Optional[ThreadPoolExecutor] = None
    ):
        """
        Initialize async event bus.
        
        Args:
            name: Name of the event bus
            middleware: List of middleware to apply to all events
            error_handler: Global error handler
            executor: Thread pool executor for sync handlers
        """
        super().__init__(name, middleware, error_handler)
        self.executor = executor or ThreadPoolExecutor(max_workers=4)
        self._loop: Optional[asyncio.AbstractEventLoop] = None

    async def async_publish(self, event: Event) -> None:
        """
        Publish an event asynchronously.
        
        Args:
            event: Event to publish
        """
        # Add sequence number
        with self._lock:
            event.context.sequence_number = len(self._event_queue)
            self._event_queue.append(event)
        
        # Process through middleware (sync for now)
        await self._async_dispatch_event(event)

    async def _async_dispatch_event(self, event: Event) -> None:
        """
        Dispatch event to handlers asynchronously.
        
        Args:
            event: Event to dispatch
        """
        # Get handlers for this event type
        with self._lock:
            handlers = self._handlers.get(event.type, []).copy()
            
            if EventType.CUSTOM in self._handlers:
                handlers.extend(self._handlers[EventType.CUSTOM])
                handlers.sort(key=lambda r: r.priority.value)
        
        # Process handlers
        tasks = []
        for registration in handlers:
            if not event.propagate:
                break
                
            if not registration.should_handle(event):
                continue
            
            if registration.is_async:
                # Async handler
                task = self._handle_async(registration, event)
            else:
                # Sync handler - run in executor
                task = self._handle_sync_in_executor(registration, event)
            
            tasks.append(task)
        
        # Wait for all handlers
        if tasks:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Handle errors
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    handler_name = handlers[i].name
                    logger.error(f"Error in handler '{handler_name}': {result}")
                    
                    if self.error_handler:
                        self.error_handler(event, result)

    async def _handle_async(
        self, registration: HandlerRegistration, event: Event
    ) -> None:
        """Handle event with async handler."""
        try:
            await registration.handler.handle(event)
        except Exception as e:
            logger.error(f"Error in async handler '{registration.name}': {e}")
            raise

    async def _handle_sync_in_executor(
        self, registration: HandlerRegistration, event: Event
    ) -> None:
        """Handle event with sync handler in executor."""
        loop = asyncio.get_event_loop()
        try:
            await loop.run_in_executor(
                self.executor,
                registration.handler.handle,
                event
            )
        except Exception as e:
            logger.error(f"Error in sync handler '{registration.name}': {e}")
            raise

    async def async_emit(
        self,
        event_type: EventType,
        name: str,
        source: Any,
        data: Optional[dict[str, Any]] = None,
        **kwargs
    ) -> None:
        """
        Async convenience method to create and publish an event.
        
        Args:
            event_type: Type of event
            name: Event name
            source: Event source
            data: Event data
            **kwargs: Additional context fields
        """
        from .types import EventContext
        
        context = EventContext(
            source=str(source),
            source_type=type(source),
            **kwargs
        )
        
        event = Event(
            type=event_type,
            name=name,
            context=context,
            data=data or {}
        )
        
        await self.async_publish(event)

    def __del__(self):
        """Cleanup executor on deletion."""
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=False)


# Global event bus instance
_global_bus: Optional[EventBus] = None
_global_bus_lock = threading.Lock()


def GlobalEventBus() -> EventBus:
    """
    Get the global event bus instance.
    
    Creates one if it doesn't exist.
    """
    global _global_bus
    
    if _global_bus is None:
        with _global_bus_lock:
            if _global_bus is None:
                _global_bus = EventBus(name="global")
    
    return _global_bus