"""
Event system for k-bert.

Provides a decoupled publish-subscribe event system with support for:
- Synchronous and asynchronous event handlers
- Event priorities and filtering
- Middleware for event processing
- Integration with the existing callback system
"""

from .bus import AsyncEventBus, EventBus, GlobalEventBus
from .handlers import (
    AsyncEventHandler,
    EventFilter,
    EventHandler,
    HandlerRegistration,
)
from .middleware import EventMiddleware, MiddlewareChain
from .types import (
    Event,
    EventContext,
    EventPriority,
    EventType,
    TrainingEvent,
)

__all__ = [
    # Event bus
    "EventBus",
    "AsyncEventBus",
    "GlobalEventBus",
    # Handlers
    "EventHandler",
    "AsyncEventHandler",
    "EventFilter",
    "HandlerRegistration",
    # Middleware
    "EventMiddleware",
    "MiddlewareChain",
    # Types
    "Event",
    "EventType",
    "EventPriority",
    "EventContext",
    "TrainingEvent",
]