"""
Event middleware for processing events before they reach handlers.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Awaitable, Callable, Optional, Protocol

from loguru import logger

from .types import Event, EventContext


class EventMiddleware(Protocol):
    """Protocol for event middleware."""

    def process(self, event: Event, next: Callable[[Event], None]) -> None:
        """
        Process an event and optionally call the next middleware.
        
        Args:
            event: The event to process
            next: Function to call the next middleware in the chain
        """
        ...


class AsyncEventMiddleware(Protocol):
    """Protocol for asynchronous event middleware."""

    async def process(
        self, event: Event, next: Callable[[Event], Awaitable[None]]
    ) -> None:
        """
        Process an event asynchronously and optionally call the next middleware.
        
        Args:
            event: The event to process
            next: Async function to call the next middleware in the chain
        """
        ...


@dataclass
class MiddlewareChain:
    """Chain of middleware to process events."""

    middlewares: list[EventMiddleware]

    def process(self, event: Event, final_handler: Callable[[Event], None]) -> None:
        """
        Process event through the middleware chain.
        
        Args:
            event: The event to process
            final_handler: Handler to call after all middleware
        """
        def make_next(index: int) -> Callable[[Event], None]:
            def next_handler(evt: Event) -> None:
                if index < len(self.middlewares):
                    self.middlewares[index].process(evt, make_next(index + 1))
                else:
                    final_handler(evt)
            return next_handler

        if self.middlewares:
            self.middlewares[0].process(event, make_next(1))
        else:
            final_handler(event)


class LoggingMiddleware:
    """Middleware that logs all events."""

    def __init__(
        self,
        log_level: str = "DEBUG",
        include_data: bool = False,
        exclude_types: Optional[set[str]] = None
    ):
        """
        Initialize logging middleware.
        
        Args:
            log_level: Log level to use
            include_data: Whether to include event data in logs
            exclude_types: Event types to exclude from logging
        """
        self.log_level = log_level
        self.include_data = include_data
        self.exclude_types = exclude_types or set()

    def process(self, event: Event, next: Callable[[Event], None]) -> None:
        """Log event and continue processing."""
        if event.type.name not in self.exclude_types:
            message = f"Event: {event.type.name} from {event.context.source}"
            
            if self.include_data and event.data:
                message += f" with data: {event.data}"
            
            logger.log(self.log_level, message)
        
        next(event)


class FilteringMiddleware:
    """Middleware that filters events based on criteria."""

    def __init__(
        self,
        filter_func: Callable[[Event], bool],
        on_filtered: Optional[Callable[[Event], None]] = None
    ):
        """
        Initialize filtering middleware.
        
        Args:
            filter_func: Function to determine if event should be processed
            on_filtered: Optional callback for filtered events
        """
        self.filter_func = filter_func
        self.on_filtered = on_filtered

    def process(self, event: Event, next: Callable[[Event], None]) -> None:
        """Filter events based on criteria."""
        if self.filter_func(event):
            next(event)
        elif self.on_filtered:
            self.on_filtered(event)


class EnrichmentMiddleware:
    """Middleware that enriches events with additional data."""

    def __init__(self, enrichment_func: Callable[[Event], None]):
        """
        Initialize enrichment middleware.
        
        Args:
            enrichment_func: Function to enrich the event
        """
        self.enrichment_func = enrichment_func

    def process(self, event: Event, next: Callable[[Event], None]) -> None:
        """Enrich event and continue processing."""
        self.enrichment_func(event)
        next(event)


class ValidationMiddleware:
    """Middleware that validates events."""

    def __init__(
        self,
        validator: Callable[[Event], bool],
        on_invalid: Optional[Callable[[Event], None]] = None
    ):
        """
        Initialize validation middleware.
        
        Args:
            validator: Function to validate the event
            on_invalid: Optional callback for invalid events
        """
        self.validator = validator
        self.on_invalid = on_invalid

    def process(self, event: Event, next: Callable[[Event], None]) -> None:
        """Validate event before processing."""
        if self.validator(event):
            next(event)
        else:
            logger.warning(f"Invalid event: {event}")
            if self.on_invalid:
                self.on_invalid(event)


class ErrorHandlingMiddleware:
    """Middleware that handles errors in event processing."""

    def __init__(
        self,
        error_handler: Callable[[Event, Exception], None],
        continue_on_error: bool = True
    ):
        """
        Initialize error handling middleware.
        
        Args:
            error_handler: Function to handle errors
            continue_on_error: Whether to continue processing after error
        """
        self.error_handler = error_handler
        self.continue_on_error = continue_on_error

    def process(self, event: Event, next: Callable[[Event], None]) -> None:
        """Process event with error handling."""
        try:
            next(event)
        except Exception as e:
            logger.error(f"Error processing event {event}: {e}")
            self.error_handler(event, e)
            
            if not self.continue_on_error:
                raise


class MetricsMiddleware:
    """Middleware that collects metrics about events."""

    def __init__(self):
        """Initialize metrics middleware."""
        self.event_counts: dict[str, int] = {}
        self.error_counts: dict[str, int] = {}
        self.processing_times: dict[str, list[float]] = {}

    def process(self, event: Event, next: Callable[[Event], None]) -> None:
        """Collect metrics and continue processing."""
        import time
        
        event_type = event.type.name
        
        # Increment event count
        self.event_counts[event_type] = self.event_counts.get(event_type, 0) + 1
        
        # Time processing
        start_time = time.time()
        try:
            next(event)
        except Exception as e:
            # Track errors
            self.error_counts[event_type] = self.error_counts.get(event_type, 0) + 1
            raise
        finally:
            # Track processing time
            elapsed = time.time() - start_time
            if event_type not in self.processing_times:
                self.processing_times[event_type] = []
            self.processing_times[event_type].append(elapsed)

    def get_metrics(self) -> dict[str, any]:
        """Get collected metrics."""
        return {
            "event_counts": self.event_counts.copy(),
            "error_counts": self.error_counts.copy(),
            "average_processing_times": {
                event_type: sum(times) / len(times) if times else 0
                for event_type, times in self.processing_times.items()
            }
        }


class BatchingMiddleware:
    """Middleware that batches events for processing."""

    def __init__(
        self,
        batch_size: int = 10,
        flush_interval: float = 1.0,
        batch_processor: Optional[Callable[[list[Event]], None]] = None
    ):
        """
        Initialize batching middleware.
        
        Args:
            batch_size: Maximum batch size
            flush_interval: Time interval to flush batch (seconds)
            batch_processor: Optional processor for batched events
        """
        self.batch_size = batch_size
        self.flush_interval = flush_interval
        self.batch_processor = batch_processor
        self.batch: list[Event] = []
        self._last_flush = None

    def process(self, event: Event, next: Callable[[Event], None]) -> None:
        """Add event to batch and process when full."""
        import time
        
        self.batch.append(event)
        current_time = time.time()
        
        # Check if we should flush
        should_flush = (
            len(self.batch) >= self.batch_size or
            (self._last_flush and current_time - self._last_flush >= self.flush_interval)
        )
        
        if should_flush:
            self._flush_batch()
        
        # Continue processing individual event
        next(event)

    def _flush_batch(self) -> None:
        """Flush the current batch."""
        import time
        
        if self.batch and self.batch_processor:
            self.batch_processor(self.batch.copy())
        
        self.batch.clear()
        self._last_flush = time.time()