"""
Event types and data structures for the event system.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import Any, Optional

from core.protocols.training import TrainingState


class EventType(Enum):
    """Event types for the k-bert system."""

    # Training lifecycle events
    TRAINING_STARTED = auto()
    TRAINING_COMPLETED = auto()
    TRAINING_FAILED = auto()
    TRAINING_CANCELLED = auto()

    # Epoch events
    EPOCH_STARTED = auto()
    EPOCH_COMPLETED = auto()
    EPOCH_SKIPPED = auto()

    # Batch events
    BATCH_STARTED = auto()
    BATCH_COMPLETED = auto()
    BATCH_FAILED = auto()

    # Evaluation events
    EVALUATION_STARTED = auto()
    EVALUATION_COMPLETED = auto()
    EVALUATION_FAILED = auto()

    # Checkpoint events
    CHECKPOINT_SAVED = auto()
    CHECKPOINT_LOADED = auto()
    CHECKPOINT_DELETED = auto()
    CHECKPOINT_FAILED = auto()

    # Model events
    MODEL_LOADED = auto()
    MODEL_SAVED = auto()
    MODEL_COMPILED = auto()
    MODEL_UPDATED = auto()

    # Data events
    DATA_LOADING_STARTED = auto()
    DATA_LOADING_COMPLETED = auto()
    DATA_BATCH_PREPARED = auto()
    DATA_AUGMENTATION_APPLIED = auto()

    # Optimization events
    OPTIMIZER_STEP_STARTED = auto()
    OPTIMIZER_STEP_COMPLETED = auto()
    LEARNING_RATE_UPDATED = auto()
    GRADIENT_CLIPPED = auto()

    # Metric events
    METRIC_LOGGED = auto()
    METRIC_IMPROVED = auto()
    METRIC_PLATEAU = auto()

    # System events
    MEMORY_WARNING = auto()
    DEVICE_CHANGED = auto()
    CONFIG_UPDATED = auto()
    PLUGIN_LOADED = auto()
    PLUGIN_UNLOADED = auto()

    # Custom events (for user-defined events)
    CUSTOM = auto()


class EventPriority(Enum):
    """Priority levels for event handlers."""

    CRITICAL = 0  # Highest priority
    HIGH = 10
    NORMAL = 50
    LOW = 100
    DEFERRED = 1000  # Lowest priority

    def __lt__(self, other):
        """Compare priorities (lower value = higher priority)."""
        if isinstance(other, EventPriority):
            return self.value < other.value
        return NotImplemented


@dataclass
class EventContext:
    """Context information for an event."""

    # Source information
    source: str  # Component that generated the event
    source_type: type  # Type of the source component

    # Timing information
    timestamp: datetime = field(default_factory=datetime.now)
    sequence_number: int = 0

    # Metadata
    metadata: dict[str, Any] = field(default_factory=dict)
    tags: set[str] = field(default_factory=set)

    # Error information (if applicable)
    error: Optional[Exception] = None
    error_handled: bool = False

    def add_tag(self, tag: str) -> None:
        """Add a tag to the context."""
        self.tags.add(tag)

    def has_tag(self, tag: str) -> bool:
        """Check if context has a specific tag."""
        return tag in self.tags

    def set_metadata(self, key: str, value: Any) -> None:
        """Set metadata value."""
        self.metadata[key] = value

    def get_metadata(self, key: str, default: Any = None) -> Any:
        """Get metadata value."""
        return self.metadata.get(key, default)


@dataclass
class Event:
    """Base event class."""

    # Event identification
    type: EventType
    name: str
    context: EventContext

    # Event data
    data: dict[str, Any] = field(default_factory=dict)

    # Event control
    propagate: bool = True  # Whether to continue propagating to other handlers
    handled: bool = False  # Whether the event has been handled

    @property
    def is_error_event(self) -> bool:
        """Check if this is an error event."""
        return self.context.error is not None

    def stop_propagation(self) -> None:
        """Stop event propagation to subsequent handlers."""
        self.propagate = False

    def mark_handled(self) -> None:
        """Mark the event as handled."""
        self.handled = True

    def get_data(self, key: str, default: Any = None) -> Any:
        """Get event data value."""
        return self.data.get(key, default)

    def set_data(self, key: str, value: Any) -> None:
        """Set event data value."""
        self.data[key] = value

    def __str__(self) -> str:
        """String representation of the event."""
        return (
            f"Event(type={self.type.name}, name={self.name}, "
            f"source={self.context.source}, timestamp={self.context.timestamp})"
        )


@dataclass
class TrainingEvent(Event):
    """
    Specialized event for training-related events.
    
    Includes training state information for easier access.
    """

    training_state: Optional[TrainingState] = None

    @classmethod
    def create(
        cls,
        event_type: EventType,
        name: str,
        source: str,
        source_type: type,
        training_state: Optional[TrainingState] = None,
        data: Optional[dict[str, Any]] = None,
        **kwargs
    ) -> "TrainingEvent":
        """Create a new training event."""
        context = EventContext(
            source=source,
            source_type=source_type,
            **kwargs
        )
        
        return cls(
            type=event_type,
            name=name,
            context=context,
            data=data or {},
            training_state=training_state
        )

    @property
    def epoch(self) -> Optional[int]:
        """Get current epoch from training state."""
        return self.training_state.epoch if self.training_state else None

    @property
    def global_step(self) -> Optional[int]:
        """Get global step from training state."""
        return self.training_state.global_step if self.training_state else None

    @property
    def loss(self) -> Optional[float]:
        """Get current loss from event data."""
        return self.get_data("loss")

    @property
    def metrics(self) -> Optional[dict[str, float]]:
        """Get metrics from event data."""
        return self.get_data("metrics")