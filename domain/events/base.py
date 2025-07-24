"""Base domain event class.

All domain events inherit from this base class.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict
from uuid import uuid4


@dataclass
class DomainEvent:
    """Base class for all domain events.
    
    Domain events are immutable records of something that has happened
    in the domain. They are used for:
    - Audit logging
    - Event sourcing
    - Integration between bounded contexts
    - Triggering side effects
    """
    event_id: str = field(default_factory=lambda: str(uuid4()))
    event_type: str = field(init=False)
    occurred_at: datetime = field(default_factory=datetime.now)
    aggregate_id: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Set event type from class name."""
        self.event_type = self.__class__.__name__
    
    @property
    def event_name(self) -> str:
        """Human-readable event name."""
        # Convert CamelCase to space-separated
        import re
        return re.sub(r'(?<!^)(?=[A-Z])', ' ', self.event_type)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert event to dictionary for serialization."""
        return {
            "event_id": self.event_id,
            "event_type": self.event_type,
            "occurred_at": self.occurred_at.isoformat(),
            "aggregate_id": self.aggregate_id,
            "metadata": self.metadata,
            "data": {
                k: v for k, v in self.__dict__.items()
                if k not in ["event_id", "event_type", "occurred_at", "aggregate_id", "metadata"]
            }
        }