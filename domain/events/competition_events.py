"""Competition-related domain events."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

from .base import DomainEvent


@dataclass
class CompetitionJoined(DomainEvent):
    """Event raised when joining a competition."""
    competition_id: str = ""
    competition_name: str = ""
    platform: str = ""
    team_name: Optional[str] = None
    
    def __post_init__(self):
        super().__post_init__()
        self.aggregate_id = self.competition_id


@dataclass
class CompetitionDataDownloaded(DomainEvent):
    """Event raised when competition data is downloaded."""
    competition_id: str = ""
    data_size_mb: float = 0.0
    file_count: int = 0
    download_duration_seconds: float = 0.0
    
    def __post_init__(self):
        super().__post_init__()
        self.aggregate_id = self.competition_id


@dataclass
class CompetitionDeadlineApproaching(DomainEvent):
    """Event raised when competition deadline is approaching."""
    competition_id: str = ""
    deadline: datetime = field(default_factory=datetime.now)
    days_remaining: int = 0
    hours_remaining: int = 0
    submission_count: int = 0
    submissions_remaining: int = 0
    
    def __post_init__(self):
        super().__post_init__()
        self.aggregate_id = self.competition_id
        
        # Add urgency to metadata
        if self.days_remaining <= 1:
            self.metadata["urgency"] = "critical"
        elif self.days_remaining <= 3:
            self.metadata["urgency"] = "high"
        elif self.days_remaining <= 7:
            self.metadata["urgency"] = "medium"
        else:
            self.metadata["urgency"] = "low"


@dataclass
class CompetitionCompleted(DomainEvent):
    """Event raised when competition ends."""
    competition_id: str = ""
    final_rank: Optional[int] = None
    total_participants: Optional[int] = None
    final_score: Optional[float] = None
    total_submissions: int = 0
    best_submission_id: Optional[str] = None
    
    def __post_init__(self):
        super().__post_init__()
        self.aggregate_id = self.competition_id
        
        # Add achievement level to metadata
        if self.final_rank and self.total_participants:
            percentile = (1 - self.final_rank / self.total_participants) * 100
            if percentile >= 99:
                self.metadata["achievement"] = "top_1_percent"
            elif percentile >= 95:
                self.metadata["achievement"] = "top_5_percent"
            elif percentile >= 90:
                self.metadata["achievement"] = "top_10_percent"
            elif percentile >= 75:
                self.metadata["achievement"] = "top_25_percent"