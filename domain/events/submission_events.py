"""Submission-related domain events."""

from dataclasses import dataclass
from typing import Optional, List

from .base import DomainEvent


@dataclass
class SubmissionCreated(DomainEvent):
    """Event raised when a submission is created."""
    submission_id: str
    competition_id: str
    experiment_id: str
    submission_type: str
    model_count: int
    
    def __post_init__(self):
        super().__post_init__()
        self.aggregate_id = self.submission_id


@dataclass
class SubmissionValidated(DomainEvent):
    """Event raised when a submission passes validation."""
    submission_id: str
    competition_id: str
    is_valid: bool
    errors: List[str] = None
    warnings: List[str] = None
    
    def __post_init__(self):
        super().__post_init__()
        self.aggregate_id = self.submission_id
        if self.errors is None:
            self.errors = []
        if self.warnings is None:
            self.warnings = []
        
        # Add validation status to metadata
        if self.is_valid:
            self.metadata["validation_status"] = "passed"
        else:
            self.metadata["validation_status"] = "failed"
            self.metadata["error_count"] = len(self.errors)


@dataclass
class SubmissionSubmitted(DomainEvent):
    """Event raised when a submission is submitted to competition."""
    submission_id: str
    competition_id: str
    submission_number: int  # Which submission this is for the competition
    daily_submissions_used: int
    daily_submissions_limit: int
    
    def __post_init__(self):
        super().__post_init__()
        self.aggregate_id = self.submission_id
        
        # Track submission quota usage
        usage_ratio = self.daily_submissions_used / self.daily_submissions_limit
        if usage_ratio >= 1.0:
            self.metadata["quota_status"] = "exhausted"
        elif usage_ratio >= 0.8:
            self.metadata["quota_status"] = "nearly_exhausted"
        else:
            self.metadata["quota_status"] = "available"


@dataclass
class SubmissionScored(DomainEvent):
    """Event raised when a submission receives a score."""
    submission_id: str
    competition_id: str
    public_score: float
    public_rank: Optional[int] = None
    score_improvement: Optional[float] = None
    is_best_score: bool = False
    
    def __post_init__(self):
        super().__post_init__()
        self.aggregate_id = self.submission_id
        
        # Categorize score improvement
        if self.score_improvement is not None:
            if self.score_improvement > 0.05:
                self.metadata["improvement_level"] = "major"
            elif self.score_improvement > 0.01:
                self.metadata["improvement_level"] = "moderate"
            elif self.score_improvement > 0:
                self.metadata["improvement_level"] = "minor"
            else:
                self.metadata["improvement_level"] = "none"


@dataclass
class LeaderboardUpdated(DomainEvent):
    """Event raised when leaderboard is updated."""
    competition_id: str
    submission_id: str
    old_rank: Optional[int]
    new_rank: int
    total_teams: int
    top_score: float
    score_gap_to_top: float
    
    def __post_init__(self):
        super().__post_init__()
        self.aggregate_id = self.competition_id
        
        # Calculate percentile
        percentile = (1 - self.new_rank / self.total_teams) * 100
        self.metadata["percentile"] = round(percentile, 2)
        
        # Categorize position
        if percentile >= 99:
            self.metadata["tier"] = "elite"
        elif percentile >= 90:
            self.metadata["tier"] = "top"
        elif percentile >= 75:
            self.metadata["tier"] = "upper"
        elif percentile >= 50:
            self.metadata["tier"] = "middle"
        else:
            self.metadata["tier"] = "lower"


@dataclass
class LeaderboardPositionChanged(DomainEvent):
    """Event raised when leaderboard position changes significantly."""
    competition_id: str
    old_rank: int
    new_rank: int
    direction: str  # "up" or "down"
    positions_changed: int
    cause: str  # "new_submission", "others_improved", "rescoring"
    
    def __post_init__(self):
        super().__post_init__()
        self.aggregate_id = self.competition_id
        
        # Categorize change significance
        if self.positions_changed >= 50:
            self.metadata["change_magnitude"] = "massive"
        elif self.positions_changed >= 20:
            self.metadata["change_magnitude"] = "large"
        elif self.positions_changed >= 5:
            self.metadata["change_magnitude"] = "moderate"
        else:
            self.metadata["change_magnitude"] = "small"