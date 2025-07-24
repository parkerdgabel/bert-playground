"""Competition entity and related domain objects.

This module contains the core competition abstractions for Kaggle and similar
ML competition platforms. These are pure domain objects with no external dependencies.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional, Dict, Any, List


class Platform(Enum):
    """Competition platform types."""
    KAGGLE = "kaggle"
    DRIVENDATA = "drivendata"
    AICROWD = "aicrowd"
    ZINDI = "zindi"
    CODALAB = "codalab"
    CUSTOM = "custom"


class CompetitionType(Enum):
    """Types of ML competitions."""
    CLASSIFICATION = "classification"
    REGRESSION = "regression"
    MULTI_CLASS = "multi_class"
    MULTI_LABEL = "multi_label"
    RANKING = "ranking"
    SEGMENTATION = "segmentation"
    OBJECT_DETECTION = "object_detection"
    NLP = "nlp"
    TIME_SERIES = "time_series"
    RECOMMENDATION = "recommendation"
    CUSTOM = "custom"


class OptimizationDirection(Enum):
    """Direction for metric optimization."""
    MINIMIZE = "minimize"
    MAXIMIZE = "maximize"


class SubmissionFileFormat(Enum):
    """Supported submission file formats."""
    CSV = "csv"
    JSON = "json"
    PARQUET = "parquet"
    NPY = "npy"
    CUSTOM = "custom"


@dataclass(frozen=True)
class CompetitionId:
    """Value object for competition identifier."""
    value: str
    
    def __post_init__(self):
        if not self.value or not isinstance(self.value, str):
            raise ValueError("Competition ID must be a non-empty string")


@dataclass(frozen=True)
class EvaluationCriteria:
    """Competition evaluation criteria.
    
    Defines how submissions are scored and ranked.
    """
    metric_name: str
    metric_type: str  # e.g., "log_loss", "rmse", "auc", "map@k"
    direction: OptimizationDirection
    precision: int = 5  # decimal places for scoring
    
    @property
    def is_better(self) -> callable:
        """Returns a function to compare two scores."""
        if self.direction == OptimizationDirection.MINIMIZE:
            return lambda a, b: a < b
        else:
            return lambda a, b: a > b
    
    def format_score(self, score: float) -> str:
        """Format score according to competition precision."""
        return f"{score:.{self.precision}f}"


@dataclass(frozen=True)
class SubmissionRules:
    """Rules and constraints for competition submissions."""
    file_format: SubmissionFileFormat
    max_file_size_mb: float
    required_columns: List[str]
    submission_id_column: str
    prediction_columns: List[str]
    max_daily_submissions: int
    max_total_submissions: Optional[int] = None
    allows_team_merging: bool = True
    allows_gpu: bool = True
    time_limit_seconds: Optional[int] = None
    
    def validate_columns(self, columns: List[str]) -> bool:
        """Check if submission has required columns."""
        return all(col in columns for col in self.required_columns)


@dataclass
class CompetitionTimeline:
    """Competition timeline and important dates."""
    start_date: datetime
    end_date: datetime
    final_submission_deadline: datetime
    team_merger_deadline: Optional[datetime] = None
    
    @property
    def is_active(self) -> bool:
        """Check if competition is currently active."""
        now = datetime.now()
        return self.start_date <= now <= self.end_date
    
    @property
    def days_remaining(self) -> int:
        """Days until competition ends."""
        if not self.is_active:
            return 0
        return (self.end_date - datetime.now()).days
    
    @property
    def allows_late_submission(self) -> bool:
        """Check if late submissions are allowed."""
        now = datetime.now()
        return now <= self.final_submission_deadline


@dataclass
class DatasetInfo:
    """Information about competition dataset."""
    train_size: int
    test_size: int
    feature_count: int
    target_column: Optional[str]
    sample_submission_available: bool
    external_data_allowed: bool
    data_description: str
    file_formats: List[str] = field(default_factory=list)
    total_size_mb: Optional[float] = None
    
    @property
    def total_samples(self) -> int:
        """Total number of samples across train and test."""
        return self.train_size + self.test_size


@dataclass
class LeaderboardInfo:
    """Leaderboard configuration and state."""
    public_percentage: float  # e.g., 0.3 for 30% public
    private_percentage: float  # e.g., 0.7 for 70% private
    show_public_scores: bool = True
    show_team_names: bool = True
    update_frequency: str = "immediate"  # immediate, daily, weekly
    
    @property
    def has_private_leaderboard(self) -> bool:
        """Check if competition has private leaderboard."""
        return self.private_percentage > 0


@dataclass
class PrizeInfo:
    """Competition prizes and incentives."""
    total_prize_pool: float
    currency: str = "USD"
    prize_distribution: Dict[int, float] = field(default_factory=dict)  # rank -> amount
    has_gpu_credits: bool = False
    has_cloud_credits: bool = False
    other_prizes: List[str] = field(default_factory=list)


@dataclass
class Competition:
    """Competition aggregate root.
    
    Represents a complete ML competition with all its rules,
    timelines, and evaluation criteria.
    """
    id: CompetitionId
    name: str
    platform: Platform
    competition_type: CompetitionType
    evaluation: EvaluationCriteria
    timeline: CompetitionTimeline
    rules: SubmissionRules
    dataset_info: DatasetInfo
    leaderboard: LeaderboardInfo
    description: str
    prize_info: Optional[PrizeInfo] = None
    tags: List[str] = field(default_factory=list)
    difficulty_level: Optional[str] = None  # beginner, intermediate, expert
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate competition consistency."""
        if self.timeline.end_date <= self.timeline.start_date:
            raise ValueError("Competition end date must be after start date")
        
        if self.timeline.final_submission_deadline < self.timeline.end_date:
            raise ValueError("Final submission deadline cannot be before competition end")
        
        total_percentage = self.leaderboard.public_percentage + self.leaderboard.private_percentage
        if abs(total_percentage - 1.0) > 0.001:
            raise ValueError("Public and private leaderboard percentages must sum to 1.0")
    
    @property
    def is_nlp_competition(self) -> bool:
        """Check if this is an NLP competition suitable for BERT."""
        return (
            self.competition_type == CompetitionType.NLP or
            "nlp" in self.tags or
            "text" in self.tags or
            "bert" in self.tags
        )
    
    @property
    def allows_ensembling(self) -> bool:
        """Check if competition rules allow model ensembling."""
        # Most competitions allow ensembling unless explicitly forbidden
        return self.metadata.get("allows_ensembling", True)
    
    @property
    def requires_code_submission(self) -> bool:
        """Check if competition requires code submission."""
        return self.metadata.get("code_competition", False)
    
    def get_optimal_validation_strategy(self) -> str:
        """Suggest validation strategy based on competition type."""
        if self.competition_type == CompetitionType.TIME_SERIES:
            return "time_series_split"
        elif self.leaderboard.private_percentage > 0.5:
            # High private percentage suggests need for robust validation
            return "stratified_kfold_5"
        else:
            return "stratified_kfold_3"
    
    def estimate_submission_budget(self) -> int:
        """Estimate total submissions available."""
        days_remaining = self.timeline.days_remaining
        daily_limit = self.rules.max_daily_submissions
        total_limit = self.rules.max_total_submissions
        
        estimated = days_remaining * daily_limit
        
        if total_limit:
            return min(estimated, total_limit)
        return estimated


@dataclass
class CompetitionSnapshot:
    """Point-in-time snapshot of competition state."""
    competition_id: CompetitionId
    timestamp: datetime
    days_elapsed: int
    days_remaining: int
    total_teams: int
    total_submissions: int
    top_public_score: float
    median_public_score: float
    my_best_score: Optional[float] = None
    my_rank: Optional[int] = None
    my_submissions_used: int = 0