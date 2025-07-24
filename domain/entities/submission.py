"""Submission entity and related domain objects.

This module contains submission tracking abstractions for managing
competition submissions, validation, and leaderboard tracking.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional, Dict, Any, List, Tuple
from uuid import uuid4

from .competition import CompetitionId, SubmissionRules
from .experiment import ExperimentId


class SubmissionStatus(Enum):
    """Status of a submission."""
    PREPARING = "preparing"
    VALIDATING = "validating"
    VALIDATED = "validated"
    SUBMITTED = "submitted"
    SCORING = "scoring"
    SCORED = "scored"
    FAILED = "failed"
    INVALIDATED = "invalidated"


class SubmissionType(Enum):
    """Type of submission."""
    SINGLE_MODEL = "single_model"
    ENSEMBLE = "ensemble"
    STACKED = "stacked"
    BLENDED = "blended"
    PSEUDO_LABELED = "pseudo_labeled"
    POST_PROCESSED = "post_processed"


@dataclass(frozen=True)
class SubmissionId:
    """Value object for submission identifier."""
    value: str
    
    def __post_init__(self):
        if not self.value or not isinstance(self.value, str):
            raise ValueError("Submission ID must be a non-empty string")
    
    @classmethod
    def generate(cls) -> "SubmissionId":
        """Generate a new submission ID."""
        return cls(f"sub_{uuid4().hex[:12]}")


@dataclass
class PredictionData:
    """Container for submission predictions."""
    sample_ids: List[str]
    predictions: List[float]  # Can be probabilities or regression values
    confidence_scores: Optional[List[float]] = None
    prediction_ranks: Optional[List[int]] = None  # For ranking problems
    
    def __post_init__(self):
        """Validate prediction data consistency."""
        if len(self.sample_ids) != len(self.predictions):
            raise ValueError("Number of sample IDs must match number of predictions")
        
        if self.confidence_scores and len(self.confidence_scores) != len(self.predictions):
            raise ValueError("Number of confidence scores must match predictions")
    
    @property
    def size(self) -> int:
        """Number of predictions."""
        return len(self.predictions)
    
    @property
    def has_confidence(self) -> bool:
        """Check if confidence scores are available."""
        return self.confidence_scores is not None
    
    def get_prediction_for_sample(self, sample_id: str) -> Optional[float]:
        """Get prediction for a specific sample."""
        try:
            idx = self.sample_ids.index(sample_id)
            return self.predictions[idx]
        except ValueError:
            return None
    
    def get_high_confidence_predictions(self, threshold: float = 0.9) -> List[Tuple[str, float]]:
        """Get predictions with high confidence."""
        if not self.confidence_scores:
            return []
        
        return [
            (sid, pred)
            for sid, pred, conf in zip(self.sample_ids, self.predictions, self.confidence_scores)
            if conf >= threshold
        ]


@dataclass
class SubmissionValidation:
    """Validation results for a submission."""
    is_valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    validation_timestamp: datetime = field(default_factory=datetime.now)
    
    # Validation details
    format_valid: bool = True
    columns_valid: bool = True
    values_valid: bool = True
    size_valid: bool = True
    
    def add_error(self, error: str, error_type: str = "general"):
        """Add validation error."""
        self.errors.append(f"[{error_type}] {error}")
        self.is_valid = False
        
        # Update specific validation flags
        if error_type == "format":
            self.format_valid = False
        elif error_type == "columns":
            self.columns_valid = False
        elif error_type == "values":
            self.values_valid = False
        elif error_type == "size":
            self.size_valid = False
    
    def add_warning(self, warning: str):
        """Add validation warning (non-fatal)."""
        self.warnings.append(warning)
    
    @property
    def error_count(self) -> int:
        """Total number of errors."""
        return len(self.errors)
    
    @property
    def has_warnings(self) -> bool:
        """Check if there are any warnings."""
        return len(self.warnings) > 0


@dataclass
class SubmissionScores:
    """Scores for a submission."""
    public_score: Optional[float] = None
    private_score: Optional[float] = None
    scoring_timestamp: Optional[datetime] = None
    public_rank: Optional[int] = None
    private_rank: Optional[int] = None
    total_submissions: Optional[int] = None  # Total submissions on leaderboard
    
    @property
    def has_public_score(self) -> bool:
        """Check if public score is available."""
        return self.public_score is not None
    
    @property
    def has_private_score(self) -> bool:
        """Check if private score is available."""
        return self.private_score is not None
    
    @property
    def public_percentile(self) -> Optional[float]:
        """Calculate percentile rank on public leaderboard."""
        if self.public_rank and self.total_submissions:
            return 100.0 * (1 - self.public_rank / self.total_submissions)
        return None
    
    @property
    def score_gap(self) -> Optional[float]:
        """Gap between public and private scores."""
        if self.public_score is not None and self.private_score is not None:
            return abs(self.public_score - self.private_score)
        return None
    
    @property
    def is_overfitting(self) -> Optional[bool]:
        """Check if submission might be overfitting to public LB."""
        if self.score_gap is None:
            return None
        # Consider overfitting if gap > 5% of public score
        return self.score_gap > (abs(self.public_score) * 0.05)


@dataclass
class SubmissionMetadata:
    """Metadata about the submission."""
    model_names: List[str]
    ensemble_weights: Optional[List[float]] = None
    preprocessing_version: str = "1.0"
    feature_version: str = "1.0"
    post_processing: List[str] = field(default_factory=list)
    compute_resources: Dict[str, Any] = field(default_factory=dict)
    generation_time_seconds: Optional[float] = None
    notes: str = ""
    
    @property
    def is_ensemble(self) -> bool:
        """Check if this is an ensemble submission."""
        return len(self.model_names) > 1
    
    @property
    def ensemble_size(self) -> int:
        """Number of models in ensemble."""
        return len(self.model_names)
    
    def add_post_processing(self, step: str):
        """Add post-processing step."""
        self.post_processing.append(step)


@dataclass
class Submission:
    """Submission aggregate root.
    
    Represents a complete competition submission with predictions,
    validation, scores, and metadata.
    """
    id: SubmissionId
    competition_id: CompetitionId
    experiment_id: ExperimentId
    submission_type: SubmissionType
    predictions: PredictionData
    validation: Optional[SubmissionValidation] = None
    scores: SubmissionScores = field(default_factory=SubmissionScores)
    metadata: SubmissionMetadata = field(default_factory=SubmissionMetadata)
    status: SubmissionStatus = SubmissionStatus.PREPARING
    created_at: datetime = field(default_factory=datetime.now)
    submitted_at: Optional[datetime] = None
    file_path: Optional[str] = None
    file_size_mb: Optional[float] = None
    
    def validate_against_rules(self, rules: SubmissionRules) -> SubmissionValidation:
        """Validate submission against competition rules."""
        validation = SubmissionValidation(is_valid=True)
        
        # Check file size
        if self.file_size_mb and self.file_size_mb > rules.max_file_size_mb:
            validation.add_error(
                f"File size {self.file_size_mb:.2f}MB exceeds limit {rules.max_file_size_mb}MB",
                "size"
            )
        
        # Check required columns (assuming predictions have been formatted)
        # This would normally check the actual file columns
        if len(self.predictions.sample_ids) == 0:
            validation.add_error("No predictions found", "format")
        
        # Check prediction values
        if any(pred < 0 for pred in self.predictions.predictions):
            validation.add_warning("Negative predictions found - ensure this is intended")
        
        # Update submission validation
        self.validation = validation
        if validation.is_valid:
            self.status = SubmissionStatus.VALIDATED
        else:
            self.status = SubmissionStatus.FAILED
        
        return validation
    
    def mark_submitted(self) -> None:
        """Mark submission as submitted."""
        if self.status != SubmissionStatus.VALIDATED:
            raise ValueError("Can only submit validated submissions")
        
        self.status = SubmissionStatus.SUBMITTED
        self.submitted_at = datetime.now()
    
    def update_scores(self, public_score: float, public_rank: int, total_submissions: int):
        """Update submission with public leaderboard scores."""
        self.scores.public_score = public_score
        self.scores.public_rank = public_rank
        self.scores.total_submissions = total_submissions
        self.scores.scoring_timestamp = datetime.now()
        self.status = SubmissionStatus.SCORED
    
    def update_private_scores(self, private_score: float, private_rank: int):
        """Update submission with private leaderboard scores."""
        self.scores.private_score = private_score
        self.scores.private_rank = private_rank
    
    @property
    def is_successful(self) -> bool:
        """Check if submission was successful."""
        return (
            self.status == SubmissionStatus.SCORED and
            self.scores.has_public_score
        )
    
    @property
    def submission_age_days(self) -> float:
        """Days since submission was created."""
        delta = datetime.now() - self.created_at
        return delta.total_seconds() / (24 * 3600)
    
    @property
    def is_recent(self) -> bool:
        """Check if submission is recent (< 1 day old)."""
        return self.submission_age_days < 1
    
    def compare_to(self, other: "Submission") -> str:
        """Compare this submission to another."""
        if not (self.scores.has_public_score and other.scores.has_public_score):
            return "Cannot compare - missing scores"
        
        score_diff = self.scores.public_score - other.scores.public_score
        
        if abs(score_diff) < 0.0001:
            return "Submissions have equal scores"
        elif score_diff > 0:
            improvement = (score_diff / other.scores.public_score) * 100
            return f"This submission is {improvement:.2f}% better"
        else:
            degradation = (abs(score_diff) / other.scores.public_score) * 100
            return f"This submission is {degradation:.2f}% worse"


@dataclass
class SubmissionHistory:
    """Track submission history for a competition."""
    competition_id: CompetitionId
    submissions: List[Submission] = field(default_factory=list)
    
    def add_submission(self, submission: Submission):
        """Add submission to history."""
        if submission.competition_id != self.competition_id:
            raise ValueError("Submission is for different competition")
        self.submissions.append(submission)
    
    @property
    def total_submissions(self) -> int:
        """Total number of submissions."""
        return len(self.submissions)
    
    @property
    def successful_submissions(self) -> List[Submission]:
        """Get only successful submissions."""
        return [s for s in self.submissions if s.is_successful]
    
    @property
    def best_submission(self) -> Optional[Submission]:
        """Get best submission by public score."""
        scored = [s for s in self.submissions if s.scores.has_public_score]
        if not scored:
            return None
        # Assumes higher is better - would need competition context for direction
        return max(scored, key=lambda s: s.scores.public_score)
    
    @property
    def latest_submission(self) -> Optional[Submission]:
        """Get most recent submission."""
        if not self.submissions:
            return None
        return max(self.submissions, key=lambda s: s.created_at)
    
    def get_submissions_today(self) -> List[Submission]:
        """Get submissions made today."""
        today = datetime.now().date()
        return [
            s for s in self.submissions
            if s.submitted_at and s.submitted_at.date() == today
        ]
    
    def get_score_progression(self) -> List[Tuple[datetime, float]]:
        """Get score progression over time."""
        scored = [
            (s.submitted_at, s.scores.public_score)
            for s in self.submissions
            if s.submitted_at and s.scores.has_public_score
        ]
        return sorted(scored, key=lambda x: x[0])