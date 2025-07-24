"""Experiment entity and related domain objects.

This module contains experiment tracking abstractions for managing
hypotheses, approaches, and results in ML competitions.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional, Dict, Any, List, Set
from uuid import uuid4

from .competition import CompetitionId


class ExperimentStatus(Enum):
    """Status of an experiment."""
    PLANNED = "planned"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    ABANDONED = "abandoned"


class ExperimentApproach(Enum):
    """High-level approach for the experiment."""
    BASELINE = "baseline"
    FEATURE_ENGINEERING = "feature_engineering"
    MODEL_ARCHITECTURE = "model_architecture"
    HYPERPARAMETER_TUNING = "hyperparameter_tuning"
    ENSEMBLE = "ensemble"
    DATA_AUGMENTATION = "data_augmentation"
    TRANSFER_LEARNING = "transfer_learning"
    PSEUDO_LABELING = "pseudo_labeling"
    ADVERSARIAL_TRAINING = "adversarial_training"
    CUSTOM = "custom"


@dataclass(frozen=True)
class ExperimentId:
    """Value object for experiment identifier."""
    value: str
    
    def __post_init__(self):
        if not self.value or not isinstance(self.value, str):
            raise ValueError("Experiment ID must be a non-empty string")
    
    @classmethod
    def generate(cls) -> "ExperimentId":
        """Generate a new experiment ID."""
        return cls(f"exp_{uuid4().hex[:12]}")


@dataclass(frozen=True)
class Hypothesis:
    """Scientific hypothesis for the experiment.
    
    Represents what we expect to achieve and why.
    """
    description: str
    rationale: str
    expected_improvement: float  # Expected score improvement
    baseline_score: Optional[float] = None
    confidence_level: float = 0.5  # 0-1, how confident we are
    assumptions: List[str] = field(default_factory=list)
    risks: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        if not 0 <= self.confidence_level <= 1:
            raise ValueError("Confidence level must be between 0 and 1")
        
        if self.expected_improvement < 0:
            raise ValueError("Expected improvement should be non-negative")
    
    @property
    def target_score(self) -> Optional[float]:
        """Calculate target score if baseline is known."""
        if self.baseline_score is not None:
            return self.baseline_score + self.expected_improvement
        return None


@dataclass
class ExperimentConfig:
    """Configuration for running the experiment."""
    model_type: str  # e.g., "bert-base", "modernbert-large"
    preprocessing_steps: List[str]
    feature_engineering: Dict[str, Any]
    hyperparameters: Dict[str, Any]
    validation_strategy: str  # e.g., "5fold", "time_series_split"
    augmentation_config: Optional[Dict[str, Any]] = None
    ensemble_config: Optional[Dict[str, Any]] = None
    resource_limits: Dict[str, Any] = field(default_factory=dict)
    
    def get_total_folds(self) -> int:
        """Extract number of folds from validation strategy."""
        if "fold" in self.validation_strategy:
            # Extract number from strings like "5fold", "10fold"
            import re
            match = re.match(r"(\d+)fold", self.validation_strategy)
            if match:
                return int(match.group(1))
        return 1  # Single train/val split


@dataclass
class ExperimentMetrics:
    """Metrics collected during experiment."""
    train_scores: List[float]
    validation_scores: List[float]
    test_scores: List[float] = field(default_factory=list)
    cv_mean: Optional[float] = None
    cv_std: Optional[float] = None
    best_fold: Optional[int] = None
    worst_fold: Optional[int] = None
    oof_score: Optional[float] = None  # Out-of-fold score
    
    @property
    def validation_mean(self) -> float:
        """Mean validation score across folds."""
        if self.validation_scores:
            return sum(self.validation_scores) / len(self.validation_scores)
        return 0.0
    
    @property
    def validation_std(self) -> float:
        """Standard deviation of validation scores."""
        if len(self.validation_scores) < 2:
            return 0.0
        mean = self.validation_mean
        variance = sum((x - mean) ** 2 for x in self.validation_scores) / len(self.validation_scores)
        return variance ** 0.5
    
    @property
    def is_stable(self) -> bool:
        """Check if CV scores are stable (low variance)."""
        if self.cv_std is None:
            return True
        # Consider stable if CV std is less than 1% of mean
        return self.cv_std < (self.cv_mean * 0.01) if self.cv_mean else True


@dataclass
class ExperimentArtifacts:
    """Artifacts produced by the experiment."""
    model_paths: List[str] = field(default_factory=list)
    prediction_paths: List[str] = field(default_factory=list)
    feature_importance_path: Optional[str] = None
    oof_predictions_path: Optional[str] = None
    test_predictions_path: Optional[str] = None
    visualizations: List[str] = field(default_factory=list)
    logs_path: Optional[str] = None
    config_path: Optional[str] = None
    
    @property
    def has_models(self) -> bool:
        """Check if experiment produced models."""
        return len(self.model_paths) > 0
    
    @property
    def has_predictions(self) -> bool:
        """Check if experiment produced predictions."""
        return len(self.prediction_paths) > 0 or self.test_predictions_path is not None


@dataclass
class ExperimentInsights:
    """Insights and learnings from the experiment."""
    key_findings: List[str]
    surprising_results: List[str] = field(default_factory=list)
    next_steps: List[str] = field(default_factory=list)
    failed_attempts: List[str] = field(default_factory=list)
    performance_bottlenecks: List[str] = field(default_factory=list)
    
    def add_finding(self, finding: str, is_surprising: bool = False):
        """Add a new finding to the experiment."""
        if is_surprising:
            self.surprising_results.append(finding)
        else:
            self.key_findings.append(finding)


@dataclass
class ExperimentResults:
    """Complete results of an experiment."""
    metrics: ExperimentMetrics
    artifacts: ExperimentArtifacts
    insights: ExperimentInsights
    execution_time_hours: float
    peak_memory_gb: float
    submission_scores: Dict[str, float] = field(default_factory=dict)  # submission_id -> score
    
    @property
    def is_successful(self) -> bool:
        """Check if experiment was successful."""
        return (
            self.artifacts.has_models and
            self.metrics.validation_mean > 0 and
            len(self.insights.key_findings) > 0
        )
    
    @property
    def efficiency_score(self) -> float:
        """Calculate efficiency (performance per hour)."""
        if self.execution_time_hours > 0:
            return self.metrics.validation_mean / self.execution_time_hours
        return 0.0


@dataclass
class Experiment:
    """Experiment aggregate root.
    
    Represents a complete ML experiment with hypothesis,
    approach, configuration, and results.
    """
    id: ExperimentId
    competition_id: CompetitionId
    name: str
    hypothesis: Hypothesis
    approach: ExperimentApproach
    config: ExperimentConfig
    status: ExperimentStatus
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    results: Optional[ExperimentResults] = None
    parent_experiment_id: Optional[ExperimentId] = None  # For iterative experiments
    tags: Set[str] = field(default_factory=set)
    notes: str = ""
    
    def __post_init__(self):
        """Initialize experiment state."""
        if self.started_at and self.created_at and self.started_at < self.created_at:
            raise ValueError("Experiment cannot start before it was created")
        
        if self.completed_at and self.started_at and self.completed_at < self.started_at:
            raise ValueError("Experiment cannot complete before it started")
    
    def start(self) -> None:
        """Mark experiment as started."""
        if self.status != ExperimentStatus.PLANNED:
            raise ValueError(f"Cannot start experiment in {self.status} status")
        self.status = ExperimentStatus.IN_PROGRESS
        self.started_at = datetime.now()
    
    def complete(self, results: ExperimentResults) -> None:
        """Mark experiment as completed with results."""
        if self.status != ExperimentStatus.IN_PROGRESS:
            raise ValueError(f"Cannot complete experiment in {self.status} status")
        self.status = ExperimentStatus.COMPLETED
        self.completed_at = datetime.now()
        self.results = results
    
    def fail(self, reason: str) -> None:
        """Mark experiment as failed."""
        self.status = ExperimentStatus.FAILED
        self.completed_at = datetime.now()
        self.notes += f"\nFailed: {reason}"
    
    def abandon(self, reason: str) -> None:
        """Mark experiment as abandoned."""
        self.status = ExperimentStatus.ABANDONED
        self.completed_at = datetime.now()
        self.notes += f"\nAbandoned: {reason}"
    
    @property
    def duration_hours(self) -> Optional[float]:
        """Calculate experiment duration in hours."""
        if self.started_at and self.completed_at:
            delta = self.completed_at - self.started_at
            return delta.total_seconds() / 3600
        return None
    
    @property
    def hypothesis_validated(self) -> Optional[bool]:
        """Check if hypothesis was validated."""
        if not self.results or not self.hypothesis.target_score:
            return None
        
        actual_improvement = self.results.metrics.validation_mean - self.hypothesis.baseline_score
        return actual_improvement >= self.hypothesis.expected_improvement
    
    def get_lineage(self) -> List[ExperimentId]:
        """Get experiment lineage (parent chain)."""
        lineage = [self.id]
        if self.parent_experiment_id:
            lineage.append(self.parent_experiment_id)
            # In practice, would recursively fetch parent experiments
        return lineage
    
    def suggest_next_experiment(self) -> Optional[str]:
        """Suggest next experiment based on results."""
        if not self.results:
            return None
        
        if self.results.insights.next_steps:
            return self.results.insights.next_steps[0]
        
        # Basic heuristics
        if self.results.metrics.validation_std > 0.05:
            return "Try more stable validation strategy or increase folds"
        
        if self.approach == ExperimentApproach.BASELINE:
            return "Try feature engineering or model architecture improvements"
        
        return None


@dataclass
class ExperimentComparison:
    """Comparison between multiple experiments."""
    experiment_ids: List[ExperimentId]
    best_experiment_id: ExperimentId
    metric_comparison: Dict[ExperimentId, float]
    improvement_over_baseline: Dict[ExperimentId, float]
    insights: List[str]
    
    @property
    def best_score(self) -> float:
        """Get best score across experiments."""
        return max(self.metric_comparison.values())
    
    @property
    def score_range(self) -> float:
        """Get range of scores."""
        scores = list(self.metric_comparison.values())
        return max(scores) - min(scores) if scores else 0.0