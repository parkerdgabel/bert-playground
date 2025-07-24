"""Experiment-related domain events."""

from dataclasses import dataclass
from typing import Optional, List, Dict, Any

from .base import DomainEvent


@dataclass
class ExperimentStarted(DomainEvent):
    """Event raised when an experiment starts."""
    experiment_id: str
    experiment_name: str
    competition_id: str
    hypothesis: str
    approach: str
    estimated_duration_hours: float
    
    def __post_init__(self):
        super().__post_init__()
        self.aggregate_id = self.experiment_id


@dataclass
class ExperimentCompleted(DomainEvent):
    """Event raised when an experiment completes successfully."""
    experiment_id: str
    competition_id: str
    duration_hours: float
    validation_score: float
    improvement_over_baseline: Optional[float] = None
    models_produced: int = 0
    
    def __post_init__(self):
        super().__post_init__()
        self.aggregate_id = self.experiment_id
        
        # Add performance level to metadata
        if self.improvement_over_baseline:
            if self.improvement_over_baseline > 0.05:
                self.metadata["impact"] = "breakthrough"
            elif self.improvement_over_baseline > 0.02:
                self.metadata["impact"] = "significant"
            elif self.improvement_over_baseline > 0:
                self.metadata["impact"] = "incremental"
            else:
                self.metadata["impact"] = "no_improvement"


@dataclass
class ExperimentFailed(DomainEvent):
    """Event raised when an experiment fails."""
    experiment_id: str
    competition_id: str
    failure_reason: str
    failure_stage: str  # "training", "validation", "prediction"
    can_retry: bool = True
    suggested_fixes: List[str] = None
    
    def __post_init__(self):
        super().__post_init__()
        self.aggregate_id = self.experiment_id
        if self.suggested_fixes is None:
            self.suggested_fixes = []


@dataclass
class HypothesisValidated(DomainEvent):
    """Event raised when a hypothesis is validated."""
    experiment_id: str
    hypothesis: str
    expected_improvement: float
    actual_improvement: float
    confidence_level: float
    
    def __post_init__(self):
        super().__post_init__()
        self.aggregate_id = self.experiment_id
        
        # Categorize validation result
        improvement_ratio = self.actual_improvement / self.expected_improvement if self.expected_improvement > 0 else 0
        
        if improvement_ratio >= 1.0:
            self.metadata["validation_result"] = "exceeded_expectations"
        elif improvement_ratio >= 0.8:
            self.metadata["validation_result"] = "met_expectations"
        elif improvement_ratio >= 0.5:
            self.metadata["validation_result"] = "partially_validated"
        else:
            self.metadata["validation_result"] = "not_validated"


@dataclass
class InsightDiscovered(DomainEvent):
    """Event raised when a significant insight is discovered."""
    experiment_id: str
    insight: str
    evidence: str
    impact_level: str  # "high", "medium", "low"
    actionable: bool
    recommended_actions: List[str] = None
    
    def __post_init__(self):
        super().__post_init__()
        self.aggregate_id = self.experiment_id
        if self.recommended_actions is None:
            self.recommended_actions = []
        
        # Tag insights by type
        insight_lower = self.insight.lower()
        if "overfit" in insight_lower:
            self.metadata["insight_type"] = "overfitting"
        elif "feature" in insight_lower:
            self.metadata["insight_type"] = "feature_importance"
        elif "data" in insight_lower:
            self.metadata["insight_type"] = "data_quality"
        elif "model" in insight_lower:
            self.metadata["insight_type"] = "model_behavior"
        else:
            self.metadata["insight_type"] = "general"