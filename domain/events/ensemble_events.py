"""Ensemble-related domain events."""

from dataclasses import dataclass
from typing import List, Dict, Optional

from .base import DomainEvent


@dataclass
class EnsembleCreated(DomainEvent):
    """Event raised when an ensemble is created."""
    ensemble_id: str
    ensemble_name: str
    method: str
    model_count: int
    base_model_ids: List[str]
    
    def __post_init__(self):
        super().__post_init__()
        self.aggregate_id = self.ensemble_id
        
        # Categorize ensemble size
        if self.model_count >= 10:
            self.metadata["size_category"] = "large"
        elif self.model_count >= 5:
            self.metadata["size_category"] = "medium"
        else:
            self.metadata["size_category"] = "small"


@dataclass
class EnsembleOptimized(DomainEvent):
    """Event raised when ensemble weights are optimized."""
    ensemble_id: str
    optimization_method: str
    old_score: float
    new_score: float
    improvement: float
    weights: Dict[str, float]
    optimization_duration_seconds: float
    
    def __post_init__(self):
        super().__post_init__()
        self.aggregate_id = self.ensemble_id
        
        # Calculate improvement percentage
        if self.old_score > 0:
            improvement_pct = (self.improvement / self.old_score) * 100
            self.metadata["improvement_percentage"] = round(improvement_pct, 2)
        
        # Categorize optimization success
        if self.improvement > 0.01:
            self.metadata["optimization_result"] = "significant_improvement"
        elif self.improvement > 0:
            self.metadata["optimization_result"] = "minor_improvement"
        else:
            self.metadata["optimization_result"] = "no_improvement"


@dataclass
class ModelAddedToEnsemble(DomainEvent):
    """Event raised when a model is added to ensemble."""
    ensemble_id: str
    model_id: str
    model_score: float
    ensemble_size_before: int
    ensemble_size_after: int
    expected_improvement: Optional[float] = None
    
    def __post_init__(self):
        super().__post_init__()
        self.aggregate_id = self.ensemble_id


@dataclass
class ModelRemovedFromEnsemble(DomainEvent):
    """Event raised when a model is removed from ensemble."""
    ensemble_id: str
    model_id: str
    removal_reason: str  # "underperforming", "redundant", "manual"
    contribution_score: float
    ensemble_size_before: int
    ensemble_size_after: int
    
    def __post_init__(self):
        super().__post_init__()
        self.aggregate_id = self.ensemble_id
        
        # Validate ensemble size
        if self.ensemble_size_after < 2:
            self.metadata["warning"] = "ensemble_too_small"


@dataclass
class EnsemblePerformanceImproved(DomainEvent):
    """Event raised when ensemble performance improves significantly."""
    ensemble_id: str
    old_best_score: float
    new_best_score: float
    improvement: float
    trigger: str  # "optimization", "model_addition", "model_removal", "method_change"
    leaderboard_impact: Optional[str] = None  # "rank_improved", "top_10", etc.
    
    def __post_init__(self):
        super().__post_init__()
        self.aggregate_id = self.ensemble_id
        
        # Calculate improvement percentage
        if self.old_best_score > 0:
            improvement_pct = (self.improvement / self.old_best_score) * 100
            self.metadata["improvement_percentage"] = round(improvement_pct, 2)
            
            # Categorize improvement magnitude
            if improvement_pct >= 5:
                self.metadata["magnitude"] = "breakthrough"
            elif improvement_pct >= 2:
                self.metadata["magnitude"] = "major"
            elif improvement_pct >= 1:
                self.metadata["magnitude"] = "moderate"
            else:
                self.metadata["magnitude"] = "minor"