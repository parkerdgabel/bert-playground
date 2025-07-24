"""Ensemble entity and related domain objects.

This module contains ensemble abstractions for combining multiple
models to improve competition performance.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional, Dict, Any, List, Tuple, Set
from uuid import uuid4
import math

from .model import BertModel
from .experiment import ExperimentId


class EnsembleMethod(Enum):
    """Methods for combining model predictions."""
    SIMPLE_AVERAGE = "simple_average"
    WEIGHTED_AVERAGE = "weighted_average"
    VOTING = "voting"
    RANK_AVERAGE = "rank_average"
    STACKING = "stacking"
    BLENDING = "blending"
    BAYESIAN_OPTIMIZATION = "bayesian_optimization"
    GEOMETRIC_MEAN = "geometric_mean"
    HARMONIC_MEAN = "harmonic_mean"
    POWER_AVERAGE = "power_average"


class VotingType(Enum):
    """Type of voting for ensemble."""
    HARD = "hard"  # Use predicted classes
    SOFT = "soft"  # Use predicted probabilities


@dataclass(frozen=True)
class EnsembleId:
    """Value object for ensemble identifier."""
    value: str
    
    def __post_init__(self):
        if not self.value or not isinstance(self.value, str):
            raise ValueError("Ensemble ID must be a non-empty string")
    
    @classmethod
    def generate(cls) -> "EnsembleId":
        """Generate a new ensemble ID."""
        return cls(f"ens_{uuid4().hex[:12]}")


@dataclass(frozen=True)
class ModelWeight:
    """Weight assigned to a model in ensemble."""
    model_id: str
    weight: float
    rationale: str = ""  # Why this weight was chosen
    
    def __post_init__(self):
        if self.weight < 0:
            raise ValueError("Model weight must be non-negative")
        if self.weight > 1:
            raise ValueError("Model weight should not exceed 1.0")


@dataclass
class EnsembleConfig:
    """Configuration for ensemble methods."""
    method: EnsembleMethod
    weights: Optional[List[ModelWeight]] = None
    voting_type: Optional[VotingType] = None
    power: float = 1.0  # For power averaging
    optimization_metric: Optional[str] = None
    optimization_cv_folds: int = 3
    use_rank_transform: bool = False
    clip_predictions: bool = True
    clip_range: Tuple[float, float] = (0.0, 1.0)
    
    def __post_init__(self):
        """Validate configuration."""
        if self.method == EnsembleMethod.WEIGHTED_AVERAGE and not self.weights:
            raise ValueError("Weighted average requires weights")
        
        if self.method == EnsembleMethod.VOTING and not self.voting_type:
            raise ValueError("Voting method requires voting type")
        
        if self.weights:
            total_weight = sum(w.weight for w in self.weights)
            if abs(total_weight - 1.0) > 0.001:
                raise ValueError("Weights must sum to 1.0")
    
    @property
    def requires_optimization(self) -> bool:
        """Check if ensemble requires optimization."""
        return self.method in [
            EnsembleMethod.WEIGHTED_AVERAGE,
            EnsembleMethod.BAYESIAN_OPTIMIZATION,
            EnsembleMethod.STACKING
        ]
    
    def get_weight_for_model(self, model_id: str) -> float:
        """Get weight for specific model."""
        if not self.weights:
            return 1.0  # Equal weight
        
        for weight in self.weights:
            if weight.model_id == model_id:
                return weight.weight
        return 0.0


@dataclass
class ModelContribution:
    """Analysis of a model's contribution to ensemble."""
    model_id: str
    solo_score: float
    ensemble_score_without: float  # Ensemble score without this model
    contribution: float  # How much this model improves ensemble
    correlation_with_ensemble: float
    diversity_score: float  # How different from other models
    
    @property
    def is_beneficial(self) -> bool:
        """Check if model improves ensemble."""
        return self.contribution > 0
    
    @property
    def is_redundant(self) -> bool:
        """Check if model is redundant (high correlation, low contribution)."""
        return self.correlation_with_ensemble > 0.95 and self.contribution < 0.001


@dataclass
class EnsemblePerformance:
    """Performance metrics for ensemble."""
    individual_scores: Dict[str, float]  # model_id -> score
    ensemble_score: float
    improvement_over_best: float
    improvement_over_average: float
    validation_scores: List[float] = field(default_factory=list)
    test_correlation_matrix: Optional[List[List[float]]] = None
    
    @property
    def best_individual_score(self) -> float:
        """Best score among individual models."""
        return max(self.individual_scores.values()) if self.individual_scores else 0.0
    
    @property
    def average_individual_score(self) -> float:
        """Average score of individual models."""
        if not self.individual_scores:
            return 0.0
        return sum(self.individual_scores.values()) / len(self.individual_scores)
    
    @property
    def is_effective(self) -> bool:
        """Check if ensemble is better than best individual."""
        return self.improvement_over_best > 0
    
    @property
    def diversity_index(self) -> float:
        """Calculate diversity index from correlation matrix."""
        if not self.test_correlation_matrix:
            return 0.0
        
        n = len(self.test_correlation_matrix)
        if n < 2:
            return 0.0
        
        # Average pairwise correlation (excluding diagonal)
        total_corr = 0
        count = 0
        for i in range(n):
            for j in range(i+1, n):
                total_corr += self.test_correlation_matrix[i][j]
                count += 1
        
        avg_corr = total_corr / count if count > 0 else 1.0
        # Diversity is inverse of correlation
        return 1.0 - avg_corr


@dataclass
class StackingConfig:
    """Configuration for stacking ensemble."""
    meta_model_type: str  # e.g., "linear", "lightgbm", "neural"
    use_original_features: bool = False
    use_proba_features: bool = True
    cv_predictions: bool = True  # Use CV predictions for training
    meta_features: List[str] = field(default_factory=list)
    
    @property
    def feature_count(self) -> int:
        """Total number of features for meta-model."""
        count = len(self.meta_features)
        if self.use_original_features:
            count += 100  # Placeholder for original features
        return count


@dataclass
class ModelEnsemble:
    """Model ensemble aggregate root.
    
    Represents a combination of multiple models working together
    to produce better predictions than any individual model.
    """
    id: EnsembleId
    name: str
    experiment_ids: List[ExperimentId]
    base_model_ids: List[str]
    config: EnsembleConfig
    performance: Optional[EnsemblePerformance] = None
    stacking_config: Optional[StackingConfig] = None
    created_at: datetime = field(default_factory=datetime.now)
    optimized_at: Optional[datetime] = None
    model_contributions: List[ModelContribution] = field(default_factory=list)
    tags: Set[str] = field(default_factory=set)
    notes: str = ""
    
    def __post_init__(self):
        """Validate ensemble consistency."""
        if len(self.base_model_ids) < 2:
            raise ValueError("Ensemble must have at least 2 models")
        
        if len(set(self.base_model_ids)) != len(self.base_model_ids):
            raise ValueError("Ensemble cannot have duplicate models")
    
    @property
    def size(self) -> int:
        """Number of models in ensemble."""
        return len(self.base_model_ids)
    
    @property
    def is_optimized(self) -> bool:
        """Check if ensemble has been optimized."""
        return self.optimized_at is not None
    
    @property
    def is_stacking(self) -> bool:
        """Check if this is a stacking ensemble."""
        return self.config.method == EnsembleMethod.STACKING
    
    @property
    def requires_meta_model(self) -> bool:
        """Check if ensemble requires meta-model training."""
        return self.config.method in [EnsembleMethod.STACKING, EnsembleMethod.BLENDING]
    
    def add_model(self, model_id: str, weight: Optional[float] = None):
        """Add a model to the ensemble."""
        if model_id in self.base_model_ids:
            raise ValueError(f"Model {model_id} already in ensemble")
        
        self.base_model_ids.append(model_id)
        
        if weight is not None and self.config.weights is not None:
            # Re-normalize weights
            self.config.weights.append(ModelWeight(model_id, weight))
            self._normalize_weights()
    
    def remove_model(self, model_id: str):
        """Remove a model from the ensemble."""
        if model_id not in self.base_model_ids:
            raise ValueError(f"Model {model_id} not in ensemble")
        
        if len(self.base_model_ids) <= 2:
            raise ValueError("Cannot remove model - ensemble must have at least 2 models")
        
        self.base_model_ids.remove(model_id)
        
        # Remove from weights if present
        if self.config.weights:
            self.config.weights = [w for w in self.config.weights if w.model_id != model_id]
            self._normalize_weights()
    
    def _normalize_weights(self):
        """Normalize weights to sum to 1.0."""
        if not self.config.weights:
            return
        
        total = sum(w.weight for w in self.config.weights)
        if total > 0:
            for w in self.config.weights:
                w = ModelWeight(w.model_id, w.weight / total, w.rationale)
    
    def optimize_weights(self, validation_scores: Dict[str, List[float]]):
        """Optimize ensemble weights based on validation scores."""
        if self.config.method != EnsembleMethod.WEIGHTED_AVERAGE:
            raise ValueError("Weight optimization only for weighted average")
        
        # Simple optimization: weight by performance
        model_means = {
            model_id: sum(scores) / len(scores)
            for model_id, scores in validation_scores.items()
        }
        
        # Convert to weights (higher score = higher weight)
        total_score = sum(model_means.values())
        self.config.weights = [
            ModelWeight(
                model_id,
                score / total_score,
                f"Weight based on CV score: {score:.5f}"
            )
            for model_id, score in model_means.items()
        ]
        
        self.optimized_at = datetime.now()
    
    def calculate_ensemble_prediction(self, predictions: Dict[str, List[float]]) -> List[float]:
        """Calculate ensemble predictions from individual model predictions.
        
        This is a simplified version - actual implementation would be more complex.
        """
        if not predictions:
            return []
        
        # Get number of samples from first model
        n_samples = len(next(iter(predictions.values())))
        
        if self.config.method == EnsembleMethod.SIMPLE_AVERAGE:
            # Simple average
            ensemble_preds = []
            for i in range(n_samples):
                avg = sum(preds[i] for preds in predictions.values()) / len(predictions)
                ensemble_preds.append(avg)
            return ensemble_preds
        
        elif self.config.method == EnsembleMethod.WEIGHTED_AVERAGE:
            # Weighted average
            ensemble_preds = []
            for i in range(n_samples):
                weighted_sum = sum(
                    preds[i] * self.config.get_weight_for_model(model_id)
                    for model_id, preds in predictions.items()
                )
                ensemble_preds.append(weighted_sum)
            return ensemble_preds
        
        elif self.config.method == EnsembleMethod.GEOMETRIC_MEAN:
            # Geometric mean
            ensemble_preds = []
            for i in range(n_samples):
                product = 1.0
                for preds in predictions.values():
                    product *= max(preds[i], 1e-10)  # Avoid zero
                geom_mean = product ** (1.0 / len(predictions))
                ensemble_preds.append(geom_mean)
            return ensemble_preds
        
        else:
            raise NotImplementedError(f"Method {self.config.method} not implemented")
    
    def analyze_model_contributions(self, validation_data: Dict[str, Any]) -> List[ModelContribution]:
        """Analyze each model's contribution to ensemble performance."""
        contributions = []
        
        # This would need actual implementation with validation data
        # For now, return placeholder
        for model_id in self.base_model_ids:
            contribution = ModelContribution(
                model_id=model_id,
                solo_score=0.85,  # Placeholder
                ensemble_score_without=0.88,  # Placeholder
                contribution=0.02,  # Placeholder
                correlation_with_ensemble=0.9,  # Placeholder
                diversity_score=0.7  # Placeholder
            )
            contributions.append(contribution)
        
        self.model_contributions = contributions
        return contributions
    
    def get_pruning_candidates(self, min_contribution: float = 0.001) -> List[str]:
        """Get models that could be removed without hurting performance."""
        if not self.model_contributions:
            return []
        
        candidates = [
            contrib.model_id
            for contrib in self.model_contributions
            if contrib.contribution < min_contribution or contrib.is_redundant
        ]
        
        # Never prune below minimum ensemble size
        max_removals = len(self.base_model_ids) - 2
        return candidates[:max_removals]
    
    def suggest_additions(self, available_models: List[str]) -> List[str]:
        """Suggest models to add based on diversity."""
        # Models not already in ensemble
        candidates = [m for m in available_models if m not in self.base_model_ids]
        
        # In practice, would analyze correlation with current ensemble
        # For now, return top candidates
        return candidates[:3]


@dataclass
class EnsembleBuilder:
    """Builder for creating optimized ensembles."""
    name: str
    candidate_models: List[str]
    validation_scores: Dict[str, float]
    target_size: int = 5
    diversity_threshold: float = 0.3
    min_individual_score: float = 0.0
    
    def build_greedy_ensemble(self) -> ModelEnsemble:
        """Build ensemble by greedily adding models that improve score."""
        # Start with best model
        sorted_models = sorted(
            self.candidate_models,
            key=lambda m: self.validation_scores.get(m, 0),
            reverse=True
        )
        
        selected = [sorted_models[0]]
        
        # Greedily add models
        for model in sorted_models[1:]:
            if len(selected) >= self.target_size:
                break
            
            # Check if model meets minimum score
            if self.validation_scores.get(model, 0) < self.min_individual_score:
                continue
            
            # In practice, would check if model improves ensemble
            # For now, just add if score is good
            if self.validation_scores.get(model, 0) > 0.8:  # Placeholder threshold
                selected.append(model)
        
        # Create ensemble
        ensemble = ModelEnsemble(
            id=EnsembleId.generate(),
            name=self.name,
            experiment_ids=[],  # Would be filled in practice
            base_model_ids=selected,
            config=EnsembleConfig(
                method=EnsembleMethod.SIMPLE_AVERAGE
            )
        )
        
        return ensemble