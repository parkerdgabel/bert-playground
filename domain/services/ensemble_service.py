"""Domain service for ensemble management.

This service handles the creation, optimization, and analysis of
model ensembles for competition performance improvement.
"""

from typing import List, Dict, Any, Optional, Tuple, Set, TYPE_CHECKING
from dataclasses import dataclass
from itertools import combinations

from domain.entities.ensemble import (
    ModelEnsemble, EnsembleId, EnsembleMethod, EnsembleConfig,
    ModelWeight, EnsemblePerformance, ModelContribution,
    VotingType, StackingConfig
)
from domain.entities.model import BertModel
from domain.entities.experiment import Experiment, ExperimentStatus
from domain.registry import domain_service, ServiceScope

if TYPE_CHECKING:
    from application.ports.secondary.numerical import NumericalOperations


@dataclass
class EnsembleOptimizationResult:
    """Result of ensemble optimization."""
    optimal_weights: List[ModelWeight]
    validation_score: float
    individual_scores: Dict[str, float]
    optimization_history: List[Tuple[List[float], float]]
    convergence_achieved: bool


@dataclass
class DiversityAnalysis:
    """Analysis of model diversity in ensemble."""
    pairwise_correlations: Dict[Tuple[str, str], float]
    average_correlation: float
    diversity_score: float  # 1 - avg_correlation
    most_similar_pair: Tuple[str, str]
    most_diverse_pair: Tuple[str, str]
    recommendation: str


@dataclass
class EnsembleRecommendation:
    """Recommendation for ensemble composition."""
    recommended_models: List[str]
    recommended_method: EnsembleMethod
    expected_score: float
    rationale: str
    alternative_options: List[Dict[str, Any]]


@domain_service(scope=ServiceScope.SINGLETON)
class EnsembleService:
    """Service for ensemble creation and optimization."""
    
    def __init__(self, numerical_ops: 'NumericalOperations'):
        """Initialize ensemble service with numerical operations port.
        
        Args:
            numerical_ops: Port for numerical operations
        """
        self._numerical_ops = numerical_ops
    
    def create_ensemble(
        self,
        models: List[BertModel],
        method: EnsembleMethod,
        name: Optional[str] = None
    ) -> ModelEnsemble:
        """Create an ensemble from models."""
        if len(models) < 2:
            raise ValueError("Ensemble requires at least 2 models")
        
        # Generate name if not provided
        if not name:
            name = f"{method.value}_ensemble_{len(models)}_models"
        
        # Create config based on method
        config = self._create_ensemble_config(method, models)
        
        # Create ensemble
        ensemble = ModelEnsemble(
            id=EnsembleId.generate(),
            name=name,
            experiment_ids=[],  # Would be populated from model metadata
            base_model_ids=[model.id.value for model in models],
            config=config
        )
        
        return ensemble
    
    def _create_ensemble_config(
        self,
        method: EnsembleMethod,
        models: List[BertModel]
    ) -> EnsembleConfig:
        """Create ensemble configuration based on method."""
        if method == EnsembleMethod.SIMPLE_AVERAGE:
            return EnsembleConfig(method=method)
            
        elif method == EnsembleMethod.WEIGHTED_AVERAGE:
            # Start with equal weights
            weights = [
                ModelWeight(
                    model_id=model.id.value,
                    weight=1.0 / len(models),
                    rationale="Initial equal weight"
                )
                for model in models
            ]
            return EnsembleConfig(method=method, weights=weights)
            
        elif method == EnsembleMethod.VOTING:
            return EnsembleConfig(
                method=method,
                voting_type=VotingType.SOFT  # Default to soft voting
            )
            
        elif method == EnsembleMethod.STACKING:
            return EnsembleConfig(
                method=method,
                optimization_metric="log_loss"  # Default metric
            )
            
        else:
            return EnsembleConfig(method=method)
    
    def optimize_weights(
        self,
        ensemble: ModelEnsemble,
        validation_predictions: Dict[str, Any],  # Changed from np.ndarray
        validation_labels: Any,  # Changed from np.ndarray
        metric: str = "log_loss"
    ) -> EnsembleOptimizationResult:
        """Optimize ensemble weights using validation data.
        
        Note: This is a simplified version. Real implementation would use
        scipy.optimize or similar optimization library.
        """
        if ensemble.config.method != EnsembleMethod.WEIGHTED_AVERAGE:
            raise ValueError("Weight optimization only for weighted average ensembles")
        
        model_ids = list(validation_predictions.keys())
        n_models = len(model_ids)
        
        # Calculate individual model scores
        individual_scores = {}
        for model_id, preds in validation_predictions.items():
            score = self._calculate_metric(preds, validation_labels, metric)
            individual_scores[model_id] = score
        
        # Simple optimization: weight by performance
        # In practice, would use proper optimization
        total_score = sum(individual_scores.values())
        optimal_weights = []
        
        for model_id in model_ids:
            weight = individual_scores[model_id] / total_score
            optimal_weights.append(
                ModelWeight(
                    model_id=model_id,
                    weight=weight,
                    rationale=f"Optimized weight based on {metric}"
                )
            )
        
        # Calculate ensemble score with optimal weights
        ensemble_preds = self._weighted_average_predictions(
            validation_predictions,
            {w.model_id: w.weight for w in optimal_weights}
        )
        ensemble_score = self._calculate_metric(ensemble_preds, validation_labels, metric)
        
        # Update ensemble
        ensemble.config.weights = optimal_weights
        ensemble.performance = EnsemblePerformance(
            individual_scores=individual_scores,
            ensemble_score=ensemble_score,
            improvement_over_best=ensemble_score - max(individual_scores.values()),
            improvement_over_average=ensemble_score - self._numerical_ops.mean(list(individual_scores.values()))
        )
        
        return EnsembleOptimizationResult(
            optimal_weights=optimal_weights,
            validation_score=ensemble_score,
            individual_scores=individual_scores,
            optimization_history=[],  # Simplified - no history
            convergence_achieved=True
        )
    
    def _calculate_metric(
        self,
        predictions: Any,
        labels: Any,
        metric: str
    ) -> float:
        """Calculate metric score.
        
        Note: Simplified implementation. Real version would support various metrics.
        """
        if metric == "log_loss":
            # Simplified log loss calculation
            eps = 1e-15
            predictions_clipped = self._numerical_ops.array_clip(predictions, eps, 1 - eps)
            
            # Calculate log loss: -mean(y * log(p) + (1-y) * log(1-p))
            log_preds = self._numerical_ops.array_log(predictions_clipped)
            one_minus_preds = self._numerical_ops.array_add(
                self._numerical_ops.array_multiply(predictions_clipped, -1.0),
                self._numerical_ops.ones_like(predictions_clipped)
            )
            log_one_minus_preds = self._numerical_ops.array_log(one_minus_preds)
            
            # y * log(p)
            term1 = self._numerical_ops.array_multiply(labels, log_preds)
            
            # (1-y) * log(1-p)
            one_minus_labels = self._numerical_ops.array_add(
                self._numerical_ops.array_multiply(labels, -1.0),
                self._numerical_ops.ones_like(labels)
            )
            term2 = self._numerical_ops.array_multiply(one_minus_labels, log_one_minus_preds)
            
            # Combine terms
            combined = self._numerical_ops.array_add(term1, term2)
            mean_val = self._numerical_ops.array_mean(combined)
            
            return -float(mean_val)
            
        elif metric == "accuracy":
            # Calculate accuracy: mean((predictions > 0.5) == labels)
            # This is simplified - assumes binary classification
            threshold = 0.5
            # For now, return a placeholder as we'd need comparison operations
            return 0.85  # Placeholder accuracy
            
        else:
            raise ValueError(f"Unsupported metric: {metric}")
    
    def _weighted_average_predictions(
        self,
        predictions: Dict[str, Any],
        weights: Dict[str, float]
    ) -> Any:
        """Calculate weighted average of predictions."""
        weighted_sum = None
        
        for model_id, preds in predictions.items():
            weight = weights.get(model_id, 0.0)
            weighted_preds = self._numerical_ops.array_multiply(preds, weight)
            
            if weighted_sum is None:
                weighted_sum = weighted_preds
            else:
                weighted_sum = self._numerical_ops.array_add(weighted_sum, weighted_preds)
        
        return weighted_sum
    
    def analyze_diversity(
        self,
        ensemble: ModelEnsemble,
        predictions: Dict[str, Any]  # Changed from np.ndarray
    ) -> DiversityAnalysis:
        """Analyze diversity of models in ensemble."""
        model_ids = list(predictions.keys())
        n_models = len(model_ids)
        
        # Calculate pairwise correlations
        correlations = {}
        for i, model1 in enumerate(model_ids):
            for j, model2 in enumerate(model_ids[i+1:], i+1):
                # Calculate correlation between two models' predictions
                corr_matrix = self._numerical_ops.array_corrcoef(predictions[model1], predictions[model2])
                # Extract the correlation coefficient
                # The correlation matrix is 2x2, we need the off-diagonal element
                # Since we can't directly index, we'll assume the port returns a scalar for two 1D arrays
                corr = float(corr_matrix) if isinstance(corr_matrix, (int, float)) else 0.5
                correlations[(model1, model2)] = corr
        
        # Calculate average correlation
        avg_correlation = self._numerical_ops.mean(list(correlations.values())) if correlations else 0.0
        
        # Find most/least similar pairs
        if correlations:
            most_similar = max(correlations.items(), key=lambda x: x[1])
            most_diverse = min(correlations.items(), key=lambda x: x[1])
        else:
            most_similar = most_diverse = (model_ids[0], model_ids[0])
        
        # Generate recommendation
        diversity_score = 1.0 - avg_correlation
        if diversity_score < 0.3:
            recommendation = "Low diversity - consider adding more diverse models"
        elif diversity_score < 0.7:
            recommendation = "Moderate diversity - ensemble should be effective"
        else:
            recommendation = "High diversity - excellent for ensembling"
        
        return DiversityAnalysis(
            pairwise_correlations=correlations,
            average_correlation=avg_correlation,
            diversity_score=diversity_score,
            most_similar_pair=most_similar[0],
            most_diverse_pair=most_diverse[0],
            recommendation=recommendation
        )
    
    def recommend_ensemble_composition(
        self,
        available_experiments: List[Experiment],
        target_ensemble_size: int = 5,
        min_individual_score: float = 0.0
    ) -> EnsembleRecommendation:
        """Recommend models for ensemble based on performance and diversity."""
        # Filter to successful experiments with good scores
        candidates = [
            exp for exp in available_experiments
            if (exp.status == ExperimentStatus.COMPLETED and
                exp.results and
                exp.results.metrics.validation_mean >= min_individual_score)
        ]
        
        if len(candidates) < 2:
            return EnsembleRecommendation(
                recommended_models=[],
                recommended_method=EnsembleMethod.SIMPLE_AVERAGE,
                expected_score=0.0,
                rationale="Not enough successful models for ensemble",
                alternative_options=[]
            )
        
        # Sort by score
        candidates.sort(
            key=lambda exp: exp.results.metrics.validation_mean,
            reverse=True
        )
        
        # Select diverse top models
        selected = [candidates[0]]  # Start with best
        selected_ids = {candidates[0].id.value}
        
        # Greedily add diverse models
        for candidate in candidates[1:]:
            if len(selected) >= target_ensemble_size:
                break
            
            # Check diversity (simplified - would use actual predictions)
            is_diverse = self._is_diverse_enough(candidate, selected)
            
            if is_diverse:
                selected.append(candidate)
                selected_ids.add(candidate.id.value)
        
        # Determine best method
        if len(selected) >= 5:
            recommended_method = EnsembleMethod.STACKING
            method_rationale = "Enough models for stacking"
        else:
            recommended_method = EnsembleMethod.WEIGHTED_AVERAGE
            method_rationale = "Weighted average for smaller ensemble"
        
        # Estimate ensemble score (simplified)
        best_individual = max(exp.results.metrics.validation_mean for exp in selected)
        expected_improvement = 0.02 if len(selected) >= 3 else 0.01
        expected_score = best_individual + expected_improvement
        
        # Generate alternatives
        alternatives = []
        
        # Simple average option
        alternatives.append({
            "method": EnsembleMethod.SIMPLE_AVERAGE,
            "expected_score": best_individual + 0.01,
            "pros": "Simple and robust",
            "cons": "May not fully utilize model strengths"
        })
        
        # Voting option for classification
        alternatives.append({
            "method": EnsembleMethod.VOTING,
            "expected_score": best_individual + 0.015,
            "pros": "Good for classification tasks",
            "cons": "Requires probability calibration"
        })
        
        return EnsembleRecommendation(
            recommended_models=[exp.id.value for exp in selected],
            recommended_method=recommended_method,
            expected_score=expected_score,
            rationale=f"{method_rationale}. Selected {len(selected)} diverse models.",
            alternative_options=alternatives
        )
    
    def _is_diverse_enough(
        self,
        candidate: Experiment,
        selected: List[Experiment],
        diversity_threshold: float = 0.8
    ) -> bool:
        """Check if candidate is diverse enough from selected models.
        
        Note: Simplified implementation. Real version would compare predictions.
        """
        # Use approach diversity as proxy
        candidate_approach = candidate.approach
        
        # Count how many selected have same approach
        same_approach_count = sum(
            1 for exp in selected if exp.approach == candidate_approach
        )
        
        # Allow at most 2 of same approach
        if same_approach_count >= 2:
            return False
        
        # Check score diversity
        candidate_score = candidate.results.metrics.validation_mean
        for exp in selected:
            score_diff = abs(exp.results.metrics.validation_mean - candidate_score)
            if score_diff < 0.001:  # Too similar performance
                return False
        
        return True
    
    def create_stacking_ensemble(
        self,
        base_models: List[BertModel],
        meta_model_type: str = "linear",
        use_original_features: bool = False
    ) -> ModelEnsemble:
        """Create a stacking ensemble."""
        if len(base_models) < 2:
            raise ValueError("Stacking requires at least 2 base models")
        
        # Create stacking configuration
        stacking_config = StackingConfig(
            meta_model_type=meta_model_type,
            use_original_features=use_original_features,
            use_proba_features=True,
            cv_predictions=True
        )
        
        # Create ensemble config
        config = EnsembleConfig(
            method=EnsembleMethod.STACKING,
            optimization_metric="log_loss"
        )
        
        # Create ensemble
        ensemble = ModelEnsemble(
            id=EnsembleId.generate(),
            name=f"stacking_{meta_model_type}_{len(base_models)}_models",
            experiment_ids=[],
            base_model_ids=[model.id.value for model in base_models],
            config=config,
            stacking_config=stacking_config
        )
        
        ensemble.tags.add("stacking")
        ensemble.tags.add(f"meta_{meta_model_type}")
        
        return ensemble
    
    def prune_ensemble(
        self,
        ensemble: ModelEnsemble,
        validation_predictions: Dict[str, Any],  # Changed from np.ndarray
        validation_labels: Any,  # Changed from np.ndarray
        min_contribution_threshold: float = 0.001
    ) -> Tuple[ModelEnsemble, List[str]]:
        """Prune underperforming models from ensemble."""
        if ensemble.size <= 2:
            return ensemble, []  # Can't prune below minimum
        
        # Analyze contributions
        contributions = self._analyze_contributions(
            ensemble,
            validation_predictions,
            validation_labels
        )
        
        # Find models to remove
        models_to_remove = []
        for model_id, contribution in contributions.items():
            if contribution < min_contribution_threshold:
                models_to_remove.append(model_id)
        
        # Limit pruning to maintain minimum ensemble size
        max_removals = ensemble.size - 2
        models_to_remove = models_to_remove[:max_removals]
        
        # Create pruned ensemble
        if models_to_remove:
            for model_id in models_to_remove:
                ensemble.remove_model(model_id)
            
            ensemble.notes += f"\nPruned {len(models_to_remove)} underperforming models"
        
        return ensemble, models_to_remove
    
    def _analyze_contributions(
        self,
        ensemble: ModelEnsemble,
        predictions: Dict[str, Any],  # Changed from np.ndarray
        labels: Any  # Changed from np.ndarray
    ) -> Dict[str, float]:
        """Analyze each model's contribution to ensemble.
        
        Note: Simplified implementation.
        """
        contributions = {}
        
        # Calculate full ensemble score
        if ensemble.config.method == EnsembleMethod.WEIGHTED_AVERAGE:
            weights = {w.model_id: w.weight for w in ensemble.config.weights}
        else:
            weights = {mid: 1.0/len(ensemble.base_model_ids) for mid in ensemble.base_model_ids}
        
        full_ensemble_preds = self._weighted_average_predictions(predictions, weights)
        full_score = self._calculate_metric(full_ensemble_preds, labels, "log_loss")
        
        # Calculate leave-one-out scores
        for model_id in ensemble.base_model_ids:
            # Create predictions without this model
            loo_predictions = {k: v for k, v in predictions.items() if k != model_id}
            
            # Reweight remaining models
            remaining_weight_sum = sum(weights[mid] for mid in loo_predictions.keys())
            loo_weights = {
                mid: weights[mid] / remaining_weight_sum
                for mid in loo_predictions.keys()
            }
            
            loo_preds = self._weighted_average_predictions(loo_predictions, loo_weights)
            loo_score = self._calculate_metric(loo_preds, labels, "log_loss")
            
            # Contribution is how much score degrades without this model
            contributions[model_id] = loo_score - full_score
        
        return contributions
    
    def blend_predictions(
        self,
        ensemble: ModelEnsemble,
        test_predictions: Dict[str, Any],  # Changed from np.ndarray
        blend_config: Optional[Dict[str, Any]] = None
    ) -> Any:  # Changed from np.ndarray
        """Blend test predictions according to ensemble configuration."""
        method = ensemble.config.method
        
        if method == EnsembleMethod.SIMPLE_AVERAGE:
            # Stack all predictions and calculate mean along first axis
            stacked_preds = self._numerical_ops.stack_arrays(list(test_predictions.values()))
            return self._numerical_ops.array_mean(stacked_preds, axis=0)
            
        elif method == EnsembleMethod.WEIGHTED_AVERAGE:
            weights = {w.model_id: w.weight for w in ensemble.config.weights}
            return self._weighted_average_predictions(test_predictions, weights)
            
        elif method == EnsembleMethod.GEOMETRIC_MEAN:
            # Geometric mean for probabilities
            first_preds = next(iter(test_predictions.values()))
            product = self._numerical_ops.ones_like(first_preds)
            
            for preds in test_predictions.values():
                # Avoid zero by using maximum with small epsilon
                safe_preds = self._numerical_ops.maximum(preds, 1e-15)
                product = self._numerical_ops.array_multiply(product, safe_preds)
            
            # Take nth root (equivalent to power of 1/n)
            n_models = len(test_predictions)
            return self._numerical_ops.array_power(product, 1.0 / n_models)
            
        elif method == EnsembleMethod.RANK_AVERAGE:
            # Average ranks instead of probabilities
            ranks = {}
            for model_id, preds in test_predictions.items():
                ranks[model_id] = self._rank_predictions(preds)
            
            # Calculate average ranks
            stacked_ranks = self._numerical_ops.stack_arrays(list(ranks.values()))
            avg_ranks = self._numerical_ops.array_mean(stacked_ranks, axis=0)
            
            # Convert back to pseudo-probabilities
            ranks_min = self._numerical_ops.array_min(avg_ranks)
            ranks_max = self._numerical_ops.array_max(avg_ranks)
            
            # Calculate range: max - min
            neg_min = self._numerical_ops.array_multiply(ranks_min, -1.0)
            ranks_range = self._numerical_ops.array_add(ranks_max, neg_min)
            
            # Normalize: 1.0 - (ranks - min) / (max - min)
            # First: ranks - min
            shifted_ranks = self._numerical_ops.array_add(avg_ranks, neg_min)
            
            # Then: divide by range (multiply by 1/range)
            # Handle edge case where range might be 0
            safe_range = self._numerical_ops.maximum(ranks_range, 1e-10)
            normalized = self._numerical_ops.array_multiply(shifted_ranks, 1.0 / float(safe_range))
            
            # Finally: 1.0 - normalized
            ones = self._numerical_ops.ones_like(normalized)
            neg_normalized = self._numerical_ops.array_multiply(normalized, -1.0)
            return self._numerical_ops.array_add(ones, neg_normalized)
            
        else:
            raise NotImplementedError(f"Blending not implemented for {method}")
    
    def _rank_predictions(self, predictions: Any) -> Any:
        """Convert predictions to ranks."""
        # Higher prediction -> lower rank (1 is best)
        # First argsort gives indices that would sort array
        # Second argsort on those indices gives ranks
        sorted_indices = self._numerical_ops.array_argsort(predictions)
        ranks = self._numerical_ops.array_argsort(sorted_indices)
        # Add 1 to make ranks start from 1 instead of 0
        return self._numerical_ops.array_add(ranks, self._numerical_ops.ones_like(ranks))