"""Domain service for competition management.

This service contains business logic for analyzing competitions,
suggesting strategies, and managing the competition lifecycle.
"""

from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass

from domain.entities.competition import (
    Competition, CompetitionType, OptimizationDirection,
    CompetitionSnapshot
)
from domain.entities.experiment import Experiment, ExperimentApproach, ExperimentStatus
from domain.entities.submission import Submission, SubmissionHistory
from domain.value_objects.validation_strategy import (
    ValidationStrategy, StratifiedKFoldStrategy, TimeSeriesSplitStrategy,
    GroupKFoldStrategy, ValidationStrategyType
)
from domain.registry import domain_service, ServiceScope


@dataclass
class CompetitionAnalysis:
    """Analysis results for a competition."""
    competition_id: str
    difficulty_assessment: str
    recommended_approaches: List[ExperimentApproach]
    validation_strategy: ValidationStrategy
    estimated_iterations: int
    key_challenges: List[str]
    success_factors: List[str]
    similar_competitions: List[str]


@dataclass
class ApproachSuggestion:
    """Suggested approach for competition."""
    approach: ExperimentApproach
    priority: int  # 1-5, 1 being highest
    rationale: str
    expected_impact: str  # "high", "medium", "low"
    complexity: str  # "simple", "moderate", "complex"
    prerequisites: List[str]


@dataclass
class SubmissionStrategy:
    """Strategy for managing submissions."""
    daily_submission_plan: Dict[int, int]  # day -> number of submissions
    experimentation_phase_days: int
    optimization_phase_days: int
    ensemble_phase_days: int
    reserve_submissions: int


@domain_service(scope=ServiceScope.SINGLETON)
class CompetitionService:
    """Service for competition analysis and strategy."""
    
    def analyze_competition(self, competition: Competition) -> CompetitionAnalysis:
        """Analyze competition and provide insights."""
        # Assess difficulty
        difficulty = self._assess_difficulty(competition)
        
        # Recommend approaches based on competition type
        approaches = self._recommend_approaches(competition)
        
        # Suggest validation strategy
        validation_strategy = self._suggest_validation_strategy(competition)
        
        # Estimate iterations needed
        iterations = self._estimate_iterations(competition)
        
        # Identify challenges and success factors
        challenges = self._identify_challenges(competition)
        success_factors = self._identify_success_factors(competition)
        
        return CompetitionAnalysis(
            competition_id=competition.id.value,
            difficulty_assessment=difficulty,
            recommended_approaches=approaches,
            validation_strategy=validation_strategy,
            estimated_iterations=iterations,
            key_challenges=challenges,
            success_factors=success_factors,
            similar_competitions=[]  # Would query historical data
        )
    
    def _assess_difficulty(self, competition: Competition) -> str:
        """Assess competition difficulty."""
        factors = {
            "beginner": 0,
            "intermediate": 0,
            "expert": 0
        }
        
        # Large dataset
        if competition.dataset_info.total_samples > 1_000_000:
            factors["expert"] += 1
        elif competition.dataset_info.total_samples > 100_000:
            factors["intermediate"] += 1
        else:
            factors["beginner"] += 1
        
        # Time constraint
        if competition.timeline.days_remaining < 30:
            factors["expert"] += 1
        elif competition.timeline.days_remaining < 60:
            factors["intermediate"] += 1
        else:
            factors["beginner"] += 1
        
        # Competition type
        if competition.competition_type in [CompetitionType.TIME_SERIES, CompetitionType.RANKING]:
            factors["expert"] += 1
        elif competition.competition_type == CompetitionType.MULTI_LABEL:
            factors["intermediate"] += 1
        
        # Prize pool (higher prizes attract more experts)
        if competition.prize_info and competition.prize_info.total_prize_pool > 50000:
            factors["expert"] += 2
        
        # Return highest scoring difficulty
        return max(factors.items(), key=lambda x: x[1])[0]
    
    def _recommend_approaches(self, competition: Competition) -> List[ExperimentApproach]:
        """Recommend approaches based on competition type."""
        approaches = []
        
        # Always start with baseline
        approaches.append(ExperimentApproach.BASELINE)
        
        # NLP competitions benefit from transfer learning
        if competition.is_nlp_competition:
            approaches.extend([
                ExperimentApproach.TRANSFER_LEARNING,
                ExperimentApproach.DATA_AUGMENTATION,
                ExperimentApproach.MODEL_ARCHITECTURE
            ])
        
        # Time series needs special treatment
        if competition.competition_type == CompetitionType.TIME_SERIES:
            approaches.append(ExperimentApproach.FEATURE_ENGINEERING)
        
        # Most competitions benefit from ensembling
        if competition.allows_ensembling:
            approaches.append(ExperimentApproach.ENSEMBLE)
        
        # If external data allowed, pseudo-labeling might help
        if competition.dataset_info.external_data_allowed:
            approaches.append(ExperimentApproach.PSEUDO_LABELING)
        
        return approaches
    
    def _suggest_validation_strategy(self, competition: Competition) -> ValidationStrategy:
        """Suggest appropriate validation strategy."""
        if competition.competition_type == CompetitionType.TIME_SERIES:
            return TimeSeriesSplitStrategy(
                n_splits=5,
                test_size=None,
                gap=0
            )
        
        # High private percentage needs robust validation
        if competition.leaderboard.private_percentage > 0.5:
            return StratifiedKFoldStrategy(
                n_splits=5,
                shuffle=True,
                stratify_column="target"
            )
        
        # Default stratified k-fold
        return StratifiedKFoldStrategy(
            n_splits=3,
            shuffle=True,
            stratify_column="target"
        )
    
    def _estimate_iterations(self, competition: Competition) -> int:
        """Estimate number of experiments needed."""
        base_iterations = 10  # Minimum experiments
        
        # Adjust based on competition duration
        if competition.timeline.days_remaining > 60:
            base_iterations *= 2
        
        # Adjust based on difficulty
        difficulty = self._assess_difficulty(competition)
        if difficulty == "expert":
            base_iterations *= 2
        elif difficulty == "intermediate":
            base_iterations = int(base_iterations * 1.5)
        
        # Adjust based on dataset size
        if competition.dataset_info.total_samples > 1_000_000:
            base_iterations = int(base_iterations * 0.7)  # Fewer but longer experiments
        
        return base_iterations
    
    def _identify_challenges(self, competition: Competition) -> List[str]:
        """Identify key challenges in the competition."""
        challenges = []
        
        # Large dataset
        if competition.dataset_info.total_samples > 1_000_000:
            challenges.append("Large dataset requires efficient data loading and sampling")
        
        # High private test percentage
        if competition.leaderboard.private_percentage > 0.7:
            challenges.append("High private test percentage - avoid overfitting to public LB")
        
        # Limited submissions
        if competition.rules.max_daily_submissions <= 2:
            challenges.append("Limited daily submissions - careful experiment planning needed")
        
        # Time series
        if competition.competition_type == CompetitionType.TIME_SERIES:
            challenges.append("Time series requires careful validation to avoid leakage")
        
        # Code competition
        if competition.requires_code_submission:
            challenges.append("Code submission requires inference optimization")
        
        return challenges
    
    def _identify_success_factors(self, competition: Competition) -> List[str]:
        """Identify success factors for the competition."""
        factors = []
        
        # NLP competition
        if competition.is_nlp_competition:
            factors.append("Leverage pre-trained BERT models for transfer learning")
        
        # Allows external data
        if competition.dataset_info.external_data_allowed:
            factors.append("Find and utilize relevant external datasets")
        
        # Ensemble allowed
        if competition.allows_ensembling:
            factors.append("Build diverse models for effective ensembling")
        
        # Long duration
        if competition.timeline.days_remaining > 60:
            factors.append("Systematic experimentation with proper tracking")
        
        return factors
    
    def suggest_approaches(
        self,
        competition: Competition,
        completed_experiments: List[Experiment]
    ) -> List[ApproachSuggestion]:
        """Suggest next approaches based on what's been tried."""
        suggestions = []
        
        # Get what's already been tried
        tried_approaches = {exp.approach for exp in completed_experiments}
        
        # Always suggest baseline first
        if ExperimentApproach.BASELINE not in tried_approaches:
            suggestions.append(ApproachSuggestion(
                approach=ExperimentApproach.BASELINE,
                priority=1,
                rationale="Establish baseline performance for comparison",
                expected_impact="medium",
                complexity="simple",
                prerequisites=[]
            ))
        
        # Feature engineering after baseline
        if (ExperimentApproach.BASELINE in tried_approaches and
            ExperimentApproach.FEATURE_ENGINEERING not in tried_approaches):
            suggestions.append(ApproachSuggestion(
                approach=ExperimentApproach.FEATURE_ENGINEERING,
                priority=2,
                rationale="Improve model inputs with domain-specific features",
                expected_impact="high",
                complexity="moderate",
                prerequisites=["baseline_complete"]
            ))
        
        # Model architecture for NLP
        if (competition.is_nlp_competition and
            ExperimentApproach.MODEL_ARCHITECTURE not in tried_approaches):
            suggestions.append(ApproachSuggestion(
                approach=ExperimentApproach.MODEL_ARCHITECTURE,
                priority=2,
                rationale="Try different BERT variants (base, large, custom)",
                expected_impact="high",
                complexity="moderate",
                prerequisites=["gpu_available"]
            ))
        
        # Ensemble when we have multiple good models
        good_models = [exp for exp in completed_experiments 
                      if exp.status == ExperimentStatus.COMPLETED and
                      exp.results and exp.results.metrics.validation_mean > 0.8]
        
        if (len(good_models) >= 3 and
            ExperimentApproach.ENSEMBLE not in tried_approaches):
            suggestions.append(ApproachSuggestion(
                approach=ExperimentApproach.ENSEMBLE,
                priority=1,
                rationale="Combine diverse models for better performance",
                expected_impact="high",
                complexity="simple",
                prerequisites=["multiple_models"]
            ))
        
        # Sort by priority
        suggestions.sort(key=lambda x: x.priority)
        
        return suggestions
    
    def create_submission_strategy(
        self,
        competition: Competition,
        current_snapshot: CompetitionSnapshot
    ) -> SubmissionStrategy:
        """Create submission strategy for remaining competition time."""
        days_remaining = competition.timeline.days_remaining
        daily_limit = competition.rules.max_daily_submissions
        
        # Allocate time to different phases
        experimentation_days = int(days_remaining * 0.6)
        optimization_days = int(days_remaining * 0.3)
        ensemble_days = days_remaining - experimentation_days - optimization_days
        
        # Create daily plan
        daily_plan = {}
        
        # Experimentation phase - use most submissions
        for day in range(experimentation_days):
            daily_plan[day] = daily_limit - 1  # Keep 1 in reserve
        
        # Optimization phase - moderate usage
        for day in range(experimentation_days, experimentation_days + optimization_days):
            daily_plan[day] = max(1, daily_limit // 2)
        
        # Ensemble phase - conservative
        for day in range(experimentation_days + optimization_days, days_remaining):
            daily_plan[day] = 1
        
        # Calculate reserves
        total_available = days_remaining * daily_limit
        total_planned = sum(daily_plan.values())
        reserve = total_available - total_planned
        
        return SubmissionStrategy(
            daily_submission_plan=daily_plan,
            experimentation_phase_days=experimentation_days,
            optimization_phase_days=optimization_days,
            ensemble_phase_days=ensemble_days,
            reserve_submissions=reserve
        )
    
    def analyze_leaderboard_position(
        self,
        competition: Competition,
        submission_history: SubmissionHistory,
        current_rank: int,
        top_scores: List[float]
    ) -> Dict[str, Any]:
        """Analyze current leaderboard position and suggest improvements."""
        best_submission = submission_history.best_submission
        if not best_submission or not best_submission.scores.public_score:
            return {"error": "No scored submissions yet"}
        
        my_score = best_submission.scores.public_score
        top_score = top_scores[0] if top_scores else my_score
        
        # Calculate gaps
        gap_to_top = abs(top_score - my_score)
        percentile = (1 - current_rank / len(top_scores)) * 100 if top_scores else 0
        
        # Analyze trend
        progression = submission_history.get_score_progression()
        is_improving = len(progression) >= 2 and progression[-1][1] > progression[-2][1]
        
        # Suggestions based on position
        suggestions = []
        
        if percentile < 10:  # Top 10%
            suggestions.append("Focus on small optimizations and ensemble refinement")
            suggestions.append("Study top solutions from similar competitions")
        elif percentile < 30:  # Top 30%
            suggestions.append("Try more diverse model architectures")
            suggestions.append("Investigate feature engineering opportunities")
        else:  # Below top 30%
            suggestions.append("Revisit data understanding and validation strategy")
            suggestions.append("Consider major approach changes")
        
        # Check for overfitting
        recent_subs = submission_history.successful_submissions[-5:]
        if recent_subs:
            cv_scores = [s.metadata.notes for s in recent_subs]  # Would need actual CV scores
            # This is placeholder logic
            suggestions.append("Monitor CV/LB correlation to avoid overfitting")
        
        return {
            "current_rank": current_rank,
            "percentile": percentile,
            "gap_to_top": gap_to_top,
            "is_improving": is_improving,
            "suggestions": suggestions,
            "best_score": my_score,
            "submissions_used": submission_history.total_submissions
        }
    
    def suggest_final_push_strategy(
        self,
        competition: Competition,
        days_remaining: int,
        current_best_score: float,
        available_models: List[str]
    ) -> List[str]:
        """Suggest strategy for final days of competition."""
        strategies = []
        
        if days_remaining <= 3:
            # Very final phase
            strategies.append("Focus only on ensemble optimization")
            strategies.append("No new model training - too risky")
            strategies.append("Create multiple ensemble variations")
            
        elif days_remaining <= 7:
            # Final week
            strategies.append("Finalize best models and start ensembling")
            strategies.append("Try different ensemble methods (average, stacking)")
            strategies.append("Reserve submissions for final ensemble tuning")
            
        else:
            # Still time for experiments
            strategies.append("Continue experimentation but start planning ensemble")
            strategies.append("Ensure model diversity for later ensembling")
            strategies.append("Track CV/LB correlation carefully")
        
        # Add specific suggestions based on model count
        if len(available_models) < 3:
            strategies.append("Need more diverse models for effective ensembling")
        elif len(available_models) > 10:
            strategies.append("Consider model pruning - too many models can hurt")
        
        return strategies