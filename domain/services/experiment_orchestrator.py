"""Domain service for experiment orchestration.

This service manages the experiment lifecycle, from hypothesis
formation through execution and analysis.
"""

from typing import List, Dict, Any, Optional, Tuple, Set
from datetime import datetime, timedelta
from dataclasses import dataclass
import statistics

from domain.entities.experiment import (
    Experiment, ExperimentId, ExperimentStatus, ExperimentApproach,
    Hypothesis, ExperimentConfig, ExperimentResults, ExperimentMetrics,
    ExperimentArtifacts, ExperimentInsights, ExperimentComparison
)
from domain.entities.competition import Competition
from domain.entities.ensemble import ModelEnsemble, EnsembleMethod
from domain.value_objects.validation_strategy import (
    ValidationStrategy, CrossValidationSummary, ValidationResult
)
from domain.registry import domain_service, ServiceScope


@dataclass
class ExperimentDesign:
    """Complete experiment design."""
    hypothesis: Hypothesis
    approach: ExperimentApproach
    config: ExperimentConfig
    validation_strategy: ValidationStrategy
    success_criteria: List[str]
    risk_mitigation: Dict[str, str]


@dataclass
class ExperimentPlan:
    """Plan for a series of experiments."""
    experiments: List[ExperimentDesign]
    dependencies: Dict[str, List[str]]  # experiment_name -> dependencies
    estimated_duration_hours: float
    resource_requirements: Dict[str, Any]


@dataclass
class ExperimentInsight:
    """Insight gained from experiment."""
    finding: str
    evidence: str
    impact: str  # "high", "medium", "low"
    actionable: bool
    next_steps: List[str]


@domain_service(scope=ServiceScope.SINGLETON)
class ExperimentOrchestrator:
    """Orchestrates ML experiments for competitions."""
    
    def design_experiment(
        self,
        hypothesis: Hypothesis,
        competition: Competition,
        previous_results: Optional[List[ExperimentResults]] = None
    ) -> ExperimentDesign:
        """Design an experiment based on hypothesis and competition."""
        # Determine approach based on hypothesis
        approach = self._determine_approach(hypothesis)
        
        # Create configuration
        config = self._create_config(hypothesis, competition, approach)
        
        # Select validation strategy
        validation = self._select_validation_strategy(competition, approach)
        
        # Define success criteria
        success_criteria = self._define_success_criteria(hypothesis, competition)
        
        # Identify risks and mitigation
        risks = self._identify_risks(approach, competition)
        
        return ExperimentDesign(
            hypothesis=hypothesis,
            approach=approach,
            config=config,
            validation_strategy=validation,
            success_criteria=success_criteria,
            risk_mitigation=risks
        )
    
    def _determine_approach(self, hypothesis: Hypothesis) -> ExperimentApproach:
        """Determine experiment approach from hypothesis."""
        description = hypothesis.description.lower()
        
        if "baseline" in description or "simple" in description:
            return ExperimentApproach.BASELINE
        elif "feature" in description or "engineering" in description:
            return ExperimentApproach.FEATURE_ENGINEERING
        elif "architecture" in description or "model" in description:
            return ExperimentApproach.MODEL_ARCHITECTURE
        elif "ensemble" in description or "blend" in description:
            return ExperimentApproach.ENSEMBLE
        elif "augment" in description:
            return ExperimentApproach.DATA_AUGMENTATION
        elif "transfer" in description or "pretrain" in description:
            return ExperimentApproach.TRANSFER_LEARNING
        elif "pseudo" in description:
            return ExperimentApproach.PSEUDO_LABELING
        else:
            return ExperimentApproach.CUSTOM
    
    def _create_config(
        self,
        hypothesis: Hypothesis,
        competition: Competition,
        approach: ExperimentApproach
    ) -> ExperimentConfig:
        """Create experiment configuration."""
        # Base configuration
        config = ExperimentConfig(
            model_type="bert-base",  # Default
            preprocessing_steps=["tokenize", "pad", "truncate"],
            feature_engineering={},
            hyperparameters={
                "learning_rate": 2e-5,
                "batch_size": 32,
                "max_epochs": 10,
                "warmup_ratio": 0.1
            },
            validation_strategy="5fold"
        )
        
        # Adjust based on approach
        if approach == ExperimentApproach.MODEL_ARCHITECTURE:
            config.model_type = "modernbert-base"  # Try modern variant
            config.hyperparameters["learning_rate"] = 1e-5  # Lower LR for larger model
            
        elif approach == ExperimentApproach.DATA_AUGMENTATION:
            config.augmentation_config = {
                "synonym_replacement": 0.1,
                "random_deletion": 0.1,
                "random_swap": 0.1
            }
            
        elif approach == ExperimentApproach.ENSEMBLE:
            config.ensemble_config = {
                "method": "weighted_average",
                "n_models": 5
            }
        
        # Adjust for competition size
        if competition.dataset_info.total_samples > 1_000_000:
            config.hyperparameters["batch_size"] = 64  # Larger batches
            config.validation_strategy = "3fold"  # Fewer folds for speed
        
        return config
    
    def _select_validation_strategy(
        self,
        competition: Competition,
        approach: ExperimentApproach
    ) -> ValidationStrategy:
        """Select appropriate validation strategy."""
        # Import here to avoid circular dependency
        from domain.value_objects.validation_strategy import (
            StratifiedKFoldStrategy, TimeSeriesSplitStrategy
        )
        
        # Use competition's suggested strategy
        suggested = competition.get_optimal_validation_strategy()
        
        if "time_series" in suggested:
            return TimeSeriesSplitStrategy(n_splits=5)
        elif "stratified" in suggested:
            n_splits = int(suggested.split("_")[-1]) if "_" in suggested else 5
            return StratifiedKFoldStrategy(n_splits=n_splits)
        else:
            # Default stratified 5-fold
            return StratifiedKFoldStrategy(n_splits=5)
    
    def _define_success_criteria(
        self,
        hypothesis: Hypothesis,
        competition: Competition
    ) -> List[str]:
        """Define success criteria for experiment."""
        criteria = []
        
        # Target score achievement
        if hypothesis.target_score:
            criteria.append(f"Achieve validation score >= {hypothesis.target_score:.5f}")
        
        # Improvement over baseline
        if hypothesis.baseline_score:
            criteria.append(f"Improve over baseline by {hypothesis.expected_improvement:.5f}")
        
        # Stability
        criteria.append("CV standard deviation < 0.01")
        
        # Leaderboard improvement
        criteria.append("Public leaderboard score improves")
        
        # Training stability
        criteria.append("No training instabilities or NaN losses")
        
        return criteria
    
    def _identify_risks(
        self,
        approach: ExperimentApproach,
        competition: Competition
    ) -> Dict[str, str]:
        """Identify risks and mitigation strategies."""
        risks = {}
        
        # Common risks
        risks["overfitting"] = "Monitor train/val gap, use early stopping"
        risks["time_constraint"] = "Set hard time limits, use smaller data samples"
        
        # Approach-specific risks
        if approach == ExperimentApproach.MODEL_ARCHITECTURE:
            risks["memory_overflow"] = "Use gradient accumulation, reduce batch size"
            risks["slow_training"] = "Use mixed precision, optimize data loading"
            
        elif approach == ExperimentApproach.ENSEMBLE:
            risks["model_correlation"] = "Ensure diverse base models"
            risks["complexity"] = "Start with simple averaging"
            
        elif approach == ExperimentApproach.PSEUDO_LABELING:
            risks["noise_amplification"] = "Use high confidence threshold"
            risks["distribution_shift"] = "Validate on trusted labels only"
        
        # Competition-specific risks
        if competition.leaderboard.private_percentage > 0.7:
            risks["public_lb_overfit"] = "Trust CV more than public LB"
        
        return risks
    
    def create_experiment_plan(
        self,
        competition: Competition,
        available_time_hours: float,
        resource_constraints: Dict[str, Any]
    ) -> ExperimentPlan:
        """Create a complete experiment plan for the competition."""
        experiments = []
        
        # Phase 1: Baseline
        baseline_hypothesis = Hypothesis(
            description="Establish baseline with standard BERT",
            rationale="Need reference point for improvements",
            expected_improvement=0.0,
            confidence_level=0.9
        )
        baseline_design = self.design_experiment(baseline_hypothesis, competition)
        experiments.append(baseline_design)
        
        # Phase 2: Quick wins
        if competition.is_nlp_competition:
            # Try better model
            model_hypothesis = Hypothesis(
                description="Use ModernBERT for improved performance",
                rationale="ModernBERT has shown improvements over classic BERT",
                expected_improvement=0.02,
                confidence_level=0.7
            )
            experiments.append(self.design_experiment(model_hypothesis, competition))
        
        # Phase 3: Feature engineering
        feature_hypothesis = Hypothesis(
            description="Engineer domain-specific features",
            rationale="Domain knowledge can improve model understanding",
            expected_improvement=0.015,
            confidence_level=0.6
        )
        experiments.append(self.design_experiment(feature_hypothesis, competition))
        
        # Phase 4: Ensemble (if time permits)
        estimated_time_so_far = len(experiments) * 10  # 10 hours per experiment estimate
        if estimated_time_so_far < available_time_hours * 0.7:
            ensemble_hypothesis = Hypothesis(
                description="Create weighted ensemble of best models",
                rationale="Ensemble typically improves single model performance",
                expected_improvement=0.025,
                confidence_level=0.8
            )
            experiments.append(self.design_experiment(ensemble_hypothesis, competition))
        
        # Define dependencies
        dependencies = {
            "baseline": [],
            "modernbert": ["baseline"],
            "features": ["baseline"],
            "ensemble": ["baseline", "modernbert", "features"]
        }
        
        # Calculate resource requirements
        total_gpu_hours = sum(exp.config.hyperparameters.get("max_epochs", 10) * 2 
                             for exp in experiments)
        
        return ExperimentPlan(
            experiments=experiments,
            dependencies=dependencies,
            estimated_duration_hours=len(experiments) * 10,
            resource_requirements={
                "gpu_hours": total_gpu_hours,
                "memory_gb": 16,
                "storage_gb": 50
            }
        )
    
    def analyze_experiment_results(
        self,
        experiment: Experiment,
        cv_results: CrossValidationSummary
    ) -> List[ExperimentInsight]:
        """Analyze experiment results and extract insights."""
        insights = []
        
        if not experiment.results:
            return insights
        
        results = experiment.results
        
        # Performance insights
        if cv_results.is_stable:
            insights.append(ExperimentInsight(
                finding="Stable cross-validation scores",
                evidence=f"CV std {cv_results.std_val_score:.5f} < 1% of mean",
                impact="high",
                actionable=True,
                next_steps=["Safe to trust CV for model selection"]
            ))
        else:
            insights.append(ExperimentInsight(
                finding="High variance in CV scores",
                evidence=f"CV std {cv_results.std_val_score:.5f}",
                impact="high",
                actionable=True,
                next_steps=["Investigate fold differences", "Consider different validation strategy"]
            ))
        
        # Overfitting check
        fold_overfit = [r for r in cv_results.fold_results if r.is_overfitting]
        if fold_overfit:
            insights.append(ExperimentInsight(
                finding=f"Overfitting detected in {len(fold_overfit)} folds",
                evidence="Train score significantly higher than validation",
                impact="high",
                actionable=True,
                next_steps=["Add regularization", "Reduce model complexity", "Increase dropout"]
            ))
        
        # Efficiency insights
        if results.efficiency_score > 0.01:  # Good efficiency
            insights.append(ExperimentInsight(
                finding="Good training efficiency",
                evidence=f"Score/hour ratio: {results.efficiency_score:.4f}",
                impact="medium",
                actionable=False,
                next_steps=["Current approach is efficient"]
            ))
        
        # Hypothesis validation
        if experiment.hypothesis_validated is not None:
            if experiment.hypothesis_validated:
                insights.append(ExperimentInsight(
                    finding="Hypothesis validated",
                    evidence=f"Achieved expected improvement of {experiment.hypothesis.expected_improvement:.5f}",
                    impact="high",
                    actionable=True,
                    next_steps=["Build on this approach", "Try similar improvements"]
                ))
            else:
                insights.append(ExperimentInsight(
                    finding="Hypothesis not validated",
                    evidence="Did not achieve expected improvement",
                    impact="medium",
                    actionable=True,
                    next_steps=["Revise assumptions", "Try different approach"]
                ))
        
        # Fold-specific insights
        if cv_results.worst_fold is not None:
            insights.append(ExperimentInsight(
                finding=f"Fold {cv_results.worst_fold} performed worst",
                evidence=f"Score gap: {cv_results.cv_score_range[1] - cv_results.cv_score_range[0]:.5f}",
                impact="medium",
                actionable=True,
                next_steps=["Investigate data in worst fold", "Check for data quality issues"]
            ))
        
        return insights
    
    def compare_experiments(
        self,
        experiments: List[Experiment]
    ) -> ExperimentComparison:
        """Compare multiple experiments."""
        if not experiments:
            raise ValueError("No experiments to compare")
        
        # Get experiments with results
        completed = [exp for exp in experiments if exp.results]
        if not completed:
            raise ValueError("No completed experiments")
        
        # Extract scores
        metric_comparison = {
            exp.id: exp.results.metrics.validation_mean
            for exp in completed
        }
        
        # Find best
        best_id = max(metric_comparison.items(), key=lambda x: x[1])[0]
        
        # Calculate improvements over baseline
        baseline = next((exp for exp in completed 
                        if exp.approach == ExperimentApproach.BASELINE), None)
        
        improvements = {}
        if baseline:
            baseline_score = baseline.results.metrics.validation_mean
            for exp in completed:
                improvements[exp.id] = exp.results.metrics.validation_mean - baseline_score
        
        # Generate insights
        insights = []
        
        # Best approach
        best_exp = next(exp for exp in completed if exp.id == best_id)
        insights.append(f"Best approach: {best_exp.approach.value} with score {metric_comparison[best_id]:.5f}")
        
        # Approach effectiveness
        approach_scores = {}
        for exp in completed:
            approach = exp.approach.value
            if approach not in approach_scores:
                approach_scores[approach] = []
            approach_scores[approach].append(exp.results.metrics.validation_mean)
        
        for approach, scores in approach_scores.items():
            avg_score = statistics.mean(scores)
            insights.append(f"{approach}: avg score {avg_score:.5f}")
        
        # Consistency
        score_std = statistics.stdev(metric_comparison.values()) if len(metric_comparison) > 1 else 0
        if score_std < 0.01:
            insights.append("Experiments show consistent results")
        else:
            insights.append("High variance between experiments - investigate differences")
        
        return ExperimentComparison(
            experiment_ids=list(metric_comparison.keys()),
            best_experiment_id=best_id,
            metric_comparison=metric_comparison,
            improvement_over_baseline=improvements,
            insights=insights
        )
    
    def suggest_next_experiments(
        self,
        completed_experiments: List[Experiment],
        competition: Competition,
        remaining_time_hours: float
    ) -> List[Hypothesis]:
        """Suggest next experiments based on results."""
        suggestions = []
        
        if not completed_experiments:
            # Start with baseline
            suggestions.append(Hypothesis(
                description="Establish baseline performance",
                rationale="Need reference point",
                expected_improvement=0.0,
                confidence_level=0.9
            ))
            return suggestions
        
        # Analyze what worked
        comparison = self.compare_experiments(completed_experiments)
        best_exp = next(exp for exp in completed_experiments 
                        if exp.id == comparison.best_experiment_id)
        
        # Build on successful approaches
        if best_exp.approach == ExperimentApproach.MODEL_ARCHITECTURE:
            suggestions.append(Hypothesis(
                description="Fine-tune the successful architecture",
                rationale="Further optimize what's working",
                expected_improvement=0.01,
                confidence_level=0.7
            ))
        
        # Try ensemble if we have good models
        good_models = [exp for exp in completed_experiments 
                      if exp.results and exp.results.metrics.validation_mean > 0.85]
        
        if len(good_models) >= 3:
            suggestions.append(Hypothesis(
                description="Create ensemble of top models",
                rationale=f"Have {len(good_models)} strong models to combine",
                expected_improvement=0.02,
                confidence_level=0.8
            ))
        
        # Try augmentation if not done
        augmentation_tried = any(exp.approach == ExperimentApproach.DATA_AUGMENTATION 
                               for exp in completed_experiments)
        
        if not augmentation_tried and competition.is_nlp_competition:
            suggestions.append(Hypothesis(
                description="Apply text augmentation techniques",
                rationale="Increase training data variety",
                expected_improvement=0.015,
                confidence_level=0.6
            ))
        
        return suggestions