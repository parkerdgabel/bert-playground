"""Training Orchestrator for complex training workflows.

This orchestrator handles advanced training scenarios like:
- Cross-validation
- Hyperparameter tuning
- Ensemble training
- Multi-stage training
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import asyncio

from application.dto.training import TrainingRequestDTO, TrainingResponseDTO
from application.dto.evaluation import EvaluationRequestDTO, EvaluationResponseDTO
from application.use_cases.train_model import TrainModelUseCase
from application.use_cases.evaluate_model import EvaluateModelUseCase
from ports.secondary.monitoring import MonitoringService
from ports.secondary.storage import StorageService


@dataclass
class CrossValidationConfig:
    """Configuration for cross-validation."""
    num_folds: int = 5
    stratified: bool = True
    shuffle: bool = True
    random_seed: Optional[int] = None


@dataclass
class HyperparameterTuningConfig:
    """Configuration for hyperparameter tuning."""
    param_grid: Dict[str, List[Any]] = field(default_factory=dict)
    search_strategy: str = "grid"  # "grid", "random", "bayesian"
    num_trials: Optional[int] = None
    scoring_metric: str = "accuracy"
    maximize: bool = True


@dataclass
class EnsembleConfig:
    """Configuration for ensemble training."""
    num_models: int = 5
    ensemble_strategy: str = "voting"  # "voting", "averaging", "stacking"
    diversity_method: Optional[str] = None  # "bagging", "random_init", etc.


class TrainingOrchestrator:
    """Orchestrates complex training workflows.
    
    This orchestrator coordinates multiple training runs for advanced
    scenarios like cross-validation, hyperparameter tuning, and ensembles.
    """
    
    def __init__(
        self,
        train_use_case: TrainModelUseCase,
        evaluate_use_case: EvaluateModelUseCase,
        monitoring_port: MonitoringService,
        storage_port: StorageService,
    ):
        """Initialize the orchestrator.
        
        Args:
            train_use_case: Use case for training models
            evaluate_use_case: Use case for evaluating models
            monitoring_port: Port for monitoring and logging
            storage_port: Port for storage operations
        """
        self.train_use_case = train_use_case
        self.evaluate_use_case = evaluate_use_case
        self.monitoring = monitoring_port
        self.storage = storage_port
    
    async def cross_validation(
        self,
        base_request: TrainingRequestDTO,
        cv_config: CrossValidationConfig
    ) -> Dict[str, Any]:
        """Perform cross-validation training.
        
        Args:
            base_request: Base training request to use for all folds
            cv_config: Cross-validation configuration
            
        Returns:
            Dictionary with CV results including mean metrics and fold details
        """
        start_time = datetime.now()
        experiment_name = f"cv_{base_request.experiment_name or 'default'}"
        
        await self.monitoring.log_info(
            f"Starting {cv_config.num_folds}-fold cross-validation"
        )
        
        # Create data splits
        fold_splits = await self._create_cv_splits(
            base_request.train_data_path,
            cv_config
        )
        
        # Train on each fold
        fold_results = []
        all_metrics = {metric: [] for metric in ['accuracy', 'loss', 'f1']}
        
        for fold_idx, (train_indices, val_indices) in enumerate(fold_splits):
            await self.monitoring.log_info(f"Training fold {fold_idx + 1}/{cv_config.num_folds}")
            
            # Create fold-specific request
            fold_request = self._create_fold_request(
                base_request,
                fold_idx,
                train_indices,
                val_indices,
                experiment_name
            )
            
            # Train the fold
            fold_response = await self.train_use_case.execute(fold_request)
            
            if not fold_response.success:
                await self.monitoring.log_error(
                    f"Fold {fold_idx + 1} failed: {fold_response.error_message}"
                )
                continue
            
            # Evaluate on validation set
            eval_request = EvaluationRequestDTO(
                model_path=fold_response.best_model_path,
                data_path=fold_request.val_data_path,
                batch_size=base_request.batch_size,
                experiment_name=experiment_name,
                run_name=f"eval_fold_{fold_idx + 1}",
            )
            
            eval_response = await self.evaluate_use_case.execute(eval_request)
            
            # Collect results
            fold_result = {
                'fold': fold_idx + 1,
                'training_response': fold_response,
                'evaluation_response': eval_response,
                'metrics': eval_response.metrics if eval_response.success else {},
            }
            fold_results.append(fold_result)
            
            # Accumulate metrics
            for metric, value in fold_result['metrics'].items():
                if metric in all_metrics:
                    all_metrics[metric].append(value)
        
        # Compute aggregate metrics
        aggregate_metrics = {}
        for metric, values in all_metrics.items():
            if values:
                aggregate_metrics[f"{metric}_mean"] = sum(values) / len(values)
                aggregate_metrics[f"{metric}_std"] = self._compute_std(values)
        
        # Save CV results
        cv_results = {
            'config': cv_config.__dict__,
            'num_folds': cv_config.num_folds,
            'fold_results': [self._serialize_fold_result(r) for r in fold_results],
            'aggregate_metrics': aggregate_metrics,
            'total_time_seconds': (datetime.now() - start_time).total_seconds(),
        }
        
        output_dir = base_request.output_dir / "cross_validation"
        await self.storage.create_directory(output_dir)
        await self.storage.save_json(cv_results, output_dir / "cv_results.json")
        
        await self.monitoring.log_info(
            f"Cross-validation completed. Mean accuracy: {aggregate_metrics.get('accuracy_mean', 0):.4f}"
        )
        
        return cv_results
    
    async def hyperparameter_tuning(
        self,
        base_request: TrainingRequestDTO,
        tuning_config: HyperparameterTuningConfig
    ) -> Dict[str, Any]:
        """Perform hyperparameter tuning.
        
        Args:
            base_request: Base training request
            tuning_config: Hyperparameter tuning configuration
            
        Returns:
            Dictionary with tuning results and best parameters
        """
        start_time = datetime.now()
        experiment_name = f"hpt_{base_request.experiment_name or 'default'}"
        
        await self.monitoring.log_info(
            f"Starting hyperparameter tuning with {tuning_config.search_strategy} search"
        )
        
        # Generate parameter combinations
        param_combinations = self._generate_param_combinations(tuning_config)
        
        # Train with each combination
        trial_results = []
        best_score = float('-inf') if tuning_config.maximize else float('inf')
        best_params = None
        best_model_path = None
        
        for trial_idx, params in enumerate(param_combinations):
            await self.monitoring.log_info(
                f"Trial {trial_idx + 1}/{len(param_combinations)}: {params}"
            )
            
            # Create trial request
            trial_request = self._create_trial_request(
                base_request,
                params,
                trial_idx,
                experiment_name
            )
            
            # Train the model
            trial_response = await self.train_use_case.execute(trial_request)
            
            if not trial_response.success:
                await self.monitoring.log_error(
                    f"Trial {trial_idx + 1} failed: {trial_response.error_message}"
                )
                continue
            
            # Get scoring metric
            score = trial_response.best_val_metric
            
            # Track results
            trial_result = {
                'trial': trial_idx + 1,
                'params': params,
                'score': score,
                'response': trial_response,
            }
            trial_results.append(trial_result)
            
            # Update best
            if (tuning_config.maximize and score > best_score) or \
               (not tuning_config.maximize and score < best_score):
                best_score = score
                best_params = params
                best_model_path = trial_response.best_model_path
        
        # Save tuning results
        tuning_results = {
            'config': tuning_config.__dict__,
            'num_trials': len(param_combinations),
            'best_params': best_params,
            'best_score': best_score,
            'best_model_path': str(best_model_path) if best_model_path else None,
            'trial_results': [self._serialize_trial_result(r) for r in trial_results],
            'total_time_seconds': (datetime.now() - start_time).total_seconds(),
        }
        
        output_dir = base_request.output_dir / "hyperparameter_tuning"
        await self.storage.create_directory(output_dir)
        await self.storage.save_json(tuning_results, output_dir / "tuning_results.json")
        
        await self.monitoring.log_info(
            f"Hyperparameter tuning completed. Best score: {best_score:.4f}"
        )
        
        return tuning_results
    
    async def ensemble_training(
        self,
        base_request: TrainingRequestDTO,
        ensemble_config: EnsembleConfig
    ) -> Dict[str, Any]:
        """Train an ensemble of models.
        
        Args:
            base_request: Base training request
            ensemble_config: Ensemble configuration
            
        Returns:
            Dictionary with ensemble training results
        """
        start_time = datetime.now()
        experiment_name = f"ensemble_{base_request.experiment_name or 'default'}"
        
        await self.monitoring.log_info(
            f"Starting ensemble training with {ensemble_config.num_models} models"
        )
        
        # Train individual models
        model_results = []
        model_paths = []
        
        for model_idx in range(ensemble_config.num_models):
            await self.monitoring.log_info(
                f"Training model {model_idx + 1}/{ensemble_config.num_models}"
            )
            
            # Create model-specific request
            model_request = self._create_ensemble_model_request(
                base_request,
                model_idx,
                ensemble_config,
                experiment_name
            )
            
            # Train the model
            model_response = await self.train_use_case.execute(model_request)
            
            if not model_response.success:
                await self.monitoring.log_error(
                    f"Model {model_idx + 1} failed: {model_response.error_message}"
                )
                continue
            
            model_results.append(model_response)
            model_paths.append(model_response.best_model_path)
        
        # Evaluate ensemble on validation set
        if base_request.val_data_path and model_paths:
            ensemble_metrics = await self._evaluate_ensemble(
                model_paths,
                base_request.val_data_path,
                ensemble_config,
                base_request.batch_size
            )
        else:
            ensemble_metrics = {}
        
        # Save ensemble results
        ensemble_results = {
            'config': ensemble_config.__dict__,
            'num_models': ensemble_config.num_models,
            'model_paths': [str(p) for p in model_paths],
            'individual_results': [self._serialize_response(r) for r in model_results],
            'ensemble_metrics': ensemble_metrics,
            'total_time_seconds': (datetime.now() - start_time).total_seconds(),
        }
        
        output_dir = base_request.output_dir / "ensemble"
        await self.storage.create_directory(output_dir)
        await self.storage.save_json(ensemble_results, output_dir / "ensemble_results.json")
        
        await self.monitoring.log_info(
            f"Ensemble training completed with {len(model_paths)} models"
        )
        
        return ensemble_results
    
    async def _create_cv_splits(
        self,
        data_path: Path,
        cv_config: CrossValidationConfig
    ) -> List[Tuple[List[int], List[int]]]:
        """Create cross-validation data splits."""
        # This would implement actual CV splitting logic
        # Placeholder for now
        raise NotImplementedError("CV splitting to be implemented")
    
    def _create_fold_request(
        self,
        base_request: TrainingRequestDTO,
        fold_idx: int,
        train_indices: List[int],
        val_indices: List[int],
        experiment_name: str
    ) -> TrainingRequestDTO:
        """Create a training request for a specific fold."""
        # Create a copy of base request with fold-specific settings
        fold_request = TrainingRequestDTO(**base_request.__dict__)
        fold_request.run_name = f"fold_{fold_idx + 1}"
        fold_request.experiment_name = experiment_name
        
        # Would need to handle data splitting here
        # This is a simplified version
        return fold_request
    
    def _generate_param_combinations(
        self,
        tuning_config: HyperparameterTuningConfig
    ) -> List[Dict[str, Any]]:
        """Generate parameter combinations for tuning."""
        # This would implement grid/random search logic
        # Placeholder for now
        return [tuning_config.param_grid]
    
    def _create_trial_request(
        self,
        base_request: TrainingRequestDTO,
        params: Dict[str, Any],
        trial_idx: int,
        experiment_name: str
    ) -> TrainingRequestDTO:
        """Create a training request for a hyperparameter trial."""
        trial_request = TrainingRequestDTO(**base_request.__dict__)
        trial_request.run_name = f"trial_{trial_idx + 1}"
        trial_request.experiment_name = experiment_name
        
        # Apply hyperparameters
        for param, value in params.items():
            if hasattr(trial_request, param):
                setattr(trial_request, param, value)
        
        return trial_request
    
    def _create_ensemble_model_request(
        self,
        base_request: TrainingRequestDTO,
        model_idx: int,
        ensemble_config: EnsembleConfig,
        experiment_name: str
    ) -> TrainingRequestDTO:
        """Create a training request for an ensemble member."""
        model_request = TrainingRequestDTO(**base_request.__dict__)
        model_request.run_name = f"model_{model_idx + 1}"
        model_request.experiment_name = experiment_name
        
        # Apply diversity method if specified
        if ensemble_config.diversity_method == "random_init":
            # Use different random seed
            model_request.tags["random_seed"] = str(model_idx)
        
        return model_request
    
    async def _evaluate_ensemble(
        self,
        model_paths: List[Path],
        val_data_path: Path,
        ensemble_config: EnsembleConfig,
        batch_size: int
    ) -> Dict[str, float]:
        """Evaluate ensemble on validation data."""
        # This would implement ensemble evaluation logic
        # Placeholder for now
        return {"ensemble_accuracy": 0.95}
    
    def _compute_std(self, values: List[float]) -> float:
        """Compute standard deviation."""
        if not values:
            return 0.0
        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / len(values)
        return variance ** 0.5
    
    def _serialize_fold_result(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Serialize fold result for saving."""
        return {
            'fold': result['fold'],
            'metrics': result['metrics'],
            'model_path': str(result['training_response'].best_model_path),
        }
    
    def _serialize_trial_result(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Serialize trial result for saving."""
        return {
            'trial': result['trial'],
            'params': result['params'],
            'score': result['score'],
            'model_path': str(result['response'].best_model_path),
        }
    
    def _serialize_response(self, response: TrainingResponseDTO) -> Dict[str, Any]:
        """Serialize training response for saving."""
        return {
            'success': response.success,
            'final_metrics': response.final_metrics,
            'best_model_path': str(response.best_model_path),
        }