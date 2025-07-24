"""Workflow Orchestrator for end-to-end ML workflows.

This orchestrator handles complete ML workflows from data preparation
through training, evaluation, and deployment.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime
from enum import Enum

from application.dto.training import TrainingRequestDTO, TrainingResponseDTO
from application.dto.evaluation import EvaluationRequestDTO, EvaluationResponseDTO
from application.dto.prediction import PredictionRequestDTO, PredictionResponseDTO
from application.use_cases.train_model import TrainModelUseCase
from application.use_cases.evaluate_model import EvaluateModelUseCase
from application.use_cases.predict import PredictUseCase
from application.ports.secondary.monitoring import MonitoringService
from application.ports.secondary.storage import StorageService


class WorkflowStage(Enum):
    """Stages in an ML workflow."""
    DATA_PREPARATION = "data_preparation"
    TRAINING = "training"
    EVALUATION = "evaluation"
    PREDICTION = "prediction"
    DEPLOYMENT = "deployment"


@dataclass
class WorkflowConfig:
    """Configuration for a complete ML workflow."""
    
    # Workflow metadata
    name: str
    description: Optional[str] = None
    
    # Stage configuration
    stages: List[WorkflowStage] = field(
        default_factory=lambda: [
            WorkflowStage.DATA_PREPARATION,
            WorkflowStage.TRAINING,
            WorkflowStage.EVALUATION,
        ]
    )
    
    # Data configuration
    raw_data_path: Path = None
    train_split_ratio: float = 0.8
    val_split_ratio: float = 0.1
    test_split_ratio: float = 0.1
    
    # Training configuration
    train_config: Optional[TrainingRequestDTO] = None
    
    # Evaluation configuration
    eval_metrics: List[str] = field(default_factory=lambda: ["accuracy", "f1", "precision", "recall"])
    eval_on_test: bool = True
    
    # Deployment configuration
    deploy_threshold: Optional[float] = None  # Minimum metric value for deployment
    deploy_metric: str = "accuracy"
    
    # Output configuration
    output_dir: Path = Path("workflows")
    save_intermediate: bool = True
    
    # Failure handling
    stop_on_failure: bool = True
    retry_failed_stages: int = 0


@dataclass
class WorkflowResult:
    """Result of a complete workflow execution."""
    
    # Status
    success: bool
    completed_stages: List[WorkflowStage] = field(default_factory=list)
    failed_stage: Optional[WorkflowStage] = None
    error_message: Optional[str] = None
    
    # Stage results
    stage_results: Dict[str, Any] = field(default_factory=dict)
    
    # Final outputs
    final_model_path: Optional[Path] = None
    evaluation_metrics: Dict[str, float] = field(default_factory=dict)
    deployment_status: Optional[str] = None
    
    # Timing
    start_time: datetime = None
    end_time: datetime = None
    stage_times: Dict[str, float] = field(default_factory=dict)
    
    # Artifacts
    artifacts: Dict[str, Path] = field(default_factory=dict)


class WorkflowOrchestrator:
    """Orchestrates complete ML workflows.
    
    This orchestrator manages end-to-end workflows including:
    - Data preparation and splitting
    - Model training
    - Evaluation on multiple datasets
    - Prediction generation
    - Model deployment decisions
    """
    
    def __init__(
        self,
        train_use_case: TrainModelUseCase,
        evaluate_use_case: EvaluateModelUseCase,
        predict_use_case: PredictUseCase,
        monitoring_port: MonitoringService,
        storage_port: StorageService,
    ):
        """Initialize the orchestrator.
        
        Args:
            train_use_case: Use case for training models
            evaluate_use_case: Use case for evaluating models
            predict_use_case: Use case for generating predictions
            monitoring_port: Port for monitoring and logging
            storage_port: Port for storage operations
        """
        self.train_use_case = train_use_case
        self.evaluate_use_case = evaluate_use_case
        self.predict_use_case = predict_use_case
        self.monitoring = monitoring_port
        self.storage = storage_port
    
    async def execute_workflow(self, config: WorkflowConfig) -> WorkflowResult:
        """Execute a complete ML workflow.
        
        Args:
            config: Workflow configuration
            
        Returns:
            WorkflowResult with execution details
        """
        result = WorkflowResult(
            success=True,
            start_time=datetime.now()
        )
        
        workflow_dir = config.output_dir / config.name
        await self.storage.create_directory(workflow_dir)
        
        await self.monitoring.log_info(f"Starting workflow: {config.name}")
        
        try:
            # Execute each stage
            for stage in config.stages:
                stage_start = datetime.now()
                
                await self.monitoring.log_info(f"Executing stage: {stage.value}")
                
                try:
                    if stage == WorkflowStage.DATA_PREPARATION:
                        await self._execute_data_preparation(config, workflow_dir, result)
                    
                    elif stage == WorkflowStage.TRAINING:
                        await self._execute_training(config, workflow_dir, result)
                    
                    elif stage == WorkflowStage.EVALUATION:
                        await self._execute_evaluation(config, workflow_dir, result)
                    
                    elif stage == WorkflowStage.PREDICTION:
                        await self._execute_prediction(config, workflow_dir, result)
                    
                    elif stage == WorkflowStage.DEPLOYMENT:
                        await self._execute_deployment(config, workflow_dir, result)
                    
                    # Record stage completion
                    result.completed_stages.append(stage)
                    result.stage_times[stage.value] = (datetime.now() - stage_start).total_seconds()
                    
                    await self.monitoring.log_info(
                        f"Stage {stage.value} completed in {result.stage_times[stage.value]:.2f}s"
                    )
                    
                except Exception as e:
                    await self.monitoring.log_error(f"Stage {stage.value} failed: {str(e)}")
                    
                    if config.retry_failed_stages > 0:
                        # Retry logic would go here
                        pass
                    
                    if config.stop_on_failure:
                        result.success = False
                        result.failed_stage = stage
                        result.error_message = str(e)
                        break
            
            # Finalize workflow
            result.end_time = datetime.now()
            
            # Save workflow summary
            await self._save_workflow_summary(config, result, workflow_dir)
            
            if result.success:
                await self.monitoring.log_info(
                    f"Workflow {config.name} completed successfully"
                )
            else:
                await self.monitoring.log_error(
                    f"Workflow {config.name} failed at stage {result.failed_stage.value}"
                )
            
            return result
            
        except Exception as e:
            await self.monitoring.log_error(f"Workflow failed: {str(e)}")
            result.success = False
            result.error_message = str(e)
            result.end_time = datetime.now()
            return result
    
    async def _execute_data_preparation(
        self,
        config: WorkflowConfig,
        workflow_dir: Path,
        result: WorkflowResult
    ) -> None:
        """Execute data preparation stage."""
        await self.monitoring.log_info("Preparing data splits")
        
        # Create data directories
        data_dir = workflow_dir / "data"
        await self.storage.create_directory(data_dir)
        
        # Split data (placeholder - actual implementation would split the data)
        # This would:
        # 1. Load raw data from config.raw_data_path
        # 2. Split into train/val/test based on ratios
        # 3. Save splits to data_dir
        
        # Record artifacts
        result.artifacts['train_data'] = data_dir / "train.csv"
        result.artifacts['val_data'] = data_dir / "val.csv"
        result.artifacts['test_data'] = data_dir / "test.csv"
        
        # Save data statistics
        data_stats = {
            'train_samples': 1000,  # Placeholder
            'val_samples': 200,
            'test_samples': 200,
        }
        
        await self.storage.save_json(
            data_stats,
            data_dir / "data_statistics.json"
        )
        
        result.stage_results['data_preparation'] = data_stats
    
    async def _execute_training(
        self,
        config: WorkflowConfig,
        workflow_dir: Path,
        result: WorkflowResult
    ) -> None:
        """Execute training stage."""
        if not config.train_config:
            raise ValueError("Training configuration not provided")
        
        # Update paths in training config
        train_config = TrainingRequestDTO(**config.train_config.__dict__)
        train_config.train_data_path = result.artifacts.get('train_data', config.raw_data_path)
        train_config.val_data_path = result.artifacts.get('val_data')
        train_config.output_dir = workflow_dir / "training"
        train_config.experiment_name = f"workflow_{config.name}"
        
        # Execute training
        training_response = await self.train_use_case.execute(train_config)
        
        if not training_response.success:
            raise RuntimeError(f"Training failed: {training_response.error_message}")
        
        # Record results
        result.final_model_path = training_response.best_model_path
        result.artifacts['model'] = training_response.best_model_path
        result.artifacts['training_history'] = workflow_dir / "training" / "history.json"
        
        result.stage_results['training'] = {
            'final_loss': training_response.final_train_loss,
            'best_val_loss': training_response.best_val_loss,
            'total_epochs': training_response.total_epochs,
            'early_stopped': training_response.early_stopped,
        }
    
    async def _execute_evaluation(
        self,
        config: WorkflowConfig,
        workflow_dir: Path,
        result: WorkflowResult
    ) -> None:
        """Execute evaluation stage."""
        if not result.final_model_path:
            raise ValueError("No model available for evaluation")
        
        eval_dir = workflow_dir / "evaluation"
        await self.storage.create_directory(eval_dir)
        
        # Evaluate on validation set
        val_eval_request = EvaluationRequestDTO(
            model_path=result.final_model_path,
            data_path=result.artifacts.get('val_data', config.raw_data_path),
            metrics=config.eval_metrics,
            output_dir=eval_dir / "validation",
            save_predictions=True,
            save_confusion_matrix=True,
            error_analysis=True,
            experiment_name=f"workflow_{config.name}",
            run_name="eval_validation",
        )
        
        val_eval_response = await self.evaluate_use_case.execute(val_eval_request)
        
        if not val_eval_response.success:
            raise RuntimeError(f"Validation evaluation failed: {val_eval_response.error_message}")
        
        # Update evaluation metrics
        for metric, value in val_eval_response.metrics.items():
            result.evaluation_metrics[f"val_{metric}"] = value
        
        # Evaluate on test set if configured
        if config.eval_on_test and result.artifacts.get('test_data'):
            test_eval_request = EvaluationRequestDTO(
                model_path=result.final_model_path,
                data_path=result.artifacts['test_data'],
                metrics=config.eval_metrics,
                output_dir=eval_dir / "test",
                save_predictions=True,
                save_confusion_matrix=True,
                error_analysis=True,
                experiment_name=f"workflow_{config.name}",
                run_name="eval_test",
            )
            
            test_eval_response = await self.evaluate_use_case.execute(test_eval_request)
            
            if test_eval_response.success:
                for metric, value in test_eval_response.metrics.items():
                    result.evaluation_metrics[f"test_{metric}"] = value
        
        result.stage_results['evaluation'] = result.evaluation_metrics
        result.artifacts['evaluation_report'] = eval_dir / "report.json"
        
        # Save evaluation report
        await self.storage.save_json(
            result.evaluation_metrics,
            result.artifacts['evaluation_report']
        )
    
    async def _execute_prediction(
        self,
        config: WorkflowConfig,
        workflow_dir: Path,
        result: WorkflowResult
    ) -> None:
        """Execute prediction stage."""
        if not result.final_model_path:
            raise ValueError("No model available for prediction")
        
        # Generate predictions on test data
        predict_request = PredictionRequestDTO(
            model_path=result.final_model_path,
            data_path=result.artifacts.get('test_data', config.raw_data_path),
            output_path=workflow_dir / "predictions" / "test_predictions.csv",
            include_probabilities=True,
            track_predictions=True,
            experiment_name=f"workflow_{config.name}",
            run_name="predict_test",
        )
        
        predict_response = await self.predict_use_case.execute(predict_request)
        
        if not predict_response.success:
            raise RuntimeError(f"Prediction failed: {predict_response.error_message}")
        
        result.artifacts['predictions'] = predict_response.output_path
        result.stage_results['prediction'] = {
            'num_predictions': predict_response.num_predictions,
            'output_path': str(predict_response.output_path),
        }
    
    async def _execute_deployment(
        self,
        config: WorkflowConfig,
        workflow_dir: Path,
        result: WorkflowResult
    ) -> None:
        """Execute deployment stage."""
        await self.monitoring.log_info("Checking deployment criteria")
        
        # Check if model meets deployment threshold
        deploy_metric_value = result.evaluation_metrics.get(
            f"test_{config.deploy_metric}",
            result.evaluation_metrics.get(f"val_{config.deploy_metric}", 0.0)
        )
        
        should_deploy = True
        if config.deploy_threshold is not None:
            should_deploy = deploy_metric_value >= config.deploy_threshold
        
        if should_deploy:
            # Copy model to deployment directory
            deploy_dir = workflow_dir / "deployment"
            await self.storage.create_directory(deploy_dir)
            
            # In a real implementation, this would:
            # 1. Package the model for deployment
            # 2. Create deployment configuration
            # 3. Push to model registry or deployment platform
            
            result.deployment_status = "deployed"
            result.artifacts['deployment_model'] = deploy_dir / "model"
            
            await self.monitoring.log_info(
                f"Model deployed! {config.deploy_metric}: {deploy_metric_value:.4f}"
            )
        else:
            result.deployment_status = "not_deployed"
            await self.monitoring.log_warning(
                f"Model did not meet deployment threshold. "
                f"{config.deploy_metric}: {deploy_metric_value:.4f} < {config.deploy_threshold}"
            )
        
        result.stage_results['deployment'] = {
            'deployed': should_deploy,
            'metric_value': deploy_metric_value,
            'threshold': config.deploy_threshold,
        }
    
    async def _save_workflow_summary(
        self,
        config: WorkflowConfig,
        result: WorkflowResult,
        workflow_dir: Path
    ) -> None:
        """Save workflow execution summary."""
        summary = {
            'workflow_name': config.name,
            'description': config.description,
            'success': result.success,
            'completed_stages': [s.value for s in result.completed_stages],
            'failed_stage': result.failed_stage.value if result.failed_stage else None,
            'error_message': result.error_message,
            'start_time': result.start_time.isoformat(),
            'end_time': result.end_time.isoformat() if result.end_time else None,
            'total_time_seconds': (result.end_time - result.start_time).total_seconds() if result.end_time else None,
            'stage_times': result.stage_times,
            'stage_results': result.stage_results,
            'evaluation_metrics': result.evaluation_metrics,
            'deployment_status': result.deployment_status,
            'artifacts': {k: str(v) for k, v in result.artifacts.items()},
        }
        
        await self.storage.save_json(
            summary,
            workflow_dir / "workflow_summary.json"
        )