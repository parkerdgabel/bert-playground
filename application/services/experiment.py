"""Experiment Tracker Service - manages experiment tracking and comparison."""

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple
import json
import pandas as pd
from enum import Enum

from application.ports.secondary.storage import StorageService as StoragePort
from application.ports.secondary.monitoring import MonitoringService as MonitoringPort


class ExperimentStatus(Enum):
    """Status of an experiment."""
    CREATED = "created"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class ExperimentConfig:
    """Configuration for an experiment."""
    name: str
    description: Optional[str] = None
    tags: Dict[str, str] = field(default_factory=dict)
    
    # Hyperparameters to track
    hyperparameters: Dict[str, Any] = field(default_factory=dict)
    
    # Metrics to track
    primary_metric: str = "eval_loss"
    metrics_to_track: List[str] = field(default_factory=lambda: ["loss", "accuracy"])
    
    # Comparison settings
    baseline_experiment: Optional[str] = None
    comparison_metrics: List[str] = field(default_factory=lambda: ["eval_loss", "eval_accuracy"])
    
    # Storage settings
    save_artifacts: bool = True
    save_checkpoints: bool = True
    save_visualizations: bool = True


@dataclass
class Experiment:
    """Represents a single experiment."""
    id: str
    name: str
    status: ExperimentStatus
    config: ExperimentConfig
    
    # Metadata
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    
    # Results
    metrics: Dict[str, List[float]] = field(default_factory=dict)
    best_metrics: Dict[str, float] = field(default_factory=dict)
    artifacts: Dict[str, Path] = field(default_factory=dict)
    
    # System info
    system_info: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert experiment to dictionary."""
        return {
            "id": self.id,
            "name": self.name,
            "status": self.status.value,
            "config": self.config.__dict__,
            "created_at": self.created_at.isoformat(),
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "metrics": self.metrics,
            "best_metrics": self.best_metrics,
            "artifacts": {k: str(v) for k, v in self.artifacts.items()},
            "system_info": self.system_info
        }


@dataclass
class ExperimentTracker:
    """Service for tracking and managing experiments.
    
    This service provides comprehensive experiment tracking including
    metrics, artifacts, comparisons, and visualizations.
    """
    
    storage_port: StoragePort
    monitoring_port: MonitoringPort
    
    # Storage paths
    experiments_dir: Path = Path("experiments")
    current_experiment: Optional[Experiment] = None
    
    async def create_experiment(
        self,
        config: ExperimentConfig,
        system_info: Optional[Dict[str, Any]] = None
    ) -> Experiment:
        """Create a new experiment.
        
        Args:
            config: Experiment configuration
            system_info: System information to track
            
        Returns:
            Created experiment
        """
        # Generate experiment ID
        experiment_id = f"{config.name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Create experiment
        experiment = Experiment(
            id=experiment_id,
            name=config.name,
            status=ExperimentStatus.CREATED,
            config=config,
            created_at=datetime.now(),
            system_info=system_info or await self._get_system_info()
        )
        
        # Create experiment directory
        experiment_dir = self.experiments_dir / experiment_id
        await self.storage_port.create_directory(experiment_dir)
        
        # Save initial experiment metadata
        metadata_path = experiment_dir / "experiment.json"
        await self.storage_port.save_json(experiment.to_dict(), metadata_path)
        
        # Set as current experiment
        self.current_experiment = experiment
        
        await self.monitoring_port.log_info(f"Created experiment: {experiment_id}")
        
        return experiment
    
    async def start_experiment(self, experiment_id: Optional[str] = None) -> None:
        """Start an experiment.
        
        Args:
            experiment_id: ID of experiment to start (uses current if None)
        """
        experiment = await self._get_experiment(experiment_id)
        
        experiment.status = ExperimentStatus.RUNNING
        experiment.started_at = datetime.now()
        
        # Log experiment start
        await self.monitoring_port.start_run(
            name=experiment.name,
            experiment=experiment.id,
            tags=experiment.config.tags
        )
        
        # Log hyperparameters
        if experiment.config.hyperparameters:
            await self.monitoring_port.log_params(experiment.config.hyperparameters)
        
        # Save updated metadata
        await self._save_experiment(experiment)
        
        await self.monitoring_port.log_info(f"Started experiment: {experiment.id}")
    
    async def log_metrics(
        self,
        metrics: Dict[str, float],
        step: Optional[int] = None,
        experiment_id: Optional[str] = None
    ) -> None:
        """Log metrics for an experiment.
        
        Args:
            metrics: Dictionary of metric values
            step: Optional step number
            experiment_id: ID of experiment (uses current if None)
        """
        experiment = await self._get_experiment(experiment_id)
        
        # Add to metrics history
        for name, value in metrics.items():
            if name not in experiment.metrics:
                experiment.metrics[name] = []
            experiment.metrics[name].append(value)
            
            # Update best metrics
            if name in experiment.config.metrics_to_track:
                if name not in experiment.best_metrics:
                    experiment.best_metrics[name] = value
                else:
                    # Determine if lower or higher is better
                    if "loss" in name or "error" in name:
                        experiment.best_metrics[name] = min(experiment.best_metrics[name], value)
                    else:
                        experiment.best_metrics[name] = max(experiment.best_metrics[name], value)
        
        # Log to monitoring port
        await self.monitoring_port.log_metrics(metrics, step=step)
        
        # Save metrics to file
        metrics_file = self.experiments_dir / experiment.id / "metrics.json"
        await self.storage_port.save_json(experiment.metrics, metrics_file)
    
    async def log_artifact(
        self,
        artifact_path: Path,
        artifact_type: str,
        experiment_id: Optional[str] = None
    ) -> None:
        """Log an artifact for an experiment.
        
        Args:
            artifact_path: Path to the artifact
            artifact_type: Type of artifact (model, config, etc.)
            experiment_id: ID of experiment (uses current if None)
        """
        experiment = await self._get_experiment(experiment_id)
        
        if experiment.config.save_artifacts:
            # Copy artifact to experiment directory
            artifact_dir = self.experiments_dir / experiment.id / "artifacts"
            await self.storage_port.create_directory(artifact_dir)
            
            dest_path = artifact_dir / f"{artifact_type}_{artifact_path.name}"
            await self.storage_port.copy_file(artifact_path, dest_path)
            
            # Track artifact
            experiment.artifacts[artifact_type] = dest_path
            
            await self.monitoring_port.log_info(
                f"Logged artifact: {artifact_type} -> {dest_path}"
            )
    
    async def complete_experiment(
        self,
        status: ExperimentStatus = ExperimentStatus.COMPLETED,
        experiment_id: Optional[str] = None
    ) -> None:
        """Complete an experiment.
        
        Args:
            status: Final status of the experiment
            experiment_id: ID of experiment (uses current if None)
        """
        experiment = await self._get_experiment(experiment_id)
        
        experiment.status = status
        experiment.completed_at = datetime.now()
        
        # Generate summary
        summary = await self._generate_experiment_summary(experiment)
        
        # Save summary
        summary_path = self.experiments_dir / experiment.id / "summary.json"
        await self.storage_port.save_json(summary, summary_path)
        
        # Save final metadata
        await self._save_experiment(experiment)
        
        # End monitoring run
        await self.monitoring_port.end_run()
        
        await self.monitoring_port.log_info(
            f"Completed experiment: {experiment.id} with status {status.value}"
        )
        
        # Clear current experiment if it matches
        if self.current_experiment and self.current_experiment.id == experiment.id:
            self.current_experiment = None
    
    async def compare_experiments(
        self,
        experiment_ids: List[str],
        metrics: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Compare multiple experiments.
        
        Args:
            experiment_ids: List of experiment IDs to compare
            metrics: Metrics to compare (uses config if None)
            
        Returns:
            Comparison results
        """
        experiments = []
        for exp_id in experiment_ids:
            exp_data = await self._load_experiment(exp_id)
            experiments.append(exp_data)
        
        # Determine metrics to compare
        if not metrics:
            # Use union of all comparison metrics
            metrics = set()
            for exp in experiments:
                metrics.update(exp["config"].get("comparison_metrics", []))
            metrics = list(metrics)
        
        # Create comparison table
        comparison_data = {
            "experiment_id": [],
            "experiment_name": [],
            "status": [],
            "duration_hours": []
        }
        
        for metric in metrics:
            comparison_data[f"best_{metric}"] = []
            comparison_data[f"final_{metric}"] = []
        
        # Add hyperparameters
        all_hyperparams = set()
        for exp in experiments:
            all_hyperparams.update(exp["config"].get("hyperparameters", {}).keys())
        
        for param in all_hyperparams:
            comparison_data[f"param_{param}"] = []
        
        # Fill comparison data
        for exp in experiments:
            comparison_data["experiment_id"].append(exp["id"])
            comparison_data["experiment_name"].append(exp["name"])
            comparison_data["status"].append(exp["status"])
            
            # Calculate duration
            if exp.get("started_at") and exp.get("completed_at"):
                start = datetime.fromisoformat(exp["started_at"])
                end = datetime.fromisoformat(exp["completed_at"])
                duration = (end - start).total_seconds() / 3600
                comparison_data["duration_hours"].append(duration)
            else:
                comparison_data["duration_hours"].append(None)
            
            # Add metrics
            for metric in metrics:
                # Best metric
                best_value = exp.get("best_metrics", {}).get(metric)
                comparison_data[f"best_{metric}"].append(best_value)
                
                # Final metric
                metric_history = exp.get("metrics", {}).get(metric, [])
                final_value = metric_history[-1] if metric_history else None
                comparison_data[f"final_{metric}"].append(final_value)
            
            # Add hyperparameters
            hyperparams = exp["config"].get("hyperparameters", {})
            for param in all_hyperparams:
                comparison_data[f"param_{param}"].append(hyperparams.get(param))
        
        # Create DataFrame for easy analysis
        comparison_df = pd.DataFrame(comparison_data)
        
        # Generate insights
        insights = await self._generate_comparison_insights(comparison_df, metrics)
        
        # Create visualizations if requested
        visualizations = {}
        if any(exp["config"].get("save_visualizations", True) for exp in experiments):
            visualizations = await self._create_comparison_visualizations(
                experiments, metrics
            )
        
        return {
            "comparison_table": comparison_df.to_dict(),
            "insights": insights,
            "visualizations": visualizations,
            "best_experiment": self._find_best_experiment(comparison_df, metrics[0])
        }
    
    async def get_experiment_history(
        self,
        experiment_id: str,
        metric: Optional[str] = None
    ) -> Dict[str, Any]:
        """Get the history of an experiment.
        
        Args:
            experiment_id: ID of the experiment
            metric: Specific metric to retrieve (all if None)
            
        Returns:
            Experiment history
        """
        experiment_data = await self._load_experiment(experiment_id)
        
        history = {
            "experiment_id": experiment_id,
            "experiment_name": experiment_data["name"],
            "metrics": {}
        }
        
        if metric:
            # Get specific metric
            if metric in experiment_data.get("metrics", {}):
                history["metrics"][metric] = experiment_data["metrics"][metric]
        else:
            # Get all metrics
            history["metrics"] = experiment_data.get("metrics", {})
        
        # Add step information if available
        first_metric = next(iter(history["metrics"].values()), [])
        history["total_steps"] = len(first_metric)
        
        return history
    
    async def search_experiments(
        self,
        filters: Optional[Dict[str, Any]] = None,
        sort_by: Optional[str] = None,
        limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Search for experiments based on filters.
        
        Args:
            filters: Filters to apply (status, tags, metrics)
            sort_by: Metric to sort by
            limit: Maximum number of results
            
        Returns:
            List of matching experiments
        """
        experiments = []
        
        # Load all experiments
        if self.experiments_dir.exists():
            for exp_dir in self.experiments_dir.iterdir():
                if exp_dir.is_dir():
                    try:
                        exp_data = await self._load_experiment(exp_dir.name)
                        experiments.append(exp_data)
                    except Exception as e:
                        await self.monitoring_port.log_warning(
                            f"Failed to load experiment {exp_dir.name}: {str(e)}"
                        )
        
        # Apply filters
        if filters:
            filtered = []
            for exp in experiments:
                include = True
                
                # Status filter
                if "status" in filters and exp.get("status") != filters["status"]:
                    include = False
                
                # Tag filters
                if "tags" in filters:
                    exp_tags = exp.get("config", {}).get("tags", {})
                    for tag_key, tag_value in filters["tags"].items():
                        if exp_tags.get(tag_key) != tag_value:
                            include = False
                            break
                
                # Metric filters (e.g., best_eval_loss < 0.5)
                if "metrics" in filters:
                    for metric_filter in filters["metrics"]:
                        metric_name = metric_filter["name"]
                        operator = metric_filter["operator"]
                        value = metric_filter["value"]
                        
                        metric_value = exp.get("best_metrics", {}).get(metric_name)
                        if metric_value is None:
                            include = False
                            break
                        
                        if operator == "<" and not metric_value < value:
                            include = False
                        elif operator == ">" and not metric_value > value:
                            include = False
                        elif operator == "=" and not metric_value == value:
                            include = False
                
                if include:
                    filtered.append(exp)
            
            experiments = filtered
        
        # Sort
        if sort_by:
            # Sort by metric value
            experiments.sort(
                key=lambda x: x.get("best_metrics", {}).get(sort_by, float('inf')),
                reverse="loss" not in sort_by  # Higher is better unless it's a loss
            )
        
        # Limit
        if limit:
            experiments = experiments[:limit]
        
        return experiments
    
    async def _get_experiment(self, experiment_id: Optional[str] = None) -> Experiment:
        """Get experiment by ID or current experiment."""
        if experiment_id:
            # Load specific experiment
            exp_data = await self._load_experiment(experiment_id)
            return self._experiment_from_dict(exp_data)
        elif self.current_experiment:
            return self.current_experiment
        else:
            raise ValueError("No experiment specified and no current experiment")
    
    async def _save_experiment(self, experiment: Experiment) -> None:
        """Save experiment metadata."""
        metadata_path = self.experiments_dir / experiment.id / "experiment.json"
        await self.storage_port.save_json(experiment.to_dict(), metadata_path)
    
    async def _load_experiment(self, experiment_id: str) -> Dict[str, Any]:
        """Load experiment data from disk."""
        metadata_path = self.experiments_dir / experiment_id / "experiment.json"
        if not metadata_path.exists():
            raise ValueError(f"Experiment not found: {experiment_id}")
        
        return await self.storage_port.load_json(metadata_path)
    
    def _experiment_from_dict(self, data: Dict[str, Any]) -> Experiment:
        """Create Experiment object from dictionary."""
        config = ExperimentConfig(**data["config"])
        
        exp = Experiment(
            id=data["id"],
            name=data["name"],
            status=ExperimentStatus(data["status"]),
            config=config,
            created_at=datetime.fromisoformat(data["created_at"])
        )
        
        if data.get("started_at"):
            exp.started_at = datetime.fromisoformat(data["started_at"])
        if data.get("completed_at"):
            exp.completed_at = datetime.fromisoformat(data["completed_at"])
        
        exp.metrics = data.get("metrics", {})
        exp.best_metrics = data.get("best_metrics", {})
        exp.artifacts = {k: Path(v) for k, v in data.get("artifacts", {}).items()}
        exp.system_info = data.get("system_info", {})
        
        return exp
    
    async def _get_system_info(self) -> Dict[str, Any]:
        """Get system information for experiment tracking."""
        import platform
        import os
        
        return {
            "platform": platform.platform(),
            "python_version": platform.python_version(),
            "cpu_count": os.cpu_count(),
            "hostname": platform.node(),
            "user": os.getenv("USER", "unknown")
        }
    
    async def _generate_experiment_summary(self, experiment: Experiment) -> Dict[str, Any]:
        """Generate comprehensive experiment summary."""
        duration = None
        if experiment.started_at and experiment.completed_at:
            duration = (experiment.completed_at - experiment.started_at).total_seconds()
        
        summary = {
            "experiment_id": experiment.id,
            "experiment_name": experiment.name,
            "status": experiment.status.value,
            "duration_seconds": duration,
            "duration_hours": duration / 3600 if duration else None,
            "best_metrics": experiment.best_metrics,
            "final_metrics": {
                name: values[-1] if values else None
                for name, values in experiment.metrics.items()
            },
            "num_metrics_logged": sum(len(values) for values in experiment.metrics.values()),
            "artifacts": list(experiment.artifacts.keys()),
            "hyperparameters": experiment.config.hyperparameters,
            "tags": experiment.config.tags,
            "system_info": experiment.system_info
        }
        
        # Add metric improvements
        improvements = {}
        for name, values in experiment.metrics.items():
            if len(values) > 1:
                first = values[0]
                last = values[-1]
                improvement = last - first
                improvement_pct = (improvement / first * 100) if first != 0 else 0
                improvements[name] = {
                    "absolute": improvement,
                    "percentage": improvement_pct
                }
        summary["improvements"] = improvements
        
        return summary
    
    async def _generate_comparison_insights(
        self,
        comparison_df: pd.DataFrame,
        metrics: List[str]
    ) -> Dict[str, Any]:
        """Generate insights from experiment comparison."""
        insights = {
            "best_performers": {},
            "correlations": {},
            "recommendations": []
        }
        
        # Find best performers for each metric
        for metric in metrics:
            best_col = f"best_{metric}"
            if best_col in comparison_df.columns:
                if "loss" in metric or "error" in metric:
                    best_idx = comparison_df[best_col].idxmin()
                else:
                    best_idx = comparison_df[best_col].idxmax()
                
                if pd.notna(best_idx):
                    insights["best_performers"][metric] = {
                        "experiment_id": comparison_df.loc[best_idx, "experiment_id"],
                        "value": comparison_df.loc[best_idx, best_col]
                    }
        
        # Calculate correlations between hyperparameters and metrics
        param_cols = [col for col in comparison_df.columns if col.startswith("param_")]
        metric_cols = [col for col in comparison_df.columns if col.startswith("best_")]
        
        if param_cols and metric_cols:
            # Convert to numeric where possible
            for col in param_cols + metric_cols:
                comparison_df[col] = pd.to_numeric(comparison_df[col], errors='coerce')
            
            # Calculate correlations
            correlations = comparison_df[param_cols + metric_cols].corr()
            
            # Extract significant correlations
            for param in param_cols:
                param_name = param.replace("param_", "")
                insights["correlations"][param_name] = {}
                
                for metric in metric_cols:
                    metric_name = metric.replace("best_", "")
                    corr_value = correlations.loc[param, metric]
                    
                    if pd.notna(corr_value) and abs(corr_value) > 0.5:
                        insights["correlations"][param_name][metric_name] = corr_value
        
        # Generate recommendations
        if insights["correlations"]:
            for param, metric_corrs in insights["correlations"].items():
                for metric, corr in metric_corrs.items():
                    if abs(corr) > 0.7:
                        direction = "increase" if corr > 0 else "decrease"
                        insights["recommendations"].append(
                            f"Strong correlation ({corr:.2f}) between {param} and {metric}. "
                            f"Consider {direction}ing {param} to improve {metric}."
                        )
        
        return insights
    
    async def _create_comparison_visualizations(
        self,
        experiments: List[Dict[str, Any]],
        metrics: List[str]
    ) -> Dict[str, Any]:
        """Create visualizations for experiment comparison."""
        # This would create actual plots using matplotlib/plotly
        # For now, return metadata about what visualizations would be created
        
        visualizations = {
            "learning_curves": f"learning_curves_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png",
            "metric_comparison": f"metric_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png",
            "hyperparameter_impact": f"hyperparam_impact_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        }
        
        # Save visualization paths
        viz_dir = self.experiments_dir / "visualizations"
        await self.storage_port.create_directory(viz_dir)
        
        return visualizations
    
    def _find_best_experiment(
        self,
        comparison_df: pd.DataFrame,
        primary_metric: str
    ) -> str:
        """Find the best experiment based on primary metric."""
        metric_col = f"best_{primary_metric}"
        
        if metric_col not in comparison_df.columns:
            return comparison_df.iloc[0]["experiment_id"]
        
        if "loss" in primary_metric or "error" in primary_metric:
            best_idx = comparison_df[metric_col].idxmin()
        else:
            best_idx = comparison_df[metric_col].idxmax()
        
        if pd.notna(best_idx):
            return comparison_df.loc[best_idx, "experiment_id"]
        
        return comparison_df.iloc[0]["experiment_id"]