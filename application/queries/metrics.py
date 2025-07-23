"""Get Training Metrics Query - retrieves training metrics and history."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Dict, Any, List, Union
from datetime import datetime
import pandas as pd

from ports.secondary.storage import StorageService as StoragePort
from ports.secondary.monitoring import MonitoringService as MonitoringPort


@dataclass
class MetricsRequest:
    """Request for training metrics."""
    run_id: Optional[str] = None
    experiment_name: Optional[str] = None
    model_path: Optional[Path] = None
    
    # Filtering options
    metric_names: Optional[List[str]] = None
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    
    # Aggregation options
    aggregate_by_epoch: bool = False
    include_system_metrics: bool = False
    include_validation_metrics: bool = True
    
    # Output options
    format: str = "dict"  # "dict", "dataframe", "csv"
    limit: Optional[int] = None  # Limit number of records


@dataclass
class MetricsResponse:
    """Response containing training metrics."""
    success: bool
    error_message: Optional[str] = None
    
    # Metrics data
    metrics: Dict[str, List[float]] = field(default_factory=dict)
    steps: Optional[List[int]] = None
    epochs: Optional[List[int]] = None
    timestamps: Optional[List[datetime]] = None
    
    # Summary statistics
    summary: Dict[str, Dict[str, float]] = field(default_factory=dict)
    
    # Best values
    best_metrics: Dict[str, float] = field(default_factory=dict)
    best_metric_steps: Dict[str, int] = field(default_factory=dict)
    
    # Training info
    total_steps: Optional[int] = None
    total_epochs: Optional[int] = None
    training_time_hours: Optional[float] = None
    
    # System metrics
    system_metrics: Optional[Dict[str, Any]] = None
    
    # Raw data (if requested)
    dataframe: Optional[pd.DataFrame] = None


@dataclass
class GetTrainingMetricsQuery:
    """Query to retrieve training metrics and history.
    
    This query handles read-only operations to get training metrics,
    learning curves, and performance history.
    """
    
    storage_port: StoragePort
    monitoring_port: MonitoringPort
    
    async def execute(self, request: MetricsRequest) -> MetricsResponse:
        """Execute the query to get training metrics.
        
        Args:
            request: Request specifying what metrics to retrieve
            
        Returns:
            Response with training metrics
        """
        try:
            # Determine source of metrics
            if request.run_id:
                # Load from MLflow or tracking system
                metrics_data = await self._load_from_tracking(request.run_id)
            elif request.experiment_name:
                # Load latest run from experiment
                metrics_data = await self._load_from_experiment(request.experiment_name)
            elif request.model_path:
                # Load from model directory
                metrics_data = await self._load_from_model_path(request.model_path)
            else:
                return MetricsResponse(
                    success=False,
                    error_message="Must provide either run_id, experiment_name, or model_path"
                )
            
            if not metrics_data:
                return MetricsResponse(
                    success=False,
                    error_message="No metrics found for the specified criteria"
                )
            
            # Process metrics
            processed_metrics = await self._process_metrics(metrics_data, request)
            
            # Create response
            response = MetricsResponse(success=True)
            
            # Extract metrics arrays
            response.metrics = processed_metrics["metrics"]
            response.steps = processed_metrics.get("steps")
            response.epochs = processed_metrics.get("epochs")
            response.timestamps = processed_metrics.get("timestamps")
            
            # Calculate summary statistics
            response.summary = self._calculate_summary(response.metrics)
            
            # Find best metrics
            best_info = self._find_best_metrics(
                response.metrics,
                response.steps or list(range(len(next(iter(response.metrics.values())))))
            )
            response.best_metrics = best_info["values"]
            response.best_metric_steps = best_info["steps"]
            
            # Add training info
            response.total_steps = processed_metrics.get("total_steps")
            response.total_epochs = processed_metrics.get("total_epochs")
            response.training_time_hours = processed_metrics.get("training_time_hours")
            
            # Add system metrics if requested
            if request.include_system_metrics:
                response.system_metrics = processed_metrics.get("system_metrics")
            
            # Convert to requested format
            if request.format == "dataframe":
                response.dataframe = self._to_dataframe(processed_metrics)
            elif request.format == "csv":
                # Store as string in dataframe field
                df = self._to_dataframe(processed_metrics)
                response.dataframe = df  # Caller can use df.to_csv()
            
            return response
            
        except Exception as e:
            await self.monitoring_port.log_error(f"Failed to get metrics: {str(e)}")
            return MetricsResponse(
                success=False,
                error_message=str(e)
            )
    
    async def _load_from_tracking(self, run_id: str) -> Dict[str, Any]:
        """Load metrics from tracking system (e.g., MLflow)."""
        # This would integrate with MLflow or similar tracking system
        # For now, we'll look for saved metrics in standard locations
        
        # Try output directory structure
        output_dir = Path("output")
        for run_dir in output_dir.glob("run_*"):
            run_info_file = run_dir / "run_info.json"
            if run_info_file.exists():
                run_info = await self.storage_port.load_json(run_info_file)
                if run_info.get("run_id") == run_id:
                    return await self._load_metrics_from_dir(run_dir)
        
        return {}
    
    async def _load_from_experiment(self, experiment_name: str) -> Dict[str, Any]:
        """Load metrics from the latest run in an experiment."""
        # Find runs for this experiment
        output_dir = Path("output")
        experiment_runs = []
        
        for run_dir in output_dir.glob("run_*"):
            run_info_file = run_dir / "run_info.json"
            if run_info_file.exists():
                run_info = await self.storage_port.load_json(run_info_file)
                if run_info.get("experiment_name") == experiment_name:
                    experiment_runs.append((run_dir, run_info.get("start_time")))
        
        if not experiment_runs:
            return {}
        
        # Get the latest run
        latest_run = max(experiment_runs, key=lambda x: x[1])
        return await self._load_metrics_from_dir(latest_run[0])
    
    async def _load_from_model_path(self, model_path: Path) -> Dict[str, Any]:
        """Load metrics from model directory."""
        metrics_data = {}
        
        # Load training history
        history_file = model_path / "training_history.json"
        if history_file.exists():
            history = await self.storage_port.load_json(history_file)
            metrics_data["train_history"] = history.get("train_history", [])
            metrics_data["val_history"] = history.get("val_history", [])
            metrics_data["config"] = history.get("config", {})
        
        # Load final metrics
        metrics_file = model_path / "metrics.json"
        if metrics_file.exists():
            final_metrics = await self.storage_port.load_json(metrics_file)
            metrics_data["final_metrics"] = final_metrics
        
        # Load system metrics if available
        system_metrics_file = model_path / "system_metrics.json"
        if system_metrics_file.exists():
            metrics_data["system_metrics"] = await self.storage_port.load_json(system_metrics_file)
        
        return metrics_data
    
    async def _load_metrics_from_dir(self, run_dir: Path) -> Dict[str, Any]:
        """Load all metrics from a run directory."""
        metrics_data = {}
        
        # Standard files to check
        files_to_load = {
            "training_history.json": ["train_history", "val_history"],
            "metrics.json": ["final_metrics"],
            "system_metrics.json": ["system_metrics"],
            "run_info.json": ["run_info"]
        }
        
        for filename, keys in files_to_load.items():
            file_path = run_dir / filename
            if file_path.exists():
                data = await self.storage_port.load_json(file_path)
                for key in keys:
                    if key in data:
                        metrics_data[key] = data[key]
                    else:
                        # The file itself might be the data
                        metrics_data[key] = data
        
        return metrics_data
    
    async def _process_metrics(
        self,
        metrics_data: Dict[str, Any],
        request: MetricsRequest
    ) -> Dict[str, Any]:
        """Process raw metrics data according to request parameters."""
        processed = {
            "metrics": {},
            "steps": [],
            "epochs": [],
            "timestamps": []
        }
        
        # Extract training history
        train_history = metrics_data.get("train_history", [])
        val_history = metrics_data.get("val_history", [])
        
        # Combine histories
        all_metrics = {}
        
        # Process training metrics
        for record in train_history:
            step = record.get("step", len(processed["steps"]))
            processed["steps"].append(step)
            
            if "epoch" in record:
                processed["epochs"].append(record["epoch"])
            
            if "timestamp" in record:
                processed["timestamps"].append(
                    datetime.fromisoformat(record["timestamp"])
                )
            
            # Extract metrics
            for key, value in record.items():
                if key not in ["step", "epoch", "timestamp"]:
                    if key not in all_metrics:
                        all_metrics[key] = []
                    all_metrics[key].append(value)
        
        # Process validation metrics
        if request.include_validation_metrics:
            for i, record in enumerate(val_history):
                for key, value in record.items():
                    if key not in ["step", "epoch", "timestamp"]:
                        val_key = f"val_{key}"
                        if val_key not in all_metrics:
                            all_metrics[val_key] = [None] * len(train_history)
                        
                        # Find corresponding training step
                        if i < len(all_metrics[val_key]):
                            all_metrics[val_key][i] = value
        
        # Filter metrics if requested
        if request.metric_names:
            all_metrics = {
                k: v for k, v in all_metrics.items()
                if k in request.metric_names
            }
        
        # Apply time filtering
        if request.start_time or request.end_time:
            time_mask = self._create_time_mask(
                processed["timestamps"],
                request.start_time,
                request.end_time
            )
            
            # Filter all arrays
            processed["steps"] = [s for s, m in zip(processed["steps"], time_mask) if m]
            processed["epochs"] = [e for e, m in zip(processed["epochs"], time_mask) if m]
            processed["timestamps"] = [t for t, m in zip(processed["timestamps"], time_mask) if m]
            
            for key in all_metrics:
                all_metrics[key] = [v for v, m in zip(all_metrics[key], time_mask) if m]
        
        # Aggregate by epoch if requested
        if request.aggregate_by_epoch and processed["epochs"]:
            all_metrics, processed = self._aggregate_by_epoch(all_metrics, processed)
        
        # Apply limit
        if request.limit and processed["steps"]:
            # Sample evenly across the range
            indices = self._sample_indices(len(processed["steps"]), request.limit)
            
            processed["steps"] = [processed["steps"][i] for i in indices]
            if processed["epochs"]:
                processed["epochs"] = [processed["epochs"][i] for i in indices]
            if processed["timestamps"]:
                processed["timestamps"] = [processed["timestamps"][i] for i in indices]
            
            for key in all_metrics:
                all_metrics[key] = [all_metrics[key][i] for i in indices]
        
        processed["metrics"] = all_metrics
        
        # Add summary info
        if metrics_data.get("run_info"):
            run_info = metrics_data["run_info"]
            start_time = datetime.fromisoformat(run_info.get("start_time"))
            end_time = datetime.fromisoformat(run_info.get("end_time"))
            processed["training_time_hours"] = (end_time - start_time).total_seconds() / 3600
        
        processed["total_steps"] = max(processed["steps"]) if processed["steps"] else None
        processed["total_epochs"] = max(processed["epochs"]) if processed["epochs"] else None
        
        # Add system metrics
        if request.include_system_metrics and "system_metrics" in metrics_data:
            processed["system_metrics"] = metrics_data["system_metrics"]
        
        return processed
    
    def _create_time_mask(
        self,
        timestamps: List[datetime],
        start_time: Optional[datetime],
        end_time: Optional[datetime]
    ) -> List[bool]:
        """Create boolean mask for time filtering."""
        mask = []
        for ts in timestamps:
            include = True
            if start_time and ts < start_time:
                include = False
            if end_time and ts > end_time:
                include = False
            mask.append(include)
        return mask
    
    def _aggregate_by_epoch(
        self,
        metrics: Dict[str, List[float]],
        processed: Dict[str, Any]
    ) -> tuple[Dict[str, List[float]], Dict[str, Any]]:
        """Aggregate metrics by epoch."""
        if not processed["epochs"]:
            return metrics, processed
        
        # Group by epoch
        epoch_groups = {}
        for i, epoch in enumerate(processed["epochs"]):
            if epoch not in epoch_groups:
                epoch_groups[epoch] = []
            epoch_groups[epoch].append(i)
        
        # Aggregate metrics
        aggregated_metrics = {}
        for metric_name, values in metrics.items():
            aggregated_metrics[metric_name] = []
            for epoch in sorted(epoch_groups.keys()):
                indices = epoch_groups[epoch]
                epoch_values = [values[i] for i in indices if i < len(values)]
                if epoch_values:
                    # Use mean for aggregation
                    aggregated_metrics[metric_name].append(
                        sum(epoch_values) / len(epoch_values)
                    )
        
        # Update processed data
        new_processed = {
            "steps": list(range(len(epoch_groups))),
            "epochs": sorted(epoch_groups.keys()),
            "timestamps": []  # Aggregated data doesn't have specific timestamps
        }
        
        return aggregated_metrics, new_processed
    
    def _sample_indices(self, total: int, limit: int) -> List[int]:
        """Sample indices evenly across the range."""
        if limit >= total:
            return list(range(total))
        
        # Sample evenly
        step = total / limit
        indices = [int(i * step) for i in range(limit)]
        return indices
    
    def _calculate_summary(self, metrics: Dict[str, List[float]]) -> Dict[str, Dict[str, float]]:
        """Calculate summary statistics for each metric."""
        summary = {}
        
        for metric_name, values in metrics.items():
            if not values:
                continue
            
            # Filter out None values
            valid_values = [v for v in values if v is not None]
            if not valid_values:
                continue
            
            summary[metric_name] = {
                "mean": sum(valid_values) / len(valid_values),
                "min": min(valid_values),
                "max": max(valid_values),
                "final": valid_values[-1],
                "std": self._calculate_std(valid_values)
            }
            
            # Add trend (positive = improving)
            if len(valid_values) > 1:
                trend = valid_values[-1] - valid_values[0]
                summary[metric_name]["trend"] = trend
                summary[metric_name]["trend_percent"] = (trend / valid_values[0]) * 100
        
        return summary
    
    def _calculate_std(self, values: List[float]) -> float:
        """Calculate standard deviation."""
        if len(values) < 2:
            return 0.0
        
        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / (len(values) - 1)
        return variance ** 0.5
    
    def _find_best_metrics(
        self,
        metrics: Dict[str, List[float]],
        steps: List[int]
    ) -> Dict[str, Dict[str, Union[float, int]]]:
        """Find best value and step for each metric."""
        best_values = {}
        best_steps = {}
        
        for metric_name, values in metrics.items():
            if not values:
                continue
            
            # Filter out None values with their steps
            valid_pairs = [(v, s) for v, s in zip(values, steps) if v is not None]
            if not valid_pairs:
                continue
            
            # Determine if lower or higher is better
            if "loss" in metric_name.lower() or "error" in metric_name.lower():
                # Lower is better
                best_value, best_step = min(valid_pairs, key=lambda x: x[0])
            else:
                # Higher is better (accuracy, f1, etc.)
                best_value, best_step = max(valid_pairs, key=lambda x: x[0])
            
            best_values[metric_name] = best_value
            best_steps[metric_name] = best_step
        
        return {"values": best_values, "steps": best_steps}
    
    def _to_dataframe(self, processed_metrics: Dict[str, Any]) -> pd.DataFrame:
        """Convert metrics to pandas DataFrame."""
        data = {}
        
        # Add index columns
        if processed_metrics.get("steps"):
            data["step"] = processed_metrics["steps"]
        if processed_metrics.get("epochs"):
            data["epoch"] = processed_metrics["epochs"]
        if processed_metrics.get("timestamps"):
            data["timestamp"] = processed_metrics["timestamps"]
        
        # Add metric columns
        for metric_name, values in processed_metrics["metrics"].items():
            data[metric_name] = values
        
        return pd.DataFrame(data)