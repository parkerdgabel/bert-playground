"""TensorBoard implementation of MonitoringService."""

import os
from typing import Dict, Any, Optional, List
from datetime import datetime
import json

from infrastructure.di import adapter, Scope
from application.ports.secondary.monitoring import MonitoringService
from ..base import BaseMonitoringAdapter, BaseProgressBar
from .writer import TensorBoardWriter
# object removed - not defined in ports
from domain.entities.training import TrainingSession


@adapter(MonitoringService, scope=Scope.SINGLETON)
class TensorBoardMonitoringAdapter(BaseMonitoringAdapter):
    """TensorBoard implementation of the MonitoringService."""
    
    def __init__(self, log_dir: str = "./runs", tag_prefix: str = ""):
        """Initialize TensorBoard monitoring adapter.
        
        Args:
            log_dir: Base directory for TensorBoard logs
            tag_prefix: Prefix for all tags
        """
        super().__init__()
        self.log_dir = log_dir
        self.tag_prefix = tag_prefix
        self._writer = None
        self._run_dir = None
        self._metadata_file = None
    
    def _get_tag(self, tag: str) -> str:
        """Get tag with prefix.
        
        Args:
            tag: Original tag
            
        Returns:
            Tag with prefix
        """
        if self.tag_prefix:
            return f"{self.tag_prefix}/{tag}"
        return tag
    
    def log_metrics(
        self,
        metrics: Dict[str, Any],
        step: Optional[int] = None,
        epoch: Optional[int] = None,
    ) -> None:
        """Log metrics.
        
        Args:
            metrics: Dictionary of metric name -> value
            step: Optional global step
            epoch: Optional epoch number
        """
        if not self._writer:
            return
        
        # Log each metric
        for name, value in metrics.items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                tag = self._get_tag(name)
                self._writer.add_scalar(tag, float(value), global_step=step)
        
        # Log epoch as a separate metric if provided
        if epoch is not None:
            self._writer.add_scalar(self._get_tag("epoch"), epoch, global_step=step)
        
        # Store in internal tracking
        for name, value in metrics.items():
            if isinstance(value, (int, float)):
                if name not in self._run_metrics:
                    self._run_metrics[name] = []
                self._run_metrics[name].append({
                    "value": float(value),
                    "step": step,
                    "epoch": epoch,
                    "timestamp": datetime.now().isoformat()
                })
        
        # Flush periodically
        if step and step % 100 == 0:
            self._writer.flush()
    
    def log_hyperparameters(self, params: Dict[str, Any]) -> None:
        """Log hyperparameters.
        
        Args:
            params: Dictionary of hyperparameters
        """
        if not self._writer:
            return
        
        # Convert params to string representation for TensorBoard
        hparam_dict = {}
        metric_dict = {}  # Empty for now, will be populated with final metrics
        
        for k, v in params.items():
            if isinstance(v, (int, float, str, bool)):
                hparam_dict[k] = v
            else:
                hparam_dict[k] = str(v)
        
        # Log hyperparameters
        self._writer.add_hparams(hparam_dict, metric_dict)
        
        # Also save to metadata file
        if self._metadata_file:
            metadata = {
                "hyperparameters": params,
                "timestamp": datetime.now().isoformat()
            }
            with open(self._metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
        
        self._run_params.update(params)
    
    def log_artifact(
        self,
        path: str,
        artifact_type: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Log an artifact (file, model, etc.).
        
        Args:
            path: Path to artifact
            artifact_type: Type of artifact
            metadata: Optional metadata
        """
        if not self._run_dir:
            return
        
        # Create artifacts directory
        artifacts_dir = os.path.join(self._run_dir, "artifacts")
        os.makedirs(artifacts_dir, exist_ok=True)
        
        # Copy artifact to run directory
        import shutil
        
        if os.path.isfile(path):
            dest_path = os.path.join(artifacts_dir, os.path.basename(path))
            shutil.copy2(path, dest_path)
        elif os.path.isdir(path):
            dest_path = os.path.join(artifacts_dir, os.path.basename(path))
            shutil.copytree(path, dest_path, dirs_exist_ok=True)
        else:
            return
        
        # Save metadata
        if metadata or artifact_type:
            meta_file = dest_path + ".metadata.json"
            meta_data = {
                "source_path": path,
                "artifact_type": artifact_type,
                "timestamp": datetime.now().isoformat(),
                "metadata": metadata or {}
            }
            with open(meta_file, 'w') as f:
                json.dump(meta_data, f, indent=2)
        
        self._run_artifacts.append(path)
    
    def start_run(
        self,
        run_name: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None,
    ) -> str:
        """Start a new monitoring run.
        
        Args:
            run_name: Optional run name
            tags: Optional tags for the run
            
        Returns:
            Run ID
        """
        # Generate run ID
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_id = f"{run_name or 'run'}_{timestamp}"
        
        # Create run directory
        self._run_dir = os.path.join(self.log_dir, run_id)
        os.makedirs(self._run_dir, exist_ok=True)
        
        # Create TensorBoard writer
        self._writer = TensorBoardWriter(self._run_dir)
        
        # Save run metadata
        self._metadata_file = os.path.join(self._run_dir, "metadata.json")
        metadata = {
            "run_id": run_id,
            "run_name": run_name,
            "tags": tags or {},
            "start_time": datetime.now().isoformat(),
            "tensorboard_enabled": self._writer.enabled
        }
        with open(self._metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Log tags as text
        if tags:
            tags_text = "\n".join([f"{k}: {v}" for k, v in tags.items()])
            self._writer.add_text("tags", tags_text)
        
        self._active_run_id = run_id
        return run_id
    
    def end_run(self, status: Optional[str] = None) -> None:
        """End current monitoring run.
        
        Args:
            status: Optional run status
        """
        if not self._writer:
            return
        
        # Update metadata with end time and status
        if self._metadata_file and os.path.exists(self._metadata_file):
            with open(self._metadata_file, 'r') as f:
                metadata = json.load(f)
            
            metadata["end_time"] = datetime.now().isoformat()
            metadata["status"] = status or "FINISHED"
            metadata["final_metrics"] = {
                name: history[-1]["value"] if history else None
                for name, history in self._run_metrics.items()
            }
            
            with open(self._metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
        
        # Close writer
        self._writer.close()
        self._writer = None
        self._run_dir = None
        self._metadata_file = None
        self._active_run_id = None
    
    def log_message(
        self,
        message: str,
        level: str = "INFO",
        context: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Log a message.
        
        Args:
            message: Message to log
            level: Log level (DEBUG, INFO, WARNING, ERROR)
            context: Optional context information
        """
        if not self._writer:
            return
        
        # Format message with context
        formatted_message = f"[{level}] {message}"
        if context:
            context_str = ", ".join([f"{k}={v}" for k, v in context.items()])
            formatted_message += f" ({context_str})"
        
        # Log as text
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self._writer.add_text(
            f"logs/{level.lower()}",
            f"{timestamp}: {formatted_message}"
        )
        
        # Also write to a log file
        if self._run_dir:
            log_file = os.path.join(self._run_dir, "messages.log")
            with open(log_file, 'a') as f:
                f.write(f"{timestamp} [{level}] {message}\n")
                if context:
                    f.write(f"  Context: {json.dumps(context)}\n")
    
    def create_progress_bar(
        self,
        total: int,
        description: str,
        unit: str = "it",
    ) -> object:
        """Create a progress bar.
        
        Args:
            total: Total number of items
            description: Progress bar description
            unit: Unit name
            
        Returns:
            Progress bar instance
        """
        return TensorBoardProgressBar(
            total=total,
            description=description,
            unit=unit,
            adapter=self
        )
    
    def get_run_metrics(
        self,
        run_id: Optional[str] = None,
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Get metrics for a run.
        
        Args:
            run_id: Optional run ID (current run if None)
            
        Returns:
            Dictionary of metric histories
        """
        if run_id is None:
            # Return current run metrics
            return self._run_metrics
        
        # Load metrics from saved run
        run_dir = os.path.join(self.log_dir, run_id)
        metrics_file = os.path.join(run_dir, "metrics.jsonl")
        
        if not os.path.exists(metrics_file):
            return {}
        
        metrics = {}
        with open(metrics_file, 'r') as f:
            for line in f:
                data = json.loads(line)
                if data.get("type") == "scalar":
                    tag = data["tag"]
                    if tag not in metrics:
                        metrics[tag] = []
                    metrics[tag].append({
                        "value": data["value"],
                        "step": data.get("step"),
                        "timestamp": datetime.fromtimestamp(data["walltime"]).isoformat()
                    })
        
        return metrics
    
    def compare_runs(
        self,
        run_ids: List[str],
        metrics: Optional[List[str]] = None,
    ) -> Dict[str, Dict[str, Any]]:
        """Compare multiple runs.
        
        Args:
            run_ids: List of run IDs to compare
            metrics: Optional list of metrics to compare
            
        Returns:
            Comparison results
        """
        comparison = {}
        
        for run_id in run_ids:
            run_dir = os.path.join(self.log_dir, run_id)
            metadata_file = os.path.join(run_dir, "metadata.json")
            
            if not os.path.exists(metadata_file):
                continue
            
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
            
            run_data = {
                "metadata": metadata,
                "metrics": {}
            }
            
            # Get final metrics
            if "final_metrics" in metadata:
                if metrics:
                    run_data["metrics"] = {
                        m: metadata["final_metrics"].get(m)
                        for m in metrics
                        if m in metadata["final_metrics"]
                    }
                else:
                    run_data["metrics"] = metadata["final_metrics"]
            
            comparison[run_id] = run_data
        
        return comparison
    
    def log_training_session(self, session: TrainingSession) -> None:
        """Log complete training session information.
        
        Args:
            session: Training session object
        """
        # First call parent implementation
        super().log_training_session(session)
        
        # Log training curves as histograms
        if session.state.train_loss_history:
            self._writer.add_histogram(
                self._get_tag("training/loss_distribution"),
                session.state.train_loss_history
            )
        
        if session.state.learning_rate_history:
            self._writer.add_histogram(
                self._get_tag("training/lr_distribution"),
                session.state.learning_rate_history
            )
        
        # Create summary text
        summary = f"""
        Training Session Summary
        =======================
        Session ID: {session.session_id}
        Total Epochs: {session.state.epoch}
        Total Steps: {session.state.global_step}
        Best Metric: {session.state.best_metric}
        Best Metric Epoch: {session.state.best_metric_epoch}
        
        Configuration:
        - Epochs: {session.config.num_epochs}
        - Batch Size: {session.config.batch_size}
        - Learning Rate: {session.config.learning_rate}
        - Optimizer: {session.config.optimizer_type.value}
        
        Checkpoints:
        {chr(10).join(session.checkpoint_paths)}
        """
        
        self._writer.add_text("summary", summary)


class TensorBoardProgressBar(BaseProgressBar):
    """Progress bar that logs to TensorBoard."""
    
    def __init__(self, total: int, description: str, unit: str, adapter: TensorBoardMonitoringAdapter):
        """Initialize TensorBoard progress bar.
        
        Args:
            total: Total number of items
            description: Progress bar description
            unit: Unit name
            adapter: TensorBoard adapter instance
        """
        super().__init__(total, description)
        self.unit = unit
        self.adapter = adapter
        self.start_time = datetime.now()
    
    def update(self, n: int = 1) -> None:
        """Update progress.
        
        Args:
            n: Number of items completed
        """
        super().update(n)
        
        # Log progress percentage
        progress_pct = (self.current / self.total) * 100
        self.adapter.log_metrics({
            f"progress/{self.description}": progress_pct
        })
    
    def set_postfix(self, **kwargs: Any) -> None:
        """Set postfix values and log them.
        
        Args:
            **kwargs: Key-value pairs to display
        """
        super().set_postfix(**kwargs)
        
        # Log postfix values as metrics
        metrics = {}
        for k, v in kwargs.items():
            if isinstance(v, (int, float)):
                metrics[f"{self.description}/{k}"] = v
        
        if metrics:
            self.adapter.log_metrics(metrics)
    
    def close(self) -> None:
        """Close progress bar and log final stats."""
        elapsed = (datetime.now() - self.start_time).total_seconds()
        
        # Log final statistics
        self.adapter.log_metrics({
            f"{self.description}/total_time_seconds": elapsed,
            f"{self.description}/items_per_second": self.current / elapsed if elapsed > 0 else 0
        })