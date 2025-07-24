"""Console/Rich implementation of MonitoringService."""

import sys
from typing import Dict, Any, Optional, List
from datetime import datetime
import json
import uuid

from infrastructure.di import adapter, Scope
from ports.secondary.monitoring import MonitoringService
from ..base import BaseMonitoringAdapter, BaseProgressBar
from .formatters import MetricsFormatter, TableFormatter
# object removed - not defined in ports
from domain.entities.training import TrainingSession


@adapter(MonitoringService, scope=Scope.SINGLETON)
class ConsoleMonitoringAdapter(BaseMonitoringAdapter):
    """Console implementation of the MonitoringService using Rich for beautiful output."""
    
    def __init__(self, verbosity: str = "normal", use_rich: bool = True):
        """Initialize console monitoring adapter.
        
        Args:
            verbosity: Verbosity level (quiet, normal, verbose)
            use_rich: Whether to use Rich for output
        """
        super().__init__()
        self.verbosity = verbosity
        self.use_rich = use_rich and self._try_import_rich()
        self._console = None
        self._live = None
        self._current_table = None
        
        if self.use_rich:
            from rich.console import Console
            from rich.theme import Theme
            
            # Create custom theme
            custom_theme = Theme({
                "info": "cyan",
                "warning": "yellow",
                "error": "bold red",
                "debug": "dim",
                "metric": "green",
                "param": "magenta",
                "progress": "blue",
            })
            
            self._console = Console(theme=custom_theme)
    
    def _try_import_rich(self) -> bool:
        """Try to import Rich library.
        
        Returns:
            True if Rich is available
        """
        try:
            import rich
            return True
        except ImportError:
            return False
    
    def _should_log(self, level: str) -> bool:
        """Check if should log based on verbosity.
        
        Args:
            level: Log level
            
        Returns:
            True if should log
        """
        if self.verbosity == "quiet":
            return level in ["ERROR", "WARNING"]
        elif self.verbosity == "normal":
            return level != "DEBUG"
        else:  # verbose
            return True
    
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
        if not self._should_log("INFO"):
            return
        
        # Add step/epoch to metrics display
        display_metrics = metrics.copy()
        if step is not None:
            display_metrics["step"] = step
        if epoch is not None:
            display_metrics["epoch"] = epoch
        
        # Format metrics line
        metrics_line = MetricsFormatter.format_metrics_line(display_metrics)
        
        if self.use_rich:
            self._console.print(f"[metric]{metrics_line}[/metric]")
        else:
            print(f"METRICS: {metrics_line}")
        
        # Store metrics
        for name, value in metrics.items():
            if isinstance(value, (int, float)):
                if name not in self._run_metrics:
                    self._run_metrics[name] = []
                self._run_metrics[name].append({
                    "value": value,
                    "step": step,
                    "epoch": epoch,
                    "timestamp": datetime.now().isoformat()
                })
    
    def log_hyperparameters(self, params: Dict[str, Any]) -> None:
        """Log hyperparameters.
        
        Args:
            params: Dictionary of hyperparameters
        """
        if not self._should_log("INFO"):
            return
        
        if self.verbosity == "verbose":
            # Show full table in verbose mode
            if self.use_rich:
                from rich.table import Table
                
                table = Table(title="Hyperparameters", show_header=True)
                table.add_column("Parameter", style="param")
                table.add_column("Value", style="white")
                
                for key, value in sorted(params.items()):
                    table.add_row(key, str(value))
                
                self._console.print(table)
            else:
                print("\nHyperparameters:")
                for line in TableFormatter.format_hyperparameters_table(params):
                    print(f"  {line}")
        else:
            # Show compact version
            key_params = {
                k: v for k, v in params.items()
                if k in ["learning_rate", "batch_size", "num_epochs", "optimizer_type"]
            }
            params_line = MetricsFormatter.format_metrics_line(key_params)
            
            if self.use_rich:
                self._console.print(f"[param]Hyperparameters: {params_line}[/param]")
            else:
                print(f"HYPERPARAMETERS: {params_line}")
        
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
        if not self._should_log("INFO"):
            return
        
        artifact_info = f"Artifact saved: {path}"
        if artifact_type:
            artifact_info += f" (type: {artifact_type})"
        
        if self.use_rich:
            self._console.print(f"[dim]{artifact_info}[/dim]")
        else:
            print(artifact_info)
        
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
        run_id = f"{run_name or 'run'}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
        self._active_run_id = run_id
        
        if self.use_rich:
            from rich.panel import Panel
            from rich.text import Text
            
            # Create run info text
            info_lines = [
                f"Run ID: {run_id}",
                f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
            ]
            
            if tags:
                info_lines.append("Tags:")
                for k, v in tags.items():
                    info_lines.append(f"  {k}: {v}")
            
            info_text = Text("\n".join(info_lines))
            
            panel = Panel(
                info_text,
                title=f"Starting Run: {run_name or 'Training'}",
                border_style="green"
            )
            self._console.print(panel)
        else:
            print(f"\n{'='*60}")
            print(f"Starting Run: {run_name or 'Training'}")
            print(f"Run ID: {run_id}")
            print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            if tags:
                print("Tags:")
                for k, v in tags.items():
                    print(f"  {k}: {v}")
            print(f"{'='*60}\n")
        
        return run_id
    
    def end_run(self, status: Optional[str] = None) -> None:
        """End current monitoring run.
        
        Args:
            status: Optional run status
        """
        if not self._active_run_id:
            return
        
        # Calculate run duration
        # Note: In a real implementation, we'd track start time
        
        if self.use_rich:
            from rich.panel import Panel
            
            # Summary statistics
            summary_lines = [
                f"Run ID: {self._active_run_id}",
                f"Status: {status or 'FINISHED'}",
                f"Ended: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
            ]
            
            # Add final metrics if available
            if self._run_metrics:
                summary_lines.append("\nFinal Metrics:")
                for name, history in sorted(self._run_metrics.items())[:5]:
                    if history:
                        final_value = history[-1]["value"]
                        formatted = MetricsFormatter.format_metric_value(final_value)
                        summary_lines.append(f"  {name}: {formatted}")
            
            panel = Panel(
                "\n".join(summary_lines),
                title="Run Completed",
                border_style="green" if status != "FAILED" else "red"
            )
            self._console.print(panel)
        else:
            print(f"\n{'='*60}")
            print(f"Run Completed: {self._active_run_id}")
            print(f"Status: {status or 'FINISHED'}")
            print(f"Ended: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"{'='*60}\n")
        
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
        if not self._should_log(level):
            return
        
        # Format context if provided
        if context and self.verbosity == "verbose":
            context_str = json.dumps(context, indent=2)
            full_message = f"{message}\nContext: {context_str}"
        else:
            full_message = message
        
        if self.use_rich:
            # Map levels to styles
            style_map = {
                "DEBUG": "debug",
                "INFO": "info",
                "WARNING": "warning",
                "ERROR": "error"
            }
            style = style_map.get(level, "white")
            
            self._console.print(f"[{style}]{level}: {full_message}[/{style}]")
        else:
            print(f"{level}: {full_message}")
    
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
        if self.use_rich and self._should_log("INFO"):
            return RichProgressBar(
                total=total,
                description=description,
                unit=unit,
                console=self._console
            )
        else:
            return ConsoleProgressBar(
                total=total,
                description=description,
                unit=unit,
                show=self._should_log("INFO")
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
        # For console adapter, we only have current run metrics
        if run_id and run_id != self._active_run_id:
            return {}
        
        return self._run_metrics
    
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
        # Console adapter doesn't persist run data
        # Return empty comparison
        return {run_id: {"metrics": {}, "params": {}} for run_id in run_ids}
    
    def log_training_session(self, session: TrainingSession) -> None:
        """Log complete training session information.
        
        Args:
            session: Training session object
        """
        # First call parent implementation
        super().log_training_session(session)
        
        if self.use_rich and self._should_log("INFO"):
            from rich.tree import Tree
            from rich.panel import Panel
            
            # Create a tree view of the session
            tree = Tree(f"Training Session: {session.session_id}")
            
            # Configuration branch
            config_branch = tree.add("Configuration")
            config_branch.add(f"Epochs: {session.config.num_epochs}")
            config_branch.add(f"Batch Size: {session.config.batch_size}")
            config_branch.add(f"Learning Rate: {session.config.learning_rate}")
            config_branch.add(f"Optimizer: {session.config.optimizer_type.value}")
            
            # Results branch
            results_branch = tree.add("Results")
            results_branch.add(f"Total Epochs: {session.state.epoch}")
            results_branch.add(f"Total Steps: {session.state.global_step}")
            if session.state.best_metric:
                results_branch.add(f"Best Metric: {MetricsFormatter.format_metric_value(session.state.best_metric)}")
                results_branch.add(f"Best Epoch: {session.state.best_metric_epoch}")
            
            # Checkpoints branch
            if session.checkpoint_paths:
                ckpt_branch = tree.add("Checkpoints")
                for path in session.checkpoint_paths[-3:]:  # Show last 3
                    ckpt_branch.add(path)
            
            self._console.print(Panel(tree, title="Session Summary", border_style="green"))


class ConsoleProgressBar(BaseProgressBar):
    """Simple console progress bar."""
    
    def __init__(self, total: int, description: str, unit: str, show: bool = True):
        """Initialize console progress bar.
        
        Args:
            total: Total number of items
            description: Progress bar description
            unit: Unit name
            show: Whether to show the progress bar
        """
        super().__init__(total, description)
        self.unit = unit
        self.show = show
        self.last_update_time = datetime.now()
        self.update_interval = 0.1  # Update every 100ms
    
    def update(self, n: int = 1) -> None:
        """Update progress.
        
        Args:
            n: Number of items completed
        """
        super().update(n)
        
        if not self.show:
            return
        
        # Check if should update display
        now = datetime.now()
        if (now - self.last_update_time).total_seconds() < self.update_interval:
            return
        
        # Calculate progress
        progress = self.current / self.total
        bar_length = 40
        filled_length = int(bar_length * progress)
        
        # Create progress bar
        bar = "█" * filled_length + "░" * (bar_length - filled_length)
        
        # Format postfix
        postfix_str = ""
        if self.postfix:
            postfix_items = [f"{k}={MetricsFormatter.format_metric_value(v)}" 
                           for k, v in self.postfix.items()]
            postfix_str = " | " + " | ".join(postfix_items)
        
        # Print progress bar
        print(f"\r{self.description}: |{bar}| {self.current}/{self.total} [{progress*100:.0f}%]{postfix_str}", end="")
        sys.stdout.flush()
        
        self.last_update_time = now
    
    def close(self) -> None:
        """Close progress bar."""
        if self.show:
            print()  # New line after progress bar


class RichProgressBar(BaseProgressBar):
    """Rich library progress bar."""
    
    def __init__(self, total: int, description: str, unit: str, console):
        """Initialize Rich progress bar.
        
        Args:
            total: Total number of items
            description: Progress bar description
            unit: Unit name
            console: Rich console instance
        """
        super().__init__(total, description)
        self.unit = unit
        self.console = console
        
        # Import Rich components
        from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeRemainingColumn
        
        # Create progress bar
        self.progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeRemainingColumn(),
            console=console,
            transient=True
        )
        
        # Start progress and add task
        self.progress.start()
        self.task_id = self.progress.add_task(description, total=total)
    
    def update(self, n: int = 1) -> None:
        """Update progress.
        
        Args:
            n: Number of items completed
        """
        super().update(n)
        self.progress.update(self.task_id, advance=n)
    
    def set_description(self, description: str) -> None:
        """Update description.
        
        Args:
            description: New description
        """
        super().set_description(description)
        self.progress.update(self.task_id, description=description)
    
    def set_postfix(self, **kwargs: Any) -> None:
        """Set postfix values.
        
        Args:
            **kwargs: Key-value pairs to display
        """
        super().set_postfix(**kwargs)
        
        # Update description with postfix
        postfix_str = " | ".join([
            f"{k}={MetricsFormatter.format_metric_value(v)}"
            for k, v in kwargs.items()
        ])
        
        if postfix_str:
            full_description = f"{self.description} | {postfix_str}"
            self.progress.update(self.task_id, description=full_description)
    
    def close(self) -> None:
        """Close progress bar."""
        self.progress.stop()