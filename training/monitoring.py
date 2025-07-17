"""Comprehensive monitoring system for MLX training.

This module provides unified monitoring capabilities including MLflow tracking,
real-time metrics, rich console output, and performance visualization.
"""

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

import mlflow
import mlx.core as mx
import numpy as np
from loguru import logger
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from training.rich_display_manager import RichDisplayManager

from training.config import TrainingConfig
from training.memory_manager import AppleSiliconMemoryManager
from training.performance_profiler import AppleSiliconProfiler
from utils.mlflow_central import MLflowCentral


@dataclass
class MetricTracker:
    """Tracks a single metric over time."""

    name: str
    values: list[float] = field(default_factory=list)
    steps: list[int] = field(default_factory=list)
    timestamps: list[float] = field(default_factory=list)
    best_value: float | None = None
    best_step: int | None = None
    higher_is_better: bool = True

    def update(self, value: float, step: int) -> bool:
        """Update metric and return True if new best."""
        timestamp = time.time()
        self.values.append(value)
        self.steps.append(step)
        self.timestamps.append(timestamp)

        # Update best value
        is_new_best = False
        if self.best_value is None or self.higher_is_better and value > self.best_value or not self.higher_is_better and value < self.best_value:
            self.best_value = value
            self.best_step = step
            is_new_best = True

        return is_new_best

    def get_recent_average(self, n: int = 5) -> float:
        """Get average of last n values."""
        if not self.values:
            return 0.0
        return np.mean(self.values[-n:])

    def get_trend(self, n: int = 10) -> str:
        """Get trend direction for last n values."""
        if len(self.values) < 2:
            return "→"

        recent_values = self.values[-n:]
        if len(recent_values) < 2:
            return "→"

        # Simple trend calculation
        diff = recent_values[-1] - recent_values[0]
        if abs(diff) < 0.001:
            return "→"
        elif diff > 0:
            return "↗" if self.higher_is_better else "↘"
        else:
            return "↘" if self.higher_is_better else "↗"


@dataclass
class TrainingMetrics:
    """Container for all training metrics."""

    # Core metrics
    train_loss: MetricTracker = field(
        default_factory=lambda: MetricTracker("train_loss", higher_is_better=False)
    )
    train_accuracy: MetricTracker = field(
        default_factory=lambda: MetricTracker("train_accuracy")
    )
    val_loss: MetricTracker = field(
        default_factory=lambda: MetricTracker("val_loss", higher_is_better=False)
    )
    val_accuracy: MetricTracker = field(
        default_factory=lambda: MetricTracker("val_accuracy")
    )

    # Performance metrics
    learning_rate: MetricTracker = field(
        default_factory=lambda: MetricTracker("learning_rate", higher_is_better=False)
    )
    step_time: MetricTracker = field(
        default_factory=lambda: MetricTracker("step_time", higher_is_better=False)
    )
    throughput: MetricTracker = field(
        default_factory=lambda: MetricTracker("throughput")
    )
    memory_usage: MetricTracker = field(
        default_factory=lambda: MetricTracker("memory_usage", higher_is_better=False)
    )

    def get_all_trackers(self) -> dict[str, MetricTracker]:
        """Get all metric trackers as a dictionary."""
        return {
            "train_loss": self.train_loss,
            "train_accuracy": self.train_accuracy,
            "val_loss": self.val_loss,
            "val_accuracy": self.val_accuracy,
            "learning_rate": self.learning_rate,
            "step_time": self.step_time,
            "throughput": self.throughput,
            "memory_usage": self.memory_usage,
        }


class MLflowTracker:
    """MLflow integration for experiment tracking."""

    def __init__(self, config: TrainingConfig):
        """Initialize MLflow tracking.

        Args:
            config: Training configuration
        """
        self.config = config
        self.mlflow_central = MLflowCentral()
        self.run_id: str | None = None
        self.experiment_id: str | None = None

        # Initialize MLflow if enabled
        if config.monitoring.enable_mlflow:
            self._initialize_mlflow()

    def _initialize_mlflow(self) -> None:
        """Initialize MLflow experiment tracking with enhanced error handling."""
        try:
            # Initialize central MLflow configuration with validation
            self.mlflow_central.initialize(
                experiment_name=self.config.experiment_name or "mlx_training_v2"
            )

            # Validate connection before proceeding
            connection_status = self.mlflow_central.validate_connection()
            if connection_status["status"] != "CONNECTED":
                raise Exception(f"MLflow connection failed: {connection_status['message']}")

            # Start run with tags
            tags = {
                "model_type": getattr(self.config, "model_type", "unknown"),
                "optimization_level": self.config.optimization_level.value,
                "batch_size": str(self.config.batch_size),
                "learning_rate": str(self.config.learning_rate),
                "optimizer": self.config.optimizer.value,
                "apple_silicon": str(mx.metal.is_available()),
            }

            # Add git information if available
            try:
                import subprocess

                git_commit = (
                    subprocess.check_output(
                        ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
                    )
                    .decode()
                    .strip()
                )
                tags["git_commit"] = git_commit[:8]
            except Exception:
                logger.debug("Git information not available")

            self.run = mlflow.start_run(tags=tags)
            self.run_id = self.run.info.run_id

            # Log configuration parameters
            self._log_config_params()

            logger.info(f"Started MLflow run: {self.run_id}")

        except Exception as e:
            logger.warning(f"Failed to initialize MLflow: {e}")
            self.config.monitoring.enable_mlflow = False
            
            # Provide specific error handling and suggestions
            error_msg = str(e).lower()
            
            if "sqlite" in error_msg or "database" in error_msg:
                logger.info("MLflow database issue detected:")
                logger.info("  • Ensure the mlruns directory is writable")
                logger.info("  • Check if database file is corrupted")
                logger.info("  • Run 'uv run python mlx_bert_cli.py mlflow-health' for diagnosis")
                
            elif "permission" in error_msg:
                logger.info("MLflow permission issue detected:")
                logger.info("  • Check file permissions for the MLflow directory")
                logger.info("  • Ensure write access to mlruns and artifact directories")
                logger.info("  • Try running with appropriate user privileges")
                
            elif "connection" in error_msg or "network" in error_msg:
                logger.info("MLflow connection issue detected:")
                logger.info("  • Check your network connection or MLflow server status")
                logger.info("  • Verify tracking URI is accessible")
                logger.info("  • Check if MLflow server is running")
                
            elif "configuration" in error_msg:
                logger.info("MLflow configuration issue detected:")
                logger.info("  • Check MLflow configuration parameters")
                logger.info("  • Verify tracking URI and artifact root settings")
                logger.info("  • Run 'uv run python mlx_bert_cli.py mlflow-health' for diagnosis")
                
            else:
                logger.info("General MLflow issue detected:")
                logger.info("  • Check MLflow installation and version")
                logger.info("  • Verify all dependencies are installed")
                logger.info("  • Run 'uv run python mlx_bert_cli.py mlflow-health' for diagnosis")
                
            logger.info("Training will continue without MLflow tracking")

    def _log_config_params(self) -> None:
        """Log training configuration as MLflow parameters."""
        if not self.config.monitoring.enable_mlflow:
            return

        try:
            # Core training parameters
            mlflow.log_param("epochs", self.config.epochs)
            mlflow.log_param("batch_size", self.config.batch_size)
            mlflow.log_param("learning_rate", self.config.learning_rate)
            mlflow.log_param("optimizer", self.config.optimizer.value)
            mlflow.log_param("optimization_level", self.config.optimization_level.value)

            # Memory configuration
            mlflow.log_param(
                "dynamic_batch_sizing", self.config.memory.dynamic_batch_sizing
            )
            mlflow.log_param(
                "unified_memory_fraction", self.config.memory.unified_memory_fraction
            )
            mlflow.log_param(
                "gradient_accumulation_steps",
                self.config.mlx_optimization.gradient_accumulation_steps,
            )

            # MLX optimization
            mlflow.log_param("enable_jit", self.config.mlx_optimization.enable_jit)
            mlflow.log_param(
                "mixed_precision", self.config.mlx_optimization.mixed_precision
            )
            mlflow.log_param(
                "memory_pool_size", self.config.mlx_optimization.memory_pool_size
            )

            # Advanced features
            mlflow.log_param("label_smoothing", self.config.advanced.label_smoothing)
            mlflow.log_param("weight_decay", self.config.advanced.weight_decay)
            mlflow.log_param(
                "gradient_clipping", self.config.advanced.gradient_clipping
            )
            mlflow.log_param("warmup_steps", self.config.advanced.warmup_steps)

        except Exception as e:
            logger.warning(f"Failed to log config parameters: {e}")

    def log_metrics(self, metrics: dict[str, float], step: int) -> None:
        """Log metrics to MLflow.

        Args:
            metrics: Dictionary of metric names to values
            step: Training step
        """
        if not self.config.monitoring.enable_mlflow:
            return

        try:
            for name, value in metrics.items():
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    mlflow.log_metric(name, value, step=step)
                else:
                    logger.debug(f"Skipping non-numeric metric {name}: {value}")
        except Exception as e:
            logger.warning(f"Failed to log metrics to MLflow: {e}")
            # Optionally disable MLflow if it keeps failing
            if "Connection" in str(e) or "HTTP" in str(e):
                logger.warning("MLflow connection issues detected, disabling MLflow tracking")
                self.config.monitoring.enable_mlflow = False

    def log_artifacts(self, artifacts: dict[str, str]) -> None:
        """Log artifacts to MLflow.

        Args:
            artifacts: Dictionary of artifact names to file paths
        """
        if not self.config.monitoring.enable_mlflow:
            return

        try:
            for name, path in artifacts.items():
                if Path(path).exists():
                    mlflow.log_artifact(path, artifact_path=name)
        except Exception as e:
            logger.warning(f"Failed to log artifacts: {e}")

    def end_run(self, status: str = "FINISHED") -> None:
        """End MLflow run.

        Args:
            status: Run status (FINISHED, FAILED, KILLED)
        """
        if not self.config.monitoring.enable_mlflow:
            return

        try:
            mlflow.end_run(status=status)
            logger.info(f"Ended MLflow run: {self.run_id} with status: {status}")
        except Exception as e:
            logger.warning(f"Failed to end MLflow run: {e}")


class RichConsoleMonitor:
    """Rich console monitoring for real-time training progress."""

    def __init__(self, config: TrainingConfig, display_manager: Optional[RichDisplayManager] = None):
        """Initialize rich console monitor.

        Args:
            config: Training configuration
            display_manager: Optional shared display manager
        """
        self.config = config
        self.console = Console()
        self.display_manager = display_manager
        self.training_task_id: Optional[str] = None
        self.epoch_task_id: Optional[str] = None

        # Training state
        self.start_time = time.time()
        self.last_update_time = time.time()

        # Display configuration
        self.enabled = config.monitoring.enable_rich_console
        self.update_frequency = max(
            1, config.monitoring.log_frequency // 4
        )  # Update 4x per log

    def start_training(self, total_epochs: int, steps_per_epoch: int) -> None:
        """Start training progress display.

        Args:
            total_epochs: Total number of epochs
            steps_per_epoch: Steps per epoch
        """
        if not self.enabled:
            return

        # Use display manager if available
        if self.display_manager:
            self.training_task_id = self.display_manager.create_progress_task(
                "training", "Overall Progress", total_epochs
            )
            self.epoch_task_id = self.display_manager.create_progress_task(
                "epoch", "Current Epoch", steps_per_epoch
            )
        else:
            # Fallback to console output
            self.console.print(f"[bold blue]Starting training: {total_epochs} epochs, {steps_per_epoch} steps/epoch[/bold blue]")

    def update_training_progress(
        self,
        epoch: int,
        step: int,
        metrics: TrainingMetrics,
        memory_manager: AppleSiliconMemoryManager,
        profiler: AppleSiliconProfiler,
    ) -> None:
        """Update training progress display.

        Args:
            epoch: Current epoch
            step: Current step
            metrics: Training metrics
            memory_manager: Memory manager for metrics
            profiler: Profiler for performance metrics
        """
        if not self.enabled:
            return

        # Update via display manager if available
        if self.display_manager:
            # Update progress tasks
            if self.training_task_id:
                self.display_manager.update_progress_task(
                    self.training_task_id, advance=0, 
                    description=f"Epoch {epoch} - Overall Progress"
                )
            if self.epoch_task_id:
                self.display_manager.update_progress_task(
                    self.epoch_task_id, advance=1,
                    description=f"Step {step} - Current Epoch"
                )
            
            # Update metrics in display manager
            metrics_dict = self._extract_metrics_dict(metrics, memory_manager, profiler)
            self.display_manager.update_metrics(metrics_dict)
            
            # Update status
            status = f"Epoch {epoch}, Step {step}"
            if metrics.train_loss.values:
                status += f", Loss: {metrics.train_loss.values[-1]:.4f}"
            self.display_manager.update_status(status)
        else:
            # Fallback to console logging
            loss_str = f"{metrics.train_loss.values[-1]:.4f}" if metrics.train_loss.values else "N/A"
            self.console.print(f"[dim]Epoch {epoch}, Step {step}, Loss: {loss_str}[/dim]")
    
    def _extract_metrics_dict(
        self,
        metrics: TrainingMetrics,
        memory_manager: AppleSiliconMemoryManager,
        profiler: AppleSiliconProfiler,
    ) -> Dict[str, Any]:
        """Extract metrics into a dictionary for display manager."""
        metrics_dict = {}
        
        # Training metrics
        if metrics.train_loss.values:
            metrics_dict["Train Loss"] = metrics.train_loss.values[-1]
        if metrics.train_accuracy.values:
            metrics_dict["Train Accuracy"] = metrics.train_accuracy.values[-1]
        if metrics.val_loss.values:
            metrics_dict["Val Loss"] = metrics.val_loss.values[-1]
        if metrics.val_accuracy.values:
            metrics_dict["Val Accuracy"] = metrics.val_accuracy.values[-1]
        if metrics.learning_rate.values:
            metrics_dict["Learning Rate"] = f"{metrics.learning_rate.values[-1]:.2e}"
        
        # Memory metrics
        try:
            memory_metrics = memory_manager.get_current_metrics()
            metrics_dict["Memory Usage"] = f"{memory_metrics.memory_percentage:.1f}%"
            metrics_dict["Memory Used"] = f"{memory_metrics.memory_used_mb:.0f}MB"
        except Exception as e:
            metrics_dict["Memory"] = "Error"
        
        # Performance metrics
        try:
            if profiler.metrics_history:
                latest_perf = profiler.metrics_history[-1]
                metrics_dict["Step Time"] = f"{latest_perf.step_time_seconds:.3f}s"
                metrics_dict["Throughput"] = f"{latest_perf.samples_per_second:.1f} samples/s"
                if latest_perf.tokens_per_second > 0:
                    metrics_dict["Token Throughput"] = f"{latest_perf.tokens_per_second:.0f} tokens/s"
        except Exception as e:
            metrics_dict["Performance"] = "Error"
        
        return metrics_dict

    def _create_display_panel(
        self,
        metrics: TrainingMetrics | None = None,
        memory_manager: AppleSiliconMemoryManager | None = None,
        profiler: AppleSiliconProfiler | None = None,
    ) -> Panel:
        """Create the main display panel.

        Args:
            metrics: Training metrics
            memory_manager: Memory manager
            profiler: Performance profiler

        Returns:
            Rich panel with training information
        """
        # Create main layout
        layout_table = Table.grid(padding=1)
        layout_table.add_column(ratio=1)
        layout_table.add_column(ratio=1)

        # Left column: Progress and metrics
        left_content = [self.progress] if self.progress else []

        if metrics:
            left_content.append(self._create_metrics_table(metrics))

        # Right column: System info
        right_content = []

        if memory_manager:
            right_content.append(self._create_memory_table(memory_manager))

        if profiler:
            right_content.append(self._create_performance_table(profiler))

        # Add content to layout
        if left_content and right_content:
            layout_table.add_row(
                Panel(
                    "\n".join([str(c) for c in left_content]), title="Training Progress"
                ),
                Panel(
                    "\n".join([str(c) for c in right_content]), title="System Status"
                ),
            )
        elif left_content:
            layout_table.add_row(
                Panel(
                    "\n".join([str(c) for c in left_content]), title="Training Progress"
                )
            )

        return Panel(
            layout_table,
            title=f"MLX Training Monitor - {self.config.experiment_name or 'MLX Training'}",
        )

    def _create_metrics_table(self, metrics: TrainingMetrics) -> Table:
        """Create metrics display table.

        Args:
            metrics: Training metrics

        Returns:
            Rich table with metrics
        """
        table = Table(
            title="Training Metrics", show_header=True, header_style="bold magenta"
        )
        table.add_column("Metric", style="cyan")
        table.add_column("Current", justify="right", style="green")
        table.add_column("Best", justify="right", style="blue")
        table.add_column("Trend", justify="center", style="yellow")

        for name, tracker in metrics.get_all_trackers().items():
            if tracker.values:
                current = f"{tracker.values[-1]:.4f}"
                best = (
                    f"{tracker.best_value:.4f}"
                    if tracker.best_value is not None
                    else "N/A"
                )
                trend = tracker.get_trend()

                table.add_row(name.replace("_", " ").title(), current, best, trend)

        return table

    def _create_memory_table(self, memory_manager: AppleSiliconMemoryManager) -> Table:
        """Create memory status table.

        Args:
            memory_manager: Memory manager

        Returns:
            Rich table with memory info
        """
        table = Table(title="Memory Status", show_header=True, header_style="bold red")
        table.add_column("Component", style="cyan")
        table.add_column("Value", justify="right", style="white")

        try:
            metrics = memory_manager.get_current_metrics()

            table.add_row("Total Memory", f"{metrics.total_memory_gb:.1f} GB")
            table.add_row("Used Memory", f"{metrics.used_memory_gb:.1f} GB")
            table.add_row("Usage", f"{metrics.memory_percentage:.1%}")

            # Add color coding for memory usage
            usage_color = "green"
            if metrics.memory_percentage > 0.85:
                usage_color = "red"
            elif metrics.memory_percentage > 0.75:
                usage_color = "yellow"

            table.rows[-1] = (
                table.rows[-1][0],
                Text(table.rows[-1][1], style=usage_color),
            )

        except Exception as e:
            table.add_row("Error", str(e))

        return table

    def _create_performance_table(self, profiler: AppleSiliconProfiler) -> Table:
        """Create performance status table.

        Args:
            profiler: Performance profiler

        Returns:
            Rich table with performance info
        """
        table = Table(title="Performance", show_header=True, header_style="bold green")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", justify="right", style="white")

        try:
            if profiler.metrics_history:
                latest = profiler.metrics_history[-1]

                table.add_row("Step Time", f"{latest.step_time_seconds:.3f}s")
                table.add_row(
                    "Throughput", f"{latest.samples_per_second:.1f} samples/s"
                )

                if latest.tokens_per_second > 0:
                    table.add_row(
                        "Token Throughput", f"{latest.tokens_per_second:.0f} tokens/s"
                    )

                if profiler.is_apple_silicon:
                    table.add_row("Apple Silicon", "✓")
                    if latest.neural_engine_utilization > 0:
                        table.add_row(
                            "Neural Engine", f"{latest.neural_engine_utilization:.1f}%"
                        )
                    if latest.gpu_utilization > 0:
                        table.add_row("GPU Usage", f"{latest.gpu_utilization:.1f}%")

        except Exception as e:
            table.add_row("Error", str(e))

        return table

    def stop_training(self) -> None:
        """Stop training progress display."""
        if self.display_manager:
            # Remove progress tasks
            if self.training_task_id:
                self.display_manager.remove_progress_task(self.training_task_id)
            if self.epoch_task_id:
                self.display_manager.remove_progress_task(self.epoch_task_id)
            
            # Update status
            elapsed = time.time() - self.start_time
            self.display_manager.update_status(f"Training completed in {elapsed:.1f}s")
        else:
            # Show final summary
            elapsed = time.time() - self.start_time
            self.console.print(
                f"\n[bold green]Training completed in {elapsed:.1f} seconds[/bold green]"
            )


class ComprehensiveMonitor:
    """Unified monitoring system combining all monitoring components."""

    def __init__(
        self,
        config: TrainingConfig,
        memory_manager: AppleSiliconMemoryManager,
        profiler: AppleSiliconProfiler,
        display_manager: Optional[RichDisplayManager] = None,
    ):
        """Initialize comprehensive monitoring system.

        Args:
            config: Training configuration
            memory_manager: Memory manager
            profiler: Performance profiler
            display_manager: Optional shared display manager
        """
        self.config = config
        self.memory_manager = memory_manager
        self.profiler = profiler
        self.display_manager = display_manager

        # Initialize monitoring components
        self.metrics = TrainingMetrics()
        self.mlflow_tracker = MLflowTracker(config)
        self.console_monitor = RichConsoleMonitor(config, display_manager)

        # State tracking
        self.start_time = time.time()
        self.last_log_time = time.time()
        self.step_count = 0

        logger.info("Initialized comprehensive monitoring system")

    def start_training(self, total_epochs: int, steps_per_epoch: int) -> None:
        """Start training monitoring.

        Args:
            total_epochs: Total number of epochs
            steps_per_epoch: Steps per epoch
        """
        self.console_monitor.start_training(total_epochs, steps_per_epoch)
        logger.info(
            f"Started training monitoring: {total_epochs} epochs, {steps_per_epoch} steps/epoch"
        )

    def log_step(
        self,
        step: int,
        epoch: int,
        train_loss: float,
        train_accuracy: float | None = None,
        learning_rate: float | None = None,
        batch_size: int | None = None,
    ) -> None:
        """Log training step metrics.

        Args:
            step: Global step number
            epoch: Current epoch
            train_loss: Training loss
            train_accuracy: Training accuracy (optional)
            learning_rate: Current learning rate (optional)
            batch_size: Current batch size (optional)
        """
        self.step_count = step

        # Update metrics
        self.metrics.train_loss.update(train_loss, step)
        if train_accuracy is not None:
            self.metrics.train_accuracy.update(train_accuracy, step)
        if learning_rate is not None:
            self.metrics.learning_rate.update(learning_rate, step)

        # Get performance metrics
        if self.profiler.metrics_history:
            latest_perf = self.profiler.metrics_history[-1]
            self.metrics.step_time.update(latest_perf.step_time_seconds, step)
            self.metrics.throughput.update(latest_perf.samples_per_second, step)

        # Get memory metrics
        memory_metrics = self.memory_manager.get_current_metrics()
        self.metrics.memory_usage.update(memory_metrics.memory_percentage, step)

        # Log to MLflow
        if step % self.config.monitoring.log_frequency == 0:
            mlflow_metrics = {
                "train_loss": train_loss,
                "memory_usage": memory_metrics.memory_percentage,
            }

            if train_accuracy is not None:
                mlflow_metrics["train_accuracy"] = train_accuracy
            if learning_rate is not None:
                mlflow_metrics["learning_rate"] = learning_rate
            if batch_size is not None:
                mlflow_metrics["batch_size"] = batch_size

            self.mlflow_tracker.log_metrics(mlflow_metrics, step)

        # Update console display
        if step % self.console_monitor.update_frequency == 0:
            self.console_monitor.update_training_progress(
                epoch, step % 1000, self.metrics, self.memory_manager, self.profiler
            )

    def log_validation(
        self,
        step: int,
        val_loss: float,
        val_accuracy: float | None = None,
        additional_metrics: dict[str, float] | None = None,
    ) -> bool:
        """Log validation metrics.

        Args:
            step: Global step number
            val_loss: Validation loss
            val_accuracy: Validation accuracy (optional)
            additional_metrics: Additional validation metrics

        Returns:
            True if validation metrics improved
        """
        # Update validation metrics
        loss_improved = self.metrics.val_loss.update(val_loss, step)
        acc_improved = False

        if val_accuracy is not None:
            acc_improved = self.metrics.val_accuracy.update(val_accuracy, step)

        # Prepare MLflow metrics
        mlflow_metrics = {"val_loss": val_loss}
        if val_accuracy is not None:
            mlflow_metrics["val_accuracy"] = val_accuracy
        if additional_metrics:
            mlflow_metrics.update(additional_metrics)

        # Log to MLflow
        self.mlflow_tracker.log_metrics(mlflow_metrics, step)

        # Log significant improvements
        if loss_improved or acc_improved:
            logger.info(f"Validation metrics improved at step {step}")
            if loss_improved:
                logger.info(f"  Best val_loss: {val_loss:.4f}")
            if acc_improved:
                logger.info(f"  Best val_accuracy: {val_accuracy:.4f}")

        return loss_improved or acc_improved

    def save_checkpoint_artifacts(
        self,
        checkpoint_path: str,
        additional_artifacts: dict[str, str] | None = None,
    ) -> None:
        """Save checkpoint and other artifacts.

        Args:
            checkpoint_path: Path to model checkpoint
            additional_artifacts: Additional artifacts to save
        """
        artifacts = {"model_checkpoint": checkpoint_path}

        if additional_artifacts:
            artifacts.update(additional_artifacts)

        # Save performance reports
        try:
            output_dir = Path(self.config.output_dir)

            # Save memory report
            memory_report_path = output_dir / "memory_report.json"
            self.memory_manager.save_memory_report(memory_report_path)
            artifacts["memory_report"] = str(memory_report_path)

            # Save performance report
            performance_report_path = output_dir / "performance_report.json"
            self.profiler.save_performance_report(performance_report_path)
            artifacts["performance_report"] = str(performance_report_path)

        except Exception as e:
            logger.warning(f"Failed to save monitoring reports: {e}")

        # Log to MLflow
        self.mlflow_tracker.log_artifacts(artifacts)

    def end_training(self, status: str = "FINISHED") -> dict[str, Any]:
        """End training monitoring and return summary.

        Args:
            status: Training status

        Returns:
            Training summary statistics
        """
        # Stop console monitoring
        self.console_monitor.stop_training()

        # Get final summaries
        performance_summary = self.profiler.get_performance_summary()
        memory_recommendations = self.memory_manager.get_memory_recommendations()

        # Create training summary
        elapsed_time = time.time() - self.start_time

        training_summary = {
            "status": status,
            "elapsed_time_seconds": elapsed_time,
            "total_steps": self.step_count,
            "average_step_time": elapsed_time / max(1, self.step_count),
            "performance_summary": performance_summary,
            "memory_recommendations": memory_recommendations,
            "best_metrics": {
                "train_loss": self.metrics.train_loss.best_value,
                "train_accuracy": self.metrics.train_accuracy.best_value,
                "val_loss": self.metrics.val_loss.best_value,
                "val_accuracy": self.metrics.val_accuracy.best_value,
            },
        }

        # Log summary to MLflow
        summary_metrics = {
            "final_train_loss": self.metrics.train_loss.values[-1]
            if self.metrics.train_loss.values
            else 0,
            "best_train_loss": self.metrics.train_loss.best_value or 0,
            "best_val_loss": self.metrics.val_loss.best_value or 0,
            "best_val_accuracy": self.metrics.val_accuracy.best_value or 0,
            "total_training_time": elapsed_time,
            "average_step_time": elapsed_time / max(1, self.step_count),
        }
        self.mlflow_tracker.log_metrics(summary_metrics, self.step_count)

        # End MLflow run
        self.mlflow_tracker.end_run(status)

        logger.info(f"Training monitoring ended with status: {status}")
        logger.info(f"Total time: {elapsed_time:.1f}s, Steps: {self.step_count}")

        return training_summary
