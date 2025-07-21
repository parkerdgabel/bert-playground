"""MLflow status dashboard and monitoring tools.

This module provides real-time monitoring and dashboard capabilities
for MLflow experiments and runs.
"""

import queue
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import mlflow
from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from utils.mlflow_central import MLflowCentral


class MLflowDashboard:
    """Real-time MLflow dashboard."""

    def __init__(self, refresh_interval: float = 5.0):
        """Initialize dashboard.

        Args:
            refresh_interval: Dashboard refresh interval in seconds
        """
        self.console = Console()
        self.refresh_interval = refresh_interval
        self.mlflow_central = MLflowCentral()
        self.running = False
        self.update_thread = None
        self.data_queue = queue.Queue()

        # Dashboard state
        self.experiments_data = []
        self.runs_data = []
        self.system_stats = {}
        self.alerts = []

        # Initialize MLflow
        self.mlflow_central.initialize()

    def start(self):
        """Start the dashboard."""
        self.running = True

        # Start background update thread
        self.update_thread = threading.Thread(target=self._update_loop)
        self.update_thread.daemon = True
        self.update_thread.start()

        # Start live dashboard
        with Live(
            self._create_dashboard(), refresh_per_second=1 / self.refresh_interval
        ) as live:
            try:
                while self.running:
                    # Check for updates
                    try:
                        update_data = self.data_queue.get_nowait()
                        self._process_update(update_data)
                    except queue.Empty:
                        pass

                    # Refresh display
                    live.update(self._create_dashboard())
                    time.sleep(self.refresh_interval)

            except KeyboardInterrupt:
                self.running = False
                self.console.print("\n[yellow]Dashboard stopped by user[/yellow]")

    def stop(self):
        """Stop the dashboard."""
        self.running = False
        if self.update_thread and self.update_thread.is_alive():
            self.update_thread.join(timeout=2.0)

    def _update_loop(self):
        """Background update loop."""
        while self.running:
            try:
                # Collect data
                experiments = self._get_experiments_data()
                runs = self._get_runs_data()
                system_stats = self._get_system_stats()
                alerts = self._check_alerts()

                # Queue update
                self.data_queue.put(
                    {
                        "experiments": experiments,
                        "runs": runs,
                        "system_stats": system_stats,
                        "alerts": alerts,
                    }
                )

            except Exception as e:
                self.data_queue.put({"error": f"Data update failed: {str(e)}"})

            time.sleep(self.refresh_interval)

    def _process_update(self, update_data: dict[str, Any]):
        """Process update data."""
        if "error" in update_data:
            self.alerts.append(
                {
                    "type": "ERROR",
                    "message": update_data["error"],
                    "timestamp": datetime.now(),
                }
            )
        else:
            self.experiments_data = update_data.get("experiments", [])
            self.runs_data = update_data.get("runs", [])
            self.system_stats = update_data.get("system_stats", {})
            new_alerts = update_data.get("alerts", [])

            # Add new alerts
            for alert in new_alerts:
                alert["timestamp"] = datetime.now()
                self.alerts.append(alert)

            # Keep only recent alerts (last 100)
            self.alerts = self.alerts[-100:]

    def _get_experiments_data(self) -> list[dict[str, Any]]:
        """Get experiments data."""
        try:
            experiments = mlflow.search_experiments()

            experiment_data = []
            for exp in experiments:
                # Get run count and latest activity
                runs = mlflow.search_runs(
                    experiment_ids=[exp.experiment_id],
                    max_results=1,
                    order_by=["start_time DESC"],
                )

                run_count = len(mlflow.search_runs(experiment_ids=[exp.experiment_id]))

                latest_run = None
                if len(runs) > 0:
                    latest_run = runs.iloc[0]

                experiment_data.append(
                    {
                        "id": exp.experiment_id,
                        "name": exp.name,
                        "run_count": run_count,
                        "latest_run": latest_run.start_time
                        if latest_run is not None
                        else None,
                        "status": exp.lifecycle_stage,
                        "artifact_location": exp.artifact_location,
                    }
                )

            return experiment_data

        except Exception as e:
            return [{"error": str(e)}]

    def _get_runs_data(self) -> list[dict[str, Any]]:
        """Get recent runs data."""
        try:
            # Get runs from last 24 hours
            runs = mlflow.search_runs(max_results=50, order_by=["start_time DESC"])

            if len(runs) == 0:
                return []

            runs_data = []
            for _, run in runs.iterrows():
                # Calculate duration
                duration = None
                if run.end_time and run.start_time:
                    duration = (run.end_time - run.start_time).total_seconds()

                runs_data.append(
                    {
                        "run_id": run.run_id,
                        "experiment_id": run.experiment_id,
                        "status": run.status,
                        "start_time": run.start_time,
                        "end_time": run.end_time,
                        "duration": duration,
                        "metrics": dict(run.drop(["params", "tags"]).dropna())
                        if hasattr(run, "drop")
                        else {},
                        "params": getattr(run, "params", {})
                        if hasattr(run, "params")
                        else {},
                    }
                )

            return runs_data

        except Exception as e:
            return [{"error": str(e)}]

    def _get_system_stats(self) -> dict[str, Any]:
        """Get system statistics."""
        try:
            stats = {}

            # Database size
            connection_status = self.mlflow_central.validate_connection()
            if connection_status["status"] == "CONNECTED":
                stats.update(connection_status.get("details", {}))

            # Memory usage
            try:
                import psutil

                process = psutil.Process()
                stats["memory_usage_mb"] = process.memory_info().rss / (1024 * 1024)
                stats["cpu_percent"] = process.cpu_percent()
            except ImportError:
                stats["memory_usage_mb"] = "N/A"
                stats["cpu_percent"] = "N/A"

            # Disk usage
            try:
                mlruns_path = Path("mlruns")
                if mlruns_path.exists():
                    total_size = sum(
                        f.stat().st_size for f in mlruns_path.rglob("*") if f.is_file()
                    )
                    stats["disk_usage_mb"] = total_size / (1024 * 1024)
                else:
                    stats["disk_usage_mb"] = 0
            except Exception:
                stats["disk_usage_mb"] = "N/A"

            return stats

        except Exception as e:
            return {"error": str(e)}

    def _check_alerts(self) -> list[dict[str, Any]]:
        """Check for alerts and warnings."""
        alerts = []

        # Check for failed runs
        try:
            failed_runs = mlflow.search_runs(
                filter_string="status = 'FAILED'",
                max_results=10,
                order_by=["start_time DESC"],
            )

            if len(failed_runs) > 0:
                alerts.append(
                    {
                        "type": "WARNING",
                        "message": f"{len(failed_runs)} failed runs detected",
                    }
                )
        except Exception:
            pass

        # Check database size
        if self.system_stats.get("database_size_mb", 0) > 100:
            alerts.append(
                {
                    "type": "WARNING",
                    "message": f"Large database size: {self.system_stats['database_size_mb']:.1f} MB",
                }
            )

        # Check disk usage
        if self.system_stats.get("disk_usage_mb", 0) > 500:
            alerts.append(
                {
                    "type": "WARNING",
                    "message": f"High disk usage: {self.system_stats['disk_usage_mb']:.1f} MB",
                }
            )

        return alerts

    def _create_dashboard(self) -> Layout:
        """Create the main dashboard layout."""
        layout = Layout()

        # Split into header and main content
        layout.split_column(Layout(name="header", size=3), Layout(name="main"))

        # Split main content
        layout["main"].split_row(Layout(name="left"), Layout(name="right"))

        # Split left column
        layout["left"].split_column(
            Layout(name="experiments", ratio=1), Layout(name="runs", ratio=2)
        )

        # Split right column
        layout["right"].split_column(
            Layout(name="system", ratio=1), Layout(name="alerts", ratio=1)
        )

        # Fill layouts
        layout["header"].update(self._create_header())
        layout["experiments"].update(self._create_experiments_table())
        layout["runs"].update(self._create_runs_table())
        layout["system"].update(self._create_system_panel())
        layout["alerts"].update(self._create_alerts_panel())

        return layout

    def _create_header(self) -> Panel:
        """Create header panel."""
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        header_text = Text()
        header_text.append("MLflow Dashboard", style="bold blue")
        header_text.append(" | ", style="dim")
        header_text.append(f"Last Updated: {current_time}", style="dim")
        header_text.append(" | ", style="dim")
        header_text.append("Press Ctrl+C to exit", style="dim")

        return Panel(header_text, title="MLflow Real-time Dashboard")

    def _create_experiments_table(self) -> Panel:
        """Create experiments table."""
        table = Table(title="Experiments")
        table.add_column("Name", style="cyan")
        table.add_column("Runs", justify="right")
        table.add_column("Status", style="green")
        table.add_column("Latest Run", style="dim")

        for exp in self.experiments_data:
            if "error" in exp:
                table.add_row("ERROR", str(exp["error"]), "-", "-")
            else:
                latest_run = exp["latest_run"]
                latest_str = latest_run.strftime("%H:%M:%S") if latest_run else "Never"

                table.add_row(
                    exp["name"], str(exp["run_count"]), exp["status"], latest_str
                )

        return Panel(table, title="Experiments")

    def _create_runs_table(self) -> Panel:
        """Create runs table."""
        table = Table(title="Recent Runs")
        table.add_column("Run ID", style="cyan", max_width=12)
        table.add_column("Status", style="green")
        table.add_column("Duration", justify="right")
        table.add_column("Metrics", style="dim")
        table.add_column("Start Time", style="dim")

        for run in self.runs_data[:10]:  # Show only first 10 runs
            if "error" in run:
                table.add_row("ERROR", str(run["error"]), "-", "-", "-")
            else:
                # Format duration
                duration_str = "Running"
                if run["duration"]:
                    duration_str = f"{run['duration']:.1f}s"
                elif run["status"] == "FAILED":
                    duration_str = "Failed"

                # Format metrics
                metrics = run.get("metrics", {})
                metrics_str = ""
                if "train_loss" in metrics:
                    metrics_str = f"Loss: {metrics['train_loss']:.3f}"
                elif "val_accuracy" in metrics:
                    metrics_str = f"Acc: {metrics['val_accuracy']:.3f}"

                # Format start time
                start_time = run["start_time"]
                start_str = start_time.strftime("%H:%M:%S") if start_time else "Unknown"

                table.add_row(
                    run["run_id"][:8] + "...",
                    run["status"],
                    duration_str,
                    metrics_str,
                    start_str,
                )

        return Panel(table, title="Recent Runs")

    def _create_system_panel(self) -> Panel:
        """Create system statistics panel."""
        stats_text = Text()

        if "error" in self.system_stats:
            stats_text.append(f"Error: {self.system_stats['error']}", style="red")
        else:
            # Connection status
            connection_status = self.mlflow_central.validate_connection()
            status_style = (
                "green" if connection_status["status"] == "CONNECTED" else "red"
            )
            stats_text.append(
                f"Connection: {connection_status['status']}\n", style=status_style
            )

            # Database size
            if "database_size_mb" in self.system_stats:
                db_size = self.system_stats["database_size_mb"]
                stats_text.append(f"Database: {db_size:.1f} MB\n")

            # Disk usage
            if "disk_usage_mb" in self.system_stats:
                disk_usage = self.system_stats["disk_usage_mb"]
                stats_text.append(f"Disk Usage: {disk_usage:.1f} MB\n")

            # Memory usage
            if "memory_usage_mb" in self.system_stats:
                memory = self.system_stats["memory_usage_mb"]
                if memory != "N/A":
                    stats_text.append(f"Memory: {memory:.1f} MB\n")

            # CPU usage
            if "cpu_percent" in self.system_stats:
                cpu = self.system_stats["cpu_percent"]
                if cpu != "N/A":
                    stats_text.append(f"CPU: {cpu:.1f}%\n")

            # Experiment count
            if "experiment_count" in self.system_stats:
                exp_count = self.system_stats["experiment_count"]
                stats_text.append(f"Experiments: {exp_count}\n")

        return Panel(stats_text, title="System Status")

    def _create_alerts_panel(self) -> Panel:
        """Create alerts panel."""
        alerts_text = Text()

        if not self.alerts:
            alerts_text.append("No alerts", style="green")
        else:
            # Show latest 5 alerts
            for alert in self.alerts[-5:]:
                timestamp = alert["timestamp"].strftime("%H:%M:%S")
                alert_type = alert["type"]
                message = alert["message"]

                style = "red" if alert_type == "ERROR" else "yellow"
                alerts_text.append(
                    f"[{timestamp}] {alert_type}: {message}\n", style=style
                )

        return Panel(alerts_text, title="Alerts")


class MLflowMonitor:
    """MLflow monitoring utilities."""

    def __init__(self):
        """Initialize monitor."""
        self.mlflow_central = MLflowCentral()
        self.mlflow_central.initialize()

    def get_experiment_summary(
        self, experiment_name: str | None = None
    ) -> dict[str, Any]:
        """Get experiment summary statistics."""
        try:
            if experiment_name:
                experiment = mlflow.get_experiment_by_name(experiment_name)
                if not experiment:
                    return {"error": f"Experiment '{experiment_name}' not found"}
                experiment_ids = [experiment.experiment_id]
            else:
                experiments = mlflow.search_experiments()
                experiment_ids = [exp.experiment_id for exp in experiments]

            runs = mlflow.search_runs(experiment_ids=experiment_ids)

            if len(runs) == 0:
                return {"message": "No runs found"}

            # Calculate statistics
            total_runs = len(runs)
            successful_runs = len(runs[runs.status == "FINISHED"])
            failed_runs = len(runs[runs.status == "FAILED"])
            running_runs = len(runs[runs.status == "RUNNING"])

            # Calculate durations
            completed_runs = runs[runs.status == "FINISHED"]
            if len(completed_runs) > 0:
                durations = (
                    completed_runs.end_time - completed_runs.start_time
                ).dt.total_seconds()
                avg_duration = durations.mean()
                total_duration = durations.sum()
            else:
                avg_duration = 0
                total_duration = 0

            # Get metric statistics
            metric_stats = {}
            for col in runs.columns:
                if col.startswith("metrics."):
                    metric_name = col.replace("metrics.", "")
                    metric_values = runs[col].dropna()
                    if len(metric_values) > 0:
                        metric_stats[metric_name] = {
                            "count": len(metric_values),
                            "mean": metric_values.mean(),
                            "std": metric_values.std(),
                            "min": metric_values.min(),
                            "max": metric_values.max(),
                        }

            return {
                "total_runs": total_runs,
                "successful_runs": successful_runs,
                "failed_runs": failed_runs,
                "running_runs": running_runs,
                "success_rate": successful_runs / total_runs if total_runs > 0 else 0,
                "avg_duration_seconds": avg_duration,
                "total_duration_seconds": total_duration,
                "metric_statistics": metric_stats,
            }

        except Exception as e:
            return {"error": str(e)}

    def get_run_comparison(self, run_ids: list[str]) -> dict[str, Any]:
        """Compare multiple runs."""
        try:
            runs_data = []

            for run_id in run_ids:
                run = mlflow.get_run(run_id)

                runs_data.append(
                    {
                        "run_id": run_id,
                        "status": run.info.status,
                        "start_time": run.info.start_time,
                        "end_time": run.info.end_time,
                        "params": run.data.params,
                        "metrics": run.data.metrics,
                        "tags": run.data.tags,
                    }
                )

            return {"runs": runs_data, "comparison": self._compare_runs(runs_data)}

        except Exception as e:
            return {"error": str(e)}

    def _compare_runs(self, runs_data: list[dict[str, Any]]) -> dict[str, Any]:
        """Compare runs and identify differences."""
        if len(runs_data) < 2:
            return {"message": "Need at least 2 runs for comparison"}

        # Compare parameters
        all_params = set()
        for run in runs_data:
            all_params.update(run["params"].keys())

        param_differences = {}
        for param in all_params:
            values = [run["params"].get(param, "N/A") for run in runs_data]
            if len(set(values)) > 1:  # Parameter differs between runs
                param_differences[param] = values

        # Compare metrics
        all_metrics = set()
        for run in runs_data:
            all_metrics.update(run["metrics"].keys())

        metric_comparison = {}
        for metric in all_metrics:
            values = [run["metrics"].get(metric, None) for run in runs_data]
            valid_values = [v for v in values if v is not None]
            if valid_values:
                metric_comparison[metric] = {
                    "values": values,
                    "min": min(valid_values),
                    "max": max(valid_values),
                    "best_run": run_ids[values.index(max(valid_values))],
                }

        return {
            "parameter_differences": param_differences,
            "metric_comparison": metric_comparison,
        }

    def cleanup_old_runs(
        self, experiment_name: str, keep_best_n: int = 10
    ) -> dict[str, Any]:
        """Clean up old runs, keeping only the best N."""
        try:
            experiment = mlflow.get_experiment_by_name(experiment_name)
            if not experiment:
                return {"error": f"Experiment '{experiment_name}' not found"}

            # Get all runs
            runs = mlflow.search_runs(
                experiment_ids=[experiment.experiment_id],
                order_by=["metrics.val_accuracy DESC"],
            )

            if len(runs) <= keep_best_n:
                return {"message": f"Only {len(runs)} runs found, no cleanup needed"}

            # Mark runs for deletion
            runs_to_delete = runs.iloc[keep_best_n:]
            deleted_count = 0

            for _, run in runs_to_delete.iterrows():
                try:
                    mlflow.delete_run(run.run_id)
                    deleted_count += 1
                except Exception as e:
                    # Log error but continue
                    logger.debug(f"Failed to delete run {run.run_id}: {e}")

            return {
                "message": f"Deleted {deleted_count} runs",
                "kept_runs": keep_best_n,
                "deleted_runs": deleted_count,
            }

        except Exception as e:
            return {"error": str(e)}


def main():
    """Main function to run MLflow dashboard."""
    import argparse

    parser = argparse.ArgumentParser(description="MLflow Dashboard")
    parser.add_argument(
        "--refresh-interval",
        type=float,
        default=5.0,
        help="Dashboard refresh interval in seconds",
    )

    args = parser.parse_args()

    # Create and start dashboard
    dashboard = MLflowDashboard(refresh_interval=args.refresh_interval)

    try:
        dashboard.start()
    except KeyboardInterrupt:
        dashboard.stop()


if __name__ == "__main__":
    main()
