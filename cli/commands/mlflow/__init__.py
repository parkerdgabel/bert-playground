"""MLflow experiment tracking and management commands."""

from typer import Typer

from .dashboard import dashboard_command
from .experiments import clean_command, list_experiments_command
from .health import health_command
from .runs import list_runs_command
from .server import restart_command, server_command
from .test import test_command

# Create MLflow sub-app
mlflow_app = Typer(
    name="mlflow",
    help="MLflow experiment tracking and management",
    no_args_is_help=True,
    rich_markup_mode="rich",
)

# Register commands with improved organization
mlflow_app.command("server", help="Start MLflow tracking server")(server_command)
mlflow_app.command("restart", help="Restart MLflow server")(restart_command)
mlflow_app.command("health", help="Check MLflow health and configuration")(
    health_command
)
mlflow_app.command("experiments", help="List experiments")(list_experiments_command)
mlflow_app.command("runs", help="List runs for an experiment")(list_runs_command)
mlflow_app.command("clean", help="Clean up experiments and runs")(clean_command)
mlflow_app.command("dashboard", help="Launch real-time monitoring dashboard")(
    dashboard_command
)
mlflow_app.command("test", help="Run comprehensive MLflow test suite")(test_command)

__all__ = ["mlflow_app"]
