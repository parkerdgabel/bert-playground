"""MLflow management commands."""

import typer

# Create the mlflow commands app
app = typer.Typer(
    help="MLflow experiment tracking and model registry",
    no_args_is_help=True,
)

# Import commands
from .server import server_command
from .experiments import experiments_command
from .runs import runs_command
from .ui import ui_command
from .health import health_command

# Register commands
app.command(name="server")(server_command)
app.command(name="experiments")(experiments_command)
app.command(name="runs")(runs_command)
app.command(name="ui")(ui_command)
app.command(name="health")(health_command)

__all__ = ["app"]