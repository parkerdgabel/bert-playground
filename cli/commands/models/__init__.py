"""Model management commands."""

import typer

# Create the models commands app
app = typer.Typer(
    help="Model management, serving, and deployment",
    no_args_is_help=True,
)

# Import commands
from .evaluate import evaluate_command
from .export import export_command
from .registry import registry_command
from .serve import serve_command

# Register commands
app.command(name="serve")(serve_command)
app.command(name="registry")(registry_command)
app.command(name="evaluate")(evaluate_command)
app.command(name="export")(export_command)

__all__ = ["app"]
