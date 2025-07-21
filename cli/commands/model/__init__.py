"""Model management and serving commands."""

from typer import Typer

from .convert import convert_command
from .evaluate import evaluate_command
from .export import export_command
from .inspect import inspect_command
from .list import list_models_command
from .serve import serve_command

# Create model sub-app
model_app = Typer(
    name="model",
    help="Model management, export, and serving",
    no_args_is_help=True,
    rich_markup_mode="rich",
)

# Register commands with improved organization
model_app.command("list", help="List available models and checkpoints")(
    list_models_command
)
model_app.command("inspect", help="Inspect model architecture and parameters")(
    inspect_command
)
model_app.command("export", help="Export model to different formats")(export_command)
model_app.command("convert", help="Convert between model formats")(convert_command)
model_app.command("serve", help="Serve model as REST API")(serve_command)
model_app.command("evaluate", help="Evaluate model performance")(evaluate_command)

__all__ = ["model_app"]
