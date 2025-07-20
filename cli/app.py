"""Main CLI application setup and configuration."""

import typer
from rich.console import Console
from typing import Optional
import sys
from pathlib import Path

# Import command groups
from .commands.core import app as core_app
from .commands.kaggle import kaggle_app
from .commands.mlflow import mlflow_app
from .commands.model import model_app

# Initialize the main app
app = typer.Typer(
    name="bert",
    help="MLX BERT CLI - The ultimate BERT playground for Kaggle competitions on Apple Silicon",
    add_completion=True,
    no_args_is_help=True,
    rich_markup_mode="rich",
    pretty_exceptions_enable=True,
    pretty_exceptions_show_locals=False,
)

# Add command groups
app.add_typer(kaggle_app, name="kaggle", help="Kaggle competition workflows")
app.add_typer(mlflow_app, name="mlflow", help="MLflow experiment tracking")
app.add_typer(model_app, name="model", help="Model management and serving")

# Global options
console = Console()

def version_callback(value: bool):
    if value:
        from . import __version__
        console.print(f"MLX BERT CLI version {__version__}")
        raise typer.Exit()

@app.callback()
def callback(
    version: bool = typer.Option(
        None, "--version", "-v", 
        callback=version_callback, is_eager=True,
        help="Show version"
    ),
    verbose: bool = typer.Option(False, "--verbose", "-V", help="Enable verbose output"),
    quiet: bool = typer.Option(False, "--quiet", "-q", help="Minimal output"),
):
    """MLX BERT CLI - The ultimate BERT playground for Kaggle competitions."""
    # Set global verbosity
    if verbose:
        import os
        os.environ["BERT_CLI_VERBOSE"] = "1"
    elif quiet:
        import os
        os.environ["BERT_CLI_QUIET"] = "1"

# Core commands at root level for convenience
from .commands.core.train_simple import train_command
from .commands.core.predict import predict_command
from .commands.core.benchmark import benchmark_command
from .commands.core.info import info_command

app.command(name="train", help="Train a BERT model")(train_command)
app.command(name="predict", help="Generate predictions")(predict_command)
app.command(name="benchmark", help="Run performance benchmarks")(benchmark_command)
app.command(name="info", help="Show system information")(info_command)

# Additional root-level commands (to be implemented)
# @app.command()
# def init(
#     name: str = typer.Argument("my-bert-project", help="Project name"),
#     template: str = typer.Option("default", help="Project template to use"),
#     force: bool = typer.Option(False, "--force", "-f", help="Overwrite existing files"),
# ):
#     """Initialize a new BERT project with best practices."""
#     from .commands.core.init import init_project
#     init_project(name, template, force)

# @app.command()
# def config(
#     action: str = typer.Argument(..., help="Action: init, validate, show, edit"),
#     config_path: Optional[Path] = typer.Option(None, "--config", "-c", help="Config file path"),
#     interactive: bool = typer.Option(False, "--interactive", "-i", help="Interactive mode"),
# ):
#     """Manage project configuration."""
#     from .commands.core.config import config_command
#     config_command(action, config_path, interactive)

# @app.command()
# def interactive():
#     """Start interactive CLI mode."""
#     from .commands.core.interactive import interactive_mode
#     interactive_mode()

if __name__ == "__main__":
    app()