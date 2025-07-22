"""Main CLI application setup and configuration."""

import typer
from rich.console import Console

# Import command groups
from .commands.kaggle import kaggle_app
from .commands.mlflow import mlflow_app
from .commands.model import model_app
from .commands.config import app as config_app
from .commands.competition import app as competition_app
from .commands.project import app as project_app

# Initialize the main app
app = typer.Typer(
    name="k-bert",
    help="K-BERT: State-of-the-art BERT models for Kaggle competitions on Apple Silicon",
    add_completion=True,
    no_args_is_help=True,
    rich_markup_mode="rich",
    pretty_exceptions_enable=True,
    pretty_exceptions_show_locals=False,
)

# Add command groups
app.add_typer(config_app, name="config", help="Configuration management")
app.add_typer(competition_app, name="competition", help="Kaggle competition management")
app.add_typer(project_app, name="project", help="Project management")
app.add_typer(kaggle_app, name="kaggle", help="Kaggle competition workflows")
app.add_typer(mlflow_app, name="mlflow", help="MLflow experiment tracking")
app.add_typer(model_app, name="model", help="Model management and serving")

# Global options
console = Console()


def version_callback(value: bool):
    if value:
        from . import __version__

        console.print(f"[bold cyan]k-bert[/bold cyan] version [green]{__version__}[/green]")
        raise typer.Exit()


@app.callback()
def callback(
    version: bool = typer.Option(
        None,
        "--version",
        "-v",
        callback=version_callback,
        is_eager=True,
        help="Show version",
    ),
    verbose: bool = typer.Option(
        False, "--verbose", "-V", help="Enable verbose output"
    ),
    quiet: bool = typer.Option(False, "--quiet", "-q", help="Minimal output"),
):
    """K-BERT: State-of-the-art BERT models for Kaggle competitions."""
    # Set global verbosity
    if verbose:
        import os

        os.environ["BERT_CLI_VERBOSE"] = "1"
    elif quiet:
        import os

        os.environ["BERT_CLI_QUIET"] = "1"


# Core commands at root level for convenience
# Config-first versions are now the main versions
from .commands.core.benchmark import benchmark_command
from .commands.core.info import info_command
from .commands.core.predict import predict_command
from .commands.core.train import train_command
from .commands.project.run import run_command

app.command(name="train", help="Train a BERT model")(train_command)
app.command(name="predict", help="Generate predictions")(predict_command)
app.command(name="benchmark", help="Run performance benchmarks")(benchmark_command)
app.command(name="info", help="Show system information")(info_command)
app.command(name="run", help="Run project with configuration")(run_command)


if __name__ == "__main__":
    app()
