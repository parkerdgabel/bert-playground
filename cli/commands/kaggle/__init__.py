"""Kaggle integration commands."""

import typer

# Create the kaggle commands app
app = typer.Typer(
    help="Kaggle competition workflows",
    no_args_is_help=True,
)

# Import and register subcommands
from .competitions import competitions_app
from .submissions import submissions_app
from .datasets import datasets_app

app.add_typer(competitions_app, name="competitions", help="Manage competitions")
app.add_typer(submissions_app, name="submit", help="Submit predictions") 
app.add_typer(datasets_app, name="datasets", help="Manage datasets")

# Direct commands at kaggle level
from .download import download_command
from .leaderboard import leaderboard_command

app.command(name="download")(download_command)
app.command(name="leaderboard")(leaderboard_command)

__all__ = ["app"]