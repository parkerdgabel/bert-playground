"""Kaggle competition integration commands."""

from typer import Typer

from .competitions import competitions_command
from .datasets import datasets_command  
from .submit import submit_command, auto_submit_command
from .leaderboard import leaderboard_command
from .history import history_command
from .download import download_competition_command, download_dataset_command

# Create Kaggle sub-app
kaggle_app = Typer(
    name="kaggle",
    help="Kaggle competition integration commands",
    no_args_is_help=True,
    rich_markup_mode="rich"
)

# Register commands with improved names
kaggle_app.command("competitions", help="List and search Kaggle competitions")(competitions_command)
kaggle_app.command("datasets", help="List and search Kaggle datasets")(datasets_command)
kaggle_app.command("download", help="Download competition data")(download_competition_command)
kaggle_app.command("download-dataset", help="Download a specific dataset")(download_dataset_command)
kaggle_app.command("submit", help="Submit predictions to a competition")(submit_command)
kaggle_app.command("auto-submit", help="Generate and submit predictions automatically")(auto_submit_command)
kaggle_app.command("leaderboard", help="View competition leaderboard")(leaderboard_command)
kaggle_app.command("history", help="View your submission history")(history_command)

__all__ = ["kaggle_app"]