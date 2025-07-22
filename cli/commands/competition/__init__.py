"""Competition management commands for k-bert CLI."""

import typer

from .download import download_command
from .submit import submit_command
from .list import list_command
from .info import info_command
from .init import init_command

# Create the competition command group
app = typer.Typer(
    name="competition",
    help="Manage Kaggle competitions",
    no_args_is_help=True,
)

# Register commands
app.command("download")(download_command)
app.command("submit")(submit_command)
app.command("list")(list_command)
app.command("info")(info_command)
app.command("init")(init_command)