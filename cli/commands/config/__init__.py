"""Configuration management commands for k-bert CLI."""

import typer

from .init import init_command
from .get import get_command
from .set import set_command
from .list import list_command
from .validate import validate_command

# Create the config command group
app = typer.Typer(
    name="config",
    help="Manage k-bert configuration",
    no_args_is_help=True,
)

# Register commands
app.command("init")(init_command)
app.command("get")(get_command)
app.command("set")(set_command)
app.command("list")(list_command)
app.command("validate")(validate_command)