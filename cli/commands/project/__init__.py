"""Project management commands for k-bert CLI."""

import typer

from .init import init_command
from .run import run_command
from .template import template_command

# Create the project command group
app = typer.Typer(
    name="project",
    help="Project management and execution",
    no_args_is_help=True,
)

# Register commands
app.command("init")(init_command)
app.command("run")(run_command)
app.command("template")(template_command)