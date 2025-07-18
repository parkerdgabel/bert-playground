"""Project initialization command."""

import typer
from pathlib import Path

def init_project(name: str, template: str, force: bool):
    """Initialize a new BERT project."""
    typer.echo(f"Initializing project {name} - implementation pending")