"""Configuration management command."""

from pathlib import Path

import typer


def config_command(action: str, config_path: Path | None, interactive: bool):
    """Manage project configuration."""
    typer.echo(f"Config {action} - implementation pending")
