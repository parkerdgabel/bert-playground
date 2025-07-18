"""System information command."""

import typer

from ...utils import handle_errors

@handle_errors
def info_command(
    mlx: bool = typer.Option(False, "--mlx", help="Show MLX information"),
    models: bool = typer.Option(False, "--models", help="List available models"),
    datasets: bool = typer.Option(False, "--datasets", help="List configured datasets"),
    all: bool = typer.Option(False, "--all", "-a", help="Show all information"),
):
    """Display system and project information.
    
    Examples:
        bert info
        bert info --mlx
        bert info --all
    """
    # Implementation will be migrated from original CLI
    typer.echo("Info command - implementation pending")