"""Initialize configuration command."""

from pathlib import Path

import typer
from rich.console import Console
from rich.prompt import Confirm, Prompt

from ...config import ConfigManager
from ...utils import handle_errors


console = Console()


@handle_errors
def init_command(
    interactive: bool = typer.Option(
        True,
        "--interactive/--no-interactive",
        help="Run in interactive mode",
    ),
    force: bool = typer.Option(
        False,
        "--force",
        "-f",
        help="Overwrite existing configuration",
    ),
):
    """Initialize k-bert configuration.
    
    This command creates a user configuration file at ~/.k-bert/config.yaml
    with default settings. In interactive mode, it will prompt for common
    configuration values.
    
    Examples:
        # Interactive setup
        k-bert config init
        
        # Non-interactive with defaults
        k-bert config init --no-interactive
        
        # Force overwrite existing config
        k-bert config init --force
    """
    manager = ConfigManager()
    
    # Check if config already exists
    if manager.user_config_path.exists() and not force:
        console.print(
            f"[yellow]Configuration already exists at {manager.user_config_path}[/yellow]"
        )
        
        if interactive:
            if not Confirm.ask("Overwrite existing configuration?", default=False):
                console.print("[red]Configuration initialization cancelled.[/red]")
                raise typer.Exit(0)
        else:
            console.print(
                "[red]Use --force to overwrite existing configuration.[/red]"
            )
            raise typer.Exit(1)
    
    # Initialize configuration
    try:
        config = manager.init_user_config(interactive=interactive)
        
        if not interactive:
            console.print(
                f"[green]Created default configuration at {manager.user_config_path}[/green]"
            )
        
        # Show next steps
        console.print("\n[bold]Next steps:[/bold]")
        console.print("1. Set your Kaggle credentials:")
        console.print("   [cyan]k-bert config set kaggle.username YOUR_USERNAME[/cyan]")
        console.print("   [cyan]k-bert config set kaggle.key YOUR_API_KEY[/cyan]")
        console.print("\n2. Download competition data:")
        console.print("   [cyan]k-bert competition download titanic[/cyan]")
        console.print("\n3. Start training:")
        console.print("   [cyan]k-bert train --train data/train.csv --val data/val.csv[/cyan]")
        
    except Exception as e:
        console.print(f"[red]Failed to initialize configuration: {e}[/red]")
        raise typer.Exit(1)