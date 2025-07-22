"""Set configuration value command."""

import json
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console

from ...config import ConfigManager, ConfigValidationError
from ...utils import handle_errors


console = Console()


def _parse_value(value_str: str):
    """Parse a string value to appropriate type."""
    # Try to parse as JSON first
    try:
        return json.loads(value_str)
    except json.JSONDecodeError:
        pass
    
    # Check for boolean values
    if value_str.lower() in ["true", "yes", "on"]:
        return True
    elif value_str.lower() in ["false", "no", "off"]:
        return False
    
    # Check for numeric values
    try:
        if "." in value_str:
            return float(value_str)
        else:
            return int(value_str)
    except ValueError:
        pass
    
    # Check for path values
    if "/" in value_str or "\\" in value_str:
        return Path(value_str)
    
    # Return as string
    return value_str


@handle_errors
def set_command(
    key: str = typer.Argument(
        ...,
        help="Configuration key (e.g., kaggle.username)",
    ),
    value: str = typer.Argument(
        ...,
        help="Value to set",
    ),
    global_config: bool = typer.Option(
        True,
        "--global/--project",
        help="Set in global config (default) or project config",
    ),
    no_save: bool = typer.Option(
        False,
        "--no-save",
        help="Don't save to file (preview only)",
    ),
):
    """Set configuration values.
    
    This command sets configuration values in either the global user
    configuration (~/.k-bert/config.yaml) or the project configuration.
    
    Values are automatically parsed to appropriate types:
    - Booleans: true/false, yes/no, on/off
    - Numbers: integers and floats
    - Paths: strings containing / or \\
    - JSON: valid JSON strings for complex values
    
    Examples:
        # Set simple values
        k-bert config set kaggle.username myusername
        k-bert config set training.default_epochs 10
        k-bert config set training.save_best_only true
        
        # Set complex values with JSON
        k-bert config set mlflow.tags '{"experiment": "v2", "model": "bert"}'
        
        # Set in project config
        k-bert config set --project kaggle.default_competition titanic
        
        # Preview without saving
        k-bert config set --no-save training.default_batch_size 64
    """
    manager = ConfigManager()
    
    # Parse the value
    parsed_value = _parse_value(value)
    
    try:
        if global_config:
            # Set in user config
            manager.set_value(key, parsed_value, save=not no_save)
            
            if no_save:
                console.print(
                    f"[yellow]Preview:[/yellow] Would set [cyan]{key}[/cyan] = "
                    f"[green]{parsed_value}[/green] in global config"
                )
            else:
                console.print(
                    f"[green]✓[/green] Set [cyan]{key}[/cyan] = "
                    f"[green]{parsed_value}[/green] in global config"
                )
        else:
            # Set in project config
            project_config = manager.load_project_config()
            
            if project_config is None:
                console.print(
                    "[red]No project configuration found. "
                    "Run 'k-bert init' to create one.[/red]"
                )
                raise typer.Exit(1)
            
            # Navigate to the correct nested location
            parts = key.split(".")
            current = project_config
            
            # Create nested structure if needed
            for i, part in enumerate(parts[:-1]):
                if not hasattr(current, part):
                    console.print(f"[red]Invalid configuration key: {key}[/red]")
                    raise typer.Exit(1)
                    
                next_val = getattr(current, part)
                if next_val is None:
                    # Create the nested object
                    from ...config.schemas import (
                        KaggleConfig, ModelConfig, TrainingConfig,
                        MLflowConfig, DataConfig, LoggingConfig
                    )
                    
                    type_map = {
                        "kaggle": KaggleConfig,
                        "models": ModelConfig,
                        "training": TrainingConfig,
                        "mlflow": MLflowConfig,
                        "data": DataConfig,
                        "logging": LoggingConfig,
                    }
                    
                    if part in type_map:
                        setattr(current, part, type_map[part]())
                        next_val = getattr(current, part)
                    else:
                        console.print(f"[red]Cannot create nested config: {part}[/red]")
                        raise typer.Exit(1)
                
                current = next_val
            
            # Set the final value
            final_key = parts[-1]
            if hasattr(current, final_key):
                setattr(current, final_key, parsed_value)
            else:
                console.print(f"[red]Invalid configuration key: {key}[/red]")
                raise typer.Exit(1)
            
            if not no_save:
                manager.save_project_config(project_config)
            
            if no_save:
                console.print(
                    f"[yellow]Preview:[/yellow] Would set [cyan]{key}[/cyan] = "
                    f"[green]{parsed_value}[/green] in project config"
                )
            else:
                console.print(
                    f"[green]✓[/green] Set [cyan]{key}[/cyan] = "
                    f"[green]{parsed_value}[/green] in project config"
                )
        
        # Show related configuration
        if key.startswith("kaggle.") and not no_save:
            if key == "kaggle.username":
                console.print(
                    "\n[dim]Don't forget to set your API key:[/dim]\n"
                    "  [cyan]k-bert config set kaggle.key YOUR_API_KEY[/cyan]"
                )
            elif key == "kaggle.key":
                console.print(
                    "\n[dim]API key set. You can now download competition data:[/dim]\n"
                    "  [cyan]k-bert competition download titanic[/cyan]"
                )
    
    except ConfigValidationError as e:
        console.print(f"[red]Configuration validation error:[/red]")
        console.print(str(e))
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"[red]Failed to set configuration: {e}[/red]")
        raise typer.Exit(1)