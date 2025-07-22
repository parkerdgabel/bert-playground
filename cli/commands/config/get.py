"""Get configuration value command."""

import json
from typing import Optional

import typer
from rich.console import Console
from rich.syntax import Syntax
from rich.table import Table

from ...config import ConfigManager
from ...utils import handle_errors


console = Console()


@handle_errors
def get_command(
    key: Optional[str] = typer.Argument(
        None,
        help="Configuration key (e.g., kaggle.username)",
    ),
    all: bool = typer.Option(
        False,
        "--all",
        "-a",
        help="Show all configuration values",
    ),
    json_output: bool = typer.Option(
        False,
        "--json",
        help="Output in JSON format",
    ),
):
    """Get configuration values.
    
    This command retrieves configuration values from the merged configuration
    (combining defaults, user config, project config, and environment variables).
    
    Examples:
        # Get a specific value
        k-bert config get kaggle.username
        
        # Get all values
        k-bert config get --all
        
        # Get as JSON
        k-bert config get --all --json
    """
    manager = ConfigManager()
    
    if all:
        # Show all configuration values
        all_values = manager.list_all_values()
        
        if json_output:
            console.print(json.dumps(all_values, indent=2))
        else:
            # Create a table
            table = Table(title="K-BERT Configuration", show_header=True)
            table.add_column("Key", style="cyan")
            table.add_column("Value", style="green")
            
            # Sort keys for better display
            for key in sorted(all_values.keys()):
                value = all_values[key]
                
                # Format value for display
                if isinstance(value, (dict, list)):
                    value_str = json.dumps(value, indent=2)
                elif value is None:
                    value_str = "[dim]<not set>[/dim]"
                else:
                    value_str = str(value)
                
                table.add_row(key, value_str)
            
            console.print(table)
    
    elif key:
        # Get specific value
        value = manager.get_value(key)
        
        if value is None:
            console.print(f"[yellow]Configuration key '{key}' not found or not set.[/yellow]")
            raise typer.Exit(1)
        
        if json_output:
            console.print(json.dumps({key: value}, indent=2))
        else:
            # Format value for display
            if isinstance(value, (dict, list)):
                # Pretty print complex values
                syntax = Syntax(
                    json.dumps(value, indent=2),
                    "json",
                    theme="monokai",
                    line_numbers=False,
                )
                console.print(f"[cyan]{key}:[/cyan]")
                console.print(syntax)
            else:
                console.print(f"[cyan]{key}:[/cyan] [green]{value}[/green]")
    
    else:
        # No key specified and not --all
        console.print("[red]Please specify a key or use --all to show all values.[/red]")
        console.print("\nExamples:")
        console.print("  k-bert config get kaggle.username")
        console.print("  k-bert config get --all")
        raise typer.Exit(1)