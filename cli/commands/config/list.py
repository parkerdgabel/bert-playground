"""List configuration command."""

import json
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table
from rich.tree import Tree

from ...config import ConfigManager
from ...utils import handle_errors


console = Console()


def _build_tree(data: dict, tree: Tree, show_values: bool = False) -> None:
    """Recursively build a tree from nested dictionary."""
    for key, value in data.items():
        if isinstance(value, dict):
            branch = tree.add(f"[cyan]{key}[/cyan]")
            _build_tree(value, branch, show_values)
        else:
            if show_values:
                if value is None:
                    display_value = "[dim]<not set>[/dim]"
                elif isinstance(value, bool):
                    display_value = f"[green]{value}[/green]" if value else f"[red]{value}[/red]"
                elif isinstance(value, (int, float)):
                    display_value = f"[yellow]{value}[/yellow]"
                elif isinstance(value, str) and len(value) > 50:
                    display_value = f"[green]{value[:47]}...[/green]"
                else:
                    display_value = f"[green]{value}[/green]"
                tree.add(f"[cyan]{key}:[/cyan] {display_value}")
            else:
                tree.add(f"[cyan]{key}[/cyan]")


@handle_errors
def list_command(
    values: bool = typer.Option(
        False,
        "--values",
        "-v",
        help="Show configuration values",
    ),
    source: Optional[str] = typer.Option(
        None,
        "--source",
        "-s",
        help="Show only from specific source (user, project, defaults)",
    ),
    json_output: bool = typer.Option(
        False,
        "--json",
        help="Output in JSON format",
    ),
):
    """List configuration structure and values.
    
    This command displays the configuration structure in a tree format,
    optionally showing values. You can filter by configuration source.
    
    Examples:
        # Show configuration structure
        k-bert config list
        
        # Show structure with values
        k-bert config list --values
        
        # Show only user configuration
        k-bert config list --source user --values
        
        # Output as JSON
        k-bert config list --json
    """
    manager = ConfigManager()
    
    # Determine which config to show
    if source == "user":
        config = manager.load_user_config()
        title = "User Configuration"
    elif source == "project":
        config = manager.load_project_config()
        if config is None:
            console.print("[yellow]No project configuration found.[/yellow]")
            raise typer.Exit(0)
        title = "Project Configuration"
    elif source == "defaults":
        from ...config import get_default_config
        config = get_default_config()
        title = "Default Configuration"
    else:
        # Show merged config
        config = manager.get_merged_config()
        title = "Merged Configuration (All Sources)"
    
    # Convert to dict
    config_dict = config.to_dict() if hasattr(config, 'to_dict') else config.model_dump(exclude_none=True)
    
    if json_output:
        console.print(json.dumps(config_dict, indent=2))
    else:
        # Create tree view
        tree = Tree(f"[bold]{title}[/bold]")
        _build_tree(config_dict, tree, show_values=values)
        console.print(tree)
        
        # Show sources info if showing merged config
        if source is None:
            console.print("\n[dim]Configuration sources (priority order):[/dim]")
            console.print("  1. CLI arguments (highest)")
            console.print("  2. Environment variables (K_BERT_*)")
            console.print(f"  3. Project config ({manager.project_config_path or 'none found'})")
            console.print(f"  4. User config ({manager.user_config_path})")
            console.print("  5. System defaults (lowest)")
        
        if not values:
            console.print("\n[dim]Tip: Use --values to show configuration values[/dim]")