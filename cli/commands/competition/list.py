"""List competitions command."""

from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

from ...config import get_config, COMPETITION_PROFILES
from ...utils import handle_errors


console = Console()


@handle_errors
def list_command(
    active: bool = typer.Option(
        False,
        "--active",
        "-a",
        help="Show only active competitions",
    ),
    search: Optional[str] = typer.Option(
        None,
        "--search",
        "-s",
        help="Search competitions by keyword",
    ),
    category: Optional[str] = typer.Option(
        None,
        "--category",
        "-c",
        help="Filter by category",
    ),
    sort_by: str = typer.Option(
        "deadline",
        "--sort",
        help="Sort by: deadline, prize, teams, entries",
    ),
    limit: int = typer.Option(
        20,
        "--limit",
        "-n",
        help="Number of competitions to show",
    ),
    show_profiles: bool = typer.Option(
        False,
        "--profiles",
        help="Show built-in competition profiles instead",
    ),
):
    """List Kaggle competitions.
    
    This command lists available Kaggle competitions with their details.
    Can filter by status, category, or search term.
    
    Examples:
        # List all competitions
        k-bert competition list
        
        # List active competitions
        k-bert competition list --active
        
        # Search for NLP competitions
        k-bert competition list --search nlp
        
        # Show built-in profiles
        k-bert competition list --profiles
    """
    # Show built-in profiles if requested
    if show_profiles:
        table = Table(title="Built-in Competition Profiles", show_header=True)
        table.add_column("Competition", style="cyan")
        table.add_column("Type", style="green")
        table.add_column("Metrics", style="yellow")
        table.add_column("Features", style="blue")
        
        for name, profile in COMPETITION_PROFILES.items():
            metrics = ", ".join(profile.metrics)
            features = f"{len(profile.feature_columns or [])} features"
            if profile.text_columns:
                features += f", {len(profile.text_columns)} text"
            
            table.add_row(
                name,
                profile.type.replace("_", " ").title(),
                metrics,
                features,
            )
        
        console.print(table)
        console.print(
            "\n[dim]These profiles have optimized settings for k-bert. "
            "Use them with:[/dim]\n"
            "  [cyan]k-bert competition init <name>[/cyan]"
        )
        return
    
    # Get configuration
    config = get_config()
    
    # Check Kaggle credentials
    if not config.kaggle.username or not config.kaggle.key:
        console.print(
            "[yellow]Kaggle credentials not configured.[/yellow]\n"
            "Showing built-in profiles instead.\n"
        )
        list_command(
            active=active,
            search=search,
            category=category,
            sort_by=sort_by,
            limit=limit,
            show_profiles=True,
        )
        return
    
    try:
        # Set up Kaggle API
        import os
        os.environ["KAGGLE_USERNAME"] = config.kaggle.username
        os.environ["KAGGLE_KEY"] = config.kaggle.key
        
        import kaggle
        from kaggle.api.kaggle_api_extended import KaggleApi
        
        api = KaggleApi()
        api.authenticate()
        
        # Get competitions list
        console.print("[dim]Fetching competitions from Kaggle...[/dim]")
        
        competitions = api.competitions_list(
            group="general" if not active else None,
            category=category,
            sort_by=sort_by,
            page=1,
            search=search,
        )
        
        if not competitions:
            console.print("[yellow]No competitions found matching criteria.[/yellow]")
            return
        
        # Create table
        table = Table(title="Kaggle Competitions", show_header=True)
        table.add_column("Competition", style="cyan", no_wrap=True)
        table.add_column("Deadline", style="yellow")
        table.add_column("Prize", style="green")
        table.add_column("Teams", style="blue")
        table.add_column("Category", style="magenta")
        
        # Add competitions to table
        count = 0
        for comp in competitions:
            if count >= limit:
                break
            
            # Format deadline
            deadline = "No deadline"
            if hasattr(comp, 'deadline') and comp.deadline:
                from datetime import datetime
                if isinstance(comp.deadline, datetime):
                    deadline = comp.deadline.strftime("%Y-%m-%d")
                else:
                    deadline = str(comp.deadline)
            
            # Format prize
            prize = comp.rewardDisplay if hasattr(comp, 'rewardDisplay') else "Knowledge"
            
            # Format teams
            teams = str(comp.teamCount) if hasattr(comp, 'teamCount') else "0"
            
            # Get category
            category_name = comp.category if hasattr(comp, 'category') else "General"
            
            # Add profile indicator if we have a built-in profile
            name = comp.ref
            if name in COMPETITION_PROFILES:
                name = f"{name} ✓"
            
            table.add_row(
                name,
                deadline,
                prize,
                teams,
                category_name,
            )
            count += 1
        
        console.print(table)
        
        if count < len(competitions):
            console.print(
                f"\n[dim]Showing {count} of {len(competitions)} competitions. "
                f"Use --limit to see more.[/dim]"
            )
        
        # Show tips
        console.print(
            "\n[bold]Tips:[/bold]\n"
            "  • ✓ indicates competitions with built-in k-bert profiles\n"
            "  • Use [cyan]k-bert competition download <name>[/cyan] to get data\n"
            "  • Use [cyan]k-bert competition info <name>[/cyan] for details"
        )
    
    except ImportError:
        console.print(
            "[red]Kaggle package not installed.[/red]\n"
            "Showing built-in profiles instead.\n"
        )
        list_command(
            active=active,
            search=search,
            category=category,
            sort_by=sort_by,
            limit=limit,
            show_profiles=True,
        )
    except Exception as e:
        console.print(f"[red]Failed to fetch competitions: {e}[/red]")
        console.print("\nShowing built-in profiles instead.\n")
        list_command(
            active=active,
            search=search,
            category=category,
            sort_by=sort_by,
            limit=limit,
            show_profiles=True,
        )