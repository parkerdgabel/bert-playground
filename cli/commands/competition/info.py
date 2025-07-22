"""Show competition information command."""

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from ...config import get_config, COMPETITION_PROFILES
from ...utils import handle_errors


console = Console()


@handle_errors
def info_command(
    competition: str = typer.Argument(
        ...,
        help="Competition name (e.g., titanic)",
    ),
    files: bool = typer.Option(
        False,
        "--files",
        "-f",
        help="List competition files",
    ),
    leaderboard: bool = typer.Option(
        False,
        "--leaderboard",
        "-l",
        help="Show top leaderboard entries",
    ),
):
    """Show detailed information about a competition.
    
    This command displays information about a Kaggle competition including
    description, evaluation metric, prizes, and timeline.
    
    Examples:
        # Show competition info
        k-bert competition info titanic
        
        # Show competition files
        k-bert competition info titanic --files
        
        # Show leaderboard
        k-bert competition info titanic --leaderboard
    """
    # Check if we have a built-in profile
    if competition in COMPETITION_PROFILES:
        profile = COMPETITION_PROFILES[competition]
        
        # Show built-in profile info
        console.print(
            Panel(
                f"[bold cyan]{competition.upper()}[/bold cyan]\n"
                f"Built-in k-bert Competition Profile",
                style="cyan",
            )
        )
        
        # Basic info
        info_table = Table(show_header=False, box=None)
        info_table.add_column("Field", style="dim")
        info_table.add_column("Value")
        
        info_table.add_row("Type", profile.type.replace("_", " ").title())
        info_table.add_row("Metrics", ", ".join(profile.metrics))
        info_table.add_row("Target Column", profile.target_column or "N/A")
        info_table.add_row("ID Column", profile.id_column or "N/A")
        
        if profile.text_columns:
            info_table.add_row("Text Columns", ", ".join(profile.text_columns))
        
        if profile.recommended_models:
            info_table.add_row("Recommended Models", profile.recommended_models[0])
        
        info_table.add_row("Batch Size", str(profile.recommended_batch_size))
        info_table.add_row("Max Length", str(profile.recommended_max_length))
        
        console.print(info_table)
        
        # Quick start
        console.print(
            "\n[bold]Quick Start:[/bold]\n"
            f"1. Download data: [cyan]k-bert competition download {competition}[/cyan]\n"
            f"2. Initialize project: [cyan]k-bert competition init {competition}[/cyan]\n"
            f"3. Start training: [cyan]k-bert run[/cyan]"
        )
    
    # Try to fetch from Kaggle API
    config = get_config()
    
    if not config.kaggle.username or not config.kaggle.key:
        if competition not in COMPETITION_PROFILES:
            console.print(
                "[red]Kaggle credentials not configured and no built-in profile found.[/red]\n"
                "Configure credentials to see full competition details:\n"
                "  [cyan]k-bert config set kaggle.username YOUR_USERNAME[/cyan]\n"
                "  [cyan]k-bert config set kaggle.key YOUR_API_KEY[/cyan]"
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
        
        # Get competition info
        console.print(f"\n[dim]Fetching competition details from Kaggle...[/dim]")
        
        try:
            # Get competition details
            comp_list = api.competitions_list(search=competition)
            comp = None
            
            # Find exact match
            for c in comp_list:
                if c.ref == competition:
                    comp = c
                    break
            
            if not comp:
                console.print(
                    f"[yellow]Competition '{competition}' not found on Kaggle.[/yellow]"
                )
                return
            
            # Show competition panel
            console.print(
                Panel(
                    f"[bold cyan]{comp.title}[/bold cyan]\n"
                    f"{comp.description[:200]}...",
                    style="cyan",
                )
            )
            
            # Competition details
            details_table = Table(show_header=False, box=None)
            details_table.add_column("Field", style="dim")
            details_table.add_column("Value")
            
            details_table.add_row("URL", f"https://www.kaggle.com/c/{competition}")
            details_table.add_row("Deadline", str(comp.deadline) if comp.deadline else "No deadline")
            details_table.add_row("Prize", comp.rewardDisplay)
            details_table.add_row("Teams", str(comp.teamCount))
            details_table.add_row("Category", comp.category)
            
            console.print(details_table)
            
            # Show files if requested
            if files:
                console.print("\n[bold]Competition Files:[/bold]")
                try:
                    comp_files = api.competition_list_files(competition)
                    
                    file_table = Table(show_header=True)
                    file_table.add_column("File", style="cyan")
                    file_table.add_column("Size", style="green")
                    
                    for f in comp_files:
                        size_mb = f.size / (1024 * 1024) if hasattr(f, 'size') else 0
                        file_table.add_row(f.name, f"{size_mb:.1f} MB")
                    
                    console.print(file_table)
                except Exception as e:
                    console.print(f"[yellow]Could not fetch file list: {e}[/yellow]")
            
            # Show leaderboard if requested
            if leaderboard:
                console.print("\n[bold]Leaderboard (Top 10):[/bold]")
                try:
                    lb = api.competition_leaderboard(competition)[:10]
                    
                    lb_table = Table(show_header=True)
                    lb_table.add_column("Rank", style="yellow")
                    lb_table.add_column("Team", style="cyan")
                    lb_table.add_column("Score", style="green")
                    lb_table.add_column("Entries", style="blue")
                    
                    for entry in lb:
                        lb_table.add_row(
                            str(entry.rank) if hasattr(entry, 'rank') else "?",
                            entry.teamName[:30] if hasattr(entry, 'teamName') else "Unknown",
                            str(entry.score) if hasattr(entry, 'score') else "?",
                            str(entry.entries) if hasattr(entry, 'entries') else "?",
                        )
                    
                    console.print(lb_table)
                except Exception as e:
                    console.print(f"[yellow]Could not fetch leaderboard: {e}[/yellow]")
            
        except Exception as e:
            if "403" in str(e):
                console.print(
                    f"[yellow]Access denied. You may need to accept competition rules.[/yellow]"
                )
            else:
                console.print(f"[red]Failed to fetch competition info: {e}[/red]")
    
    except ImportError:
        if competition not in COMPETITION_PROFILES:
            console.print(
                "[red]Kaggle package not installed and no built-in profile found.[/red]"
            )