"""Track competition progress command."""

from typing import Optional
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

from ...config import get_config
from ...config.config_manager import ConfigManager
from ...utils import handle_errors


console = Console()


@handle_errors
def track_command(
    competition: Optional[str] = typer.Argument(
        None,
        help="Competition to track (uses project config if not specified)"
    ),
):
    """Track competition submissions and performance.
    
    Shows your submission history, best scores, and leaderboard position
    for the specified competition.
    
    Examples:
        # Track current project competition
        k-bert competition track
        
        # Track specific competition  
        k-bert competition track titanic
    """
    # Get competition name
    if not competition:
        # Try to get from project config
        config_manager = ConfigManager()
        project_config = config_manager.load_project_config()
        
        if project_config and hasattr(project_config, 'competition'):
            competition = project_config.competition
        else:
            console.print(
                "[red]No competition specified and no project configuration found.[/red]\n"
                "Usage: [cyan]k-bert competition track <competition>[/cyan]"
            )
            raise typer.Exit(1)
    
    # Get configuration
    config = get_config()
    
    # Check Kaggle credentials
    if not config.kaggle.username or not config.kaggle.key:
        console.print(
            "[red]Kaggle credentials not configured.[/red]\n"
            "Run: [cyan]k-bert config set kaggle.username <username>[/cyan]\n"
            "And: [cyan]k-bert config set kaggle.key <api_key>[/cyan]"
        )
        raise typer.Exit(1)
    
    try:
        # Set up Kaggle API
        import os
        os.environ["KAGGLE_USERNAME"] = config.kaggle.username
        os.environ["KAGGLE_KEY"] = config.kaggle.key
        
        import kaggle
        from kaggle.api.kaggle_api_extended import KaggleApi
        
        api = KaggleApi()
        api.authenticate()
        
        console.print(f"[dim]Fetching data for competition: {competition}[/dim]")
        
        # Get competition info
        try:
            comp_info = api.competition_view(competition)
        except Exception as e:
            console.print(f"[red]Competition '{competition}' not found or not accessible.[/red]")
            raise typer.Exit(1)
        
        # Get submission history
        submissions = api.competitions_submissions_list(competition)
        
        # Competition overview
        overview_content = (
            f"[bold]Competition:[/bold] {competition}\n"
            f"[bold]Title:[/bold] {getattr(comp_info, 'title', 'N/A')}\n"
            f"[bold]Submissions:[/bold] {len(submissions)}"
        )
        
        console.print(Panel(overview_content, title="Competition Status", border_style="blue"))
        
        if not submissions:
            console.print("[yellow]No submissions found for this competition.[/yellow]")
            return
        
        # Submissions table
        table = Table(title="Your Submissions", show_header=True)
        table.add_column("Date", style="cyan")
        table.add_column("Score", style="green") 
        table.add_column("Status", style="yellow")
        table.add_column("Description", style="dim")
        
        best_score = None
        best_score_numeric = None
        
        for sub in submissions:
            # Format date
            date_str = "N/A"
            if hasattr(sub, 'date') and sub.date:
                from datetime import datetime
                if isinstance(sub.date, datetime):
                    date_str = sub.date.strftime("%Y-%m-%d %H:%M")
                else:
                    date_str = str(sub.date)
            
            # Format score
            score_str = "N/A"
            if hasattr(sub, 'publicScore') and sub.publicScore is not None:
                score_str = f"{sub.publicScore:.5f}"
                try:
                    score_numeric = float(sub.publicScore)
                    if best_score_numeric is None or score_numeric > best_score_numeric:
                        best_score_numeric = score_numeric
                        best_score = score_str
                except (ValueError, TypeError):
                    pass
            
            # Status
            status = getattr(sub, 'status', 'complete')
            
            # Description  
            description = getattr(sub, 'description', '') or ''
            
            table.add_row(date_str, score_str, status, description)
        
        console.print(table)
        
        # Best score summary
        if best_score:
            console.print(f"\n[bold green]Your best score: {best_score}[/bold green]")
        
        # Get leaderboard position if possible
        try:
            leaderboard = api.competition_leaderboard_view(competition)
            if leaderboard and config.kaggle.username:
                for i, entry in enumerate(leaderboard, 1):
                    if entry.get('teamName') == config.kaggle.username:
                        console.print(f"[bold blue]Current leaderboard position: #{i}[/bold blue]")
                        break
        except Exception:
            # Leaderboard access might be restricted
            pass
        
        # Tips
        console.print(
            "\n[bold]Tips:[/bold]\n"
            "  • Use [cyan]k-bert competition submit <file>[/cyan] to make new submissions\n"
            "  • Use [cyan]k-bert competition leaderboard[/cyan] to see rankings\n"
            "  • Check submission logs in your Kaggle profile for details"
        )
    
    except ImportError:
        console.print(
            "[red]Kaggle package not installed.[/red]\n"
            "Install with: [cyan]uv add kaggle[/cyan]"
        )
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"[red]Failed to track competition: {e}[/red]")
        raise typer.Exit(1)