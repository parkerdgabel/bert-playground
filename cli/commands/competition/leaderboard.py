"""View competition leaderboard command."""

from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

from ...config import get_config
from ...utils import handle_errors


console = Console()


@handle_errors
def leaderboard_command(
    competition: str = typer.Argument(..., help="Competition name"),
    top: int = typer.Option(10, "--top", "-n", help="Number of entries to show"),
    page: int = typer.Option(1, "--page", "-p", help="Page number for pagination"),
):
    """View competition leaderboard.
    
    Shows the current public leaderboard for the specified competition.
    
    Examples:
        # View top 10 entries
        k-bert competition leaderboard titanic
        
        # View top 20 entries
        k-bert competition leaderboard titanic --top 20
        
        # View specific page
        k-bert competition leaderboard titanic --page 2
    """
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
        
        console.print(f"[dim]Fetching leaderboard for: {competition}[/dim]")
        
        # Get leaderboard
        try:
            leaderboard = api.competition_leaderboard_view(competition)
        except Exception as e:
            console.print(f"[red]Could not access leaderboard for '{competition}'.[/red]")
            console.print(f"[dim]Error: {e}[/dim]")
            raise typer.Exit(1)
        
        if not leaderboard:
            console.print("[yellow]No leaderboard data available.[/yellow]")
            return
        
        # Create table
        table = Table(title=f"Leaderboard: {competition}", show_header=True)
        table.add_column("Rank", style="cyan", justify="right")
        table.add_column("Team", style="bold")
        table.add_column("Score", style="green", justify="right")
        table.add_column("Entries", style="yellow", justify="right")
        table.add_column("Last Submission", style="dim")
        
        # Calculate pagination
        start_idx = (page - 1) * top
        end_idx = min(start_idx + top, len(leaderboard))
        
        if start_idx >= len(leaderboard):
            console.print(f"[yellow]Page {page} is beyond available data.[/yellow]")
            return
        
        # Add entries to table
        for i in range(start_idx, end_idx):
            entry = leaderboard[i]
            rank = i + 1
            
            # Get team name
            team_name = entry.get('teamName', 'Unknown')
            
            # Highlight current user
            if team_name == config.kaggle.username:
                team_name = f"[bold green]{team_name} (You)[/bold green]"
            
            # Get score
            score = entry.get('score', 'N/A')
            if isinstance(score, (int, float)):
                score = f"{score:.5f}"
            
            # Get submission count
            entries_count = entry.get('submissionCount', 'N/A')
            
            # Get last submission date
            last_submission = entry.get('lastSubmissionDate', '')
            if last_submission:
                try:
                    from datetime import datetime
                    if isinstance(last_submission, str):
                        # Try to parse and format date
                        dt = datetime.fromisoformat(last_submission.replace('Z', '+00:00'))
                        last_submission = dt.strftime("%Y-%m-%d")
                except Exception:
                    pass
            
            table.add_row(
                str(rank),
                team_name,
                str(score),
                str(entries_count),
                str(last_submission)
            )
        
        console.print(table)
        
        # Show pagination info
        total_entries = len(leaderboard)
        if total_entries > top:
            total_pages = (total_entries + top - 1) // top
            console.print(
                f"\n[dim]Showing entries {start_idx + 1}-{end_idx} of {total_entries} "
                f"(Page {page} of {total_pages})[/dim]"
            )
            
            if page < total_pages:
                console.print(
                    f"[dim]Use [cyan]--page {page + 1}[/cyan] to see more entries.[/dim]"
                )
        
        # Find user position if not in current view
        user_rank = None
        for i, entry in enumerate(leaderboard, 1):
            if entry.get('teamName') == config.kaggle.username:
                user_rank = i
                break
        
        if user_rank and (user_rank < start_idx + 1 or user_rank > end_idx):
            user_page = (user_rank - 1) // top + 1
            console.print(
                f"\n[bold blue]Your rank: #{user_rank}[/bold blue] "
                f"[dim](use [cyan]--page {user_page}[/cyan] to see your position)[/dim]"
            )
        
        # Tips
        console.print(
            "\n[bold]Tips:[/bold]\n"
            "  • Use [cyan]k-bert competition track[/cyan] to see your submission history\n"
            "  • Use [cyan]k-bert competition submit <file>[/cyan] to improve your score\n"
            "  • Check competition rules for submission limits"
        )
    
    except ImportError:
        console.print(
            "[red]Kaggle package not installed.[/red]\n"
            "Install with: [cyan]uv add kaggle[/cyan]"
        )
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"[red]Failed to fetch leaderboard: {e}[/red]")
        raise typer.Exit(1)