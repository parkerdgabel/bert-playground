"""Leaderboard viewing command."""

from pathlib import Path
from typing import Optional
import typer
import sys

from ...utils import (
    get_console, print_error, print_info, print_warning, print_success,
    handle_errors
)
from ...utils.console import create_table

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

@handle_errors
def leaderboard_command(
    competition: str = typer.Argument(..., help="Competition ID (e.g., titanic)"),
    top_n: int = typer.Option(20, "--top", "-t", help="Number of top entries to show"),
    save: Optional[Path] = typer.Option(None, "--save", "-s", help="Save leaderboard to file (CSV or JSON)"),
    show_change: bool = typer.Option(False, "--changes", help="Show position changes"),
    highlight_user: bool = typer.Option(True, "--highlight/--no-highlight", help="Highlight your submissions"),
):
    """View competition leaderboard.
    
    Displays the current leaderboard for a Kaggle competition, showing
    top performers and optionally highlighting your own submissions.
    
    Examples:
        # View top 20 on leaderboard
        bert kaggle leaderboard titanic
        
        # View top 50 with position changes
        bert kaggle leaderboard titanic --top 50 --changes
        
        # Save leaderboard to file
        bert kaggle leaderboard titanic --save leaderboard.csv
        
        # View without highlighting your entries
        bert kaggle leaderboard titanic --no-highlight
    """
    console = get_console()
    
    console.print(f"\n[bold blue]Kaggle Leaderboard: {competition}[/bold blue]")
    console.print("=" * 60)
    
    try:
        from utils.kaggle_integration import KaggleIntegration
    except ImportError:
        print_error(
            "Failed to import Kaggle integration. Make sure kaggle is installed:\n"
            "pip install kaggle",
            title="Import Error"
        )
        raise typer.Exit(1)
    
    try:
        kaggle = KaggleIntegration()
        
        # Get leaderboard data
        with console.status("[yellow]Fetching leaderboard...[/yellow]"):
            leaderboard = kaggle.get_leaderboard(competition, top_n=top_n)
        
        if not leaderboard:
            print_info("No leaderboard data available for this competition.")
            return
        
        # Get current user info if highlighting
        current_user = None
        if highlight_user:
            try:
                # This would need to be implemented in KaggleIntegration
                current_user = kaggle.get_current_username()
            except:
                pass
        
        # Create leaderboard table
        columns = ["Rank", "Team", "Score", "Entries"]
        if show_change:
            columns.insert(1, "Change")
        
        lb_table = create_table(f"Top {min(top_n, len(leaderboard))} Leaderboard", columns)
        
        # Add rows
        for i, entry in enumerate(leaderboard[:top_n]):
            rank = entry.get("rank", i + 1)
            team = entry.get("teamName", "Unknown")[:30]
            score = entry.get("score", 0)
            entries = entry.get("entries", 0)
            
            # Format score based on competition type
            if isinstance(score, float):
                score_str = f"{score:.6f}"
            else:
                score_str = str(score)
            
            # Check if this is the current user
            is_current_user = current_user and team == current_user
            
            # Format row
            if is_current_user:
                rank_str = f"[bold green]{rank}[/bold green]"
                team_str = f"[bold green]{team} (You)[/bold green]"
                score_str = f"[bold green]{score_str}[/bold green]"
                entries_str = f"[bold green]{entries}[/bold green]"
            else:
                rank_str = str(rank)
                team_str = team
                entries_str = str(entries)
            
            row = [rank_str, team_str, score_str, entries_str]
            
            # Add change column if requested
            if show_change:
                change = entry.get("change", 0)
                if change > 0:
                    change_str = f"[green]↑{change}[/green]"
                elif change < 0:
                    change_str = f"[red]↓{abs(change)}[/red]"
                else:
                    change_str = "→"
                row.insert(1, change_str)
            
            lb_table.add_row(*row)
        
        console.print(lb_table)
        
        # Show competition stats
        if len(leaderboard) > 0:
            best_score = leaderboard[0].get("score", 0)
            total_teams = len(leaderboard)
            console.print(f"\n[cyan]Best Score: {best_score}[/cyan]")
            console.print(f"[cyan]Total Teams: {total_teams}[/cyan]")
        
        # Save if requested
        if save:
            if save.suffix == ".csv":
                import csv
                with open(save, "w", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow(["rank", "team", "score", "entries"])
                    for entry in leaderboard:
                        writer.writerow([
                            entry.get("rank", ""),
                            entry.get("teamName", ""),
                            entry.get("score", ""),
                            entry.get("entries", "")
                        ])
            elif save.suffix == ".json":
                import json
                with open(save, "w") as f:
                    json.dump(leaderboard, f, indent=2)
            else:
                print_warning(f"Unsupported file format: {save.suffix}. Use .csv or .json")
                return
            
            print_success(f"Leaderboard saved to: {save}")
        
        # Show next steps
        console.print("\n[bold]Next steps:[/bold]")
        console.print(f"1. View your submissions: [cyan]bert kaggle history {competition}[/cyan]")
        console.print(f"2. Make a new submission: [cyan]bert kaggle submit {competition} predictions.csv[/cyan]")
        console.print(f"3. Download competition data: [cyan]bert kaggle download {competition}[/cyan]")
        
    except Exception as e:
        print_error(f"Failed to fetch leaderboard: {str(e)}", title="Leaderboard Error")
        raise typer.Exit(1)