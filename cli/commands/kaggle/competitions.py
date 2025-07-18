"""Competition listing and search commands."""

from typing import Optional
import typer
from pathlib import Path
import sys

from ...utils import (
    get_console, print_error, print_info,
    handle_errors
)
from ...utils.console import create_table

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

@handle_errors
def competitions_command(
    category: Optional[str] = typer.Option(None, "--category", "-c", help="Filter by category (e.g., 'tabular', 'nlp', 'cv')"),
    search: Optional[str] = typer.Option(None, "--search", "-s", help="Search competitions by keyword"),
    sort_by: str = typer.Option("latestDeadline", "--sort", help="Sort by: latestDeadline, prize, numberOfTeams, recentlyCreated"),
    limit: int = typer.Option(20, "--limit", "-l", help="Number of competitions to show (max: 50)"),
    active_only: bool = typer.Option(True, "--active/--all", help="Show only active competitions"),
    show_tags: bool = typer.Option(False, "--tags", help="Show competition tags"),
):
    """List and search Kaggle competitions.
    
    This command helps you discover competitions on Kaggle. You can filter
    by category, search by keywords, and sort by various criteria.
    
    Examples:
        # List all active tabular competitions
        bert kaggle competitions --category tabular
        
        # Search for NLP competitions with prize money
        bert kaggle competitions --search "nlp" --sort prize
        
        # Show recent competitions with tags
        bert kaggle competitions --sort recentlyCreated --tags
        
        # List all competitions (including completed)
        bert kaggle competitions --all --limit 30
    """
    console = get_console()
    
    console.print("\n[bold blue]Kaggle Competitions[/bold blue]")
    console.print("=" * 60)
    
    try:
        from utils.kaggle_integration import KaggleIntegration
    except ImportError as e:
        print_error(
            "Failed to import Kaggle integration. Make sure kaggle is installed:\n"
            "pip install kaggle",
            title="Import Error"
        )
        raise typer.Exit(1)
    
    try:
        kaggle = KaggleIntegration()
        competitions = kaggle.list_competitions(
            category=category,
            search=search,
            sort_by=sort_by,
            page=1  # KaggleIntegration expects 'page', not 'page_size'
        )
        
        if competitions.empty:
            print_info("No competitions found matching your criteria.")
            return
        
        # Convert DataFrame to list of dicts for easier manipulation
        competitions_list = competitions.to_dict('records')
        
        # Filter active competitions if requested
        if active_only:
            competitions_list = [c for c in competitions_list if not c.get("isCompleted", True)]
            if not competitions_list:
                print_info("No active competitions found. Use --all to see completed ones.")
                return
        
        # Create competitions table
        columns = ["Competition", "Category", "Teams", "Prize", "Deadline", "Status"]
        if show_tags:
            columns.append("Tags")
        
        comp_table = create_table("Available Competitions", columns)
        
        for comp in competitions_list[:limit]:
            # Format competition data
            comp_id = comp.get("id", "unknown")
            title = comp.get("title", "Unknown")[:50]
            # Category might not be in the response, use evaluationMetric as fallback
            category_val = comp.get("category", comp.get("evaluationMetric", "Unknown"))
            teams = str(comp.get("numTeams", 0))
            
            # Format prize
            prize = comp.get("reward", None)
            if prize:
                prize_str = prize
            else:
                prize_str = "Knowledge"
            
            # Format deadline
            deadline = comp.get("deadline", "")
            if deadline:
                from datetime import datetime
                import pandas as pd
                try:
                    # Handle pandas Timestamp objects
                    if isinstance(deadline, pd.Timestamp):
                        deadline_dt = deadline.to_pydatetime()
                    else:
                        # Handle string dates
                        deadline_dt = datetime.fromisoformat(str(deadline).replace("Z", "+00:00"))
                    
                    deadline_str = deadline_dt.strftime("%Y-%m-%d")
                    days_left = (deadline_dt - datetime.now(deadline_dt.tzinfo)).days
                    if days_left >= 0:
                        deadline_str += f" ({days_left}d)"
                except Exception as e:
                    # Fallback to string representation
                    deadline_str = str(deadline)[:10]
            else:
                deadline_str = "No deadline"
            
            # Status  
            if comp.get("isKernelsSubmissionsOnly", False):
                status = "[yellow]Kernels Only[/yellow]"
            elif comp.get("userHasEntered", False):
                status = "[cyan]Entered[/cyan]"
            else:
                status = "[green]Active[/green]"
            
            row = [f"[cyan]{comp_id}[/cyan]\n{title}", category_val, teams, prize_str, deadline_str, status]
            
            # Add tags if requested
            if show_tags:
                tags = comp.get("tags", [])
                tag_str = ", ".join(tags[:3])
                if len(tags) > 3:
                    tag_str += f" (+{len(tags)-3})"
                row.append(tag_str)
            
            comp_table.add_row(*row)
        
        console.print(comp_table)
        
        # Show summary
        console.print(f"\n[cyan]Showing {min(limit, len(competitions_list))} of {len(competitions_list)} competitions[/cyan]")
        
        # Show next steps
        console.print("\n[bold]Next steps:[/bold]")
        console.print("  • Download competition data: [cyan]bert kaggle download COMPETITION_ID[/cyan]")
        console.print("  • View leaderboard: [cyan]bert kaggle leaderboard COMPETITION_ID[/cyan]")
        console.print("  • Submit predictions: [cyan]bert kaggle submit COMPETITION_ID submission.csv[/cyan]")
        
    except Exception as e:
        print_error(f"Failed to list competitions: {str(e)}", title="Kaggle Error")
        raise typer.Exit(1)