"""Kaggle competitions commands."""

import typer
from typing import Optional

from ...utils import handle_errors, require_auth

competitions_app = typer.Typer()

@competitions_app.command(name="list")
@handle_errors
@require_auth("kaggle")
def list_competitions(
    category: Optional[str] = typer.Option(None, "--category", "-c", help="Filter by category"),
    search: Optional[str] = typer.Option(None, "--search", "-s", help="Search competitions"),
    active: bool = typer.Option(True, "--active/--all", help="Show only active competitions"),
    limit: int = typer.Option(20, "--limit", "-l", help="Number of competitions to show"),
):
    """List Kaggle competitions.
    
    Examples:
        bert kaggle competitions list
        bert kaggle competitions list --category tabular
        bert kaggle competitions list --search "natural language"
    """
    typer.echo("List competitions - implementation pending")

@competitions_app.command(name="info")
@handle_errors
@require_auth("kaggle")
def competition_info(
    competition: str = typer.Argument(..., help="Competition ID"),
    leaderboard: bool = typer.Option(False, "--leaderboard", "-l", help="Show leaderboard"),
):
    """Show detailed information about a competition.
    
    Examples:
        bert kaggle competitions info titanic
        bert kaggle competitions info titanic --leaderboard
    """
    typer.echo(f"Competition info for {competition} - implementation pending")