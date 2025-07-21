"""Competition listing command with API contracts enforced.

This is an example of how to use the contract validation utilities
to ensure API stability.
"""

import sys
from pathlib import Path

import typer

from ...utils import (
    ParameterContract,
    backward_compatible,
    get_console,
    handle_errors,
    print_error,
    print_info,
    validate_contract,
)

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))


# Define the contract for KaggleIntegration.list_competitions
class ListCompetitionsContract(ParameterContract):
    """Contract for list_competitions parameters."""

    category: str | None = None
    search: str | None = None
    sort_by: str = "latestDeadline"
    page: int = 1  # NOT page_size!


@handle_errors
@backward_compatible(
    {
        "page_size": "page",  # Support old parameter name with warning
    }
)
@validate_contract(
    expected_params={
        "category": (str, type(None)),
        "search": (str, type(None)),
        "sort_by": str,
        "limit": int,
        "active_only": bool,
        "show_tags": bool,
    }
)
def competitions_command(
    category: str | None = typer.Option(
        None, "--category", "-c", help="Filter by category"
    ),
    search: str | None = typer.Option(None, "--search", "-s", help="Search by keyword"),
    sort_by: str = typer.Option("latestDeadline", "--sort", help="Sort by field"),
    limit: int = typer.Option(20, "--limit", "-l", help="Number to show"),
    active_only: bool = typer.Option(True, "--active/--all", help="Show only active"),
    show_tags: bool = typer.Option(False, "--tags", help="Show tags"),
):
    """List Kaggle competitions with contract validation."""
    console = get_console()

    console.print("\n[bold blue]Kaggle Competitions (with Contracts)[/bold blue]")
    console.print("=" * 60)

    try:
        from utils.kaggle_integration import KaggleIntegration
    except ImportError:
        print_error("Failed to import Kaggle integration", title="Import Error")
        raise typer.Exit(1)

    try:
        kaggle = KaggleIntegration()

        # Validate parameters against contract before calling
        params = ListCompetitionsContract(
            category=category, search=search, sort_by=sort_by, page=1
        )

        # Call with validated parameters
        competitions = kaggle.list_competitions(**params.dict())

        if competitions.empty:
            print_info("No competitions found matching your criteria.")
            return

        # Rest of the implementation remains the same...
        _display_competitions(console, competitions, limit, active_only, show_tags)

    except Exception as e:
        print_error(f"Failed to list competitions: {str(e)}", title="Kaggle Error")
        raise typer.Exit(1)


def _display_competitions(console, competitions, limit, active_only, show_tags):
    """Display competitions in a formatted table."""
    # Implementation details...
    pass
