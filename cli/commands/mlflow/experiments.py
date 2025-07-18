"""MLflow experiments management commands."""

import typer
from pathlib import Path
import sys
from typing import Optional, List
from datetime import datetime

from ...utils import (
    get_console, print_success, print_error, print_warning, print_info,
    handle_errors
)
from ...utils.console import create_table

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

@handle_errors
def list_experiments_command(
    tracking_uri: Optional[str] = typer.Option(None, "--tracking-uri", "-u", help="MLflow tracking URI"),
    view: str = typer.Option("active", "--view", "-v", help="View type: active, deleted, or all"),
    sort_by: str = typer.Option("creation_time", "--sort", "-s", help="Sort by: name, creation_time, or last_update"),
    limit: int = typer.Option(20, "--limit", "-l", help="Maximum number of experiments to show"),
    filter_name: Optional[str] = typer.Option(None, "--filter", "-f", help="Filter experiments by name pattern"),
):
    """List MLflow experiments.
    
    Displays all experiments in the MLflow tracking store with detailed
    information including run counts, metrics, and lifecycle stage.
    
    Examples:
        # List all active experiments
        bert mlflow experiments
        
        # List experiments with custom sorting and filtering
        bert mlflow experiments --sort name --filter "bert-*"
        
        # Show deleted experiments
        bert mlflow experiments --view deleted
        
        # List experiments from specific tracking server
        bert mlflow experiments --tracking-uri http://localhost:5000
    """
    console = get_console()
    
    console.print("\n[bold blue]MLflow Experiments[/bold blue]")
    console.print("=" * 60)
    
    # Display query parameters
    print_info(f"View: {view}")
    print_info(f"Sort by: {sort_by}")
    print_info(f"Limit: {limit}")
    
    if tracking_uri:
        print_info(f"Tracking URI: {tracking_uri}")
    
    if filter_name:
        print_info(f"Filter: {filter_name}")
    
    # TODO: Implement actual experiment listing logic
    console.print("\n[yellow]Experiment listing functionality not yet implemented.[/yellow]")
    
    # Mock experiment table
    experiments_table = create_table(
        "Experiments",
        ["ID", "Name", "Lifecycle", "Runs", "Created", "Last Updated"]
    )
    
    # Add mock data
    experiments_table.add_row(
        "1",
        "bert-titanic-experiment",
        "[green]active[/green]",
        "15",
        "2024-01-15 10:30:00",
        "2024-01-16 14:22:00"
    )
    
    experiments_table.add_row(
        "2",
        "bert-spaceship-experiment",
        "[green]active[/green]",
        "8",
        "2024-01-14 09:15:00",
        "2024-01-15 16:45:00"
    )
    
    console.print(experiments_table)
    
    print_success("Found 2 experiments matching criteria.")


@handle_errors
def clean_command(
    experiment_ids: Optional[List[str]] = typer.Option(None, "--experiment", "-e", help="Specific experiment IDs to clean"),
    older_than: Optional[int] = typer.Option(None, "--older-than", "-o", help="Clean runs older than N days"),
    failed_only: bool = typer.Option(False, "--failed-only", help="Only clean failed runs"),
    dry_run: bool = typer.Option(True, "--dry-run/--no-dry-run", help="Show what would be cleaned without doing it"),
    keep_best: int = typer.Option(5, "--keep-best", "-k", help="Number of best runs to keep per experiment"),
    force: bool = typer.Option(False, "--force", "-f", help="Skip confirmation prompts"),
):
    """Clean up experiments and runs.
    
    Removes old, failed, or unnecessary MLflow runs and experiments to
    free up storage space and improve tracking server performance.
    
    By default runs in dry-run mode to show what would be cleaned.
    
    Examples:
        # Show what would be cleaned (dry run)
        bert mlflow clean
        
        # Clean failed runs older than 30 days
        bert mlflow clean --failed-only --older-than 30 --no-dry-run
        
        # Clean specific experiments, keeping best 10 runs
        bert mlflow clean --experiment exp1 --experiment exp2 --keep-best 10 --no-dry-run
        
        # Force clean without confirmation
        bert mlflow clean --force --no-dry-run
    """
    console = get_console()
    
    console.print("\n[bold blue]MLflow Cleanup[/bold blue]")
    console.print("=" * 60)
    
    # Display cleanup configuration
    if dry_run:
        print_warning("Running in DRY RUN mode - no changes will be made")
    else:
        print_warning("Running in LIVE mode - changes WILL be made")
    
    print_info(f"Keep best runs: {keep_best}")
    print_info(f"Failed runs only: {failed_only}")
    
    if older_than:
        print_info(f"Older than: {older_than} days")
    
    if experiment_ids:
        print_info(f"Target experiments: {', '.join(experiment_ids)}")
    
    # Confirm if not dry run and not forced
    if not dry_run and not force:
        confirm = typer.confirm("Are you sure you want to proceed with cleanup?")
        if not confirm:
            print_warning("Cleanup cancelled.")
            raise typer.Exit(0)
    
    # TODO: Implement actual cleanup logic
    console.print("\n[yellow]Cleanup functionality not yet implemented.[/yellow]")
    
    # Mock cleanup summary
    console.print("\n[bold]Cleanup Summary:[/bold]")
    console.print("  • Experiments analyzed: 3")
    console.print("  • Total runs found: 45")
    console.print("  • Runs to be cleaned: 12")
    console.print("  • Failed runs: 5")
    console.print("  • Old runs (>30 days): 7")
    console.print("  • Storage to be freed: ~1.2 GB")
    
    if dry_run:
        print_info("Run with --no-dry-run to perform actual cleanup.")
    else:
        print_success("Cleanup would be completed successfully.")