"""MLflow runs management commands."""

import typer
from pathlib import Path
import sys
from typing import Optional
from datetime import datetime

from ...utils import (
    get_console, print_success, print_error, print_warning, print_info,
    handle_errors
)
from ...utils.console import create_table

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

@handle_errors
def list_runs_command(
    experiment_id: str = typer.Argument(..., help="Experiment ID or name to list runs for"),
    status: Optional[str] = typer.Option(None, "--status", "-s", help="Filter by status: RUNNING, FINISHED, FAILED"),
    metric: Optional[str] = typer.Option(None, "--metric", "-m", help="Metric to display and sort by"),
    sort_order: str = typer.Option("desc", "--sort", help="Sort order: asc or desc"),
    limit: int = typer.Option(20, "--limit", "-l", help="Maximum number of runs to show"),
    show_params: bool = typer.Option(False, "--params", "-p", help="Show run parameters"),
    show_metrics: bool = typer.Option(False, "--metrics", help="Show all metrics"),
    output_format: str = typer.Option("table", "--format", "-f", help="Output format: table, json, csv"),
):
    """List runs for an experiment.
    
    Displays all runs within a specified MLflow experiment with their
    metrics, parameters, and status information.
    
    Examples:
        # List all runs for experiment "bert-titanic"
        bert mlflow runs bert-titanic
        
        # List failed runs sorted by accuracy
        bert mlflow runs bert-titanic --status FAILED --metric accuracy
        
        # Show top 10 runs with parameters
        bert mlflow runs bert-titanic --limit 10 --params
        
        # Export runs as CSV
        bert mlflow runs bert-titanic --format csv > runs.csv
    """
    console = get_console()
    
    console.print(f"\n[bold blue]MLflow Runs for Experiment: {experiment_id}[/bold blue]")
    console.print("=" * 60)
    
    # Display filter parameters
    if status:
        print_info(f"Status filter: {status}")
    
    if metric:
        print_info(f"Sort metric: {metric} ({sort_order})")
    
    print_info(f"Limit: {limit}")
    print_info(f"Output format: {output_format}")
    
    # TODO: Implement actual run listing logic
    console.print("\n[yellow]Run listing functionality not yet implemented.[/yellow]")
    
    # Mock runs table
    if output_format == "table":
        # Create table with appropriate columns
        columns = ["Run ID", "Status", "Start Time", "Duration"]
        
        if metric:
            columns.append(f"{metric}")
        else:
            columns.append("Accuracy")
        
        if show_params:
            columns.extend(["Learning Rate", "Batch Size"])
        
        runs_table = create_table(f"Runs for {experiment_id}", columns)
        
        # Add mock data
        runs_table.add_row(
            "a1b2c3d4e5f6",
            "[green]FINISHED[/green]",
            "2024-01-16 10:30:00",
            "2h 15m",
            "0.9234",
            "2e-5" if show_params else None,
            "32" if show_params else None
        )
        
        runs_table.add_row(
            "f6e5d4c3b2a1",
            "[green]FINISHED[/green]",
            "2024-01-16 08:15:00",
            "2h 10m",
            "0.9156",
            "1e-5" if show_params else None,
            "64" if show_params else None
        )
        
        runs_table.add_row(
            "z9y8x7w6v5u4",
            "[yellow]RUNNING[/yellow]",
            "2024-01-16 14:00:00",
            "45m",
            "-",
            "3e-5" if show_params else None,
            "16" if show_params else None
        )
        
        console.print(runs_table)
        
        print_success("Found 3 runs matching criteria.")
    
    elif output_format == "json":
        # Mock JSON output
        console.print('{"runs": [{"id": "a1b2c3d4e5f6", "status": "FINISHED", "accuracy": 0.9234}]}')
    
    elif output_format == "csv":
        # Mock CSV output
        console.print("run_id,status,start_time,duration,accuracy")
        console.print("a1b2c3d4e5f6,FINISHED,2024-01-16 10:30:00,2h 15m,0.9234")