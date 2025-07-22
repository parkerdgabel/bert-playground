"""Submit predictions to competition command."""

from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

from ...config import get_config
from ...utils import handle_errors


console = Console()


@handle_errors
def submit_command(
    competition: str = typer.Argument(
        ...,
        help="Competition name (e.g., titanic)",
    ),
    submission_file: Path = typer.Argument(
        ...,
        help="Path to submission CSV file",
        exists=True,
        dir_okay=False,
    ),
    message: Optional[str] = typer.Option(
        None,
        "--message",
        "-m",
        help="Submission message",
    ),
    quiet: bool = typer.Option(
        False,
        "--quiet",
        "-q",
        help="Don't wait for submission score",
    ),
):
    """Submit predictions to a Kaggle competition.
    
    This command submits a CSV file containing predictions to a Kaggle competition.
    The submission file must match the format specified by the competition.
    
    Examples:
        # Submit predictions with default message
        k-bert competition submit titanic predictions.csv
        
        # Submit with custom message
        k-bert competition submit titanic predictions.csv -m "ModernBERT v2"
        
        # Submit without waiting for score
        k-bert competition submit titanic predictions.csv --quiet
    """
    # Get configuration
    config = get_config()
    
    # Check Kaggle credentials
    if not config.kaggle.username or not config.kaggle.key:
        console.print(
            "[red]Kaggle credentials not configured.[/red]\n\n"
            "Please set your credentials:\n"
            "  [cyan]k-bert config set kaggle.username YOUR_USERNAME[/cyan]\n"
            "  [cyan]k-bert config set kaggle.key YOUR_API_KEY[/cyan]"
        )
        raise typer.Exit(1)
    
    # Check submission file
    if not submission_file.exists():
        console.print(f"[red]Submission file not found: {submission_file}[/red]")
        raise typer.Exit(1)
    
    if submission_file.suffix.lower() != ".csv":
        console.print("[red]Submission file must be a CSV file.[/red]")
        raise typer.Exit(1)
    
    # Use default message if not provided
    if message is None:
        message = config.kaggle.submission_message
    
    try:
        # Set up Kaggle API
        import os
        os.environ["KAGGLE_USERNAME"] = config.kaggle.username
        os.environ["KAGGLE_KEY"] = config.kaggle.key
        
        import kaggle
        from kaggle.api.kaggle_api_extended import KaggleApi
        
        api = KaggleApi()
        api.authenticate()
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            # Submit to competition
            task = progress.add_task(
                f"Submitting to {competition}...",
                total=None,
            )
            
            try:
                result = api.competition_submit(
                    str(submission_file),
                    message=message,
                    competition=competition,
                    quiet=quiet,
                )
                progress.update(task, completed=True)
                
                console.print(
                    f"\n[green]✓[/green] Successfully submitted to [cyan]{competition}[/cyan]"
                )
                console.print(f"Message: {message}")
                
                if not quiet:
                    # Try to get submission info
                    console.print("\n[dim]Waiting for submission score...[/dim]")
                    
                    import time
                    time.sleep(5)  # Give Kaggle time to process
                    
                    try:
                        submissions = api.competition_submissions(competition)
                        if submissions:
                            latest = submissions[0]
                            console.print(
                                f"\n[bold]Latest submission:[/bold]\n"
                                f"  Date: {latest.date}\n"
                                f"  Score: {latest.publicScore or 'Processing...'}\n"
                                f"  Status: {latest.status}"
                            )
                            
                            if latest.status == "complete" and latest.publicScore:
                                console.print(
                                    f"\n[green]Public score: {latest.publicScore}[/green]"
                                )
                    except Exception as e:
                        console.print(
                            "[yellow]Could not retrieve submission score. "
                            "Check the competition page for results.[/yellow]"
                        )
                
                # Show competition URL
                console.print(
                    f"\n[dim]View all submissions:[/dim]\n"
                    f"https://www.kaggle.com/c/{competition}/submissions"
                )
                
            except Exception as e:
                progress.update(task, completed=True)
                
                if "403" in str(e):
                    console.print(
                        f"[red]Access denied to competition '{competition}'.[/red]\n"
                        "Please make sure:\n"
                        "  1. You have accepted the competition rules\n"
                        "  2. The competition is still accepting submissions\n"
                        "  3. You haven't exceeded the daily submission limit"
                    )
                elif "404" in str(e):
                    console.print(
                        f"[red]Competition '{competition}' not found.[/red]"
                    )
                else:
                    console.print(f"[red]Submission failed: {e}[/red]")
                raise typer.Exit(1)
    
    except ImportError:
        console.print(
            "[red]Kaggle package not installed.[/red]\n"
            "This should have been installed with k-bert."
        )
        raise typer.Exit(1)