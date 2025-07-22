"""Download competition data command."""

from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

from ...config import get_config
from ...utils import handle_errors


console = Console()


@handle_errors
def download_command(
    competition: str = typer.Argument(
        ...,
        help="Competition name (e.g., titanic)",
    ),
    path: Optional[Path] = typer.Option(
        None,
        "--path",
        "-p",
        help="Download path (defaults to competitions/{competition})",
    ),
    force: bool = typer.Option(
        False,
        "--force",
        "-f",
        help="Force re-download even if files exist",
    ),
    unzip: bool = typer.Option(
        True,
        "--unzip/--no-unzip",
        help="Unzip downloaded files",
    ),
):
    """Download competition data from Kaggle.
    
    This command downloads all data files for a specified Kaggle competition.
    Requires Kaggle API credentials to be configured.
    
    Examples:
        # Download Titanic competition data
        k-bert competition download titanic
        
        # Download to specific directory
        k-bert competition download house-prices --path ./data/house-prices
        
        # Force re-download
        k-bert competition download titanic --force
    """
    # Get configuration
    config = get_config()
    
    # Check Kaggle credentials
    if not config.kaggle.username or not config.kaggle.key:
        console.print(
            "[red]Kaggle credentials not configured.[/red]\n\n"
            "Please set your credentials:\n"
            "  [cyan]k-bert config set kaggle.username YOUR_USERNAME[/cyan]\n"
            "  [cyan]k-bert config set kaggle.key YOUR_API_KEY[/cyan]\n\n"
            "You can find your API key at: https://www.kaggle.com/account"
        )
        raise typer.Exit(1)
    
    # Determine download path
    if path is None:
        path = config.kaggle.competitions_dir / competition
    
    # Create directory
    path.mkdir(parents=True, exist_ok=True)
    
    # Check if files already exist
    existing_files = list(path.glob("*.csv")) + list(path.glob("*.zip"))
    if existing_files and not force:
        console.print(
            f"[yellow]Competition data already exists in {path}[/yellow]\n"
            f"Found {len(existing_files)} file(s). Use --force to re-download."
        )
        raise typer.Exit(0)
    
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
            # Download competition files
            task = progress.add_task(
                f"Downloading {competition} competition data...",
                total=None,
            )
            
            try:
                api.competition_download_files(
                    competition,
                    path=str(path),
                    unzip=unzip,
                    force=force,
                    quiet=False,
                )
                progress.update(task, completed=True)
            except Exception as e:
                if "403" in str(e):
                    console.print(
                        f"[red]Access denied to competition '{competition}'.[/red]\n"
                        "Please make sure:\n"
                        "  1. You have accepted the competition rules on Kaggle\n"
                        "  2. The competition name is correct\n"
                        "  3. Your API credentials are valid"
                    )
                elif "404" in str(e):
                    console.print(
                        f"[red]Competition '{competition}' not found.[/red]\n"
                        "Use 'k-bert competition list' to see available competitions."
                    )
                else:
                    console.print(f"[red]Download failed: {e}[/red]")
                raise typer.Exit(1)
        
        # List downloaded files
        if unzip:
            downloaded_files = list(path.glob("*.csv")) + list(path.glob("*.json"))
        else:
            downloaded_files = list(path.glob("*.zip"))
        
        console.print(f"\n[green]✓[/green] Downloaded {len(downloaded_files)} file(s) to {path}")
        
        if downloaded_files:
            console.print("\nDownloaded files:")
            for file in sorted(downloaded_files):
                size_mb = file.stat().st_size / (1024 * 1024)
                console.print(f"  [cyan]{file.name}[/cyan] ({size_mb:.1f} MB)")
        
        # Check for common files and provide guidance
        train_files = [f for f in downloaded_files if "train" in f.name.lower()]
        test_files = [f for f in downloaded_files if "test" in f.name.lower()]
        
        if train_files and test_files:
            train_file = train_files[0]
            test_file = test_files[0]
            
            console.print(
                f"\n[bold]Next steps:[/bold]\n"
                f"1. Explore the data:\n"
                f"   [cyan]k-bert explore {train_file}[/cyan]\n\n"
                f"2. Initialize a project:\n"
                f"   [cyan]k-bert competition init {competition}[/cyan]\n\n"
                f"3. Start training:\n"
                f"   [cyan]k-bert train --train {train_file} --val {train_file}[/cyan]"
            )
    
    except ImportError:
        console.print(
            "[red]Kaggle package not installed.[/red]\n"
            "This should have been installed with k-bert. "
            "Try reinstalling: pip install k-bert[kaggle]"
        )
        raise typer.Exit(1)