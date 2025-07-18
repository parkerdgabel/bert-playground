"""Download commands for competitions and datasets."""

from pathlib import Path
from typing import Optional
import typer
import sys

from ...utils import (
    get_console, print_success, print_error, print_info,
    handle_errors, track_time, validate_path
)

# Add project root to path for imports  
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

@handle_errors
@track_time("Downloading competition data")
def download_competition_command(
    competition: str = typer.Argument(..., help="Competition ID (e.g., titanic)"),
    output_dir: Optional[Path] = typer.Option(None, "--output", "-o", help="Output directory"),
    unzip: bool = typer.Option(True, "--unzip/--no-unzip", help="Unzip downloaded files"),
    specific_file: Optional[str] = typer.Option(None, "--file", "-f", help="Download specific file only"),
    force: bool = typer.Option(False, "--force", help="Force re-download even if files exist"),
):
    """Download competition data from Kaggle.
    
    Downloads all competition files or a specific file to the specified directory.
    By default, files are automatically unzipped.
    
    Examples:
        # Download all competition data
        bert kaggle download titanic
        
        # Download to specific directory
        bert kaggle download titanic --output data/titanic
        
        # Download specific file only
        bert kaggle download titanic --file train.csv
        
        # Keep files zipped
        bert kaggle download titanic --no-unzip
    """
    console = get_console()
    
    console.print(f"\n[bold blue]Downloading Kaggle Competition: {competition}[/bold blue]")
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
        
        # Set default output directory if not provided
        if output_dir is None:
            output_dir = Path("data") / competition
        
        # Check if directory exists and has files
        if output_dir.exists() and any(output_dir.iterdir()) and not force:
            print_error(
                f"Output directory '{output_dir}' already exists and contains files.\n"
                "Use --force to re-download or specify a different directory.",
                title="Directory Exists"
            )
            raise typer.Exit(1)
        
        # Create output directory
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Download competition data
        output_path = kaggle.download_competition_data(
            competition=competition,
            output_dir=output_dir,
            unzip=unzip,
            specific_file=specific_file
        )
        
        print_success(f"Downloaded competition data to: {output_path}")
        
        # List downloaded files
        if output_path.is_dir():
            files = list(output_path.glob("*"))
            console.print(f"\n[cyan]Downloaded {len(files)} files:[/cyan]")
            for file in sorted(files)[:10]:
                size = file.stat().st_size / 1024 / 1024  # MB
                console.print(f"  • {file.name} ({size:.1f} MB)")
            if len(files) > 10:
                console.print(f"  ... and {len(files) - 10} more files")
        
        # Show next steps
        console.print("\n[bold]Next steps:[/bold]")
        console.print(f"1. Explore the data: [cyan]ls {output_path}[/cyan]")
        console.print(f"2. Train a model: [cyan]bert train --train {output_path}/train.csv[/cyan]")
        console.print(f"3. View competition info: [cyan]bert kaggle leaderboard {competition}[/cyan]")
        
    except Exception as e:
        print_error(f"Failed to download competition data: {str(e)}", title="Download Error")
        raise typer.Exit(1)


@handle_errors
@track_time("Downloading dataset")
def download_dataset_command(
    dataset: str = typer.Argument(..., help="Dataset identifier (e.g., username/dataset-name)"),
    output_dir: Optional[Path] = typer.Option(None, "--output", "-o", help="Output directory"),
    unzip: bool = typer.Option(True, "--unzip/--no-unzip", help="Unzip downloaded files"),
    force: bool = typer.Option(False, "--force", help="Force re-download even if files exist"),
):
    """Download a specific dataset from Kaggle.
    
    Downloads datasets using the format: username/dataset-name
    
    Examples:
        # Download a dataset
        bert kaggle download-dataset rdizzl3/nasdaq-earnings-calendar
        
        # Download to specific directory
        bert kaggle download-dataset username/dataset --output data/custom
        
        # Keep files zipped
        bert kaggle download-dataset username/dataset --no-unzip
    """
    console = get_console()
    
    console.print(f"\n[bold blue]Downloading Kaggle Dataset: {dataset}[/bold blue]")
    console.print("=" * 60)
    
    try:
        from utils.kaggle_integration import KaggleDatasetManager
    except ImportError:
        print_error(
            "Failed to import Kaggle integration. Make sure kaggle is installed:\n"
            "pip install kaggle",
            title="Import Error"
        )
        raise typer.Exit(1)
    
    try:
        dataset_manager = KaggleDatasetManager()
        
        # Parse dataset identifier
        if "/" not in dataset:
            print_error(
                "Invalid dataset format. Use: username/dataset-name",
                title="Invalid Format"
            )
            raise typer.Exit(1)
        
        # Set default output directory if not provided
        if output_dir is None:
            dataset_name = dataset.split("/")[1]
            output_dir = Path("data") / "datasets" / dataset_name
        
        # Check if directory exists
        if output_dir.exists() and any(output_dir.iterdir()) and not force:
            print_error(
                f"Output directory '{output_dir}' already exists and contains files.\n"
                "Use --force to re-download or specify a different directory.",
                title="Directory Exists"
            )
            raise typer.Exit(1)
        
        # Create output directory
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Download dataset
        output_path = dataset_manager.download_dataset(
            dataset_ref=dataset,
            output_dir=output_dir,
            unzip=unzip
        )
        
        print_success(f"Downloaded dataset to: {output_path}")
        
        # List downloaded files
        if output_path.is_dir():
            files = list(output_path.glob("*"))
            console.print(f"\n[cyan]Downloaded {len(files)} files:[/cyan]")
            for file in sorted(files)[:10]:
                size = file.stat().st_size / 1024 / 1024  # MB
                console.print(f"  • {file.name} ({size:.1f} MB)")
            if len(files) > 10:
                console.print(f"  ... and {len(files) - 10} more files")
        
        # Show next steps
        console.print("\n[bold]Next steps:[/bold]")
        console.print(f"1. Explore the data: [cyan]ls {output_path}[/cyan]")
        console.print(f"2. Create a data loader for the dataset")
        console.print(f"3. Train a model with the data")
        
    except Exception as e:
        print_error(f"Failed to download dataset: {str(e)}", title="Download Error")
        raise typer.Exit(1)