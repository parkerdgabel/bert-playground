"""MLflow import command."""

from pathlib import Path
from typing import Optional

import typer
from rich.console import Console

from ...config import get_config
from ...utils import handle_errors


console = Console()


@handle_errors
def import_command(
    path: Path = typer.Argument(..., help="Path to import file or directory"),
    experiment: Optional[str] = typer.Option(
        None,
        "--experiment", 
        "-e",
        help="Target experiment name (creates if not exists)"
    ),
    run_name: Optional[str] = typer.Option(
        None,
        "--run-name",
        "-n", 
        help="Name for imported run"
    ),
    overwrite: bool = typer.Option(
        False,
        "--overwrite",
        help="Overwrite existing run if run ID matches"
    ),
    import_artifacts: bool = typer.Option(
        True,
        "--artifacts/--no-artifacts",
        help="Import artifacts along with run data"
    ),
):
    """Import MLflow run from file or directory.
    
    Import run data that was previously exported or created externally.
    Supports MLflow run directories and exported data files.
    
    Examples:
        # Import run directory
        k-bert mlflow import ./exported_run_dir
        
        # Import to specific experiment  
        k-bert mlflow import ./run_data.zip --experiment my_experiment
        
        # Import without artifacts
        k-bert mlflow import ./run_data --no-artifacts
    """
    # Get configuration
    config = get_config()
    
    # Validate path
    if not path.exists():
        console.print(f"[red]Path not found: {path}[/red]")
        raise typer.Exit(1)
    
    try:
        import mlflow
        from mlflow.tracking import MlflowClient
        
        # Set tracking URI
        if config.mlflow and config.mlflow.tracking_uri:
            mlflow.set_tracking_uri(config.mlflow.tracking_uri)
        
        client = MlflowClient()
        
        console.print(f"[dim]Importing from: {path}[/dim]")
        
        # Handle different import types
        if path.is_file():
            # Handle file imports (ZIP, JSON, etc.)
            if path.suffix.lower() == '.zip':
                _import_from_zip(client, path, experiment, run_name, import_artifacts)
            elif path.suffix.lower() == '.json':
                _import_from_json(client, path, experiment, run_name)
            else:
                console.print(f"[red]Unsupported file format: {path.suffix}[/red]")
                console.print("Supported formats: .zip, .json")
                raise typer.Exit(1)
        
        elif path.is_dir():
            # Handle directory imports (MLflow run directories)
            _import_from_directory(client, path, experiment, run_name, import_artifacts)
        
        else:
            console.print(f"[red]Invalid path: {path}[/red]")
            raise typer.Exit(1)
        
    except ImportError:
        console.print(
            "[red]MLflow not installed.[/red]\n"
            "Install with: [cyan]uv add mlflow[/cyan]"
        )
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"[red]Failed to import run: {e}[/red]")
        raise typer.Exit(1)


def _import_from_zip(client, zip_path, experiment, run_name, import_artifacts):
    """Import from ZIP file."""
    import tempfile
    import zipfile
    
    console.print("Extracting ZIP archive...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(temp_dir)
        
        # Find the run directory
        temp_path = Path(temp_dir)
        run_dirs = list(temp_path.iterdir())
        
        if len(run_dirs) == 1 and run_dirs[0].is_dir():
            run_dir = run_dirs[0]
        else:
            run_dir = temp_path
        
        _import_from_directory(client, run_dir, experiment, run_name, import_artifacts)


def _import_from_json(client, json_path, experiment, run_name):
    """Import from JSON file."""
    import json
    
    console.print("Loading JSON data...")
    
    with open(json_path) as f:
        run_data = json.load(f)
    
    # Handle both single run and multiple runs
    if isinstance(run_data, list):
        console.print(f"Importing {len(run_data)} runs...")
        for i, run in enumerate(run_data):
            name = f"{run_name}_{i}" if run_name else None
            _create_run_from_data(client, run, experiment, name)
    else:
        _create_run_from_data(client, run_data, experiment, run_name)


def _import_from_directory(client, run_dir, experiment, run_name, import_artifacts):
    """Import from MLflow run directory."""
    console.print("Importing from directory...")
    
    # Look for meta.yaml or run metadata
    meta_file = run_dir / "meta.yaml"
    if meta_file.exists():
        import yaml
        with open(meta_file) as f:
            metadata = yaml.safe_load(f)
    else:
        metadata = {}
    
    # Look for metrics, params, tags
    metrics_dir = run_dir / "metrics"
    params_dir = run_dir / "params" 
    tags_dir = run_dir / "tags"
    artifacts_dir = run_dir / "artifacts"
    
    # Create or get experiment
    if experiment:
        try:
            exp = mlflow.get_experiment_by_name(experiment)
            if exp is None:
                exp_id = mlflow.create_experiment(experiment)
                exp = mlflow.get_experiment(exp_id)
        except Exception:
            exp_id = mlflow.create_experiment(experiment)
            exp = mlflow.get_experiment(exp_id)
    else:
        exp = mlflow.get_experiment("0")  # Default experiment
    
    # Create run
    with mlflow.start_run(experiment_id=exp.experiment_id, run_name=run_name):
        run_id = mlflow.active_run().info.run_id
        console.print(f"Created run: {run_id}")
        
        # Import metrics
        if metrics_dir.exists():
            for metric_file in metrics_dir.iterdir():
                if metric_file.is_file():
                    try:
                        with open(metric_file) as f:
                            lines = f.readlines()
                        
                        metric_name = metric_file.name
                        for line in lines:
                            parts = line.strip().split()
                            if len(parts) >= 3:
                                timestamp, value = parts[0], parts[1]
                                mlflow.log_metric(
                                    metric_name, 
                                    float(value), 
                                    step=int(timestamp) if timestamp.isdigit() else None
                                )
                    except Exception as e:
                        console.print(f"[yellow]Warning: Could not import metric {metric_file.name}: {e}[/yellow]")
        
        # Import parameters
        if params_dir.exists():
            for param_file in params_dir.iterdir():
                if param_file.is_file():
                    try:
                        with open(param_file) as f:
                            value = f.read().strip()
                        mlflow.log_param(param_file.name, value)
                    except Exception as e:
                        console.print(f"[yellow]Warning: Could not import param {param_file.name}: {e}[/yellow]")
        
        # Import tags
        if tags_dir.exists():
            for tag_file in tags_dir.iterdir():
                if tag_file.is_file():
                    try:
                        with open(tag_file) as f:
                            value = f.read().strip()
                        mlflow.set_tag(tag_file.name, value)
                    except Exception as e:
                        console.print(f"[yellow]Warning: Could not import tag {tag_file.name}: {e}[/yellow]")
        
        # Import artifacts
        if import_artifacts and artifacts_dir.exists():
            try:
                mlflow.log_artifacts(str(artifacts_dir))
                console.print("✓ Imported artifacts")
            except Exception as e:
                console.print(f"[yellow]Warning: Could not import artifacts: {e}[/yellow]")
    
    console.print(f"[bold green]✓ Run imported successfully: {run_id}[/bold green]")


def _create_run_from_data(client, run_data, experiment, run_name):
    """Create run from structured data."""
    # Create or get experiment
    if experiment:
        try:
            exp = mlflow.get_experiment_by_name(experiment)
            if exp is None:
                exp_id = mlflow.create_experiment(experiment)
                exp = mlflow.get_experiment(exp_id)
        except Exception:
            exp_id = mlflow.create_experiment(experiment)
            exp = mlflow.get_experiment(exp_id)
    else:
        exp = mlflow.get_experiment("0")  # Default experiment
    
    # Create run
    with mlflow.start_run(experiment_id=exp.experiment_id, run_name=run_name):
        run_id = mlflow.active_run().info.run_id
        
        # Log metrics
        for key, value in run_data.items():
            if key.startswith("metrics."):
                metric_name = key.replace("metrics.", "")
                try:
                    mlflow.log_metric(metric_name, float(value))
                except (ValueError, TypeError):
                    pass
            elif key.startswith("params."):
                param_name = key.replace("params.", "")
                mlflow.log_param(param_name, str(value))
            elif key.startswith("tags."):
                tag_name = key.replace("tags.", "")
                mlflow.set_tag(tag_name, str(value))
        
        console.print(f"✓ Imported run: {run_id}")
    
    return run_id