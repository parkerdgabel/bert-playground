"""Model listing command - list available models and checkpoints."""

from pathlib import Path
from typing import Optional, List
import typer
from loguru import logger
import json
from datetime import datetime
import os

from ...utils import (
    get_console, print_success, print_error, print_info, print_warning,
    handle_errors, requires_project,
    create_table, format_bytes, format_timestamp
)

# Add project root to path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

console = get_console()

@handle_errors
@requires_project()
def list_models_command(
    path: Path = typer.Option(Path("output"), "--path", "-p",
                            help="Directory to search for models"),
    format: str = typer.Option("table", "--format", "-f",
                             help="Output format: table, json, simple"),
    sort_by: str = typer.Option("date", "--sort", "-s",
                              help="Sort by: date, size, name, accuracy"),
    reverse: bool = typer.Option(False, "--reverse", "-r",
                               help="Reverse sort order"),
    filter_name: Optional[str] = typer.Option(None, "--name", "-n",
                                            help="Filter by model name pattern"),
    filter_type: Optional[str] = typer.Option(None, "--type", "-t",
                                            help="Filter by model type"),
    show_metrics: bool = typer.Option(True, "--metrics/--no-metrics",
                                    help="Show model metrics"),
    show_config: bool = typer.Option(False, "--config", "-c",
                                   help="Show model configuration"),
    limit: int = typer.Option(0, "--limit", "-l",
                            help="Limit number of results (0 for all)"),
    recursive: bool = typer.Option(True, "--recursive/--no-recursive",
                                 help="Search recursively"),
    include_mlflow: bool = typer.Option(False, "--mlflow", "-m",
                                      help="Include MLflow tracked models"),
    verbose: bool = typer.Option(False, "--verbose", "-v",
                               help="Show detailed information")
):
    """List available models and checkpoints.
    
    This command searches for trained models in the specified directory
    and displays information about each model.
    
    Examples:
        # List all models in output directory
        bert model list
        
        # List models sorted by accuracy
        bert model list --sort accuracy --reverse
        
        # Filter models by name
        bert model list --name "titanic*"
        
        # Show detailed configuration
        bert model list --config --verbose
        
        # Export as JSON
        bert model list --format json > models.json
        
        # Include MLflow models
        bert model list --mlflow
    """
    if not path.exists():
        print_error(f"Path does not exist: {path}")
        raise typer.Exit(1)
    
    # Find all models
    models = _find_models(path, recursive, filter_name, filter_type, verbose)
    
    if include_mlflow:
        mlflow_models = _find_mlflow_models(verbose)
        models.extend(mlflow_models)
    
    if not models:
        print_warning("No models found")
        return
    
    # Sort models
    models = _sort_models(models, sort_by, reverse)
    
    # Apply limit
    if limit > 0:
        models = models[:limit]
    
    # Display results
    if format == "table":
        _display_table(models, show_metrics, show_config)
    elif format == "json":
        _display_json(models)
    elif format == "simple":
        _display_simple(models)
    else:
        print_error(f"Unknown format: {format}")
        raise typer.Exit(1)
    
    # Summary
    console.print(f"\n[dim]Found {len(models)} model(s)[/dim]")


def _find_models(path: Path, recursive: bool, name_filter: Optional[str], 
                 type_filter: Optional[str], verbose: bool) -> List[dict]:
    """Find all models in the given path."""
    models = []
    
    # Patterns to identify model directories
    model_indicators = [
        "model.safetensors",
        "model.npz",
        "config.json",
        "best_model.safetensors",
        "checkpoint.mlx"
    ]
    
    # Search for models
    search_paths = [path]
    if recursive:
        search_paths.extend(path.rglob("*") if path.is_dir() else [])
    
    for search_path in search_paths:
        if not search_path.is_dir():
            continue
            
        # Check if this is a model directory
        is_model = any((search_path / indicator).exists() for indicator in model_indicators)
        
        if not is_model:
            continue
        
        # Apply filters
        if name_filter and not _matches_pattern(search_path.name, name_filter):
            continue
        
        # Extract model info
        model_info = _extract_model_info(search_path, verbose)
        
        if type_filter and model_info.get('type') != type_filter:
            continue
        
        models.append(model_info)
    
    return models


def _find_mlflow_models(verbose: bool) -> List[dict]:
    """Find models tracked by MLflow."""
    models = []
    
    try:
        import mlflow
        from mlflow.tracking import MlflowClient
        
        client = MlflowClient()
        
        # Get all experiments
        experiments = client.search_experiments()
        
        for exp in experiments:
            # Get runs with logged models
            runs = client.search_runs(
                experiment_ids=[exp.experiment_id],
                filter_string="attributes.status = 'FINISHED'",
                order_by=["attributes.end_time DESC"]
            )
            
            for run in runs:
                # Check if run has a model
                artifacts = client.list_artifacts(run.info.run_id)
                has_model = any(art.path == "model" for art in artifacts)
                
                if has_model:
                    model_info = {
                        'name': run.data.tags.get('mlflow.runName', run.info.run_id[:8]),
                        'path': f"mlflow:///{exp.name}/{run.info.run_id}/model",
                        'type': 'mlflow',
                        'date': datetime.fromtimestamp(run.info.end_time / 1000),
                        'size': 0,  # MLflow doesn't provide size easily
                        'metrics': run.data.metrics,
                        'params': run.data.params,
                        'tags': run.data.tags,
                        'experiment': exp.name,
                        'run_id': run.info.run_id
                    }
                    models.append(model_info)
        
    except Exception as e:
        if verbose:
            logger.warning(f"Failed to load MLflow models: {e}")
    
    return models


def _extract_model_info(model_path: Path, verbose: bool) -> dict:
    """Extract information about a model."""
    info = {
        'name': model_path.name,
        'path': str(model_path),
        'type': 'local',
        'date': datetime.fromtimestamp(model_path.stat().st_mtime),
        'size': _get_directory_size(model_path),
        'metrics': {},
        'config': {},
        'params': {}
    }
    
    # Load config
    config_path = model_path / "config.json"
    if config_path.exists():
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
                info['config'] = config
                info['type'] = config.get('model_type', 'unknown')
                info['params'] = {
                    'model_name': config.get('model_name'),
                    'hidden_size': config.get('hidden_size'),
                    'num_layers': config.get('num_hidden_layers'),
                    'num_heads': config.get('num_attention_heads'),
                    'vocab_size': config.get('vocab_size'),
                }
        except Exception as e:
            if verbose:
                logger.warning(f"Failed to load config from {config_path}: {e}")
    
    # Load metrics
    metrics_path = model_path / "metrics.json"
    if metrics_path.exists():
        try:
            with open(metrics_path, 'r') as f:
                info['metrics'] = json.load(f)
        except Exception as e:
            if verbose:
                logger.warning(f"Failed to load metrics from {metrics_path}: {e}")
    
    # Check for training info
    training_info_path = model_path / "training_info.json"
    if training_info_path.exists():
        try:
            with open(training_info_path, 'r') as f:
                training_info = json.load(f)
                info['metrics'].update(training_info.get('final_metrics', {}))
                info['params'].update(training_info.get('training_params', {}))
        except Exception as e:
            if verbose:
                logger.warning(f"Failed to load training info: {e}")
    
    return info


def _get_directory_size(path: Path) -> int:
    """Calculate total size of directory."""
    total = 0
    for file in path.rglob('*'):
        if file.is_file():
            total += file.stat().st_size
    return total


def _matches_pattern(name: str, pattern: str) -> bool:
    """Check if name matches glob pattern."""
    from fnmatch import fnmatch
    return fnmatch(name.lower(), pattern.lower())


def _sort_models(models: List[dict], sort_by: str, reverse: bool) -> List[dict]:
    """Sort models by specified criteria."""
    if sort_by == "date":
        models.sort(key=lambda m: m['date'], reverse=not reverse)
    elif sort_by == "size":
        models.sort(key=lambda m: m['size'], reverse=not reverse)
    elif sort_by == "name":
        models.sort(key=lambda m: m['name'].lower(), reverse=reverse)
    elif sort_by == "accuracy":
        models.sort(
            key=lambda m: m['metrics'].get('val_accuracy', 0) or m['metrics'].get('accuracy', 0),
            reverse=not reverse
        )
    
    return models


def _display_table(models: List[dict], show_metrics: bool, show_config: bool):
    """Display models in table format."""
    table = create_table("Available Models")
    
    # Basic columns
    table.add_column("Name", style="cyan", overflow="fold")
    table.add_column("Type", style="yellow")
    table.add_column("Date", style="magenta")
    table.add_column("Size", style="blue")
    
    # Metrics columns
    if show_metrics:
        table.add_column("Accuracy", style="green")
        table.add_column("Loss", style="red")
    
    # Config columns
    if show_config:
        table.add_column("Model", style="dim")
        table.add_column("Params", style="dim")
    
    table.add_column("Path", style="dim", overflow="fold")
    
    for model in models:
        row = [
            model['name'],
            model['type'],
            format_timestamp(model['date']),
            format_bytes(model['size']) if model['size'] > 0 else "N/A"
        ]
        
        if show_metrics:
            accuracy = model['metrics'].get('val_accuracy') or model['metrics'].get('accuracy')
            loss = model['metrics'].get('val_loss') or model['metrics'].get('loss')
            row.extend([
                f"{accuracy:.3f}" if accuracy else "N/A",
                f"{loss:.3f}" if loss else "N/A"
            ])
        
        if show_config:
            model_name = model['params'].get('model_name', 'N/A')
            if model_name and len(model_name) > 30:
                model_name = model_name[:27] + "..."
            
            param_count = model['params'].get('total_params')
            if param_count:
                param_str = f"{param_count/1e6:.1f}M"
            else:
                param_str = "N/A"
            
            row.extend([model_name, param_str])
        
        # Shorten path for display
        path = model['path']
        if len(path) > 50:
            path = "..." + path[-47:]
        row.append(path)
        
        table.add_row(*row)
    
    console.print(table)


def _display_json(models: List[dict]):
    """Display models in JSON format."""
    # Convert datetime objects to strings
    for model in models:
        model['date'] = model['date'].isoformat()
    
    console.print_json(data=models)


def _display_simple(models: List[dict]):
    """Display models in simple format."""
    for model in models:
        console.print(f"[cyan]{model['name']}[/cyan]")
        console.print(f"  Path: {model['path']}")
        console.print(f"  Type: {model['type']}")
        console.print(f"  Date: {format_timestamp(model['date'])}")
        console.print(f"  Size: {format_bytes(model['size'])}")
        
        if model['metrics']:
            accuracy = model['metrics'].get('val_accuracy') or model['metrics'].get('accuracy')
            if accuracy:
                console.print(f"  Accuracy: {accuracy:.3f}")
        
        console.print()