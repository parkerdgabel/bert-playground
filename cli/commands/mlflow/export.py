"""MLflow export command."""

from pathlib import Path
from typing import Optional

import typer
from rich.console import Console

from ...config import get_config
from ...utils import handle_errors


console = Console()


@handle_errors
def export_command(
    output: Path = typer.Option(
        ..., 
        "--output", 
        "-o", 
        help="Output file path"
    ),
    experiment: Optional[str] = typer.Option(
        None,
        "--experiment",
        "-e",
        help="Experiment name or ID to export"
    ),
    run_ids: Optional[str] = typer.Option(
        None,
        "--run-ids",
        "-r",
        help="Comma-separated list of run IDs"
    ),
    format: str = typer.Option(
        "csv",
        "--format",
        "-f",
        help="Export format: csv, json, parquet"
    ),
    include_artifacts: bool = typer.Option(
        False,
        "--include-artifacts",
        help="Include artifact paths in export"
    ),
    metrics_only: bool = typer.Option(
        False,
        "--metrics-only",
        help="Export only metrics data"
    ),
    params_only: bool = typer.Option(
        False,
        "--params-only", 
        help="Export only parameters data"
    ),
):
    """Export MLflow runs to file.
    
    Export experiment data including runs, metrics, parameters, and metadata
    to various formats for analysis or backup.
    
    Examples:
        # Export all runs from an experiment
        k-bert mlflow export --experiment my_exp --output runs.csv
        
        # Export specific runs
        k-bert mlflow export --run-ids run1,run2 --output selected_runs.json
        
        # Export only metrics
        k-bert mlflow export --experiment my_exp --metrics-only --output metrics.csv
    """
    # Get configuration
    config = get_config()
    
    try:
        import mlflow
        import pandas as pd
        
        # Set tracking URI
        if config.mlflow and config.mlflow.tracking_uri:
            mlflow.set_tracking_uri(config.mlflow.tracking_uri)
        
        console.print("[dim]Exporting MLflow data...[/dim]")
        
        # Determine what to export
        if run_ids:
            # Export specific runs
            run_list = [rid.strip() for rid in run_ids.split(",")]
            console.print(f"Exporting {len(run_list)} specific runs...")
            
            runs_data = []
            for run_id in run_list:
                try:
                    run = mlflow.get_run(run_id)
                    runs_data.append(_run_to_dict(run, include_artifacts))
                except Exception as e:
                    console.print(f"[yellow]Warning: Could not get run {run_id}: {e}[/yellow]")
            
            df = pd.DataFrame(runs_data)
            
        elif experiment:
            # Export runs from experiment
            console.print(f"Exporting runs from experiment: {experiment}")
            
            # Try to get experiment by name first, then by ID
            try:
                exp = mlflow.get_experiment_by_name(experiment)
                if exp is None:
                    exp = mlflow.get_experiment(experiment)
            except Exception:
                exp = mlflow.get_experiment(experiment)
            
            if exp is None:
                console.print(f"[red]Experiment '{experiment}' not found.[/red]")
                raise typer.Exit(1)
            
            # Search runs in experiment
            df = mlflow.search_runs(
                experiment_ids=[exp.experiment_id],
                output_format="pandas"
            )
            
            if include_artifacts:
                # Add artifact information
                artifact_info = []
                for _, row in df.iterrows():
                    run_id = row["run_id"]
                    try:
                        client = mlflow.tracking.MlflowClient()
                        artifacts = client.list_artifacts(run_id)
                        artifact_paths = [art.path for art in artifacts]
                        artifact_info.append(";".join(artifact_paths) if artifact_paths else "")
                    except Exception:
                        artifact_info.append("")
                
                df["artifacts"] = artifact_info
            
        else:
            console.print("[red]Must specify either --experiment or --run-ids[/red]")
            raise typer.Exit(1)
        
        if df.empty:
            console.print("[yellow]No runs found to export.[/yellow]")
            return
        
        # Filter columns based on options
        if metrics_only:
            metric_cols = [col for col in df.columns if col.startswith("metrics.")]
            base_cols = ["run_id", "experiment_id", "status", "start_time", "end_time"]
            df = df[[col for col in base_cols + metric_cols if col in df.columns]]
        
        elif params_only:
            param_cols = [col for col in df.columns if col.startswith("params.")]
            base_cols = ["run_id", "experiment_id", "status", "start_time", "end_time"]
            df = df[[col for col in base_cols + param_cols if col in df.columns]]
        
        # Create output directory if needed
        output.parent.mkdir(parents=True, exist_ok=True)
        
        # Export based on format
        if format.lower() == "csv":
            df.to_csv(output, index=False)
        elif format.lower() == "json":
            df.to_json(output, orient="records", indent=2, date_format="iso")
        elif format.lower() == "parquet":
            df.to_parquet(output, index=False)
        else:
            console.print(f"[red]Unsupported format: {format}[/red]")
            console.print("Supported formats: csv, json, parquet")
            raise typer.Exit(1)
        
        console.print(f"[bold green]✓ Exported {len(df)} runs to {output}[/bold green]")
        
        # Show summary
        console.print("\n[bold]Export Summary:[/bold]")
        console.print(f"  Runs: {len(df)}")
        console.print(f"  Format: {format.upper()}")
        console.print(f"  Size: {output.stat().st_size / 1024:.1f} KB")
        
        if not (metrics_only or params_only):
            metric_cols = len([col for col in df.columns if col.startswith("metrics.")])
            param_cols = len([col for col in df.columns if col.startswith("params.")])
            console.print(f"  Metrics: {metric_cols}")
            console.print(f"  Parameters: {param_cols}")
        
    except ImportError:
        console.print(
            "[red]MLflow not installed.[/red]\n"
            "Install with: [cyan]uv add mlflow pandas[/cyan]"
        )
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"[red]Failed to export runs: {e}[/red]")
        raise typer.Exit(1)


def _run_to_dict(run, include_artifacts=False):
    """Convert MLflow run to dictionary."""
    data = {
        "run_id": run.info.run_id,
        "experiment_id": run.info.experiment_id,
        "status": run.info.status,
        "start_time": run.info.start_time,
        "end_time": run.info.end_time,
        "lifecycle_stage": run.info.lifecycle_stage,
        "user_id": run.info.user_id,
    }
    
    # Add metrics
    for key, value in run.data.metrics.items():
        data[f"metrics.{key}"] = value
    
    # Add parameters
    for key, value in run.data.params.items():
        data[f"params.{key}"] = value
    
    # Add tags
    for key, value in run.data.tags.items():
        data[f"tags.{key}"] = value
    
    if include_artifacts:
        try:
            client = mlflow.tracking.MlflowClient()
            artifacts = client.list_artifacts(run.info.run_id)
            artifact_paths = [art.path for art in artifacts]
            data["artifacts"] = ";".join(artifact_paths) if artifact_paths else ""
        except Exception:
            data["artifacts"] = ""
    
    return data