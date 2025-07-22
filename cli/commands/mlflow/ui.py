"""MLflow UI command."""

import subprocess
import sys
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console

from ...config import get_config
from ...utils import handle_errors


console = Console()


@handle_errors
def ui_command(
    port: int = typer.Option(5000, "--port", "-p", help="Port to run MLflow UI on"),
    host: str = typer.Option("127.0.0.1", "--host", "-h", help="Host to bind to"),
    backend_store_uri: Optional[str] = typer.Option(
        None, 
        "--backend-store-uri", 
        help="Backend store URI (overrides config)"
    ),
    default_artifact_root: Optional[str] = typer.Option(
        None,
        "--default-artifact-root", 
        help="Default artifact root directory"
    ),
    dev: bool = typer.Option(
        False, "--dev", help="Run in development mode"
    ),
):
    """Start MLflow UI for experiment tracking.
    
    This command starts the MLflow UI web interface for viewing experiments,
    runs, models, and artifacts.
    
    Examples:
        # Start UI with default settings
        k-bert mlflow ui
        
        # Start on custom port and host
        k-bert mlflow ui --port 8080 --host 0.0.0.0
        
        # Use custom backend store
        k-bert mlflow ui --backend-store-uri sqlite:///mlflow.db
    """
    # Get configuration
    config = get_config()
    
    # Build mlflow ui command
    cmd = ["mlflow", "ui"]
    
    # Add host and port
    cmd.extend(["--host", host, "--port", str(port)])
    
    # Configure backend store
    store_uri = backend_store_uri
    if not store_uri and config.mlflow and config.mlflow.tracking_uri:
        if config.mlflow.tracking_uri.startswith("file://"):
            store_uri = config.mlflow.tracking_uri
        else:
            console.print(
                f"[yellow]Using MLflow tracking URI from config: {config.mlflow.tracking_uri}[/yellow]"
            )
    
    if store_uri:
        cmd.extend(["--backend-store-uri", store_uri])
    
    # Configure artifact root
    if default_artifact_root:
        cmd.extend(["--default-artifact-root", default_artifact_root])
    elif not backend_store_uri and config.mlflow:
        # Use a sensible default relative to tracking URI
        if hasattr(config.mlflow, 'artifacts_uri') and config.mlflow.artifacts_uri:
            cmd.extend(["--default-artifact-root", config.mlflow.artifacts_uri])
    
    # Development mode
    if dev:
        cmd.append("--dev")
    
    console.print(f"[bold blue]Starting MLflow UI...[/bold blue]")
    console.print(f"[dim]Command: {' '.join(cmd)}[/dim]")
    console.print(f"[bold]UI URL: http://{host}:{port}[/bold]")
    console.print("\n[yellow]Press Ctrl+C to stop the server[/yellow]\n")
    
    try:
        # Check if mlflow is available
        result = subprocess.run(
            ["mlflow", "--version"], 
            capture_output=True, 
            text=True, 
            timeout=5
        )
        if result.returncode != 0:
            raise FileNotFoundError("mlflow command not found")
        
    except (FileNotFoundError, subprocess.TimeoutExpired):
        console.print(
            "[red]MLflow CLI not found.[/red]\n"
            "Install with: [cyan]uv add mlflow[/cyan]"
        )
        raise typer.Exit(1)
    
    try:
        # Start MLflow UI
        subprocess.run(cmd, check=True)
        
    except KeyboardInterrupt:
        console.print("\n[green]MLflow UI stopped.[/green]")
    except subprocess.CalledProcessError as e:
        console.print(f"[red]Failed to start MLflow UI: {e}[/red]")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)