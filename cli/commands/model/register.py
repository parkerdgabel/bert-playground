"""Model registration command for MLflow integration."""

import json
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console

from ...config import get_config
from ...utils import handle_errors


console = Console()


@handle_errors  
def register_command(
    checkpoint: Path = typer.Argument(..., help="Path to model checkpoint"),
    name: str = typer.Option(..., "--name", "-n", help="Model name in registry"),
    stage: str = typer.Option(
        "Staging", 
        "--stage", 
        "-s", 
        help="Model stage: Staging, Production, Archived"
    ),
    description: Optional[str] = typer.Option(
        None, 
        "--description", 
        "-d", 
        help="Model description"
    ),
    tags: Optional[str] = typer.Option(
        None,
        "--tags",
        "-t", 
        help="Comma-separated tags"
    ),
    version: Optional[str] = typer.Option(
        None,
        "--version",
        "-v",
        help="Model version (auto-generated if not provided)"
    ),
):
    """Register a model in MLflow Model Registry.
    
    This command registers a trained model checkpoint in MLflow's Model Registry
    for versioning, stage management, and deployment tracking.
    
    Examples:
        # Register model with automatic versioning
        k-bert model register path/to/checkpoint --name my-bert-model
        
        # Register with specific version and stage
        k-bert model register path/to/checkpoint \\
            --name my-bert-model --version 2.1.0 --stage Production
            
        # Register with description and tags
        k-bert model register path/to/checkpoint \\
            --name my-bert-model --description "Competition winner" \\
            --tags "titanic,bert,production"
    """
    # Validate checkpoint path
    if not checkpoint.exists():
        console.print(f"[red]Checkpoint not found: {checkpoint}[/red]")
        raise typer.Exit(1)
    
    # Check for required model files
    config_file = checkpoint / "config.json"
    weights_file = checkpoint / "model.safetensors"
    
    if not config_file.exists():
        console.print(f"[red]Model config not found: {config_file}[/red]")
        raise typer.Exit(1)
        
    if not weights_file.exists():
        console.print(f"[red]Model weights not found: {weights_file}[/red]")
        raise typer.Exit(1)
    
    # Get configuration
    config = get_config()
    
    try:
        import mlflow
        import mlflow.pyfunc
        from mlflow.tracking import MlflowClient
        
        # Set tracking URI
        if config.mlflow and config.mlflow.tracking_uri:
            mlflow.set_tracking_uri(config.mlflow.tracking_uri)
        
        client = MlflowClient()
        
        console.print(f"[dim]Registering model from: {checkpoint}[/dim]")
        
        # Load model config for metadata
        with open(config_file) as f:
            model_config = json.load(f)
        
        # Prepare model metadata
        model_metadata = {
            "framework": "MLX",
            "architecture": model_config.get("model_type", "ModernBERT"),
            "parameters": model_config.get("num_parameters"),
            "vocab_size": model_config.get("vocab_size"),
            "hidden_size": model_config.get("hidden_size"),
            "num_layers": model_config.get("num_hidden_layers"),
            "num_attention_heads": model_config.get("num_attention_heads"),
            "checkpoint_path": str(checkpoint),
        }
        
        # Add user-provided metadata
        if description:
            model_metadata["description"] = description
        
        # Parse tags
        tag_dict = {}
        if tags:
            for tag in tags.split(","):
                tag = tag.strip()
                if "=" in tag:
                    key, value = tag.split("=", 1)
                    tag_dict[key.strip()] = value.strip()
                else:
                    tag_dict[tag] = "true"
        
        # Add default tags
        tag_dict.update({
            "model_type": model_config.get("model_type", "ModernBERT"),
            "framework": "MLX",
            "registered_by": "k-bert-cli",
        })
        
        # Create a simple model wrapper for MLflow
        import tempfile
        import shutil
        
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir) / "model"
            temp_path.mkdir()
            
            # Copy model files
            shutil.copy2(config_file, temp_path / "config.json")
            shutil.copy2(weights_file, temp_path / "model.safetensors")
            
            # Create MLmodel file
            mlmodel_content = f"""
artifact_path: model
flavors:
  python_function:
    env: conda.yaml
    loader_module: k_bert_model_loader
    python_version: 3.11.0
model_uuid: {model_metadata.get('model_uuid', 'generated')}
run_id: null
signature:
  inputs: '[{{"name": "text", "type": "string"}}]'
  outputs: '[{{"name": "prediction", "type": "string"}}, {{"name": "confidence", "type": "double"}}]'
utc_time_created: '{mlflow.utils.time_utils.get_current_time_millis()}'
"""
            
            with open(temp_path / "MLmodel", "w") as f:
                f.write(mlmodel_content.strip())
            
            # Create conda.yaml
            conda_content = """
name: k-bert-env
channels:
  - conda-forge
dependencies:
  - python=3.11
  - pip
  - pip:
    - mlx
    - transformers
    - safetensors
    - numpy
"""
            with open(temp_path / "conda.yaml", "w") as f:
                f.write(conda_content.strip())
            
            # Register model
            console.print(f"[dim]Registering model '{name}' in MLflow...[/dim]")
            
            # First, log the model to get a run URI
            with mlflow.start_run():
                model_uri = mlflow.pyfunc.log_model(
                    artifact_path="model",
                    python_model=None,  # We're just storing artifacts
                    artifacts={"model_path": str(temp_path)},
                    registered_model_name=name,
                )
                run_id = mlflow.active_run().info.run_id
            
            # Get the registered model version
            registered_models = client.search_registered_models(
                filter_string=f"name='{name}'"
            )
            
            if registered_models:
                latest_version = client.get_latest_versions(
                    name, stages=["None", "Staging", "Production", "Archived"]
                )[0]
                model_version = latest_version.version
            else:
                model_version = "1"
            
            # Update model version with metadata and tags
            client.update_model_version(
                name=name,
                version=model_version,
                description=description or f"K-BERT model registered from {checkpoint}"
            )
            
            # Set tags
            for key, value in tag_dict.items():
                client.set_model_version_tag(
                    name=name,
                    version=model_version,
                    key=key,
                    value=str(value)
                )
            
            # Transition to specified stage if not None/Staging
            if stage and stage != "Staging":
                client.transition_model_version_stage(
                    name=name,
                    version=model_version,
                    stage=stage,
                    archive_existing_versions=False
                )
        
        console.print(f"[bold green]✓ Model registered successfully![/bold green]")
        console.print(f"[bold]Name:[/bold] {name}")
        console.print(f"[bold]Version:[/bold] {model_version}")
        console.print(f"[bold]Stage:[/bold] {stage}")
        console.print(f"[bold]MLflow URI:[/bold] {model_uri}")
        
        # Show model details
        console.print("\n[bold]Model Details:[/bold]")
        for key, value in model_metadata.items():
            if value is not None:
                console.print(f"  {key}: {value}")
        
        if tag_dict:
            console.print("\n[bold]Tags:[/bold]")
            for key, value in tag_dict.items():
                console.print(f"  {key}: {value}")
        
        console.print(
            "\n[bold]Next steps:[/bold]\n"
            f"  • View in MLflow UI: [cyan]k-bert mlflow ui[/cyan]\n"
            f"  • Deploy model: [cyan]k-bert model serve --registry {name}:{model_version}[/cyan]\n"
            f"  • Update stage: [cyan]mlflow models transition-stage --name {name} --version {model_version} --stage Production[/cyan]"
        )
        
    except ImportError:
        console.print(
            "[red]MLflow not installed.[/red]\n"
            "Install with: [cyan]uv add mlflow[/cyan]"
        )
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"[red]Failed to register model: {e}[/red]")
        raise typer.Exit(1)