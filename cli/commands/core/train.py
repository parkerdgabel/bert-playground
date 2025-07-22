"""Training command implementation using hexagonal architecture and dependency injection."""

import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import typer
from loguru import logger
from rich.console import Console

from core.bootstrap import get_service
from core.ports.config import ConfigurationProvider
from core.ports.storage import StorageService
from core.ports.monitoring import MonitoringService
from core.events.bus import EventBus
from core.events.types import EventType
from training.components.training_orchestrator import TrainingOrchestrator
from data.factory import DatasetFactory
from models.factory import ModelFactory

console = Console()


def train_command(
    # Config-first approach: config is optional but encouraged
    config: Optional[Path] = typer.Option(
        None,
        "--config",
        "-c",
        help="Configuration file (defaults to k-bert.yaml in current directory)",
    ),
    experiment: Optional[str] = typer.Option(
        None,
        "--experiment",
        "-e",
        help="Run specific experiment from configuration",
    ),
    # Minimal overrides for common use cases
    train_data: Optional[Path] = typer.Option(
        None,
        "--train",
        help="Override training data path",
    ),
    val_data: Optional[Path] = typer.Option(
        None,
        "--val",
        help="Override validation data path", 
    ),
    epochs: Optional[int] = typer.Option(
        None,
        "--epochs",
        help="Override number of epochs",
    ),
    output_dir: Optional[Path] = typer.Option(
        None,
        "--output",
        "-o",
        help="Override output directory",
    ),
    # Control flags
    no_config: bool = typer.Option(
        False,
        "--no-config",
        help="Run with default settings (no configuration file)",
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Show configuration without running training",
    ),
    debug: bool = typer.Option(
        False,
        "--debug",
        "-d",
        help="Enable debug logging",
    ),
):
    """Train a BERT model using hexagonal architecture and dependency injection.
    
    This command uses the hexagonal architecture with dependency injection
    to abstract external dependencies through ports and adapters.
    
    Examples:
        # Train with project configuration
        k-bert train
        
        # Train with specific experiment
        k-bert train --experiment titanic_baseline
        
        # Override specific settings
        k-bert train --epochs 10 --train custom_train.csv
        
        # Dry run to see configuration
        k-bert train --dry-run
    """
    console.print("\n[bold blue]K-BERT Training (Hexagonal Architecture)[/bold blue]")
    console.print("=" * 60)
    
    # Get services through dependency injection
    config_provider = get_service(ConfigurationProvider)
    storage_service = get_service(StorageService)
    monitoring = get_service(MonitoringService) 
    event_bus = get_service(EventBus)
    
    # Configure logging level
    if debug:
        monitoring.set_level("DEBUG")
    else:
        monitoring.set_level("INFO")
    
    # Load configuration through the configuration port
    if config:
        config_provider.load_file(str(config))
    elif not no_config:
        # Look for k-bert.yaml in current directory
        config_paths = [
            Path.cwd() / "k-bert.yaml",
            Path.cwd() / "k-bert.yml", 
            Path.cwd() / ".k-bert.yaml",
        ]
        
        config_file = next((p for p in config_paths if p.exists()), None)
        if config_file:
            config_provider.load_file(str(config_file))
            console.print(f"[green]Using configuration: {config_file}[/green]")
        else:
            console.print("[yellow]No config file found, using defaults[/yellow]")
    
    # Handle no-config mode
    if no_config:
        if not train_data:
            console.print("[red]Error: Training data is required when using --no-config[/red]")
            raise typer.Exit(1)
        console.print("[yellow]Running with default configuration (--no-config)[/yellow]")
    
    # Apply CLI overrides
    if train_data:
        config_provider.set("data.train_path", str(train_data))
    if val_data:
        config_provider.set("data.val_path", str(val_data))
    if epochs is not None:
        config_provider.set("training.epochs", epochs)
    if output_dir:
        config_provider.set("training.output_dir", str(output_dir))
    
    # Handle experiment selection (simplified for now)
    if experiment:
        console.print(f"[cyan]Running experiment: {experiment}[/cyan]")
        # TODO: Load experiment config from configuration provider
    
    # Get configuration values through the configuration port
    model_name = config_provider.get("models.default_model", "answerdotai/ModernBERT-base")
    train_path = config_provider.get("data.train_path")
    val_path = config_provider.get("data.val_path") 
    training_epochs = config_provider.get("training.epochs", 5)
    batch_size = config_provider.get("data.batch_size", 32)
    learning_rate = config_provider.get("training.learning_rate", 2e-5)
    output_path = config_provider.get("training.output_dir", "output")
    
    # Validate required paths
    if not train_path:
        console.print("[red]Error: Training data path not specified in configuration[/red]")
        raise typer.Exit(1)
    
    # Display configuration summary
    console.print("\n[bold]Configuration Summary:[/bold]")
    console.print(f"  Model: {model_name}")
    console.print(f"  Training data: {train_path}")
    if val_path:
        console.print(f"  Validation data: {val_path}")
    console.print(f"  Epochs: {training_epochs}")
    console.print(f"  Batch size: {batch_size}")
    console.print(f"  Learning rate: {learning_rate}")
    console.print(f"  Output: {output_path}")
    
    if dry_run:
        console.print("\n[yellow]Dry run mode - no training performed[/yellow]")
        console.print("[green]✓ Configuration validated successfully[/green]")
        return
    
    # Create run directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{experiment}_{timestamp}" if experiment else f"run_{timestamp}"
    run_dir = Path(output_path) / run_name
    
    # Create directory through storage service
    storage_service.create_directory(str(run_dir))
    
    # Setup monitoring for this run
    monitoring.start_run(run_name, {
        "model": model_name, 
        "epochs": training_epochs,
        "train_path": train_path
    })
    
    # Emit training started event
    event_bus.emit(
        EventType.TRAINING,
        "training_started",
        source="train_command",
        data={
            "model": model_name,
            "train_path": train_path,
            "val_path": val_path, 
            "epochs": training_epochs,
            "run_name": run_name
        }
    )
    
    try:
        # Create data loaders through factory (which uses DI internally)
        dataset_factory = get_service(DatasetFactory)
        
        console.print("\n[dim]Creating data loaders...[/dim]")
        train_loader = dataset_factory.create_dataloader(
            data_path=train_path,
            batch_size=batch_size,
            shuffle=True,
            split="train"
        )
        
        val_loader = None
        if val_path:
            val_loader = dataset_factory.create_dataloader(
                data_path=val_path,
                batch_size=batch_size * 2,  # Larger batch for validation
                shuffle=False,
                split="val"
            )
        
        # Create model through factory (which uses DI internally)
        console.print("[dim]Creating model...[/dim]")
        model_factory = get_service(ModelFactory)
        
        model = model_factory.create_model(
            model_name=model_name,
            model_type="modernbert_with_head",
            head_type="binary_classification",  # TODO: Get from config
            num_labels=2
        )
        
        # Get training orchestrator through DI
        orchestrator = get_service(TrainingOrchestrator)
        
        # Configure orchestrator
        training_config = {
            "epochs": training_epochs,
            "learning_rate": learning_rate,
            "output_dir": str(run_dir),
            "run_name": run_name
        }
        
        orchestrator.configure(training_config)
        
        # Save configuration through storage service
        config_data = {
            "model": model_name,
            "train_path": train_path,
            "val_path": val_path,
            "epochs": training_epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "timestamp": timestamp,
            "run_name": run_name
        }
        
        storage_service.save_json(
            str(run_dir / "training_config.json"),
            config_data
        )
    
        # Start training
        console.print("\n[bold green]Starting training...[/bold green]\n")
        
        result = orchestrator.train(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader
        )
        
        # Emit training completed event
        event_bus.emit(
            EventType.TRAINING,
            "training_completed", 
            source="train_command",
            data={
                "run_name": run_name,
                "result": result,
                "success": True
            }
        )
        
        console.print("\n[bold green]Training completed successfully![/bold green]")
        console.print(f"Model saved to: {run_dir}")
        
        # Show next steps
        console.print("\n[bold]Next steps:[/bold]")
        console.print(f"  • Generate predictions: [cyan]k-bert predict --checkpoint {run_dir}/final --test data/test.csv[/cyan]")
        console.print(f"  • View metrics: [cyan]k-bert info --run {run_name}[/cyan]")
        
    except KeyboardInterrupt:
        monitoring.log_error("Training interrupted by user")
        event_bus.emit(
            EventType.TRAINING,
            "training_interrupted",
            source="train_command", 
            data={"run_name": run_name}
        )
        raise typer.Exit(130)
        
    except Exception as e:
        monitoring.log_error(f"Training failed: {str(e)}")
        event_bus.emit(
            EventType.TRAINING,
            "training_failed",
            source="train_command",
            data={
                "run_name": run_name,
                "error": str(e)
            }
        )
        
        if debug:
            import traceback
            traceback.print_exc()
        
        raise typer.Exit(1)
    
    finally:
        monitoring.end_run()