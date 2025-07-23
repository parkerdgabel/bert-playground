"""Thin CLI adapter for training command.

This adapter is responsible only for:
1. Parsing command-line arguments
2. Creating the TrainingRequestDTO
3. Calling the TrainModelUseCase
4. Formatting and displaying the response

No business logic should exist in this adapter.
"""

from pathlib import Path
from typing import Optional
from datetime import datetime

import typer
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn
from loguru import logger

from application.dto.training import TrainingRequestDTO
from application.use_cases.train_model import TrainModelUseCase
from core.bootstrap import get_service
from core.ports.config import ConfigurationProvider


console = Console()


def create_train_request_dto(
    config_provider: ConfigurationProvider,
    config: Optional[Path],
    experiment: Optional[str],
    train_data: Optional[Path],
    val_data: Optional[Path],
    epochs: Optional[int],
    output_dir: Optional[Path],
    no_config: bool,
) -> TrainingRequestDTO:
    """Create TrainingRequestDTO from CLI arguments and configuration.
    
    This method consolidates configuration loading and DTO creation,
    keeping the main command handler focused on UI concerns.
    """
    # Load configuration
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
    
    # Apply CLI overrides
    if train_data:
        config_provider.set("data.train_path", str(train_data))
    if val_data:
        config_provider.set("data.val_path", str(val_data))
    if epochs is not None:
        config_provider.set("training.epochs", epochs)
    if output_dir:
        config_provider.set("training.output_dir", str(output_dir))
    
    # Extract configuration values
    train_path = config_provider.get("data.train_path")
    if not train_path:
        raise ValueError("Training data path not specified")
    
    # Create DTO
    return TrainingRequestDTO(
        # Model configuration
        model_type=config_provider.get("models.type", "modernbert_with_head"),
        model_config={
            "model_name": config_provider.get("models.default_model", "answerdotai/ModernBERT-base"),
            "head_type": config_provider.get("models.head_type", "binary_classification"),
            "num_labels": config_provider.get("models.num_labels", 2),
        },
        
        # Data configuration
        train_data_path=Path(train_path),
        val_data_path=Path(val_path) if (val_path := config_provider.get("data.val_path")) else None,
        
        # Training configuration
        num_epochs=config_provider.get("training.epochs", 3),
        batch_size=config_provider.get("data.batch_size", 32),
        learning_rate=config_provider.get("training.learning_rate", 5e-5),
        weight_decay=config_provider.get("training.weight_decay", 0.01),
        max_grad_norm=config_provider.get("training.max_grad_norm", 1.0),
        
        # Optimizer configuration
        optimizer_type=config_provider.get("training.optimizer", "adamw"),
        
        # Scheduler configuration
        scheduler_type=config_provider.get("training.scheduler", "warmup_linear"),
        warmup_ratio=config_provider.get("training.warmup_ratio", 0.1),
        
        # Evaluation configuration
        eval_strategy=config_provider.get("training.eval_strategy", "epoch"),
        save_strategy=config_provider.get("training.save_strategy", "epoch"),
        
        # Early stopping
        early_stopping_patience=config_provider.get("training.early_stopping_patience"),
        early_stopping_threshold=config_provider.get("training.early_stopping_threshold", 0.0),
        metric_for_best_model=config_provider.get("training.metric_for_best_model", "eval_loss"),
        greater_is_better=config_provider.get("training.greater_is_better", False),
        
        # Advanced options
        gradient_accumulation_steps=config_provider.get("training.gradient_accumulation_steps", 1),
        use_mixed_precision=config_provider.get("training.use_mixed_precision", False),
        gradient_checkpointing=config_provider.get("training.gradient_checkpointing", False),
        
        # Output configuration
        output_dir=Path(config_provider.get("training.output_dir", "output")),
        run_name=experiment,
        experiment_name=experiment,
        
        # Logging configuration
        logging_steps=config_provider.get("training.logging_steps", 100),
        logging_first_step=config_provider.get("training.logging_first_step", True),
        
        # Checkpointing
        save_total_limit=config_provider.get("training.save_total_limit"),
        load_best_model_at_end=config_provider.get("training.load_best_model_at_end", True),
        
        # Tracking configuration
        use_mlflow=config_provider.get("mlflow.enabled", True),
        mlflow_tracking_uri=config_provider.get("mlflow.tracking_uri"),
        tags={"source": "cli", "command": "train"},
    )


def display_configuration_summary(request: TrainingRequestDTO) -> None:
    """Display configuration summary in a formatted table."""
    table = Table(title="Training Configuration", show_header=True)
    table.add_column("Parameter", style="cyan")
    table.add_column("Value", style="green")
    
    # Model info
    table.add_row("Model Type", request.model_type)
    table.add_row("Model Name", request.model_config.get("model_name", "N/A"))
    
    # Data info
    table.add_row("Training Data", str(request.train_data_path))
    if request.val_data_path:
        table.add_row("Validation Data", str(request.val_data_path))
    
    # Training params
    table.add_row("Epochs", str(request.num_epochs))
    table.add_row("Batch Size", str(request.batch_size))
    table.add_row("Learning Rate", f"{request.learning_rate:.2e}")
    table.add_row("Optimizer", request.optimizer_type)
    
    # Output info
    table.add_row("Output Directory", str(request.output_dir))
    if request.run_name:
        table.add_row("Run Name", request.run_name)
    
    console.print(table)


def display_training_results(response) -> None:
    """Display training results in a formatted manner."""
    if not response.success:
        console.print(f"\n[red]Training failed: {response.error_message}[/red]")
        return
    
    console.print("\n[bold green]Training completed successfully![/bold green]\n")
    
    # Results table
    table = Table(title="Training Results", show_header=True)
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    
    # Loss metrics
    table.add_row("Final Train Loss", f"{response.final_train_loss:.4f}")
    if response.final_val_loss is not None:
        table.add_row("Final Val Loss", f"{response.final_val_loss:.4f}")
    if response.best_val_loss is not None:
        table.add_row("Best Val Loss", f"{response.best_val_loss:.4f}")
    
    # Training info
    table.add_row("Total Epochs", str(response.total_epochs))
    table.add_row("Total Steps", str(response.total_steps))
    table.add_row("Training Time", f"{response.total_time_seconds:.1f}s")
    
    # Early stopping info
    if response.early_stopped:
        table.add_row("Early Stopped", "Yes")
        table.add_row("Stop Reason", response.stop_reason or "N/A")
    
    console.print(table)
    
    # Model paths
    if response.final_model_path:
        console.print(f"\nFinal model saved to: [cyan]{response.final_model_path}[/cyan]")
    if response.best_model_path:
        console.print(f"Best model saved to: [cyan]{response.best_model_path}[/cyan]")
    
    # Next steps
    console.print("\n[bold]Next steps:[/bold]")
    if response.best_model_path:
        console.print(f"  • Generate predictions: [cyan]k-bert predict --checkpoint {response.best_model_path} --test data/test.csv[/cyan]")
    console.print(f"  • View detailed metrics: [cyan]k-bert info --run {response.run_id}[/cyan]")
    if response.mlflow_run_url:
        console.print(f"  • View in MLflow: [cyan]{response.mlflow_run_url}[/cyan]")


async def train_command(
    # Config-first approach
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
    # Minimal overrides
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
    """Train a BERT model.
    
    This is a thin CLI adapter that:
    1. Parses arguments
    2. Creates TrainingRequestDTO
    3. Calls TrainModelUseCase
    4. Displays results
    
    All business logic is handled by the use case.
    """
    console.print("\n[bold blue]K-BERT Training[/bold blue]")
    console.print("=" * 60)
    
    try:
        # Get configuration provider
        config_provider = get_service(ConfigurationProvider)
        
        # Create request DTO
        request = create_train_request_dto(
            config_provider=config_provider,
            config=config,
            experiment=experiment,
            train_data=train_data,
            val_data=val_data,
            epochs=epochs,
            output_dir=output_dir,
            no_config=no_config,
        )
        
        # Display configuration
        display_configuration_summary(request)
        
        if dry_run:
            console.print("\n[yellow]Dry run mode - no training performed[/yellow]")
            return
        
        # Get use case
        use_case = get_service(TrainModelUseCase)
        
        # Run training with progress indicator
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
            transient=True,
        ) as progress:
            task = progress.add_task("Training model...", total=None)
            response = await use_case.execute(request)
        
        # Display results
        display_training_results(response)
        
    except ValueError as e:
        console.print(f"\n[red]Configuration error: {e}[/red]")
        raise typer.Exit(1)
    except KeyboardInterrupt:
        console.print("\n[yellow]Training interrupted by user[/yellow]")
        raise typer.Exit(130)
    except Exception as e:
        console.print(f"\n[red]Error: {e}[/red]")
        if debug:
            import traceback
            traceback.print_exc()
        raise typer.Exit(1)


# Create the Typer command
train = typer.command()(train_command)