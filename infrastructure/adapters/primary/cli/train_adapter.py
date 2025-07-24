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
from infrastructure.bootstrap import get_service
from application.ports.secondary.configuration import ConfigurationProvider


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
        default_config = Path.cwd() / "k-bert.yaml"
        if default_config.exists():
            config_provider.load_file(str(default_config))
    
    # Get configuration values with CLI overrides
    config_dict = config_provider.get_all()
    
    # Handle experiment selection
    if experiment and "experiments" in config_dict:
        if experiment in config_dict["experiments"]:
            # Merge experiment config
            exp_config = config_dict["experiments"][experiment]
            config_dict.update(exp_config)
        else:
            raise ValueError(f"Experiment '{experiment}' not found in configuration")
    
    # Apply CLI overrides
    if train_data:
        config_dict.setdefault("data", {})["train_path"] = str(train_data)
    if val_data:
        config_dict.setdefault("data", {})["val_path"] = str(val_data)
    if epochs is not None:
        config_dict.setdefault("training", {})["epochs"] = epochs
    if output_dir:
        config_dict.setdefault("training", {})["output_dir"] = str(output_dir)
    
    # Extract values for DTO
    models_config = config_dict.get("models", {})
    data_config = config_dict.get("data", {})
    training_config = config_dict.get("training", {})
    
    # Create request DTO
    return TrainingRequestDTO(
        # Model configuration
        model_type=models_config.get("type", "modernbert_with_head"),
        model_config={
            "model_name": models_config.get("default_model", "answerdotai/ModernBERT-base"),
            "head_type": models_config.get("head_type", "binary_classification"),
            "num_labels": models_config.get("num_labels", 2),
            "hidden_size": models_config.get("hidden_size", 768),
            "num_hidden_layers": models_config.get("num_hidden_layers", 12),
            "num_attention_heads": models_config.get("num_attention_heads", 12),
            "intermediate_size": models_config.get("intermediate_size", 3072),
            "max_position_embeddings": models_config.get("max_position_embeddings", 512),
        },
        
        # Data paths
        train_data_path=Path(data_config.get("train_path", "data/train.csv")),
        val_data_path=Path(data_config.get("val_path")) if data_config.get("val_path") else None,
        test_data_path=Path(data_config.get("test_path")) if data_config.get("test_path") else None,
        
        # Training configuration
        num_epochs=training_config.get("epochs", 3),
        batch_size=training_config.get("batch_size", data_config.get("batch_size", 32)),
        learning_rate=training_config.get("learning_rate", 5e-5),
        weight_decay=training_config.get("weight_decay", 0.01),
        max_grad_norm=training_config.get("max_grad_norm", 1.0),
        
        # Optimizer configuration
        optimizer_type=training_config.get("optimizer", "adamw"),
        optimizer_params={
            "beta1": training_config.get("beta1", 0.9),
            "beta2": training_config.get("beta2", 0.999),
            "epsilon": training_config.get("epsilon", 1e-8),
        },
        
        # Scheduler configuration
        scheduler_type=training_config.get("scheduler", "warmup_linear"),
        warmup_steps=training_config.get("warmup_steps"),
        warmup_ratio=training_config.get("warmup_ratio", 0.1),
        
        # Evaluation configuration
        eval_strategy=training_config.get("eval_strategy", "epoch"),
        eval_steps=training_config.get("eval_steps"),
        save_strategy=training_config.get("save_strategy", "epoch"),
        save_steps=training_config.get("save_steps"),
        
        # Early stopping
        early_stopping_patience=training_config.get("early_stopping_patience"),
        early_stopping_threshold=training_config.get("early_stopping_threshold", 0.0),
        metric_for_best_model=training_config.get("metric_for_best_model", "eval_loss"),
        greater_is_better=training_config.get("greater_is_better", False),
        
        # Output configuration
        output_dir=Path(training_config.get("output_dir", "output")),
        run_name=training_config.get("run_name") or f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        experiment_name=config_dict.get("mlflow", {}).get("experiment_name", "k-bert-experiments"),
        
        # MLflow
        use_mlflow=config_dict.get("mlflow", {}).get("enabled", True),
        mlflow_tracking_uri=config_dict.get("mlflow", {}).get("tracking_uri"),
        tags=config_dict.get("mlflow", {}).get("tags", {}),
    )


def display_configuration_summary(request: TrainingRequestDTO) -> None:
    """Display training configuration in a formatted table."""
    console.print("\n[bold]Training Configuration[/bold]")
    
    table = Table(show_header=False)
    table.add_column("Parameter", style="cyan")
    table.add_column("Value")
    
    # Model info
    table.add_row("Model Type", request.model_type)
    table.add_row("Model", request.model_config.get("model_name", "unknown"))
    
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
    ctx: typer.Context,
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
        # Get configuration provider from context or fallback
        container = ctx.obj
        if container:
            config_provider = container.resolve(ConfigurationProvider)
            use_case = container.resolve(TrainModelUseCase)
        else:
            # Fallback to bootstrap for backwards compatibility
            config_provider = get_service(ConfigurationProvider)
            use_case = get_service(TrainModelUseCase)
        
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


# Export the command function
train = train_command