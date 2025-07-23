"""Train command for the CLI.

This is a thin adapter that converts CLI arguments to DTOs and delegates
to the application layer TrainModelCommand.
"""

from pathlib import Path
from typing import Optional
import asyncio

import typer
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn
from loguru import logger

from cli.bootstrap import initialize_cli, get_command, get_config, shutdown_cli
from cli.config.loader import ConfigurationLoader
from application.commands.train import TrainModelCommand
from application.dto.training import TrainingRequestDTO


console = Console()


def create_training_request(
    config: dict,
    experiment: Optional[str] = None,
) -> TrainingRequestDTO:
    """Create TrainingRequestDTO from configuration.
    
    Args:
        config: Configuration dictionary
        experiment: Experiment name
        
    Returns:
        Training request DTO
    """
    # Extract configuration with defaults
    model_config = config.get("models", {})
    data_config = config.get("data", {})
    training_config = config.get("training", {})
    mlflow_config = config.get("mlflow", {})
    
    # Create the DTO
    return TrainingRequestDTO(
        # Model configuration
        model_type=model_config.get("type", "modernbert_with_head"),
        model_config={
            "model_name": model_config.get("default_model", "answerdotai/ModernBERT-base"),
            "head_type": model_config.get("head_type", "binary_classification"),
            "num_labels": model_config.get("num_labels", 2),
            **model_config.get("config", {})
        },
        
        # Data configuration
        train_data_path=Path(data_config["train_path"]),
        val_data_path=Path(val_path) if (val_path := data_config.get("val_path")) else None,
        test_data_path=Path(test_path) if (test_path := data_config.get("test_path")) else None,
        
        # Training configuration
        num_epochs=training_config.get("epochs", 3),
        batch_size=data_config.get("batch_size", 32),
        learning_rate=training_config.get("learning_rate", 5e-5),
        weight_decay=training_config.get("weight_decay", 0.01),
        max_grad_norm=training_config.get("max_grad_norm", 1.0),
        
        # Optimizer configuration
        optimizer_type=training_config.get("optimizer", "adamw"),
        optimizer_params=training_config.get("optimizer_params", {}),
        
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
        
        # Advanced options
        gradient_accumulation_steps=training_config.get("gradient_accumulation_steps", 1),
        use_mixed_precision=training_config.get("use_mixed_precision", False),
        gradient_checkpointing=training_config.get("gradient_checkpointing", False),
        label_smoothing_factor=training_config.get("label_smoothing_factor", 0.0),
        
        # Output configuration
        output_dir=Path(training_config.get("output_dir", "output")),
        run_name=experiment,
        experiment_name=experiment or mlflow_config.get("experiment_name", "default"),
        
        # Logging configuration
        logging_steps=training_config.get("logging_steps", 100),
        logging_first_step=training_config.get("logging_first_step", True),
        
        # Checkpointing
        save_total_limit=training_config.get("save_total_limit"),
        load_best_model_at_end=training_config.get("load_best_model_at_end", True),
        resume_from_checkpoint=Path(ckpt) if (ckpt := training_config.get("resume_from_checkpoint")) else None,
        
        # Worker configuration
        num_workers=data_config.get("num_workers", 0),
        
        # Tracking configuration
        use_mlflow=mlflow_config.get("enabled", True),
        mlflow_tracking_uri=mlflow_config.get("tracking_uri"),
        tags={"source": "cli", "command": "train"},
    )


def display_configuration(request: TrainingRequestDTO) -> None:
    """Display training configuration in a table."""
    table = Table(title="Training Configuration", show_header=True)
    table.add_column("Parameter", style="cyan")
    table.add_column("Value", style="green")
    
    # Model info
    table.add_row("Model Type", request.model_type)
    table.add_row("Model Name", request.model_config.get("model_name", "N/A"))
    table.add_row("Head Type", request.model_config.get("head_type", "N/A"))
    
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


def display_results(response) -> None:
    """Display training results."""
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
        console.print(f"  • Evaluate model: [cyan]k-bert evaluate --model {response.best_model_path}[/cyan]")
        console.print(f"  • Generate predictions: [cyan]k-bert predict --model {response.best_model_path} --input data.csv[/cyan]")
    console.print(f"  • View run details: [cyan]k-bert info --run {response.run_id}[/cyan]")
    if response.mlflow_run_url:
        console.print(f"  • View in MLflow: [cyan]{response.mlflow_run_url}[/cyan]")


def train(
    config: Optional[Path] = typer.Option(
        None,
        "--config", "-c",
        help="Configuration file (defaults to k-bert.yaml in current directory)",
    ),
    experiment: Optional[str] = typer.Option(
        None,
        "--experiment", "-e",
        help="Experiment name for tracking",
    ),
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
    batch_size: Optional[int] = typer.Option(
        None,
        "--batch-size", "-b",
        help="Override batch size",
    ),
    learning_rate: Optional[float] = typer.Option(
        None,
        "--lr",
        help="Override learning rate",
    ),
    output_dir: Optional[Path] = typer.Option(
        None,
        "--output", "-o",
        help="Override output directory",
    ),
    resume: Optional[Path] = typer.Option(
        None,
        "--resume",
        help="Resume from checkpoint",
    ),
    no_config: bool = typer.Option(
        False,
        "--no-config",
        help="Run with defaults only (no config file)",
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Show configuration without running",
    ),
    debug: bool = typer.Option(
        False,
        "--debug", "-d",
        help="Enable debug logging",
    ),
):
    """Train a BERT model.
    
    This command trains a BERT model using the configuration-first approach.
    Configuration is loaded from (in order of precedence):
    
    1. Command-line arguments
    2. --config file (if specified)
    3. k-bert.yaml in current directory
    4. ~/.k-bert/config.yaml (user config)
    
    Examples:
        # Train with project config
        k-bert train
        
        # Train with specific config
        k-bert train --config configs/production.yaml
        
        # Override specific parameters
        k-bert train --epochs 10 --lr 1e-5
        
        # Resume from checkpoint
        k-bert train --resume output/checkpoint-1000
    """
    console.print("\n[bold blue]K-BERT Training[/bold blue]")
    console.print("=" * 60)
    
    try:
        # Load configuration
        loader = ConfigurationLoader()
        configs = []
        
        if not no_config:
            # Load user config
            user_config_path = loader.find_user_config()
            if user_config_path:
                configs.append(loader.load_yaml_config(user_config_path))
            
            # Load project config
            project_config_path = loader.find_project_config()
            if project_config_path:
                configs.append(loader.load_yaml_config(project_config_path))
            
            # Load command config
            if config:
                configs.append(loader.load_yaml_config(config))
        
        # Merge configurations
        merged_config = loader.merge_configs(configs) if configs else {}
        
        # Apply CLI overrides
        merged_config = loader.apply_cli_overrides(
            merged_config,
            train_data=train_data,
            val_data=val_data,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            output_dir=output_dir,
            checkpoint=resume,
        )
        
        # Validate configuration
        errors = loader.validate_config(merged_config, "train")
        if errors:
            console.print("[red]Configuration errors:[/red]")
            for error in errors:
                console.print(f"  • {error}")
            raise typer.Exit(1)
        
        # Initialize CLI with configuration paths
        initialize_cli(
            config_path=config,
            user_config_path=user_config_path,
            project_config_path=project_config_path,
        )
        
        # Create training request
        request = create_training_request(merged_config, experiment)
        
        # Display configuration
        display_configuration(request)
        
        if dry_run:
            console.print("\n[yellow]Dry run mode - no training performed[/yellow]")
            return
        
        # Get the training command
        train_command = get_command(TrainModelCommand)
        
        # Run training with progress indicator
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
            transient=True,
        ) as progress:
            task = progress.add_task("Training model...", total=None)
            
            # Run the async command
            response = asyncio.run(train_command.execute(request))
        
        # Display results
        display_results(response)
        
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
    finally:
        # Ensure cleanup
        shutdown_cli()


if __name__ == "__main__":
    typer.run(train)