"""Training command implementation."""

from pathlib import Path
from typing import Optional, List
import typer
from loguru import logger

from ...utils import (
    get_console, print_success, print_error, print_info,
    handle_errors, track_time, requires_project,
    validate_path, validate_batch_size, validate_learning_rate, validate_epochs
)
from ...utils.console import create_progress, create_table

@handle_errors
@requires_project()
@track_time("Training BERT model")
def train_command(
    # Data arguments
    train_data: Path = typer.Option(..., "--train", "-t", help="Training data path", 
                                   callback=lambda p: validate_path(p, must_exist=True)),
    val_data: Optional[Path] = typer.Option(None, "--val", "-v", help="Validation data path",
                                          callback=lambda p: validate_path(p, must_exist=True) if p else None),
    test_data: Optional[Path] = typer.Option(None, "--test", help="Test data path",
                                           callback=lambda p: validate_path(p, must_exist=True) if p else None),
    
    # Model arguments
    model_type: str = typer.Option("modernbert", "--model", "-m", help="Model type to use"),
    pretrained: Optional[str] = typer.Option(None, "--pretrained", "-p", help="Pretrained model name or path"),
    
    # Training arguments
    epochs: int = typer.Option(3, "--epochs", "-e", help="Number of training epochs",
                              callback=validate_epochs),
    batch_size: int = typer.Option(32, "--batch-size", "-b", help="Training batch size",
                                  callback=validate_batch_size),
    learning_rate: float = typer.Option(2e-5, "--lr", "-l", help="Learning rate",
                                       callback=validate_learning_rate),
    warmup_steps: int = typer.Option(500, "--warmup", "-w", help="Number of warmup steps"),
    
    # Configuration
    config: Optional[Path] = typer.Option(None, "--config", "-c", help="Configuration file",
                                        callback=lambda p: validate_path(p, must_exist=True) if p else None),
    
    # Output arguments  
    output_dir: Path = typer.Option("output", "--output", "-o", help="Output directory"),
    experiment_name: Optional[str] = typer.Option(None, "--experiment", help="MLflow experiment name"),
    run_name: Optional[str] = typer.Option(None, "--run-name", help="Name for this training run"),
    
    # MLX-specific arguments
    use_mlx_embeddings: bool = typer.Option(False, "--mlx-embeddings", help="Use MLX embeddings"),
    gradient_accumulation: int = typer.Option(1, "--grad-accum", help="Gradient accumulation steps"),
    mixed_precision: bool = typer.Option(True, "--mixed-precision/--no-mixed-precision", help="Use mixed precision"),
    
    # Other options
    seed: int = typer.Option(42, "--seed", "-s", help="Random seed"),
    workers: int = typer.Option(4, "--workers", help="Number of data loading workers"),
    resume: Optional[Path] = typer.Option(None, "--resume", "-r", help="Resume from checkpoint"),
    early_stopping: bool = typer.Option(True, "--early-stopping/--no-early-stopping", help="Enable early stopping"),
    save_best_only: bool = typer.Option(False, "--save-best-only", help="Save only the best model"),
    
    # Advanced options
    augment: bool = typer.Option(False, "--augment", "-a", help="Enable data augmentation"),
    profile: bool = typer.Option(False, "--profile", help="Enable performance profiling"),
    debug: bool = typer.Option(False, "--debug", "-d", help="Enable debug mode"),
):
    """Train a BERT model for Kaggle competitions.
    
    This command provides comprehensive training functionality with MLX optimizations
    for Apple Silicon. It supports various model architectures, data formats, and
    training strategies.
    
    Examples:
        # Basic training
        bert train --train data/train.csv --val data/val.csv
        
        # Training with configuration file
        bert train --config configs/production.yaml
        
        # Resume training from checkpoint
        bert train --train data/train.csv --resume output/checkpoint_epoch_5
        
        # Training with MLX embeddings
        bert train --train data/train.csv --mlx-embeddings --model mlx-bert
    """
    console = get_console()
    
    # Show training configuration
    print_info("Training Configuration")
    
    config_table = create_table("Parameters", ["Parameter", "Value"])
    config_table.add_row("Model Type", model_type)
    config_table.add_row("Training Data", str(train_data))
    config_table.add_row("Validation Data", str(val_data) if val_data else "None")
    config_table.add_row("Batch Size", str(batch_size))
    config_table.add_row("Learning Rate", f"{learning_rate:.2e}")
    config_table.add_row("Epochs", str(epochs))
    config_table.add_row("Output Directory", str(output_dir))
    console.print(config_table)
    
    # Import training components
    try:
        from training.mlx_trainer import MLXTrainer
        from training.unified_config import UnifiedConfig
        from data.universal_loader import UniversalKaggleLoader
        from models.factory import UnifiedModelFactory
        from utils.mlflow_central import CentralMLflowConfig
    except ImportError as e:
        print_error(
            f"Failed to import training components: {str(e)}\n"
            "Make sure all dependencies are installed.",
            title="Import Error"
        )
        raise typer.Exit(1)
    
    # Load or create configuration
    if config:
        logger.info(f"Loading configuration from {config}")
        # Load config implementation here
        training_config = UnifiedConfig.from_yaml(str(config))
    else:
        # Create config from CLI arguments
        training_config = UnifiedConfig(
            # Model configuration
            model_type=model_type,
            pretrained_model=pretrained,
            
            # Training configuration
            num_epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            warmup_steps=warmup_steps,
            gradient_accumulation_steps=gradient_accumulation,
            
            # Data configuration
            train_path=str(train_data),
            val_path=str(val_data) if val_data else None,
            test_path=str(test_data) if test_data else None,
            num_workers=workers,
            
            # Output configuration
            output_dir=str(output_dir),
            
            # MLX configuration
            use_mlx_embeddings=use_mlx_embeddings,
            mixed_precision=mixed_precision,
            
            # Other configuration
            seed=seed,
            early_stopping_enabled=early_stopping,
            save_best_only=save_best_only,
            enable_augmentation=augment,
            enable_profiling=profile,
        )
    
    # Setup MLflow if experiment name provided
    if experiment_name:
        mlflow_config = CentralMLflowConfig()
        mlflow_config.create_experiment(experiment_name)
        if run_name:
            mlflow_config.set_run_name(run_name)
    
    # Create data loader
    with console.status("[bold blue]Loading data..."):
        data_loader = UniversalKaggleLoader(config=training_config.data)
        data_loader.setup()
        
        # Show dataset info
        dataset_info = data_loader.get_dataset_info()
        print_info(f"Dataset: {dataset_info['samples']} samples, "
                  f"{dataset_info['features']} features")
    
    # Create model
    with console.status("[bold blue]Creating model..."):
        model_factory = UnifiedModelFactory()
        model = model_factory.create_model(
            model_type=model_type,
            **training_config.model.dict()
        )
        
        print_info(f"Model created: {model.__class__.__name__}")
    
    # Create trainer
    trainer = MLXTrainer(
        model=model,
        config=training_config.training,
        data_module=data_loader,
    )
    
    # Resume if checkpoint provided
    if resume:
        logger.info(f"Resuming from checkpoint: {resume}")
        trainer.load_checkpoint(resume)
    
    # Start training
    console.print("\n[bold green]Starting training...[/bold green]\n")
    
    try:
        # Train with progress tracking
        metrics = trainer.train()
        
        # Show final results
        print_success("Training completed successfully!")
        
        results_table = create_table("Final Results", ["Metric", "Value"])
        for metric, value in metrics.items():
            if isinstance(value, float):
                results_table.add_row(metric, f"{value:.4f}")
            else:
                results_table.add_row(metric, str(value))
        console.print(results_table)
        
        # Save final model
        final_model_path = output_dir / "final_model"
        trainer.save_checkpoint(final_model_path)
        print_info(f"Model saved to: {final_model_path}")
        
        # Show next steps
        console.print("\n[bold]Next steps:[/bold]")
        console.print("1. Generate predictions: [cyan]bert predict --test data/test.csv "
                     f"--checkpoint {final_model_path}[/cyan]")
        console.print("2. View MLflow results: [cyan]bert mlflow ui[/cyan]")
        console.print("3. Submit to Kaggle: [cyan]bert kaggle submit --competition NAME[/cyan]")
        
    except KeyboardInterrupt:
        print_error("Training interrupted by user", title="Interrupted")
        raise typer.Exit(130)
    except Exception as e:
        print_error(f"Training failed: {str(e)}", title="Training Error")
        if debug:
            console.print_exception()
        raise typer.Exit(1)