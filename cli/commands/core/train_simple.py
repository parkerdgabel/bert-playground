"""Simplified training command implementation."""

from pathlib import Path
from typing import Optional
import typer
from loguru import logger
from datetime import datetime
import json
import sys

from ...utils import (
    get_console, print_success, print_error, print_info,
    handle_errors, track_time,
    validate_path, validate_batch_size, validate_learning_rate, validate_epochs
)
from ...utils.console import create_progress, create_table

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

@handle_errors
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
    model_type: str = typer.Option("modernbert", "--model", "-m", help="Model type"),
    head_type: str = typer.Option("binary_classification", "--head", help="Head type"),
    num_labels: int = typer.Option(2, "--num-labels", help="Number of labels"),
    
    # Training arguments
    epochs: int = typer.Option(3, "--epochs", "-e", help="Number of training epochs",
                              callback=validate_epochs),
    batch_size: int = typer.Option(32, "--batch-size", "-b", help="Training batch size",
                                  callback=validate_batch_size),
    learning_rate: float = typer.Option(2e-5, "--lr", "-l", help="Learning rate",
                                       callback=validate_learning_rate),
    
    # Configuration
    config: Optional[Path] = typer.Option(None, "--config", "-c", help="Configuration file",
                                        callback=lambda p: validate_path(p, must_exist=True, extensions=['.yaml', '.yml', '.json']) if p else None),
    
    # Output arguments
    output_dir: Path = typer.Option("output", "--output", "-o", help="Output directory"),
    experiment_name: Optional[str] = typer.Option("bert_training", "--experiment", help="Experiment name"),
    
    # Advanced options
    seed: int = typer.Option(42, "--seed", "-s", help="Random seed"),
    use_lora: bool = typer.Option(False, "--use-lora", help="Use LoRA adaptation"),
    debug: bool = typer.Option(False, "--debug", "-d", help="Enable debug mode"),
):
    """Train a BERT model for Kaggle competitions (simplified version).
    
    Examples:
        # Basic training
        bert train --train data/train.csv --val data/val.csv
        
        # Training with configuration file
        bert train --train data/train.csv --config configs/quick.yaml
    """
    console = get_console()
    
    # Show training configuration header
    console.print("\n[bold blue]BERT Training System[/bold blue]")
    console.print("=" * 60)
    
    # Create output directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = output_dir / f"run_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    
    # Import training components
    try:
        from data import create_dataloader
        from models import create_bert_with_head
        from models.wrapper import ModelWrapper
        from training import create_trainer, BaseTrainerConfig
        from transformers import AutoTokenizer
        import mlx.core as mx
        
    except ImportError as e:
        print_error(
            f"Failed to import training components: {str(e)}\n"
            "Make sure all dependencies are installed.",
            title="Import Error"
        )
        raise typer.Exit(1)
    
    # Load configuration if provided
    if config and config.exists():
        trainer_config = BaseTrainerConfig.load(config)
        print_success(f"Loaded configuration from {config}")
    else:
        # Create config from CLI parameters
        trainer_config = BaseTrainerConfig()
        trainer_config.optimizer.learning_rate = learning_rate
        trainer_config.training.num_epochs = epochs
        trainer_config.data.batch_size = batch_size
        trainer_config.environment.output_dir = run_dir
        trainer_config.environment.experiment_name = experiment_name
        trainer_config.environment.seed = seed
    
    # Display configuration table
    config_table = create_table("Training Configuration", ["Parameter", "Value"])
    config_table.add_row("Model Type", model_type)
    config_table.add_row("Head Type", head_type)
    config_table.add_row("Number of Labels", str(num_labels))
    config_table.add_row("Output Directory", str(run_dir))
    config_table.add_row("Batch Size", str(trainer_config.data.batch_size))
    config_table.add_row("Learning Rate", str(trainer_config.optimizer.learning_rate))
    config_table.add_row("Epochs", str(trainer_config.training.num_epochs))
    config_table.add_row("Use LoRA", "Yes" if use_lora else "No")
    console.print(config_table)
    
    # Create tokenizer
    console.print("\n[yellow]Loading tokenizer...[/yellow]")
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    
    # Create data loaders
    console.print("\n[yellow]Loading data...[/yellow]")
    
    with console.status("[yellow]Creating training data loader...[/yellow]"):
        train_loader = create_dataloader(
            data_path=train_data,
            batch_size=trainer_config.data.batch_size,
            shuffle=True,
            tokenizer=tokenizer,
            max_length=512,
            num_workers=trainer_config.data.num_workers,
            label_column="Survived" if "titanic" in str(train_data).lower() else None,
        )
    
    val_loader = None
    if val_data:
        with console.status("[yellow]Creating validation data loader...[/yellow]"):
            val_loader = create_dataloader(
                data_path=val_data,
                batch_size=trainer_config.data.eval_batch_size or trainer_config.data.batch_size * 2,
                shuffle=False,
                tokenizer=tokenizer,
                max_length=512,
                num_workers=2,
                label_column="Survived" if "titanic" in str(val_data).lower() else None,
            )
    
    # Display dataset info
    console.print(f"[green]✓ Loaded {len(train_loader.dataset)} training samples[/green]")
    if val_loader:
        console.print(f"[green]✓ Loaded {len(val_loader.dataset)} validation samples[/green]")
    
    # Create model
    with console.status("[yellow]Creating model...[/yellow]"):
        base_model = create_bert_with_head(
            model_type=model_type,
            head_type=head_type,
            num_labels=num_labels,
            use_lora=use_lora,
        )
        # Wrap model to handle dictionary inputs
        model = ModelWrapper(base_model)
    
    console.print(f"[green]✓ Created {model_type} model with {head_type} head[/green]")
    
    # Create trainer
    trainer = create_trainer(
        model=model,
        config=trainer_config,
    )
    
    # Save training config
    config_dict = {
        "model_type": model_type,
        "head_type": head_type,
        "num_labels": num_labels,
        "train_path": str(train_data),
        "val_path": str(val_data) if val_data else None,
        "timestamp": timestamp,
        "config": trainer_config.to_dict(),
    }
    
    with open(run_dir / "training_config.json", "w") as f:
        json.dump(config_dict, f, indent=2)
    
    # Start training
    console.print("\n[bold green]Starting training...[/bold green]\n")
    
    try:
        # Train model
        result = trainer.train(train_loader, val_loader)
        
        # Show final results
        print_success("Training completed successfully!")
        
        # Display results
        results_table = create_table("Training Results", ["Metric", "Value"])
        results_table.add_row("Final Train Loss", f"{result.final_train_loss:.4f}")
        if result.final_val_loss:
            results_table.add_row("Final Val Loss", f"{result.final_val_loss:.4f}")
        results_table.add_row("Best Val Loss", f"{result.best_val_loss:.4f}")
        results_table.add_row("Total Epochs", str(result.total_epochs))
        results_table.add_row("Total Time", f"{result.total_time:.1f}s")
        console.print(results_table)
        
        # Show model location
        if result.best_model_path:
            print_info(f"Best model saved to: {result.best_model_path}")
        
        # Show next steps
        console.print("\n[bold]Next steps:[/bold]")
        console.print(f"1. Generate predictions: [cyan]bert predict --test data/test.csv "
                     f"--checkpoint {result.best_model_path or run_dir}[/cyan]")
        console.print("2. View MLflow results: [cyan]bert mlflow ui[/cyan]")
        console.print("3. Submit to Kaggle: [cyan]bert kaggle submit <competition> <predictions>[/cyan]")
        
    except KeyboardInterrupt:
        print_error("Training interrupted by user", title="Interrupted")
        raise typer.Exit(130)
    except Exception as e:
        print_error(f"Training failed: {str(e)}", title="Training Error")
        if debug:
            console.print_exception()
        raise typer.Exit(1)