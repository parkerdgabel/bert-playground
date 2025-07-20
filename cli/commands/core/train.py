"""Training command implementation."""

from pathlib import Path
from typing import Optional, List
import typer
from loguru import logger
from datetime import datetime
import json
import sys

from ...utils import (
    get_console, print_success, print_error, print_info,
    handle_errors, track_time, requires_project,
    validate_path, validate_batch_size, validate_learning_rate, validate_epochs
)
from ...utils.console import create_progress, create_table

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

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
    model_type: str = typer.Option("base", "--model-type", help="Model type: base, cnn_hybrid, or mlx_embeddings"),
    model_name: str = typer.Option("answerdotai/ModernBERT-base", "--model", "-m", help="Model name or path"),
    pretrained: Optional[str] = typer.Option(None, "--pretrained", "-p", help="Pretrained model name or path"),
    
    # Training arguments
    epochs: int = typer.Option(5, "--epochs", "-e", help="Number of training epochs",
                              callback=validate_epochs),
    batch_size: int = typer.Option(32, "--batch-size", "-b", help="Training batch size",
                                  callback=validate_batch_size),
    learning_rate: float = typer.Option(2e-5, "--lr", "-l", help="Learning rate",
                                       callback=validate_learning_rate),
    warmup_ratio: float = typer.Option(0.1, "--warmup-ratio", help="Warmup ratio"),
    warmup_steps: Optional[int] = typer.Option(None, "--warmup-steps", "-w", help="Number of warmup steps (overrides ratio)"),
    
    # Configuration
    config: Optional[Path] = typer.Option(None, "--config", "-c", help="Configuration file",
                                        callback=lambda p: validate_path(p, must_exist=True, extensions=['.yaml', '.yml', '.json']) if p else None),
    
    # Output arguments
    output_dir: Path = typer.Option("output", "--output", "-o", help="Output directory"),
    experiment_name: Optional[str] = typer.Option("mlx_unified", "--experiment", help="MLflow experiment name"),
    run_name: Optional[str] = typer.Option(None, "--run-name", help="Name for this training run"),
    
    # MLX-specific arguments
    use_mlx_embeddings: bool = typer.Option(False, "--mlx-embeddings", help="Use MLX embeddings"),
    tokenizer_backend: str = typer.Option("auto", "--tokenizer-backend", 
                                        help="Tokenizer backend: auto, mlx, or huggingface"),
    gradient_accumulation: int = typer.Option(1, "--grad-accum", help="Gradient accumulation steps"),
    mixed_precision: bool = typer.Option(True, "--mixed-precision/--no-mixed-precision", 
                                       help="Use mixed precision"),
    gradient_clip: float = typer.Option(1.0, "--grad-clip", help="Gradient clipping value"),
    
    # Data loading options
    max_length: int = typer.Option(256, "--max-length", help="Maximum sequence length"),
    workers: int = typer.Option(0, "--workers", help="Number of data loading workers"),
    prefetch_size: int = typer.Option(0, "--prefetch", help="Data prefetch size"),
    
    # Training control
    seed: int = typer.Option(42, "--seed", "-s", help="Random seed"),
    resume: Optional[Path] = typer.Option(None, "--resume", "-r", help="Resume from checkpoint"),
    early_stopping_patience: int = typer.Option(3, "--early-stopping", 
                                              help="Early stopping patience (0 to disable)"),
    save_steps: int = typer.Option(100, "--save-steps", help="Checkpoint save frequency"),
    eval_steps: int = typer.Option(100, "--eval-steps", help="Evaluation frequency"),
    save_best_only: bool = typer.Option(False, "--save-best-only", help="Save only the best model"),
    
    # Advanced options
    augment: bool = typer.Option(True, "--augment/--no-augment", help="Enable data augmentation"),
    label_smoothing: float = typer.Option(0.0, "--label-smoothing", help="Label smoothing factor"),
    enable_dynamic_batching: bool = typer.Option(True, "--dynamic-batch/--no-dynamic-batch", 
                                               help="Enable dynamic batching"),
    max_batch_size: int = typer.Option(64, "--max-batch-size", 
                                     help="Maximum batch size for dynamic batching"),
    disable_mlflow: bool = typer.Option(False, "--no-mlflow", help="Disable MLflow tracking"),
    
    # CNN-specific options
    cnn_kernel_sizes: str = typer.Option("2,3,4,5", "--cnn-kernels", 
                                       help="CNN kernel sizes (comma-separated)"),
    cnn_num_filters: int = typer.Option(128, "--cnn-filters", help="Number of CNN filters"),
    use_dilated_conv: bool = typer.Option(True, "--dilated/--no-dilated", help="Use dilated convolutions"),
    
    # Debug options
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
    
    # Apply MLX compatibility patches
    from utils.mlx_patch import apply_mlx_patches
    apply_mlx_patches()
    
    # Show training configuration header
    console.print("\n[bold blue]MLX Unified Training System[/bold blue]")
    console.print("=" * 60)
    
    # Load configuration if provided
    config_overrides = {}
    if config and config.exists():
        from utils.config_loader import ConfigLoader
        config_overrides = ConfigLoader.load(config)
        print_success(f"Loaded configuration from {config}")
    
    # Create output directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = output_dir / f"run_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    
    # Import training components
    try:
        from data.factory import create_dataloader, create_dataset
        from models.factory import create_model
        from training.core.base import BaseTrainer
        from training.core.config import get_quick_test_config
        from transformers import AutoTokenizer
        
    except ImportError as e:
        print_error(
            f"Failed to import training components: {str(e)}\n"
            "Make sure all dependencies are installed.",
            title="Import Error"
        )
        raise typer.Exit(1)
    
    # Process config overrides
    batch_size_config = config_overrides.get("batch_size", batch_size)
    max_batch_size_config = config_overrides.get("max_batch_size", max_batch_size)
    eval_batch_size = min(batch_size_config * 2, max_batch_size_config)
    
    # Create data loaders
    console.print("\n[yellow]Loading data...[/yellow]")
    
    # Load tokenizer
    console.print("[yellow]Loading tokenizer...[/yellow]")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Create training data loader
    train_loader = create_dataloader(
        data_path=train_data,
        batch_size=batch_size_config,
        shuffle=True,
        num_workers=workers,
        prefetch_size=prefetch_size,
        tokenizer=tokenizer,
        split="train"
    )
    
    # Create validation loader if provided
    val_loader = None
    if val_data:
        val_loader = create_dataloader(
            data_path=val_data,
            batch_size=eval_batch_size,
            shuffle=False,
            num_workers=2,
            prefetch_size=2,
            tokenizer=tokenizer,
            split="val"
        )
    
    # Display dataset info
    train_samples = len(train_loader) * batch_size_config
    console.print(f"[green]✓ Loaded ~{train_samples} training samples ({len(train_loader)} batches)[/green]")
    if val_loader:
        val_samples = len(val_loader) * eval_batch_size
        console.print(f"[green]✓ Loaded ~{val_samples} validation samples ({len(val_loader)} batches)[/green]")
    
    # Build training configuration
    if config and "memory" in config_overrides:
        # Full config from YAML
        training_config = TrainingConfig.from_dict(config_overrides)
        if run_name:
            training_config.run_name = run_name
        training_config.output_dir = str(run_dir)
        training_config.train_path = str(train_data)
        training_config.val_path = str(val_data) if val_data else None
        
        if hasattr(training_config, 'checkpoint') and training_config.checkpoint:
            training_config.checkpoint.checkpoint_dir = str(run_dir / "checkpoints")
    else:
        # Build from CLI parameters
        warmup_steps_final = warmup_steps
        if warmup_steps_final is None:
            warmup_steps_final = int(warmup_ratio * len(train_loader) * epochs)
        
        training_config = TrainingConfig(
            # Basic parameters
            learning_rate=config_overrides.get("learning_rate", learning_rate),
            epochs=config_overrides.get("epochs", epochs),
            warmup_steps=config_overrides.get("warmup_steps", warmup_steps_final),
            batch_size=batch_size_config,
            # Data configuration
            train_path=str(train_data),
            val_path=str(val_data) if val_data else None,
            # Output
            output_dir=str(run_dir),
            experiment_name=config_overrides.get("experiment_name", experiment_name),
            run_name=run_name or f"{model_type}_{timestamp}",
            # Sub-configurations
            evaluation=EvaluationConfig(
                early_stopping_patience=config_overrides.get("early_stopping_patience", early_stopping_patience),
                eval_steps=config_overrides.get("eval_steps", eval_steps),
            ),
            checkpoint=CheckpointConfig(
                checkpoint_dir=str(run_dir / "checkpoints"),
                checkpoint_frequency=config_overrides.get("save_steps", save_steps),
                save_best_only=save_best_only,
            ),
            monitoring=MonitoringConfig(
                enable_mlflow=not disable_mlflow,
                experiment_name=config_overrides.get("experiment_name", experiment_name),
                run_name=run_name or f"{model_type}_{timestamp}",
                enable_rich_console=False,  # We manage our own console
            ),
            mlx_optimization=MLXOptimizationConfig(
                gradient_accumulation_steps=config_overrides.get("gradient_accumulation", gradient_accumulation),
                max_grad_norm=config_overrides.get("gradient_clip", gradient_clip),
            ),
            advanced=AdvancedFeatures(
                label_smoothing=config_overrides.get("label_smoothing", label_smoothing),
            ),
        )
    
    # Display configuration table
    config_table = create_table("Training Configuration", ["Parameter", "Value"])
    config_table.add_row("Model", model_name)
    config_table.add_row("Model Type", model_type)
    config_table.add_row("Output Directory", str(run_dir))
    config_table.add_row("MLX Embeddings", "Enabled" if use_mlx_embeddings else "Disabled")
    config_table.add_row("Tokenizer Backend", tokenizer_backend)
    config_table.add_row("Batch Size", str(batch_size_config))
    config_table.add_row("Learning Rate", str(training_config.learning_rate))
    config_table.add_row("Epochs", str(training_config.epochs))
    config_table.add_row("Gradient Accumulation", str(training_config.mlx_optimization.gradient_accumulation_steps))
    config_table.add_row("MLflow", "Enabled" if training_config.monitoring.enable_mlflow else "Disabled")
    config_table.add_row("Early Stopping", str(training_config.evaluation.early_stopping_patience))
    console.print(config_table)
    
    # Create model
    with console.status("[yellow]Creating model...[/yellow]"):
        if use_mlx_embeddings:
            # Create MLX embeddings model
            try:
                from models.classification import create_titanic_classifier
                model = create_titanic_classifier(
                    model_name=model_name,
                    dropout_prob=0.1,
                    use_layer_norm=False,
                    activation="relu",
                )
                model_desc = "New Architecture MLX Embeddings ModernBERT"
            except ImportError:
                # Fall back to old architecture
                from embeddings.model_wrapper import MLXEmbeddingModel
                bert_model = MLXEmbeddingModel(
                    model_name=model_name,
                    num_labels=2,
                    use_mlx_embeddings=True,
                )
                model_desc = "Legacy MLX Embeddings ModernBERT"
                model = bert_model
        elif model_type == "cnn_hybrid":
            # Parse CNN kernel sizes
            kernel_sizes = [int(k.strip()) for k in cnn_kernel_sizes.split(",")]
            
            # Create CNN-hybrid model
            bert_model = create_cnn_hybrid_model(
                model_name=model_name,
                num_labels=2,
                cnn_kernel_sizes=kernel_sizes,
                cnn_num_filters=cnn_num_filters,
                use_dilated_conv=use_dilated_conv,
                use_attention_fusion=True,
                use_highway=True,
            )
            model_desc = "CNN-Enhanced ModernBERT"
            
            # For CNN model, override config hidden_size
            bert_model.config.hidden_size = bert_model.output_hidden_size
            model = UnifiedTitanicClassifier(bert_model)
        else:
            # Create standard model
            bert_model = create_model("standard")
            model = UnifiedTitanicClassifier(bert_model)
            model_desc = "ModernBERT with TitanicClassifier"
    
    console.print(f"[green]✓ Created {model_desc} model[/green]")
    
    # Create optimizer
    optimizer = optim.AdamW(
        learning_rate=training_config.learning_rate,
        weight_decay=0.01,
    )
    
    # Create display manager
    display_manager = RichDisplayManager(console=console)
    
    # Create trainer
    trainer = MLXTrainer(
        model=model,
        config=training_config,
        optimizer=optimizer,
        display_manager=display_manager,
    )
    
    # Save training config
    full_config = {
        "model": model_name,
        "model_type": model_type,
        "use_mlx_embeddings": use_mlx_embeddings,
        "tokenizer_backend": tokenizer_backend,
        "train_path": str(train_data),
        "val_path": str(val_data) if val_data else None,
        "timestamp": timestamp,
        "learning_rate": training_config.learning_rate,
        "epochs": training_config.epochs,
        "batch_size": training_config.batch_size,
        "cnn_kernel_sizes": kernel_sizes if model_type == "cnn_hybrid" else None,
        "cnn_num_filters": cnn_num_filters if model_type == "cnn_hybrid" else None,
        "use_dilated_conv": use_dilated_conv if model_type == "cnn_hybrid" else None,
    }
    
    with open(run_dir / "training_config.json", "w") as f:
        json.dump(full_config, f, indent=2)
    
    # Resume if checkpoint provided
    if resume:
        logger.info(f"Resuming from checkpoint: {resume}")
        trainer.load_checkpoint(resume)
    
    # Start training
    console.print("\n[bold green]Starting training...[/bold green]\n")
    
    try:
        # Train model
        metrics = trainer.train(train_loader, val_loader)
        
        # Show final results
        print_success("Training completed successfully!")
        
        # Save final model
        final_model_path = run_dir / "final_model"
        model.save_pretrained(str(final_model_path))
        print_info(f"Model saved to: {final_model_path}")
        
        # Show next steps
        console.print("\n[bold]Next steps:[/bold]")
        console.print(f"1. Generate predictions: [cyan]bert predict --test data/test.csv "
                     f"--checkpoint {final_model_path}[/cyan]")
        
        if training_config.monitoring.enable_mlflow:
            console.print("2. View MLflow results: [cyan]bert mlflow ui[/cyan]")
        
        console.print("3. Submit to Kaggle: [cyan]bert kaggle submit auto --competition NAME[/cyan]")
        
    except KeyboardInterrupt:
        print_error("Training interrupted by user", title="Interrupted")
        raise typer.Exit(130)
    except Exception as e:
        print_error(f"Training failed: {str(e)}", title="Training Error")
        if debug:
            console.print_exception()
        raise typer.Exit(1)