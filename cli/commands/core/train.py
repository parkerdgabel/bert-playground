"""Training command implementation."""

from pathlib import Path
from typing import Optional, List
import typer
from loguru import logger
from datetime import datetime
import json
import sys

from ...utils import (
    handle_errors, track_time, requires_project,
    validate_path, validate_batch_size, validate_learning_rate, validate_epochs
)

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
    model_type: str = typer.Option("base", "--model-type", help="Model type: base or mlx_embeddings"),
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
    
    # Kaggle arguments
    kaggle: bool = typer.Option(False, "--kaggle", help="Use KaggleTrainer for competition optimization"),
    competition: Optional[str] = typer.Option(None, "--competition", help="Kaggle competition name (e.g., titanic)"),
    cv_folds: int = typer.Option(5, "--cv-folds", help="Number of CV folds for KaggleTrainer"),
    
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
    prefetch_size: int = typer.Option(4, "--prefetch", help="Data prefetch size (0 to disable)"),
    use_pretokenized: bool = typer.Option(False, "--pretokenize", help="Pre-tokenize data for optimal performance"),
    
    # Training control
    seed: int = typer.Option(42, "--seed", "-s", help="Random seed"),
    resume: Optional[Path] = typer.Option(None, "--resume", "-r", help="Resume from checkpoint"),
    log_level: str = typer.Option("INFO", "--log-level", help="Logging level (DEBUG, INFO, WARNING, ERROR)"),
    early_stopping_patience: int = typer.Option(3, "--early-stopping", 
                                              help="Early stopping patience (0 to disable)"),
    save_steps: int = typer.Option(100, "--save-steps", help="Checkpoint save frequency"),
    eval_steps: int = typer.Option(100, "--eval-steps", help="Evaluation frequency"),
    logging_steps: int = typer.Option(10, "--logging-steps", help="Progress logging frequency"),
    save_best_only: bool = typer.Option(False, "--save-best-only", help="Save only the best model"),
    
    # Advanced options
    augment: bool = typer.Option(True, "--augment/--no-augment", help="Enable data augmentation"),
    label_smoothing: float = typer.Option(0.0, "--label-smoothing", help="Label smoothing factor"),
    enable_dynamic_batching: bool = typer.Option(True, "--dynamic-batch/--no-dynamic-batch", 
                                               help="Enable dynamic batching"),
    max_batch_size: int = typer.Option(64, "--max-batch-size", 
                                     help="Maximum batch size for dynamic batching"),
    disable_mlflow: bool = typer.Option(False, "--no-mlflow", help="Disable MLflow tracking"),
    use_lora: bool = typer.Option(False, "--use-lora", help="Use LoRA adaptation"),
    
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
    # Apply MLX compatibility patches
    from utils.mlx_patch import apply_mlx_patches
    apply_mlx_patches()
    
    # Configure logging level
    import logging
    from loguru import logger
    log_level_upper = log_level.upper()
    logger.remove()  # Remove default handler
    # Add handler with immediate flushing to avoid buffering issues
    logger.add(sys.stderr, level=log_level_upper, enqueue=False)
    
    # Show training configuration header
    logger.info("\nMLX Unified Training System")
    logger.info("=" * 60)
    
    # Load configuration if provided
    config_overrides = {}
    if config and config.exists():
        from utils.config_loader import ConfigLoader
        config_overrides = ConfigLoader.load(config)
        logger.info(f"✓ Loaded configuration from {config}")
    
    # Create output directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = output_dir / f"run_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    
    # Import training components
    try:
        from data.factory import create_dataloader, create_dataset
        from models.factory import create_model
        from training.core.base import BaseTrainer
        from training.core.config import get_quick_test_config, BaseTrainerConfig
        from training.kaggle.trainer import KaggleTrainer
        from training.kaggle.config import KaggleTrainerConfig, get_competition_config, CompetitionProfile, CompetitionType, KaggleConfig
        from transformers import AutoTokenizer
        
    except ImportError as e:
        logger.error(
            f"Failed to import training components: {str(e)}\n"
            "Make sure all dependencies are installed."
        )
        raise typer.Exit(1)
    
    # Process config overrides
    batch_size_config = config_overrides.get("batch_size", batch_size)
    max_batch_size_config = config_overrides.get("max_batch_size", max_batch_size)
    eval_batch_size = min(batch_size_config * 2, max_batch_size_config)
    
    # Create data loaders
    logger.info("Loading data...")
    
    # Load tokenizer
    logger.info("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Get MLX-specific parameters from config if available
    mlx_prefetch_size = config_overrides.get("data", {}).get("mlx_prefetch_size", None)
    mlx_tokenizer_chunk_size = config_overrides.get("data", {}).get("mlx_tokenizer_chunk_size", 100)
    use_pretokenized_config = config_overrides.get("data", {}).get("use_pretokenized", use_pretokenized)
    
    # Check if compilation is enabled and disable prefetch if so
    # This prevents a deadlock between compiled functions and prefetch threads
    use_compilation = config_overrides.get("training", {}).get("use_compilation", True)  # Default is True
    if use_compilation and (mlx_prefetch_size is None):
        # If compilation is enabled and prefetch wasn't explicitly set, disable it
        logger.warning("Compilation is enabled - disabling prefetch to prevent deadlock")
        prefetch_size = 0
        mlx_prefetch_size = 0
    
    # Create training data loader
    train_loader = create_dataloader(
        data_path=train_data,
        batch_size=batch_size_config,
        shuffle=True,
        num_workers=workers,
        prefetch_size=prefetch_size,
        mlx_prefetch_size=mlx_prefetch_size,
        mlx_tokenizer_chunk_size=mlx_tokenizer_chunk_size,
        tokenizer=tokenizer,
        split="train",
        use_pretokenized=use_pretokenized_config,
        max_length=max_length
    )
    
    # Create validation loader if provided
    val_loader = None
    if val_data:
        val_loader = create_dataloader(
            data_path=val_data,
            batch_size=eval_batch_size,
            shuffle=False,
            num_workers=2,
            prefetch_size=prefetch_size,
            mlx_prefetch_size=mlx_prefetch_size,
            mlx_tokenizer_chunk_size=mlx_tokenizer_chunk_size,
            tokenizer=tokenizer,
            split="val",
            use_pretokenized=use_pretokenized_config,
            max_length=max_length
        )
    
    # Display dataset info
    train_samples = len(train_loader) * batch_size_config
    logger.info(f"✓ Loaded ~{train_samples} training samples ({len(train_loader)} batches)")
    if val_loader:
        val_samples = len(val_loader) * eval_batch_size
        logger.info(f"✓ Loaded ~{val_samples} validation samples ({len(val_loader)} batches)")
    
    # Build training configuration
    if kaggle:
        # Use KaggleTrainer configuration
        if competition and competition.lower() == "titanic":
            training_config = get_competition_config(CompetitionProfile.TITANIC)
            # Override CV folds from CLI
            training_config.kaggle.cv_folds = cv_folds
        else:
            # Create custom Kaggle config
            training_config = KaggleTrainerConfig(
                kaggle=KaggleConfig(
                    competition_name=competition or "titanic",
                    competition_type=CompetitionType.BINARY_CLASSIFICATION,
                    cv_folds=cv_folds,
                    enable_api=True,
                    auto_submit=False,  # Manual submission for now
                )
            )
        
        # Apply any config overrides
        if config and config_overrides:
            # Merge with YAML config
            for key, value in config_overrides.items():
                if hasattr(training_config, key) and key != 'kaggle':
                    setattr(training_config, key, value)
        
        training_config.environment.run_name = run_name or f"kaggle_{competition}_{timestamp}"
        training_config.environment.output_dir = run_dir
        
        # Override training parameters with CLI values
        training_config.training.num_epochs = epochs
        training_config.training.gradient_accumulation_steps = gradient_accumulation
        training_config.data.batch_size = batch_size
        training_config.optimizer.learning_rate = learning_rate
    elif config and config_overrides:
        # Full config from YAML - use BaseTrainerConfig
        training_config = BaseTrainerConfig.from_dict(config_overrides)
        training_config.environment.run_name = run_name or f"{model_type}_{timestamp}"
        training_config.environment.output_dir = run_dir
    else:
        # Build minimal config from CLI parameters
        training_config = get_quick_test_config()
        
        # Override with CLI values
        training_config.optimizer.learning_rate = learning_rate
        training_config.training.num_epochs = epochs
        training_config.data.batch_size = batch_size_config
        training_config.data.eval_batch_size = eval_batch_size
        training_config.training.eval_steps = eval_steps
        training_config.training.logging_steps = logging_steps
        training_config.training.save_steps = save_steps
        training_config.training.early_stopping_patience = early_stopping_patience
        training_config.training.gradient_accumulation_steps = gradient_accumulation
        training_config.training.mixed_precision = mixed_precision
        training_config.optimizer.max_grad_norm = gradient_clip
        training_config.training.label_smoothing = label_smoothing
        
        # Set environment
        training_config.environment.output_dir = run_dir
        training_config.environment.experiment_name = experiment_name
        training_config.environment.run_name = run_name or f"{model_type}_{timestamp}"
        training_config.environment.seed = seed
        
        # Disable MLflow if requested
        if disable_mlflow:
            training_config.training.report_to = []
    
    # Display configuration
    logger.info("\nTraining Configuration:")
    logger.info(f"  Model: {model_name}")
    logger.info(f"  Model Type: {model_type}")
    logger.info(f"  Output Directory: {run_dir}")
    logger.info(f"  MLX Embeddings: {'Enabled' if use_mlx_embeddings else 'Disabled'}")
    logger.info(f"  Tokenizer Backend: {tokenizer_backend}")
    logger.info(f"  Use LoRA: {'Enabled' if use_lora else 'Disabled'}")
    logger.info(f"  Batch Size: {batch_size_config}")
    logger.info(f"  Learning Rate: {training_config.optimizer.learning_rate}")
    logger.info(f"  Epochs: {training_config.training.num_epochs}")
    logger.info(f"  Gradient Accumulation: {training_config.training.gradient_accumulation_steps}")
    logger.info(f"  MLflow: {'Enabled' if 'mlflow' in training_config.training.report_to else 'Disabled'}")
    logger.info(f"  Early Stopping: {training_config.training.early_stopping_patience}\n")
    
    # Create model
    logger.info("Creating model...")
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
    else:
        # Create standard model
        if use_lora:
            # Create model with LoRA
            from models.factory import create_bert_with_lora
            bert_model, lora_adapter = create_bert_with_lora(
                head_type="binary_classification",
                num_labels=2,
                lora_preset="balanced",
            )
            model = bert_model
            model_desc = "ModernBERT with LoRA adaptation"
        else:
            bert_model = create_model(
                "modernbert_with_head",
                head_type="binary_classification",
                num_labels=2
            )
            model = bert_model
            model_desc = "ModernBERT with TitanicClassifier"
    
    logger.info(f"✓ Created {model_desc} model")
    
    # Create trainer
    if kaggle:
        # Load test data for KaggleTrainer
        test_loader = None
        if test_data:
            test_dataset = create_dataset(
                data_path=test_data,
                tokenizer=tokenizer,
                max_length=max_length,
                split="test"
            )
            test_loader = create_dataloader(
                dataset=test_dataset,
                batch_size=eval_batch_size,
                shuffle=False,
                num_workers=workers,
                prefetch_size=prefetch_size
            )
            logger.info(f"✓ Loaded test data for predictions: {len(test_dataset)} samples")
        
        trainer = KaggleTrainer(
            model=model,
            config=training_config,
            test_dataloader=test_loader,
        )
        logger.info("Using KaggleTrainer with cross-validation")
    else:
        trainer = BaseTrainer(
            model=model,
            config=training_config,
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
        "learning_rate": training_config.optimizer.learning_rate,
        "epochs": training_config.training.num_epochs,
        "batch_size": training_config.data.batch_size,
    }
    
    with open(run_dir / "training_config.json", "w") as f:
        json.dump(full_config, f, indent=2)
    
    # Resume if checkpoint provided
    if resume:
        logger.info(f"Resuming from checkpoint: {resume}")
        trainer.load_checkpoint(resume)
    
    # Start training
    logger.info("\nStarting training...")
    
    try:
        # Train model
        if kaggle:
            logger.info("Starting KaggleTrainer training")
            # KaggleTrainer.train handles CV internally if configured
            result = trainer.train(train_loader, val_loader)
            metrics = result  # Get metrics from training result
            
            # KaggleTrainer automatically generates submission if test_dataloader was provided
            if test_loader and hasattr(trainer, 'test_predictions') and trainer.test_predictions is not None:
                logger.info(f"Test predictions generated. Check {trainer.config.kaggle.submission_dir} for submission file.")
        else:
            logger.info("About to call trainer.train()")
            metrics = trainer.train(train_loader, val_loader)
        
        # Show final results
        logger.info("✓ Training completed successfully!")
        
        # Save final model
        final_model_path = run_dir / "final_model"
        trainer.save_checkpoint(final_model_path)
        logger.info(f"Model saved to: {final_model_path}")
        
        # Show next steps
        logger.info("\nNext steps:")
        logger.info(f"1. Generate predictions: bert predict --test data/test.csv --checkpoint {final_model_path}")
        
        if "mlflow" in training_config.training.report_to:
            logger.info("2. View MLflow results: bert mlflow ui")
        
        logger.info("3. Submit to Kaggle: bert kaggle submit auto --competition NAME")
        
    except KeyboardInterrupt:
        logger.error("Training interrupted by user")
        raise typer.Exit(130)
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        if debug:
            import traceback
            traceback.print_exc()
        raise typer.Exit(1)