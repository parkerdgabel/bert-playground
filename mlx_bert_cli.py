#!/usr/bin/env python3
"""Unified CLI for MLX-based ModernBERT training and inference."""

import typer
from pathlib import Path
from typing import Optional
import mlx.core as mx
import mlx.nn as nn
from rich.console import Console
from rich.table import Table
from rich.progress import track
from training.rich_display_manager import RichDisplayManager
import json
import time
from datetime import datetime
import sys
import pandas as pd

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

# Apply MLX compatibility patches for mlx-embeddings
from utils.mlx_patch import apply_mlx_patches
apply_mlx_patches()

from models.factory import create_model
from models.modernbert_cnn_hybrid import create_cnn_hybrid_model
from data import KaggleDataLoader, create_kaggle_dataloader
# Import the unified TitanicClassifier from the main classification.py file
# (not from models/classification/titanic_classifier.py which expects EmbeddingModel)
import importlib.util
spec = importlib.util.spec_from_file_location("classification", "/Users/parkergabel/PycharmProjects/bert-playground/models/classification.py")
classification_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(classification_module)
UnifiedTitanicClassifier = classification_module.TitanicClassifier
from training.mlx_trainer import MLXTrainer
from training.config import TrainingConfig
from utils.mlflow_central import setup_central_mlflow
import mlx.optimizers as optim

app = typer.Typer(
    name="mlx-bert",
    help="MLX-based ModernBERT for Kaggle competitions",
    add_completion=False,
)
console = Console()


@app.command()
def train(
    train_path: Path = typer.Option(..., "--train", "-t", help="Path to training CSV"),
    val_path: Optional[Path] = typer.Option(
        None, "--val", "-v", help="Path to validation CSV"
    ),
    output_dir: Path = typer.Option(
        "./output", "--output", "-o", help="Output directory"
    ),
    model_name: str = typer.Option(
        "answerdotai/ModernBERT-base", "--model", "-m", help="Model name or path"
    ),
    model_type: str = typer.Option(
        "base", "--model-type", help="Model type: base or cnn_hybrid"
    ),
    batch_size: int = typer.Option(32, "--batch-size", "-b", help="Batch size"),
    learning_rate: float = typer.Option(2e-5, "--lr", help="Learning rate"),
    num_epochs: int = typer.Option(5, "--epochs", "-e", help="Number of epochs"),
    max_length: int = typer.Option(256, "--max-length", help="Maximum sequence length"),
    warmup_ratio: float = typer.Option(0.1, "--warmup-ratio", help="Warmup ratio"),
    gradient_accumulation: int = typer.Option(
        1, "--grad-accum", help="Gradient accumulation steps"
    ),
    num_workers: int = typer.Option(
        4, "--workers", "-w", help="Number of data workers"
    ),
    prefetch_size: int = typer.Option(
        4, "--prefetch", help="Data prefetch size"
    ),
    experiment_name: str = typer.Option(
        "mlx_unified", "--experiment", help="MLflow experiment name"
    ),
    run_name: Optional[str] = typer.Option(
        None, "--run-name", help="MLflow run name"
    ),
    disable_mlflow: bool = typer.Option(
        False, "--no-mlflow", help="Disable MLflow tracking"
    ),
    augment: bool = typer.Option(
        True, "--augment/--no-augment", help="Enable data augmentation"
    ),
    enable_dynamic_batching: bool = typer.Option(
        True, "--dynamic-batch/--no-dynamic-batch", help="Enable dynamic batching"
    ),
    max_batch_size: int = typer.Option(
        64, "--max-batch-size", help="Maximum batch size for dynamic batching"
    ),
    early_stopping_patience: int = typer.Option(
        3, "--early-stopping", help="Early stopping patience (0 to disable)"
    ),
    gradient_clip: float = typer.Option(
        1.0, "--grad-clip", help="Gradient clipping value"
    ),
    label_smoothing: float = typer.Option(
        0.0, "--label-smoothing", help="Label smoothing factor"
    ),
    eval_steps: int = typer.Option(
        100, "--eval-steps", help="Evaluation frequency"
    ),
    save_steps: int = typer.Option(
        100, "--save-steps", help="Checkpoint save frequency"
    ),
    resume_from: Optional[str] = typer.Option(
        None, "--resume", help="Resume from checkpoint"
    ),
    config: Optional[Path] = typer.Option(
        None, "--config", "-c", help="Load config from YAML or JSON file"
    ),
    # CNN-specific options
    cnn_kernel_sizes: str = typer.Option(
        "2,3,4,5", "--cnn-kernels", help="CNN kernel sizes (comma-separated)"
    ),
    cnn_num_filters: int = typer.Option(
        128, "--cnn-filters", help="Number of CNN filters"
    ),
    use_dilated_conv: bool = typer.Option(
        True, "--dilated/--no-dilated", help="Use dilated convolutions"
    ),
    # MLX Embeddings options
    use_mlx_embeddings: bool = typer.Option(
        False, "--use-mlx-embeddings", help="Use MLX embeddings backend"
    ),
    tokenizer_backend: str = typer.Option(
        "auto", "--tokenizer-backend", help="Tokenizer backend: auto, mlx, or huggingface"
    ),
):
    """Train ModernBERT model with unified MLX trainer."""
    console.print("\n[bold blue]MLX Unified Training System[/bold blue]")
    console.print("=" * 60)

    # Load config if provided
    config_overrides = {}
    if config and config.exists():
        from utils.config_loader import ConfigLoader
        config_overrides = ConfigLoader.load(config)
        console.print(f"[green]Loaded config from {config}[/green]")

    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = output_dir / f"run_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    # Create placeholder config for data loading
    batch_size_config = config_overrides.get("batch_size", batch_size)
    max_batch_size_config = config_overrides.get("max_batch_size", max_batch_size)
    eval_batch_size = min(batch_size_config * 2, max_batch_size_config)

    # Create data loaders
    console.print("\n[yellow]Loading data...[/yellow]")
    
    # Create MLX-native Kaggle data loader
    train_loader = create_kaggle_dataloader(
        dataset_name="titanic",
        csv_path=str(train_path),
        tokenizer_name=model_name,
        batch_size=batch_size_config,
        max_length=max_length,
        shuffle=True,
        shuffle_buffer_size=1000 if augment else 100,
        prefetch_size=config_overrides.get("prefetch_size", prefetch_size),
        num_workers=config_overrides.get("num_workers", num_workers),
        tokenizer_backend=config_overrides.get("tokenizer_backend", tokenizer_backend),
    )
    
    # Create validation loader if needed
    val_loader = None
    if val_path:
        val_loader = create_kaggle_dataloader(
            dataset_name="titanic",
            csv_path=str(val_path),
            tokenizer_name=model_name,
            batch_size=eval_batch_size,
            max_length=max_length,
            shuffle=False,
            prefetch_size=2,
            num_workers=2,
            tokenizer_backend=config_overrides.get("tokenizer_backend", tokenizer_backend),
        )

    train_samples = len(train_loader) * batch_size_config
    console.print(f"[green]✓ Loaded ~{train_samples} training samples ({len(train_loader)} batches)[/green]")
    if val_loader:
        val_samples = len(val_loader) * eval_batch_size
        console.print(f"[green]✓ Loaded ~{val_samples} validation samples ({len(val_loader)} batches)[/green]")

    # Create unified training config with correct warmup steps
    from training.config import (TrainingConfig, EvaluationConfig, CheckpointConfig, 
                                  MonitoringConfig, MLXOptimizationConfig, AdvancedFeatures)
    
    # If we have a full config from YAML, use it directly
    if config and "memory" in config_overrides:
        # This is a full TrainingConfig in YAML format
        training_config = TrainingConfig.from_dict(config_overrides)
        # Override with CLI parameters if provided
        if run_name:
            training_config.run_name = run_name
        training_config.output_dir = str(run_dir)
        training_config.train_path = str(train_path)
        training_config.val_path = str(val_path) if val_path else None
        
        # Update checkpoint dir
        if hasattr(training_config, 'checkpoint') and training_config.checkpoint:
            training_config.checkpoint.checkpoint_dir = str(run_dir / "checkpoints")
    else:
        # Build config from CLI parameters with overrides
        training_config = TrainingConfig(
            # Basic parameters
            learning_rate=config_overrides.get("learning_rate", learning_rate),
            epochs=config_overrides.get("epochs", num_epochs),
            warmup_steps=config_overrides.get("warmup_steps", 
                int(config_overrides.get("warmup_ratio", warmup_ratio) * len(train_loader) * num_epochs)),
            batch_size=batch_size_config,
            # Data configuration
            train_path=str(train_path),
            val_path=str(val_path) if val_path else None,
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
            ),
            monitoring=MonitoringConfig(
                enable_mlflow=not disable_mlflow,
                experiment_name=config_overrides.get("experiment_name", experiment_name),
                run_name=run_name or f"{model_type}_{timestamp}",
                enable_rich_console=False,  # Disable Rich console to avoid conflicts
            ),
            mlx_optimization=MLXOptimizationConfig(
                gradient_accumulation_steps=config_overrides.get("gradient_accumulation", gradient_accumulation),
                max_grad_norm=config_overrides.get("gradient_clip", gradient_clip),
            ),
            advanced=AdvancedFeatures(
                label_smoothing=config_overrides.get("label_smoothing", label_smoothing),
            ),
        )

    # Display configuration
    config_table = Table(title="Training Configuration")
    config_table.add_column("Parameter", style="cyan")
    config_table.add_column("Value", style="green")

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
            # Create new architecture MLX embeddings model
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
            # Create standard model and wrap with classifier
            bert_model = create_model("standard")
            model = UnifiedTitanicClassifier(bert_model)
            model_desc = "ModernBERT with TitanicClassifier"

    console.print(f"[green]✓ Created {model_desc} model[/green]")

    # Create optimizer
    optimizer = optim.AdamW(
        learning_rate=training_config.learning_rate,
        weight_decay=0.01,  # Default weight decay
    )

    # Create display manager for Rich console output
    display_manager = RichDisplayManager(console=console)

    # Create unified trainer
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
        "train_path": str(train_path),
        "val_path": str(val_path) if val_path else None,
        "timestamp": timestamp,
        "learning_rate": training_config.learning_rate,
        "epochs": training_config.epochs,
        "batch_size": training_config.batch_size,
        "warmup_steps": training_config.warmup_steps,
        "output_dir": training_config.output_dir,
        "gradient_accumulation_steps": training_config.mlx_optimization.gradient_accumulation_steps,
        "max_grad_norm": training_config.mlx_optimization.max_grad_norm,
        "early_stopping_patience": training_config.evaluation.early_stopping_patience,
        "eval_steps": training_config.evaluation.eval_steps,
        "save_steps": training_config.checkpoint.checkpoint_frequency,
        "mlflow_enabled": training_config.monitoring.enable_mlflow,
        "label_smoothing": training_config.advanced.label_smoothing,
    }

    if model_type == "cnn_hybrid":
        full_config.update({
            "cnn_kernel_sizes": kernel_sizes,
            "cnn_num_filters": cnn_num_filters,
            "use_dilated_conv": use_dilated_conv,
        })

    with open(run_dir / "training_config.json", "w") as f:
        json.dump(full_config, f, indent=2, default=str)

    # Train model
    console.print("\n[bold green]Starting training...[/bold green]\n")

    start_time = time.time()
    
    # Use display manager as context manager for proper cleanup
    with display_manager:
        results = trainer.train(
            train_loader=train_loader,
            val_loader=val_loader,
            resume_from_checkpoint=resume_from,
        )
    
    # Training complete
    elapsed_time = time.time() - start_time
    console.print(f"\n[bold green]✓ Training completed in {elapsed_time:.1f}s[/bold green]")
    
    # Display results
    results_table = Table(title="Training Results")
    results_table.add_column("Metric", style="cyan")
    results_table.add_column("Value", style="green")
    
    results_table.add_row("Best Validation Score", f"{results['best_metric']:.4f}")
    results_table.add_row("Best Step", str(results['best_step']))
    results_table.add_row("Total Time", f"{results['total_time']:.1f}s")
    
    if results.get('final_metrics'):
        for metric, value in results['final_metrics'].items():
            results_table.add_row(metric, f"{value:.4f}")
    
    console.print(results_table)
    
    # Save results
    with open(run_dir / "training_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    
    console.print(f"\n[green]Results saved to: {run_dir}[/green]")
    
    if training_config.monitoring.enable_mlflow:
        console.print("\n[cyan]View results with: mlflow ui[/cyan]")


@app.command()
def predict(
    test_path: Path = typer.Option(..., "--test", "-t", help="Path to test CSV"),
    checkpoint: Path = typer.Option(
        ..., "--checkpoint", "-c", help="Path to model checkpoint"
    ),
    output: Path = typer.Option(
        "submission.csv", "--output", "-o", help="Output CSV path"
    ),
    batch_size: int = typer.Option(64, "--batch-size", "-b", help="Batch size"),
    max_length: int = typer.Option(256, "--max-length", help="Maximum sequence length"),
):
    """Generate predictions using trained model."""
    console.print("\n[bold blue]MLX ModernBERT Prediction[/bold blue]")
    console.print("=" * 60)

    # Load model config
    config_path = checkpoint.parent.parent / "training_config.json"
    if config_path.exists():
        with open(config_path) as f:
            train_config = json.load(f)
        model_name = train_config.get("model", "answerdotai/ModernBERT-base")
        model_type = train_config.get("model_type", "base")
    else:
        console.print("[yellow]Warning: No training config found, using defaults[/yellow]")
        model_name = "answerdotai/ModernBERT-base"
        model_type = "base"

    # Create model
    with console.status("[yellow]Loading model...[/yellow]"):
        if model_type == "cnn_hybrid":
            bert_model = create_cnn_hybrid_model(
                model_name=model_name,
                num_labels=2,
                cnn_kernel_sizes=train_config.get("cnn_kernel_sizes", [2, 3, 4, 5]),
                cnn_num_filters=train_config.get("cnn_num_filters", 128),
                use_dilated_conv=train_config.get("use_dilated_conv", True),
            )
            bert_model.config.hidden_size = bert_model.output_hidden_size
            model = UnifiedTitanicClassifier(bert_model)
        else:
            bert_model = create_model("standard")
            model = UnifiedTitanicClassifier(bert_model)
        
        # Load weights
        model.load_pretrained(str(checkpoint))

    console.print(f"[green]✓ Loaded model from {checkpoint}[/green]")

    # Create data loader for test data
    test_loader = create_kaggle_dataloader(
        dataset_name="titanic",
        csv_path=str(test_path),
        tokenizer_name=model_name,
        batch_size=batch_size,
        max_length=max_length,
        shuffle=False,
        prefetch_size=4,
        num_workers=4,
    )

    console.print(f"[green]✓ Loaded {len(test_loader)} test batches[/green]")

    # Generate predictions
    predictions = []
    with console.status("[yellow]Generating predictions...[/yellow]"):
        test_stream = test_loader.create_stream(is_training=False)
        for batch in test_stream:
            # Convert numpy arrays to MLX arrays if needed
            if not isinstance(batch["input_ids"], mx.array):
                batch["input_ids"] = mx.array(batch["input_ids"])
                batch["attention_mask"] = mx.array(batch["attention_mask"])
            
            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
            )
            preds = mx.argmax(outputs["logits"], axis=-1)
            predictions.extend(preds.tolist())

    # Save predictions
    import pandas as pd

    df = pd.read_csv(test_path)
    submission = pd.DataFrame({
        "PassengerId": df["PassengerId"],
        "Survived": predictions[:len(df)]
    })
    submission.to_csv(output, index=False)

    console.print(f"[green]✓ Predictions saved to {output}[/green]")
    console.print(f"[cyan]Total predictions: {len(predictions)}[/cyan]")


@app.command()
def benchmark(
    batch_size: int = typer.Option(32, "--batch-size", "-b", help="Batch size"),
    seq_length: int = typer.Option(256, "--seq-length", "-s", help="Sequence length"),
    steps: int = typer.Option(20, "--steps", "-n", help="Number of steps"),
    model_type: str = typer.Option("base", "--model-type", help="Model type"),
    warmup_steps: int = typer.Option(5, "--warmup", help="Warmup steps"),
):
    """Benchmark model performance."""
    console.print("\n[bold blue]MLX ModernBERT Benchmark[/bold blue]")
    console.print("=" * 60)

    # Create dummy data
    import numpy as np

    dummy_batch = {
        "input_ids": mx.ones((batch_size, seq_length), dtype=mx.int32),
        "attention_mask": mx.ones((batch_size, seq_length), dtype=mx.int32),
        "labels": mx.zeros((batch_size,), dtype=mx.int32),
    }

    # Create model
    with console.status("[yellow]Creating model...[/yellow]"):
        if model_type == "cnn_hybrid":
            bert_model = create_cnn_hybrid_model(
                model_name="answerdotai/ModernBERT-base",
                num_labels=2,
            )
            bert_model.config.hidden_size = bert_model.output_hidden_size
            model = UnifiedTitanicClassifier(bert_model)
        else:
            bert_model = create_model("standard")
            model = UnifiedTitanicClassifier(bert_model)

    # Create optimizer
    optimizer = optim.AdamW(learning_rate=2e-5)

    console.print(f"[green]Model: {model_type}[/green]")
    console.print(f"[green]Batch size: {batch_size}[/green]")
    console.print(f"[green]Sequence length: {seq_length}[/green]")

    # Warmup
    console.print("\n[yellow]Warming up...[/yellow]")
    for _ in range(warmup_steps):
        def loss_fn(model):
            outputs = model(**dummy_batch)
            return outputs["loss"]

        loss, grads = nn.value_and_grad(model, loss_fn)(model)
        optimizer.update(model, grads)
        mx.eval(loss)

    # Benchmark
    console.print("[yellow]Benchmarking...[/yellow]")
    times = []

    for step in range(steps):
        start_time = time.time()

        def loss_fn(model):
            outputs = model(**dummy_batch)
            return outputs["loss"]

        loss, grads = nn.value_and_grad(model, loss_fn)(model)
        optimizer.update(model, grads)
        mx.eval(loss)

        step_time = time.time() - start_time
        times.append(step_time)

        if (step + 1) % 5 == 0:
            console.print(f"Step {step + 1}/{steps}: {step_time:.3f}s")

    # Results
    times = np.array(times)
    console.print("\n[bold green]Benchmark Results[/bold green]")
    console.print(f"Average time per step: {times.mean():.3f}s ± {times.std():.3f}s")
    console.print(f"Throughput: {batch_size / times.mean():.1f} samples/s")
    console.print(f"Min time: {times.min():.3f}s")
    console.print(f"Max time: {times.max():.3f}s")


@app.command()
def info():
    """Display system and MLX information."""
    import platform

    console.print("\n[bold blue]System Information[/bold blue]")
    console.print("=" * 60)

    info_table = Table()
    info_table.add_column("Property", style="cyan")
    info_table.add_column("Value", style="green")

    # System info
    info_table.add_row("Platform", platform.platform())
    info_table.add_row("Python", platform.python_version())
    info_table.add_row("MLX Device", str(mx.default_device()))
    
    # MLX info
    try:
        import mlx
        info_table.add_row("MLX Version", mlx.__version__)
    except:
        info_table.add_row("MLX Version", "Unknown")

    # Memory info
    try:
        import psutil
        mem = psutil.virtual_memory()
        info_table.add_row("System Memory", f"{mem.total / 1e9:.1f} GB")
        info_table.add_row("Available Memory", f"{mem.available / 1e9:.1f} GB")
    except:
        pass

    console.print(info_table)

    # MLflow info
    console.print("\n[bold blue]MLflow Configuration[/bold blue]")
    console.print("=" * 60)
    
    from utils.mlflow_central import mlflow_central
    mlflow_central.initialize()
    
    mlflow_table = Table()
    mlflow_table.add_column("Property", style="cyan")
    mlflow_table.add_column("Value", style="green")
    
    mlflow_table.add_row("Tracking URI", mlflow_central.tracking_uri)
    mlflow_table.add_row("Artifact Root", mlflow_central.artifact_root)
    mlflow_table.add_row("Default Experiment", mlflow_central.DEFAULT_EXPERIMENT)
    
    console.print(mlflow_table)


@app.command()
def export(
    checkpoint: Path = typer.Option(
        ..., "--checkpoint", "-c", help="Path to model checkpoint"
    ),
    output_path: Path = typer.Option(
        ..., "--output", "-o", help="Output path for exported model"
    ),
    format: str = typer.Option(
        "mlx", "--format", "-f", help="Export format (mlx, onnx)"
    ),
):
    """Export model to different formats."""
    console.print(f"\n[bold blue]Exporting model to {format} format[/bold blue]")
    console.print("=" * 60)

    if format == "mlx":
        # MLX native format (already implemented)
        import shutil

        shutil.copytree(checkpoint, output_path, dirs_exist_ok=True)
        console.print(f"[green]✓ Exported to MLX format at {output_path}[/green]")
    else:
        console.print(f"[red]Format {format} not yet supported[/red]")


@app.command()
def mlflow_health():
    """Check MLflow health and configuration."""
    console.print("\n[bold blue]MLflow Health Check[/bold blue]")
    console.print("=" * 60)
    
    from utils.mlflow_health import MLflowHealthChecker
    
    health_checker = MLflowHealthChecker()
    results = health_checker.run_full_check()
    
    # Display results
    for check_name, result in results.items():
        status = "[green]✓ PASS[/green]" if result["status"] == "PASS" else "[red]✗ FAIL[/red]"
        console.print(f"{status} {check_name}: {result['message']}")
        
        if result["status"] == "FAIL" and result.get("suggestions"):
            console.print(f"  [yellow]Suggestions:[/yellow]")
            for suggestion in result["suggestions"]:
                console.print(f"    • {suggestion}")
    
    # Summary
    passed = sum(1 for r in results.values() if r["status"] == "PASS")
    total = len(results)
    
    if passed == total:
        console.print(f"\n[bold green]✓ All {total} checks passed![/bold green]")
    else:
        failed = total - passed
        console.print(f"\n[bold red]✗ {failed} of {total} checks failed[/bold red]")
        console.print("[yellow]Run the suggestions above to fix issues[/yellow]")


@app.command()
def mlflow_test(
    output: Optional[Path] = typer.Option(
        None, "--output", "-o", help="Output file for detailed report"
    ),
    category: Optional[str] = typer.Option(
        None, "--category", "-c", help="Run specific test category (health, unit, integration, performance, configuration)"
    ),
):
    """Run comprehensive MLflow test suite."""
    console.print("\n[bold blue]MLflow Comprehensive Test Suite[/bold blue]")
    console.print("=" * 60)
    
    from tests.mlflow_test_runner import MLflowTestRunner
    
    # Create test runner
    runner = MLflowTestRunner()
    
    if category:
        # Run specific category
        category_map = {
            "health": runner._run_health_check,
            "unit": runner._run_unit_tests,
            "integration": runner._run_integration_tests,
            "performance": runner._run_performance_tests,
            "configuration": runner._run_configuration_tests
        }
        
        if category in category_map:
            console.print(f"\n[bold blue]Running {category} tests only[/bold blue]")
            result = category_map[category]()
            runner.results[category] = result
            runner._generate_summary_report()
        else:
            console.print(f"[red]Unknown category: {category}[/red]")
            return
    else:
        # Run all tests
        runner.run_all_tests()
    
    # Save report if requested
    if output:
        runner.save_report(str(output))
    else:
        runner.save_report("mlflow_test_report.json")
    
    # Check overall status
    overall_status = all(r.get("status") == "PASS" for r in runner.results.values())
    if not overall_status:
        console.print("\n[bold red]Some tests failed. Please address the issues above.[/bold red]")
        raise typer.Exit(1)
    else:
        console.print("\n[bold green]All tests passed! MLflow is ready for training.[/bold green]")


@app.command()
def mlflow_dashboard(
    refresh_interval: float = typer.Option(
        5.0, "--refresh", "-r", help="Dashboard refresh interval in seconds"
    )
):
    """Launch MLflow real-time dashboard."""
    console.print("\n[bold blue]Starting MLflow Dashboard[/bold blue]")
    console.print("=" * 60)
    
    from utils.mlflow_dashboard import MLflowDashboard
    
    # Create and start dashboard
    dashboard = MLflowDashboard(refresh_interval=refresh_interval)
    
    try:
        dashboard.start()
    except KeyboardInterrupt:
        dashboard.stop()
        console.print("\n[yellow]Dashboard stopped[/yellow]")


@app.command()
def list_experiments():
    """List all MLflow experiments."""
    console.print("\n[bold blue]MLflow Experiments[/bold blue]")
    console.print("=" * 60)
    
    from utils.mlflow_central import mlflow_central
    mlflow_central.initialize()
    
    experiments = mlflow_central.list_experiments()
    
    if not experiments:
        console.print("[yellow]No experiments found[/yellow]")
        return
    
    exp_table = Table()
    exp_table.add_column("ID", style="cyan")
    exp_table.add_column("Name", style="green")
    exp_table.add_column("Artifact Location")
    exp_table.add_column("Lifecycle Stage")
    
    for exp in experiments:
        exp_table.add_row(
            exp.experiment_id,
            exp.name,
            exp.artifact_location or "",
            exp.lifecycle_stage
        )
    
    console.print(exp_table)


@app.command()
def mlflow_clean(
    all: bool = typer.Option(False, "--all", help="Remove all MLflow data"),
    experiment: Optional[str] = typer.Option(None, "--experiment", "-e", help="Remove specific experiment by name"),
    run_id: Optional[str] = typer.Option(None, "--run", "-r", help="Remove specific run by ID"),
    failed_only: bool = typer.Option(False, "--failed-only", help="Remove only failed runs"),
    before: Optional[str] = typer.Option(None, "--before", help="Remove runs before date (YYYY-MM-DD)"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Preview what will be deleted without deleting"),
    force: bool = typer.Option(False, "--force", "-f", help="Skip confirmation prompts"),
):
    """Clean up MLflow experiments and runs."""
    import mlflow
    from mlflow.entities import ViewType
    from datetime import datetime as dt
    import shutil
    
    console.print("\n[bold blue]MLflow Cleanup Tool[/bold blue]")
    console.print("=" * 60)
    
    # Initialize MLflow
    from utils.mlflow_central import mlflow_central
    mlflow_central.initialize()
    
    items_to_delete = []
    
    # Handle --all option
    if all:
        if not force and not dry_run:
            confirm = typer.confirm("⚠️  This will delete ALL MLflow data. Are you sure?")
            if not confirm:
                console.print("[yellow]Operation cancelled[/yellow]")
                return
        
        # Get all experiments except default
        experiments = mlflow.search_experiments()
        for exp in experiments:
            if exp.name != "Default":
                items_to_delete.append(("experiment", exp.experiment_id, exp.name))
        
        # Also delete all runs from default experiment
        runs = mlflow.search_runs(experiment_ids=["0"], run_view_type=ViewType.ALL)
        for _, run in runs.iterrows():
            items_to_delete.append(("run", run.run_id, f"Run from Default experiment"))
    
    # Handle specific experiment
    elif experiment:
        exp = mlflow.get_experiment_by_name(experiment)
        if not exp:
            console.print(f"[red]Experiment '{experiment}' not found[/red]")
            return
        items_to_delete.append(("experiment", exp.experiment_id, exp.name))
    
    # Handle specific run
    elif run_id:
        try:
            run = mlflow.get_run(run_id)
            items_to_delete.append(("run", run_id, f"Run {run.info.run_name}"))
        except Exception:
            console.print(f"[red]Run '{run_id}' not found[/red]")
            return
    
    # Handle filtered deletions
    else:
        # Get all runs
        runs = mlflow.search_runs(experiment_ids=None, run_view_type=ViewType.ALL)
        
        for _, run in runs.iterrows():
            should_delete = False
            
            # Filter by status
            if failed_only and run.status != "FAILED":
                continue
                
            # Filter by date
            if before:
                try:
                    cutoff_date = dt.strptime(before, "%Y-%m-%d")
                    run_date = dt.fromtimestamp(run.start_time / 1000)
                    if run_date >= cutoff_date:
                        continue
                    should_delete = True
                except ValueError:
                    console.print(f"[red]Invalid date format. Use YYYY-MM-DD[/red]")
                    return
            
            if failed_only or before:
                should_delete = True
                
            if should_delete:
                items_to_delete.append(("run", run.run_id, f"Run {run.get('tags.mlflow.runName', run.run_id)}"))
    
    # Display what will be deleted
    if not items_to_delete:
        console.print("[yellow]No items match the criteria[/yellow]")
        return
    
    console.print(f"\n[yellow]Items to delete: {len(items_to_delete)}[/yellow]")
    
    delete_table = Table()
    delete_table.add_column("Type", style="cyan")
    delete_table.add_column("ID", style="red")
    delete_table.add_column("Name", style="yellow")
    
    for item_type, item_id, item_name in items_to_delete[:10]:  # Show first 10
        delete_table.add_row(item_type.capitalize(), item_id, item_name)
    
    if len(items_to_delete) > 10:
        delete_table.add_row("...", f"... and {len(items_to_delete) - 10} more", "...")
    
    console.print(delete_table)
    
    if dry_run:
        console.print("\n[blue]Dry run mode - no changes made[/blue]")
        return
    
    # Confirm deletion
    if not force:
        confirm = typer.confirm(f"\nDelete {len(items_to_delete)} items?")
        if not confirm:
            console.print("[yellow]Operation cancelled[/yellow]")
            return
    
    # Perform deletion
    deleted_count = 0
    failed_count = 0
    
    with console.status("[bold green]Deleting items...") as status:
        for item_type, item_id, item_name in items_to_delete:
            try:
                if item_type == "experiment":
                    mlflow.delete_experiment(item_id)
                else:  # run
                    mlflow.delete_run(item_id)
                deleted_count += 1
                status.update(f"[bold green]Deleted {deleted_count}/{len(items_to_delete)} items...")
            except Exception as e:
                failed_count += 1
                console.print(f"[red]Failed to delete {item_type} {item_id}: {e}[/red]")
    
    # Summary
    console.print(f"\n[green]✓ Deleted {deleted_count} items[/green]")
    if failed_count > 0:
        console.print(f"[red]✗ Failed to delete {failed_count} items[/red]")


@app.command()
def mlflow_list(
    experiments: bool = typer.Option(False, "--experiments", "-e", help="List all experiments"),
    runs: Optional[str] = typer.Option(None, "--runs", "-r", help="List runs for specific experiment"),
    failed: bool = typer.Option(False, "--failed", help="Show only failed runs"),
    limit: int = typer.Option(20, "--limit", "-l", help="Maximum number of items to show"),
):
    """List MLflow experiments and runs."""
    import mlflow
    from mlflow.entities import ViewType
    
    console.print("\n[bold blue]MLflow Data Browser[/bold blue]")
    console.print("=" * 60)
    
    # Initialize MLflow
    from utils.mlflow_central import mlflow_central
    mlflow_central.initialize()
    
    if experiments or (not runs):
        # List experiments
        exps = mlflow.search_experiments()
        
        exp_table = Table(title="MLflow Experiments")
        exp_table.add_column("ID", style="cyan")
        exp_table.add_column("Name", style="green")
        exp_table.add_column("Total Runs", style="yellow")
        exp_table.add_column("Active Runs", style="blue")
        exp_table.add_column("Created", style="magenta")
        
        for exp in exps:
            # Count runs
            all_runs = mlflow.search_runs(
                experiment_ids=[exp.experiment_id],
                run_view_type=ViewType.ALL
            )
            active_runs = mlflow.search_runs(
                experiment_ids=[exp.experiment_id],
                run_view_type=ViewType.ACTIVE_ONLY
            )
            
            created_date = datetime.fromtimestamp(exp.creation_time / 1000).strftime("%Y-%m-%d %H:%M")
            
            exp_table.add_row(
                exp.experiment_id,
                exp.name,
                str(len(all_runs)),
                str(len(active_runs)),
                created_date
            )
        
        console.print(exp_table)
    
    if runs:
        # List runs for specific experiment
        exp = mlflow.get_experiment_by_name(runs)
        if not exp:
            console.print(f"[red]Experiment '{runs}' not found[/red]")
            return
        
        run_list = mlflow.search_runs(
            experiment_ids=[exp.experiment_id],
            run_view_type=ViewType.ALL,
            max_results=limit
        )
        
        if failed:
            run_list = run_list[run_list['status'] == 'FAILED']
        
        if run_list.empty:
            console.print(f"[yellow]No runs found for experiment '{runs}'[/yellow]")
            return
        
        run_table = Table(title=f"Runs for experiment '{runs}'")
        run_table.add_column("Run ID", style="cyan", max_width=16)
        run_table.add_column("Name", style="green", max_width=20)
        run_table.add_column("Status", style="yellow")
        run_table.add_column("Start Time", style="magenta")
        run_table.add_column("Duration", style="blue")
        run_table.add_column("Metrics", style="white")
        
        for _, run in run_list.iterrows():
            run_id = run['run_id'][:8] + "..."
            run_name = run.get('tags.mlflow.runName', 'N/A')
            status = run['status']
            
            # Format start time
            if isinstance(run['start_time'], pd.Timestamp):
                start_time = run['start_time'].strftime("%Y-%m-%d %H:%M")
            else:
                start_time = datetime.fromtimestamp(run['start_time'] / 1000).strftime("%Y-%m-%d %H:%M")
            
            # Calculate duration
            if pd.notna(run.get('end_time')):
                if isinstance(run['end_time'], pd.Timestamp) and isinstance(run['start_time'], pd.Timestamp):
                    duration_sec = (run['end_time'] - run['start_time']).total_seconds()
                else:
                    duration_sec = (run['end_time'] - run['start_time']) / 1000
                duration = f"{duration_sec / 60:.1f}m"
            else:
                duration = "Running"
            
            # Get key metrics
            metrics = []
            if 'metrics.val_accuracy' in run and pd.notna(run['metrics.val_accuracy']):
                metrics.append(f"acc={run['metrics.val_accuracy']:.3f}")
            if 'metrics.val_loss' in run and pd.notna(run['metrics.val_loss']):
                metrics.append(f"loss={run['metrics.val_loss']:.3f}")
            
            # Color code status
            if status == "FINISHED":
                status = f"[green]{status}[/green]"
            elif status == "FAILED":
                status = f"[red]{status}[/red]"
            elif status == "RUNNING":
                status = f"[blue]{status}[/blue]"
            
            run_table.add_row(
                run_id,
                run_name[:20] + "..." if len(run_name) > 20 else run_name,
                status,
                start_time,
                duration,
                ", ".join(metrics) if metrics else "N/A"
            )
        
        console.print(run_table)
        
        if len(run_list) == limit:
            console.print(f"\n[dim]Showing first {limit} runs. Use --limit to see more.[/dim]")


@app.command()
def mlflow_restart(
    port: int = typer.Option(8080, "--port", "-p", help="Port for MLflow UI"),
    host: str = typer.Option("0.0.0.0", "--host", help="Host for MLflow UI"),
):
    """Restart MLflow UI with correct configuration."""
    import subprocess
    import signal
    import os
    
    console.print("\n[bold blue]MLflow UI Restart[/bold blue]")
    console.print("=" * 60)
    
    # Kill existing MLflow processes
    with console.status("[bold yellow]Stopping existing MLflow UI...") as status:
        try:
            # Find MLflow UI processes
            result = subprocess.run(
                ["pgrep", "-f", "mlflow ui"],
                capture_output=True,
                text=True
            )
            
            if result.stdout:
                pids = result.stdout.strip().split('\n')
                for pid in pids:
                    try:
                        os.kill(int(pid), signal.SIGTERM)
                    except ProcessLookupError:
                        pass
                console.print("[green]✓ Stopped existing MLflow UI processes[/green]")
            
            # Also kill gunicorn processes
            subprocess.run(["pkill", "-f", "gunicorn.*mlflow"], capture_output=True)
            
        except Exception as e:
            console.print(f"[yellow]Warning: {e}[/yellow]")
    
    # Get absolute paths
    db_path = Path.cwd() / "mlruns" / "mlflow.db"
    artifact_path = Path.cwd() / "mlruns" / "artifacts"
    
    # Ensure paths exist
    db_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_path.mkdir(parents=True, exist_ok=True)
    
    # Start new MLflow UI
    console.print(f"\n[bold green]Starting MLflow UI on http://{host}:{port}[/bold green]")
    console.print(f"Database: {db_path}")
    console.print(f"Artifacts: {artifact_path}")
    
    cmd = [
        "uv", "run", "mlflow", "ui",
        "--backend-store-uri", f"sqlite:///{db_path}",
        "--default-artifact-root", str(artifact_path),
        "--host", host,
        "--port", str(port)
    ]
    
    # Start in background
    log_file = Path("mlflow_ui.log")
    with open(log_file, "w") as log:
        process = subprocess.Popen(
            cmd,
            stdout=log,
            stderr=log,
            start_new_session=True
        )
    
    # Wait a moment and check if it started
    time.sleep(3)
    
    if process.poll() is None:
        console.print(f"\n[green]✓ MLflow UI started successfully![/green]")
        console.print(f"[blue]View at: http://{host}:{port}[/blue]")
        console.print(f"[dim]Logs: {log_file}[/dim]")
        console.print(f"[dim]Process ID: {process.pid}[/dim]")
    else:
        console.print("[red]✗ Failed to start MLflow UI[/red]")
        console.print(f"[yellow]Check logs at: {log_file}[/yellow]")


if __name__ == "__main__":
    app()
