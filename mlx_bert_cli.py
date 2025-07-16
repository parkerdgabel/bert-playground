#!/usr/bin/env python3
"""Unified CLI for MLX-based ModernBERT training and inference."""

import typer
from pathlib import Path
from typing import Optional
import mlx.core as mx
from rich.console import Console
from rich.table import Table
from rich.progress import track
import json
import time
from datetime import datetime
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from models.factory import create_model
from models.modernbert_cnn_hybrid import create_cnn_hybrid_model
from data.unified_loader import create_unified_dataloaders, UnifiedTitanicDataPipeline
from training.trainer_v2 import TitanicTrainerV2

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
    warmup_steps: int = typer.Option(100, "--warmup", help="Warmup steps"),
    gradient_accumulation: int = typer.Option(
        1, "--grad-accum", help="Gradient accumulation steps"
    ),
    num_workers: int = typer.Option(
        4, "--workers", "-w", help="Number of data workers"
    ),
    experiment_name: str = typer.Option(
        "mlx_modernbert", "--experiment", help="MLflow experiment name"
    ),
    disable_mlflow: bool = typer.Option(
        False, "--no-mlflow", help="Disable MLflow tracking"
    ),
    augment: bool = typer.Option(
        True, "--augment/--no-augment", help="Enable data augmentation"
    ),
    config: Optional[Path] = typer.Option(
        None, "--config", "-c", help="Load config from JSON file"
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
):
    """Train ModernBERT model on Titanic dataset with MLX optimizations."""
    console.print("\n[bold blue]MLX ModernBERT Training[/bold blue]")
    console.print("=" * 60)

    # Load config if provided
    if config and config.exists():
        with open(config) as f:
            config_dict = json.load(f)
        console.print(f"[green]Loaded config from {config}[/green]")
        # Override CLI args with config values
        locals().update(config_dict)

    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = output_dir / f"run_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    # Display configuration
    config_table = Table(title="Training Configuration")
    config_table.add_column("Parameter", style="cyan")
    config_table.add_column("Value", style="yellow")

    config_items = [
        ("Model", model_name),
        ("Model Type", model_type.upper()),
        ("Train Data", str(train_path)),
        ("Val Data", str(val_path) if val_path else "None"),
        ("Batch Size", str(batch_size)),
        ("Learning Rate", f"{learning_rate:.2e}"),
        ("Epochs", str(num_epochs)),
        ("Max Length", str(max_length)),
        ("Output Dir", str(run_dir)),
        ("MLflow", "Enabled" if not disable_mlflow else "Disabled"),
        ("Augmentation", "Enabled" if augment else "Disabled"),
    ]

    if model_type == "cnn_hybrid":
        config_items.extend(
            [
                ("CNN Kernels", cnn_kernel_sizes),
                ("CNN Filters", str(cnn_num_filters)),
                ("Dilated Conv", "Enabled" if use_dilated_conv else "Disabled"),
            ]
        )

    for key, value in config_items:
        config_table.add_row(key, value)

    console.print(config_table)

    # Initialize MLX
    console.print("\n[yellow]Initializing MLX...[/yellow]")
    device = mx.default_device()
    console.print(f"[green]Using device: {device}[/green]")

    # Create data loaders
    with console.status("[yellow]Loading data...[/yellow]"):
        if model_type == "cnn_hybrid":
            # Use optimized loader for CNN-hybrid model
            train_loader = UnifiedTitanicDataPipeline(
                data_path=str(train_path),
                tokenizer_name=model_name,
                batch_size=batch_size,
                max_length=max_length,
                is_training=True,
                augment=augment,
                num_threads=num_workers,
                prefetch_size=4,
            )

            val_loader = None
            if val_path:
                val_loader = UnifiedTitanicDataPipeline(
                    data_path=str(val_path),
                    tokenizer_name=model_name,
                    batch_size=batch_size,
                    max_length=max_length,
                    is_training=False,
                    augment=False,
                    num_threads=2,
                    prefetch_size=2,
                )
        else:
            # Use standard loader for base model
            train_loader, val_loader = create_unified_dataloaders(
                train_path=str(train_path),
                val_path=str(val_path) if val_path else None,
                tokenizer_name=model_name,
                batch_size=batch_size,
                max_length=max_length,
            )

    console.print(f"[green]✓ Loaded {len(train_loader)} training samples[/green]")
    if val_loader:
        console.print(f"[green]✓ Loaded {len(val_loader)} validation samples[/green]")

    # Create model
    with console.status("[yellow]Creating model...[/yellow]"):
        from models.classification import TitanicClassifier

        if model_type == "cnn_hybrid":
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

            # For CNN model, we need to override the config hidden_size
            # to match the fusion output size
            bert_model.config.hidden_size = bert_model.output_hidden_size
        else:
            # Create base model
            bert_model = create_model("standard")
            model_desc = "OptimizedModernBertMLX"

        model = TitanicClassifier(bert_model)

    console.print(f"[green]✓ Created {model_desc} model[/green]")

    # Create trainer
    trainer = TitanicTrainerV2(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        learning_rate=learning_rate,
        warmup_steps=warmup_steps,
        gradient_accumulation_steps=gradient_accumulation,
        output_dir=str(run_dir),
        experiment_name=experiment_name,
        enable_mlflow=not disable_mlflow,
    )

    # Save config
    training_config = {
        "model": model_name,
        "model_type": model_type,
        "train_path": str(train_path),
        "val_path": str(val_path) if val_path else None,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "num_epochs": num_epochs,
        "max_length": max_length,
        "warmup_steps": warmup_steps,
        "gradient_accumulation": gradient_accumulation,
        "augmentation": augment,
        "timestamp": timestamp,
    }

    if model_type == "cnn_hybrid":
        training_config.update(
            {
                "cnn_kernel_sizes": kernel_sizes,
                "cnn_num_filters": cnn_num_filters,
                "use_dilated_conv": use_dilated_conv,
            }
        )

    with open(run_dir / "training_config.json", "w") as f:
        json.dump(training_config, f, indent=2)

    # Train model
    console.print("\n[bold green]Starting training...[/bold green]")
    start_time = time.time()

    try:
        trainer.train(num_epochs=num_epochs)

        elapsed_time = time.time() - start_time
        console.print(
            f"\n[bold green]✓ Training completed in {elapsed_time:.1f} seconds[/bold green]"
        )

        # Display results
        if trainer.training_history["val_accuracy"]:
            best_val_acc = max(trainer.training_history["val_accuracy"])
            console.print(
                f"[green]Best validation accuracy: {best_val_acc:.4f}[/green]"
            )

        console.print(f"[blue]Model saved to: {run_dir}[/blue]")

    except Exception as e:
        console.print(f"[bold red]Training failed: {e}[/bold red]")
        raise


@app.command()
def predict(
    test_path: Path = typer.Option(..., "--test", "-t", help="Path to test CSV"),
    checkpoint: Path = typer.Option(
        ..., "--checkpoint", "-c", help="Path to model checkpoint"
    ),
    output_path: Path = typer.Option(
        "./submission.csv", "--output", "-o", help="Output submission path"
    ),
    batch_size: int = typer.Option(32, "--batch-size", "-b", help="Batch size"),
    max_length: int = typer.Option(256, "--max-length", help="Maximum sequence length"),
):
    """Generate predictions for Kaggle submission."""
    console.print("\n[bold blue]MLX ModernBERT Prediction[/bold blue]")
    console.print("=" * 60)

    # Load model
    with console.status("[yellow]Loading model...[/yellow]"):
        from models.classification import TitanicClassifier

        bert_model = create_model("standard")
        model = TitanicClassifier(bert_model)

        # Load weights
        weights_path = checkpoint / "model.safetensors"
        if weights_path.exists():
            weights = mx.load(str(weights_path))
            model.load_weights(list(weights.items()))
        else:
            # Try loading from bert subdirectory (old format)
            bert_weights = checkpoint / "bert" / "model_weights.npz"
            if bert_weights.exists():
                import numpy as np

                weights = np.load(bert_weights)
                # Convert numpy weights to MLX
                mlx_weights = {k: mx.array(v) for k, v in weights.items()}
                model.load_weights(list(mlx_weights.items()))

    console.print(f"[green]✓ Loaded model from {checkpoint}[/green]")

    # Create test loader
    with console.status("[yellow]Loading test data...[/yellow]"):
        from data.unified_loader import UnifiedTitanicDataPipeline

        test_loader = UnifiedTitanicDataPipeline(
            data_path=str(test_path),
            tokenizer_name="answerdotai/ModernBERT-base",
            max_length=max_length,
            batch_size=batch_size,
            is_training=False,
            augment=False,
        )

    console.print(f"[green]✓ Loaded {len(test_loader)} test samples[/green]")

    # Generate predictions
    predictions = []
    passenger_ids = []

    model.eval()

    with console.status("[yellow]Generating predictions...[/yellow]"):
        for batch_idx, batch in enumerate(test_loader.get_dataloader()()):
            # Get predictions
            outputs = model(
                input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]
            )

            # Get predicted class (0 or 1)
            batch_preds = mx.argmax(outputs["logits"], axis=-1)
            predictions.extend(batch_preds.tolist())

            # Track passenger IDs
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, len(test_loader))
            passenger_ids.extend(range(start_idx, end_idx))

            if batch_idx >= test_loader.get_num_batches() - 1:
                break

    # Create submission
    import pandas as pd

    submission_df = pd.DataFrame(
        {
            "PassengerId": [892 + i for i in range(len(predictions))],
            "Survived": predictions,
        }
    )

    submission_df.to_csv(output_path, index=False)
    console.print(f"[green]✓ Saved predictions to {output_path}[/green]")
    console.print(f"[blue]Total predictions: {len(predictions)}[/blue]")


@app.command()
def benchmark(
    model_name: str = typer.Option(
        "answerdotai/ModernBERT-base", "--model", "-m", help="Model name"
    ),
    batch_size: int = typer.Option(32, "--batch-size", "-b", help="Batch size"),
    seq_length: int = typer.Option(256, "--seq-length", "-s", help="Sequence length"),
    num_steps: int = typer.Option(
        10, "--steps", "-n", help="Number of steps to benchmark"
    ),
):
    """Benchmark model performance."""
    console.print("\n[bold blue]MLX ModernBERT Benchmark[/bold blue]")
    console.print("=" * 60)

    # Create model
    model = create_model("standard")

    # Create dummy data
    input_ids = mx.random.randint(0, 50000, (batch_size, seq_length))
    attention_mask = mx.ones((batch_size, seq_length))
    labels = mx.random.randint(0, 2, (batch_size,))

    # Warmup
    console.print("[yellow]Warming up...[/yellow]")
    for _ in range(3):
        _ = model(input_ids, attention_mask, labels=labels)
    mx.eval(model.parameters())

    # Benchmark
    console.print(f"\n[yellow]Running {num_steps} steps...[/yellow]")
    times = []

    for step in track(range(num_steps), description="Benchmarking"):
        start = time.time()
        outputs = model(input_ids, attention_mask, labels=labels)
        mx.eval(outputs["loss"])
        elapsed = time.time() - start
        times.append(elapsed)

    # Display results
    avg_time = sum(times) / len(times)
    throughput = batch_size / avg_time

    results_table = Table(title="Benchmark Results")
    results_table.add_column("Metric", style="cyan")
    results_table.add_column("Value", style="yellow")

    results_table.add_row("Model", model_name)
    results_table.add_row("Batch Size", str(batch_size))
    results_table.add_row("Sequence Length", str(seq_length))
    results_table.add_row("Average Time/Step", f"{avg_time:.3f}s")
    results_table.add_row("Throughput", f"{throughput:.1f} samples/s")
    results_table.add_row("Min Time", f"{min(times):.3f}s")
    results_table.add_row("Max Time", f"{max(times):.3f}s")

    console.print("\n")
    console.print(results_table)


@app.command()
def info():
    """Display system and MLX information."""
    import platform

    console.print("\n[bold blue]System Information[/bold blue]")
    console.print("=" * 60)

    info_table = Table()
    info_table.add_column("Component", style="cyan")
    info_table.add_column("Value", style="yellow")

    info_items = [
        ("Platform", platform.platform()),
        ("Python", platform.python_version()),
        ("MLX Device", str(mx.default_device())),
        ("MLX Version", mx.__version__),
    ]

    for key, value in info_items:
        info_table.add_row(key, value)

    console.print(info_table)


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


if __name__ == "__main__":
    app()
