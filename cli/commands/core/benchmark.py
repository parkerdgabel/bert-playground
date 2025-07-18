"""Benchmark command implementation."""

import typer
from pathlib import Path
import sys
import time
import numpy as np
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

from ...utils import (
    get_console, print_success, print_error, print_info,
    handle_errors, track_time,
    validate_batch_size
)
from ...utils.console import create_table

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

@handle_errors
@track_time("Running benchmarks")
def benchmark_command(
    batch_size: int = typer.Option(32, "--batch-size", "-b", help="Batch size for benchmarking",
                                  callback=validate_batch_size),
    seq_length: int = typer.Option(256, "--seq-length", "-s", help="Sequence length"),
    steps: int = typer.Option(20, "--steps", "-n", help="Number of steps to run"),
    model_type: str = typer.Option("base", "--model-type", help="Model type: base, cnn_hybrid, mlx_embeddings"),
    model_name: str = typer.Option("answerdotai/ModernBERT-base", "--model", "-m", help="Model name"),
    warmup_steps: int = typer.Option(5, "--warmup", help="Warmup steps"),
    profile: bool = typer.Option(False, "--profile", "-p", help="Enable detailed profiling"),
    memory: bool = typer.Option(False, "--memory", help="Monitor memory usage"),
    compare: bool = typer.Option(False, "--compare", help="Compare different model types"),
    export_results: Optional[Path] = typer.Option(None, "--export", help="Export results to JSON"),
):
    """Run performance benchmarks.
    
    This command benchmarks model performance with various configurations
    to help optimize training and inference settings.
    
    Examples:
        # Basic benchmark
        bert benchmark --batch-size 64 --steps 100
        
        # Benchmark CNN model with profiling
        bert benchmark --model-type cnn_hybrid --profile
        
        # Compare different models
        bert benchmark --compare --export results.json
        
        # Memory usage analysis
        bert benchmark --memory --batch-size 128
    """
    console = get_console()
    
    console.print("\n[bold blue]MLX ModernBERT Benchmark[/bold blue]")
    console.print("=" * 60)
    
    # Import necessary components
    try:
        from models.factory import create_model
        from models.modernbert_cnn_hybrid import create_cnn_hybrid_model
        
        # Import classifier
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "classification",
            str(Path(__file__).parent.parent.parent.parent / "models" / "classification.py")
        )
        classification_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(classification_module)
        UnifiedTitanicClassifier = classification_module.TitanicClassifier
        
    except ImportError as e:
        print_error(
            f"Failed to import components: {str(e)}\n"
            "Make sure all dependencies are installed.",
            title="Import Error"
        )
        raise typer.Exit(1)
    
    # Create dummy data
    dummy_batch = {
        "input_ids": mx.ones((batch_size, seq_length), dtype=mx.int32),
        "attention_mask": mx.ones((batch_size, seq_length), dtype=mx.int32),
        "labels": mx.zeros((batch_size,), dtype=mx.int32),
    }
    
    # Run benchmarks for different models if compare mode
    if compare:
        model_types = ["base", "cnn_hybrid"]
        results = {}
        
        for mt in model_types:
            console.print(f"\n[yellow]Benchmarking {mt} model...[/yellow]")
            results[mt] = _run_single_benchmark(
                model_type=mt,
                model_name=model_name,
                dummy_batch=dummy_batch,
                steps=steps,
                warmup_steps=warmup_steps,
                profile=profile,
                memory=memory,
                console=console
            )
        
        # Display comparison
        _display_comparison(results, console)
        
        # Export if requested
        if export_results:
            import json
            with open(export_results, "w") as f:
                json.dump(results, f, indent=2)
            print_info(f"Results exported to {export_results}")
    
    else:
        # Run single benchmark
        results = _run_single_benchmark(
            model_type=model_type,
            model_name=model_name,
            dummy_batch=dummy_batch,
            steps=steps,
            warmup_steps=warmup_steps,
            profile=profile,
            memory=memory,
            console=console
        )
        
        # Export if requested
        if export_results:
            import json
            with open(export_results, "w") as f:
                json.dump({model_type: results}, f, indent=2)
            print_info(f"Results exported to {export_results}")


def _run_single_benchmark(
    model_type: str,
    model_name: str,
    dummy_batch: dict,
    steps: int,
    warmup_steps: int,
    profile: bool,
    memory: bool,
    console
) -> dict:
    """Run benchmark for a single model type."""
    
    # Create model
    with console.status(f"[yellow]Creating {model_type} model...[/yellow]"):
        if model_type == "cnn_hybrid":
            bert_model = create_cnn_hybrid_model(
                model_name=model_name,
                num_labels=2,
            )
            bert_model.config.hidden_size = bert_model.output_hidden_size
            model = UnifiedTitanicClassifier(bert_model)
        elif model_type == "mlx_embeddings":
            try:
                from models.classification import create_titanic_classifier
                model = create_titanic_classifier(
                    model_name=model_name,
                    dropout_prob=0.0,
                    use_layer_norm=False,
                    activation="relu",
                )
            except ImportError:
                from embeddings.model_wrapper import MLXEmbeddingModel
                model = MLXEmbeddingModel(
                    model_name=model_name,
                    num_labels=2,
                    use_mlx_embeddings=True,
                )
        else:
            bert_model = create_model("standard")
            model = UnifiedTitanicClassifier(bert_model)
    
    # Create optimizer
    optimizer = optim.AdamW(learning_rate=2e-5)
    
    # Display configuration
    config_table = create_table("Benchmark Configuration", ["Parameter", "Value"])
    config_table.add_row("Model Type", model_type)
    config_table.add_row("Batch Size", str(dummy_batch["input_ids"].shape[0]))
    config_table.add_row("Sequence Length", str(dummy_batch["input_ids"].shape[1]))
    config_table.add_row("Steps", str(steps))
    config_table.add_row("Warmup Steps", str(warmup_steps))
    console.print(config_table)
    
    # Warmup
    console.print("\n[yellow]Warming up...[/yellow]")
    for _ in range(warmup_steps):
        outputs = model(
            input_ids=dummy_batch["input_ids"],
            attention_mask=dummy_batch["attention_mask"],
        )
        loss = outputs["loss"]
        loss_value, grads = mx.value_and_grad(model, loss)(
            dummy_batch["input_ids"],
            dummy_batch["attention_mask"],
            dummy_batch["labels"],
        )
        optimizer.update(model, grads)
        mx.eval(model.parameters())
    
    # Benchmark
    console.print("\n[yellow]Running benchmark...[/yellow]")
    forward_times = []
    backward_times = []
    total_times = []
    memory_usage = []
    
    if profile:
        from utils.memory_profiler import MemoryProfiler
        profiler = MemoryProfiler()
    
    for step in range(steps):
        # Track memory before
        if memory:
            import psutil
            process = psutil.Process()
            mem_before = process.memory_info().rss / 1e9  # GB
        
        # Forward pass
        start_time = time.time()
        outputs = model(
            input_ids=dummy_batch["input_ids"],
            attention_mask=dummy_batch["attention_mask"],
        )
        loss = outputs["loss"]
        mx.eval(loss)
        forward_time = time.time() - start_time
        forward_times.append(forward_time)
        
        # Backward pass
        start_time = time.time()
        loss_value, grads = mx.value_and_grad(model, loss)(
            dummy_batch["input_ids"],
            dummy_batch["attention_mask"],
            dummy_batch["labels"],
        )
        optimizer.update(model, grads)
        mx.eval(model.parameters())
        backward_time = time.time() - start_time
        backward_times.append(backward_time)
        
        total_times.append(forward_time + backward_time)
        
        # Track memory after
        if memory:
            mem_after = process.memory_info().rss / 1e9
            memory_usage.append(mem_after - mem_before)
        
        # Show progress
        if (step + 1) % 5 == 0:
            console.print(f"Step {step + 1}/{steps} - "
                         f"Forward: {forward_time:.3f}s, "
                         f"Backward: {backward_time:.3f}s", end="\r")
    
    # Calculate statistics
    forward_times = np.array(forward_times)
    backward_times = np.array(backward_times)
    total_times = np.array(total_times)
    
    # Calculate throughput
    batch_size = dummy_batch["input_ids"].shape[0]
    samples_per_second = batch_size / total_times.mean()
    
    # Display results
    results_table = create_table(f"{model_type.upper()} Model Results", ["Metric", "Value"])
    results_table.add_row("Forward (mean)", f"{forward_times.mean():.3f}s")
    results_table.add_row("Forward (std)", f"{forward_times.std():.3f}s")
    results_table.add_row("Backward (mean)", f"{backward_times.mean():.3f}s")
    results_table.add_row("Backward (std)", f"{backward_times.std():.3f}s")
    results_table.add_row("Total (mean)", f"{total_times.mean():.3f}s")
    results_table.add_row("Min time", f"{total_times.min():.3f}s")
    results_table.add_row("Max time", f"{total_times.max():.3f}s")
    results_table.add_row("Throughput", f"{samples_per_second:.1f} samples/s")
    
    if memory and memory_usage:
        memory_usage = np.array(memory_usage)
        results_table.add_row("Memory (mean)", f"{memory_usage.mean():.2f} GB")
        results_table.add_row("Memory (max)", f"{memory_usage.max():.2f} GB")
    
    console.print("\n")
    console.print(results_table)
    
    # Return results for comparison/export
    results = {
        "forward_mean": float(forward_times.mean()),
        "forward_std": float(forward_times.std()),
        "backward_mean": float(backward_times.mean()),
        "backward_std": float(backward_times.std()),
        "total_mean": float(total_times.mean()),
        "total_min": float(total_times.min()),
        "total_max": float(total_times.max()),
        "throughput": float(samples_per_second),
    }
    
    if memory and memory_usage:
        results["memory_mean"] = float(memory_usage.mean())
        results["memory_max"] = float(memory_usage.max())
    
    return results


def _display_comparison(results: dict, console):
    """Display comparison of different model types."""
    
    console.print("\n[bold blue]Model Comparison[/bold blue]")
    
    # Create comparison table
    comparison_table = create_table("Performance Comparison", 
                                  ["Metric"] + list(results.keys()))
    
    # Add rows for each metric
    metrics = ["forward_mean", "backward_mean", "total_mean", "throughput"]
    metric_names = ["Forward (s)", "Backward (s)", "Total (s)", "Throughput (samples/s)"]
    
    for metric, name in zip(metrics, metric_names):
        row = [name]
        for model_type in results:
            value = results[model_type].get(metric, "N/A")
            if isinstance(value, float):
                if metric == "throughput":
                    row.append(f"{value:.1f}")
                else:
                    row.append(f"{value:.3f}")
            else:
                row.append(str(value))
        comparison_table.add_row(*row)
    
    console.print(comparison_table)
    
    # Find best model
    best_throughput = 0
    best_model = None
    for model_type, res in results.items():
        if res.get("throughput", 0) > best_throughput:
            best_throughput = res["throughput"]
            best_model = model_type
    
    if best_model:
        print_success(f"Best throughput: {best_model} ({best_throughput:.1f} samples/s)")


from typing import Optional