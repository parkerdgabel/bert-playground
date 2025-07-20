"""Benchmark command implementation."""

import typer
from pathlib import Path
import sys
import time
from typing import Optional, Dict, Any
import numpy as np
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import json

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
    model_type: str = typer.Option("modernbert", "--model-type", help="Model type: bert, modernbert, bert-lora, modernbert-lora"),
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
        
        # Benchmark ModernBERT with profiling
        bert benchmark --model-type modernbert --profile
        
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
        from models.factory import create_model, MODEL_REGISTRY
        
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
        model_types = ["bert-binary", "modernbert-binary", "bert-lora-binary", "modernbert-lora-binary"]
        results = {}
        
        for mt in model_types:
            console.print(f"\n[yellow]Benchmarking {mt} model...[/yellow]")
            results[mt] = _run_single_benchmark(
                mt, dummy_batch, steps, warmup_steps, 
                profile, memory, console
            )
        
        # Display comparison results
        _display_comparison(results, console)
        
        # Export results if requested
        if export_results:
            with open(export_results, "w") as f:
                json.dump(results, f, indent=2)
            print_success(f"Results exported to {export_results}")
            
    else:
        # Run single benchmark
        results = _run_single_benchmark(
            model_type, dummy_batch, steps, warmup_steps,
            profile, memory, console
        )
        
        # Display results
        _display_results(results, console)
        
        # Export results if requested
        if export_results:
            with open(export_results, "w") as f:
                json.dump({model_type: results}, f, indent=2)
            print_success(f"Results exported to {export_results}")


def _run_single_benchmark(
    model_type: str,
    dummy_batch: Dict[str, mx.array],
    steps: int,
    warmup_steps: int,
    profile: bool,
    memory: bool,
    console
) -> Dict[str, Any]:
    """Run benchmark for a single model type."""
    
    # Import here to avoid issues with module loading
    from models.factory import create_model, MODEL_REGISTRY
    
    # Create model based on type
    if model_type in MODEL_REGISTRY:
        model = MODEL_REGISTRY[model_type](num_labels=2)
        model_desc = model_type
    else:
        # Default to modernbert with binary classification
        model = create_model("modernbert_with_head", head_type="binary_classification", num_labels=2)
        model_desc = "ModernBERT Binary"
    
    console.print(f"Created {model_desc} model")
    
    # Create optimizer
    optimizer = optim.AdamW(learning_rate=2e-5)
    
    # Warmup
    console.print(f"Running {warmup_steps} warmup steps...")
    for _ in range(warmup_steps):
        loss = _forward_backward_step(model, dummy_batch, optimizer)
    
    # Benchmark
    console.print(f"Running {steps} benchmark steps...")
    forward_times = []
    backward_times = []
    total_times = []
    losses = []
    memory_usage = []
    
    for step in range(steps):
        step_start = time.time()
        
        # Forward pass
        forward_start = time.time()
        outputs = model(
            input_ids=dummy_batch["input_ids"],
            attention_mask=dummy_batch["attention_mask"],
        )
        
        # Compute loss
        if isinstance(outputs, dict) and "logits" in outputs:
            logits = outputs["logits"]
        elif hasattr(outputs, "logits"):
            logits = outputs.logits
        else:
            logits = outputs
            
        loss = nn.losses.cross_entropy(
            logits,
            dummy_batch["labels"],
            reduction="mean"
        )
        forward_time = time.time() - forward_start
        
        # Backward pass
        backward_start = time.time()
        loss_val, grads = mx.value_and_grad(lambda m: loss)(model)
        optimizer.update(model, grads)
        mx.eval(model.parameters())
        backward_time = time.time() - backward_start
        
        total_time = time.time() - step_start
        
        # Record metrics
        forward_times.append(forward_time)
        backward_times.append(backward_time)
        total_times.append(total_time)
        losses.append(float(loss))
        
        if memory:
            # Simple memory tracking
            memory_usage.append(mx.metal.get_active_memory() / 1024**3)  # GB
        
        # Show progress
        if (step + 1) % 5 == 0:
            console.print(f"Step {step + 1}/{steps} - Loss: {loss:.4f}", end="\r")
    
    # Calculate statistics
    results = {
        "model_type": model_type,
        "batch_size": dummy_batch["input_ids"].shape[0],
        "seq_length": dummy_batch["input_ids"].shape[1],
        "steps": steps,
        "forward_time": {
            "mean": np.mean(forward_times),
            "std": np.std(forward_times),
            "min": np.min(forward_times),
            "max": np.max(forward_times),
        },
        "backward_time": {
            "mean": np.mean(backward_times),
            "std": np.std(backward_times),
            "min": np.min(backward_times),
            "max": np.max(backward_times),
        },
        "total_time": {
            "mean": np.mean(total_times),
            "std": np.std(total_times),
            "min": np.min(total_times),
            "max": np.max(total_times),
        },
        "throughput": {
            "samples_per_second": dummy_batch["input_ids"].shape[0] / np.mean(total_times),
            "tokens_per_second": (dummy_batch["input_ids"].shape[0] * dummy_batch["input_ids"].shape[1]) / np.mean(total_times),
        },
        "loss": {
            "final": losses[-1],
            "mean": np.mean(losses),
        }
    }
    
    if memory:
        results["memory"] = {
            "mean_gb": np.mean(memory_usage),
            "max_gb": np.max(memory_usage),
        }
    
    return results


def _forward_backward_step(model, batch, optimizer):
    """Perform a single forward-backward step."""
    def loss_fn(model):
        outputs = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
        )
        
        if isinstance(outputs, dict) and "logits" in outputs:
            logits = outputs["logits"]
        elif hasattr(outputs, "logits"):
            logits = outputs.logits
        else:
            logits = outputs
            
        return nn.losses.cross_entropy(
            logits,
            batch["labels"],
            reduction="mean"
        )
    
    loss, grads = mx.value_and_grad(loss_fn)(model)
    optimizer.update(model, grads)
    mx.eval(model.parameters())
    
    return loss


def _display_results(results: Dict[str, Any], console):
    """Display benchmark results."""
    console.print("\n[bold green]Benchmark Results[/bold green]")
    console.print("=" * 60)
    
    # Create results table
    table = create_table("Performance Metrics", ["Metric", "Value"])
    
    table.add_row("Model Type", results["model_type"])
    table.add_row("Batch Size", str(results["batch_size"]))
    table.add_row("Sequence Length", str(results["seq_length"]))
    table.add_row("Steps", str(results["steps"]))
    table.add_row("", "")  # Separator
    
    # Timing metrics
    table.add_row("Forward Time (ms)", f"{results['forward_time']['mean']*1000:.2f} ± {results['forward_time']['std']*1000:.2f}")
    table.add_row("Backward Time (ms)", f"{results['backward_time']['mean']*1000:.2f} ± {results['backward_time']['std']*1000:.2f}")
    table.add_row("Total Time (ms)", f"{results['total_time']['mean']*1000:.2f} ± {results['total_time']['std']*1000:.2f}")
    table.add_row("", "")  # Separator
    
    # Throughput metrics
    table.add_row("Samples/Second", f"{results['throughput']['samples_per_second']:.2f}")
    table.add_row("Tokens/Second", f"{results['throughput']['tokens_per_second']:.2f}")
    table.add_row("", "")  # Separator
    
    # Loss
    table.add_row("Final Loss", f"{results['loss']['final']:.4f}")
    
    # Memory if available
    if "memory" in results:
        table.add_row("", "")  # Separator
        table.add_row("Avg Memory (GB)", f"{results['memory']['mean_gb']:.2f}")
        table.add_row("Max Memory (GB)", f"{results['memory']['max_gb']:.2f}")
    
    console.print(table)


def _display_comparison(results: Dict[str, Dict[str, Any]], console):
    """Display comparison of multiple benchmark results."""
    console.print("\n[bold green]Benchmark Comparison[/bold green]")
    console.print("=" * 60)
    
    # Create comparison table
    table = create_table("Model Comparison", 
                        ["Model", "Forward (ms)", "Backward (ms)", "Total (ms)", "Samples/s", "Loss"])
    
    for model_type, result in results.items():
        table.add_row(
            model_type,
            f"{result['forward_time']['mean']*1000:.2f}",
            f"{result['backward_time']['mean']*1000:.2f}",
            f"{result['total_time']['mean']*1000:.2f}",
            f"{result['throughput']['samples_per_second']:.2f}",
            f"{result['loss']['final']:.4f}"
        )
    
    console.print(table)
    
    # Find best performing model
    best_throughput = max(results.items(), key=lambda x: x[1]['throughput']['samples_per_second'])
    console.print(f"\n[bold green]Best throughput:[/bold green] {best_throughput[0]} ({best_throughput[1]['throughput']['samples_per_second']:.2f} samples/s)")
    
    best_speed = min(results.items(), key=lambda x: x[1]['total_time']['mean'])
    console.print(f"[bold green]Fastest per step:[/bold green] {best_speed[0]} ({best_speed[1]['total_time']['mean']*1000:.2f} ms/step)")