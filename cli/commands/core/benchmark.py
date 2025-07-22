"""Benchmark command implementation with config-first approach."""

import json
import sys
import time
from pathlib import Path
from typing import Optional, Dict, Any

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
import typer
from loguru import logger
from rich.console import Console
from rich.table import Table

from ...utils import (
    handle_errors,
    track_time,
    print_error,
    print_success,
)
from ...config import ConfigManager

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

console = Console()


@handle_errors
@track_time("Running benchmark")
def benchmark_command(
    config: Optional[Path] = typer.Option(
        None,
        "--config",
        "-c",
        help="Configuration file (defaults to k-bert.yaml)",
    ),
    steps: int = typer.Option(
        20,
        "--steps",
        "-n",
        help="Number of benchmark steps",
    ),
    warmup: int = typer.Option(
        5,
        "--warmup",
        help="Number of warmup steps",
    ),
    batch_size: Optional[int] = typer.Option(
        None,
        "--batch-size",
        "-b",
        help="Override batch size",
    ),
    seq_length: Optional[int] = typer.Option(
        None,
        "--seq-length",
        "-s",
        help="Override sequence length",
    ),
    no_config: bool = typer.Option(
        False,
        "--no-config",
        help="Run with default settings",
    ),
    compare: bool = typer.Option(
        False,
        "--compare",
        help="Compare different configurations",
    ),
    memory: bool = typer.Option(
        False,
        "--memory",
        help="Track memory usage",
    ),
    export: Optional[Path] = typer.Option(
        None,
        "--export",
        help="Export results to JSON",
    ),
    debug: bool = typer.Option(
        False,
        "--debug",
        "-d",
        help="Enable debug logging",
    ),
):
    """Benchmark model performance.

    This command benchmarks training performance using your configuration.
    It helps identify optimal batch sizes and settings for your hardware.
    
    Examples:
        # Benchmark with project config
        k-bert benchmark
        
        # Benchmark with specific batch size
        k-bert benchmark --batch-size 64
        
        # Compare different configurations
        k-bert benchmark --compare
        
        # Export results
        k-bert benchmark --export results.json
        
        # Run without config
        k-bert benchmark --no-config --batch-size 32
    """
    # Configure logging
    log_level = "DEBUG" if debug else "INFO"
    logger.remove()
    logger.add(sys.stderr, level=log_level, enqueue=False)
    
    console.print("\n[bold blue]K-BERT Performance Benchmark[/bold blue]")
    console.print("=" * 60)
    
    # Load configuration
    if no_config:
        console.print("[yellow]Running with default configuration (--no-config)[/yellow]")
        config_overrides = {
            'data': {
                'batch_size': batch_size or 32,
                'max_length': seq_length or 256,
            },
            'models': {
                'default_model': 'answerdotai/ModernBERT-base',
                'head': {'type': 'binary_classification'},
            }
        }
        merged_config = ConfigManager().get_merged_config(cli_overrides=config_overrides)
    else:
        # Find configuration file
        if config is None:
            config_paths = [
                Path.cwd() / "k-bert.yaml",
                Path.cwd() / "k-bert.yml",
            ]
            config = next((p for p in config_paths if p.exists()), None)
            
            if config is None:
                print_error(
                    "No configuration file found. Create one with 'k-bert config init' "
                    "or use --no-config.",
                    title="Configuration Required"
                )
                raise typer.Exit(1)
        
        console.print(f"[green]Using configuration: {config}[/green]")
        
        # Build CLI overrides
        cli_overrides = {}
        if batch_size:
            cli_overrides.setdefault('data', {})['batch_size'] = batch_size
        if seq_length:
            cli_overrides.setdefault('data', {})['max_length'] = seq_length
        
        config_manager = ConfigManager()
        merged_config = config_manager.get_merged_config(
            cli_overrides=cli_overrides,
            project_path=config,
            validate=True
        )
    
    # Extract settings
    batch_size = merged_config.data.batch_size
    seq_length = merged_config.data.max_length
    model_name = merged_config.models.default_model
    
    # Display configuration
    console.print("\n[bold]Benchmark Configuration:[/bold]")
    console.print(f"  Model: {model_name}")
    console.print(f"  Batch size: {batch_size}")
    console.print(f"  Sequence length: {seq_length}")
    console.print(f"  Steps: {steps} (+ {warmup} warmup)")
    if memory:
        console.print(f"  Memory tracking: Enabled")
    
    # Import components
    try:
        from models.factory import create_model
        
    except ImportError as e:
        print_error(
            f"Failed to import components: {str(e)}",
            title="Import Error"
        )
        raise typer.Exit(1)
    
    # Run benchmark
    if compare:
        # Compare different batch sizes
        batch_sizes = [8, 16, 32, 64, 128]
        results = {}
        
        console.print("\n[yellow]Comparing batch sizes...[/yellow]")
        
        for bs in batch_sizes:
            if bs > 64 and seq_length > 128:
                # Skip large batch sizes for long sequences
                continue
            
            console.print(f"\n[cyan]Benchmarking batch size: {bs}[/cyan]")
            try:
                results[f"batch_{bs}"] = _run_single_benchmark(
                    model_name=model_name,
                    batch_size=bs,
                    seq_length=seq_length,
                    steps=steps,
                    warmup_steps=warmup,
                    track_memory=memory,
                )
            except Exception as e:
                console.print(f"[red]Failed with batch size {bs}: {e}[/red]")
                continue
        
        # Display comparison
        _display_comparison(results, console)
        
    else:
        # Single benchmark
        results = _run_single_benchmark(
            model_name=model_name,
            batch_size=batch_size,
            seq_length=seq_length,
            steps=steps,
            warmup_steps=warmup,
            track_memory=memory,
        )
        
        # Display results
        _display_results(results, console)
    
    # Export results if requested
    if export:
        export.parent.mkdir(parents=True, exist_ok=True)
        with open(export, "w") as f:
            json.dump(results if not compare else results, f, indent=2)
        print_success(f"Results exported to: {export}", title="Export Complete")
    
    # Show recommendations
    console.print("\n[bold]Recommendations:[/bold]")
    
    if not compare:
        throughput = results["throughput"]["samples_per_second"]
        if throughput < 10:
            console.print("  • Consider reducing batch size for better performance")
        elif throughput > 100:
            console.print("  • Try increasing batch size to improve GPU utilization")
        
        if memory and "memory" in results:
            mem_usage = results["memory"]["max_gb"]
            if mem_usage > 8:
                console.print("  • High memory usage - consider reducing batch size")
    else:
        # Find optimal batch size
        best_batch = max(results.items(), key=lambda x: x[1]["throughput"]["samples_per_second"])
        console.print(f"  • Optimal batch size: {best_batch[0].split('_')[1]}")
        console.print(f"    ({best_batch[1]['throughput']['samples_per_second']:.1f} samples/sec)")


def _run_single_benchmark(
    model_name: str,
    batch_size: int,
    seq_length: int,
    steps: int,
    warmup_steps: int,
    track_memory: bool = False,
) -> Dict[str, Any]:
    """Run a single benchmark configuration."""
    from models.factory import create_model
    
    # Create model
    model = create_model(
        model_name=model_name,
        model_type="modernbert_with_head",
        head_type="binary_classification",
        num_labels=2
    )
    
    # Create optimizer
    optimizer = optim.AdamW(learning_rate=2e-5)
    
    # Create dummy batch
    dummy_batch = {
        "input_ids": mx.ones((batch_size, seq_length), dtype=mx.int32),
        "attention_mask": mx.ones((batch_size, seq_length), dtype=mx.int32),
        "labels": mx.zeros((batch_size,), dtype=mx.int32),
    }
    
    # Define loss function
    def loss_fn(model, batch):
        outputs = model(**batch)
        
        if "loss" in outputs:
            return outputs["loss"]
        
        # Compute loss manually
        if isinstance(outputs, dict) and "logits" in outputs:
            logits = outputs["logits"]
        else:
            logits = outputs
        
        return nn.losses.cross_entropy(logits, batch["labels"], reduction="mean")
    
    # Create value_and_grad function
    value_and_grad_fn = nn.value_and_grad(model, loss_fn)
    
    # Warmup
    console.print(f"[dim]Running {warmup_steps} warmup steps...[/dim]")
    for _ in range(warmup_steps):
        loss, grads = value_and_grad_fn(model, dummy_batch)
        optimizer.update(model, grads)
        mx.eval(model.parameters())
    
    # Benchmark
    console.print(f"[dim]Running {steps} benchmark steps...[/dim]")
    
    step_times = []
    losses = []
    memory_usage = []
    
    for step in range(steps):
        start_time = time.time()
        
        # Forward and backward pass
        loss, grads = value_and_grad_fn(model, dummy_batch)
        
        # Update
        optimizer.update(model, grads)
        mx.eval(model.parameters())
        
        step_time = time.time() - start_time
        step_times.append(step_time)
        losses.append(float(loss))
        
        if track_memory:
            memory_usage.append(mx.metal.get_active_memory() / 1024**3)
        
        if (step + 1) % 5 == 0:
            console.print(f"Step {step + 1}/{steps} - {step_time:.3f}s", end="\r")
    
    # Calculate statistics
    results = {
        "model": model_name,
        "batch_size": batch_size,
        "seq_length": seq_length,
        "steps": steps,
        "step_time": {
            "mean": np.mean(step_times),
            "std": np.std(step_times),
            "min": np.min(step_times),
            "max": np.max(step_times),
        },
        "throughput": {
            "samples_per_second": batch_size / np.mean(step_times),
            "tokens_per_second": (batch_size * seq_length) / np.mean(step_times),
        },
        "loss": {
            "final": losses[-1],
            "mean": np.mean(losses),
        },
    }
    
    if track_memory and memory_usage:
        results["memory"] = {
            "mean_gb": np.mean(memory_usage),
            "max_gb": np.max(memory_usage),
        }
    
    return results


def _display_results(results: Dict[str, Any], console: Console):
    """Display benchmark results in a table."""
    table = Table(title="Benchmark Results")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    
    # Basic info
    table.add_row("Model", results["model"])
    table.add_row("Batch Size", str(results["batch_size"]))
    table.add_row("Sequence Length", str(results["seq_length"]))
    table.add_row("Steps", str(results["steps"]))
    
    # Timing
    table.add_section()
    table.add_row(
        "Avg Step Time",
        f"{results['step_time']['mean']:.3f}s ± {results['step_time']['std']:.3f}s"
    )
    table.add_row(
        "Min/Max Time",
        f"{results['step_time']['min']:.3f}s / {results['step_time']['max']:.3f}s"
    )
    
    # Throughput
    table.add_section()
    table.add_row(
        "Samples/Second",
        f"{results['throughput']['samples_per_second']:.1f}"
    )
    table.add_row(
        "Tokens/Second",
        f"{results['throughput']['tokens_per_second']:.0f}"
    )
    
    # Loss
    table.add_section()
    table.add_row("Final Loss", f"{results['loss']['final']:.4f}")
    
    # Memory
    if "memory" in results:
        table.add_section()
        table.add_row(
            "Memory Usage",
            f"{results['memory']['mean_gb']:.2f} GB (max: {results['memory']['max_gb']:.2f} GB)"
        )
    
    console.print(table)


def _display_comparison(results: Dict[str, Dict[str, Any]], console: Console):
    """Display comparison of multiple benchmark runs."""
    table = Table(title="Batch Size Comparison")
    table.add_column("Batch Size", style="cyan")
    table.add_column("Step Time (s)", style="yellow")
    table.add_column("Samples/s", style="green")
    table.add_column("Tokens/s", style="green")
    table.add_column("Memory (GB)", style="magenta")
    
    for name, result in sorted(results.items()):
        batch_size = str(result["batch_size"])
        step_time = f"{result['step_time']['mean']:.3f}"
        samples_per_sec = f"{result['throughput']['samples_per_second']:.1f}"
        tokens_per_sec = f"{result['throughput']['tokens_per_second']:.0f}"
        
        memory = "-"
        if "memory" in result:
            memory = f"{result['memory']['max_gb']:.2f}"
        
        table.add_row(
            batch_size,
            step_time,
            samples_per_sec,
            tokens_per_sec,
            memory
        )
    
    console.print(table)
    
    # Find best configuration
    best = max(results.items(), key=lambda x: x[1]["throughput"]["samples_per_second"])
    console.print(
        f"\n[bold green]Best throughput:[/bold green] "
        f"Batch size {best[1]['batch_size']} "
        f"({best[1]['throughput']['samples_per_second']:.1f} samples/s)"
    )