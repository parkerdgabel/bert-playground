"""Benchmark command implementation."""

import json
import sys
import time
from pathlib import Path
from typing import Any, Optional

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
import typer
from loguru import logger

from ...utils import (
    get_console,
    handle_errors,
    print_error,
    print_success,
    track_time,
    validate_batch_size,
    validate_path,
)
from ...utils.console import create_table
from ...config import ConfigManager
from ...plugins import ComponentRegistry

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))


@handle_errors
@track_time("Running benchmarks")
def benchmark_command(
    batch_size: int = typer.Option(
        32,
        "--batch-size",
        "-b",
        help="Batch size for benchmarking",
        callback=validate_batch_size,
    ),
    seq_length: int = typer.Option(256, "--seq-length", "-s", help="Sequence length"),
    steps: int = typer.Option(20, "--steps", "-n", help="Number of steps to run"),
    model_type: str = typer.Option(
        "modernbert",
        "--model-type",
        help="Model type: bert, modernbert, bert-lora, modernbert-lora",
    ),
    model_name: str = typer.Option(
        "answerdotai/ModernBERT-base", "--model", "-m", help="Model name"
    ),
    warmup_steps: int = typer.Option(5, "--warmup", help="Warmup steps"),
    profile: bool = typer.Option(
        False, "--profile", "-p", help="Enable detailed profiling"
    ),
    memory: bool = typer.Option(False, "--memory", help="Monitor memory usage"),
    compare: bool = typer.Option(
        False, "--compare", help="Compare different model types"
    ),
    test_compilation: bool = typer.Option(
        False, "--test-compilation", help="Test with and without MLX compilation"
    ),
    export_results: Path | None = typer.Option(
        None, "--export", help="Export results to JSON"
    ),
    config: Optional[Path] = typer.Option(
        None,
        "--config",
        help="Configuration file to use",
        callback=lambda p: validate_path(
            p, must_exist=True, extensions=[".yaml", ".yml", ".json"]
        )
        if p
        else None,
    ),
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
    
    # Load configuration
    config_manager = ConfigManager()
    merged_config = config_manager.get_merged_config(
        cli_overrides={
            'data': {
                'batch_size': batch_size,
                'max_length': seq_length,
            },
            'models': {
                'default_model': model_name,
            }
        },
        project_path=config,
        validate=True
    )
    
    # Extract configuration values
    batch_size = merged_config.data.batch_size
    seq_length = merged_config.data.max_length
    model_name = merged_config.models.default_model
    
    # Configure logging
    from utils.logging_utils import bind_context, log_timing, MetricsLogger, lazy_debug
    
    # Create benchmark logger
    bench_log = bind_context(
        command="benchmark",
        model_type=model_type,
        batch_size=batch_size,
        seq_length=seq_length
    )
    bench_log.info("Starting benchmark")

    # Import necessary components
    try:
        with log_timing("import_models", level="DEBUG"):
            from models.factory import MODEL_REGISTRY, create_model

    except ImportError as e:
        bench_log.error(f"Failed to import components: {str(e)}")
        print_error(
            f"Failed to import components: {str(e)}\n"
            "Make sure all dependencies are installed.",
            title="Import Error",
        )
        raise typer.Exit(1)

    # Create dummy data
    dummy_batch = {
        "input_ids": mx.ones((batch_size, seq_length), dtype=mx.int32),
        "attention_mask": mx.ones((batch_size, seq_length), dtype=mx.int32),
        "labels": mx.zeros((batch_size,), dtype=mx.int32),
        "token_type_ids": mx.zeros(
            (batch_size, seq_length), dtype=mx.int32
        ),  # Some models may need this
    }

    # Test compilation if requested
    if test_compilation:
        console.print("\n[yellow]Testing MLX compilation impact...[/yellow]")
        bench_log.info("Testing compilation impact")

        # Run without compilation
        console.print("\n[cyan]Without compilation:[/cyan]")
        results_no_compile = _run_single_benchmark(
            model_type,
            dummy_batch,
            steps,
            warmup_steps,
            profile,
            memory,
            console,
            use_compilation=False,
        )

        # Run with compilation
        console.print("\n[cyan]With compilation:[/cyan]")
        results_compile = _run_single_benchmark(
            model_type,
            dummy_batch,
            steps,
            warmup_steps,
            profile,
            memory,
            console,
            use_compilation=True,
        )

        # Display comparison
        _display_compilation_comparison(results_no_compile, results_compile, console)

        # Export results if requested
        if export_results:
            with open(export_results, "w") as f:
                json.dump(
                    {
                        "without_compilation": results_no_compile,
                        "with_compilation": results_compile,
                    },
                    f,
                    indent=2,
                )
            print_success(f"Results exported to {export_results}")

        return

    # Run benchmarks for different models if compare mode
    if compare:
        model_types = [
            "bert-binary",
            "modernbert-binary",
            "bert-lora-binary",
            "modernbert-lora-binary",
        ]
        results = {}

        for mt in model_types:
            console.print(f"\n[yellow]Benchmarking {mt} model...[/yellow]")
            results[mt] = _run_single_benchmark(
                mt, dummy_batch, steps, warmup_steps, profile, memory, console
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
            model_type, dummy_batch, steps, warmup_steps, profile, memory, console
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
    dummy_batch: dict[str, mx.array],
    steps: int,
    warmup_steps: int,
    profile: bool,
    memory: bool,
    console,
    use_compilation: bool = False,
) -> dict[str, Any]:
    """Run benchmark for a single model type."""
    from utils.logging_utils import bind_context, log_timing, ProgressTracker, lazy_debug
    
    # Create benchmark context
    bench_log = bind_context(
        benchmark_type="single",
        model=model_type,
        compilation=use_compilation
    )
    bench_log.info("Starting single benchmark")

    # Import here to avoid issues with module loading
    from models.factory import MODEL_REGISTRY, create_model

    # Create model based on type
    if model_type in MODEL_REGISTRY:
        model = MODEL_REGISTRY[model_type](num_labels=2)
        model_desc = model_type
    else:
        # Default to modernbert with binary classification
        model = create_model(
            "modernbert_with_head", head_type="binary_classification", num_labels=2
        )
        model_desc = "ModernBERT Binary"

    console.print(f"Created {model_desc} model")
    bench_log.info(f"Created {model_desc} model")

    # Create optimizer
    optimizer = optim.AdamW(learning_rate=2e-5)

    # Setup compiled functions if requested
    train_step_fn = None
    if use_compilation:
        try:
            from training.core.compiled import create_compiled_train_step

            # Create simple config for compilation
            class SimpleConfig:
                class training:
                    mixed_precision = False
                    label_smoothing = 0.0
                    gradient_accumulation_steps = 1

                class optimizer:
                    max_grad_norm = 1.0

            config = SimpleConfig()
            compiled_step, state = create_compiled_train_step(
                model, optimizer, config, gradient_accumulator=None
            )
            train_step_fn = compiled_step
            console.print("[green]Using compiled training step[/green]")
        except Exception as e:
            bench_log.warning(f"Failed to compile: {e}")
            console.print(f"[yellow]Failed to compile: {e}[/yellow]")
            use_compilation = False

    # Warmup
    console.print(f"Running {warmup_steps} warmup steps...")
    with log_timing("warmup", steps=warmup_steps):
        for _ in range(warmup_steps):
            if use_compilation and train_step_fn:
                _ = train_step_fn(dummy_batch)
            else:
                loss = _forward_backward_step(model, dummy_batch, optimizer)

    # Benchmark
    console.print(f"Running {steps} benchmark steps...")
    forward_times = []
    backward_times = []
    total_times = []
    losses = []
    memory_usage = []
    
    # Use ProgressTracker for benchmark steps
    bench_log.info(f"Running {steps} benchmark steps")

    for step in range(steps):
        step_start = time.time()

        if use_compilation and train_step_fn:
            # Use compiled function
            loss, metrics = train_step_fn(dummy_batch)
            mx.eval(loss)  # Force evaluation

            # For timing, we can't separate forward/backward in compiled mode
            total_time = time.time() - step_start
            forward_time = total_time * 0.4  # Estimate
            backward_time = total_time * 0.6  # Estimate
        else:
            # Standard execution
            # Forward pass
            forward_start = time.time()
            outputs = model(**dummy_batch)

            # Get loss
            if "loss" in outputs:
                loss = outputs["loss"]
            else:
                # Compute loss if not provided
                if isinstance(outputs, dict) and "logits" in outputs:
                    logits = outputs["logits"]
                elif hasattr(outputs, "logits"):
                    logits = outputs.logits
                else:
                    logits = outputs

                loss = nn.losses.cross_entropy(
                    logits, dummy_batch["labels"], reduction="mean"
                )
            forward_time = time.time() - forward_start

            # Backward pass
            backward_start = time.time()

            # Define loss function for gradients
            def loss_fn(model, batch):
                outputs = model(**batch)
                return outputs["loss"] if "loss" in outputs else loss

            # Compute gradients
            value_and_grad_fn = nn.value_and_grad(model, loss_fn)
            _, grads = value_and_grad_fn(model, dummy_batch)

            # Update
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
            "samples_per_second": dummy_batch["input_ids"].shape[0]
            / np.mean(total_times),
            "tokens_per_second": (
                dummy_batch["input_ids"].shape[0] * dummy_batch["input_ids"].shape[1]
            )
            / np.mean(total_times),
        },
        "loss": {
            "final": losses[-1],
            "mean": np.mean(losses),
        },
    }

    if memory:
        results["memory"] = {
            "mean_gb": np.mean(memory_usage),
            "max_gb": np.max(memory_usage),
        }

    return results


def _forward_backward_step(model, batch, optimizer):
    """Perform a single forward-backward step."""

    def loss_fn(model, batch):
        outputs = model(**batch)

        # Model should return loss when labels are provided
        if "loss" in outputs:
            return outputs["loss"]

        # Fallback to computing loss manually
        if isinstance(outputs, dict) and "logits" in outputs:
            logits = outputs["logits"]
        elif hasattr(outputs, "logits"):
            logits = outputs.logits
        else:
            logits = outputs

        return nn.losses.cross_entropy(logits, batch["labels"], reduction="mean")

    # Create value_and_grad function
    value_and_grad_fn = nn.value_and_grad(model, loss_fn)

    # Compute loss and gradients
    loss, grads = value_and_grad_fn(model, batch)

    # Update model
    optimizer.update(model, grads)

    # Force evaluation
    mx.eval(model.parameters())

    return loss


def _display_results(results: dict[str, Any], console):
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
    table.add_row(
        "Forward Time (ms)",
        f"{results['forward_time']['mean'] * 1000:.2f} ± {results['forward_time']['std'] * 1000:.2f}",
    )
    table.add_row(
        "Backward Time (ms)",
        f"{results['backward_time']['mean'] * 1000:.2f} ± {results['backward_time']['std'] * 1000:.2f}",
    )
    table.add_row(
        "Total Time (ms)",
        f"{results['total_time']['mean'] * 1000:.2f} ± {results['total_time']['std'] * 1000:.2f}",
    )
    table.add_row("", "")  # Separator

    # Throughput metrics
    table.add_row(
        "Samples/Second", f"{results['throughput']['samples_per_second']:.2f}"
    )
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


def _display_comparison(results: dict[str, dict[str, Any]], console):
    """Display comparison of multiple benchmark results."""
    console.print("\n[bold green]Benchmark Comparison[/bold green]")
    console.print("=" * 60)

    # Create comparison table
    table = create_table(
        "Model Comparison",
        ["Model", "Forward (ms)", "Backward (ms)", "Total (ms)", "Samples/s", "Loss"],
    )

    for model_type, result in results.items():
        table.add_row(
            model_type,
            f"{result['forward_time']['mean'] * 1000:.2f}",
            f"{result['backward_time']['mean'] * 1000:.2f}",
            f"{result['total_time']['mean'] * 1000:.2f}",
            f"{result['throughput']['samples_per_second']:.2f}",
            f"{result['loss']['final']:.4f}",
        )

    console.print(table)

    # Find best performing model
    best_throughput = max(
        results.items(), key=lambda x: x[1]["throughput"]["samples_per_second"]
    )
    console.print(
        f"\n[bold green]Best throughput:[/bold green] {best_throughput[0]} ({best_throughput[1]['throughput']['samples_per_second']:.2f} samples/s)"
    )

    best_speed = min(results.items(), key=lambda x: x[1]["total_time"]["mean"])
    console.print(
        f"[bold green]Fastest per step:[/bold green] {best_speed[0]} ({best_speed[1]['total_time']['mean'] * 1000:.2f} ms/step)"
    )


def _display_compilation_comparison(
    results_no_compile: dict[str, Any], results_compile: dict[str, Any], console
):
    """Display comparison between compiled and non-compiled results."""
    console.print("\n[bold green]Compilation Impact[/bold green]")
    console.print("=" * 60)

    # Create comparison table
    table = create_table(
        "Compilation Comparison",
        ["Metric", "Without Compilation", "With Compilation", "Speedup"],
    )

    # Calculate speedups
    forward_speedup = (
        results_no_compile["forward_time"]["mean"]
        / results_compile["forward_time"]["mean"]
    )
    backward_speedup = (
        results_no_compile["backward_time"]["mean"]
        / results_compile["backward_time"]["mean"]
    )
    total_speedup = (
        results_no_compile["total_time"]["mean"] / results_compile["total_time"]["mean"]
    )
    throughput_speedup = (
        results_compile["throughput"]["samples_per_second"]
        / results_no_compile["throughput"]["samples_per_second"]
    )

    # Add rows
    table.add_row(
        "Forward Time (ms)",
        f"{results_no_compile['forward_time']['mean'] * 1000:.2f}",
        f"{results_compile['forward_time']['mean'] * 1000:.2f}",
        f"{forward_speedup:.2f}x",
    )

    table.add_row(
        "Backward Time (ms)",
        f"{results_no_compile['backward_time']['mean'] * 1000:.2f}",
        f"{results_compile['backward_time']['mean'] * 1000:.2f}",
        f"{backward_speedup:.2f}x",
    )

    table.add_row(
        "Total Time (ms)",
        f"{results_no_compile['total_time']['mean'] * 1000:.2f}",
        f"{results_compile['total_time']['mean'] * 1000:.2f}",
        f"{total_speedup:.2f}x",
    )

    table.add_row("", "", "", "")  # Separator

    table.add_row(
        "Samples/Second",
        f"{results_no_compile['throughput']['samples_per_second']:.2f}",
        f"{results_compile['throughput']['samples_per_second']:.2f}",
        f"{throughput_speedup:.2f}x",
    )

    table.add_row(
        "Tokens/Second",
        f"{results_no_compile['throughput']['tokens_per_second']:.2f}",
        f"{results_compile['throughput']['tokens_per_second']:.2f}",
        f"{throughput_speedup:.2f}x",
    )

    console.print(table)

    # Summary
    console.print(f"\n[bold green]Overall speedup:[/bold green] {total_speedup:.2f}x")
    console.print(
        f"[bold green]Throughput improvement:[/bold green] {(throughput_speedup - 1) * 100:.1f}%"
    )
