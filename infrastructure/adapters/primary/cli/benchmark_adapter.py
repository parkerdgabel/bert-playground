"""Thin CLI adapter for benchmark command.

This adapter handles benchmarking of model performance.
"""

from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

from infrastructure.bootstrap import get_service
from infrastructure.adapters.primary.cli.base import CLIAdapter, ProgressContext, format_time, format_size


console = Console()


async def benchmark_command(
    model: Optional[Path] = typer.Option(
        None,
        "--model",
        "-m",
        help="Path to model checkpoint (uses default if not specified)",
    ),
    batch_sizes: str = typer.Option(
        "1,8,16,32,64",
        "--batch-sizes",
        "-b",
        help="Comma-separated list of batch sizes to test",
    ),
    sequence_lengths: str = typer.Option(
        "128,256,512",
        "--sequence-lengths",
        "-s",
        help="Comma-separated list of sequence lengths to test",
    ),
    warmup_steps: int = typer.Option(
        5,
        "--warmup",
        "-w",
        help="Number of warmup iterations",
    ),
    num_steps: int = typer.Option(
        20,
        "--steps",
        "-n",
        help="Number of benchmark iterations",
    ),
    use_compile: bool = typer.Option(
        True,
        "--compile/--no-compile",
        help="Use MLX compilation",
    ),
    memory_profile: bool = typer.Option(
        False,
        "--memory",
        help="Include memory profiling",
    ),
    export_results: Optional[Path] = typer.Option(
        None,
        "--export",
        help="Export results to file",
    ),
):
    """Benchmark model performance across different configurations.
    
    This command tests model inference speed and memory usage
    with various batch sizes and sequence lengths.
    """
    console.print("\n[bold blue]K-BERT Performance Benchmark[/bold blue]")
    console.print("=" * 60)
    
    # Parse batch sizes and sequence lengths
    try:
        batch_size_list = [int(x.strip()) for x in batch_sizes.split(",")]
        seq_length_list = [int(x.strip()) for x in sequence_lengths.split(",")]
    except ValueError:
        console.print("[red]Error: Invalid batch sizes or sequence lengths[/red]")
        raise typer.Exit(1)
    
    # Display configuration
    config_table = Table(title="Benchmark Configuration", show_header=True)
    config_table.add_column("Parameter", style="cyan")
    config_table.add_column("Value", style="green")
    
    config_table.add_row("Model", str(model) if model else "Default")
    config_table.add_row("Batch Sizes", batch_sizes)
    config_table.add_row("Sequence Lengths", sequence_lengths)
    config_table.add_row("Warmup Steps", str(warmup_steps))
    config_table.add_row("Benchmark Steps", str(num_steps))
    config_table.add_row("Use Compilation", "Yes" if use_compile else "No")
    config_table.add_row("Memory Profiling", "Yes" if memory_profile else "No")
    
    console.print(config_table)
    
    # In a real implementation, this would:
    # 1. Create a BenchmarkRequestDTO
    # 2. Call a BenchmarkUseCase
    # 3. Display results
    
    # For now, show a placeholder
    console.print("\n[yellow]Note: Benchmark use case not yet implemented[/yellow]")
    console.print("[dim]This would run performance tests and display results[/dim]")
    
    # Example results table
    console.print("\n[bold]Example Results:[/bold]")
    results_table = Table(title="Inference Performance", show_header=True)
    results_table.add_column("Batch Size", style="cyan")
    results_table.add_column("Seq Length", style="cyan")
    results_table.add_column("Throughput (samples/s)", style="green")
    results_table.add_column("Latency (ms)", style="yellow")
    results_table.add_column("Memory (MB)", style="magenta")
    
    # Example data
    results_table.add_row("32", "128", "245.6", "130.4", "512")
    results_table.add_row("32", "256", "142.3", "224.7", "768")
    results_table.add_row("64", "128", "367.8", "174.1", "924")
    
    console.print(results_table)


# Create the Typer command
benchmark = benchmark_command