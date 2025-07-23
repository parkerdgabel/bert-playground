"""Main CLI application adapter.

This module creates the main Typer application and registers all command adapters.
It acts as the entry point for the CLI, delegating to individual command adapters.
"""

import typer
from rich.console import Console

from adapters.primary.cli.train_adapter import train
from adapters.primary.cli.predict_adapter import predict
from adapters.primary.cli.benchmark_adapter import benchmark
from adapters.primary.cli.config_adapter import config_app


# Create the main Typer app
app = typer.Typer(
    name="k-bert",
    help="K-BERT: MLX-based ModernBERT for Kaggle competitions",
    no_args_is_help=True,
    rich_markup_mode="rich",
)

# Create console for output
console = Console()


# Register command adapters
app.command(name="train")(train)
app.command(name="predict")(predict)
app.command(name="benchmark")(benchmark)

# Register subcommand apps
app.add_typer(config_app, name="config")


# Add version command
@app.command()
def version():
    """Display K-BERT version information."""
    from cli._version import __version__
    
    console.print(f"[bold blue]K-BERT[/bold blue] version {__version__}")


# Add a simple info command
@app.command()
def info():
    """Display system and configuration information."""
    import platform
    import mlx
    
    console.print("\n[bold blue]K-BERT System Information[/bold blue]")
    console.print("=" * 60)
    
    # System info
    console.print("\n[bold]System:[/bold]")
    console.print(f"  Platform: {platform.system()} {platform.release()}")
    console.print(f"  Python: {platform.python_version()}")
    console.print(f"  MLX: {mlx.__version__}")
    
    # Hardware info
    console.print("\n[bold]Hardware:[/bold]")
    if hasattr(mlx.core, 'metal'):
        console.print(f"  Metal device: Available")
    else:
        console.print(f"  Metal device: Not available")
    
    console.print("\n[dim]For detailed configuration, use: k-bert config show[/dim]")


def main():
    """Main entry point for the CLI."""
    app()


if __name__ == "__main__":
    main()