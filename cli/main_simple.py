"""Simplified main entry point for testing the K-BERT CLI structure.

This version doesn't require all adapters to be fully functional.
"""

import typer
from rich.console import Console
from pathlib import Path
from typing import Optional

from cli import __version__


# Create the main Typer app
app = typer.Typer(
    name="k-bert",
    help="K-BERT: MLX-based ModernBERT for Kaggle competitions",
    no_args_is_help=True,
    rich_markup_mode="rich",
    add_completion=False,
)

# Create console for output
console = Console()


# Version command
@app.command()
def version():
    """Display K-BERT version information."""
    console.print(f"[bold blue]K-BERT[/bold blue] version {__version__}")


# Info command
@app.command()
def info():
    """Display system and configuration information."""
    import platform
    
    console.print("\n[bold blue]K-BERT System Information[/bold blue]")
    console.print("=" * 60)
    
    # System info
    console.print("\n[bold]System:[/bold]")
    console.print(f"  Platform: {platform.system()} {platform.release()}")
    console.print(f"  Python: {platform.python_version()}")
    
    # Try to import MLX
    try:
        import mlx.core as mx
        console.print(f"  MLX: {getattr(mx, '__version__', 'unknown')}")
        console.print(f"  Metal device: {'Available' if hasattr(mx, 'metal') else 'Not available'}")
    except ImportError:
        console.print(f"  MLX: Not installed")
    
    console.print("\n[dim]For detailed configuration, use: k-bert config show[/dim]")


# Train command (stub)
@app.command()
def train(
    config: Optional[Path] = typer.Option(None, "--config", "-c", help="Configuration file"),
    epochs: Optional[int] = typer.Option(None, "--epochs", help="Number of epochs"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Show configuration without running"),
):
    """Train a BERT model."""
    console.print("\n[bold blue]K-BERT Training[/bold blue]")
    console.print("=" * 60)
    
    console.print("\n[yellow]Training command implementation in progress[/yellow]")
    console.print(f"Config: {config}")
    console.print(f"Epochs: {epochs}")
    console.print(f"Dry run: {dry_run}")
    
    console.print("\n[dim]The full hexagonal architecture is being implemented.[/dim]")


# Evaluate command (stub)
@app.command()
def evaluate(
    model: Optional[Path] = typer.Option(None, "--model", "-m", help="Model path"),
    data: Optional[Path] = typer.Option(None, "--data", "-d", help="Data path"),
):
    """Evaluate a trained model."""
    console.print("\n[bold blue]K-BERT Evaluation[/bold blue]")
    console.print("=" * 60)
    
    console.print("\n[yellow]Evaluate command implementation in progress[/yellow]")
    console.print(f"Model: {model}")
    console.print(f"Data: {data}")


# Predict command (stub)
@app.command()
def predict(
    model: Optional[Path] = typer.Option(None, "--model", "-m", help="Model path"),
    input: Optional[Path] = typer.Option(None, "--input", "-i", help="Input data"),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Output path"),
):
    """Generate predictions."""
    console.print("\n[bold blue]K-BERT Prediction[/bold blue]")
    console.print("=" * 60)
    
    console.print("\n[yellow]Predict command implementation in progress[/yellow]")
    console.print(f"Model: {model}")
    console.print(f"Input: {input}")
    console.print(f"Output: {output}")


# Config command group
config_app = typer.Typer(help="Configuration management commands")


@config_app.command()
def init(
    user: bool = typer.Option(False, "--user", help="Initialize user configuration"),
    force: bool = typer.Option(False, "--force", "-f", help="Overwrite existing"),
):
    """Initialize K-BERT configuration."""
    import yaml
    
    # Determine target path
    if user:
        config_dir = Path.home() / ".k-bert"
        config_dir.mkdir(exist_ok=True)
        config_path = config_dir / "config.yaml"
        config_type = "user"
    else:
        config_path = Path.cwd() / "k-bert.yaml"
        config_type = "project"
    
    # Check if exists
    if config_path.exists() and not force:
        console.print(f"[yellow]Configuration already exists at {config_path}[/yellow]")
        console.print("Use --force to overwrite")
        raise typer.Exit(1)
    
    # Create default configuration
    default_config = {
        "models": {
            "type": "modernbert_with_head",
            "default_model": "answerdotai/ModernBERT-base",
        },
        "data": {
            "batch_size": 32,
            "train_path": "data/train.csv",
            "val_path": "data/val.csv",
        },
        "training": {
            "epochs": 3,
            "learning_rate": 5e-5,
            "output_dir": "output",
        },
    }
    
    # Write configuration
    with open(config_path, 'w') as f:
        yaml.dump(default_config, f, default_flow_style=False, sort_keys=False)
    
    console.print(f"[green]Created {config_type} configuration at {config_path}[/green]")


@config_app.command()
def show():
    """Show current configuration."""
    from cli.config.loader import ConfigurationLoader
    import yaml
    
    loader = ConfigurationLoader()
    configs = []
    sources = []
    
    # Load configs
    if user_path := loader.find_user_config():
        configs.append(loader.load_yaml_config(user_path))
        sources.append(f"User: {user_path}")
    
    if project_path := loader.find_project_config():
        configs.append(loader.load_yaml_config(project_path))
        sources.append(f"Project: {project_path}")
    
    if not configs:
        console.print("[yellow]No configuration files found[/yellow]")
        raise typer.Exit(1)
    
    # Merge and display
    merged = loader.merge_configs(configs) if configs else {}
    
    console.print("[bold blue]Current Configuration[/bold blue]")
    console.print("\nSources:")
    for source in sources:
        console.print(f"  • {source}")
    
    console.print("\n[bold]Merged Configuration:[/bold]")
    yaml_output = yaml.dump(merged, default_flow_style=False, sort_keys=False)
    
    from rich.syntax import Syntax
    syntax = Syntax(yaml_output, "yaml", theme="monokai", line_numbers=False)
    console.print(syntax)


# Register config subcommands
app.add_typer(config_app, name="config")


def main():
    """Main entry point for the CLI."""
    app()


if __name__ == "__main__":
    main()