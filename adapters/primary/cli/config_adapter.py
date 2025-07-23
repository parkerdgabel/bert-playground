"""Thin CLI adapter for configuration commands.

This adapter handles configuration display and validation.
"""

from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.syntax import Syntax
from rich.panel import Panel
import yaml

from infrastructure.bootstrap import get_service
from ports.secondary.configuration import ConfigurationProvider
from adapters.primary.cli.base import CLIAdapter


console = Console()


async def config_show_command(
    config: Optional[Path] = typer.Option(
        None,
        "--config",
        "-c",
        help="Configuration file to display",
    ),
    section: Optional[str] = typer.Option(
        None,
        "--section",
        "-s",
        help="Show only specific section",
    ),
    defaults: bool = typer.Option(
        False,
        "--defaults",
        help="Show default configuration",
    ),
):
    """Display current configuration.
    
    This command shows the active configuration with proper formatting.
    """
    console.print("\n[bold blue]K-BERT Configuration[/bold blue]")
    console.print("=" * 60)
    
    try:
        config_provider = get_service(ConfigurationProvider)
        
        # Load configuration if specified
        if config:
            config_provider.load_file(str(config))
            console.print(f"\nLoaded from: [cyan]{config}[/cyan]")
        
        # Get configuration
        if section:
            config_data = config_provider.get(section, {})
            if not config_data:
                console.print(f"[yellow]Section '{section}' not found[/yellow]")
                return
        else:
            # Get all configuration
            config_data = {
                "models": config_provider.get("models", {}),
                "training": config_provider.get("training", {}),
                "data": config_provider.get("data", {}),
                "logging": config_provider.get("logging", {}),
                "mlflow": config_provider.get("mlflow", {}),
            }
        
        # Format as YAML
        yaml_str = yaml.dump(config_data, default_flow_style=False, sort_keys=False)
        
        # Display with syntax highlighting
        syntax = Syntax(yaml_str, "yaml", theme="monokai", line_numbers=True)
        panel = Panel(syntax, title=f"Configuration{f' - {section}' if section else ''}")
        console.print(panel)
        
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)


async def config_validate_command(
    config: Path = typer.Argument(
        ...,
        help="Configuration file to validate",
    ),
    strict: bool = typer.Option(
        False,
        "--strict",
        help="Enable strict validation",
    ),
):
    """Validate a configuration file.
    
    This command checks if a configuration file is valid and reports any issues.
    """
    console.print("\n[bold blue]Configuration Validation[/bold blue]")
    console.print("=" * 60)
    
    if not config.exists():
        console.print(f"[red]Error: Configuration file not found: {config}[/red]")
        raise typer.Exit(1)
    
    try:
        # In a real implementation, this would:
        # 1. Create a ConfigValidationRequestDTO
        # 2. Call a ConfigValidationUseCase
        # 3. Display validation results
        
        # For now, basic validation
        with open(config) as f:
            config_data = yaml.safe_load(f)
        
        console.print(f"\n[green]✓[/green] Configuration file is valid YAML")
        
        # Check required sections
        required_sections = ["models", "training", "data"]
        missing_sections = [s for s in required_sections if s not in config_data]
        
        if missing_sections:
            console.print(f"\n[yellow]Warning: Missing sections: {', '.join(missing_sections)}[/yellow]")
        else:
            console.print(f"[green]✓[/green] All required sections present")
        
        # Display summary
        console.print(f"\n[bold]Configuration Summary:[/bold]")
        console.print(f"  Model: {config_data.get('models', {}).get('default_model', 'Not specified')}")
        console.print(f"  Training epochs: {config_data.get('training', {}).get('epochs', 'Not specified')}")
        console.print(f"  Batch size: {config_data.get('data', {}).get('batch_size', 'Not specified')}")
        
        if strict:
            console.print("\n[dim]Strict validation not yet implemented[/dim]")
        
    except yaml.YAMLError as e:
        console.print(f"[red]Invalid YAML: {e}[/red]")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)


async def config_init_command(
    output: Path = typer.Option(
        Path("k-bert.yaml"),
        "--output",
        "-o",
        help="Output file path",
    ),
    force: bool = typer.Option(
        False,
        "--force",
        "-f",
        help="Overwrite existing file",
    ),
    minimal: bool = typer.Option(
        False,
        "--minimal",
        help="Create minimal configuration",
    ),
):
    """Initialize a new configuration file.
    
    This command creates a new k-bert.yaml configuration file with defaults.
    """
    console.print("\n[bold blue]Initialize Configuration[/bold blue]")
    console.print("=" * 60)
    
    if output.exists() and not force:
        console.print(f"[yellow]File already exists: {output}[/yellow]")
        console.print("Use --force to overwrite")
        raise typer.Exit(1)
    
    try:
        # In a real implementation, this would call a use case
        # For now, create a basic config
        
        if minimal:
            config_template = {
                "models": {
                    "default_model": "answerdotai/ModernBERT-base"
                },
                "training": {
                    "epochs": 3,
                    "learning_rate": 2e-5
                },
                "data": {
                    "batch_size": 32
                }
            }
        else:
            config_template = {
                "models": {
                    "default_model": "answerdotai/ModernBERT-base",
                    "type": "modernbert_with_head",
                    "head_type": "binary_classification",
                    "num_labels": 2
                },
                "training": {
                    "epochs": 3,
                    "batch_size": 32,
                    "learning_rate": 2e-5,
                    "weight_decay": 0.01,
                    "warmup_ratio": 0.1,
                    "eval_strategy": "epoch",
                    "save_strategy": "epoch",
                    "logging_steps": 100,
                    "output_dir": "output"
                },
                "data": {
                    "batch_size": 32,
                    "max_length": 512,
                    "train_path": null,
                    "val_path": null
                },
                "logging": {
                    "level": "INFO",
                    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
                },
                "mlflow": {
                    "enabled": true,
                    "tracking_uri": "file:./mlruns",
                    "experiment_name": "k-bert-experiments"
                }
            }
        
        # Write configuration
        with open(output, 'w') as f:
            yaml.dump(config_template, f, default_flow_style=False, sort_keys=False)
        
        console.print(f"\n[green]✓[/green] Configuration created: [cyan]{output}[/cyan]")
        console.print("\n[bold]Next steps:[/bold]")
        console.print(f"  1. Edit [cyan]{output}[/cyan] to set your data paths")
        console.print(f"  2. Run training: [cyan]k-bert train[/cyan]")
        
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)


# Create config subcommand app
config_app = typer.Typer(help="Configuration management commands")

# Register subcommands
config_app.command(name="show")(config_show_command)
config_app.command(name="validate")(config_validate_command)
config_app.command(name="init")(config_init_command)