"""Info command for the CLI.

This command displays system and configuration information.
"""

from pathlib import Path
from typing import Optional
import platform
import json

import typer
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.syntax import Syntax
from loguru import logger

from cli.bootstrap import initialize_cli, get_service, get_config, shutdown_cli
from cli.config.loader import ConfigurationLoader


console = Console()


def info(
    config: bool = typer.Option(
        False,
        "--config", "-c",
        help="Show current configuration",
    ),
    system: bool = typer.Option(
        False,
        "--system", "-s",
        help="Show system information",
    ),
    models: bool = typer.Option(
        False,
        "--models", "-m",
        help="Show available models",
    ),
    adapters: bool = typer.Option(
        False,
        "--adapters", "-a",
        help="Show registered adapters",
    ),
    run: Optional[str] = typer.Option(
        None,
        "--run", "-r",
        help="Show information about a specific run",
    ),
    all: bool = typer.Option(
        False,
        "--all",
        help="Show all available information",
    ),
    json_output: bool = typer.Option(
        False,
        "--json",
        help="Output in JSON format",
    ),
):
    """Display system and configuration information.
    
    This command shows various information about the K-BERT system,
    including configuration, available models, and system capabilities.
    
    Examples:
        # Show all information
        k-bert info --all
        
        # Show current configuration
        k-bert info --config
        
        # Show system information
        k-bert info --system
        
        # Show specific run details
        k-bert info --run run_20250723_120000
    """
    # If no specific option is selected, default to system info
    if not any([config, system, models, adapters, run, all]):
        system = True
    
    # Initialize CLI (lightweight mode for info command)
    try:
        loader = ConfigurationLoader()
        user_config_path = loader.find_user_config()
        project_config_path = loader.find_project_config()
        
        initialize_cli(
            user_config_path=user_config_path,
            project_config_path=project_config_path,
        )
    except Exception as e:
        # Continue even if initialization fails
        logger.warning(f"Failed to initialize CLI: {e}")
    
    info_data = {}
    
    try:
        # Collect system information
        if system or all:
            info_data["system"] = get_system_info()
            if not json_output:
                display_system_info(info_data["system"])
        
        # Collect configuration information
        if config or all:
            info_data["configuration"] = get_configuration_info()
            if not json_output:
                display_configuration_info(info_data["configuration"])
        
        # Collect model information
        if models or all:
            info_data["models"] = get_models_info()
            if not json_output:
                display_models_info(info_data["models"])
        
        # Collect adapter information
        if adapters or all:
            info_data["adapters"] = get_adapters_info()
            if not json_output:
                display_adapters_info(info_data["adapters"])
        
        # Show specific run information
        if run:
            info_data["run"] = get_run_info(run)
            if not json_output:
                display_run_info(info_data["run"], run)
        
        # Output JSON if requested
        if json_output:
            console.print_json(data=info_data)
        
    except Exception as e:
        console.print(f"[red]Error collecting information: {e}[/red]")
        raise typer.Exit(1)
    finally:
        # Ensure cleanup
        shutdown_cli()


def get_system_info() -> dict:
    """Collect system information."""
    import mlx.core as mx
    
    info = {
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "machine": platform.machine(),
            "processor": platform.processor(),
        },
        "python": {
            "version": platform.python_version(),
            "implementation": platform.python_implementation(),
        },
        "mlx": {
            "version": getattr(mx, "__version__", "unknown"),
            "metal_available": hasattr(mx, "metal") and mx.metal.is_available(),
            "default_device": str(mx.default_device()),
        },
        "k_bert": {
            "version": "0.1.0",
            "cli_version": "0.1.0",
        }
    }
    
    # Add memory info if available
    try:
        import psutil
        memory = psutil.virtual_memory()
        info["memory"] = {
            "total_gb": round(memory.total / (1024**3), 2),
            "available_gb": round(memory.available / (1024**3), 2),
            "percent_used": memory.percent,
        }
    except ImportError:
        pass
    
    return info


def get_configuration_info() -> dict:
    """Collect configuration information."""
    loader = ConfigurationLoader()
    
    info = {
        "config_files": {
            "user_config": str(path) if (path := loader.find_user_config()) else None,
            "project_config": str(path) if (path := loader.find_project_config()) else None,
        },
        "current_config": {}
    }
    
    # Load and include current configuration
    try:
        configs = []
        if info["config_files"]["user_config"]:
            configs.append(loader.load_yaml_config(Path(info["config_files"]["user_config"])))
        if info["config_files"]["project_config"]:
            configs.append(loader.load_yaml_config(Path(info["config_files"]["project_config"])))
        
        if configs:
            info["current_config"] = loader.merge_configs(configs)
    except Exception as e:
        info["config_error"] = str(e)
    
    return info


def get_models_info() -> dict:
    """Collect information about available models."""
    info = {
        "supported_architectures": [
            "modernbert",
            "bert",
            "modernbert_with_head",
            "bert_with_lora",
        ],
        "default_models": {
            "modernbert": "answerdotai/ModernBERT-base",
            "bert": "bert-base-uncased",
        },
        "supported_heads": [
            "classification",
            "binary_classification",
            "regression",
            "multi_label_classification",
        ],
    }
    
    # Check for local model checkpoints
    output_dir = Path("output")
    if output_dir.exists():
        checkpoints = []
        for run_dir in output_dir.iterdir():
            if run_dir.is_dir() and run_dir.name.startswith("run_"):
                checkpoint_dir = run_dir / "checkpoints"
                if checkpoint_dir.exists():
                    checkpoints.append(str(run_dir))
        info["local_checkpoints"] = checkpoints
    
    return info


def get_adapters_info() -> dict:
    """Collect information about registered adapters."""
    info = {
        "primary_adapters": ["CLI", "Web API (planned)", "gRPC (planned)"],
        "secondary_adapters": {
            "compute": ["MLX", "PyTorch (planned)"],
            "storage": ["Filesystem", "S3 (planned)", "GCS (planned)"],
            "monitoring": ["Console", "MLflow", "Weights & Biases (planned)"],
            "data": ["MLX DataLoader", "Streaming (planned)"],
            "tokenizer": ["HuggingFace", "Custom (planned)"],
        },
        "active_adapters": {
            "compute": "MLX",
            "storage": "Filesystem",
            "monitoring": "Console + MLflow",
            "data": "MLX DataLoader",
            "tokenizer": "HuggingFace",
        }
    }
    
    return info


def get_run_info(run_id: str) -> dict:
    """Get information about a specific training run."""
    info = {"run_id": run_id}
    
    # Check for run directory
    run_dir = Path("output") / run_id
    if not run_dir.exists():
        # Try with run_ prefix
        run_dir = Path("output") / f"run_{run_id}"
    
    if run_dir.exists():
        info["path"] = str(run_dir)
        
        # Check for various artifacts
        if (run_dir / "training_config.json").exists():
            with open(run_dir / "training_config.json") as f:
                info["config"] = json.load(f)
        
        if (run_dir / "training_result.json").exists():
            with open(run_dir / "training_result.json") as f:
                info["result"] = json.load(f)
        
        if (run_dir / "metrics.jsonl").exists():
            metrics = []
            with open(run_dir / "metrics.jsonl") as f:
                for line in f:
                    metrics.append(json.loads(line))
            info["metrics_history"] = metrics
        
        # Check for checkpoints
        checkpoint_dir = run_dir / "checkpoints"
        if checkpoint_dir.exists():
            checkpoints = list(checkpoint_dir.glob("checkpoint-*"))
            info["checkpoints"] = [str(cp) for cp in checkpoints]
        
        # Check for logs
        if (run_dir / "training.log").exists():
            info["log_file"] = str(run_dir / "training.log")
    else:
        info["error"] = f"Run directory not found: {run_id}"
    
    return info


def display_system_info(info: dict):
    """Display system information."""
    console.print("\n[bold blue]System Information[/bold blue]")
    
    table = Table(show_header=True)
    table.add_column("Component", style="cyan")
    table.add_column("Value", style="green")
    
    # Platform info
    table.add_row("Platform", f"{info['platform']['system']} {info['platform']['release']}")
    table.add_row("Machine", info['platform']['machine'])
    
    # Python info
    table.add_row("Python", info['python']['version'])
    
    # MLX info
    table.add_row("MLX Version", info['mlx']['version'])
    table.add_row("Metal Available", "Yes" if info['mlx']['metal_available'] else "No")
    table.add_row("Default Device", info['mlx']['default_device'])
    
    # Memory info
    if 'memory' in info:
        table.add_row(
            "Memory",
            f"{info['memory']['available_gb']:.1f} / {info['memory']['total_gb']:.1f} GB available"
        )
    
    # K-BERT info
    table.add_row("K-BERT Version", info['k_bert']['version'])
    
    console.print(table)


def display_configuration_info(info: dict):
    """Display configuration information."""
    console.print("\n[bold blue]Configuration[/bold blue]")
    
    # Config files
    table = Table(show_header=True)
    table.add_column("Config Type", style="cyan")
    table.add_column("Path", style="green")
    
    table.add_row(
        "User Config",
        info['config_files']['user_config'] or "[dim]Not found[/dim]"
    )
    table.add_row(
        "Project Config",
        info['config_files']['project_config'] or "[dim]Not found[/dim]"
    )
    
    console.print(table)
    
    # Current configuration
    if info.get('current_config'):
        console.print("\n[bold]Current Configuration:[/bold]")
        # Pretty print as YAML
        import yaml
        config_yaml = yaml.dump(info['current_config'], default_flow_style=False)
        syntax = Syntax(config_yaml, "yaml", theme="monokai", line_numbers=False)
        console.print(Panel(syntax, title="Merged Configuration", border_style="blue"))


def display_models_info(info: dict):
    """Display model information."""
    console.print("\n[bold blue]Model Information[/bold blue]")
    
    # Supported architectures
    console.print("\n[bold]Supported Architectures:[/bold]")
    for arch in info['supported_architectures']:
        console.print(f"  • {arch}")
    
    # Default models
    console.print("\n[bold]Default Models:[/bold]")
    table = Table(show_header=True)
    table.add_column("Architecture", style="cyan")
    table.add_column("Default Model", style="green")
    
    for arch, model in info['default_models'].items():
        table.add_row(arch, model)
    
    console.print(table)
    
    # Local checkpoints
    if info.get('local_checkpoints'):
        console.print(f"\n[bold]Local Checkpoints ({len(info['local_checkpoints'])}):[/bold]")
        for checkpoint in info['local_checkpoints'][:5]:
            console.print(f"  • {checkpoint}")
        if len(info['local_checkpoints']) > 5:
            console.print(f"  [dim]... and {len(info['local_checkpoints']) - 5} more[/dim]")


def display_adapters_info(info: dict):
    """Display adapter information."""
    console.print("\n[bold blue]Adapter Information[/bold blue]")
    
    # Active adapters
    console.print("\n[bold]Active Adapters:[/bold]")
    table = Table(show_header=True)
    table.add_column("Port Type", style="cyan")
    table.add_column("Active Adapter", style="green")
    
    for port_type, adapter in info['active_adapters'].items():
        table.add_row(port_type.title(), adapter)
    
    console.print(table)
    
    # Available adapters
    console.print("\n[bold]Available Adapters:[/bold]")
    for port_type, adapters in info['secondary_adapters'].items():
        console.print(f"\n[cyan]{port_type.title()}:[/cyan]")
        for adapter in adapters:
            if "(planned)" in adapter:
                console.print(f"  • [dim]{adapter}[/dim]")
            else:
                console.print(f"  • {adapter}")


def display_run_info(info: dict, run_id: str):
    """Display run information."""
    console.print(f"\n[bold blue]Run Information: {run_id}[/bold blue]")
    
    if "error" in info:
        console.print(f"[red]{info['error']}[/red]")
        return
    
    # Basic info
    console.print(f"\nPath: [cyan]{info['path']}[/cyan]")
    
    # Training result
    if "result" in info:
        result = info["result"]
        table = Table(title="Training Result", show_header=True)
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")
        
        if "final_train_loss" in result:
            table.add_row("Final Train Loss", f"{result['final_train_loss']:.4f}")
        if "final_val_loss" in result:
            table.add_row("Final Val Loss", f"{result['final_val_loss']:.4f}")
        if "best_val_loss" in result:
            table.add_row("Best Val Loss", f"{result['best_val_loss']:.4f}")
        if "total_epochs" in result:
            table.add_row("Total Epochs", str(result['total_epochs']))
        if "training_time_seconds" in result:
            table.add_row("Training Time", f"{result['training_time_seconds']:.1f}s")
        
        console.print(table)
    
    # Checkpoints
    if "checkpoints" in info:
        console.print(f"\n[bold]Checkpoints ({len(info['checkpoints'])}):[/bold]")
        for checkpoint in info['checkpoints'][:3]:
            console.print(f"  • {Path(checkpoint).name}")
        if len(info['checkpoints']) > 3:
            console.print(f"  [dim]... and {len(info['checkpoints']) - 3} more[/dim]")
    
    # Log file
    if "log_file" in info:
        console.print(f"\nLog file: [cyan]{info['log_file']}[/cyan]")


if __name__ == "__main__":
    typer.run(info)