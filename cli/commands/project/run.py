"""Run k-bert project command."""

from pathlib import Path
from typing import Optional
import os
import sys

import typer
from rich.console import Console

from ...config import ConfigManager, get_config
from ...plugins import load_project_plugins
from ...utils import handle_errors


console = Console()


@handle_errors
def run_command(
    config: Optional[Path] = typer.Option(
        None,
        "--config",
        "-c",
        help="Project configuration file (defaults to k-bert.yaml)",
    ),
    experiment: Optional[str] = typer.Option(
        None,
        "--experiment",
        "-e",
        help="Run specific experiment from configuration",
    ),
    train_data: Optional[Path] = typer.Option(
        None,
        "--train",
        help="Override training data path",
    ),
    val_data: Optional[Path] = typer.Option(
        None,
        "--val",
        help="Override validation data path",
    ),
    test_data: Optional[Path] = typer.Option(
        None,
        "--test",
        help="Override test data path",
    ),
    output_dir: Optional[Path] = typer.Option(
        None,
        "--output",
        "-o",
        help="Override output directory",
    ),
    reload_plugins: bool = typer.Option(
        False,
        "--reload-plugins",
        help="Force reload project plugins",
    ),
    debug: bool = typer.Option(
        False,
        "--debug",
        "-d",
        help="Enable debug mode",
    ),
):
    """Run k-bert project with configuration.
    
    This command loads the project configuration, custom components,
    and runs training with all specified settings. It automatically
    discovers and loads plugins from the project structure.
    
    Examples:
        # Run with default configuration
        k-bert run
        
        # Run specific experiment
        k-bert run --experiment full_pipeline
        
        # Override data paths
        k-bert run --train custom_train.csv --val custom_val.csv
        
        # Use different config file
        k-bert run --config configs/experiments/advanced.yaml
    """
    # Find project root (directory containing k-bert.yaml)
    project_root = _find_project_root()
    if not project_root:
        console.print(
            "[red]No k-bert project found.[/red]\n"
            "Run this command from a k-bert project directory or use "
            "'k-bert project init' to create one."
        )
        raise typer.Exit(1)
    
    console.print(f"[cyan]Running k-bert project from: {project_root}[/cyan]")
    
    # Load project plugins
    console.print("[dim]Loading project plugins...[/dim]")
    plugin_results = load_project_plugins(project_root, override=reload_plugins)
    
    if plugin_results:
        total_plugins = sum(plugin_results.values())
        console.print(f"[green]✓ Loaded {total_plugins} plugins[/green]")
        
        if debug:
            for path, count in plugin_results.items():
                console.print(f"  • {path}: {count} components")
    
    # Load configuration
    config_manager = ConfigManager()
    
    # Determine config file
    if config is None:
        config = project_root / "k-bert.yaml"
        if not config.exists():
            console.print(
                "[red]No k-bert.yaml found in project.[/red]\n"
                "Create one or specify with --config"
            )
            raise typer.Exit(1)
    
    # Load project configuration
    project_config = config_manager.load_project_config(config)
    if not project_config:
        console.print(f"[red]Failed to load configuration from {config}[/red]")
        raise typer.Exit(1)
    
    # Handle experiment selection
    if experiment:
        if not project_config.experiments:
            console.print("[red]No experiments defined in configuration.[/red]")
            raise typer.Exit(1)
        
        # Find experiment
        exp_config = None
        for exp in project_config.experiments:
            if exp.get("name") == experiment:
                exp_config = exp
                break
        
        if not exp_config:
            available = [exp.get("name", "unnamed") for exp in project_config.experiments]
            console.print(
                f"[red]Experiment '{experiment}' not found.[/red]\n"
                f"Available experiments: {', '.join(available)}"
            )
            raise typer.Exit(1)
        
        console.print(f"[cyan]Running experiment: {experiment}[/cyan]")
        if "description" in exp_config:
            console.print(f"[dim]{exp_config['description']}[/dim]")
    
    # Build training configuration
    merged_config = config_manager.get_merged_config(
        project_path=config,
        validate=True
    )
    
    # Apply experiment overrides
    if experiment and exp_config and "config" in exp_config:
        import copy
        exp_overrides = copy.deepcopy(exp_config["config"])
        merged_config = merged_config.merge(exp_overrides)
    
    # Apply CLI overrides
    cli_overrides = {}
    
    # Data paths
    if train_data:
        cli_overrides.setdefault("data", {})["train_path"] = str(train_data)
    if val_data:
        cli_overrides.setdefault("data", {})["val_path"] = str(val_data)
    if test_data:
        cli_overrides.setdefault("data", {})["test_path"] = str(test_data)
    
    # Output directory
    if output_dir:
        cli_overrides.setdefault("training", {})["output_dir"] = str(output_dir)
    
    if cli_overrides:
        merged_config = merged_config.merge(cli_overrides)
    
    # Get data paths from configuration
    data_paths = _get_data_paths(project_config, merged_config, project_root)
    
    if not data_paths.get("train"):
        console.print(
            "[red]No training data specified.[/red]\n"
            "Specify in configuration or use --train"
        )
        raise typer.Exit(1)
    
    # Import and run training
    try:
        from ...commands.core.train import train_command
        
        # Prepare arguments for train command
        train_args = {
            "train_data": Path(data_paths["train"]),
            "val_data": Path(data_paths["val"]) if data_paths.get("val") else None,
            "test_data": Path(data_paths["test"]) if data_paths.get("test") else None,
            "epochs": merged_config.training.default_epochs,
            "batch_size": merged_config.training.default_batch_size,
            "learning_rate": merged_config.training.default_learning_rate,
            "output_dir": Path(merged_config.training.output_dir),
            "seed": merged_config.training.seed,
            "mixed_precision": merged_config.training.mixed_precision,
            "gradient_clip": merged_config.training.max_grad_norm,
            "early_stopping_patience": merged_config.training.early_stopping_patience,
            "save_best_only": merged_config.training.save_best_only,
            "warmup_ratio": merged_config.training.warmup_ratio,
            "gradient_accumulation": merged_config.training.gradient_accumulation_steps,
            "label_smoothing": getattr(merged_config.training, "label_smoothing", 0.0),
            "debug": debug,
        }
        
        # Add custom component settings
        if hasattr(merged_config.training, "use_custom_head") and merged_config.training.use_custom_head:
            # Custom head will be loaded from plugins
            train_args["model_type"] = "custom"
        
        # Add MLX settings
        train_args["max_length"] = merged_config.data.max_length
        train_args["workers"] = merged_config.data.num_workers
        train_args["prefetch_size"] = merged_config.data.prefetch_size
        train_args["use_pretokenized"] = merged_config.data.use_pretokenized
        
        # Add model settings
        train_args["model_name"] = merged_config.models.default_model
        train_args["use_mlx_embeddings"] = merged_config.models.use_mlx_embeddings
        train_args["use_lora"] = merged_config.models.use_lora
        
        # Disable MLflow if not configured
        if not merged_config.mlflow.auto_log:
            train_args["disable_mlflow"] = True
        else:
            train_args["experiment_name"] = merged_config.mlflow.default_experiment
            
            # Add run name
            run_name = f"{project_config.name}"
            if experiment:
                run_name += f"_{experiment}"
            train_args["run_name"] = run_name
        
        # Show configuration summary
        console.print("\n[bold]Training Configuration:[/bold]")
        console.print(f"  Model: {merged_config.models.default_model}")
        console.print(f"  Epochs: {merged_config.training.default_epochs}")
        console.print(f"  Batch size: {merged_config.training.default_batch_size}")
        console.print(f"  Learning rate: {merged_config.training.default_learning_rate}")
        console.print(f"  Training data: {data_paths['train']}")
        
        if plugin_results:
            console.print(f"  Custom components: {total_plugins} loaded")
        
        console.print()
        
        # Run training
        train_command(**train_args)
        
    except Exception as e:
        console.print(f"[red]Training failed: {e}[/red]")
        if debug:
            import traceback
            traceback.print_exc()
        raise typer.Exit(1)


def _find_project_root(start_path: Optional[Path] = None) -> Optional[Path]:
    """Find project root by looking for k-bert.yaml."""
    current = Path(start_path or os.getcwd())
    
    # Check current and parent directories
    for _ in range(5):  # Max 5 levels up
        if (current / "k-bert.yaml").exists():
            return current
        
        if current.parent == current:
            break
        
        current = current.parent
    
    return None


def _get_data_paths(
    project_config,
    merged_config,
    project_root: Path
) -> dict:
    """Extract data paths from configuration."""
    paths = {}
    
    # Check project config first
    if hasattr(project_config, "data"):
        data_config = project_config.data
        if isinstance(data_config, dict):
            for key in ["train_path", "val_path", "test_path"]:
                if key in data_config:
                    path = Path(data_config[key])
                    if not path.is_absolute():
                        path = project_root / path
                    paths[key.replace("_path", "")] = str(path)
    
    # Check merged config
    if hasattr(merged_config, "data"):
        for key in ["train_path", "val_path", "test_path"]:
            value = getattr(merged_config.data, key, None)
            if value:
                path = Path(value)
                if not path.is_absolute():
                    path = project_root / path
                paths[key.replace("_path", "")] = str(path)
    
    return paths