"""Training command implementation with config-first approach."""

import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import typer
from loguru import logger
from rich.console import Console

from ...utils import (
    handle_errors,
    track_time,
    print_error,
    print_success,
    print_info,
)
from ...config import ConfigManager
from ...plugins import ComponentRegistry, load_project_plugins

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

console = Console()


@handle_errors
@track_time("Training BERT model")
def train_command(
    # Config-first approach: config is optional but encouraged
    config: Optional[Path] = typer.Option(
        None,
        "--config",
        "-c",
        help="Configuration file (defaults to k-bert.yaml in current directory)",
    ),
    experiment: Optional[str] = typer.Option(
        None,
        "--experiment",
        "-e",
        help="Run specific experiment from configuration",
    ),
    # Minimal overrides for common use cases
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
    epochs: Optional[int] = typer.Option(
        None,
        "--epochs",
        help="Override number of epochs",
    ),
    output_dir: Optional[Path] = typer.Option(
        None,
        "--output",
        "-o",
        help="Override output directory",
    ),
    # Control flags
    no_config: bool = typer.Option(
        False,
        "--no-config",
        help="Run with default settings (no configuration file)",
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Show configuration without running training",
    ),
    debug: bool = typer.Option(
        False,
        "--debug",
        "-d",
        help="Enable debug logging",
    ),
):
    """Train a BERT model using configuration file.

    This command uses a configuration-first approach. By default, it looks for
    k-bert.yaml in the current directory or uses the specified config file.
    
    Examples:
        # Train with project configuration
        k-bert train
        
        # Train with specific experiment
        k-bert train --experiment titanic_baseline
        
        # Override specific settings
        k-bert train --epochs 10 --train custom_train.csv
        
        # Run without config (uses defaults)
        k-bert train --no-config --train data/train.csv --val data/val.csv
        
        # Dry run to see configuration
        k-bert train --dry-run
    """
    # Configure logging
    log_level = "DEBUG" if debug else "INFO"
    logger.remove()
    logger.add(sys.stderr, level=log_level, enqueue=False)
    
    console.print("\n[bold blue]K-BERT Training[/bold blue]")
    console.print("=" * 60)
    
    # Handle no-config mode
    if no_config:
        if not train_data:
            print_error(
                "Training data is required when using --no-config",
                title="Missing Required Data"
            )
            raise typer.Exit(1)
        
        # Use minimal defaults
        config_overrides = {
            'data': {
                'train_path': str(train_data),
                'val_path': str(val_data) if val_data else None,
                'batch_size': 32,
                'max_length': 256,
            },
            'training': {
                'default_epochs': epochs or 5,
                'default_batch_size': 32,
                'default_learning_rate': 2e-5,
                'output_dir': str(output_dir or Path('output')),
            },
            'models': {
                'default_model': 'answerdotai/ModernBERT-base',
                'head': {'type': 'binary_classification'},
            }
        }
        
        console.print("[yellow]Running with default configuration (--no-config)[/yellow]")
        merged_config = ConfigManager().get_merged_config(cli_overrides=config_overrides)
    else:
        # Config-first approach
        # Find configuration file
        if config is None:
            # Look for k-bert.yaml in current directory
            config_paths = [
                Path.cwd() / "k-bert.yaml",
                Path.cwd() / "k-bert.yml",
                Path.cwd() / ".k-bert.yaml",
            ]
            
            config = next((p for p in config_paths if p.exists()), None)
            
            if config is None:
                print_error(
                    "No configuration file found. Create one with 'k-bert config init' "
                    "or use --no-config to run with defaults.",
                    title="Configuration Required"
                )
                console.print("\n[dim]Looked for:[/dim]")
                for p in config_paths:
                    console.print(f"  • {p}")
                raise typer.Exit(1)
        
        console.print(f"[green]Using configuration: {config}[/green]")
        
        # Load project plugins if in a project directory
        project_root = config.parent
        if (project_root / "src").exists():
            console.print("[dim]Loading project plugins...[/dim]")
            plugin_results = load_project_plugins(project_root)
            if plugin_results:
                total = sum(plugin_results.values())
                console.print(f"[green]✓ Loaded {total} custom components[/green]")
        
        # Build CLI overrides
        cli_overrides = {}
        if train_data:
            cli_overrides.setdefault('data', {})['train_path'] = str(train_data)
        if val_data:
            cli_overrides.setdefault('data', {})['val_path'] = str(val_data)
        if epochs is not None:
            cli_overrides.setdefault('training', {})['default_epochs'] = epochs
        if output_dir:
            cli_overrides.setdefault('training', {})['output_dir'] = str(output_dir)
        
        # Load and merge configuration
        config_manager = ConfigManager()
        
        # Handle experiment selection
        if experiment:
            # Load project config to get experiments
            project_config = config_manager.load_project_config(config)
            if not project_config or not project_config.experiments:
                print_error(
                    f"No experiments found in {config}",
                    title="Experiment Not Found"
                )
                raise typer.Exit(1)
            
            # Find experiment
            exp_config = None
            for exp in project_config.experiments:
                if exp.get('name') == experiment:
                    exp_config = exp.get('config', {})
                    break
            
            if exp_config is None:
                available = [e.get('name', 'unnamed') for e in project_config.experiments]
                print_error(
                    f"Experiment '{experiment}' not found. Available: {', '.join(available)}",
                    title="Experiment Not Found"
                )
                raise typer.Exit(1)
            
            console.print(f"[cyan]Running experiment: {experiment}[/cyan]")
            
            # Merge experiment config with CLI overrides
            import copy
            exp_overrides = copy.deepcopy(exp_config)
            for key, value in cli_overrides.items():
                if key in exp_overrides:
                    exp_overrides[key].update(value)
                else:
                    exp_overrides[key] = value
            cli_overrides = exp_overrides
        
        merged_config = config_manager.get_merged_config(
            cli_overrides=cli_overrides,
            project_path=config,
            validate=True
        )
    
    # Validate required paths
    if not merged_config.data.train_path:
        print_error(
            "Training data path not specified in configuration",
            title="Missing Training Data"
        )
        raise typer.Exit(1)
    
    # Display configuration summary
    console.print("\n[bold]Configuration Summary:[/bold]")
    console.print(f"  Model: {merged_config.models.default_model}")
    console.print(f"  Training data: {merged_config.data.train_path}")
    if merged_config.data.val_path:
        console.print(f"  Validation data: {merged_config.data.val_path}")
    console.print(f"  Epochs: {merged_config.training.default_epochs}")
    console.print(f"  Batch size: {merged_config.training.default_batch_size}")
    console.print(f"  Learning rate: {merged_config.training.default_learning_rate}")
    console.print(f"  Output: {merged_config.training.output_dir}")
    
    # Check for custom components
    registry = ComponentRegistry.get_registry()
    if registry.get('heads'):
        console.print(f"  Custom heads: {', '.join(registry['heads'].keys())}")
    if registry.get('augmenters'):
        console.print(f"  Custom augmenters: {', '.join(registry['augmenters'].keys())}")
    
    if dry_run:
        console.print("\n[yellow]Dry run mode - no training performed[/yellow]")
        
        # Show full configuration if debug
        if debug:
            console.print("\n[bold]Full Configuration:[/bold]")
            import json
            console.print(json.dumps(merged_config.model_dump(), indent=2))
        
        print_success("Configuration validated successfully", title="Dry Run Complete")
        return
    
    # Create output directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{experiment}_{timestamp}" if experiment else f"run_{timestamp}"
    run_dir = Path(merged_config.training.output_dir) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    
    # Set up logging to file
    from utils.logging_utils import add_file_logger
    
    log_file = run_dir / "training.log"
    add_file_logger(
        file_path=log_file,
        level=log_level,
        rotation="500 MB",
        retention="30 days",
        compression="zip"
    )
    
    logger.info(f"Starting training run: {run_name}")
    logger.info(f"Configuration: {config or 'defaults'}")
    logger.info(f"Output directory: {run_dir}")
    
    # Import training components
    try:
        from transformers import AutoTokenizer
        from data.factory import create_dataloader
        from models.factory import create_model
        from training.core.base import BaseTrainer
        from training.core.config import BaseTrainerConfig
        
    except ImportError as e:
        print_error(
            f"Failed to import training components: {str(e)}",
            title="Import Error"
        )
        raise typer.Exit(1)
    
    # Convert merged config to training config
    config_dict = merged_config.model_dump()
    
    # Create training configuration
    training_config = BaseTrainerConfig.from_dict({
        'optimizer': {
            'learning_rate': merged_config.training.default_learning_rate,
            'weight_decay': merged_config.training.weight_decay,
            'max_grad_norm': merged_config.training.max_grad_norm,
        },
        'training': {
            'num_epochs': merged_config.training.default_epochs,
            'gradient_accumulation_steps': merged_config.training.gradient_accumulation_steps,
            'mixed_precision': merged_config.training.mixed_precision,
            'early_stopping_patience': merged_config.training.early_stopping_patience,
            'label_smoothing': merged_config.training.label_smoothing,
            'save_steps': merged_config.training.save_steps,
            'eval_steps': merged_config.training.eval_steps,
            'logging_steps': merged_config.training.logging_steps,
            'warmup_ratio': merged_config.training.warmup_ratio,
        },
        'data': {
            'batch_size': merged_config.data.batch_size,
            'eval_batch_size': merged_config.data.eval_batch_size or merged_config.data.batch_size * 2,
        },
        'environment': {
            'output_dir': run_dir,
            'experiment_name': merged_config.mlflow.default_experiment if merged_config.mlflow.auto_log else None,
            'run_name': run_name,
            'seed': merged_config.training.seed,
        }
    })
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(merged_config.models.default_model)
    
    # Create data loaders
    console.print("\n[dim]Loading data...[/dim]")
    
    train_loader = create_dataloader(
        data_path=Path(merged_config.data.train_path),
        batch_size=merged_config.data.batch_size,
        shuffle=True,
        num_workers=merged_config.data.num_workers,
        prefetch_size=merged_config.data.prefetch_size,
        tokenizer=tokenizer,
        max_length=merged_config.data.max_length,
        split="train",
        use_pretokenized=merged_config.data.use_pretokenized,
    )
    
    val_loader = None
    if merged_config.data.val_path:
        val_loader = create_dataloader(
            data_path=Path(merged_config.data.val_path),
            batch_size=merged_config.data.eval_batch_size or merged_config.data.batch_size * 2,
            shuffle=False,
            num_workers=merged_config.data.num_workers,
            prefetch_size=merged_config.data.prefetch_size,
            tokenizer=tokenizer,
            max_length=merged_config.data.max_length,
            split="val",
            use_pretokenized=merged_config.data.use_pretokenized,
        )
    
    # Create model
    console.print("[dim]Creating model...[/dim]")
    
    # Check for custom head
    head_type = None
    custom_head = None
    if hasattr(merged_config.models, 'head') and merged_config.models.head:
        head_type = merged_config.models.head.type
        if head_type in registry.get('heads', {}):
            logger.info(f"Using custom head: {head_type}")
            head_class = ComponentRegistry.get_component('heads', head_type)
            head_config = merged_config.models.head.config if hasattr(merged_config.models.head, 'config') else {}
            custom_head = head_class(config=head_config)
    
    # Create model
    if custom_head:
        model = create_model(
            model_name=merged_config.models.default_model,
            model_type="modernbert",
            custom_head=custom_head
        )
    else:
        model = create_model(
            model_name=merged_config.models.default_model,
            model_type="modernbert_with_head",
            head_type=head_type or "binary_classification",
            num_labels=2  # TODO: Get from config
        )
    
    # Create trainer
    trainer = BaseTrainer(
        model=model,
        config=training_config,
    )
    
    # Save configuration
    config_dict['timestamp'] = timestamp
    config_dict['run_name'] = run_name
    
    with open(run_dir / "training_config.json", "w") as f:
        import json
        json.dump(config_dict, f, indent=2)
    
    # Start training
    console.print("\n[bold green]Starting training...[/bold green]\n")
    
    try:
        metrics = trainer.train(train_loader, val_loader)
        
        # Save final model
        final_model_path = run_dir / "final_model"
        trainer.save_checkpoint(final_model_path)
        
        print_success(
            f"Training complete!\n"
            f"Model saved to: {final_model_path}\n"
            f"Logs: {log_file}",
            title="Training Complete"
        )
        
        # Show next steps
        console.print("\n[bold]Next steps:[/bold]")
        console.print(f"  • Generate predictions: [cyan]k-bert predict --checkpoint {final_model_path} --test data/test.csv[/cyan]")
        console.print(f"  • View logs: [cyan]cat {log_file}[/cyan]")
        if merged_config.mlflow.auto_log:
            console.print(f"  • View in MLflow: [cyan]mlflow ui[/cyan]")
        
    except KeyboardInterrupt:
        logger.error("Training interrupted by user")
        raise typer.Exit(130)
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        if debug:
            import traceback
            traceback.print_exc()
        raise typer.Exit(1)