"""Evaluate command for the CLI.

This is a thin adapter that converts CLI arguments to DTOs and delegates
to the application layer EvaluateModelCommand.
"""

from pathlib import Path
from typing import Optional
import asyncio

import typer
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn
from loguru import logger

from cli.bootstrap import initialize_cli, get_command, shutdown_cli
from cli.config.loader import ConfigurationLoader
from application.commands.evaluate import EvaluateModelCommand
from application.dto.evaluation import EvaluationRequestDTO


console = Console()


def create_evaluation_request(
    config: dict,
    split: str = "test",
) -> EvaluationRequestDTO:
    """Create EvaluationRequestDTO from configuration.
    
    Args:
        config: Configuration dictionary
        split: Data split to evaluate on
        
    Returns:
        Evaluation request DTO
    """
    # Extract configuration
    model_config = config.get("models", {})
    data_config = config.get("data", {})
    eval_config = config.get("evaluation", {})
    
    # Determine model path
    model_path = None
    checkpoint_path = None
    
    if model_config.get("model_path"):
        model_path = Path(model_config["model_path"])
    elif config.get("training", {}).get("resume_from_checkpoint"):
        checkpoint_path = Path(config["training"]["resume_from_checkpoint"])
    
    # Determine data path based on split
    data_path = None
    if split == "train":
        data_path = data_config.get("train_path")
    elif split == "val" or split == "validation":
        data_path = data_config.get("val_path")
    elif split == "test":
        data_path = data_config.get("test_path")
    
    if not data_path:
        raise ValueError(f"No data path configured for split '{split}'")
    
    # Create the DTO
    return EvaluationRequestDTO(
        model_path=model_path,
        checkpoint_path=checkpoint_path,
        data_path=Path(data_path),
        batch_size=eval_config.get("batch_size", data_config.get("batch_size", 32)),
        metrics=eval_config.get("metrics", ["accuracy", "precision", "recall", "f1"]),
        num_workers=data_config.get("num_workers", 0),
        output_dir=Path(eval_config.get("output_dir", "output/evaluation")),
        save_predictions=eval_config.get("save_predictions", True),
        save_confusion_matrix=eval_config.get("save_confusion_matrix", True),
        use_mixed_precision=eval_config.get("use_mixed_precision", False),
        device=eval_config.get("device"),
    )


def display_configuration(request: EvaluationRequestDTO) -> None:
    """Display evaluation configuration in a table."""
    table = Table(title="Evaluation Configuration", show_header=True)
    table.add_column("Parameter", style="cyan")
    table.add_column("Value", style="green")
    
    # Model info
    if request.model_path:
        table.add_row("Model Path", str(request.model_path))
    elif request.checkpoint_path:
        table.add_row("Checkpoint Path", str(request.checkpoint_path))
    
    # Data info
    table.add_row("Data Path", str(request.data_path))
    table.add_row("Batch Size", str(request.batch_size))
    
    # Metrics
    table.add_row("Metrics", ", ".join(request.metrics))
    
    # Output
    table.add_row("Output Directory", str(request.output_dir))
    table.add_row("Save Predictions", "Yes" if request.save_predictions else "No")
    table.add_row("Save Confusion Matrix", "Yes" if request.save_confusion_matrix else "No")
    
    console.print(table)


def display_results(response) -> None:
    """Display evaluation results."""
    if not response.success:
        console.print(f"\n[red]Evaluation failed: {response.error_message}[/red]")
        return
    
    console.print("\n[bold green]Evaluation completed successfully![/bold green]\n")
    
    # Metrics table
    table = Table(title="Evaluation Metrics", show_header=True)
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    
    # Display metrics
    for metric_name, value in response.metrics.items():
        if isinstance(value, float):
            table.add_row(metric_name.title(), f"{value:.4f}")
        else:
            table.add_row(metric_name.title(), str(value))
    
    console.print(table)
    
    # Additional info
    console.print(f"\nTotal Samples: {response.total_samples}")
    console.print(f"Evaluation Time: {response.evaluation_time_seconds:.1f}s")
    
    # Output files
    if response.predictions_path:
        console.print(f"\nPredictions saved to: [cyan]{response.predictions_path}[/cyan]")
    if response.confusion_matrix_path:
        console.print(f"Confusion matrix saved to: [cyan]{response.confusion_matrix_path}[/cyan]")
    if response.metrics_path:
        console.print(f"Metrics saved to: [cyan]{response.metrics_path}[/cyan]")
    
    # Class-specific metrics if available
    if response.per_class_metrics:
        console.print("\n[bold]Per-Class Metrics:[/bold]")
        class_table = Table(show_header=True)
        class_table.add_column("Class", style="cyan")
        class_table.add_column("Precision", style="green")
        class_table.add_column("Recall", style="green")
        class_table.add_column("F1-Score", style="green")
        class_table.add_column("Support", style="yellow")
        
        for class_name, metrics in response.per_class_metrics.items():
            class_table.add_row(
                str(class_name),
                f"{metrics.get('precision', 0):.3f}",
                f"{metrics.get('recall', 0):.3f}",
                f"{metrics.get('f1-score', 0):.3f}",
                str(metrics.get('support', 0))
            )
        
        console.print(class_table)


def evaluate(
    model: Optional[Path] = typer.Option(
        None,
        "--model", "-m",
        help="Path to model directory or checkpoint",
    ),
    checkpoint: Optional[Path] = typer.Option(
        None,
        "--checkpoint", "-c",
        help="Path to checkpoint file",
    ),
    data: Optional[Path] = typer.Option(
        None,
        "--data", "-d",
        help="Override data path",
    ),
    split: str = typer.Option(
        "test",
        "--split", "-s",
        help="Data split to evaluate (train/val/test)",
    ),
    config: Optional[Path] = typer.Option(
        None,
        "--config",
        help="Configuration file",
    ),
    batch_size: Optional[int] = typer.Option(
        None,
        "--batch-size", "-b",
        help="Override batch size",
    ),
    output_dir: Optional[Path] = typer.Option(
        None,
        "--output", "-o",
        help="Override output directory",
    ),
    metrics: Optional[str] = typer.Option(
        None,
        "--metrics",
        help="Comma-separated list of metrics to compute",
    ),
    no_save: bool = typer.Option(
        False,
        "--no-save",
        help="Don't save predictions and confusion matrix",
    ),
    debug: bool = typer.Option(
        False,
        "--debug",
        help="Enable debug logging",
    ),
):
    """Evaluate a trained BERT model.
    
    This command evaluates a trained model on a dataset and computes metrics.
    
    Examples:
        # Evaluate on test set
        k-bert evaluate --model output/best_model
        
        # Evaluate checkpoint on validation set
        k-bert evaluate --checkpoint output/checkpoint-1000 --split val
        
        # Evaluate with specific metrics
        k-bert evaluate --model output/best_model --metrics accuracy,f1,confusion_matrix
    """
    console.print("\n[bold blue]K-BERT Model Evaluation[/bold blue]")
    console.print("=" * 60)
    
    try:
        # Load configuration
        loader = ConfigurationLoader()
        configs = []
        
        # Load user config
        user_config_path = loader.find_user_config()
        if user_config_path:
            configs.append(loader.load_yaml_config(user_config_path))
        
        # Load project config
        project_config_path = loader.find_project_config()
        if project_config_path:
            configs.append(loader.load_yaml_config(project_config_path))
        
        # Load command config
        if config:
            configs.append(loader.load_yaml_config(config))
        
        # Merge configurations
        merged_config = loader.merge_configs(configs) if configs else {}
        
        # Apply CLI overrides
        if model:
            merged_config.setdefault("models", {})["model_path"] = str(model)
        if checkpoint:
            merged_config.setdefault("training", {})["resume_from_checkpoint"] = str(checkpoint)
        if data:
            merged_config.setdefault("data", {})[f"{split}_path"] = str(data)
        if batch_size:
            merged_config.setdefault("evaluation", {})["batch_size"] = batch_size
        if output_dir:
            merged_config.setdefault("evaluation", {})["output_dir"] = str(output_dir)
        if metrics:
            merged_config.setdefault("evaluation", {})["metrics"] = metrics.split(",")
        if no_save:
            merged_config.setdefault("evaluation", {})["save_predictions"] = False
            merged_config.setdefault("evaluation", {})["save_confusion_matrix"] = False
        
        # Validate configuration
        errors = loader.validate_config(merged_config, "evaluate")
        if errors:
            console.print("[red]Configuration errors:[/red]")
            for error in errors:
                console.print(f"  • {error}")
            raise typer.Exit(1)
        
        # Initialize CLI
        initialize_cli(
            config_path=config,
            user_config_path=user_config_path,
            project_config_path=project_config_path,
        )
        
        # Create evaluation request
        request = create_evaluation_request(merged_config, split)
        
        # Display configuration
        display_configuration(request)
        
        # Get the evaluation command
        eval_command = get_command(EvaluateModelCommand)
        
        # Run evaluation with progress indicator
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
            transient=True,
        ) as progress:
            task = progress.add_task("Evaluating model...", total=None)
            
            # Run the async command
            response = asyncio.run(eval_command.execute(request))
        
        # Display results
        display_results(response)
        
    except ValueError as e:
        console.print(f"\n[red]Configuration error: {e}[/red]")
        raise typer.Exit(1)
    except KeyboardInterrupt:
        console.print("\n[yellow]Evaluation interrupted by user[/yellow]")
        raise typer.Exit(130)
    except Exception as e:
        console.print(f"\n[red]Error: {e}[/red]")
        if debug:
            import traceback
            traceback.print_exc()
        raise typer.Exit(1)
    finally:
        # Ensure cleanup
        shutdown_cli()


if __name__ == "__main__":
    typer.run(evaluate)