"""Thin CLI adapter for evaluate command.

This adapter is responsible only for:
1. Parsing command-line arguments
2. Creating the EvaluationRequestDTO
3. Calling the EvaluateModelUseCase
4. Formatting and displaying the response

No business logic should exist in this adapter.
"""

from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn
from loguru import logger

from application.dto.evaluation import EvaluationRequestDTO
from application.use_cases.evaluate_model import EvaluateModelUseCase
from infrastructure.bootstrap import get_service
from ports.secondary.configuration import ConfigurationProvider


console = Console()


def create_evaluation_request_dto(
    config_provider: ConfigurationProvider,
    model: Optional[Path],
    checkpoint: Optional[Path],
    data: Optional[Path],
    split: str,
    config: Optional[Path],
    batch_size: Optional[int],
    output_dir: Optional[Path],
    metrics: Optional[str],
    no_save: bool,
) -> EvaluationRequestDTO:
    """Create EvaluationRequestDTO from CLI arguments and configuration."""
    # Load configuration file if specified
    if config:
        config_provider.load_file(str(config))
    else:
        # Look for k-bert.yaml in current directory
        config_paths = [
            Path.cwd() / "k-bert.yaml",
            Path.cwd() / "k-bert.yml",
            Path.cwd() / ".k-bert.yaml",
        ]
        
        config_file = next((p for p in config_paths if p.exists()), None)
        if config_file:
            config_provider.load_file(str(config_file))
    
    # Apply CLI overrides
    if model:
        config_provider.set("models.model_path", str(model))
    if checkpoint:
        config_provider.set("training.resume_from_checkpoint", str(checkpoint))
    if data:
        config_provider.set(f"data.{split}_path", str(data))
    if batch_size is not None:
        config_provider.set("evaluation.batch_size", batch_size)
    if output_dir:
        config_provider.set("evaluation.output_dir", str(output_dir))
    if metrics:
        config_provider.set("evaluation.metrics", metrics.split(","))
    if no_save:
        config_provider.set("evaluation.save_predictions", False)
        config_provider.set("evaluation.save_confusion_matrix", False)
    
    # Determine model/checkpoint path
    model_path = None
    checkpoint_path = None
    
    if model_path_str := config_provider.get("models.model_path"):
        model_path = Path(model_path_str)
    elif checkpoint_path_str := config_provider.get("training.resume_from_checkpoint"):
        checkpoint_path = Path(checkpoint_path_str)
    else:
        raise ValueError("No model or checkpoint path specified")
    
    # Determine data path based on split
    data_path = None
    if split == "train":
        data_path = config_provider.get("data.train_path")
    elif split in ["val", "validation"]:
        data_path = config_provider.get("data.val_path")
    elif split == "test":
        data_path = config_provider.get("data.test_path")
    
    if not data_path:
        raise ValueError(f"No data path configured for split '{split}'")
    
    # Create DTO
    return EvaluationRequestDTO(
        model_path=model_path,
        checkpoint_path=checkpoint_path,
        data_path=Path(data_path),
        batch_size=config_provider.get("evaluation.batch_size", 
                                     config_provider.get("data.batch_size", 32)),
        metrics=config_provider.get("evaluation.metrics", 
                                  ["accuracy", "precision", "recall", "f1"]),
        num_workers=config_provider.get("data.num_workers", 0),
        output_dir=Path(config_provider.get("evaluation.output_dir", "output/evaluation")),
        save_predictions=config_provider.get("evaluation.save_predictions", True),
        save_confusion_matrix=config_provider.get("evaluation.save_confusion_matrix", True),
        use_mixed_precision=config_provider.get("evaluation.use_mixed_precision", False),
        device=config_provider.get("evaluation.device"),
    )


def display_configuration_summary(request: EvaluationRequestDTO) -> None:
    """Display evaluation configuration in a formatted table."""
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


def display_evaluation_results(response) -> None:
    """Display evaluation results in a formatted manner."""
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


async def evaluate_command(
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
    
    This is a thin CLI adapter that:
    1. Parses arguments
    2. Creates EvaluationRequestDTO
    3. Calls EvaluateModelUseCase
    4. Displays results
    
    All business logic is handled by the use case.
    
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
        # Get configuration provider
        config_provider = get_service(ConfigurationProvider)
        
        # Create request DTO
        request = create_evaluation_request_dto(
            config_provider=config_provider,
            model=model,
            checkpoint=checkpoint,
            data=data,
            split=split,
            config=config,
            batch_size=batch_size,
            output_dir=output_dir,
            metrics=metrics,
            no_save=no_save,
        )
        
        # Display configuration
        display_configuration_summary(request)
        
        # Get use case
        use_case = get_service(EvaluateModelUseCase)
        
        # Run evaluation with progress indicator
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
            transient=True,
        ) as progress:
            task = progress.add_task("Evaluating model...", total=None)
            response = await use_case.execute(request)
        
        # Display results
        display_evaluation_results(response)
        
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


# Create the Typer command
evaluate = evaluate_command