"""Predict command for the CLI.

This is a thin adapter that converts CLI arguments to DTOs and delegates
to the application layer PredictCommand.
"""

from pathlib import Path
from typing import Optional
import asyncio

import typer
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from loguru import logger

from cli.bootstrap import initialize_cli, get_command, shutdown_cli
from cli.config.loader import ConfigurationLoader
from application.commands.predict import PredictCommand
from application.dto.prediction import PredictionRequestDTO


console = Console()


def create_prediction_request(
    config: dict,
    output_path: Optional[Path] = None,
) -> PredictionRequestDTO:
    """Create PredictionRequestDTO from configuration.
    
    Args:
        config: Configuration dictionary
        output_path: Override output path
        
    Returns:
        Prediction request DTO
    """
    # Extract configuration
    model_config = config.get("models", {})
    data_config = config.get("data", {})
    predict_config = config.get("prediction", {})
    
    # Determine model path
    model_path = None
    checkpoint_path = None
    
    if model_config.get("model_path"):
        model_path = Path(model_config["model_path"])
    elif config.get("training", {}).get("resume_from_checkpoint"):
        checkpoint_path = Path(config["training"]["resume_from_checkpoint"])
    
    # Get data path
    data_path = data_config.get("test_path")
    if not data_path:
        raise ValueError("Test data path not specified")
    
    # Determine output path
    if output_path:
        final_output_path = output_path
    else:
        final_output_path = Path(predict_config.get("output_path", "predictions/submission.csv"))
    
    # Create the DTO
    return PredictionRequestDTO(
        model_path=model_path,
        checkpoint_path=checkpoint_path,
        data_path=Path(data_path),
        output_path=final_output_path,
        batch_size=predict_config.get("batch_size", data_config.get("batch_size", 32)),
        num_workers=data_config.get("num_workers", 0),
        output_format=predict_config.get("output_format", "csv"),
        include_probabilities=predict_config.get("include_probabilities", False),
        probability_threshold=predict_config.get("probability_threshold"),
        use_mixed_precision=predict_config.get("use_mixed_precision", False),
        device=predict_config.get("device"),
        id_column=predict_config.get("id_column", "id"),
        prediction_column=predict_config.get("prediction_column", "prediction"),
    )


def display_configuration(request: PredictionRequestDTO) -> None:
    """Display prediction configuration in a table."""
    table = Table(title="Prediction Configuration", show_header=True)
    table.add_column("Parameter", style="cyan")
    table.add_column("Value", style="green")
    
    # Model info
    if request.model_path:
        table.add_row("Model Path", str(request.model_path))
    elif request.checkpoint_path:
        table.add_row("Checkpoint Path", str(request.checkpoint_path))
    
    # Data info
    table.add_row("Input Data", str(request.data_path))
    table.add_row("Batch Size", str(request.batch_size))
    
    # Output info
    table.add_row("Output Path", str(request.output_path))
    table.add_row("Output Format", request.output_format)
    table.add_row("Include Probabilities", "Yes" if request.include_probabilities else "No")
    
    if request.probability_threshold is not None:
        table.add_row("Probability Threshold", f"{request.probability_threshold:.2f}")
    
    console.print(table)


def display_results(response) -> None:
    """Display prediction results."""
    if not response.success:
        console.print(f"\n[red]Prediction failed: {response.error_message}[/red]")
        return
    
    console.print("\n[bold green]Predictions completed successfully![/bold green]\n")
    
    # Results summary
    console.print(f"Total Predictions: {response.total_predictions}")
    console.print(f"Prediction Time: {response.prediction_time_seconds:.1f}s")
    console.print(f"Output saved to: [cyan]{response.output_path}[/cyan]")
    
    # Class distribution if available
    if response.class_distribution:
        console.print("\n[bold]Prediction Distribution:[/bold]")
        dist_table = Table(show_header=True)
        dist_table.add_column("Class", style="cyan")
        dist_table.add_column("Count", style="green")
        dist_table.add_column("Percentage", style="yellow")
        
        total = sum(response.class_distribution.values())
        for class_name, count in sorted(response.class_distribution.items()):
            percentage = (count / total) * 100
            dist_table.add_row(
                str(class_name),
                str(count),
                f"{percentage:.1f}%"
            )
        
        console.print(dist_table)
    
    # Sample predictions if available
    if response.sample_predictions:
        console.print("\n[bold]Sample Predictions:[/bold]")
        sample_table = Table(show_header=True)
        sample_table.add_column("ID", style="cyan")
        sample_table.add_column("Prediction", style="green")
        if response.sample_predictions[0].get("probability") is not None:
            sample_table.add_column("Probability", style="yellow")
        
        for pred in response.sample_predictions[:5]:
            row = [str(pred["id"]), str(pred["prediction"])]
            if pred.get("probability") is not None:
                row.append(f"{pred['probability']:.3f}")
            sample_table.add_row(*row)
        
        console.print(sample_table)
        console.print(f"[dim](Showing first 5 of {len(response.sample_predictions)} predictions)[/dim]")
    
    # Next steps
    console.print("\n[bold]Next steps:[/bold]")
    console.print(f"  • Submit predictions: [cyan]kaggle competitions submit -f {response.output_path} -m 'K-BERT submission'[/cyan]")
    console.print(f"  • Analyze predictions: [cyan]python analyze_predictions.py {response.output_path}[/cyan]")


def predict(
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
    input: Optional[Path] = typer.Option(
        None,
        "--input", "-i",
        help="Input data path",
    ),
    output: Optional[Path] = typer.Option(
        None,
        "--output", "-o",
        help="Output predictions path",
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
    format: Optional[str] = typer.Option(
        None,
        "--format", "-f",
        help="Output format (csv/json/parquet)",
    ),
    include_probs: bool = typer.Option(
        False,
        "--include-probs",
        help="Include prediction probabilities",
    ),
    threshold: Optional[float] = typer.Option(
        None,
        "--threshold", "-t",
        help="Probability threshold for binary classification",
    ),
    id_column: Optional[str] = typer.Option(
        None,
        "--id-column",
        help="Name of ID column in input data",
    ),
    debug: bool = typer.Option(
        False,
        "--debug",
        help="Enable debug logging",
    ),
):
    """Generate predictions using a trained BERT model.
    
    This command generates predictions for a dataset using a trained model.
    
    Examples:
        # Generate predictions with default settings
        k-bert predict --model output/best_model
        
        # Generate predictions with probabilities
        k-bert predict --model output/best_model --include-probs
        
        # Generate predictions with custom threshold
        k-bert predict --model output/best_model --threshold 0.7
        
        # Generate predictions in JSON format
        k-bert predict --model output/best_model --format json
    """
    console.print("\n[bold blue]K-BERT Prediction[/bold blue]")
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
        if input:
            merged_config.setdefault("data", {})["test_path"] = str(input)
        if batch_size:
            merged_config.setdefault("prediction", {})["batch_size"] = batch_size
        if format:
            merged_config.setdefault("prediction", {})["output_format"] = format
        if include_probs:
            merged_config.setdefault("prediction", {})["include_probabilities"] = True
        if threshold is not None:
            merged_config.setdefault("prediction", {})["probability_threshold"] = threshold
        if id_column:
            merged_config.setdefault("prediction", {})["id_column"] = id_column
        
        # Validate configuration
        errors = loader.validate_config(merged_config, "predict")
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
        
        # Create prediction request
        request = create_prediction_request(merged_config, output)
        
        # Display configuration
        display_configuration(request)
        
        # Get the prediction command
        predict_command = get_command(PredictCommand)
        
        # Run prediction with progress bar
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("Generating predictions...", total=None)
            
            # Run the async command
            response = asyncio.run(predict_command.execute(request))
        
        # Display results
        display_results(response)
        
    except ValueError as e:
        console.print(f"\n[red]Configuration error: {e}[/red]")
        raise typer.Exit(1)
    except KeyboardInterrupt:
        console.print("\n[yellow]Prediction interrupted by user[/yellow]")
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
    typer.run(predict)