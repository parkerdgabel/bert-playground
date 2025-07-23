"""Thin CLI adapter for prediction command.

This adapter is responsible only for:
1. Parsing command-line arguments
2. Creating the PredictionRequestDTO
3. Calling the PredictUseCase
4. Formatting and displaying the response
"""

from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

from application.dto.prediction import PredictionRequestDTO, PredictionFormat
from application.use_cases.predict import PredictUseCase
from infrastructure.bootstrap import get_service
from adapters.primary.cli.base import CLIAdapter, ProgressContext, format_time


console = Console()


class PredictCLIAdapter(CLIAdapter[PredictionRequestDTO, 'PredictionResponseDTO']):
    """CLI adapter for the predict command."""
    
    def create_request_dto(
        self,
        checkpoint: Path,
        test_data: Path,
        output: Optional[Path],
        output_format: str,
        batch_size: int,
        include_probabilities: bool,
        include_embeddings: bool,
        threshold: Optional[float],
        top_k: Optional[int],
    ) -> PredictionRequestDTO:
        """Create PredictionRequestDTO from CLI arguments."""
        # Determine output path
        if output is None:
            output = Path("predictions.csv")
        
        # Convert format string to enum
        try:
            format_enum = PredictionFormat(output_format.lower())
        except ValueError:
            raise ValueError(f"Invalid output format: {output_format}")
        
        return PredictionRequestDTO(
            model_path=checkpoint,
            data_path=test_data,
            batch_size=batch_size,
            output_format=format_enum,
            output_path=output,
            include_probabilities=include_probabilities,
            include_embeddings=include_embeddings,
            probability_threshold=threshold,
            top_k_predictions=top_k,
        )
    
    def display_results(self, response) -> None:
        """Display prediction results."""
        if not response.success:
            self.console.print(f"\n[red]Prediction failed: {response.error_message}[/red]")
            return
        
        self.console.print("\n[bold green]Predictions generated successfully![/bold green]\n")
        
        # Results table
        table = Table(title="Prediction Summary", show_header=True)
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")
        
        table.add_row("Total Predictions", str(response.num_predictions))
        table.add_row("Output File", str(response.output_path))
        table.add_row("Format", response.output_format.value if response.output_format else "N/A")
        table.add_row("Processing Time", format_time(response.prediction_time_seconds))
        table.add_row("Speed", f"{response.samples_per_second:.1f} samples/s")
        
        self.console.print(table)
        
        # Distribution table if available
        if response.prediction_distribution:
            dist_table = Table(title="Prediction Distribution", show_header=True)
            dist_table.add_column("Class", style="cyan")
            dist_table.add_column("Count", style="yellow")
            dist_table.add_column("Percentage", style="green")
            
            for label, count in response.prediction_distribution.items():
                percentage = (count / response.num_predictions) * 100
                dist_table.add_row(label, str(count), f"{percentage:.1f}%")
            
            self.console.print("\n")
            self.console.print(dist_table)
        
        # Confidence stats if available
        if response.confidence_stats:
            conf_table = Table(title="Confidence Statistics", show_header=True)
            conf_table.add_column("Statistic", style="cyan")
            conf_table.add_column("Value", style="green")
            
            for stat, value in response.confidence_stats.items():
                conf_table.add_row(stat.replace('_', ' ').title(), f"{value:.4f}")
            
            self.console.print("\n")
            self.console.print(conf_table)
        
        # Sample predictions if available
        if response.sample_predictions:
            self.console.print(f"\n[bold]Sample predictions (first 5):[/bold]")
            for i, pred in enumerate(response.sample_predictions[:5]):
                self.console.print(f"  {i+1}. {pred}")


async def predict_command(
    checkpoint: Path = typer.Argument(
        ...,
        help="Path to model checkpoint directory",
    ),
    test_data: Path = typer.Argument(
        ...,
        help="Path to test data file",
    ),
    output: Optional[Path] = typer.Option(
        None,
        "--output",
        "-o",
        help="Output file path (defaults to predictions.csv)",
    ),
    output_format: str = typer.Option(
        "csv",
        "--format",
        "-f",
        help="Output format: csv, json, parquet, numpy",
    ),
    batch_size: int = typer.Option(
        32,
        "--batch-size",
        "-b",
        help="Batch size for prediction",
    ),
    include_probabilities: bool = typer.Option(
        True,
        "--probabilities/--no-probabilities",
        help="Include prediction probabilities",
    ),
    include_embeddings: bool = typer.Option(
        False,
        "--embeddings",
        help="Include model embeddings in output",
    ),
    threshold: Optional[float] = typer.Option(
        None,
        "--threshold",
        "-t",
        help="Probability threshold for binary classification",
    ),
    top_k: Optional[int] = typer.Option(
        None,
        "--top-k",
        help="Return top K predictions for each sample",
    ),
    debug: bool = typer.Option(
        False,
        "--debug",
        "-d",
        help="Enable debug logging",
    ),
):
    """Generate predictions using a trained model.
    
    This is a thin CLI adapter that:
    1. Parses arguments
    2. Creates PredictionRequestDTO
    3. Calls PredictUseCase
    4. Displays results
    
    All business logic is handled by the use case.
    """
    console.print("\n[bold blue]K-BERT Prediction[/bold blue]")
    console.print("=" * 60)
    
    adapter = PredictCLIAdapter(console)
    
    try:
        # Create request DTO
        request = adapter.create_request_dto(
            checkpoint=checkpoint,
            test_data=test_data,
            output=output,
            output_format=output_format,
            batch_size=batch_size,
            include_probabilities=include_probabilities,
            include_embeddings=include_embeddings,
            threshold=threshold,
            top_k=top_k,
        )
        
        # Display configuration
        config = {
            "model_checkpoint": request.model_path,
            "test_data": request.data_path,
            "output_path": request.output_path,
            "output_format": request.output_format.value,
            "batch_size": request.batch_size,
            "include_probabilities": request.include_probabilities,
        }
        
        if request.probability_threshold is not None:
            config["probability_threshold"] = request.probability_threshold
        if request.top_k_predictions is not None:
            config["top_k_predictions"] = request.top_k_predictions
        
        adapter.display_config_table("Prediction Configuration", config)
        
        # Get use case
        use_case = get_service(PredictUseCase)
        
        # Run prediction with progress indicator
        with ProgressContext(console, "Generating predictions...") as progress:
            response = await use_case.execute(request)
        
        # Display results
        adapter.display_results(response)
        
    except ValueError as e:
        adapter.display_error(e, debug)
        raise typer.Exit(1)
    except KeyboardInterrupt:
        console.print("\n[yellow]Prediction interrupted by user[/yellow]")
        raise typer.Exit(130)
    except Exception as e:
        adapter.display_error(e, debug)
        raise typer.Exit(1)


# Create the Typer command
predict = predict_command