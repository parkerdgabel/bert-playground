"""Submission commands for Kaggle competitions."""

import json
import sys
from datetime import datetime
from pathlib import Path

import typer

from ...utils import (
    get_console,
    handle_errors,
    print_error,
    print_info,
    print_success,
    print_warning,
    track_time,
    validate_path,
)
from ...utils.console import create_table

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))


@handle_errors
@track_time("Submitting to Kaggle")
def submit_command(
    competition: str = typer.Argument(..., help="Competition ID (e.g., titanic)"),
    submission_file: Path = typer.Argument(
        ...,
        help="Path to submission CSV file",
        callback=lambda p: validate_path(p, must_exist=True),
    ),
    message: str = typer.Option(
        "Submission from MLX-BERT CLI",
        "--message",
        "-m",
        help="Submission message (max 512 chars)",
    ),
    checkpoint: Path | None = typer.Option(
        None, "--checkpoint", "-c", help="Associated model checkpoint for tracking"
    ),
    track_mlflow: bool = typer.Option(
        True, "--track/--no-track", help="Track submission in MLflow"
    ),
):
    """Submit predictions to a Kaggle competition.

    Submits a CSV file to a Kaggle competition and optionally tracks the
    submission in MLflow for experiment management.

    Examples:
        # Basic submission
        bert kaggle submit titanic predictions.csv

        # Submission with custom message
        bert kaggle submit titanic predictions.csv -m "CNN model with augmentation"

        # Track checkpoint with submission
        bert kaggle submit titanic predictions.csv --checkpoint output/best_model

        # Submit without MLflow tracking
        bert kaggle submit titanic predictions.csv --no-track
    """
    console = get_console()

    console.print(
        f"\n[bold blue]Submitting to Kaggle Competition: {competition}[/bold blue]"
    )
    console.print("=" * 60)

    # Validate submission file
    if not submission_file.suffix == ".csv":
        print_error("Submission file must be a CSV file", title="Invalid File Type")
        raise typer.Exit(1)

    # Truncate message if too long
    if len(message) > 512:
        message = message[:509] + "..."
        print_warning("Message truncated to 512 characters")

    try:
        import mlflow

        from utils.kaggle_integration import KaggleIntegration
    except ImportError as e:
        print_error(
            f"Failed to import required packages: {str(e)}\n"
            "Make sure kaggle and mlflow are installed:\n"
            "pip install kaggle mlflow",
            title="Import Error",
        )
        raise typer.Exit(1)

    try:
        kaggle = KaggleIntegration()

        # Show submission details
        submission_table = create_table("Submission Details", ["Field", "Value"])
        submission_table.add_row("Competition", competition)
        submission_table.add_row("File", str(submission_file))
        submission_table.add_row(
            "File Size", f"{submission_file.stat().st_size / 1024:.1f} KB"
        )
        submission_table.add_row("Message", message)
        if checkpoint:
            submission_table.add_row("Checkpoint", str(checkpoint))
        console.print(submission_table)

        # Submit to Kaggle
        with console.status("[yellow]Submitting to Kaggle...[/yellow]"):
            kaggle.submit_predictions(
                competition_id=competition,
                submission_file=submission_file,
                message=message,
            )

        print_success("Successfully submitted to Kaggle!")
        submit_log.info("Successfully submitted to Kaggle")

        # Track in MLflow if enabled
        if track_mlflow and checkpoint:
            try:
                # Initialize MLflow
                from utils.mlflow_central import mlflow_central

                mlflow_central.initialize()

                # Find or create run
                checkpoint_info_path = checkpoint / "checkpoint_info.json"
                if checkpoint_info_path.exists():
                    with open(checkpoint_info_path) as f:
                        checkpoint_info = json.load(f)
                    run_id = checkpoint_info.get("run_id")

                    if run_id:
                        with mlflow.start_run(run_id=run_id):
                            # Log submission info
                            mlflow.log_param(
                                f"kaggle_submission_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                                {
                                    "competition": competition,
                                    "submission_file": str(submission_file),
                                    "message": message,
                                    "timestamp": datetime.now().isoformat(),
                                },
                            )

                            # Log submission file as artifact
                            mlflow.log_artifact(
                                str(submission_file), "kaggle_submissions"
                            )

                        console.print("[green]✓ Submission tracked in MLflow[/green]")

            except Exception as e:
                print_warning(f"Could not track submission in MLflow: {str(e)}")

        # Show next steps
        console.print("\n[bold]Next steps:[/bold]")
        console.print(
            f"1. View submission status: [cyan]bert kaggle history {competition}[/cyan]"
        )
        console.print(
            f"2. Check leaderboard: [cyan]bert kaggle leaderboard {competition}[/cyan]"
        )
        console.print("3. Wait for scoring to complete (usually takes a few minutes)")

    except Exception as e:
        print_error(f"Failed to submit to Kaggle: {str(e)}", title="Submission Error")
        raise typer.Exit(1)


@handle_errors
@track_time("Auto-submitting to Kaggle")
def auto_submit_command(
    competition: str = typer.Argument(..., help="Competition ID (e.g., titanic)"),
    checkpoint: Path = typer.Argument(
        ...,
        help="Model checkpoint directory",
        callback=lambda p: validate_path(p, must_exist=True),
    ),
    test_data: Path = typer.Argument(
        ...,
        help="Test data CSV file",
        callback=lambda p: validate_path(p, must_exist=True),
    ),
    config: Path | None = typer.Option(
        None, "--config", "-c", help="Competition config file"
    ),
    message_template: str = typer.Option(
        "Auto-submission: {model_name} - Score: {val_score:.4f}",
        "--template",
        help="Message template with placeholders",
    ),
    batch_size: int = typer.Option(
        64, "--batch-size", "-b", help="Prediction batch size"
    ),
    output_dir: Path = typer.Option(
        "submissions", "--output", "-o", help="Output directory"
    ),
):
    """Automatically generate and submit predictions from a checkpoint.
    
    This command loads a model checkpoint, generates predictions on test data,
    and submits them to Kaggle automatically. It uses checkpoint metadata to
    create an informative submission message.
    
    Examples:
        # Basic auto-submission
        bert kaggle auto-submit titanic output/best_model data/test.csv
        
        # Custom message template
        bert kaggle auto-submit titanic output/model data/test.csv \\
            --template "MLX-BERT: {model_type} - LR: {learning_rate}"
        
        # Use competition config
        bert kaggle auto-submit titanic output/model data/test.csv \\
            --config configs/titanic.yaml
    """
    console = get_console()

    console.print(f"\n[bold blue]Auto-Submit to Kaggle: {competition}[/bold blue]")
    console.print("=" * 60)

    try:
        import pandas as pd
        import yaml

        from utils.evaluation import ModelEvaluator
        from utils.kaggle_integration import CompetitionConfig, KaggleIntegration
    except ImportError as e:
        print_error(
            f"Failed to import required packages: {str(e)}", title="Import Error"
        )
        raise typer.Exit(1)

    try:
        # Load checkpoint info
        checkpoint_info_path = checkpoint / "checkpoint_info.json"
        training_config_path = checkpoint.parent.parent / "training_config.json"

        if not checkpoint_info_path.exists():
            print_error(
                "No checkpoint_info.json found. Is this a valid checkpoint?",
                title="Invalid Checkpoint",
            )
            raise typer.Exit(1)

        with open(checkpoint_info_path) as f:
            checkpoint_info = json.load(f)

        # Load training config if available
        training_config = {}
        if training_config_path.exists():
            with open(training_config_path) as f:
                training_config = json.load(f)

        # Show checkpoint info
        info_table = create_table("Checkpoint Information", ["Property", "Value"])
        info_table.add_row("Model Type", training_config.get("model_type", "Unknown"))
        info_table.add_row("Model Name", training_config.get("model", "Unknown"))
        info_table.add_row(
            "Best Score", f"{checkpoint_info.get('best_val_accuracy', 0):.4f}"
        )
        info_table.add_row("Epoch", str(checkpoint_info.get("epoch", "Unknown")))
        info_table.add_row(
            "Learning Rate", str(training_config.get("learning_rate", "Unknown"))
        )
        console.print(info_table)

        # Load competition config if provided
        comp_config = None
        if config and config.exists():
            with open(config) as f:
                comp_config = CompetitionConfig(**yaml.safe_load(f))
            print_info(f"Loaded competition config from {config}")

        # Generate predictions
        output_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        submission_file = output_dir / f"{competition}_submission_{timestamp}.csv"

        console.print("\n[yellow]Generating predictions...[/yellow]")

        # Import and run prediction
        from cli.commands.core.predict import predict_command

        # Call predict command programmatically
        predict_command(
            test_data=test_data,
            checkpoint=checkpoint,
            output=submission_file,
            batch_size=batch_size,
            dataset_name=competition,
            format="csv",
            probability=False,
        )

        # Format submission message
        format_data = {
            "model_name": training_config.get("model", "MLX-BERT"),
            "model_type": training_config.get("model_type", "base"),
            "val_score": checkpoint_info.get("best_val_accuracy", 0),
            "learning_rate": training_config.get("learning_rate", 0),
            "batch_size": training_config.get("batch_size", 0),
            "epochs": training_config.get("epochs", 0),
            "checkpoint_epoch": checkpoint_info.get("epoch", 0),
        }

        try:
            message = message_template.format(**format_data)
        except KeyError as e:
            print_warning(f"Invalid placeholder in message template: {e}")
            message = f"Auto-submission from {checkpoint.name}"

        # Submit to Kaggle
        kaggle = KaggleIntegration()

        with console.status("[yellow]Submitting to Kaggle...[/yellow]"):
            kaggle.submit_predictions(
                competition_id=competition,
                submission_file=submission_file,
                message=message,
            )

        print_success("Successfully submitted to Kaggle!")
        print_info(f"Submission file: {submission_file}")
        print_info(f"Message: {message}")

        # Log to MLflow if available
        try:
            import mlflow

            from utils.mlflow_central import mlflow_central

            mlflow_central.initialize()

            run_id = checkpoint_info.get("run_id")
            if run_id:
                with mlflow.start_run(run_id=run_id):
                    mlflow.log_artifact(str(submission_file), "kaggle_submissions")
                    mlflow.log_metric(f"kaggle_submission_{timestamp}", 1)
                console.print("[green]✓ Submission tracked in MLflow[/green]")
        except:
            pass

        # Show next steps
        console.print("\n[bold]Next steps:[/bold]")
        console.print(
            f"1. Check submission status: [cyan]bert kaggle history {competition}[/cyan]"
        )
        console.print(
            f"2. View leaderboard: [cyan]bert kaggle leaderboard {competition}[/cyan]"
        )
        console.print(f"3. Submission file saved: [cyan]{submission_file}[/cyan]")

    except Exception as e:
        print_error(f"Auto-submission failed: {str(e)}", title="Auto-Submit Error")
        raise typer.Exit(1)
