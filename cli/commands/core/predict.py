"""Prediction command implementation with config-first approach."""

import json
import sys
from pathlib import Path
from typing import Optional

import mlx.core as mx
import pandas as pd
import typer
from loguru import logger
from rich.console import Console

from ...utils import (
    handle_errors,
    track_time,
    print_error,
    print_success,
    validate_path,
)
from ...config import ConfigManager

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

console = Console()


@handle_errors
@track_time("Generating predictions")
def predict_command(
    checkpoint: Path = typer.Argument(
        ...,
        help="Model checkpoint path",
        callback=lambda p: validate_path(p, must_exist=True),
    ),
    test_data: Optional[Path] = typer.Option(
        None,
        "--test",
        "-t",
        help="Test data path (defaults to config value)",
    ),
    output: Path = typer.Option(
        "submission.csv",
        "--output",
        "-o",
        help="Output file path",
    ),
    config: Optional[Path] = typer.Option(
        None,
        "--config",
        "-c",
        help="Configuration file (defaults to k-bert.yaml or checkpoint config)",
    ),
    batch_size: Optional[int] = typer.Option(
        None,
        "--batch-size",
        "-b",
        help="Override batch size",
    ),
    no_config: bool = typer.Option(
        False,
        "--no-config",
        help="Run with defaults (no configuration file)",
    ),
    probability: bool = typer.Option(
        False,
        "--probability",
        help="Output probabilities instead of classes",
    ),
    format: str = typer.Option(
        "csv",
        "--format",
        "-f",
        help="Output format (csv, json, parquet)",
    ),
    debug: bool = typer.Option(
        False,
        "--debug",
        "-d",
        help="Enable debug logging",
    ),
):
    """Generate predictions using a trained model.

    This command loads a model checkpoint and generates predictions.
    Configuration is loaded from (in order of priority):
    1. Specified config file
    2. Config saved with checkpoint
    3. k-bert.yaml in current directory
    4. Defaults (with --no-config)
    
    Examples:
        # Predict using checkpoint's config
        k-bert predict output/run_20240115/final_model
        
        # Predict with specific test data
        k-bert predict output/model --test data/test_final.csv
        
        # Output probabilities as JSON
        k-bert predict output/model --format json --probability
        
        # Run without config
        k-bert predict output/model --no-config --test data/test.csv
    """
    # Configure logging
    log_level = "DEBUG" if debug else "INFO"
    logger.remove()
    logger.add(sys.stderr, level=log_level, enqueue=False)
    
    console.print("\n[bold blue]K-BERT Prediction[/bold blue]")
    console.print("=" * 60)
    
    # Load configuration
    merged_config = None
    
    if no_config:
        # Use minimal defaults
        if not test_data:
            print_error(
                "Test data is required when using --no-config",
                title="Missing Test Data"
            )
            raise typer.Exit(1)
        
        console.print("[yellow]Running with default configuration (--no-config)[/yellow]")
        config_overrides = {
            'data': {
                'test_path': str(test_data),
                'batch_size': batch_size or 64,
                'max_length': 256,
                'num_workers': 4,
            }
        }
        merged_config = ConfigManager().get_merged_config(cli_overrides=config_overrides)
    else:
        # Try to load configuration in order of priority
        config_manager = ConfigManager()
        config_loaded = False
        
        # 1. Try specified config file
        if config:
            console.print(f"[green]Using configuration: {config}[/green]")
            config_loaded = True
        else:
            # 2. Try checkpoint config
            checkpoint_config = checkpoint.parent.parent / "training_config.json"
            if checkpoint_config.exists():
                console.print(f"[green]Using checkpoint configuration: {checkpoint_config}[/green]")
                # Load the training config to extract model info
                with open(checkpoint_config) as f:
                    train_config = json.load(f)
                config_loaded = True
            else:
                # 3. Try project config
                project_configs = [
                    Path.cwd() / "k-bert.yaml",
                    Path.cwd() / "k-bert.yml",
                ]
                config = next((p for p in project_configs if p.exists()), None)
                if config:
                    console.print(f"[green]Using project configuration: {config}[/green]")
                    config_loaded = True
        
        if not config_loaded:
            print_error(
                "No configuration found. Create one with 'k-bert config init' "
                "or use --no-config with explicit test data path.",
                title="Configuration Required"
            )
            raise typer.Exit(1)
        
        # Build CLI overrides
        cli_overrides = {}
        if test_data:
            cli_overrides.setdefault('data', {})['test_path'] = str(test_data)
        if batch_size:
            cli_overrides.setdefault('data', {})['batch_size'] = batch_size
        
        merged_config = config_manager.get_merged_config(
            cli_overrides=cli_overrides,
            project_path=config,
            validate=True
        )
    
    # Validate test data path
    test_path = Path(merged_config.data.test_path) if merged_config.data.test_path else None
    if not test_path or not test_path.exists():
        print_error(
            f"Test data not found: {test_path}",
            title="Missing Test Data"
        )
        raise typer.Exit(1)
    
    # Display configuration
    console.print("\n[bold]Configuration:[/bold]")
    console.print(f"  Checkpoint: {checkpoint}")
    console.print(f"  Test data: {test_path}")
    console.print(f"  Batch size: {merged_config.data.batch_size}")
    console.print(f"  Output: {output} ({format})")
    if probability:
        console.print(f"  Mode: Probability output")
    
    # Set up logging
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = output.parent / f"predict_{timestamp}.log"
    
    from utils.logging_utils import add_file_logger
    add_file_logger(
        file_path=log_file,
        level=log_level,
        rotation="100 MB",
        retention="7 days",
        compression="zip"
    )
    
    # Import components
    try:
        from transformers import AutoTokenizer
        from data import create_dataloader
        from models.factory import create_model_from_checkpoint
        
    except ImportError as e:
        print_error(
            f"Failed to import components: {str(e)}",
            title="Import Error"
        )
        raise typer.Exit(1)
    
    # Load model and tokenizer
    console.print("\n[dim]Loading model...[/dim]")
    
    # Get model name from checkpoint config if available
    model_name = merged_config.models.default_model
    checkpoint_config_path = checkpoint.parent.parent / "training_config.json"
    if checkpoint_config_path.exists():
        with open(checkpoint_config_path) as f:
            checkpoint_config = json.load(f)
            model_name = checkpoint_config.get("model", model_name)
    
    # Load model
    try:
        model = create_model_from_checkpoint(checkpoint)
        console.print(f"[green]✓ Loaded model from {checkpoint}[/green]")
    except Exception as e:
        print_error(f"Failed to load model: {str(e)}", title="Model Loading Error")
        raise typer.Exit(1)
    
    # Create tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Create data loader
    console.print("[dim]Loading test data...[/dim]")
    
    test_loader = create_dataloader(
        data_path=test_path,
        batch_size=merged_config.data.batch_size,
        shuffle=False,
        num_workers=merged_config.data.num_workers,
        tokenizer=tokenizer,
        max_length=merged_config.data.max_length,
        split="test",
    )
    
    console.print(f"[green]✓ Loaded {len(test_loader)} batches[/green]")
    
    # Generate predictions
    predictions = []
    probabilities = []
    
    model.eval()
    total_samples = 0
    
    console.print("\n[dim]Generating predictions...[/dim]")
    
    with console.status("[yellow]Processing batches...[/yellow]"):
        for batch_idx, batch in enumerate(test_loader):
            # Forward pass
            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
            )
            
            # Extract predictions
            if isinstance(outputs, dict) and "predictions" in outputs:
                preds = outputs["predictions"]
                if hasattr(preds, "tolist"):
                    predictions.extend(preds.tolist())
                else:
                    predictions.extend(preds)
                
                if probability and "probabilities_2class" in outputs:
                    probs = outputs["probabilities_2class"]
                    if hasattr(probs, "tolist"):
                        probabilities.extend(probs.tolist())
                    else:
                        probabilities.extend(probs)
            else:
                # Compute from logits
                if isinstance(outputs, dict) and "logits" in outputs:
                    logits = outputs["logits"]
                elif hasattr(outputs, "logits"):
                    logits = outputs.logits
                else:
                    logits = outputs
                
                if probability:
                    probs = mx.softmax(logits, axis=-1)
                    probabilities.extend(probs.tolist())
                
                preds = mx.argmax(logits, axis=-1)
                predictions.extend(preds.tolist() if preds.ndim > 0 else [int(preds.item())])
            
            total_samples += len(batch["input_ids"])
            
            if (batch_idx + 1) % 10 == 0:
                console.print(f"Processed {total_samples} samples...", end="\r")
    
    console.print(f"\n[green]✓ Generated predictions for {total_samples} samples[/green]")
    
    # Load original test data for IDs
    df = pd.read_csv(test_path)
    
    # Prepare output based on format
    output.parent.mkdir(parents=True, exist_ok=True)
    
    if format == "csv":
        # Check if this is a known competition format
        if "PassengerId" in df.columns:
            # Titanic format
            submission = pd.DataFrame({
                "PassengerId": df["PassengerId"],
                "Survived": predictions[:len(df)]
            })
        elif "id" in df.columns:
            # Generic format with id column
            submission = pd.DataFrame({
                "id": df["id"],
                "prediction": predictions[:len(df)]
            })
        else:
            # Default format
            submission = pd.DataFrame({
                "index": range(len(predictions[:len(df)])),
                "prediction": predictions[:len(df)]
            })
        
        if probability and probabilities:
            prob_df = pd.DataFrame(
                probabilities[:len(df)],
                columns=[f"prob_class_{i}" for i in range(len(probabilities[0]))]
            )
            submission = pd.concat([submission, prob_df], axis=1)
        
        submission.to_csv(output, index=False)
    
    elif format == "json":
        results = []
        for i in range(min(len(predictions), len(df))):
            result = {"index": i, "prediction": int(predictions[i])}
            
            # Add ID columns if present
            if "PassengerId" in df.columns:
                result["PassengerId"] = int(df.iloc[i]["PassengerId"])
            elif "id" in df.columns:
                result["id"] = int(df.iloc[i]["id"])
            
            if probability and i < len(probabilities):
                result["probabilities"] = probabilities[i]
            
            results.append(result)
        
        with open(output, "w") as f:
            json.dump(results, f, indent=2)
    
    elif format == "parquet":
        # Same as CSV but save as parquet
        if "PassengerId" in df.columns:
            submission = pd.DataFrame({
                "PassengerId": df["PassengerId"],
                "Survived": predictions[:len(df)]
            })
        else:
            submission = pd.DataFrame({
                "id": range(len(predictions)),
                "prediction": predictions
            })
        
        if probability and probabilities:
            prob_df = pd.DataFrame(
                probabilities[:len(df)],
                columns=[f"prob_class_{i}" for i in range(len(probabilities[0]))]
            )
            submission = pd.concat([submission, prob_df], axis=1)
        
        submission.to_parquet(output, index=False)
    
    else:
        print_error(f"Unsupported format: {format}", title="Format Error")
        raise typer.Exit(1)
    
    print_success(
        f"Predictions saved to: {output}\n"
        f"Total predictions: {len(predictions)}\n"
        f"Log file: {log_file}",
        title="Prediction Complete"
    )
    
    # Show submission command if applicable
    if "PassengerId" in df.columns and format == "csv":
        console.print("\n[bold]Next steps:[/bold]")
        console.print(f"  • Submit to Kaggle: [cyan]k-bert competition submit titanic {output}[/cyan]")
        console.print(f"  • View predictions: [cyan]head -20 {output}[/cyan]")