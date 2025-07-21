"""Prediction command implementation."""

from pathlib import Path
from typing import Optional
import typer
from loguru import logger
import json
import sys
import pandas as pd
import mlx.core as mx

from ...utils import (
    get_console, print_success, print_error, print_info,
    handle_errors, track_time, requires_project,
    validate_path, validate_batch_size
)

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

@handle_errors
@requires_project()
@track_time("Generating predictions")
def predict_command(
    test_data: Path = typer.Option(..., "--test", "-t", help="Test data path",
                                  callback=lambda p: validate_path(p, must_exist=True)),
    checkpoint: Path = typer.Option(..., "--checkpoint", "-c", help="Model checkpoint path",
                                   callback=lambda p: validate_path(p, must_exist=True)),
    output: Path = typer.Option("submission.csv", "--output", "-o", help="Output file path"),
    batch_size: int = typer.Option(64, "--batch-size", "-b", help="Prediction batch size",
                                  callback=validate_batch_size),
    max_length: int = typer.Option(256, "--max-length", help="Maximum sequence length"),
    format: str = typer.Option("csv", "--format", "-f", help="Output format (csv, json, parquet)"),
    dataset_name: str = typer.Option("titanic", "--dataset", help="Dataset name for loader"),
    num_workers: int = typer.Option(4, "--workers", help="Number of data loading workers"),
    no_header: bool = typer.Option(False, "--no-header", help="Exclude header from output"),
    probability: bool = typer.Option(False, "--probability", help="Output probabilities instead of classes"),
):
    """Generate predictions using a trained model.
    
    This command loads a trained model checkpoint and generates predictions
    on test data. It supports various output formats and can output either
    class predictions or probabilities.
    
    Examples:
        # Basic prediction
        bert predict --test data/test.csv --checkpoint output/best_model
        
        # Output probabilities as JSON
        bert predict --test data/test.csv --checkpoint output/model --format json --probability
        
        # Custom output path
        bert predict --test data/test.csv --checkpoint output/model -o predictions/submission.csv
    """
    console = get_console()
    
    console.print("\n[bold blue]MLX ModernBERT Prediction[/bold blue]")
    console.print("=" * 60)
    
    # Import necessary components
    try:
        from data import create_dataloader
        from models.factory import create_model_from_checkpoint
        from transformers import AutoTokenizer
        
    except ImportError as e:
        print_error(
            f"Failed to import components: {str(e)}\n"
            "Make sure all dependencies are installed.",
            title="Import Error"
        )
        raise typer.Exit(1)
    
    # Load model configuration
    config_path = checkpoint.parent.parent / "training_config.json"
    if config_path.exists():
        with open(config_path) as f:
            train_config = json.load(f)
        model_name = train_config.get("model", "answerdotai/ModernBERT-base")
        print_info(f"Loaded training configuration from {config_path}")
    else:
        console.print("[yellow]Warning: No training config found, using defaults[/yellow]")
        model_name = "answerdotai/ModernBERT-base"
    
    # Load model
    console.print(f"Loading model from {checkpoint}...")
    try:
        model = create_model_from_checkpoint(checkpoint)
        console.print(f"[green]✓ Loaded model from {checkpoint}[/green]")
    except Exception as e:
        print_error(f"Failed to load model: {str(e)}", title="Model Loading Error")
        raise typer.Exit(1)
    
    # Create tokenizer
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        console.print(f"[green]✓ Loaded tokenizer: {model_name}[/green]")
    except Exception as e:
        print_error(f"Failed to load tokenizer: {str(e)}", title="Tokenizer Error")
        raise typer.Exit(1)
    
    # Create data loader
    test_loader = create_dataloader(
        data_path=test_data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        tokenizer=tokenizer,
        max_length=max_length,
        split="test",
    )
    
    console.print(f"[green]✓ Created test data loader[/green]")
    
    # Generate predictions
    predictions = []
    probabilities = []
    
    model.eval()
    total_samples = 0
    
    with console.status("[yellow]Generating predictions...[/yellow]"):
        for batch_idx, batch in enumerate(test_loader):
            # Forward pass (MLX doesn't need no_grad context)
            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
            )
            
            # Debug outputs
            logger.debug(f"Batch {batch_idx} outputs keys: {outputs.keys() if isinstance(outputs, dict) else 'not a dict'}")
            
            # Get predictions - model already returns predictions
            if isinstance(outputs, dict) and "predictions" in outputs:
                # Use pre-computed predictions
                preds = outputs["predictions"]
                
                # Convert to list
                if hasattr(preds, 'tolist'):
                    predictions.extend(preds.tolist())
                else:
                    predictions.extend(preds)
                
                # Handle probabilities if requested
                if probability and "probabilities_2class" in outputs:
                    probs = outputs["probabilities_2class"]
                    if hasattr(probs, 'tolist'):
                        probabilities.extend(probs.tolist())
                    else:
                        probabilities.extend(probs)
                        
            else:
                # Fallback to computing from logits
                if isinstance(outputs, dict) and "logits" in outputs:
                    logits = outputs["logits"]
                elif hasattr(outputs, "logits"):
                    logits = outputs.logits
                else:
                    logits = outputs
                
                if probability:
                    # Calculate probabilities
                    probs = mx.softmax(logits, axis=-1)
                    probabilities.extend(probs.tolist())
                
                # Get class predictions
                preds = mx.argmax(logits, axis=-1)
                
                # Convert to list (handle both single and batch predictions)
                if preds.ndim == 0:
                    predictions.append(int(preds.item()))
                elif preds.ndim == 1:
                    predictions.extend(preds.tolist())
                else:
                    # If 2D or higher, flatten first dimension
                    batch_preds = preds.reshape(-1).tolist()
                    predictions.extend(batch_preds)
            
            total_samples += len(batch["input_ids"])
            
            # Show progress
            if (batch_idx + 1) % 10 == 0:
                console.print(f"Processed {total_samples} samples...", end="\r")
    
    console.print(f"\n[green]✓ Generated predictions for {total_samples} samples[/green]")
    
    # Load original test data for IDs
    df = pd.read_csv(test_data)
    
    # Prepare output based on format
    if format == "csv":
        # Create submission dataframe
        if dataset_name == "titanic":
            submission = pd.DataFrame({
                "PassengerId": df["PassengerId"],
                "Survived": predictions[:len(df)]
            })
        else:
            # Generic format
            submission = pd.DataFrame({
                "id": range(len(predictions)),
                "prediction": predictions
            })
            
        if probability:
            # Add probability columns
            prob_df = pd.DataFrame(probabilities[:len(df)], columns=[f"prob_class_{i}" for i in range(len(probabilities[0]))])
            submission = pd.concat([submission, prob_df], axis=1)
        
        # Save CSV
        submission.to_csv(output, index=False, header=not no_header)
        
    elif format == "json":
        # Create JSON output
        results = []
        for i in range(len(predictions[:len(df)])):
            result = {
                "index": i,
                "prediction": int(predictions[i])
            }
            if dataset_name == "titanic" and "PassengerId" in df.columns:
                result["PassengerId"] = int(df.iloc[i]["PassengerId"])
            
            if probability:
                result["probabilities"] = probabilities[i]
            
            results.append(result)
        
        # Save JSON
        with open(output, "w") as f:
            json.dump(results, f, indent=2)
            
    elif format == "parquet":
        # Create parquet output
        if dataset_name == "titanic":
            submission = pd.DataFrame({
                "PassengerId": df["PassengerId"],
                "Survived": predictions[:len(df)]
            })
        else:
            submission = pd.DataFrame({
                "id": range(len(predictions)),
                "prediction": predictions
            })
            
        if probability:
            prob_df = pd.DataFrame(probabilities[:len(df)], columns=[f"prob_class_{i}" for i in range(len(probabilities[0]))])
            submission = pd.concat([submission, prob_df], axis=1)
        
        # Save parquet
        submission.to_parquet(output, index=False)
    
    else:
        print_error(f"Unsupported format: {format}", title="Format Error")
        raise typer.Exit(1)
    
    print_success(
        f"Predictions saved to {output}\n"
        f"Format: {format}\n"
        f"Total predictions: {len(predictions)}",
        title="Prediction Complete"
    )