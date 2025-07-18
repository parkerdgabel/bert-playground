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
        from data import create_kaggle_dataloader
        from models.factory import create_model
        from models.modernbert_cnn_hybrid import create_cnn_hybrid_model
        
        # Import classifier
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "classification",
            str(Path(__file__).parent.parent.parent.parent / "models" / "classification.py")
        )
        classification_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(classification_module)
        UnifiedTitanicClassifier = classification_module.TitanicClassifier
        
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
        model_type = train_config.get("model_type", "base")
        use_mlx_embeddings = train_config.get("use_mlx_embeddings", False)
        tokenizer_backend = train_config.get("tokenizer_backend", "auto")
        print_info(f"Loaded training configuration from {config_path}")
    else:
        console.print("[yellow]Warning: No training config found, using defaults[/yellow]")
        model_name = "answerdotai/ModernBERT-base"
        model_type = "base"
        use_mlx_embeddings = False
        tokenizer_backend = "auto"
    
    # Create model
    with console.status("[yellow]Loading model...[/yellow]"):
        if use_mlx_embeddings:
            # Create MLX embeddings model
            try:
                from models.classification import create_titanic_classifier
                model = create_titanic_classifier(
                    model_name=model_name,
                    dropout_prob=0.0,  # No dropout for inference
                    use_layer_norm=False,
                    activation="relu",
                )
                model_desc = "MLX Embeddings ModernBERT"
            except ImportError:
                from embeddings.model_wrapper import MLXEmbeddingModel
                model = MLXEmbeddingModel(
                    model_name=model_name,
                    num_labels=2,
                    use_mlx_embeddings=True,
                )
                model_desc = "Legacy MLX Embeddings ModernBERT"
        elif model_type == "cnn_hybrid":
            bert_model = create_cnn_hybrid_model(
                model_name=model_name,
                num_labels=2,
                cnn_kernel_sizes=train_config.get("cnn_kernel_sizes", [2, 3, 4, 5]),
                cnn_num_filters=train_config.get("cnn_num_filters", 128),
                use_dilated_conv=train_config.get("use_dilated_conv", True),
            )
            bert_model.config.hidden_size = bert_model.output_hidden_size
            model = UnifiedTitanicClassifier(bert_model)
            model_desc = "CNN-Enhanced ModernBERT"
        else:
            bert_model = create_model("standard")
            model = UnifiedTitanicClassifier(bert_model)
            model_desc = "Standard ModernBERT"
        
        # Load weights
        model.load_pretrained(str(checkpoint))
        
    console.print(f"[green]✓ Loaded {model_desc} from {checkpoint}[/green]")
    
    # Create data loader
    test_loader = create_kaggle_dataloader(
        dataset_name=dataset_name,
        csv_path=str(test_data),
        tokenizer_name=model_name,
        batch_size=batch_size,
        max_length=max_length,
        shuffle=False,
        prefetch_size=4,
        num_workers=num_workers,
        tokenizer_backend=tokenizer_backend,
    )
    
    console.print(f"[green]✓ Loaded {len(test_loader)} test batches[/green]")
    
    # Generate predictions
    predictions = []
    probabilities = []
    
    with console.status("[yellow]Generating predictions...[/yellow]"):
        test_stream = test_loader.create_stream(is_training=False)
        
        for batch_idx, batch in enumerate(test_stream):
            # Convert numpy arrays to MLX arrays if needed
            if not isinstance(batch["input_ids"], mx.array):
                batch["input_ids"] = mx.array(batch["input_ids"])
                batch["attention_mask"] = mx.array(batch["attention_mask"])
            
            # Forward pass
            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
            )
            
            # Get predictions
            logits = outputs["logits"]
            
            if probability:
                # Calculate probabilities
                probs = mx.softmax(logits, axis=-1)
                probabilities.extend(probs.tolist())
            
            # Get class predictions
            preds = mx.argmax(logits, axis=-1)
            predictions.extend(preds.tolist())
            
            # Show progress
            if (batch_idx + 1) % 10 == 0:
                console.print(f"Processed {(batch_idx + 1) * batch_size} samples...", end="\r")
    
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
    
    # Show results
    print_success(f"Predictions saved to {output}")
    console.print(f"[cyan]Total predictions: {len(predictions[:len(df)])}[/cyan]")
    
    if probability:
        console.print("[cyan]Output includes probability scores[/cyan]")
    
    # Show sample predictions
    console.print("\n[bold]Sample predictions:[/bold]")
    sample_size = min(5, len(predictions))
    for i in range(sample_size):
        if probability:
            console.print(f"  Sample {i+1}: Class {predictions[i]} (probs: {[f'{p:.3f}' for p in probabilities[i]]})")
        else:
            console.print(f"  Sample {i+1}: Class {predictions[i]}")
    
    # Next steps
    console.print("\n[bold]Next steps:[/bold]")
    console.print(f"1. Submit to Kaggle: [cyan]bert kaggle submit create --competition {dataset_name} --file {output}[/cyan]")
    console.print("2. Evaluate locally: [cyan]bert model evaluate --predictions {} --ground-truth labels.csv[/cyan]".format(output))