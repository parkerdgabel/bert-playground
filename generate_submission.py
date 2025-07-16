#!/usr/bin/env python3
"""Generate Kaggle submission from a trained model."""

import argparse
import pandas as pd
import mlx.core as mx
from pathlib import Path
from datetime import datetime
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn

from models.modernbert_cnn_hybrid import CNNEnhancedModernBERT, CNNHybridConfig
from models.modernbert_optimized import OptimizedModernBertMLX
from models.classification_head import TitanicClassifier
from data.optimized_loader import OptimizedTitanicDataPipeline

console = Console()

def load_model(model_path: Path):
    """Load model from checkpoint."""
    # Check format
    if (model_path / "model.safetensors").exists():
        # Load config to determine model type
        config_path = model_path / "config.json"
        if config_path.exists():
            import json
            with open(config_path) as f:
                config_dict = json.load(f)
            
            if "cnn_kernel_sizes" in config_dict:
                # CNN hybrid model
                config = CNNHybridConfig(**config_dict)
                
                # Fix for models saved with incorrect hidden_size
                if config.hidden_size != 768 and hasattr(config, 'fusion_hidden_size'):
                    console.print(f"[yellow]Correcting config.hidden_size from {config.hidden_size} to 768[/yellow]")
                    config.hidden_size = 768  # Fix for base BERT
                
                model = CNNEnhancedModernBERT(config)
                
                weights = mx.load(str(model_path / "model.safetensors"))
                model.load_weights(list(weights.items()))
                console.print(f"[green]Loaded {len(weights)} parameters[/green]")
                return model, "cnn_hybrid"
            else:
                # Standard optimized model
                model = OptimizedModernBertMLX.from_pretrained(str(model_path))
                return model, "standard"
    else:
        raise ValueError(f"Unsupported model format in {model_path}")

def generate_predictions(model, test_loader, model_type: str):
    """Generate predictions for test data."""
    model.eval()
    all_predictions = []
    all_ids = []
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        num_batches = test_loader.get_num_batches()
        task = progress.add_task("Generating predictions...", total=num_batches)
        
        for batch_idx, batch in enumerate(test_loader.get_dataloader()()):
            # Run inference
            with mx.stream(mx.cpu):
                outputs = model(
                    input_ids=batch['input_ids'],
                    attention_mask=batch['attention_mask']
                )
                predictions = mx.argmax(outputs['logits'], axis=1)
                all_predictions.extend(predictions.tolist())
            
            # Track passenger IDs
            batch_size = batch['input_ids'].shape[0]
            start_idx = batch_idx * test_loader.batch_size
            batch_ids = list(range(start_idx, start_idx + batch_size))
            all_ids.extend(batch_ids)
            
            progress.update(task, advance=1)
    
    return all_ids, all_predictions

def main():
    parser = argparse.ArgumentParser(description="Generate Kaggle submission")
    parser.add_argument("model_path", help="Path to model checkpoint")
    parser.add_argument("--test-data", default="data/titanic/test.csv", 
                       help="Path to test data")
    parser.add_argument("--output", help="Output submission file path")
    parser.add_argument("--batch-size", type=int, default=32, 
                       help="Batch size for inference")
    
    args = parser.parse_args()
    
    # Set default output path if not provided
    if not args.output:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name = Path(args.model_path).parent.name
        args.output = f"submission_{model_name}_{timestamp}.csv"
    
    console.print(Panel.fit(
        f"🚀 Generating Kaggle Submission\n\n"
        f"Model: {args.model_path}\n"
        f"Test Data: {args.test_data}\n"
        f"Output: {args.output}",
        style="bold green"
    ))
    
    # Load model
    console.print("\n[blue]Loading model...[/blue]")
    model, model_type = load_model(Path(args.model_path))
    console.print(f"✅ Loaded {model_type} model")
    
    # Load test data
    console.print("\n[blue]Loading test data...[/blue]")
    test_loader = OptimizedTitanicDataPipeline(
        data_path=args.test_data,
        batch_size=args.batch_size,
        is_training=False,
        augment=False,
        num_threads=2,
        prefetch_size=2,
    )
    console.print(f"✅ Loaded {len(test_loader)} test samples")
    
    # Generate predictions
    console.print("\n[blue]Generating predictions...[/blue]")
    passenger_ids, predictions = generate_predictions(model, test_loader, model_type)
    
    # Load original test data to get actual PassengerIds
    test_df = pd.read_csv(args.test_data)
    
    # Create submission
    submission_df = pd.DataFrame({
        'PassengerId': test_df['PassengerId'].iloc[:len(predictions)],
        'Survived': predictions
    })
    
    # Save submission
    submission_df.to_csv(args.output, index=False)
    
    # Display statistics
    survival_rate = submission_df['Survived'].mean()
    console.print(f"\n✅ Submission generated successfully!")
    console.print(f"   Total predictions: {len(predictions)}")
    console.print(f"   Survival rate: {survival_rate:.3f}")
    console.print(f"   Saved to: {args.output}")
    
    # Show sample predictions
    console.print("\n[yellow]Sample predictions:[/yellow]")
    print(submission_df.head(10))


if __name__ == "__main__":
    main()