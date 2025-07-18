"""Model evaluation command - evaluate model performance on test data."""

from pathlib import Path
from typing import Optional, List, Dict, Any
import typer
from loguru import logger
import json
import sys
import time
from datetime import datetime

from ...utils import (
    get_console, print_success, print_error, print_info, print_warning,
    handle_errors, track_time, requires_project,
    validate_path, create_table, create_progress
)

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

console = get_console()

@handle_errors
@requires_project()
@track_time("Evaluating model")
def evaluate_command(
    checkpoint: Path = typer.Option(..., "--checkpoint", "-c",
                                  help="Path to model checkpoint",
                                  callback=lambda p: validate_path(p, must_exist=True)),
    test_data: Path = typer.Option(..., "--test", "-t",
                                 help="Test data file (CSV format)",
                                 callback=lambda p: validate_path(p, must_exist=True, extensions=['.csv'])),
    metrics: List[str] = typer.Option(["accuracy", "f1", "precision", "recall"],
                                    "--metrics", "-m",
                                    help="Metrics to compute"),
    batch_size: int = typer.Option(32, "--batch-size", "-b",
                                 help="Batch size for evaluation"),
    max_length: int = typer.Option(256, "--max-length",
                                 help="Maximum sequence length"),
    num_workers: int = typer.Option(4, "--workers", "-w",
                              help="Number of data loading workers"),
    output: Optional[Path] = typer.Option(None, "--output", "-o",
                                        help="Save evaluation results to file"),
    save_predictions: bool = typer.Option(False, "--save-predictions", "-p",
                                        help="Save all predictions to CSV"),
    confusion_matrix: bool = typer.Option(False, "--confusion-matrix",
                                        help="Generate confusion matrix"),
    per_class: bool = typer.Option(False, "--per-class",
                                 help="Show per-class metrics"),
    error_analysis: bool = typer.Option(False, "--error-analysis", "-e",
                                      help="Perform error analysis"),
    mlflow_run: Optional[str] = typer.Option(None, "--mlflow-run",
                                           help="Log results to MLflow run"),
    compare_with: Optional[Path] = typer.Option(None, "--compare", "-C",
                                              help="Compare with another model"),
    device: str = typer.Option("auto", "--device", "-d",
                             help="Device to use: auto, cpu, gpu"),
    verbose: bool = typer.Option(False, "--verbose", "-v",
                               help="Show detailed evaluation information")
):
    """Evaluate model performance on test data.
    
    This command provides comprehensive model evaluation with various
    metrics and analysis options.
    
    Supported metrics:
    • accuracy: Overall classification accuracy
    • f1: F1 score (macro/micro/weighted)
    • precision: Precision score
    • recall: Recall score
    • auc: Area under ROC curve
    • loss: Average loss value
    
    Examples:
        # Basic evaluation
        bert model evaluate -c output/run_001/best_model -t data/test.csv
        
        # Full evaluation with all analyses
        bert model evaluate -c output/run_001/best_model -t data/test.csv \\
            --confusion-matrix --per-class --error-analysis
        
        # Save predictions and results
        bert model evaluate -c output/run_001/best_model -t data/test.csv \\
            --save-predictions --output results.json
        
        # Compare two models
        bert model evaluate -c output/run_001/best_model -t data/test.csv \\
            --compare output/run_002/best_model
        
        # Log to MLflow
        bert model evaluate -c output/run_001/best_model -t data/test.csv \\
            --mlflow-run abc123def456
    """
    import mlx.core as mx
    
    # Set device
    if device == "auto":
        device = mx.default_device()
    elif device == "cpu":
        mx.set_default_device(mx.cpu)
    elif device == "gpu":
        if mx.metal.is_available():
            mx.set_default_device(mx.gpu)
        else:
            print_warning("GPU not available, using CPU")
            mx.set_default_device(mx.cpu)
    
    console.print(f"\n[bold blue]Model Evaluation[/bold blue]")
    console.print(f"Checkpoint: {checkpoint}")
    console.print(f"Test data: {test_data}")
    console.print(f"Metrics: {', '.join(metrics)}")
    console.print(f"Device: {mx.default_device()}")
    console.print()
    
    try:
        # Load test data
        test_dataset = _load_test_data(test_data, max_length, verbose)
        console.print(f"Loaded {len(test_dataset)} test samples")
        
        # Load model
        model, tokenizer, config = _load_model_for_eval(checkpoint, verbose)
        
        # Run evaluation
        results = _evaluate_model(
            model=model,
            tokenizer=tokenizer,
            dataset=test_dataset,
            batch_size=batch_size,
            metrics=metrics,
            verbose=verbose
        )
        
        # Display results
        _display_results(results, metrics)
        
        # Additional analyses
        if confusion_matrix:
            _display_confusion_matrix(results)
        
        if per_class:
            _display_per_class_metrics(results, config)
        
        if error_analysis:
            _perform_error_analysis(results, test_dataset, config)
        
        # Save predictions if requested
        if save_predictions:
            pred_path = output.parent / f"{output.stem}_predictions.csv" if output else Path("predictions.csv")
            _save_predictions(results, test_dataset, pred_path)
            print_success(f"Predictions saved to: {pred_path}")
        
        # Compare with another model
        if compare_with:
            _compare_models(checkpoint, compare_with, test_dataset, metrics, batch_size, max_length)
        
        # Save results
        if output:
            _save_results(results, output, checkpoint, test_data, metrics)
            print_success(f"Results saved to: {output}")
        
        # Log to MLflow
        if mlflow_run:
            _log_to_mlflow(results, mlflow_run, checkpoint)
            print_success(f"Results logged to MLflow run: {mlflow_run}")
        
    except Exception as e:
        print_error(f"Evaluation failed: {e}")
        if verbose:
            logger.exception("Detailed error:")
        raise typer.Exit(1)


def _load_test_data(test_file: Path, max_length: int, verbose: bool) -> List[Dict[str, Any]]:
    """Load test data from CSV file."""
    import pandas as pd
    from data.text_templates import TabularToTextConverter
    
    if verbose:
        console.print("[dim]Loading test data...[/dim]")
    
    # Load CSV
    df = pd.read_csv(test_file)
    
    # Check for required columns
    if 'text' not in df.columns and 'label' not in df.columns:
        # Try to convert tabular data to text
        converter = TabularToTextConverter()
        df = converter.convert_dataframe(df)
    
    # Create dataset
    dataset = []
    for idx, row in df.iterrows():
        sample = {
            'id': idx,
            'text': row.get('text', ''),
            'label': row.get('label', row.get('target', -1))
        }
        
        # Add any additional columns
        for col in df.columns:
            if col not in ['text', 'label', 'target']:
                sample[col] = row[col]
        
        dataset.append(sample)
    
    return dataset


def _load_model_for_eval(checkpoint: Path, verbose: bool) -> tuple:
    """Load model for evaluation."""
    import mlx.core as mx
    from models.modernbert_optimized import ModernBertModel
    from models.classification_head import ClassificationHead
    from embeddings.tokenizer_wrapper import UnifiedTokenizer
    
    if verbose:
        console.print("[dim]Loading model...[/dim]")
    
    # Load config
    config_path = checkpoint / "config.json"
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Initialize model
    model = ModernBertModel(config)
    classification_head = ClassificationHead(
        hidden_size=config['hidden_size'],
        num_labels=config.get('num_labels', 2),
        dropout=config.get('classifier_dropout', 0.1)
    )
    
    # Load weights
    weights_path = checkpoint / "model.safetensors"
    if weights_path.exists():
        from safetensors import safe_open
        weights = {}
        with safe_open(weights_path, framework="mlx") as f:
            for key in f.keys():
                weights[key] = f.get_tensor(key)
        
        # Load weights into models
        model.load_weights(list(weights.items()))
        classification_head.load_weights(list(weights.items()))
    else:
        # Try npz format
        weights_path = checkpoint / "model.npz"
        if weights_path.exists():
            weights = mx.load(str(weights_path))
            model.load_weights(list(weights.items()))
            classification_head.load_weights(list(weights.items()))
    
    # Load tokenizer
    tokenizer = UnifiedTokenizer(
        model_name=config.get('model_name', 'answerdotai/ModernBERT-base'),
        backend='auto',
        checkpoint_path=checkpoint
    )
    
    # Set to eval mode
    model.eval()
    classification_head.eval()
    
    # Create combined model for easier evaluation
    class CombinedModel:
        def __init__(self, encoder, head):
            self.encoder = encoder
            self.head = head
        
        def __call__(self, input_ids, attention_mask):
            outputs = self.encoder(input_ids, attention_mask)
            logits = self.head(outputs['last_hidden_state'][:, 0, :])
            return logits
    
    combined_model = CombinedModel(model, classification_head)
    
    return combined_model, tokenizer, config


def _evaluate_model(model, tokenizer, dataset: List[Dict[str, Any]], 
                   batch_size: int, metrics: List[str], verbose: bool) -> Dict[str, Any]:
    """Run model evaluation."""
    import mlx.core as mx
    import numpy as np
    from sklearn.metrics import (
        accuracy_score, f1_score, precision_score, recall_score,
        confusion_matrix, classification_report, roc_auc_score
    )
    
    results = {
        'predictions': [],
        'labels': [],
        'probabilities': [],
        'texts': [],
        'ids': []
    }
    
    # Create progress bar
    progress = create_progress()
    task = progress.add_task("[cyan]Evaluating...", total=len(dataset))
    
    with progress:
        # Process in batches
        for i in range(0, len(dataset), batch_size):
            batch = dataset[i:i + batch_size]
            
            # Prepare batch
            texts = [sample['text'] for sample in batch]
            labels = [sample['label'] for sample in batch]
            ids = [sample['id'] for sample in batch]
            
            # Tokenize
            encoded = tokenizer(
                texts,
                max_length=256,
                padding=True,
                truncation=True,
                return_tensors='mlx'
            )
            
            # Forward pass
            with mx.no_grad():
                logits = model(
                    encoded['input_ids'],
                    encoded['attention_mask']
                )
                probs = mx.softmax(logits, axis=-1)
            
            # Get predictions
            preds = mx.argmax(logits, axis=-1)
            
            # Store results
            results['predictions'].extend(preds.tolist())
            results['labels'].extend(labels)
            results['probabilities'].extend(probs.tolist())
            results['texts'].extend(texts)
            results['ids'].extend(ids)
            
            progress.update(task, advance=len(batch))
    
    # Convert to numpy for sklearn metrics
    y_true = np.array(results['labels'])
    y_pred = np.array(results['predictions'])
    y_prob = np.array(results['probabilities'])
    
    # Compute metrics
    computed_metrics = {}
    
    if 'accuracy' in metrics:
        computed_metrics['accuracy'] = accuracy_score(y_true, y_pred)
    
    if 'f1' in metrics:
        computed_metrics['f1_macro'] = f1_score(y_true, y_pred, average='macro')
        computed_metrics['f1_micro'] = f1_score(y_true, y_pred, average='micro')
        computed_metrics['f1_weighted'] = f1_score(y_true, y_pred, average='weighted')
    
    if 'precision' in metrics:
        computed_metrics['precision_macro'] = precision_score(y_true, y_pred, average='macro')
        computed_metrics['precision_weighted'] = precision_score(y_true, y_pred, average='weighted')
    
    if 'recall' in metrics:
        computed_metrics['recall_macro'] = recall_score(y_true, y_pred, average='macro')
        computed_metrics['recall_weighted'] = recall_score(y_true, y_pred, average='weighted')
    
    if 'auc' in metrics and len(np.unique(y_true)) == 2:
        # AUC for binary classification
        computed_metrics['auc'] = roc_auc_score(y_true, y_prob[:, 1])
    
    # Add confusion matrix
    results['confusion_matrix'] = confusion_matrix(y_true, y_pred)
    results['classification_report'] = classification_report(y_true, y_pred, output_dict=True)
    results['metrics'] = computed_metrics
    
    return results


def _display_results(results: Dict[str, Any], metrics: List[str]):
    """Display evaluation results."""
    console.print("\n[bold green]Evaluation Results[/bold green]")
    
    table = create_table("Metrics")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="yellow")
    
    for metric, value in results['metrics'].items():
        if isinstance(value, float):
            table.add_row(metric.replace('_', ' ').title(), f"{value:.4f}")
        else:
            table.add_row(metric.replace('_', ' ').title(), str(value))
    
    console.print(table)


def _display_confusion_matrix(results: Dict[str, Any]):
    """Display confusion matrix."""
    console.print("\n[bold green]Confusion Matrix[/bold green]")
    
    cm = results['confusion_matrix']
    n_classes = cm.shape[0]
    
    # Create table
    table = create_table("")
    
    # Add header
    table.add_column("True \\ Pred", style="bold")
    for i in range(n_classes):
        table.add_column(f"Class {i}", style="cyan")
    
    # Add rows
    for i in range(n_classes):
        row = [f"Class {i}"]
        for j in range(n_classes):
            value = cm[i, j]
            # Highlight diagonal
            if i == j:
                row.append(f"[bold green]{value}[/bold green]")
            else:
                row.append(str(value))
        table.add_row(*row)
    
    console.print(table)


def _display_per_class_metrics(results: Dict[str, Any], config: dict):
    """Display per-class metrics."""
    console.print("\n[bold green]Per-Class Metrics[/bold green]")
    
    report = results['classification_report']
    label_map = config.get('label_map', {})
    
    table = create_table("Class Performance")
    table.add_column("Class", style="cyan")
    table.add_column("Precision", style="yellow")
    table.add_column("Recall", style="yellow")
    table.add_column("F1-Score", style="green")
    table.add_column("Support", style="blue")
    
    for class_id, metrics in report.items():
        if class_id in ['accuracy', 'macro avg', 'weighted avg']:
            continue
        
        class_name = label_map.get(int(class_id), f"Class {class_id}")
        table.add_row(
            class_name,
            f"{metrics['precision']:.3f}",
            f"{metrics['recall']:.3f}",
            f"{metrics['f1-score']:.3f}",
            str(metrics['support'])
        )
    
    # Add averages
    table.add_row("", "", "", "", "")
    for avg_type in ['macro avg', 'weighted avg']:
        if avg_type in report:
            table.add_row(
                f"[bold]{avg_type}[/bold]",
                f"{report[avg_type]['precision']:.3f}",
                f"{report[avg_type]['recall']:.3f}",
                f"{report[avg_type]['f1-score']:.3f}",
                str(report[avg_type]['support'])
            )
    
    console.print(table)


def _perform_error_analysis(results: Dict[str, Any], dataset: List[Dict[str, Any]], 
                           config: dict):
    """Perform error analysis."""
    console.print("\n[bold green]Error Analysis[/bold green]")
    
    # Find misclassified samples
    errors = []
    for i, (pred, true) in enumerate(zip(results['predictions'], results['labels'])):
        if pred != true:
            errors.append({
                'id': results['ids'][i],
                'text': results['texts'][i][:100] + "..." if len(results['texts'][i]) > 100 else results['texts'][i],
                'true_label': true,
                'predicted_label': pred,
                'confidence': max(results['probabilities'][i])
            })
    
    console.print(f"Total errors: {len(errors)} / {len(results['predictions'])} ({len(errors)/len(results['predictions'])*100:.1f}%)")
    
    if errors:
        # Show most confident errors
        errors.sort(key=lambda x: x['confidence'], reverse=True)
        
        console.print("\n[bold]Most Confident Errors:[/bold]")
        table = create_table("Error Examples")
        table.add_column("ID", style="cyan")
        table.add_column("Text", style="white", overflow="fold", max_width=50)
        table.add_column("True", style="green")
        table.add_column("Predicted", style="red")
        table.add_column("Confidence", style="yellow")
        
        label_map = config.get('label_map', {})
        
        for error in errors[:10]:
            true_label = label_map.get(error['true_label'], str(error['true_label']))
            pred_label = label_map.get(error['predicted_label'], str(error['predicted_label']))
            
            table.add_row(
                str(error['id']),
                error['text'],
                true_label,
                pred_label,
                f"{error['confidence']:.3f}"
            )
        
        console.print(table)


def _save_predictions(results: Dict[str, Any], dataset: List[Dict[str, Any]], 
                     output_path: Path):
    """Save predictions to CSV."""
    import pandas as pd
    
    # Create dataframe
    df_data = []
    for i in range(len(results['predictions'])):
        row = {
            'id': results['ids'][i],
            'text': results['texts'][i],
            'true_label': results['labels'][i],
            'predicted_label': results['predictions'][i],
            'confidence': max(results['probabilities'][i])
        }
        
        # Add probability for each class
        for j, prob in enumerate(results['probabilities'][i]):
            row[f'prob_class_{j}'] = prob
        
        df_data.append(row)
    
    df = pd.DataFrame(df_data)
    df.to_csv(output_path, index=False)


def _compare_models(model1_path: Path, model2_path: Path, dataset: List[Dict[str, Any]],
                   metrics: List[str], batch_size: int, max_length: int):
    """Compare two models on the same test set."""
    console.print("\n[bold blue]Model Comparison[/bold blue]")
    
    # Evaluate both models
    model1, tokenizer1, config1 = _load_model_for_eval(model1_path, False)
    results1 = _evaluate_model(model1, tokenizer1, dataset, batch_size, metrics, False)
    
    model2, tokenizer2, config2 = _load_model_for_eval(model2_path, False)
    results2 = _evaluate_model(model2, tokenizer2, dataset, batch_size, metrics, False)
    
    # Compare results
    table = create_table("Model Comparison")
    table.add_column("Metric", style="cyan")
    table.add_column(f"Model 1\n{model1_path.name}", style="yellow")
    table.add_column(f"Model 2\n{model2_path.name}", style="green")
    table.add_column("Difference", style="magenta")
    
    for metric in results1['metrics']:
        val1 = results1['metrics'][metric]
        val2 = results2['metrics'][metric]
        diff = val2 - val1
        
        # Format difference
        if diff > 0:
            diff_str = f"[green]+{diff:.4f}[/green]"
        elif diff < 0:
            diff_str = f"[red]{diff:.4f}[/red]"
        else:
            diff_str = "0.0000"
        
        table.add_row(
            metric.replace('_', ' ').title(),
            f"{val1:.4f}",
            f"{val2:.4f}",
            diff_str
        )
    
    console.print(table)


def _save_results(results: Dict[str, Any], output_path: Path, 
                 checkpoint: Path, test_data: Path, metrics: List[str]):
    """Save evaluation results to file."""
    # Prepare results for saving
    save_data = {
        'timestamp': datetime.now().isoformat(),
        'checkpoint': str(checkpoint),
        'test_data': str(test_data),
        'metrics': results['metrics'],
        'confusion_matrix': results['confusion_matrix'].tolist(),
        'classification_report': results['classification_report'],
        'total_samples': len(results['predictions']),
        'evaluation_settings': {
            'metrics': metrics
        }
    }
    
    with open(output_path, 'w') as f:
        json.dump(save_data, f, indent=2)


def _log_to_mlflow(results: Dict[str, Any], run_id: str, checkpoint: Path):
    """Log results to MLflow."""
    try:
        import mlflow
        
        with mlflow.start_run(run_id=run_id):
            # Log metrics
            for metric, value in results['metrics'].items():
                mlflow.log_metric(f"test_{metric}", value)
            
            # Log confusion matrix as artifact
            import matplotlib.pyplot as plt
            import seaborn as sns
            
            plt.figure(figsize=(8, 6))
            sns.heatmap(results['confusion_matrix'], annot=True, fmt='d', cmap='Blues')
            plt.title('Confusion Matrix')
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')
            
            cm_path = Path("confusion_matrix.png")
            plt.savefig(cm_path)
            mlflow.log_artifact(cm_path)
            cm_path.unlink()
            
            # Log model info
            mlflow.log_param("evaluation_checkpoint", str(checkpoint))
            mlflow.log_param("test_samples", len(results['predictions']))
        
        console.print("[dim]Results logged to MLflow[/dim]")
        
    except Exception as e:
        print_warning(f"Failed to log to MLflow: {e}")