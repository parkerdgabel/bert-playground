"""Unified evaluation utilities for model evaluation, validation, and submission generation."""

import os
import json
import subprocess
import mlx.core as mx
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Union
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
from rich import box
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, roc_curve, classification_report
)
from loguru import logger
import warnings

# Import unified modules
from models.modernbert import ModernBertModel, ModernBertConfig
from models.modernbert_cnn_hybrid import CNNEnhancedModernBERT
from models.classification import TitanicClassifier
from data.unified_loader import UnifiedTitanicDataPipeline
from utils.mlflow_helper import UnifiedMLflowTracker

console = Console()


class UnifiedModelEvaluator:
    """Unified model evaluator for comprehensive performance analysis and submission generation."""
    
    def __init__(
        self,
        output_dir: str = "evaluation_results",
        submission_dir: str = "kaggle_submissions",
        enable_mlflow: bool = False,
        mlflow_experiment: Optional[str] = None
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.submission_dir = Path(submission_dir)
        self.submission_dir.mkdir(exist_ok=True)
        self.console = console
        
        # Configure logging
        log_file = self.output_dir / f"evaluation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        logger.add(log_file, format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}", level="INFO")
        
        # MLflow tracking
        self.enable_mlflow = enable_mlflow
        self.mlflow_tracker = None
        if enable_mlflow and mlflow_experiment:
            self.mlflow_tracker = UnifiedMLflowTracker(
                experiment_name=mlflow_experiment,
                base_dir="./mlflow"
            )
    
    def find_best_models(self, search_dirs: Optional[List[str]] = None) -> List[Tuple[str, str]]:
        """Find all trained best models in specified directories."""
        models = []
        
        # Default search patterns
        if search_dirs is None:
            search_patterns = [
                ("output/best_model_accuracy", "Best Model (Accuracy)"),
                ("output/best_model_loss", "Best Model (Loss)"),
                ("output/run_*/best_model_accuracy", "Run Best (Accuracy)"),
                ("output/production_*/best_model_accuracy", "Production (Accuracy)"),
                ("output/cnn_hybrid*/best_model", "CNN Hybrid"),
                ("evaluation_*/best_model", "Evaluation Best"),
            ]
        else:
            search_patterns = [(d, Path(d).name) for d in search_dirs]
        
        # Search for models
        from glob import glob
        for pattern, name in search_patterns:
            if "*" in pattern:
                matches = glob(pattern)
                for match in matches:
                    if os.path.exists(match):
                        models.append((match, f"{name} - {Path(match).parent.name}"))
            else:
                if os.path.exists(pattern):
                    models.append((pattern, name))
        
        logger.info(f"Found {len(models)} models to evaluate")
        return models
    
    def load_model(self, model_path: Union[str, Path]) -> Tuple[nn.Module, Dict]:
        """Load a model from checkpoint with automatic type detection."""
        model_path = Path(model_path)
        logger.info(f"Loading model from {model_path}")
        
        # Load model info
        model_info = self.load_model_info(model_path)
        
        # Load config
        config_path = model_path / "config.json"
        if config_path.exists():
            with open(config_path) as f:
                config_dict = json.load(f)
            
            # Determine model type
            if "cnn_kernel_sizes" in config_dict:
                # CNN hybrid model
                from models.modernbert_cnn_hybrid import CNNHybridConfig
                config = CNNHybridConfig(**config_dict)
                base_model = CNNEnhancedModernBERT(config)
            else:
                # Standard ModernBERT
                config = ModernBertConfig(**config_dict)
                base_model = ModernBertModel(config)
        else:
            # Default config
            logger.warning("No config found, using default ModernBERT config")
            config = ModernBertConfig()
            base_model = ModernBertModel(config)
        
        # Wrap with classifier if needed
        if hasattr(base_model, 'classifier'):
            # CNN hybrid already has classifier
            model = base_model
        else:
            # Wrap with TitanicClassifier
            model = TitanicClassifier(base_model)
        
        # Load weights
        weights_path = model_path / "model.safetensors"
        if weights_path.exists():
            from safetensors.mlx import load_model
            load_model(model, str(weights_path))
            logger.info("Loaded model weights from safetensors")
        else:
            logger.warning("No weights found, using random initialization")
        
        return model, model_info
    
    def load_model_info(self, model_path: Path) -> Dict:
        """Load comprehensive model information."""
        info = {
            "path": str(model_path),
            "name": model_path.name,
            "parent": model_path.parent.name,
            "created": datetime.fromtimestamp(model_path.stat().st_mtime).isoformat()
        }
        
        # Load training state
        state_file = model_path / "training_state.json"
        if state_file.exists():
            with open(state_file) as f:
                state = json.load(f)
                info.update({
                    "training_state": state,
                    "best_val_accuracy": state.get("best_val_accuracy", 0),
                    "final_metrics": state.get("metrics", {}),
                    "global_steps": state.get("global_step", 0),
                    "training_history": state.get("training_history", {})
                })
        
        # Load config
        config_file = model_path / "config.json"
        if config_file.exists():
            with open(config_file) as f:
                config = json.load(f)
                info["config"] = config
                info["model_type"] = "cnn_hybrid" if "cnn_kernel_sizes" in config else "standard"
        
        return info
    
    def evaluate_on_validation(
        self,
        model: nn.Module,
        val_path: str = "data/titanic/val.csv",
        batch_size: int = 32
    ) -> Dict[str, float]:
        """Run comprehensive evaluation on validation set."""
        console.print(Panel.fit("📊 Evaluating on Validation Set", style="bold blue"))
        
        # Load validation data
        val_loader = UnifiedTitanicDataPipeline(
            data_path=val_path,
            batch_size=batch_size,
            is_training=False,
            augment=False,
            optimization_level="standard"
        )
        
        # Evaluation mode
        model.eval()
        
        # Collect predictions and labels
        all_preds = []
        all_labels = []
        all_probs = []
        total_loss = 0
        num_batches = 0
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            console=console
        ) as progress:
            task = progress.add_task("Evaluating...", total=len(val_loader))
            
            for batch in val_loader:
                # Forward pass
                outputs = model(batch['input_ids'], batch['attention_mask'], batch['labels'])
                
                # Get predictions
                logits = outputs['logits']
                probs = mx.softmax(logits, axis=-1)
                preds = mx.argmax(logits, axis=-1)
                
                # Collect results
                all_preds.extend(preds.tolist())
                all_labels.extend(batch['labels'].tolist())
                all_probs.extend(probs[:, 1].tolist())  # Probability of positive class
                
                if 'loss' in outputs:
                    total_loss += float(outputs['loss'])
                    num_batches += 1
                
                progress.update(task, advance=1)
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(all_labels, all_preds),
            'precision': precision_score(all_labels, all_preds, average='binary'),
            'recall': recall_score(all_labels, all_preds, average='binary'),
            'f1': f1_score(all_labels, all_preds, average='binary'),
            'avg_loss': total_loss / num_batches if num_batches > 0 else 0
        }
        
        # Calculate AUC if possible
        try:
            metrics['auc'] = roc_auc_score(all_labels, all_probs)
        except:
            metrics['auc'] = 0
        
        # Store detailed results
        self.val_results = {
            'predictions': all_preds,
            'labels': all_labels,
            'probabilities': all_probs,
            'metrics': metrics,
            'confusion_matrix': confusion_matrix(all_labels, all_preds)
        }
        
        return metrics
    
    def generate_submission(
        self,
        model: nn.Module,
        test_path: str = "data/titanic/test.csv",
        submission_name: Optional[str] = None,
        batch_size: int = 32
    ) -> str:
        """Generate Kaggle submission file."""
        console.print(Panel.fit("🚀 Generating Submission", style="bold green"))
        
        # Load test data
        test_df = pd.read_csv(test_path)
        test_loader = UnifiedTitanicDataPipeline(
            data_path=test_path,
            batch_size=batch_size,
            is_training=False,
            augment=False,
            optimization_level="standard"
        )
        
        # Generate predictions
        model.eval()
        all_preds = []
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            console=console
        ) as progress:
            task = progress.add_task("Predicting...", total=len(test_loader))
            
            for batch in test_loader:
                outputs = model(batch['input_ids'], batch['attention_mask'])
                preds = mx.argmax(outputs['logits'], axis=-1)
                all_preds.extend(preds.tolist())
                progress.update(task, advance=1)
        
        # Create submission DataFrame
        submission = pd.DataFrame({
            'PassengerId': test_df['PassengerId'],
            'Survived': all_preds
        })
        
        # Save submission
        if submission_name is None:
            submission_name = f"submission_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        
        submission_path = self.submission_dir / submission_name
        submission.to_csv(submission_path, index=False)
        
        logger.info(f"Saved submission to {submission_path}")
        return str(submission_path)
    
    def visualize_results(self, save_path: Optional[Path] = None):
        """Create comprehensive visualization of evaluation results."""
        if not hasattr(self, 'val_results'):
            logger.warning("No validation results to visualize")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle('Model Evaluation Results', fontsize=16)
        
        # Confusion Matrix
        cm = self.val_results['confusion_matrix']
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0, 0])
        axes[0, 0].set_title('Confusion Matrix')
        axes[0, 0].set_xlabel('Predicted')
        axes[0, 0].set_ylabel('Actual')
        
        # ROC Curve
        if 'probabilities' in self.val_results:
            fpr, tpr, _ = roc_curve(self.val_results['labels'], self.val_results['probabilities'])
            auc = self.val_results['metrics'].get('auc', 0)
            axes[0, 1].plot(fpr, tpr, label=f'ROC (AUC = {auc:.3f})')
            axes[0, 1].plot([0, 1], [0, 1], 'k--')
            axes[0, 1].set_xlabel('False Positive Rate')
            axes[0, 1].set_ylabel('True Positive Rate')
            axes[0, 1].set_title('ROC Curve')
            axes[0, 1].legend()
        
        # Metrics Bar Chart
        metrics = self.val_results['metrics']
        metric_names = ['accuracy', 'precision', 'recall', 'f1']
        metric_values = [metrics.get(m, 0) for m in metric_names]
        axes[1, 0].bar(metric_names, metric_values)
        axes[1, 0].set_ylim(0, 1)
        axes[1, 0].set_title('Performance Metrics')
        axes[1, 0].set_ylabel('Score')
        
        # Probability Distribution
        if 'probabilities' in self.val_results:
            probs = self.val_results['probabilities']
            labels = self.val_results['labels']
            axes[1, 1].hist([p for p, l in zip(probs, labels) if l == 0], 
                          alpha=0.5, label='Class 0', bins=20)
            axes[1, 1].hist([p for p, l in zip(probs, labels) if l == 1], 
                          alpha=0.5, label='Class 1', bins=20)
            axes[1, 1].set_xlabel('Predicted Probability')
            axes[1, 1].set_ylabel('Count')
            axes[1, 1].set_title('Prediction Probability Distribution')
            axes[1, 1].legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"Saved visualization to {save_path}")
        else:
            plt.show()
    
    def create_evaluation_report(self, model_info: Dict, metrics: Dict) -> Table:
        """Create a rich table with evaluation results."""
        table = Table(title="Model Evaluation Report", box=box.ROUNDED)
        
        # Add columns
        table.add_column("Metric", style="cyan", no_wrap=True)
        table.add_column("Value", style="magenta")
        
        # Model info
        table.add_row("Model Path", str(model_info['path']))
        table.add_row("Model Type", model_info.get('model_type', 'unknown'))
        table.add_row("Training Steps", str(model_info.get('global_steps', 'N/A')))
        table.add_section()
        
        # Performance metrics
        for metric, value in metrics.items():
            if isinstance(value, float):
                table.add_row(metric.title(), f"{value:.4f}")
            else:
                table.add_row(metric.title(), str(value))
        
        return table
    
    def evaluate_and_submit(
        self,
        model_path: str,
        submit_to_kaggle: bool = False,
        competition: str = "titanic"
    ) -> Dict:
        """Complete evaluation pipeline with optional Kaggle submission."""
        results = {}
        
        # Load model
        model, model_info = self.load_model(model_path)
        results['model_info'] = model_info
        
        # Start MLflow run if enabled
        if self.mlflow_tracker:
            self.mlflow_tracker.start_run(
                run_name=f"eval_{Path(model_path).name}",
                tags={"evaluation": "true", "model_path": model_path}
            )
            self.mlflow_tracker.log_params({
                "model_type": model_info.get('model_type', 'unknown'),
                "model_path": model_path
            })
        
        # Evaluate on validation
        val_metrics = self.evaluate_on_validation(model)
        results['validation_metrics'] = val_metrics
        
        # Log metrics to MLflow
        if self.mlflow_tracker:
            self.mlflow_tracker.log_metrics(val_metrics)
        
        # Display results
        table = self.create_evaluation_report(model_info, val_metrics)
        console.print(table)
        
        # Generate submission
        submission_path = self.generate_submission(model)
        results['submission_path'] = submission_path
        
        # Visualize results
        viz_path = self.output_dir / f"evaluation_{Path(model_path).name}.png"
        self.visualize_results(viz_path)
        
        # Submit to Kaggle if requested
        if submit_to_kaggle:
            self.submit_to_kaggle(submission_path, competition)
        
        # End MLflow run
        if self.mlflow_tracker:
            self.mlflow_tracker.log_artifacts([str(viz_path), submission_path])
            self.mlflow_tracker.end_run()
        
        return results
    
    def submit_to_kaggle(self, submission_path: str, competition: str = "titanic"):
        """Submit to Kaggle competition."""
        console.print(Panel.fit("📤 Submitting to Kaggle", style="bold yellow"))
        
        try:
            cmd = [
                "kaggle", "competitions", "submit",
                "-c", competition,
                "-f", submission_path,
                "-m", f"MLX ModernBERT submission {datetime.now().strftime('%Y-%m-%d %H:%M')}"
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                console.print("✅ Successfully submitted to Kaggle!", style="bold green")
                logger.info(f"Kaggle submission successful: {result.stdout}")
            else:
                console.print("❌ Submission failed!", style="bold red")
                logger.error(f"Kaggle submission failed: {result.stderr}")
        except Exception as e:
            console.print(f"❌ Error submitting to Kaggle: {e}", style="bold red")
            logger.error(f"Kaggle submission error: {e}")
    
    def compare_models(self, model_paths: List[str]) -> pd.DataFrame:
        """Compare multiple models and return comparison DataFrame."""
        results = []
        
        for model_path in model_paths:
            console.print(f"\n🔍 Evaluating: {model_path}")
            
            try:
                # Load and evaluate model
                model, model_info = self.load_model(model_path)
                metrics = self.evaluate_on_validation(model)
                
                # Combine results
                result = {
                    'model_path': model_path,
                    'model_type': model_info.get('model_type', 'unknown'),
                    **metrics
                }
                results.append(result)
                
            except Exception as e:
                logger.error(f"Failed to evaluate {model_path}: {e}")
                console.print(f"❌ Error: {e}", style="red")
        
        # Create comparison DataFrame
        df = pd.DataFrame(results)
        df = df.sort_values('accuracy', ascending=False)
        
        # Save comparison
        comparison_path = self.output_dir / f"model_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        df.to_csv(comparison_path, index=False)
        
        return df


# Convenience functions
def evaluate_single_model(
    model_path: str,
    submit: bool = False,
    enable_mlflow: bool = False
) -> Dict:
    """Evaluate a single model with optional submission."""
    evaluator = UnifiedModelEvaluator(enable_mlflow=enable_mlflow)
    return evaluator.evaluate_and_submit(model_path, submit_to_kaggle=submit)


def compare_all_models(output_dir: str = "evaluation_results") -> pd.DataFrame:
    """Find and compare all available models."""
    evaluator = UnifiedModelEvaluator(output_dir=output_dir)
    models = evaluator.find_best_models()
    
    if not models:
        logger.warning("No models found to evaluate")
        return pd.DataFrame()
    
    model_paths = [m[0] for m in models]
    return evaluator.compare_models(model_paths)


def generate_submission_only(
    model_path: str,
    test_path: str = "data/titanic/test.csv",
    submission_name: Optional[str] = None
) -> str:
    """Generate submission file without full evaluation."""
    evaluator = UnifiedModelEvaluator()
    model, _ = evaluator.load_model(model_path)
    return evaluator.generate_submission(model, test_path, submission_name)