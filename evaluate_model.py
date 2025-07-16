#!/usr/bin/env python3
"""Evaluate a single trained model with comprehensive performance analysis."""

import os
import json
import argparse
import mlx.core as mx
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
from rich import box
import matplotlib.pyplot as plt
import seaborn as sns
from loguru import logger

# Import model classes
from models.modernbert_mlx import ModernBertMLX
from models.modernbert_optimized import OptimizedModernBertMLX
from models.modernbert_cnn_hybrid import CNNEnhancedModernBERT
from models.classification_head import TitanicClassifier
from data.optimized_loader import OptimizedTitanicDataPipeline

console = Console()

class ModelEvaluator:
    """Comprehensive evaluation for a single model."""
    
    def __init__(self, model_path: str, output_dir: str = "evaluation_results"):
        self.model_path = Path(model_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.console = console
        
        # Configure logging
        log_file = self.output_dir / f"evaluation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        logger.add(log_file, format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}", level="INFO")
        
    def load_model_info(self):
        """Load model information and metrics."""
        info = {
            "path": str(self.model_path),
            "name": self.model_path.name,
            "parent": self.model_path.parent.name
        }
        
        # Load training state if available
        state_file = self.model_path / "training_state.json"
        if state_file.exists():
            with open(state_file) as f:
                state = json.load(f)
                info["training_state"] = state
                info["best_val_accuracy"] = state.get("best_val_accuracy", 0)
                info["final_metrics"] = state.get("metrics", {})
                info["global_steps"] = state.get("global_step", 0)
        
        # Load config if available
        config_file = self.model_path / "config.json"
        if config_file.exists():
            with open(config_file) as f:
                info["config"] = json.load(f)
                info["model_type"] = "cnn_hybrid" if "cnn_kernel_sizes" in info["config"] else "standard"
        else:
            info["model_type"] = "standard"
        
        # Check for model files
        if (self.model_path / "model.safetensors").exists():
            info["model_format"] = "safetensors"
        elif (self.model_path / "bert").exists():
            info["model_format"] = "legacy"
        else:
            info["model_format"] = "unknown"
        
        return info
    
    def evaluate_on_validation(self, batch_size: int = 32):
        """Run evaluation on validation set."""
        console.print(Panel.fit("📊 Evaluating on Validation Set", style="bold blue"))
        
        # Load validation data
        val_loader = OptimizedTitanicDataPipeline(
            data_path="data/titanic/val.csv",
            batch_size=batch_size,
            is_training=False,
            augment=False,
            num_threads=2,
            prefetch_size=2,
        )
        
        # Load model based on format
        model_info = self.load_model_info()
        
        try:
            if model_info["model_format"] == "safetensors":
                # Load CNN hybrid model
                if model_info["model_type"] == "cnn_hybrid":
                    from models.modernbert_cnn_hybrid import CNNEnhancedModernBERT, CNNHybridConfig
                    # Load config and fix hidden_size for embeddings if needed
                    config = CNNHybridConfig(**model_info["config"])
                    
                    # Fix for models saved with incorrect hidden_size
                    # The base ModernBERT uses 768, but config might have been saved with fusion_hidden_size
                    if config.hidden_size != 768 and hasattr(config, 'fusion_hidden_size'):
                        logger.warning(f"Correcting config.hidden_size from {config.hidden_size} to 768")
                        config.hidden_size = 768  # Base BERT hidden size
                    
                    model = CNNEnhancedModernBERT(config)
                    
                    # Load weights
                    weights_path = self.model_path / "model.safetensors"
                    if weights_path.exists():
                        weights = mx.load(str(weights_path))
                        model.load_weights(list(weights.items()))
                        logger.info(f"Loaded {len(weights)} parameters from {weights_path}")
                else:
                    from models.modernbert_optimized import OptimizedModernBertMLX
                    model = OptimizedModernBertMLX.from_pretrained(str(self.model_path))
                
                # CNN hybrid model already has classifier built-in, don't wrap it
                    
            elif model_info["model_format"] == "legacy":
                # Load legacy format model
                bert_model = ModernBertMLX.from_pretrained(str(self.model_path / "bert"))
                model = TitanicClassifier(bert_model)
                
                # Load classifier weights
                classifier_weights = self.model_path / "classifier_weights.npz"
                if classifier_weights.exists():
                    weights = mx.load(str(classifier_weights))
                    # Apply classifier weights
                    model.classifier.weight = weights["weight"]
                    model.classifier.bias = weights["bias"]
            else:
                raise ValueError(f"Unknown model format: {model_info['model_format']}")
            
            console.print(f"✅ Loaded model: {model_info['model_type']} format: {model_info['model_format']}")
            
        except Exception as e:
            console.print(f"[red]❌ Failed to load model: {e}[/red]")
            return None
        
        # Run evaluation
        model.eval()
        all_predictions = []
        all_labels = []
        all_probs = []
        total_loss = 0
        num_batches = val_loader.get_num_batches()
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            console=self.console
        ) as progress:
            task = progress.add_task("Evaluating batches...", total=num_batches)
            
            for batch_idx, batch in enumerate(val_loader.get_dataloader()()):
                outputs = model(
                    input_ids=batch['input_ids'],
                    attention_mask=batch['attention_mask'],
                    labels=batch['labels']
                )
                
                # Get predictions
                probs = mx.softmax(outputs['logits'], axis=-1)
                predictions = mx.argmax(outputs['logits'], axis=1)
                
                all_predictions.extend(predictions.tolist())
                all_labels.extend(batch['labels'].tolist())
                all_probs.extend(probs[:, 1].tolist())  # Probability of survival
                
                if outputs.get('loss') is not None:
                    total_loss += outputs['loss'].item()
                
                progress.update(task, advance=1)
        
        # Calculate metrics
        accuracy = sum(p == l for p, l in zip(all_predictions, all_labels)) / len(all_labels)
        avg_loss = total_loss / num_batches if num_batches > 0 else 0
        
        # Calculate additional metrics
        from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score
        
        precision = precision_score(all_labels, all_predictions)
        recall = recall_score(all_labels, all_predictions)
        f1 = f1_score(all_labels, all_predictions)
        auc = roc_auc_score(all_labels, all_probs)
        cm = confusion_matrix(all_labels, all_predictions)
        
        metrics = {
            "accuracy": accuracy,
            "loss": avg_loss,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "auc": auc,
            "confusion_matrix": cm.tolist(),
            "num_samples": len(all_labels),
            "predictions": all_predictions,
            "labels": all_labels,
            "probabilities": all_probs
        }
        
        return metrics
    
    def create_visualizations(self, metrics: dict, model_info: dict):
        """Create performance visualizations."""
        viz_dir = self.output_dir / "visualizations"
        viz_dir.mkdir(exist_ok=True)
        
        # 1. Confusion Matrix
        plt.figure(figsize=(8, 6))
        cm = np.array(metrics["confusion_matrix"])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=['Not Survived', 'Survived'],
                    yticklabels=['Not Survived', 'Survived'])
        plt.title(f'Confusion Matrix - {model_info["name"]}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(viz_dir / "confusion_matrix.png", dpi=150)
        plt.close()
        
        # 2. ROC Curve
        from sklearn.metrics import roc_curve
        fpr, tpr, _ = roc_curve(metrics["labels"], metrics["probabilities"])
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, 
                label=f'ROC curve (AUC = {metrics["auc"]:.3f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve - {model_info["name"]}')
        plt.legend(loc="lower right")
        plt.tight_layout()
        plt.savefig(viz_dir / "roc_curve.png", dpi=150)
        plt.close()
        
        # 3. Probability Distribution
        plt.figure(figsize=(10, 6))
        survived_probs = [p for p, l in zip(metrics["probabilities"], metrics["labels"]) if l == 1]
        not_survived_probs = [p for p, l in zip(metrics["probabilities"], metrics["labels"]) if l == 0]
        
        plt.hist(not_survived_probs, bins=30, alpha=0.5, label='Not Survived', color='red')
        plt.hist(survived_probs, bins=30, alpha=0.5, label='Survived', color='green')
        plt.xlabel('Predicted Probability of Survival')
        plt.ylabel('Count')
        plt.title(f'Prediction Probability Distribution - {model_info["name"]}')
        plt.legend()
        plt.tight_layout()
        plt.savefig(viz_dir / "probability_distribution.png", dpi=150)
        plt.close()
        
        # 4. Training History (if available)
        if "training_state" in model_info and "training_history" in model_info["training_state"]:
            history = model_info["training_state"]["training_history"]
            
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            fig.suptitle(f'Training History - {model_info["name"]}', fontsize=16)
            
            # Plot losses and accuracies
            if history.get("train_loss"):
                axes[0, 0].plot(history["train_loss"])
                axes[0, 0].set_title("Training Loss")
                axes[0, 0].set_xlabel("Step")
                axes[0, 0].set_ylabel("Loss")
            
            if history.get("train_accuracy"):
                axes[0, 1].plot(history["train_accuracy"])
                axes[0, 1].set_title("Training Accuracy")
                axes[0, 1].set_xlabel("Step")
                axes[0, 1].set_ylabel("Accuracy")
            
            if history.get("val_loss"):
                axes[1, 0].plot(history["val_loss"], color='orange')
                axes[1, 0].set_title("Validation Loss")
                axes[1, 0].set_xlabel("Step")
                axes[1, 0].set_ylabel("Loss")
            
            if history.get("val_accuracy"):
                axes[1, 1].plot(history["val_accuracy"], color='orange')
                axes[1, 1].set_title("Validation Accuracy")
                axes[1, 1].set_xlabel("Step")
                axes[1, 1].set_ylabel("Accuracy")
            
            plt.tight_layout()
            plt.savefig(viz_dir / "training_history.png", dpi=150)
            plt.close()
        
        console.print(f"📊 Visualizations saved to: {viz_dir}")
    
    def display_results(self, metrics: dict, model_info: dict):
        """Display evaluation results in a rich format."""
        # Performance metrics table
        table = Table(title="📈 Model Performance Metrics", box=box.ROUNDED)
        table.add_column("Metric", style="cyan", no_wrap=True)
        table.add_column("Value", justify="right", style="bold green")
        
        table.add_row("Accuracy", f"{metrics['accuracy']:.4f}")
        table.add_row("Precision", f"{metrics['precision']:.4f}")
        table.add_row("Recall", f"{metrics['recall']:.4f}")
        table.add_row("F1 Score", f"{metrics['f1_score']:.4f}")
        table.add_row("AUC", f"{metrics['auc']:.4f}")
        table.add_row("Average Loss", f"{metrics['loss']:.4f}")
        table.add_row("Total Samples", str(metrics['num_samples']))
        
        console.print(table)
        
        # Confusion matrix
        cm = np.array(metrics["confusion_matrix"])
        cm_table = Table(title="🎯 Confusion Matrix", box=box.ROUNDED)
        cm_table.add_column("", style="cyan")
        cm_table.add_column("Pred: Not Survived", justify="center")
        cm_table.add_column("Pred: Survived", justify="center")
        
        cm_table.add_row("True: Not Survived", str(cm[0, 0]), str(cm[0, 1]))
        cm_table.add_row("True: Survived", str(cm[1, 0]), str(cm[1, 1]))
        
        console.print(cm_table)
        
        # Model info panel
        info_text = f"""
Model Path: {model_info['path']}
Model Type: {model_info['model_type']}
Model Format: {model_info['model_format']}
Training Steps: {model_info.get('global_steps', 'N/A')}
Best Val Accuracy (Training): {model_info.get('best_val_accuracy', 'N/A')}
"""
        console.print(Panel(info_text.strip(), title="🔍 Model Information", box=box.ROUNDED))
    
    def save_report(self, metrics: dict, model_info: dict):
        """Save detailed evaluation report."""
        report_path = self.output_dir / f"evaluation_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        report = {
            "timestamp": datetime.now().isoformat(),
            "model_info": model_info,
            "metrics": {k: v for k, v in metrics.items() if k not in ["predictions", "labels", "probabilities"]},
            "confusion_matrix": metrics["confusion_matrix"]
        }
        
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)
        
        console.print(f"💾 Report saved to: {report_path}")
        
        # Also save predictions
        pred_df = pd.DataFrame({
            "PassengerId": range(len(metrics["predictions"])),
            "TrueLabel": metrics["labels"],
            "Prediction": metrics["predictions"],
            "Probability": metrics["probabilities"]
        })
        pred_path = self.output_dir / "validation_predictions.csv"
        pred_df.to_csv(pred_path, index=False)
        console.print(f"💾 Predictions saved to: {pred_path}")
    
    def run(self):
        """Run complete evaluation pipeline."""
        console.print(Panel.fit(
            f"🚀 Model Evaluation Pipeline\n\nModel: {self.model_path}",
            style="bold green"
        ))
        
        # Load model info
        console.print("\n" + "="*60)
        model_info = self.load_model_info()
        
        # Run validation evaluation
        metrics = self.evaluate_on_validation()
        
        if metrics:
            # Display results
            console.print("\n" + "="*60)
            self.display_results(metrics, model_info)
            
            # Create visualizations
            console.print("\n" + "="*60)
            self.create_visualizations(metrics, model_info)
            
            # Save report
            self.save_report(metrics, model_info)
            
            console.print(Panel.fit(
                f"✅ Evaluation Complete!\n\nResults saved to: {self.output_dir}",
                style="bold green"
            ))
        else:
            console.print(Panel.fit(
                "❌ Evaluation Failed",
                style="bold red"
            ))


def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained model")
    parser.add_argument("model_path", help="Path to model checkpoint directory")
    parser.add_argument("--output-dir", default="evaluation_results", 
                       help="Output directory for results")
    parser.add_argument("--batch-size", type=int, default=32,
                       help="Batch size for evaluation")
    
    args = parser.parse_args()
    
    evaluator = ModelEvaluator(args.model_path, args.output_dir)
    evaluator.run()


if __name__ == "__main__":
    main()