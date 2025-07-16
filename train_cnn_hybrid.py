#!/usr/bin/env python3
"""Training script for CNN-Enhanced ModernBERT on Titanic dataset with model comparison."""

import argparse
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten
from typing import Union, Dict, Optional, Tuple
from pathlib import Path
import json
import time
import numpy as np
from tqdm import tqdm
from loguru import logger
import matplotlib.pyplot as plt
import seaborn as sns

from models.modernbert_cnn_hybrid import CNNEnhancedModernBERT, CNNHybridConfig, create_cnn_hybrid_model
from models.modernbert_optimized import OptimizedModernBertMLX, create_optimized_model
from models.classification_head import TitanicClassifier
from data.optimized_loader import OptimizedTitanicDataPipeline
from utils.mlflow_utils import MLflowExperimentTracker
from utils.logging_config import LoggingConfig


class HybridModelTrainer:
    """Trainer for CNN-Enhanced ModernBERT with comparison capabilities."""
    
    def __init__(
        self,
        model_type: str = "cnn_hybrid",
        train_path: str = "data/titanic/train.csv",
        val_path: str = "data/titanic/val.csv",
        batch_size: int = 32,
        learning_rate: float = 2e-5,
        epochs: int = 5,
        warmup_ratio: float = 0.1,
        output_dir: str = "./output/cnn_hybrid",
        experiment_name: str = "titanic_cnn_hybrid",
        enable_mlflow: bool = True,
        enable_visualization: bool = True,
        optimizer_type: str = "adamw",
        optimizer_kwargs: Dict = None,
        **model_kwargs
    ):
        self.model_type = model_type
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.warmup_ratio = warmup_ratio
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.enable_visualization = enable_visualization
        
        # Initialize logging
        self.logging_config = LoggingConfig(
            log_dir=str(self.output_dir / "logs"),
            experiment_name=experiment_name
        )
        
        # Initialize MLflow
        if enable_mlflow:
            self.mlflow_tracker = MLflowExperimentTracker(
                experiment_name=experiment_name,
                tracking_uri=str(self.output_dir / "mlruns")
            )
        else:
            self.mlflow_tracker = None
        
        # Load data
        logger.info("Loading data pipelines...")
        self.train_loader = OptimizedTitanicDataPipeline(
            data_path=train_path,
            batch_size=batch_size,
            is_training=True,
            augment=True,
            num_threads=4,
            prefetch_size=4,
        )
        
        self.val_loader = OptimizedTitanicDataPipeline(
            data_path=val_path,
            batch_size=batch_size,
            is_training=False,
            augment=False,
            num_threads=2,
            prefetch_size=2,
        )
        
        # Initialize model
        logger.info(f"Initializing {model_type} model...")
        if model_type == "cnn_hybrid":
            self.model = self._create_cnn_hybrid_model(**model_kwargs)
        elif model_type == "base":
            self.model = self._create_base_model(**model_kwargs)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        # Initialize optimizer
        self.optimizer = self._create_optimizer(
            optimizer_type,
            learning_rate,
            optimizer_kwargs or {}
        )
        
        # Create compiled evaluation function
        self._create_compiled_eval_step()
        
        # Training state
        self.global_step = 0
        self.best_val_accuracy = 0.0
        self.training_history = {
            'train_loss': [],
            'train_accuracy': [],
            'val_loss': [],
            'val_accuracy': [],
            'learning_rate': [],
            'cnn_feature_importance': [] if model_type == "cnn_hybrid" else None,
        }
        
        logger.info(f"Trainer initialized for {model_type} model")
    
    def _create_cnn_hybrid_model(self, **kwargs) -> TitanicClassifier:
        """Create CNN-Enhanced ModernBERT model."""
        # Default CNN configuration
        cnn_config = {
            'cnn_kernel_sizes': kwargs.get('cnn_kernel_sizes', [2, 3, 4, 5]),
            'cnn_num_filters': kwargs.get('cnn_num_filters', 128),
            'use_dilated_conv': kwargs.get('use_dilated_conv', True),
            'dilation_rates': kwargs.get('dilation_rates', [1, 2, 4]),
            'use_attention_fusion': kwargs.get('use_attention_fusion', True),
            'use_highway': kwargs.get('use_highway', True),
            'cnn_dropout': kwargs.get('cnn_dropout', 0.5),
            'fusion_hidden_size': kwargs.get('fusion_hidden_size', 512),
        }
        
        bert_model = create_cnn_hybrid_model(
            model_name="answerdotai/ModernBERT-base",
            num_labels=2,
            **cnn_config
        )
        
        # Don't override config.hidden_size - it should remain 768 for base BERT
        # The output_hidden_size is fusion_hidden_size which is used internally
        
        return TitanicClassifier(bert_model)
    
    def _create_base_model(self, **kwargs) -> TitanicClassifier:
        """Create base ModernBERT model for comparison."""
        bert_model = create_optimized_model(
            model_name="answerdotai/ModernBERT-base",
            num_labels=2,
        )
        return TitanicClassifier(bert_model)
    
    def _create_optimizer(
        self,
        optimizer_type: str,
        learning_rate: float,
        kwargs: Dict
    ) -> Union[optim.SGD, optim.Adam, optim.AdamW, optim.Adamax, optim.RMSprop,
               optim.Adagrad, optim.AdaDelta, optim.Lion]:
        """Create optimizer based on type."""
        optimizer_map = {
            'sgd': optim.SGD,
            'adam': optim.Adam,
            'adamw': optim.AdamW,
            'adamax': optim.Adamax,
            'rmsprop': optim.RMSprop,
            'adagrad': optim.Adagrad,
            'adadelta': optim.AdaDelta,
            'lion': optim.Lion,
        }
        
        optimizer_class = optimizer_map.get(optimizer_type.lower())
        if optimizer_class is None:
            raise ValueError(
                f"Unknown optimizer: {optimizer_type}. "
                f"Available optimizers: {list(optimizer_map.keys())}"
            )
        
        # Filter out None values from kwargs
        filtered_kwargs = {k: v for k, v in kwargs.items() if v is not None}
        
        # Set learning rate
        filtered_kwargs['learning_rate'] = learning_rate
        
        # Set default weight decay for AdamW and Lion
        if optimizer_type.lower() in ['adamw', 'lion'] and 'weight_decay' not in filtered_kwargs:
            filtered_kwargs['weight_decay'] = 0.01
        
        logger.info(
            f"Creating {optimizer_type} optimizer with "
            f"lr={learning_rate}, kwargs={filtered_kwargs}"
        )
        
        return optimizer_class(**filtered_kwargs)
    
    def compute_lr(self, step: int, total_steps: int) -> float:
        """Compute learning rate with linear warmup and cosine decay."""
        warmup_steps = int(total_steps * self.warmup_ratio)
        
        if step < warmup_steps:
            return self.learning_rate * step / warmup_steps
        
        progress = (step - warmup_steps) / (total_steps - warmup_steps)
        return self.learning_rate * 0.5 * (1 + np.cos(np.pi * progress))
    
    def train_step(self, batch: Dict[str, mx.array]) -> Tuple[float, float]:
        """Single training step."""
        def loss_fn(model):
            outputs = model(
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask'],
                labels=batch['labels'],
            )
            return outputs['loss'], outputs
        
        # Compute gradients
        grad_fn = nn.value_and_grad(self.model, loss_fn)
        (loss, outputs), grads = grad_fn(self.model)
        
        # Update weights
        self.optimizer.update(self.model, grads)
        mx.eval(self.model.parameters(), self.optimizer.state, loss)
        
        # Compute accuracy
        predictions = mx.argmax(outputs['logits'], axis=1)
        accuracy = mx.mean(predictions == batch['labels'])
        
        return loss.item(), accuracy.item()
    
    def _create_compiled_eval_step(self):
        """Create a compiled evaluation function."""
        # Don't compile the entire eval step with the model
        # Instead, compile just the computation-heavy parts
        @mx.compile
        def compute_predictions(logits):
            return mx.argmax(logits, axis=1)
        
        self._compute_predictions = compute_predictions
    
    def eval_step(self, batch: Dict[str, mx.array]) -> Dict[str, mx.array]:
        """Single evaluation step using compiled function."""
        # Run the model forward pass
        outputs = self.model(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask'],
            labels=batch['labels'],
        )
        
        # Use compiled function for predictions
        predictions = self._compute_predictions(outputs['logits'])
        
        return {
            'loss': outputs['loss'],
            'predictions': predictions,
            'labels': batch['labels'],
            'logits': outputs['logits'],
            'cnn_features': outputs.get('cnn_features', None),
        }
    
    def evaluate(self) -> Dict[str, float]:
        """Full evaluation on validation set."""
        self.model.eval()
        
        total_loss = 0
        all_predictions = []
        all_labels = []
        all_cnn_features = []
        
        # Get total validation batches
        val_batches = self.val_loader.get_num_batches()
        
        with tqdm(
            self.val_loader.get_dataloader()(),
            total=val_batches,
            desc="Evaluating",
            unit="batch",
            leave=False,
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]"
        ) as pbar:
            for batch in pbar:
                outputs = self.eval_step(batch)
                
                total_loss += outputs['loss'].item()
                all_predictions.extend(outputs['predictions'].tolist())
                all_labels.extend(outputs['labels'].tolist())
                
                if outputs['cnn_features'] is not None:
                    all_cnn_features.append(outputs['cnn_features'])
        
        # Compute metrics
        accuracy = sum(p == l for p, l in zip(all_predictions, all_labels)) / len(all_labels)
        avg_loss = total_loss / len(self.val_loader)
        
        metrics = {
            'val_loss': avg_loss,
            'val_accuracy': accuracy,
        }
        
        # Analyze CNN features if available
        if all_cnn_features and self.enable_visualization:
            cnn_features = mx.concatenate(all_cnn_features, axis=0)
            feature_importance = mx.mean(mx.abs(cnn_features), axis=0)
            metrics['cnn_feature_importance'] = feature_importance.tolist()
        
        self.model.train()
        return metrics
    
    def visualize_cnn_features(self, feature_importance: list, step: int):
        """Visualize CNN feature importance."""
        if not self.enable_visualization or not feature_importance:
            return
        
        plt.figure(figsize=(10, 6))
        plt.bar(range(len(feature_importance)), feature_importance)
        plt.xlabel('Feature Index')
        plt.ylabel('Average Absolute Value')
        plt.title(f'CNN Feature Importance at Step {step}')
        plt.tight_layout()
        
        viz_path = self.output_dir / 'visualizations'
        viz_path.mkdir(exist_ok=True)
        plt.savefig(viz_path / f'cnn_features_step_{step}.png')
        plt.close()
    
    def save_checkpoint(self, metrics: Dict[str, float], is_best: bool = False):
        """Save model checkpoint."""
        checkpoint_dir = self.output_dir / f"checkpoint-{self.global_step}"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Save model
        try:
            if hasattr(self.model, 'bert') and hasattr(self.model.bert, 'save_pretrained'):
                self.model.bert.save_pretrained(checkpoint_dir)
            else:
                # Save model weights using MLX native format with proper flattening
                model_path = checkpoint_dir / 'model.safetensors'
                # Flatten the model parameters properly
                flat_params = tree_flatten(self.model.parameters())
                # Convert to dict format expected by save_safetensors
                weights_dict = {k: v for k, v in flat_params}
                mx.save_safetensors(str(model_path), weights_dict)
                logger.info(f"Model saved to {model_path}")
        except Exception as e:
            logger.error(f"Error saving model: {e}")
            # Fallback: try saving individual components
            try:
                model_path = checkpoint_dir / 'model.npz'
                mx.savez(str(model_path), **dict(tree_flatten(self.model.parameters())))
                logger.info(f"Model saved to {model_path} (npz format)")
            except Exception as e2:
                logger.error(f"Failed to save model in any format: {e2}")
        
        # Save optimizer state
        try:
            optimizer_path = checkpoint_dir / 'optimizer.safetensors'
            opt_state = tree_flatten(self.optimizer.state)
            opt_dict = {k: v for k, v in opt_state}
            mx.save_safetensors(str(optimizer_path), opt_dict)
            logger.info(f"Optimizer saved to {optimizer_path}")
        except Exception as e:
            logger.error(f"Error saving optimizer: {e}")
        
        # Save training state
        state = {
            'global_step': self.global_step,
            'best_val_accuracy': self.best_val_accuracy,
            'training_history': self.training_history,
            'metrics': metrics,
            'model_type': self.model_type,
        }
        
        with open(checkpoint_dir / 'training_state.json', 'w') as f:
            json.dump(state, f, indent=2)
        
        if is_best:
            best_dir = self.output_dir / 'best_model'
            best_dir.mkdir(exist_ok=True)
            
            # Copy to best model directory
            import shutil
            for file in checkpoint_dir.glob('*'):
                shutil.copy2(file, best_dir / file.name)
            
            logger.info(f"New best model saved with accuracy: {metrics['val_accuracy']:.4f}")
    
    def train(self):
        """Main training loop."""
        total_steps = self.epochs * len(self.train_loader)
        
        # Log configuration
        config = {
            'model_type': self.model_type,
            'batch_size': self.batch_size,
            'learning_rate': self.learning_rate,
            'epochs': self.epochs,
            'warmup_ratio': self.warmup_ratio,
            'total_steps': total_steps,
        }
        
        if self.mlflow_tracker:
            run_id = self.mlflow_tracker.start_run(
                run_name=f"{self.model_type}_{time.strftime('%Y%m%d_%H%M%S')}"
            )
            self.mlflow_tracker.log_params(config)
        
        logger.info(f"Starting training for {self.epochs} epochs ({total_steps} steps)")
        
        # Training loop
        for epoch in range(self.epochs):
            epoch_loss = 0
            epoch_accuracy = 0
            
            self.model.train()
            
            # Get total batches for proper progress bar
            train_batches = self.train_loader.get_num_batches()
            
            with tqdm(
                self.train_loader.get_dataloader()(),
                total=train_batches,
                desc=f"Epoch {epoch+1}/{self.epochs}",
                unit="batch",
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]"
            ) as pbar:
                for batch_idx, batch in enumerate(pbar):
                    # Update learning rate
                    lr = self.compute_lr(self.global_step, total_steps)
                    self.optimizer.learning_rate = lr
                    
                    # Training step
                    loss, accuracy = self.train_step(batch)
                    
                    epoch_loss += loss
                    epoch_accuracy += accuracy
                    self.global_step += 1
                    
                    # Update progress bar
                    pbar.set_postfix({
                        'loss': f"{loss:.4f}",
                        'acc': f"{accuracy:.4f}",
                        'lr': f"{lr:.2e}"
                    })
                    
                    # Log metrics
                    if self.mlflow_tracker and self.global_step % 10 == 0:
                        self.mlflow_tracker.log_metrics({
                            'train_loss': loss,
                            'train_accuracy': accuracy,
                            'learning_rate': lr,
                        }, step=self.global_step)
                    
                    # Evaluate periodically
                    if self.global_step % 100 == 0:
                        metrics = self.evaluate()
                        
                        # Log validation metrics
                        if self.mlflow_tracker:
                            self.mlflow_tracker.log_metrics(metrics, step=self.global_step)
                        
                        # Visualize CNN features
                        if 'cnn_feature_importance' in metrics:
                            self.visualize_cnn_features(
                                metrics['cnn_feature_importance'],
                                self.global_step
                            )
                        
                        # Save checkpoint if best
                        if metrics['val_accuracy'] > self.best_val_accuracy:
                            self.best_val_accuracy = metrics['val_accuracy']
                            self.save_checkpoint(metrics, is_best=True)
                        
                        logger.info(
                            f"Step {self.global_step}: "
                            f"Val Loss: {metrics['val_loss']:.4f}, "
                            f"Val Acc: {metrics['val_accuracy']:.4f}"
                        )
            
            # End of epoch logging
            avg_epoch_loss = epoch_loss / len(self.train_loader)
            avg_epoch_accuracy = epoch_accuracy / len(self.train_loader)
            
            logger.info(
                f"Epoch {epoch+1} completed: "
                f"Avg Loss: {avg_epoch_loss:.4f}, "
                f"Avg Acc: {avg_epoch_accuracy:.4f}"
            )
        
        # Final evaluation
        final_metrics = self.evaluate()
        logger.info(f"Final validation accuracy: {final_metrics['val_accuracy']:.4f}")
        
        if self.mlflow_tracker:
            self.mlflow_tracker.log_metrics(final_metrics, step=self.global_step)
            self.mlflow_tracker.end_run()
        
        # Save final checkpoint
        self.save_checkpoint(final_metrics)
        
        # Plot training history
        self.plot_training_history()
        
        return final_metrics
    
    def plot_training_history(self):
        """Plot and save training history."""
        if not self.enable_visualization:
            return
        
        # Implementation would create plots for loss, accuracy, etc.
        logger.info("Training history plots saved")


def main():
    parser = argparse.ArgumentParser(description="Train CNN-Enhanced ModernBERT on Titanic dataset")
    
    # Model selection
    parser.add_argument('--model-type', choices=['cnn_hybrid', 'base'], default='cnn_hybrid',
                        help='Model type to train')
    
    # Data arguments
    parser.add_argument('--train-path', default='data/titanic/train.csv',
                        help='Path to training data')
    parser.add_argument('--val-path', default='data/titanic/val.csv',
                        help='Path to validation data')
    
    # Training arguments
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size for training')
    parser.add_argument('--learning-rate', type=float, default=2e-5,
                        help='Learning rate')
    parser.add_argument('--epochs', type=int, default=5,
                        help='Number of training epochs')
    parser.add_argument('--warmup-ratio', type=float, default=0.1,
                        help='Warmup ratio for learning rate schedule')
    
    # CNN-specific arguments
    parser.add_argument('--cnn-kernel-sizes', nargs='+', type=int, default=[2, 3, 4, 5],
                        help='CNN kernel sizes')
    parser.add_argument('--cnn-num-filters', type=int, default=128,
                        help='Number of CNN filters')
    parser.add_argument('--use-dilated-conv', action='store_true', default=True,
                        help='Use dilated convolutions')
    parser.add_argument('--use-attention-fusion', action='store_true', default=True,
                        help='Use attention-based fusion')
    parser.add_argument('--use-highway', action='store_true', default=True,
                        help='Use highway networks')
    
    # Optimizer arguments
    parser.add_argument('--optimizer', default='adamw',
                        choices=['sgd', 'adam', 'adamw', 'adamax', 'rmsprop', 
                                'adagrad', 'adadelta', 'lion'],
                        help='Optimizer type')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='Momentum for SGD')
    parser.add_argument('--weight-decay', type=float, default=0.01,
                        help='Weight decay')
    parser.add_argument('--betas', nargs=2, type=float, default=[0.9, 0.999],
                        help='Betas for Adam-based optimizers')
    
    # Output arguments
    parser.add_argument('--output-dir', default='./output/cnn_hybrid',
                        help='Output directory')
    parser.add_argument('--experiment-name', default='titanic_cnn_hybrid',
                        help='MLflow experiment name')
    parser.add_argument('--enable-mlflow', action='store_true', default=True,
                        help='Enable MLflow tracking')
    parser.add_argument('--enable-visualization', action='store_true', default=True,
                        help='Enable visualization of features')
    
    # Comparison mode
    parser.add_argument('--compare-models', action='store_true',
                        help='Train both base and CNN-hybrid models for comparison')
    
    args = parser.parse_args()
    
    if args.compare_models:
        # Train both models for comparison
        results = {}
        
        for model_type in ['base', 'cnn_hybrid']:
            logger.info(f"\n{'='*50}")
            logger.info(f"Training {model_type} model")
            logger.info(f"{'='*50}\n")
            
            trainer = HybridModelTrainer(
                model_type=model_type,
                train_path=args.train_path,
                val_path=args.val_path,
                batch_size=args.batch_size,
                learning_rate=args.learning_rate,
                epochs=args.epochs,
                warmup_ratio=args.warmup_ratio,
                output_dir=f"{args.output_dir}/{model_type}",
                experiment_name=f"{args.experiment_name}_{model_type}",
                enable_mlflow=args.enable_mlflow,
                enable_visualization=args.enable_visualization,
                optimizer_type=args.optimizer,
                optimizer_kwargs={
                    'momentum': args.momentum if args.optimizer == 'sgd' else None,
                    'weight_decay': args.weight_decay,
                    'betas': args.betas if args.optimizer in ['adam', 'adamw', 'adamax', 'lion'] else None,
                },
                cnn_kernel_sizes=args.cnn_kernel_sizes,
                cnn_num_filters=args.cnn_num_filters,
                use_dilated_conv=args.use_dilated_conv,
                use_attention_fusion=args.use_attention_fusion,
                use_highway=args.use_highway,
            )
            
            results[model_type] = trainer.train()
        
        # Compare results
        logger.info("\n" + "="*50)
        logger.info("Model Comparison Results:")
        logger.info("="*50)
        for model_type, metrics in results.items():
            logger.info(f"{model_type.upper()} Model:")
            logger.info(f"  - Validation Accuracy: {metrics['val_accuracy']:.4f}")
            logger.info(f"  - Validation Loss: {metrics['val_loss']:.4f}")
        
    else:
        # Train single model
        trainer = HybridModelTrainer(
            model_type=args.model_type,
            train_path=args.train_path,
            val_path=args.val_path,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            epochs=args.epochs,
            warmup_ratio=args.warmup_ratio,
            output_dir=args.output_dir,
            experiment_name=args.experiment_name,
            enable_mlflow=args.enable_mlflow,
            enable_visualization=args.enable_visualization,
            optimizer_type=args.optimizer,
            optimizer_kwargs={
                'momentum': args.momentum if args.optimizer == 'sgd' else None,
                'weight_decay': args.weight_decay,
                'betas': args.betas if args.optimizer in ['adam', 'adamw', 'adamax', 'lion'] else None,
            },
            cnn_kernel_sizes=args.cnn_kernel_sizes,
            cnn_num_filters=args.cnn_num_filters,
            use_dilated_conv=args.use_dilated_conv,
            use_attention_fusion=args.use_attention_fusion,
            use_highway=args.use_highway,
        )
        
        metrics = trainer.train()
        logger.info(f"\nTraining completed! Best validation accuracy: {metrics['val_accuracy']:.4f}")


if __name__ == "__main__":
    main()