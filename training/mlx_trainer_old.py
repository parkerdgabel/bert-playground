"""Unified MLX trainer combining all optimizations and features.

This module implements a consolidated trainer that merges:
- MLX-specific optimizations from MLXOptimizedTrainer
- MLflow integration from TitanicTrainerV2
- Best practices from both implementations
"""

import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
from loguru import logger
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
)
from tqdm import tqdm

from data.unified_loader import UnifiedTitanicDataPipeline
from utils.logging_config import log_execution_time
from utils.mlflow_central import setup_central_mlflow
# from utils.visualization import ExperimentVisualizer  # TODO: Add visualization support


@dataclass
class UnifiedTrainingConfig:
    """Unified configuration for MLX training."""
    
    # Basic training parameters
    learning_rate: float = 5e-5
    num_epochs: int = 3
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    early_stopping_patience: int = 3
    early_stopping_threshold: float = 0.001
    
    # MLX optimizations
    base_batch_size: int = 32
    max_batch_size: int = 64
    enable_dynamic_batching: bool = True
    gradient_accumulation_steps: int = 1
    lazy_eval_interval: int = 10
    memory_threshold: float = 0.8
    
    # Evaluation settings
    eval_batch_size: int = 64
    eval_steps: int = 100
    save_steps: int = 100
    
    # Data pipeline
    num_workers: int = 8
    prefetch_size: int = 4
    enable_caching: bool = True
    cache_dir: str = "cache/tokenized"
    
    # MLflow settings
    enable_mlflow: bool = True
    experiment_name: str = "mlx_unified"
    run_name: Optional[str] = None
    tracking_uri: Optional[str] = None
    log_models: bool = True
    
    # Quantization
    enable_quantization: bool = False
    quantization_bits: int = 4
    quantization_group_size: int = 64
    
    # Output settings
    output_dir: str = "output/unified"
    checkpoint_dir: Optional[str] = None
    save_total_limit: int = 3
    
    # Advanced features
    enable_profiling: bool = True
    profile_memory_steps: int = 50
    enable_visualization: bool = True
    gradient_clip_val: float = 1.0
    label_smoothing: float = 0.0
    
    # Training modifiers
    training_modifiers: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Post-initialization setup."""
        if self.checkpoint_dir is None:
            self.checkpoint_dir = os.path.join(self.output_dir, "checkpoints")
        
        # Create directories
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        if self.enable_caching:
            os.makedirs(self.cache_dir, exist_ok=True)


class MLXTrainer:
    """Unified MLX trainer with all optimizations and features."""
    
    def __init__(
        self,
        model: nn.Module,
        config: UnifiedTrainingConfig,
        optimizer: Optional[optim.Optimizer] = None,
    ):
        """Initialize the unified trainer.
        
        Args:
            model: MLX model to train
            config: Training configuration
            optimizer: Optional optimizer (created if not provided)
        """
        self.model = model
        self.config = config
        
        # Create optimizer if not provided
        if optimizer is None:
            self.optimizer = optim.AdamW(
                learning_rate=config.learning_rate,
                weight_decay=config.weight_decay,
            )
        else:
            self.optimizer = optimizer
        
        # Training state
        self.global_step = 0
        self.current_epoch = 0
        self.best_metric = -float("inf")
        self.best_metric_step = 0
        self.current_batch_size = config.base_batch_size
        
        # MLX optimization state
        self.accumulated_loss = None
        self.accumulated_grads = None
        self.steps_since_eval = 0
        self.max_steps = 0
        
        # Early stopping
        self.early_stopping_counter = 0
        self.last_best_metric = -float("inf")
        
        # Memory profiling
        self.memory_history = []
        
        # Visualization
        self.visualizer = None
        # TODO: Add visualization support when ExperimentVisualizer is ready
        # if config.enable_visualization:
        #     self.visualizer = ExperimentVisualizer(save_dir=config.output_dir)
        
        # MLflow setup
        self.mlflow_run = None
        if config.enable_mlflow:
            self._setup_mlflow()
        
        logger.info(
            f"Initialized MLX Unified Trainer:\n"
            f"  Model: {model.__class__.__name__}\n"
            f"  Batch Size: {config.base_batch_size}-{config.max_batch_size}\n"
            f"  Learning Rate: {config.learning_rate}\n"
            f"  Epochs: {config.num_epochs}\n"
            f"  MLflow: {'Enabled' if config.enable_mlflow else 'Disabled'}\n"
            f"  Output Dir: {config.output_dir}"
        )
    
    def _setup_mlflow(self):
        """Setup MLflow tracking."""
        import mlflow
        
        # Setup central MLflow configuration
        mlflow_central = setup_central_mlflow(
            experiment_name=self.config.experiment_name,
            tracking_uri=self.config.tracking_uri,
        )
        
        # Log configuration
        logger.info(
            f"MLflow tracking configured:\n"
            f"  URI: {mlflow_central.tracking_uri}\n"
            f"  Experiment: {self.config.experiment_name}"
        )
    
    def get_memory_usage(self) -> float:
        """Get current memory usage as a fraction."""
        # Placeholder for actual memory monitoring
        # In production, integrate with system memory monitoring
        return 0.5
    
    def adjust_batch_size(self) -> int:
        """Dynamically adjust batch size based on memory usage."""
        if not self.config.enable_dynamic_batching:
            return self.current_batch_size
        
        memory_usage = self.get_memory_usage()
        
        if memory_usage < 0.5 and self.current_batch_size < self.config.max_batch_size:
            self.current_batch_size = min(
                self.current_batch_size * 2, self.config.max_batch_size
            )
            logger.info(f"Increased batch size to {self.current_batch_size}")
        elif memory_usage > self.config.memory_threshold:
            self.current_batch_size = max(
                self.current_batch_size // 2, self.config.base_batch_size // 2
            )
            logger.info(f"Decreased batch size to {self.current_batch_size}")
        
        return self.current_batch_size
    
    def get_learning_rate(self) -> float:
        """Get current learning rate with warmup and decay."""
        if self.max_steps == 0:
            return self.config.learning_rate
        
        warmup_steps = int(self.config.warmup_ratio * self.max_steps)
        
        if self.global_step < warmup_steps:
            # Linear warmup
            lr = self.config.learning_rate * self.global_step / warmup_steps
        else:
            # Cosine decay
            progress = (self.global_step - warmup_steps) / max(
                1, self.max_steps - warmup_steps
            )
            lr = self.config.learning_rate * 0.5 * (
                1 + mx.cos(mx.array(np.pi * progress))
            )
        
        return float(lr)
    
    @log_execution_time
    def train_step(
        self, batch: Dict[str, mx.array]
    ) -> Tuple[float, Dict[str, float]]:
        """Execute a training step with MLX optimizations."""
        
        def loss_fn(model, inputs, labels):
            outputs = model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                labels=labels,
            )
            
            loss = outputs["loss"]
            
            # Apply label smoothing if configured
            if self.config.label_smoothing > 0:
                # Simple label smoothing implementation
                n_classes = outputs["logits"].shape[-1]
                smooth_loss = -mx.log(
                    mx.softmax(outputs["logits"], axis=-1)
                ).mean(axis=-1)
                loss = (
                    (1 - self.config.label_smoothing) * loss
                    + self.config.label_smoothing * smooth_loss.mean()
                )
            
            return loss
        
        # Compute loss and gradients (lazy)
        loss_and_grad_fn = nn.value_and_grad(self.model, loss_fn)
        loss, grads = loss_and_grad_fn(
            self.model,
            {
                "input_ids": batch["input_ids"],
                "attention_mask": batch["attention_mask"],
            },
            batch["labels"].squeeze(),
        )
        
        # Scale for gradient accumulation
        if self.config.gradient_accumulation_steps > 1:
            loss = loss / self.config.gradient_accumulation_steps
        
        # Accumulate gradients
        if self.accumulated_grads is None:
            self.accumulated_grads = grads
            self.accumulated_loss = loss
        else:
            # Tree-based accumulation for nested structures
            def accumulate_tree(acc, new):
                if isinstance(new, dict):
                    for k, v in new.items():
                        if k in acc:
                            acc[k] = accumulate_tree(acc[k], v)
                        else:
                            acc[k] = v
                else:
                    return acc + new
                return acc
            
            self.accumulated_grads = accumulate_tree(self.accumulated_grads, grads)
            self.accumulated_loss = self.accumulated_loss + loss
        
        self.steps_since_eval += 1
        
        # Update when accumulation complete
        if (self.global_step + 1) % self.config.gradient_accumulation_steps == 0:
            # Force evaluation
            mx.eval(self.accumulated_grads)
            
            # Gradient clipping
            if self.config.gradient_clip_val > 0:
                self._clip_gradients(self.accumulated_grads)
            
            # Update parameters
            self.optimizer.update(self.model, self.accumulated_grads)
            
            # Update learning rate
            new_lr = self.get_learning_rate()
            self.optimizer.learning_rate = new_lr
            
            # Get loss value
            mx.eval(self.accumulated_loss)
            loss_value = float(self.accumulated_loss)
            
            # Reset accumulation
            self.accumulated_grads = None
            self.accumulated_loss = None
            
            metrics = {
                "loss": loss_value,
                "learning_rate": new_lr,
                "batch_size": self.current_batch_size,
                "memory_usage": self.get_memory_usage(),
                "epoch": self.current_epoch,
            }
            
            return loss_value, metrics
        else:
            return 0.0, {"accumulating": True}
    
    def _clip_gradients(self, grads: Dict[str, Any]) -> None:
        """Clip gradients by global norm."""
        # Flatten all gradients
        all_grads = []
        
        def flatten_grads(g):
            if isinstance(g, dict):
                for v in g.values():
                    flatten_grads(v)
            elif isinstance(g, list):
                for v in g:
                    flatten_grads(v)
            elif hasattr(g, 'reshape'):
                all_grads.append(g.reshape(-1))
        
        flatten_grads(grads)
        
        # Compute global norm
        total_norm = mx.sqrt(sum(mx.sum(g**2) for g in all_grads))
        
        # Clip if needed
        clip_coef = self.config.gradient_clip_val / (total_norm + 1e-6)
        if clip_coef < 1:
            def clip_tree(g):
                if isinstance(g, dict):
                    return {k: clip_tree(v) for k, v in g.items()}
                elif isinstance(g, list):
                    return [clip_tree(v) for v in g]
                elif hasattr(g, '__mul__'):
                    return g * clip_coef
                else:
                    return g
            
            grads = clip_tree(grads)
    
    def force_eval_if_needed(self):
        """Force evaluation periodically to prevent graph buildup."""
        if self.steps_since_eval >= self.config.lazy_eval_interval:
            if self.accumulated_loss is not None:
                mx.eval(self.accumulated_loss)
            self.steps_since_eval = 0
    
    @log_execution_time
    def evaluate(
        self,
        dataloader: UnifiedTitanicDataPipeline,
        phase: str = "val",
        max_batches: Optional[int] = None,
    ) -> Dict[str, float]:
        """Evaluate the model."""
        logger.info(f"Starting {phase} evaluation")
        
        # Temporarily use eval batch size
        original_batch_size = dataloader.batch_size
        if hasattr(dataloader, "batch_size"):
            dataloader.batch_size = self.config.eval_batch_size
        
        total_loss = 0
        all_predictions = []
        all_labels = []
        all_probs = []
        num_batches = 0
        
        # Create progress bar
        total_batches = max_batches or dataloader.get_num_batches()
        eval_pbar = tqdm(
            total=total_batches,
            desc=f"Evaluating {phase}",
            leave=False,
        )
        
        for batch in dataloader.get_dataloader():
            # Forward pass
            outputs = self.model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch.get("labels"),
            )
            
            if "labels" in batch:
                total_loss = total_loss + outputs["loss"]
                all_labels.extend(batch["labels"].squeeze().tolist())
            
            # Get predictions
            predictions = mx.argmax(outputs["logits"], axis=-1)
            probs = mx.softmax(outputs["logits"], axis=-1)
            
            # Force eval periodically
            if num_batches % 10 == 0:
                mx.eval(predictions, probs)
            
            all_predictions.extend(predictions.tolist())
            if probs.shape[-1] == 2:
                all_probs.extend(probs[:, 1].tolist())
            
            num_batches += 1
            eval_pbar.update(1)
            
            if max_batches and num_batches >= max_batches:
                break
        
        eval_pbar.close()
        
        # Force final eval
        if total_loss != 0:
            mx.eval(total_loss)
        
        # Restore batch size
        if hasattr(dataloader, "batch_size"):
            dataloader.batch_size = original_batch_size
        
        # Calculate metrics
        metrics = {}
        if all_labels:
            avg_loss = float(total_loss) / num_batches
            accuracy = accuracy_score(all_labels, all_predictions)
            
            # Handle binary/multiclass metrics
            average = "binary" if len(set(all_labels)) == 2 else "weighted"
            precision, recall, f1, _ = precision_recall_fscore_support(
                all_labels, all_predictions, average=average, zero_division=0
            )
            
            metrics = {
                f"{phase}_loss": avg_loss,
                f"{phase}_accuracy": accuracy,
                f"{phase}_precision": precision,
                f"{phase}_recall": recall,
                f"{phase}_f1": f1,
            }
            
            # AUC for binary classification
            if all_probs and len(set(all_labels)) == 2:
                try:
                    auc = roc_auc_score(all_labels, all_probs)
                    metrics[f"{phase}_auc"] = auc
                except Exception:
                    pass
            
            logger.info(
                f"{phase.capitalize()} Results: "
                f"Loss={avg_loss:.4f}, "
                f"Acc={accuracy:.4f}, "
                f"F1={f1:.4f}"
            )
        
        return metrics
    
    def should_stop_early(self, current_metric: float) -> bool:
        """Check if training should stop early."""
        if not self.config.early_stopping_patience:
            return False
        
        # Check if metric improved
        if current_metric > self.last_best_metric + self.config.early_stopping_threshold:
            self.early_stopping_counter = 0
            self.last_best_metric = current_metric
        else:
            self.early_stopping_counter += 1
        
        should_stop = self.early_stopping_counter >= self.config.early_stopping_patience
        
        if should_stop:
            logger.info(
                f"Early stopping triggered: "
                f"No improvement for {self.early_stopping_counter} evaluations"
            )
        
        return should_stop
    
    def train(
        self,
        train_dataloader: UnifiedTitanicDataPipeline,
        val_dataloader: Optional[UnifiedTitanicDataPipeline] = None,
        test_dataloader: Optional[UnifiedTitanicDataPipeline] = None,
        resume_from_checkpoint: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Main training loop."""
        # Resume from checkpoint if specified
        if resume_from_checkpoint:
            self.load_checkpoint(resume_from_checkpoint)
        
        # Start MLflow run
        if self.config.enable_mlflow:
            import mlflow
            
            with mlflow.start_run(run_name=self.config.run_name) as run:
                self.mlflow_run = run
                
                # Log configuration
                mlflow.log_params({
                    "model_type": self.model.__class__.__name__,
                    "learning_rate": self.config.learning_rate,
                    "batch_size": f"{self.config.base_batch_size}-{self.config.max_batch_size}",
                    "num_epochs": self.config.num_epochs,
                    "gradient_accumulation": self.config.gradient_accumulation_steps,
                    "warmup_ratio": self.config.warmup_ratio,
                })
                
                # Train with MLflow tracking
                return self._train_loop(train_dataloader, val_dataloader, test_dataloader)
        else:
            # Train without MLflow
            return self._train_loop(train_dataloader, val_dataloader, test_dataloader)
    
    def _train_loop(
        self,
        train_dataloader: UnifiedTitanicDataPipeline,
        val_dataloader: Optional[UnifiedTitanicDataPipeline] = None,
        test_dataloader: Optional[UnifiedTitanicDataPipeline] = None,
    ) -> Dict[str, Any]:
        """Internal training loop."""
        # Calculate total steps
        steps_per_epoch = train_dataloader.get_num_batches()
        self.max_steps = steps_per_epoch * self.config.num_epochs
        
        logger.info(
            f"Starting training: "
            f"{self.config.num_epochs} epochs, "
            f"{steps_per_epoch} steps/epoch, "
            f"{self.max_steps} total steps"
        )
        
        # Training history
        history = {
            "train_loss": [],
            "val_metrics": [],
            "learning_rates": [],
        }
        
        start_time = time.time()
        
        for epoch in range(self.current_epoch, self.config.num_epochs):
            self.current_epoch = epoch
            epoch_start = time.time()
            epoch_loss = 0
            epoch_steps = 0
            
            # Adjust batch size
            if self.config.enable_dynamic_batching:
                self.adjust_batch_size()
            
            # Training progress bar
            epoch_pbar = tqdm(
                total=steps_per_epoch,
                desc=f"Epoch {epoch + 1}/{self.config.num_epochs}",
                dynamic_ncols=True,
            )
            
            for batch_idx, batch in enumerate(train_dataloader.get_dataloader()):
                self.global_step += 1
                
                # Training step
                loss, metrics = self.train_step(batch)
                
                # Force eval periodically
                self.force_eval_if_needed()
                
                # Update progress
                if "loss" in metrics:
                    epoch_loss += metrics["loss"]
                    epoch_steps += 1
                    
                    # Update progress bar
                    epoch_pbar.update(1)
                    epoch_pbar.set_postfix({
                        "Loss": f"{metrics['loss']:.4f}",
                        "LR": f"{metrics['learning_rate']:.2e}",
                        "BS": self.current_batch_size,
                    })
                    
                    # Log to MLflow
                    if self.config.enable_mlflow and self.mlflow_run:
                        import mlflow
                        mlflow.log_metrics(metrics, step=self.global_step)
                
                # Memory profiling
                if (
                    self.config.enable_profiling
                    and self.global_step % self.config.profile_memory_steps == 0
                ):
                    self.memory_history.append({
                        "step": self.global_step,
                        "memory": self.get_memory_usage(),
                        "batch_size": self.current_batch_size,
                    })
                
                # Evaluation
                if val_dataloader and self.global_step % self.config.eval_steps == 0:
                    val_metrics = self.evaluate(val_dataloader, "val")
                    history["val_metrics"].append(val_metrics)
                    
                    # Log validation metrics
                    if self.config.enable_mlflow and self.mlflow_run:
                        import mlflow
                        mlflow.log_metrics(val_metrics, step=self.global_step)
                    
                    # Save best model
                    val_score = val_metrics.get("val_accuracy", 0)
                    if val_score > self.best_metric:
                        self.best_metric = val_score
                        self.best_metric_step = self.global_step
                        self.save_checkpoint("best")
                        logger.info(f"New best model: {val_score:.4f}")
                    
                    # Early stopping
                    if self.should_stop_early(val_score):
                        logger.info("Early stopping triggered")
                        break
                
                # Save checkpoint
                if self.global_step % self.config.save_steps == 0:
                    self.save_checkpoint(f"checkpoint-{self.global_step}")
                
                if batch_idx >= steps_per_epoch - 1:
                    break
            
            epoch_pbar.close()
            
            # Epoch summary
            avg_loss = epoch_loss / max(1, epoch_steps)
            epoch_time = time.time() - epoch_start
            
            logger.info(
                f"Epoch {epoch + 1} completed: "
                f"Loss={avg_loss:.4f}, "
                f"Time={epoch_time:.1f}s, "
                f"Steps/sec={epoch_steps / epoch_time:.1f}"
            )
            
            history["train_loss"].append(avg_loss)
            
            # Visualization
            # TODO: Add visualization when ready
            # if self.visualizer and history["val_metrics"]:
            #     self.visualizer.plot_metrics(history)
            
            # Check early stopping at epoch level
            if self.early_stopping_counter >= self.config.early_stopping_patience:
                break
        
        # Final evaluation
        final_metrics = {}
        
        if val_dataloader:
            logger.info("Running final validation")
            val_metrics = self.evaluate(val_dataloader, "val")
            final_metrics.update(val_metrics)
        
        if test_dataloader:
            logger.info("Running test evaluation")
            test_metrics = self.evaluate(test_dataloader, "test")
            final_metrics.update(test_metrics)
        
        # Training summary
        total_time = time.time() - start_time
        logger.info(
            f"Training completed in {total_time:.1f}s\n"
            f"Best validation score: {self.best_metric:.4f} at step {self.best_metric_step}"
        )
        
        # Save final artifacts
        if self.config.enable_mlflow and self.mlflow_run:
            import mlflow
            
            # Log final metrics
            mlflow.log_metrics(final_metrics)
            
            # Log model
            if self.config.log_models:
                mlflow.log_artifact(
                    os.path.join(self.config.checkpoint_dir, "best"),
                    "model",
                )
            
            # Log training history
            with open(os.path.join(self.config.output_dir, "history.json"), "w") as f:
                json.dump(history, f, indent=2)
            mlflow.log_artifact(os.path.join(self.config.output_dir, "history.json"))
        
        # Save memory profile
        if self.config.enable_profiling:
            self._save_memory_profile()
        
        return {
            "best_metric": self.best_metric,
            "best_step": self.best_metric_step,
            "final_metrics": final_metrics,
            "history": history,
            "total_time": total_time,
        }
    
    def save_checkpoint(self, name: str) -> None:
        """Save model checkpoint."""
        checkpoint_path = Path(self.config.checkpoint_dir) / name
        checkpoint_path.mkdir(exist_ok=True, parents=True)
        
        # Save model
        self.model.save_pretrained(str(checkpoint_path))
        
        # Save optimizer state
        optimizer_path = checkpoint_path / "optimizer.safetensors"
        try:
            from mlx.utils import tree_flatten
            state_flat = dict(tree_flatten(self.optimizer.state))
            mx.save_safetensors(str(optimizer_path), state_flat)
        except Exception as e:
            logger.warning(f"Failed to save optimizer: {e}")
        
        # Save trainer state
        state = {
            "global_step": self.global_step,
            "current_epoch": self.current_epoch,
            "best_metric": self.best_metric,
            "best_metric_step": self.best_metric_step,
            "config": self.config.__dict__,
        }
        
        with open(checkpoint_path / "trainer_state.json", "w") as f:
            json.dump(state, f, indent=2)
        
        logger.info(f"Checkpoint saved: {checkpoint_path}")
        
        # Manage checkpoint limit
        self._cleanup_checkpoints()
    
    def load_checkpoint(self, checkpoint_name: str) -> bool:
        """Load checkpoint."""
        checkpoint_path = Path(self.config.checkpoint_dir) / checkpoint_name
        
        if not checkpoint_path.exists():
            logger.error(f"Checkpoint not found: {checkpoint_path}")
            return False
        
        try:
            # Load trainer state
            with open(checkpoint_path / "trainer_state.json", "r") as f:
                state = json.load(f)
            
            self.global_step = state["global_step"]
            self.current_epoch = state["current_epoch"]
            self.best_metric = state["best_metric"]
            self.best_metric_step = state["best_metric_step"]
            
            # Load optimizer
            optimizer_path = checkpoint_path / "optimizer.safetensors"
            if optimizer_path.exists():
                from mlx.utils import tree_unflatten
                state_flat = mx.load(str(optimizer_path))
                self.optimizer.state = tree_unflatten(list(state_flat.items()))
            
            logger.info(f"Checkpoint loaded: {checkpoint_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")
            return False
    
    def _cleanup_checkpoints(self) -> None:
        """Remove old checkpoints keeping only the latest N."""
        if self.config.save_total_limit <= 0:
            return
        
        # Get all checkpoints
        checkpoints = []
        checkpoint_dir = Path(self.config.checkpoint_dir)
        
        for path in checkpoint_dir.iterdir():
            if path.is_dir() and path.name.startswith("checkpoint-"):
                try:
                    step = int(path.name.split("-")[1])
                    checkpoints.append((step, path))
                except (IndexError, ValueError):
                    continue
        
        # Sort by step
        checkpoints.sort(key=lambda x: x[0], reverse=True)
        
        # Remove old checkpoints
        for _, path in checkpoints[self.config.save_total_limit:]:
            logger.info(f"Removing old checkpoint: {path}")
            import shutil
            shutil.rmtree(path)
    
    def _save_memory_profile(self) -> None:
        """Save memory profiling data."""
        if not self.memory_history:
            return
        
        profile_path = Path(self.config.output_dir) / "memory_profile.json"
        with open(profile_path, "w") as f:
            json.dump(self.memory_history, f, indent=2)
        
        logger.info(f"Memory profile saved: {profile_path}")
        
        # Log to MLflow
        if self.config.enable_mlflow and self.mlflow_run:
            import mlflow
            mlflow.log_artifact(str(profile_path))