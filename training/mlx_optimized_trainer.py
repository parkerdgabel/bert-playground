"""MLX-optimized trainer with best practices for performance.

This module implements MLX training optimizations including:
- Lazy computation with explicit eval() calls
- Dynamic batch sizing based on memory
- Gradient accumulation for effective larger batches
- Memory-efficient training patterns
- Optimized data pipeline with prefetching
"""

import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
from loguru import logger
from rich.progress import Progress, SpinnerColumn, TextColumn
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score

from data.unified_loader import OptimizedTitanicDataPipeline
from utils.logging_config import log_execution_time


@dataclass
class OptimizedTrainingConfig:
    """Configuration for optimized training."""
    
    # Basic training params
    learning_rate: float = 5e-5
    num_epochs: int = 3
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    
    # Optimization params
    base_batch_size: int = 32  # Base batch size
    max_batch_size: int = 64   # Maximum batch size for dynamic batching
    gradient_accumulation_steps: int = 1
    
    # Memory optimization
    eval_batch_size: int = 64  # Larger batch for evaluation
    lazy_eval_interval: int = 10  # Force eval every N steps
    memory_threshold: float = 0.8  # Memory usage threshold for dynamic batching
    
    # Data pipeline
    num_workers: int = 8
    prefetch_size: int = 4
    
    # Checkpointing
    save_steps: int = 100
    eval_steps: int = 100
    checkpoint_dir: str = "output/optimized"
    
    # Profiling
    enable_profiling: bool = True
    profile_memory_steps: int = 50


class MLXOptimizedTrainer:
    """Optimized trainer implementing MLX best practices."""
    
    def __init__(
        self,
        model: nn.Module,
        optimizer: optim.Optimizer,
        config: OptimizedTrainingConfig,
    ):
        self.model = model
        self.optimizer = optimizer
        self.config = config
        
        # Training state
        self.global_step = 0
        self.best_metric = -float("inf")
        self.current_batch_size = config.base_batch_size
        
        # Memory profiling
        self.memory_history = []
        
        # Setup lazy computation tracking
        self.accumulated_loss = None
        self.accumulated_grads = None
        self.steps_since_eval = 0
        self.max_steps = 0  # Will be set during training
        
        # Create output directory
        os.makedirs(config.checkpoint_dir, exist_ok=True)
        
        logger.info(
            f"Initialized MLX Optimized Trainer with config: "
            f"batch_size={config.base_batch_size}-{config.max_batch_size}, "
            f"grad_accum={config.gradient_accumulation_steps}, "
            f"workers={config.num_workers}, "
            f"prefetch={config.prefetch_size}"
        )
    
    def get_memory_usage(self) -> float:
        """Get current memory usage as a fraction."""
        # MLX doesn't have direct memory profiling yet
        # This is a placeholder - in practice you'd monitor system memory
        return 0.5  # Return 50% as placeholder
    
    def adjust_batch_size(self) -> int:
        """Dynamically adjust batch size based on memory usage."""
        memory_usage = self.get_memory_usage()
        
        if memory_usage < 0.5 and self.current_batch_size < self.config.max_batch_size:
            # Increase batch size if memory usage is low
            self.current_batch_size = min(
                self.current_batch_size * 2, self.config.max_batch_size
            )
            logger.info(f"Increased batch size to {self.current_batch_size}")
        elif memory_usage > self.config.memory_threshold:
            # Decrease batch size if memory usage is high
            self.current_batch_size = max(
                self.current_batch_size // 2, self.config.base_batch_size // 2
            )
            logger.info(f"Decreased batch size to {self.current_batch_size}")
        
        return self.current_batch_size
    
    def get_learning_rate(self) -> float:
        """Linear warmup and cosine decay."""
        # Handle case when max_steps is not set yet
        if self.max_steps == 0:
            return self.config.learning_rate
            
        warmup_steps = int(self.config.warmup_ratio * self.max_steps)
        
        if self.global_step < warmup_steps:
            lr = self.config.learning_rate * self.global_step / warmup_steps
        else:
            progress = (self.global_step - warmup_steps) / max(
                1, self.max_steps - warmup_steps
            )
            lr = self.config.learning_rate * 0.5 * (1 + mx.cos(mx.array(np.pi * progress)))
        
        return float(lr)
    
    @log_execution_time
    def train_step_lazy(
        self, batch: Dict[str, mx.array]
    ) -> Tuple[mx.array, Dict[str, float]]:
        """Training step with lazy computation and gradient accumulation."""
        
        # Define loss function
        def loss_fn(model, inputs, labels):
            outputs = model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                labels=labels,
            )
            return outputs["loss"]
        
        # Get loss and gradients (lazy - not computed yet)
        loss_and_grad_fn = nn.value_and_grad(self.model, loss_fn)
        loss, grads = loss_and_grad_fn(
            self.model,
            {
                "input_ids": batch["input_ids"],
                "attention_mask": batch["attention_mask"],
            },
            batch["labels"].squeeze(),
        )
        
        # Scale loss for gradient accumulation
        if self.config.gradient_accumulation_steps > 1:
            loss = loss / self.config.gradient_accumulation_steps
        
        # Accumulate gradients (still lazy)
        if self.accumulated_grads is None:
            self.accumulated_grads = grads
            self.accumulated_loss = loss
        else:
            # Accumulate gradients recursively
            def accumulate_recursive(acc_dict, new_dict):
                for key in new_dict:
                    if isinstance(new_dict[key], dict):
                        accumulate_recursive(acc_dict[key], new_dict[key])
                    else:
                        acc_dict[key] = acc_dict[key] + new_dict[key]
            
            accumulate_recursive(self.accumulated_grads, grads)
            self.accumulated_loss = self.accumulated_loss + loss
        
        self.steps_since_eval += 1
        
        # Update parameters only when accumulation is complete
        if (self.global_step + 1) % self.config.gradient_accumulation_steps == 0:
            # Force evaluation of accumulated gradients
            mx.eval(self.accumulated_grads)
            
            # Update parameters
            self.optimizer.update(self.model, self.accumulated_grads)
            
            # Update learning rate
            new_lr = self.get_learning_rate()
            self.optimizer.learning_rate = new_lr
            
            # Get metrics (force eval for logging)
            mx.eval(self.accumulated_loss)
            loss_value = float(self.accumulated_loss)
            
            # Reset accumulation
            self.accumulated_grads = None
            self.accumulated_loss = None
            
            metrics = {
                "loss": loss_value,
                "learning_rate": float(new_lr),
                "batch_size": self.current_batch_size,
                "memory_usage": self.get_memory_usage(),
            }
            
            return loss_value, metrics
        else:
            # Return dummy metrics during accumulation
            return loss, {"accumulating": True}
    
    def force_eval_if_needed(self):
        """Force evaluation periodically to prevent graph buildup."""
        if self.steps_since_eval >= self.config.lazy_eval_interval:
            if self.accumulated_loss is not None:
                mx.eval(self.accumulated_loss)
            self.steps_since_eval = 0
            logger.debug(f"Forced eval at step {self.global_step}")
    
    @log_execution_time
    def evaluate_lazy(
        self, dataloader: OptimizedTitanicDataPipeline, phase: str = "val"
    ) -> Dict[str, float]:
        """Evaluate with larger batch size and lazy computation."""
        logger.info(f"Starting {phase} evaluation with batch_size={self.config.eval_batch_size}")
        
        # Temporarily increase batch size for evaluation
        original_batch_size = dataloader.batch_size
        dataloader.batch_size = self.config.eval_batch_size
        dataloader._initialize_optimized_stream()  # Reinitialize with new batch size
        
        total_loss = 0
        all_predictions = []
        all_labels = []
        all_probs = []
        
        num_batches = 0
        for batch in dataloader.get_dataloader()():
            # Forward pass (lazy)
            outputs = self.model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"].squeeze() if "labels" in batch else None,
            )
            
            if "labels" in batch:
                # Accumulate loss lazily
                total_loss = total_loss + outputs["loss"]
                all_labels.extend(batch["labels"].squeeze().tolist())
            
            # Get predictions (lazy)
            predictions = mx.argmax(outputs["logits"], axis=-1)
            probs = mx.softmax(outputs["logits"], axis=-1)
            
            # Force eval periodically to avoid memory buildup
            if num_batches % 10 == 0:
                mx.eval(predictions, probs)
            
            all_predictions.extend(predictions.tolist())
            all_probs.extend(probs[:, 1].tolist())
            
            num_batches += 1
            
            if num_batches >= dataloader.get_num_batches():
                break
        
        # Force final eval
        mx.eval(total_loss)
        
        # Restore original batch size
        dataloader.batch_size = original_batch_size
        dataloader._initialize_optimized_stream()
        
        # Calculate metrics
        metrics = {}
        if all_labels:
            accuracy = accuracy_score(all_labels, all_predictions)
            precision, recall, f1, support = precision_recall_fscore_support(
                all_labels, all_predictions, average="binary"
            )
            
            try:
                auc = roc_auc_score(all_labels, all_probs)
            except Exception:
                auc = 0.0
            
            metrics = {
                f"{phase}_loss": float(total_loss) / num_batches,
                f"{phase}_accuracy": accuracy,
                f"{phase}_precision": precision,
                f"{phase}_recall": recall,
                f"{phase}_f1": f1,
                f"{phase}_auc": auc,
            }
            
            logger.info(
                f"{phase.capitalize()} Results - "
                f"Loss: {metrics[f'{phase}_loss']:.4f}, "
                f"Acc: {accuracy:.4f}, "
                f"F1: {f1:.4f}, "
                f"AUC: {auc:.4f}"
            )
        
        return metrics
    
    def train(
        self,
        train_dataloader: OptimizedTitanicDataPipeline,
        val_dataloader: Optional[OptimizedTitanicDataPipeline] = None,
        test_dataloader: Optional[OptimizedTitanicDataPipeline] = None,
    ):
        """Main training loop with MLX optimizations."""
        # Calculate total steps
        steps_per_epoch = train_dataloader.get_num_batches()
        self.max_steps = steps_per_epoch * self.config.num_epochs
        
        logger.info(
            f"Starting optimized training: "
            f"{self.config.num_epochs} epochs, "
            f"{steps_per_epoch} steps/epoch, "
            f"{self.max_steps} total steps"
        )
        
        # Initialize data pipeline with optimal settings
        train_dataloader.num_threads = self.config.num_workers
        train_dataloader.prefetch_size = self.config.prefetch_size
        train_dataloader._initialize_optimized_stream()
        
        start_time = time.time()
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            transient=True,
        ) as progress:
            
            for epoch in range(self.config.num_epochs):
                epoch_start = time.time()
                epoch_loss = 0
                epoch_steps = 0
                
                # Adjust batch size based on memory
                self.current_batch_size = self.adjust_batch_size()
                if self.current_batch_size != train_dataloader.batch_size:
                    train_dataloader.batch_size = self.current_batch_size
                    train_dataloader._initialize_optimized_stream()
                
                task = progress.add_task(
                    f"[cyan]Epoch {epoch + 1}/{self.config.num_epochs}", total=steps_per_epoch
                )
                
                for batch in train_dataloader.get_dataloader()():
                    self.global_step += 1
                    
                    # Training step with lazy computation
                    loss, metrics = self.train_step_lazy(batch)
                    
                    # Force eval periodically
                    self.force_eval_if_needed()
                    
                    # Update progress only on actual updates
                    if "loss" in metrics:
                        epoch_loss += metrics["loss"]
                        epoch_steps += 1
                        
                        progress.update(
                            task,
                            advance=1,
                            description=f"[cyan]Epoch {epoch + 1}/{self.config.num_epochs} - "
                            f"Loss: {metrics['loss']:.4f}, "
                            f"LR: {metrics['learning_rate']:.2e}, "
                            f"Mem: {metrics['memory_usage']:.1%}",
                        )
                    
                    # Profile memory periodically
                    if (
                        self.config.enable_profiling
                        and self.global_step % self.config.profile_memory_steps == 0
                    ):
                        self.memory_history.append(
                            {
                                "step": self.global_step,
                                "memory": self.get_memory_usage(),
                                "batch_size": self.current_batch_size,
                            }
                        )
                    
                    # Evaluate periodically
                    if val_dataloader and self.global_step % self.config.eval_steps == 0:
                        val_metrics = self.evaluate_lazy(val_dataloader, "val")
                        
                        # Save best model
                        if val_metrics.get("val_accuracy", 0) > self.best_metric:
                            self.best_metric = val_metrics["val_accuracy"]
                            self.save_checkpoint("best")
                            logger.info(
                                f"New best model saved with accuracy: {self.best_metric:.4f}"
                            )
                    
                    # Save checkpoint periodically
                    if self.global_step % self.config.save_steps == 0:
                        self.save_checkpoint(f"checkpoint-{self.global_step}")
                
                # Epoch summary
                avg_loss = epoch_loss / max(1, epoch_steps)
                epoch_time = time.time() - epoch_start
                logger.info(
                    f"Epoch {epoch + 1} completed in {epoch_time:.1f}s - "
                    f"Avg Loss: {avg_loss:.4f}, "
                    f"Steps/sec: {epoch_steps / epoch_time:.1f}"
                )
        
        # Final evaluation
        if val_dataloader:
            logger.info("Running final validation...")
            val_metrics = self.evaluate_lazy(val_dataloader, "val")
        
        if test_dataloader:
            logger.info("Running test evaluation...")
            self.evaluate_lazy(test_dataloader, "test")
        
        total_time = time.time() - start_time
        logger.info(
            f"Training completed in {total_time:.1f}s - "
            f"Avg time/step: {total_time / self.global_step:.3f}s"
        )
        
        # Save memory profile
        if self.config.enable_profiling and self.memory_history:
            self._save_memory_profile()
    
    def save_checkpoint(self, name: str):
        """Save model checkpoint."""
        checkpoint_path = Path(self.config.checkpoint_dir) / name
        checkpoint_path.mkdir(exist_ok=True)
        
        # Save model
        self.model.save_pretrained(str(checkpoint_path))
        
        # Save optimizer state
        optimizer_path = checkpoint_path / "optimizer.safetensors"
        mx.save_safetensors(
            str(optimizer_path),
            {"optimizer": self.optimizer.state},
        )
        
        # Save training state
        state_path = checkpoint_path / "trainer_state.json"
        import json
        
        with open(state_path, "w") as f:
            json.dump(
                {
                    "global_step": self.global_step,
                    "best_metric": self.best_metric,
                    "current_batch_size": self.current_batch_size,
                    "config": self.config.__dict__,
                },
                f,
                indent=2,
            )
        
        logger.info(f"Checkpoint saved to {checkpoint_path}")
    
    def _save_memory_profile(self):
        """Save memory profiling data."""
        import json
        
        profile_path = Path(self.config.checkpoint_dir) / "memory_profile.json"
        with open(profile_path, "w") as f:
            json.dump(self.memory_history, f, indent=2)
        
        logger.info(f"Memory profile saved to {profile_path}")