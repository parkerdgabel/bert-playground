import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from typing import Dict, Optional, Tuple, List
from pathlib import Path
import time
from tqdm import tqdm
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
from loguru import logger
import json

from models.modernbert_mlx import ModernBertMLX, create_model
from models.classification_head import TitanicClassifier
from data.titanic_loader import TitanicDataPipeline
from utils.logging_config import LoggingConfig, ExperimentLogger, log_execution_time
from utils.mlflow_utils import MLflowExperimentTracker, MLflowModelRegistry
import mlflow


class TitanicTrainerV2:
    """Enhanced trainer with MLflow tracking and extensive logging."""
    
    def __init__(
        self,
        model: TitanicClassifier,
        train_loader: TitanicDataPipeline,
        val_loader: Optional[TitanicDataPipeline] = None,
        learning_rate: float = 2e-5,
        weight_decay: float = 0.01,
        warmup_steps: int = 100,
        max_steps: Optional[int] = None,
        eval_steps: int = 50,
        save_steps: int = 100,
        output_dir: str = "./output",
        gradient_accumulation_steps: int = 1,
        experiment_name: str = "titanic_modernbert",
        run_name: Optional[str] = None,
        log_level: str = "INFO",
        enable_mlflow: bool = True
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Training parameters
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps
        self.eval_steps = eval_steps
        self.save_steps = save_steps
        self.gradient_accumulation_steps = gradient_accumulation_steps
        
        # Initialize logging
        self.logging_config = LoggingConfig(
            log_dir=str(self.output_dir / "logs"),
            log_level=log_level,
            experiment_name=run_name or experiment_name
        )
        
        # Initialize MLflow tracking
        self.enable_mlflow = enable_mlflow
        if enable_mlflow:
            self.mlflow_tracker = MLflowExperimentTracker(
                experiment_name=experiment_name,
                tracking_uri=str(self.output_dir / "mlruns")
            )
            self.model_registry = MLflowModelRegistry(
                tracking_uri=str(self.output_dir / "mlruns")
            )
        
        # Initialize optimizer
        self.optimizer = optim.AdamW(
            learning_rate=learning_rate,
            weight_decay=weight_decay
        )
        
        # Training state
        self.global_step = 0
        self.best_val_loss = float('inf')
        self.best_val_accuracy = 0.0
        self.training_history = {
            'train_loss': [],
            'train_accuracy': [],
            'val_loss': [],
            'val_accuracy': [],
            'val_precision': [],
            'val_recall': [],
            'val_f1': [],
            'val_auc': [],
            'learning_rate': []
        }
        
        logger.info("Initialized TitanicTrainerV2 with enhanced logging and MLflow tracking")
    
    def log_configuration(self):
        """Log all configuration parameters."""
        config = {
            "model": {
                "type": "ModernBERT",
                "num_labels": self.model.bert.config.num_labels,
                "hidden_size": self.model.bert.config.hidden_size,
                "num_layers": self.model.bert.config.num_hidden_layers,
                "num_attention_heads": self.model.bert.config.num_attention_heads,
            },
            "training": {
                "learning_rate": self.learning_rate,
                "weight_decay": self.weight_decay,
                "warmup_steps": self.warmup_steps,
                "max_steps": self.max_steps,
                "eval_steps": self.eval_steps,
                "save_steps": self.save_steps,
                "gradient_accumulation_steps": self.gradient_accumulation_steps,
                "batch_size": self.train_loader.batch_size,
            },
            "data": {
                "train_samples": len(self.train_loader),
                "val_samples": len(self.val_loader) if self.val_loader else 0,
                "max_length": self.train_loader.max_length,
                "augmentation": self.train_loader.augment,
            }
        }
        
        # Log to logger
        LoggingConfig.log_hyperparameters(config["training"])
        LoggingConfig.log_model_info(config["model"])
        LoggingConfig.log_data_info(config["data"])
        
        # Log to MLflow
        if self.enable_mlflow:
            flat_config = {}
            for section, params in config.items():
                for key, value in params.items():
                    flat_config[f"{section}_{key}"] = value
            self.mlflow_tracker.log_params(flat_config)
        
        # Save configuration to file
        config_path = self.output_dir / "config.json"
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)
        logger.info(f"Saved configuration to {config_path}")
    
    def get_learning_rate(self) -> float:
        """Linear warmup and cosine decay."""
        if self.global_step < self.warmup_steps:
            lr = self.learning_rate * self.global_step / self.warmup_steps
        else:
            progress = (self.global_step - self.warmup_steps) / max(1, self.max_steps - self.warmup_steps)
            lr = self.learning_rate * 0.5 * (1 + mx.cos(mx.array(np.pi * progress)))
        
        return float(lr)
    
    @log_execution_time
    def train_step(self, batch: Dict[str, mx.array]) -> Tuple[mx.array, Dict[str, float]]:
        """Single training step with detailed logging."""
        # Log batch info (only occasionally to avoid spam)
        if self.global_step % 100 == 0:
            logger.debug(f"Batch shape: input_ids={batch['input_ids'].shape}, "
                        f"attention_mask={batch['attention_mask'].shape}, "
                        f"labels={batch['labels'].shape}")
        
        # Define loss function that will be differentiated
        def loss_fn(model, inputs, labels):
            outputs = model(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                labels=labels
            )
            return outputs['loss']
        
        # Use MLX's value_and_grad with the model
        loss_and_grad_fn = nn.value_and_grad(self.model, loss_fn)
        
        # Compute loss and gradients
        loss, grads = loss_and_grad_fn(
            self.model,
            {'input_ids': batch['input_ids'], 'attention_mask': batch['attention_mask']},
            batch['labels'].squeeze()
        )
        
        # Get outputs for metrics (run forward again for now)
        outputs = self.model(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask'],
            labels=batch['labels'].squeeze()
        )
        
        # Log gradient statistics
        if self.global_step % 50 == 0 and grads:
            grad_norms = {}
            
            # MLX returns gradients differently
            if isinstance(grads, dict):
                for name, grad in grads.items():
                    if isinstance(grad, mx.array):
                        grad_norm = float(mx.norm(grad))
                        grad_norms[f"grad_norm/{name}"] = grad_norm
            
            if grad_norms:
                logger.debug(f"Gradient norms at step {self.global_step}: "
                            f"min={min(grad_norms.values()):.4f}, "
                            f"max={max(grad_norms.values()):.4f}, "
                            f"mean={np.mean(list(grad_norms.values())):.4f}")
            else:
                logger.debug(f"No gradient norms to log at step {self.global_step}")
        
        # Update parameters
        self.optimizer.update(self.model, grads)
        
        # Update learning rate
        new_lr = self.get_learning_rate()
        self.optimizer.learning_rate = new_lr
        
        # Compute metrics
        predictions = mx.argmax(outputs['logits'], axis=-1)
        accuracy = mx.mean(predictions == batch['labels'].squeeze())
        
        metrics = {
            'loss': float(loss),
            'accuracy': float(accuracy),
            'learning_rate': float(new_lr),
            'batch_size': batch['input_ids'].shape[0]
        }
        
        # Log metrics
        LoggingConfig.log_metrics(metrics, step=self.global_step)
        
        return loss, metrics
    
    @log_execution_time
    def evaluate(self, dataloader: TitanicDataPipeline, phase: str = "val") -> Dict[str, float]:
        """Evaluate model with comprehensive metrics and logging."""
        logger.info(f"Starting {phase} evaluation...")
        
        total_loss = 0
        all_predictions = []
        all_labels = []
        all_probs = []
        all_logits = []
        
        num_batches = 0
        for batch in dataloader.get_dataloader()():
            outputs = self.model(
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask'],
                labels=batch['labels'].squeeze() if 'labels' in batch else None
            )
            
            if 'labels' in batch:
                total_loss += float(outputs['loss'])
                all_labels.extend(batch['labels'].squeeze().tolist())
            
            predictions = mx.argmax(outputs['logits'], axis=-1)
            probs = mx.softmax(outputs['logits'], axis=-1)
            
            all_predictions.extend(predictions.tolist())
            all_probs.extend(probs[:, 1].tolist())
            all_logits.extend(outputs['logits'].tolist())
            
            num_batches += 1
            
            if num_batches >= dataloader.get_num_batches():
                break
        
        metrics = {}
        
        if all_labels:
            # Calculate comprehensive metrics
            accuracy = accuracy_score(all_labels, all_predictions)
            precision, recall, f1, support = precision_recall_fscore_support(
                all_labels, all_predictions, average='binary'
            )
            
            try:
                auc = roc_auc_score(all_labels, all_probs)
            except:
                auc = 0.0
            
            # Calculate per-class support
            support_0 = all_labels.count(0)
            support_1 = all_labels.count(1)
            
            metrics = {
                f'{phase}_loss': total_loss / num_batches,
                f'{phase}_accuracy': accuracy,
                f'{phase}_precision': precision,
                f'{phase}_recall': recall,
                f'{phase}_f1': f1,
                f'{phase}_auc': auc,
                f'{phase}_support_0': support_0,
                f'{phase}_support_1': support_1,
            }
            
            # Log detailed metrics
            logger.info(f"{phase.capitalize()} Evaluation Results:")
            logger.info(f"  Loss: {metrics[f'{phase}_loss']:.4f}")
            logger.info(f"  Accuracy: {metrics[f'{phase}_accuracy']:.4f}")
            logger.info(f"  Precision: {metrics[f'{phase}_precision']:.4f}")
            logger.info(f"  Recall: {metrics[f'{phase}_recall']:.4f}")
            logger.info(f"  F1: {metrics[f'{phase}_f1']:.4f}")
            logger.info(f"  AUC: {metrics[f'{phase}_auc']:.4f}")
            
            # Log confusion matrix details
            from sklearn.metrics import confusion_matrix
            cm = confusion_matrix(all_labels, all_predictions)
            logger.info(f"  Confusion Matrix:")
            logger.info(f"    TN={cm[0,0]}, FP={cm[0,1]}")
            logger.info(f"    FN={cm[1,0]}, TP={cm[1,1]}")
            
            # Log to MLflow
            if self.enable_mlflow and phase == "val":
                self.mlflow_tracker.log_confusion_matrix(
                    np.array(all_labels),
                    np.array(all_predictions),
                    labels=['Not Survived', 'Survived'],
                    step=self.global_step
                )
                
                self.mlflow_tracker.log_roc_curve(
                    np.array(all_labels),
                    np.array(all_probs),
                    step=self.global_step
                )
        
        return metrics
    
    def train(self, num_epochs: int):
        """Enhanced training loop with comprehensive logging and tracking."""
        # Calculate total steps if not provided
        if self.max_steps is None:
            self.max_steps = num_epochs * self.train_loader.get_num_batches()
        
        # Start MLflow run
        if self.enable_mlflow:
            self.mlflow_tracker.start_run(
                run_name=f"train_{time.strftime('%Y%m%d_%H%M%S')}",
                tags={
                    "epochs": str(num_epochs),
                    "model_type": "ModernBERT",
                    "dataset": "Titanic"
                }
            )
        
        # Log configuration
        self.log_configuration()
        
        # Create experiment context
        with ExperimentLogger(
            experiment_name=f"titanic_training_{num_epochs}_epochs",
            config={
                "epochs": num_epochs,
                "max_steps": self.max_steps,
                "learning_rate": self.learning_rate
            }
        ):
            logger.info(f"Starting training for {num_epochs} epochs ({self.max_steps} steps)")
            logger.info(f"Train samples: {len(self.train_loader)}")
            if self.val_loader:
                logger.info(f"Val samples: {len(self.val_loader)}")
            
            # Training loop
            for epoch in range(num_epochs):
                logger.info(f"\n{'='*60}")
                logger.info(f"Epoch {epoch + 1}/{num_epochs}")
                logger.info(f"{'='*60}")
                
                # Training phase
                epoch_loss = 0
                epoch_metrics = {'loss': 0, 'accuracy': 0, 'learning_rate': 0}
                num_batches = 0
                
                pbar = tqdm(total=self.train_loader.get_num_batches(), desc="Training")
                
                for batch_idx, batch in enumerate(self.train_loader.get_dataloader()()):
                    try:
                        loss, metrics = self.train_step(batch)
                        
                        # Accumulate metrics
                        for key, value in metrics.items():
                            if key in epoch_metrics:
                                epoch_metrics[key] += value
                        
                        self.global_step += 1
                        num_batches += 1
                        
                        # Update progress bar
                        pbar.update(1)
                        pbar.set_postfix({
                            'loss': f"{metrics['loss']:.4f}",
                            'acc': f"{metrics['accuracy']:.4f}",
                            'lr': f"{metrics['learning_rate']:.6f}"
                        })
                        
                        # Log to MLflow
                        if self.enable_mlflow:
                            self.mlflow_tracker.log_metrics(
                                {
                                    "train_loss": metrics['loss'],
                                    "train_accuracy": metrics['accuracy'],
                                    "learning_rate": metrics['learning_rate']
                                },
                                step=self.global_step
                            )
                        
                        # Periodic evaluation
                        if self.val_loader and self.global_step % self.eval_steps == 0:
                            val_metrics = self.evaluate(self.val_loader, phase="val")
                            
                            # Log to MLflow
                            if self.enable_mlflow:
                                self.mlflow_tracker.log_metrics(val_metrics, step=self.global_step)
                            
                            # Update training history
                            for key, value in val_metrics.items():
                                if key in self.training_history:
                                    self.training_history[key].append(value)
                            
                            # Save best model
                            if val_metrics['val_loss'] < self.best_val_loss:
                                self.best_val_loss = val_metrics['val_loss']
                                self.save_checkpoint('best_model_loss')
                                logger.success(f"New best model (loss) saved at step {self.global_step}")
                            
                            if val_metrics['val_accuracy'] > self.best_val_accuracy:
                                self.best_val_accuracy = val_metrics['val_accuracy']
                                self.save_checkpoint('best_model_accuracy')
                                logger.success(f"New best model (accuracy) saved at step {self.global_step}")
                        
                        # Periodic checkpoint
                        if self.global_step % self.save_steps == 0:
                            self.save_checkpoint(f'checkpoint_{self.global_step}')
                        
                        if num_batches >= self.train_loader.get_num_batches():
                            break
                            
                    except Exception as e:
                        logger.error(f"Error in training step {self.global_step}: {str(e)}")
                        logger.exception("Detailed error:")
                        raise
                
                pbar.close()
                
                # Average epoch metrics
                for key in epoch_metrics:
                    epoch_metrics[key] /= num_batches
                
                # Update training history
                self.training_history['train_loss'].append(epoch_metrics['loss'])
                self.training_history['train_accuracy'].append(epoch_metrics['accuracy'])
                self.training_history['learning_rate'].append(epoch_metrics['learning_rate'])
                
                logger.info(f"\nEpoch {epoch + 1} Summary:")
                logger.info(f"  Average Loss: {epoch_metrics['loss']:.4f}")
                logger.info(f"  Average Accuracy: {epoch_metrics['accuracy']:.4f}")
                
                # End of epoch evaluation
                if self.val_loader:
                    val_metrics = self.evaluate(self.val_loader, phase="val")
                    
                    if self.enable_mlflow:
                        self.mlflow_tracker.log_metrics(val_metrics, step=self.global_step)
            
            # Save final model
            self.save_checkpoint('final_model')
            
            # Log training curves
            if self.enable_mlflow:
                self.mlflow_tracker.log_training_curves(self.training_history)
            
            # Save training history
            history_path = self.output_dir / "training_history.json"
            with open(history_path, "w") as f:
                json.dump(self.training_history, f, indent=2)
            logger.info(f"Saved training history to {history_path}")
            
            logger.success("\nTraining completed successfully!")
            logger.info(f"Best validation loss: {self.best_val_loss:.4f}")
            logger.info(f"Best validation accuracy: {self.best_val_accuracy:.4f}")
        
        # End MLflow run
        if self.enable_mlflow:
            # Log final artifacts
            self.mlflow_tracker.log_artifacts([
                str(self.output_dir / "config.json"),
                str(self.output_dir / "training_history.json")
            ])
            
            # Register best model
            if self.best_val_accuracy > 0:
                try:
                    model_version = self.model_registry.register_model(
                        run_id=mlflow.active_run().info.run_id,
                        model_name="titanic_modernbert",
                        model_path="models/best_model_accuracy",
                        tags={"accuracy": str(self.best_val_accuracy)}
                    )
                    logger.info(f"Registered model version: {model_version}")
                except Exception as e:
                    logger.warning(f"Failed to register model: {e}")
            
            self.mlflow_tracker.end_run()
    
    def save_checkpoint(self, name: str):
        """Save model checkpoint with enhanced logging."""
        checkpoint_dir = self.output_dir / name
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Saving checkpoint '{name}'...")
        
        # Save model
        self.model.bert.save_pretrained(str(checkpoint_dir / "bert"))
        
        # Save classifier weights
        classifier_weights = {}
        for name, param in self.model.classifier.parameters().items():
            classifier_weights[name] = np.array(param)
        
        np.savez(str(checkpoint_dir / "classifier_weights.npz"), **classifier_weights)
        
        # Save training state
        state = {
            'global_step': self.global_step,
            'best_val_loss': float(self.best_val_loss),
            'best_val_accuracy': float(self.best_val_accuracy),
            'training_history': self.training_history,
            'optimizer_state': {
                'learning_rate': float(self.optimizer.learning_rate),
                'weight_decay': float(self.optimizer.weight_decay)
            }
        }
        
        with open(checkpoint_dir / "training_state.json", "w") as f:
            json.dump(state, f, indent=2)
        
        # Log to MLflow
        if self.enable_mlflow and mlflow.active_run():
            self.mlflow_tracker.log_model(
                str(checkpoint_dir),
                name,
                metadata=state
            )
        
        logger.success(f"Checkpoint saved to {checkpoint_dir}")


def create_trainer_v2(
    train_path: str,
    val_path: Optional[str] = None,
    model_name: str = "answerdotai/ModernBERT-base",
    learning_rate: float = 2e-5,
    batch_size: int = 16,
    num_epochs: int = 3,
    output_dir: str = "./output",
    experiment_name: str = "titanic_modernbert",
    enable_mlflow: bool = True
) -> TitanicTrainerV2:
    """Create an enhanced trainer with logging and MLflow."""
    from data.titanic_loader import create_data_loaders
    
    logger.info("Creating enhanced trainer with MLflow and logging...")
    
    # Create data loaders
    train_loader, val_loader = create_data_loaders(
        train_path=train_path,
        val_path=val_path,
        batch_size=batch_size
    )
    
    # Create model
    bert_model = create_model(model_name)
    model = TitanicClassifier(bert_model)
    
    # Create trainer
    trainer = TitanicTrainerV2(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        learning_rate=learning_rate,
        output_dir=output_dir,
        experiment_name=experiment_name,
        enable_mlflow=enable_mlflow
    )
    
    return trainer