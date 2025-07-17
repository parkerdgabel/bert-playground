#!/usr/bin/env python3
"""Simple MLflow training script for Titanic with ModernBERT."""

import sys
from pathlib import Path
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from loguru import logger

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

# Apply MLX compatibility patches
from utils.mlx_patch import apply_mlx_patches
apply_mlx_patches()

# Core imports
from models.factory import create_model
from models.modernbert import ModernBertConfig
from data import create_kaggle_dataloader
from training.config import TrainingConfig
from training.mlx_trainer import MLXTrainer
from utils.mlflow_central import setup_central_mlflow

# Simple classifier for ModernBERT
class SimpleClassifier(nn.Module):
    """Simple classifier that works with any ModernBERT model."""
    
    def __init__(self, bert_model, num_classes=2, dropout=0.1):
        super().__init__()
        self.bert = bert_model
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(768, num_classes)  # ModernBERT base hidden size
        
    def __call__(self, input_ids, attention_mask=None, labels=None):
        # Forward pass through BERT
        bert_output = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        # Use pooled output (CLS token)
        pooled_output = bert_output["pooler_output"]
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        
        # Compute loss if labels provided
        loss = None
        if labels is not None:
            # Simple cross-entropy loss - make sure labels are the right shape
            labels_flat = labels.squeeze() if labels.ndim > 1 else labels
            loss = nn.losses.cross_entropy(logits, labels_flat, reduction='mean')
        
        return {
            "logits": logits,
            "loss": loss,
            "last_hidden_state": bert_output["last_hidden_state"],
            "pooler_output": pooled_output
        }


def main():
    """Main training function."""
    logger.info("Starting MLflow Titanic training")
    
    # Setup MLflow
    setup_central_mlflow(experiment_name="titanic_simple_training")
    
    # Data paths
    train_path = "data/titanic/train.csv"
    val_path = "data/titanic/val.csv"
    
    # Create data loaders
    logger.info("Loading data...")
    train_loader = create_kaggle_dataloader(
        dataset_name="titanic",
        csv_path=train_path,
        tokenizer_name="answerdotai/ModernBERT-base",
        batch_size=32,
        max_length=256,
        shuffle=True,
        shuffle_buffer_size=1000,
        prefetch_size=4,
        num_workers=4,
        tokenizer_backend="auto",
    )
    
    val_loader = create_kaggle_dataloader(
        dataset_name="titanic",
        csv_path=val_path,
        tokenizer_name="answerdotai/ModernBERT-base",
        batch_size=64,
        max_length=256,
        shuffle=False,
        prefetch_size=2,
        num_workers=2,
        tokenizer_backend="auto",
    )
    
    logger.info(f"Loaded {len(train_loader)} train batches, {len(val_loader)} val batches")
    
    # Create model
    logger.info("Creating model...")
    bert_model = create_model("standard")
    model = SimpleClassifier(bert_model)
    
    # Create training configuration
    config = TrainingConfig(
        # Basic parameters
        learning_rate=2e-5,
        epochs=3,
        batch_size=32,
        
        # Data configuration
        train_path=train_path,
        val_path=val_path,
        
        # Output
        output_dir="./output/simple_mlflow_run",
        experiment_name="titanic_simple_training"
    )
    
    # Create trainer
    logger.info("Creating trainer...")
    trainer = MLXTrainer(model, config)
    
    # Run training
    logger.info("Starting training...")
    try:
        results = trainer.train(train_loader, val_loader)
        logger.info(f"Training completed successfully!")
        logger.info(f"Best metric: {results['best_metric']:.4f}")
        logger.info(f"Total time: {results['total_time']:.1f}s")
        return 0
    except Exception as e:
        logger.error(f"Training failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())