#!/usr/bin/env python3
"""Train ModernBERT CNN Hybrid with MLX optimizations.

This script demonstrates using the optimized trainer with MLX best practices:
- Dynamic batch sizing
- Lazy computation with explicit eval()
- Gradient accumulation
- Memory-efficient data pipeline
"""

import argparse
import os
from datetime import datetime
from pathlib import Path

import mlx.core as mx
import mlx.optimizers as optim
from loguru import logger
from transformers import AutoTokenizer

from data.unified_loader import create_optimized_dataloaders
from models.modernbert_cnn_hybrid import CNNEnhancedModernBERT, ModernBERTConfig
from training.mlx_optimized_trainer import MLXOptimizedTrainer, OptimizedTrainingConfig
from utils.logging_config import LoggingConfig
from utils.mlflow_helper import MLflowHelper


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train CNN-Enhanced ModernBERT with MLX optimizations"
    )
    
    # Data arguments
    parser.add_argument(
        "--data_dir", type=str, default="data/titanic", help="Directory containing data"
    )
    parser.add_argument(
        "--train_file", type=str, default="train.csv", help="Training data filename"
    )
    parser.add_argument("--val_file", type=str, default="val.csv", help="Validation filename")
    parser.add_argument("--test_file", type=str, default="test.csv", help="Test filename")
    
    # Model arguments
    parser.add_argument(
        "--model_name",
        type=str,
        default="answerdotai/ModernBERT-base",
        help="Model name or path",
    )
    parser.add_argument(
        "--config_path", type=str, default=None, help="Path to model config JSON"
    )
    
    # Training arguments
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument(
        "--base_batch_size", type=int, default=32, help="Base batch size for training"
    )
    parser.add_argument(
        "--max_batch_size", type=int, default=64, help="Maximum batch size for dynamic sizing"
    )
    parser.add_argument(
        "--eval_batch_size", type=int, default=64, help="Batch size for evaluation"
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of gradient accumulation steps",
    )
    parser.add_argument("--learning_rate", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--warmup_ratio", type=float, default=0.1, help="Warmup ratio")
    
    # Optimization arguments
    parser.add_argument(
        "--num_workers", type=int, default=8, help="Number of data loading workers"
    )
    parser.add_argument(
        "--prefetch_size", type=int, default=4, help="Number of batches to prefetch"
    )
    parser.add_argument(
        "--lazy_eval_interval",
        type=int,
        default=10,
        help="Force evaluation every N steps",
    )
    parser.add_argument(
        "--memory_threshold",
        type=float,
        default=0.8,
        help="Memory usage threshold for batch size adjustment",
    )
    
    # Output arguments
    parser.add_argument(
        "--output_dir",
        type=str,
        default="output/optimized",
        help="Directory to save outputs",
    )
    parser.add_argument("--save_steps", type=int, default=100, help="Save checkpoint steps")
    parser.add_argument("--eval_steps", type=int, default=100, help="Evaluation steps")
    
    # Experiment tracking
    parser.add_argument(
        "--experiment_name",
        type=str,
        default="titanic_optimized",
        help="MLflow experiment name",
    )
    parser.add_argument(
        "--run_name", type=str, default=None, help="MLflow run name (auto-generated if None)"
    )
    parser.add_argument(
        "--enable_mlflow", action="store_true", help="Enable MLflow tracking"
    )
    
    # Other arguments
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--enable_profiling", action="store_true", help="Enable memory profiling"
    )
    parser.add_argument(
        "--log_level", type=str, default="INFO", help="Logging level", choices=["DEBUG", "INFO", "WARNING", "ERROR"]
    )
    
    return parser.parse_args()


def main():
    """Main training function."""
    args = parse_args()
    
    # Setup logging
    LoggingConfig.setup(
        log_dir=Path(args.output_dir) / "logs",
        log_level=args.log_level,
        prefix="optimized_training",
    )
    
    logger.info("Starting MLX Optimized Training")
    logger.info(f"Arguments: {args}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set random seed
    mx.random.seed(args.seed)
    
    # Initialize tokenizer
    logger.info(f"Loading tokenizer: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    
    # Create data loaders with optimizations
    logger.info("Creating optimized data loaders...")
    train_loader, val_loader, test_loader = create_optimized_dataloaders(
        data_dir=args.data_dir,
        tokenizer=tokenizer,
        batch_size=args.base_batch_size,
        max_length=128,
        train_file=args.train_file,
        val_file=args.val_file,
        test_file=args.test_file,
        num_threads=args.num_workers,
        prefetch_size=args.prefetch_size,
        pre_tokenize=True,  # Enable pre-tokenization
        use_mlx_data=True,  # Use MLX-Data pipeline
    )
    
    # Initialize model
    logger.info("Initializing CNN-Enhanced ModernBERT model...")
    if args.config_path:
        import json
        
        with open(args.config_path) as f:
            config_dict = json.load(f)
        config = ModernBERTConfig(**config_dict)
    else:
        config = ModernBERTConfig(
            vocab_size=tokenizer.vocab_size,
            hidden_size=768,
            num_hidden_layers=12,
            num_attention_heads=12,
            intermediate_size=3072,
            max_position_embeddings=8192,
            num_labels=2,
            pad_token_id=tokenizer.pad_token_id,
            # CNN config
            cnn_num_filters=[128, 256, 512],
            cnn_filter_sizes=[3, 5, 7],
            cnn_dropout=0.3,
        )
    
    model = CNNEnhancedModernBERT(config)
    logger.info(f"Model initialized with {sum(p.size for p in model.parameters()):,} parameters")
    
    # Initialize optimizer
    optimizer = optim.AdamW(
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    
    # Create training config
    training_config = OptimizedTrainingConfig(
        learning_rate=args.learning_rate,
        num_epochs=args.epochs,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        base_batch_size=args.base_batch_size,
        max_batch_size=args.max_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        eval_batch_size=args.eval_batch_size,
        lazy_eval_interval=args.lazy_eval_interval,
        memory_threshold=args.memory_threshold,
        num_workers=args.num_workers,
        prefetch_size=args.prefetch_size,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps,
        checkpoint_dir=args.output_dir,
        enable_profiling=args.enable_profiling,
    )
    
    # Initialize trainer
    trainer = MLXOptimizedTrainer(
        model=model,
        optimizer=optimizer,
        config=training_config,
    )
    
    # Setup MLflow if enabled
    mlflow_helper = None
    if args.enable_mlflow:
        mlflow_helper = MLflowHelper(
            experiment_name=args.experiment_name,
            tracking_uri=f"file://{args.output_dir}/mlruns",
        )
        
        run_name = args.run_name or f"optimized_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        mlflow_helper.start_run(run_name=run_name)
        
        # Log parameters
        mlflow_helper.log_params(
            {
                "model_type": "cnn_hybrid_optimized",
                "base_batch_size": args.base_batch_size,
                "max_batch_size": args.max_batch_size,
                "gradient_accumulation": args.gradient_accumulation_steps,
                "learning_rate": args.learning_rate,
                "epochs": args.epochs,
                "num_workers": args.num_workers,
                "prefetch_size": args.prefetch_size,
                "pre_tokenize": True,
                "use_mlx_data": True,
            }
        )
    
    # Train model
    logger.info("Starting optimized training...")
    try:
        trainer.train(
            train_dataloader=train_loader,
            val_dataloader=val_loader,
            test_dataloader=test_loader,
        )
        
        logger.success("Training completed successfully!")
        
        # Log final metrics if MLflow is enabled
        if mlflow_helper and hasattr(trainer, "best_metric"):
            mlflow_helper.log_metric("best_val_accuracy", trainer.best_metric)
            
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise
    finally:
        if mlflow_helper:
            mlflow_helper.end_run()
    
    # Save final model
    final_path = Path(args.output_dir) / "final_model"
    trainer.model.save_pretrained(str(final_path))
    logger.info(f"Final model saved to {final_path}")
    
    # Print memory profile summary if enabled
    if args.enable_profiling and trainer.memory_history:
        avg_memory = sum(h["memory"] for h in trainer.memory_history) / len(
            trainer.memory_history
        )
        max_memory = max(h["memory"] for h in trainer.memory_history)
        logger.info(
            f"Memory Profile Summary - Avg: {avg_memory:.1%}, Max: {max_memory:.1%}"
        )


if __name__ == "__main__":
    main()