#!/usr/bin/env python3
"""Debug script to test the optimized training step by step."""

import mlx.core as mx
import mlx.optimizers as optim
from transformers import AutoTokenizer
from loguru import logger

from data.mlx_enhanced_loader import create_enhanced_dataloaders
from models.modernbert_cnn_hybrid import CNNEnhancedModernBERT, CNNHybridConfig
from training.mlx_optimized_trainer import MLXOptimizedTrainer, OptimizedTrainingConfig

# Initialize basic components
logger.info("Initializing debug training test...")

# Set seed
mx.random.seed(42)

# Initialize tokenizer
tokenizer = AutoTokenizer.from_pretrained("answerdotai/ModernBERT-base")
logger.info("Tokenizer loaded")

# Create a simple data loader
train_loader, val_loader, test_loader = create_enhanced_dataloaders(
    data_dir="data/titanic",
    tokenizer=tokenizer,
    batch_size=8,
    max_length=128,
    train_file="train.csv",
    val_file="val.csv",
    test_file="test.csv",
    num_threads=4,
    prefetch_size=2,
    enable_augmentation=True,
    memory_map=True,
    double_buffer=True,
)
logger.info("Data loaders created")

# Initialize model
config = CNNHybridConfig(
    vocab_size=tokenizer.vocab_size,
    hidden_size=768,
    num_hidden_layers=12,
    num_attention_heads=12,
    intermediate_size=3072,
    max_position_embeddings=8192,
    num_labels=2,
    cnn_kernel_sizes=[3, 5, 7],
    cnn_num_filters=256,
    cnn_dropout=0.3,
)
model = CNNEnhancedModernBERT(config)
logger.info("Model created")

# Initialize optimizer
optimizer = optim.AdamW(learning_rate=2e-5, weight_decay=0.01)
logger.info("Optimizer created")

# Create training config
training_config = OptimizedTrainingConfig(
    learning_rate=2e-5,
    num_epochs=1,
    warmup_ratio=0.1,
    weight_decay=0.01,
    base_batch_size=8,
    max_batch_size=16,
    gradient_accumulation_steps=1,
    eval_batch_size=64,
    lazy_eval_interval=10,
    memory_threshold=0.8,
    num_workers=4,
    prefetch_size=2,
    save_steps=100,
    eval_steps=100,
    checkpoint_dir="output/debug_run",
    enable_profiling=False,
)
logger.info("Training config created")

# Initialize trainer
trainer = MLXOptimizedTrainer(
    model=model,
    optimizer=optimizer,
    config=training_config,
)
logger.info("Trainer created")

# Test single training step
logger.info("Testing single training step...")

# Get a single batch
train_loader._initialize_optimized_stream()
batch_iter = train_loader.stream()
batch = next(batch_iter)
logger.info(f"Got batch with keys: {list(batch.keys())}")

# Check the initial state
logger.info(f"Initial accumulated_grads: {trainer.accumulated_grads}")
logger.info(f"Initial accumulated_loss: {trainer.accumulated_loss}")

# Test single step
try:
    # Add some debug logging
    logger.info(f"Before training step - global_step: {trainer.global_step}")
    logger.info(f"Before training step - gradient_accumulation_steps: {trainer.config.gradient_accumulation_steps}")
    
    loss, metrics = trainer.train_step_lazy(batch)
    logger.info(f"Training step successful! Loss: {loss}, Metrics: {metrics}")
except Exception as e:
    logger.error(f"Training step failed: {e}")
    import traceback
    traceback.print_exc()

logger.info("Debug test completed")