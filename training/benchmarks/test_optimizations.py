"""
Test script to benchmark the training optimizations.
"""

import time
import numpy as np
import mlx.core as mx
import mlx.nn as nn
from typing import Dict
from loguru import logger

from training.benchmarks import PerformanceMonitor, MemoryTracker
from training.core.base import BaseTrainer
from training.core.config import get_quick_test_config


class DummyModel(nn.Module):
    """Simple model for benchmarking."""
    
    def __init__(self, input_dim: int = 768, hidden_dim: int = 256, num_classes: int = 2):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_classes)
        self.config = None
        
    def __call__(self, batch: Dict[str, mx.array]) -> Dict[str, mx.array]:
        # Handle both dict and unpacked inputs
        if isinstance(batch, dict):
            x = batch.get("input_ids", batch.get("features"))
            labels = batch.get("labels")
        else:
            x = batch
            labels = None
            
        x = self.fc1(x)
        x = nn.relu(x)
        logits = self.fc2(x)
        
        outputs = {"logits": logits}
        
        # Compute loss if labels provided
        if labels is not None:
            loss = mx.mean(nn.losses.cross_entropy(logits, labels))
            outputs["loss"] = loss
            
        return outputs


class DummyDataLoader:
    """Simple data loader for benchmarking."""
    
    def __init__(self, num_samples: int = 1000, batch_size: int = 32, 
                 input_dim: int = 768, num_classes: int = 2):
        self.num_samples = num_samples
        self.batch_size = batch_size
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.num_batches = num_samples // batch_size
        
    def __iter__(self):
        for i in range(self.num_batches):
            # Create random batch
            features = mx.random.normal((self.batch_size, self.input_dim))
            labels = mx.random.randint(0, self.num_classes, (self.batch_size,))
            
            yield {
                "features": features,
                "labels": labels,
                "input_ids": features,  # Alias for compatibility
            }
    
    def __len__(self):
        return self.num_batches


def benchmark_training_step():
    """Benchmark individual training step performance."""
    logger.info("=== Benchmarking Training Step Performance ===")
    
    # Setup
    config = get_quick_test_config()
    config.training.logging_steps = 1000  # Reduce logging frequency
    config.training.num_epochs = 1
    config.data.batch_size = 64
    
    model = DummyModel()
    trainer = BaseTrainer(model, config)
    
    # Create dummy batch
    batch = {
        "features": mx.random.normal((config.data.batch_size, 768)),
        "labels": mx.random.randint(0, 2, (config.data.batch_size,)),
        "input_ids": mx.random.normal((config.data.batch_size, 768)),
    }
    
    # Performance monitor
    monitor = PerformanceMonitor(warmup_steps=10)
    
    # Benchmark uncompiled vs compiled
    logger.info("Testing uncompiled training step...")
    uncompiled_times = []
    for i in range(50):
        start = time.perf_counter()
        loss, metrics = trainer._train_step(batch)
        mx.eval(loss)
        uncompiled_times.append(time.perf_counter() - start)
    
    logger.info("Testing compiled training step...")
    compiled_times = []
    if trainer._use_compiled:
        for i in range(50):
            start = time.perf_counter()
            loss, metrics = trainer._compiled_train_step(batch)
            mx.eval(loss)
            compiled_times.append(time.perf_counter() - start)
    
    # Results
    avg_uncompiled = np.mean(uncompiled_times[10:])  # Skip warmup
    logger.info(f"Uncompiled step time: {avg_uncompiled*1000:.2f}ms")
    
    if compiled_times:
        avg_compiled = np.mean(compiled_times[10:])
        speedup = avg_uncompiled / avg_compiled
        logger.info(f"Compiled step time: {avg_compiled*1000:.2f}ms")
        logger.info(f"Speedup from compilation: {speedup:.2f}x")
    
    return trainer


def benchmark_full_training():
    """Benchmark full training loop."""
    logger.info("\n=== Benchmarking Full Training Loop ===")
    
    # Setup
    config = get_quick_test_config()
    config.training.num_epochs = 2
    config.training.logging_steps = 50
    config.data.batch_size = 64
    config.training.eval_strategy = "epoch"
    
    model = DummyModel()
    trainer = BaseTrainer(model, config)
    
    # Create data loaders
    train_loader = DummyDataLoader(num_samples=1000, batch_size=config.data.batch_size)
    val_loader = DummyDataLoader(num_samples=200, batch_size=config.data.batch_size)
    
    # Memory tracker
    memory_tracker = MemoryTracker()
    memory_tracker.reset()
    
    # Train
    logger.info("Starting training benchmark...")
    start_time = time.perf_counter()
    
    result = trainer.train(train_loader, val_loader)
    
    total_time = time.perf_counter() - start_time
    
    # Report results
    logger.info(f"\nTraining completed in {total_time:.2f} seconds")
    logger.info(f"Average time per epoch: {total_time/config.training.num_epochs:.2f} seconds")
    logger.info(f"Throughput: {(train_loader.num_samples * config.training.num_epochs) / total_time:.1f} samples/sec")
    
    # Memory stats
    memory_stats = memory_tracker.get_stats()
    logger.info(f"\nMemory usage - Peak: {memory_stats['peak_gb']:.2f}GB")
    if "peak_used_gb" in memory_stats:
        logger.info(f"Memory usage - Peak Used: {memory_stats['peak_used_gb']:.2f}GB")


def compare_optimization_impact():
    """Compare performance with and without optimizations."""
    logger.info("\n=== Comparing Optimization Impact ===")
    
    # This would require toggling optimizations on/off
    # For now, we'll just report current performance
    
    config = get_quick_test_config()
    config.data.batch_size = 128
    
    model = DummyModel()
    trainer = BaseTrainer(model, config)
    
    # Test batch processing speed
    loader = DummyDataLoader(num_samples=512, batch_size=config.data.batch_size)
    
    logger.info("Processing batches...")
    start = time.perf_counter()
    
    for i, batch in enumerate(loader):
        if trainer._use_compiled:
            loss, metrics = trainer._compiled_train_step(batch)
        else:
            loss, metrics = trainer._train_step(batch)
        
        # Minimal evaluation to measure true performance
        if i == len(loader) - 1:
            mx.eval(loss)
    
    elapsed = time.perf_counter() - start
    throughput = loader.num_samples / elapsed
    
    logger.info(f"Batch processing time: {elapsed:.2f}s")
    logger.info(f"Throughput: {throughput:.1f} samples/sec")
    logger.info(f"Time per batch: {elapsed/len(loader)*1000:.2f}ms")


if __name__ == "__main__":
    logger.info("Starting MLX Training Optimization Benchmarks")
    logger.info(f"MLX version: {mx.__version__}")
    
    # Run benchmarks
    trainer = benchmark_training_step()
    benchmark_full_training()
    compare_optimization_impact()
    
    logger.info("\n=== Benchmark Complete ===")
    logger.info("Key optimizations implemented:")
    logger.info("✓ Removed excessive mx.eval() calls")
    logger.info("✓ Implemented lazy evaluation throughout training loop")
    logger.info("✓ Added mx.compile() for training step")
    logger.info("✓ Deferred metric conversions")
    logger.info("\nNext steps: Optimize gradient statistics and accumulation")