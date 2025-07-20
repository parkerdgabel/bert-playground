"""
Test Phase 2 optimizations - gradient stats, accumulation, and checkpointing.
"""

import time
import tempfile
import shutil
from pathlib import Path
import mlx.core as mx
import mlx.nn as nn
from loguru import logger

from training.benchmarks import PerformanceMonitor, MemoryTracker
from training.core.base import BaseTrainer
from training.core.config import get_quick_test_config
from training.core.optimization import GradientAccumulator, compute_gradient_stats
from training.benchmarks.test_optimizations import DummyModel, DummyDataLoader


def benchmark_gradient_stats():
    """Benchmark gradient statistics computation."""
    logger.info("=== Benchmarking Gradient Statistics ===")
    
    # Create dummy gradients
    dummy_grads = {
        "layer1": {
            "weight": mx.random.normal((768, 256)),
            "bias": mx.random.normal((256,))
        },
        "layer2": {
            "weight": mx.random.normal((256, 128)),
            "bias": mx.random.normal((128,))
        },
        "output": mx.random.normal((128, 10))
    }
    
    # Benchmark old vs new approach
    num_iterations = 100
    
    logger.info("Testing optimized gradient stats (norm only)...")
    start = time.perf_counter()
    for _ in range(num_iterations):
        stats = compute_gradient_stats(dummy_grads)
        # Force evaluation of grad_norm
        if hasattr(stats["grad_norm"], 'item'):
            mx.eval(stats["grad_norm"])
    optimized_time = time.perf_counter() - start
    
    logger.info("Testing detailed gradient stats...")
    from training.core.optimization import compute_gradient_stats_detailed
    start = time.perf_counter()
    for _ in range(num_iterations):
        stats = compute_gradient_stats_detailed(dummy_grads)
    detailed_time = time.perf_counter() - start
    
    logger.info(f"Optimized (norm only): {optimized_time:.3f}s ({optimized_time/num_iterations*1000:.2f}ms per call)")
    logger.info(f"Detailed stats: {detailed_time:.3f}s ({detailed_time/num_iterations*1000:.2f}ms per call)")
    logger.info(f"Speedup: {detailed_time/optimized_time:.2f}x")


def benchmark_gradient_accumulation():
    """Benchmark gradient accumulation with tree_map."""
    logger.info("\n=== Benchmarking Gradient Accumulation ===")
    
    accumulator = GradientAccumulator(accumulation_steps=4)
    
    # Create dummy gradients
    def create_grads():
        return {
            "layer1": {
                "weight": mx.random.normal((768, 256)),
                "bias": mx.random.normal((256,))
            },
            "layer2": {
                "weight": mx.random.normal((256, 128)),
                "bias": mx.random.normal((128,))
            }
        }
    
    # Benchmark accumulation
    num_iterations = 100
    start = time.perf_counter()
    
    for i in range(num_iterations):
        grads = create_grads()
        should_update = accumulator.accumulate(grads)
        
        if should_update:
            accumulated = accumulator.get_gradients()
            # Force evaluation
            mx.eval(accumulated["layer1"]["weight"])
    
    elapsed = time.perf_counter() - start
    
    logger.info(f"Accumulation time for {num_iterations} iterations: {elapsed:.3f}s")
    logger.info(f"Average per accumulation: {elapsed/num_iterations*1000:.2f}ms")


def benchmark_checkpoint_saving():
    """Benchmark checkpoint saving optimizations."""
    logger.info("\n=== Benchmarking Checkpoint Saving ===")
    
    # Create temporary directory
    with tempfile.TemporaryDirectory() as tmpdir:
        config = get_quick_test_config()
        config.environment.output_dir = Path(tmpdir)
        config.training.save_steps = 10
        
        model = DummyModel()
        trainer = BaseTrainer(model, config)
        
        # Create dummy state
        trainer.state.epoch = 5
        trainer.state.global_step = 1000
        trainer.state.train_loss = 0.123
        trainer.state.val_loss = 0.456
        
        # Benchmark checkpoint saving
        num_saves = 10
        start = time.perf_counter()
        
        for i in range(num_saves):
            checkpoint_path = trainer._save_checkpoint(is_best=False)
        
        elapsed = time.perf_counter() - start
        
        logger.info(f"Saved {num_saves} checkpoints in {elapsed:.3f}s")
        logger.info(f"Average save time: {elapsed/num_saves*1000:.0f}ms")
        
        # Check checkpoint size
        checkpoint_size = sum(f.stat().st_size for f in checkpoint_path.rglob("*") if f.is_file())
        logger.info(f"Checkpoint size: {checkpoint_size / 1024:.1f}KB")


def benchmark_full_training_phase2():
    """Benchmark full training with Phase 2 optimizations."""
    logger.info("\n=== Full Training Benchmark with Phase 2 Optimizations ===")
    
    config = get_quick_test_config()
    config.training.num_epochs = 2
    config.training.gradient_accumulation_steps = 2
    config.data.batch_size = 64
    config.training.logging_steps = 100  # Less frequent logging
    
    model = DummyModel()
    trainer = BaseTrainer(model, config)
    
    train_loader = DummyDataLoader(num_samples=1000, batch_size=config.data.batch_size)
    val_loader = DummyDataLoader(num_samples=200, batch_size=config.data.batch_size)
    
    # Memory tracking
    memory_tracker = MemoryTracker()
    memory_tracker.reset()
    
    # Training
    start_time = time.perf_counter()
    result = trainer.train(train_loader, val_loader)
    total_time = time.perf_counter() - start_time
    
    # Results
    total_samples = train_loader.num_samples * config.training.num_epochs
    throughput = total_samples / total_time
    
    logger.info(f"\nTraining completed in {total_time:.2f}s")
    logger.info(f"Throughput: {throughput:.1f} samples/sec")
    logger.info(f"With gradient accumulation: effective batch size = {config.data.batch_size * config.training.gradient_accumulation_steps}")
    
    # Memory stats
    memory_stats = memory_tracker.get_stats()
    logger.info(f"Peak memory usage: {memory_stats['peak_gb']:.2f}GB")


if __name__ == "__main__":
    logger.info("Starting Phase 2 Optimization Benchmarks")
    logger.info(f"MLX version: {mx.__version__}")
    
    # Run benchmarks
    benchmark_gradient_stats()
    benchmark_gradient_accumulation()
    benchmark_checkpoint_saving()
    benchmark_full_training_phase2()
    
    logger.info("\n=== Phase 2 Optimizations Complete ===")
    logger.info("Key improvements:")
    logger.info("✓ Gradient stats optimized with tree_flatten")
    logger.info("✓ Gradient accumulation using tree_map")
    logger.info("✓ Checkpoint saving batched and optimized")
    logger.info("✓ Removed excessive debug logging")
    logger.info("✓ Simplified JSON serialization")