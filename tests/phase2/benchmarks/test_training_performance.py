"""Performance benchmarks for training speed improvements in Phase 2.

This module benchmarks the performance of various training configurations
to ensure Phase 2 optimizations maintain or improve performance.
"""

import gc
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock

import mlx.core as mx
import pytest

from training.core.base import BaseTrainer
from training.core.config import BaseTrainerConfig
from training.benchmarks.memory_tracker import MemoryTracker
from training.benchmarks.performance_monitor import PerformanceMonitor


@dataclass
class BenchmarkResult:
    """Result of a performance benchmark."""
    name: str
    duration_seconds: float
    samples_per_second: float
    memory_used_mb: float
    peak_memory_mb: float
    metrics: Dict[str, float]


class TrainingBenchmark:
    """Base class for training benchmarks."""
    
    def __init__(self, name: str):
        self.name = name
        self.results = []
        self.memory_tracker = MemoryTracker()
        self.performance_monitor = PerformanceMonitor()
    
    @contextmanager
    def benchmark(self, description: str):
        """Context manager for benchmarking a code block."""
        # Clear caches and collect garbage
        mx.clear_caches()
        gc.collect()
        
        # Start monitoring
        self.memory_tracker.start()
        start_time = time.perf_counter()
        start_memory = self.memory_tracker.get_current_memory()
        
        try:
            yield
        finally:
            # Stop monitoring
            end_time = time.perf_counter()
            end_memory = self.memory_tracker.get_current_memory()
            peak_memory = self.memory_tracker.get_peak_memory()
            
            # Calculate metrics
            duration = end_time - start_time
            memory_used = end_memory - start_memory
            
            result = BenchmarkResult(
                name=description,
                duration_seconds=duration,
                samples_per_second=0,  # Will be calculated by specific benchmarks
                memory_used_mb=memory_used / 1024 / 1024,
                peak_memory_mb=peak_memory / 1024 / 1024,
                metrics={}
            )
            
            self.results.append(result)
            self.memory_tracker.stop()
    
    def report(self) -> str:
        """Generate a benchmark report."""
        report_lines = [
            f"\n{'='*60}",
            f"Benchmark Report: {self.name}",
            f"{'='*60}\n"
        ]
        
        for result in self.results:
            report_lines.extend([
                f"Test: {result.name}",
                f"  Duration: {result.duration_seconds:.3f}s",
                f"  Throughput: {result.samples_per_second:.1f} samples/s",
                f"  Memory Used: {result.memory_used_mb:.1f} MB",
                f"  Peak Memory: {result.peak_memory_mb:.1f} MB",
            ])
            
            if result.metrics:
                report_lines.append("  Additional Metrics:")
                for key, value in result.metrics.items():
                    report_lines.append(f"    {key}: {value:.3f}")
            
            report_lines.append("")
        
        return "\n".join(report_lines)


def create_mock_model(input_dim: int = 768, output_dim: int = 2):
    """Create a mock model for benchmarking."""
    model = MagicMock()
    model.parameters.return_value = {
        "weight": mx.random.normal((input_dim, output_dim)),
        "bias": mx.zeros((output_dim,))
    }
    
    def forward(**kwargs):
        batch_size = kwargs.get("input_ids", mx.zeros((1, 1))).shape[0]
        # Simulate some computation
        x = mx.random.normal((batch_size, input_dim))
        logits = x @ model.parameters()["weight"] + model.parameters()["bias"]
        loss = mx.mean(mx.square(logits))
        return {"loss": loss, "logits": logits}
    
    model.__call__ = MagicMock(side_effect=forward)
    model.eval = MagicMock()
    model.train = MagicMock()
    
    return model


def create_mock_dataloader(num_samples: int, batch_size: int, seq_length: int = 128):
    """Create a mock dataloader for benchmarking."""
    num_batches = num_samples // batch_size
    batches = []
    
    for i in range(num_batches):
        batch = {
            "input_ids": mx.random.randint(0, 1000, (batch_size, seq_length)),
            "attention_mask": mx.ones((batch_size, seq_length)),
            "labels": mx.random.randint(0, 2, (batch_size,))
        }
        batches.append(batch)
    
    loader = MagicMock()
    loader.__iter__.return_value = iter(batches)
    loader.__len__.return_value = len(batches)
    
    return loader


class TestTrainingPerformance:
    """Test training performance improvements."""
    
    @pytest.fixture
    def base_config(self, tmp_path):
        """Create base configuration for benchmarks."""
        config = BaseTrainerConfig()
        config.environment.output_dir = tmp_path / "benchmark"
        config.training.num_epochs = 1
        config.training.logging_steps = 1000  # Reduce logging overhead
        config.training.save_strategy = "no"  # No checkpointing
        config.training.eval_strategy = "no"  # No evaluation
        config.optimizer.learning_rate = 0.001
        return config
    
    def test_baseline_training_speed(self, base_config):
        """Benchmark baseline training speed."""
        benchmark = TrainingBenchmark("Baseline Training")
        
        # Test different batch sizes
        batch_sizes = [8, 16, 32, 64]
        num_samples = 1024
        
        for batch_size in batch_sizes:
            model = create_mock_model()
            dataloader = create_mock_dataloader(num_samples, batch_size)
            
            config = base_config.copy()
            config.data.batch_size = batch_size
            
            trainer = BaseTrainer(model, config)
            
            with benchmark.benchmark(f"Batch size {batch_size}"):
                result = trainer.train(dataloader)
            
            # Calculate throughput
            duration = benchmark.results[-1].duration_seconds
            samples_per_second = num_samples / duration
            benchmark.results[-1].samples_per_second = samples_per_second
            benchmark.results[-1].metrics["final_loss"] = float(result.final_train_loss)
        
        print(benchmark.report())
        
        # Verify performance expectations
        for result in benchmark.results:
            assert result.samples_per_second > 100  # At least 100 samples/s
            assert result.memory_used_mb < 1000  # Less than 1GB memory usage
    
    def test_compiled_vs_uncompiled(self, base_config):
        """Benchmark compiled vs uncompiled training."""
        benchmark = TrainingBenchmark("Compilation Comparison")
        
        batch_size = 32
        num_samples = 512
        
        # Test uncompiled
        model = create_mock_model()
        dataloader = create_mock_dataloader(num_samples, batch_size)
        
        config = base_config.copy()
        config.training.use_compilation = False
        
        trainer = BaseTrainer(model, config)
        
        with benchmark.benchmark("Uncompiled"):
            result = trainer.train(dataloader)
        
        uncompiled_duration = benchmark.results[-1].duration_seconds
        benchmark.results[-1].samples_per_second = num_samples / uncompiled_duration
        
        # Test compiled
        model = create_mock_model()
        dataloader = create_mock_dataloader(num_samples, batch_size)
        
        config = base_config.copy()
        config.training.use_compilation = True
        
        trainer = BaseTrainer(model, config)
        
        with benchmark.benchmark("Compiled"):
            result = trainer.train(dataloader)
        
        compiled_duration = benchmark.results[-1].duration_seconds
        benchmark.results[-1].samples_per_second = num_samples / compiled_duration
        
        print(benchmark.report())
        
        # Compilation should provide some speedup (or at least not slow down)
        speedup = uncompiled_duration / compiled_duration
        assert speedup >= 0.95  # Allow 5% regression due to compilation overhead
    
    def test_gradient_accumulation_performance(self, base_config):
        """Benchmark gradient accumulation impact."""
        benchmark = TrainingBenchmark("Gradient Accumulation")
        
        batch_size = 16
        num_samples = 512
        accumulation_steps = [1, 2, 4, 8]
        
        for acc_steps in accumulation_steps:
            model = create_mock_model()
            dataloader = create_mock_dataloader(num_samples, batch_size)
            
            config = base_config.copy()
            config.training.gradient_accumulation_steps = acc_steps
            
            trainer = BaseTrainer(model, config)
            
            with benchmark.benchmark(f"Accumulation steps: {acc_steps}"):
                result = trainer.train(dataloader)
            
            # Calculate effective batch size and throughput
            effective_batch_size = batch_size * acc_steps
            duration = benchmark.results[-1].duration_seconds
            benchmark.results[-1].samples_per_second = num_samples / duration
            benchmark.results[-1].metrics["effective_batch_size"] = effective_batch_size
        
        print(benchmark.report())
        
        # Verify gradient accumulation doesn't cause significant slowdown
        base_throughput = benchmark.results[0].samples_per_second
        for result in benchmark.results[1:]:
            # Allow up to 20% throughput reduction for larger accumulation
            assert result.samples_per_second >= base_throughput * 0.8
    
    def test_mixed_precision_performance(self, base_config):
        """Benchmark mixed precision training performance."""
        benchmark = TrainingBenchmark("Mixed Precision")
        
        batch_size = 32
        num_samples = 512
        
        # Test without mixed precision
        model = create_mock_model()
        dataloader = create_mock_dataloader(num_samples, batch_size)
        
        config = base_config.copy()
        config.training.mixed_precision = False
        
        trainer = BaseTrainer(model, config)
        
        with benchmark.benchmark("FP32"):
            result = trainer.train(dataloader)
        
        fp32_duration = benchmark.results[-1].duration_seconds
        benchmark.results[-1].samples_per_second = num_samples / fp32_duration
        
        # Test with mixed precision
        model = create_mock_model()
        dataloader = create_mock_dataloader(num_samples, batch_size)
        
        config = base_config.copy()
        config.training.mixed_precision = True
        
        trainer = BaseTrainer(model, config)
        
        with benchmark.benchmark("Mixed Precision (BF16)"):
            result = trainer.train(dataloader)
        
        mixed_duration = benchmark.results[-1].duration_seconds
        benchmark.results[-1].samples_per_second = num_samples / mixed_duration
        
        print(benchmark.report())
        
        # Mixed precision should provide speedup or at least not slow down
        speedup = fp32_duration / mixed_duration
        assert speedup >= 0.95
        
        # Memory usage should be lower with mixed precision
        fp32_memory = benchmark.results[0].peak_memory_mb
        mixed_memory = benchmark.results[1].peak_memory_mb
        assert mixed_memory <= fp32_memory * 1.1  # Allow 10% variance
    
    def test_prefetching_performance(self, base_config):
        """Benchmark data prefetching impact."""
        benchmark = TrainingBenchmark("Data Prefetching")
        
        batch_size = 32
        num_samples = 1024
        prefetch_sizes = [0, 2, 4, 8]
        
        for prefetch_size in prefetch_sizes:
            model = create_mock_model()
            
            # Create dataloader with prefetching
            dataloader = create_mock_dataloader(num_samples, batch_size)
            if prefetch_size > 0:
                # Simulate prefetching by pre-generating batches
                dataloader._prefetch_size = prefetch_size
            
            config = base_config.copy()
            config.data.prefetch_size = prefetch_size
            
            trainer = BaseTrainer(model, config)
            
            with benchmark.benchmark(f"Prefetch size: {prefetch_size}"):
                result = trainer.train(dataloader)
            
            duration = benchmark.results[-1].duration_seconds
            benchmark.results[-1].samples_per_second = num_samples / duration
        
        print(benchmark.report())
        
        # Prefetching should improve throughput
        no_prefetch_throughput = benchmark.results[0].samples_per_second
        for i, result in enumerate(benchmark.results[1:], 1):
            # Each level of prefetching should maintain or improve performance
            assert result.samples_per_second >= no_prefetch_throughput * 0.95
    
    def test_callback_overhead(self, base_config):
        """Benchmark callback system overhead."""
        benchmark = TrainingBenchmark("Callback Overhead")
        
        batch_size = 32
        num_samples = 512
        
        # Test without callbacks
        model = create_mock_model()
        dataloader = create_mock_dataloader(num_samples, batch_size)
        
        trainer = BaseTrainer(model, base_config, callbacks=[])
        
        with benchmark.benchmark("No callbacks"):
            result = trainer.train(dataloader)
        
        no_callback_duration = benchmark.results[-1].duration_seconds
        benchmark.results[-1].samples_per_second = num_samples / no_callback_duration
        
        # Test with multiple callbacks
        from training.callbacks.metrics import MetricsCallback
        from training.callbacks.progress import ProgressCallback
        from training.callbacks.early_stopping import EarlyStoppingCallback
        
        callbacks = [
            MetricsCallback(),
            ProgressCallback(),
            EarlyStoppingCallback(patience=10, metric="loss")
        ]
        
        model = create_mock_model()
        dataloader = create_mock_dataloader(num_samples, batch_size)
        
        trainer = BaseTrainer(model, base_config, callbacks=callbacks)
        
        with benchmark.benchmark("Multiple callbacks"):
            result = trainer.train(dataloader)
        
        callback_duration = benchmark.results[-1].duration_seconds
        benchmark.results[-1].samples_per_second = num_samples / callback_duration
        
        print(benchmark.report())
        
        # Callbacks should have minimal overhead (< 10%)
        overhead = (callback_duration - no_callback_duration) / no_callback_duration
        assert overhead < 0.1
    
    def test_scaling_performance(self, base_config):
        """Test how performance scales with model and data size."""
        benchmark = TrainingBenchmark("Scaling Analysis")
        
        # Test different model sizes
        model_sizes = [(256, "Small"), (768, "Medium"), (1024, "Large")]
        batch_size = 16
        num_samples = 256
        
        for hidden_dim, size_name in model_sizes:
            model = create_mock_model(input_dim=hidden_dim)
            dataloader = create_mock_dataloader(num_samples, batch_size, seq_length=128)
            
            trainer = BaseTrainer(model, base_config)
            
            with benchmark.benchmark(f"Model size: {size_name} ({hidden_dim}d)"):
                result = trainer.train(dataloader)
            
            duration = benchmark.results[-1].duration_seconds
            benchmark.results[-1].samples_per_second = num_samples / duration
            benchmark.results[-1].metrics["model_params"] = hidden_dim * 2  # Simplified
        
        print(benchmark.report())
        
        # Verify reasonable scaling
        # Throughput should decrease with model size, but not catastrophically
        throughputs = [r.samples_per_second for r in benchmark.results]
        for i in range(len(throughputs) - 1):
            # Each size increase should reduce throughput by less than 50%
            assert throughputs[i+1] >= throughputs[i] * 0.5