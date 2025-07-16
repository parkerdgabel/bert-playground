#!/usr/bin/env python3
"""Benchmark MLX optimizations.

This script benchmarks the performance improvements from MLX optimizations:
- Baseline vs optimized training
- Memory usage comparison
- Throughput analysis
- Quantization impact
"""

import argparse
import json
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import mlx.core as mx
import mlx.optimizers as optim
import numpy as np
import pandas as pd
from loguru import logger
from rich.console import Console
from rich.table import Table
from transformers import AutoTokenizer

from data.unified_loader import create_optimized_dataloaders
from data.mlx_enhanced_loader import create_enhanced_dataloaders
from models.modernbert_cnn_hybrid import CNNEnhancedModernBERT, ModernBERTConfig
from models.quantization_utils import (
    create_default_quantization_config,
    estimate_model_size,
    quantize_model_for_inference,
)
from training.trainer_v2 import Trainer, TrainingConfig
from training.mlx_optimized_trainer import MLXOptimizedTrainer, OptimizedTrainingConfig
from utils.memory_profiler import MemoryProfiler


@dataclass
class BenchmarkResult:
    """Results from a benchmark run."""
    
    name: str
    config: Dict
    
    # Timing metrics
    total_time: float
    steps_per_second: float
    samples_per_second: float
    time_per_step: float
    
    # Memory metrics
    peak_memory_mb: float
    avg_memory_mb: float
    memory_efficiency: float  # Samples per MB
    
    # Model metrics
    final_loss: float
    final_accuracy: float
    best_validation_accuracy: float
    
    # Optimization metrics
    model_size_mb: float
    compression_ratio: float


class MLXBenchmarkSuite:
    """Comprehensive benchmark suite for MLX optimizations."""
    
    def __init__(
        self,
        output_dir: str = "output/benchmarks",
        num_steps: int = 100,
        batch_sizes: List[int] = [16, 32, 64],
        enable_memory_profiling: bool = True,
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.num_steps = num_steps
        self.batch_sizes = batch_sizes
        self.enable_memory_profiling = enable_memory_profiling
        
        self.results: List[BenchmarkResult] = []
        self.console = Console()
        
        # Initialize components
        self.tokenizer = None
        self.config = None
        
    def setup(self):
        """Setup benchmark environment."""
        logger.info("Setting up benchmark environment...")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained("answerdotai/ModernBERT-base")
        
        # Create model config
        self.config = ModernBERTConfig(
            vocab_size=self.tokenizer.vocab_size,
            hidden_size=768,
            num_hidden_layers=12,
            num_attention_heads=12,
            intermediate_size=3072,
            max_position_embeddings=8192,
            num_labels=2,
            pad_token_id=self.tokenizer.pad_token_id,
            cnn_num_filters=[128, 256, 512],
            cnn_filter_sizes=[3, 5, 7],
            cnn_dropout=0.3,
        )
        
        logger.info("Benchmark environment ready")
    
    def run_baseline_benchmark(self, batch_size: int) -> BenchmarkResult:
        """Run baseline benchmark without optimizations."""
        logger.info(f"Running baseline benchmark with batch_size={batch_size}")
        
        # Create standard data loaders
        train_loader, val_loader, _ = create_optimized_dataloaders(
            tokenizer=self.tokenizer,
            batch_size=batch_size,
            num_threads=4,  # Baseline threads
            prefetch_size=1,  # Minimal prefetch
            pre_tokenize=False,  # No pre-tokenization
            use_mlx_data=False,  # Basic pipeline
        )
        
        # Create model and optimizer
        model = CNNEnhancedModernBERT(self.config)
        optimizer = optim.AdamW(learning_rate=2e-5)
        
        # Create trainer with basic config
        config = TrainingConfig(
            learning_rate=2e-5,
            num_epochs=1,
            batch_size=batch_size,
            gradient_clip=1.0,
            warmup_ratio=0.1,
            weight_decay=0.01,
            save_steps=self.num_steps + 1,  # Don't save during benchmark
            eval_steps=self.num_steps + 1,  # Don't eval during benchmark
        )
        
        trainer = Trainer(model, optimizer)
        trainer.save_config(config)
        
        # Memory profiler
        memory_profiler = None
        if self.enable_memory_profiling:
            memory_profiler = MemoryProfiler(
                log_interval=10,
                save_plots=False,
            )
        
        # Benchmark training
        start_time = time.time()
        
        step_times = []
        losses = []
        accuracies = []
        
        for step, batch in enumerate(train_loader.get_dataloader()()):
            if step >= self.num_steps:
                break
            
            step_start = time.time()
            
            loss, metrics = trainer.train_step(batch)
            mx.eval(loss)  # Force evaluation
            
            step_time = time.time() - step_start
            step_times.append(step_time)
            losses.append(float(loss))
            accuracies.append(metrics.get("accuracy", 0))
            
            if memory_profiler and step % 10 == 0:
                memory_profiler.record(step, "baseline_train")
        
        total_time = time.time() - start_time
        
        # Get memory summary
        memory_summary = {}
        if memory_profiler:
            memory_summary = memory_profiler.get_memory_summary()
        
        # Calculate metrics
        avg_step_time = np.mean(step_times)
        steps_per_second = 1.0 / avg_step_time
        samples_per_second = steps_per_second * batch_size
        
        # Model size
        model_size_info = estimate_model_size(model)
        
        result = BenchmarkResult(
            name=f"baseline_batch{batch_size}",
            config={
                "batch_size": batch_size,
                "optimizations": "none",
                "threads": 4,
                "prefetch": 1,
            },
            total_time=total_time,
            steps_per_second=steps_per_second,
            samples_per_second=samples_per_second,
            time_per_step=avg_step_time,
            peak_memory_mb=memory_summary.get("peak_rss_mb", 0),
            avg_memory_mb=memory_summary.get("current_rss_mb", 0),
            memory_efficiency=samples_per_second / max(1, memory_summary.get("current_rss_mb", 1)),
            final_loss=np.mean(losses[-10:]) if losses else 0,
            final_accuracy=np.mean(accuracies[-10:]) if accuracies else 0,
            best_validation_accuracy=0,  # Not evaluated in benchmark
            model_size_mb=model_size_info["original_size_mb"],
            compression_ratio=1.0,
        )
        
        logger.info(
            f"Baseline complete: {samples_per_second:.1f} samples/sec, "
            f"Memory: {result.peak_memory_mb:.1f} MB"
        )
        
        return result
    
    def run_optimized_benchmark(self, batch_size: int) -> BenchmarkResult:
        """Run benchmark with all optimizations."""
        logger.info(f"Running optimized benchmark with batch_size={batch_size}")
        
        # Create enhanced data loaders
        train_loader, val_loader, _ = create_enhanced_dataloaders(
            tokenizer=self.tokenizer,
            batch_size=batch_size,
            num_threads=8,  # Optimized threads
            prefetch_size=4,  # Better prefetch
            enable_augmentation=False,  # Disable for fair comparison
            memory_map=True,  # Memory mapping
            double_buffer=True,  # Double buffering
        )
        
        # Create model and optimizer
        model = CNNEnhancedModernBERT(self.config)
        optimizer = optim.AdamW(learning_rate=2e-5)
        
        # Create optimized trainer config
        config = OptimizedTrainingConfig(
            learning_rate=2e-5,
            num_epochs=1,
            base_batch_size=batch_size,
            max_batch_size=batch_size * 2,  # Allow dynamic sizing
            gradient_accumulation_steps=2,  # Effective larger batch
            eval_batch_size=batch_size * 2,
            lazy_eval_interval=10,
            num_workers=8,
            prefetch_size=4,
            save_steps=self.num_steps + 1,
            eval_steps=self.num_steps + 1,
            enable_profiling=False,
        )
        
        trainer = MLXOptimizedTrainer(model, optimizer, config)
        
        # Memory profiler
        memory_profiler = None
        if self.enable_memory_profiling:
            memory_profiler = MemoryProfiler(
                log_interval=10,
                save_plots=False,
            )
        
        # Benchmark training
        start_time = time.time()
        
        step_times = []
        losses = []
        effective_batch_sizes = []
        
        for step, batch in enumerate(train_loader.get_dataloader()()):
            if step >= self.num_steps:
                break
            
            step_start = time.time()
            
            loss, metrics = trainer.train_step_lazy(batch)
            
            # Force eval periodically
            if step % 10 == 0:
                mx.eval(loss)
            
            step_time = time.time() - step_start
            
            if "loss" in metrics:  # Only count actual updates
                step_times.append(step_time)
                losses.append(metrics["loss"])
                effective_batch_sizes.append(trainer.current_batch_size)
            
            if memory_profiler and step % 10 == 0:
                memory_profiler.record(step, "optimized_train")
        
        total_time = time.time() - start_time
        
        # Get memory summary
        memory_summary = {}
        if memory_profiler:
            memory_summary = memory_profiler.get_memory_summary()
        
        # Calculate metrics
        avg_step_time = np.mean(step_times) if step_times else 1.0
        steps_per_second = 1.0 / avg_step_time
        avg_batch_size = np.mean(effective_batch_sizes) if effective_batch_sizes else batch_size
        samples_per_second = steps_per_second * avg_batch_size
        
        # Model size
        model_size_info = estimate_model_size(model)
        
        result = BenchmarkResult(
            name=f"optimized_batch{batch_size}",
            config={
                "batch_size": batch_size,
                "avg_effective_batch": avg_batch_size,
                "optimizations": "all",
                "threads": 8,
                "prefetch": 4,
                "grad_accum": 2,
                "lazy_eval": True,
                "double_buffer": True,
            },
            total_time=total_time,
            steps_per_second=steps_per_second,
            samples_per_second=samples_per_second,
            time_per_step=avg_step_time,
            peak_memory_mb=memory_summary.get("peak_rss_mb", 0),
            avg_memory_mb=memory_summary.get("current_rss_mb", 0),
            memory_efficiency=samples_per_second / max(1, memory_summary.get("current_rss_mb", 1)),
            final_loss=np.mean(losses[-10:]) if losses else 0,
            final_accuracy=0,  # Not tracked in optimized version
            best_validation_accuracy=0,
            model_size_mb=model_size_info["original_size_mb"],
            compression_ratio=1.0,
        )
        
        logger.info(
            f"Optimized complete: {samples_per_second:.1f} samples/sec, "
            f"Memory: {result.peak_memory_mb:.1f} MB"
        )
        
        return result
    
    def run_quantization_benchmark(self, batch_size: int, bits: int = 4) -> BenchmarkResult:
        """Run benchmark with quantized model."""
        logger.info(f"Running quantization benchmark with batch_size={batch_size}, bits={bits}")
        
        # Create data loader
        train_loader, _, _ = create_enhanced_dataloaders(
            tokenizer=self.tokenizer,
            batch_size=batch_size,
            num_threads=8,
            prefetch_size=4,
        )
        
        # Create and quantize model
        model = CNNEnhancedModernBERT(self.config)
        
        # Quantize model
        quant_config = create_default_quantization_config("bert", bits=bits)
        model = quantize_model_for_inference(model, quant_config)
        
        # Get model size info
        model_size_info = estimate_model_size(model, quant_config)
        
        # Benchmark inference
        start_time = time.time()
        
        step_times = []
        
        for step, batch in enumerate(train_loader.get_dataloader()()):
            if step >= self.num_steps:
                break
            
            step_start = time.time()
            
            # Forward pass only (inference)
            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
            )
            mx.eval(outputs["logits"])
            
            step_time = time.time() - step_start
            step_times.append(step_time)
        
        total_time = time.time() - start_time
        
        # Calculate metrics
        avg_step_time = np.mean(step_times)
        steps_per_second = 1.0 / avg_step_time
        samples_per_second = steps_per_second * batch_size
        
        result = BenchmarkResult(
            name=f"quantized_{bits}bit_batch{batch_size}",
            config={
                "batch_size": batch_size,
                "quantization": f"{bits}bit",
                "inference_only": True,
            },
            total_time=total_time,
            steps_per_second=steps_per_second,
            samples_per_second=samples_per_second,
            time_per_step=avg_step_time,
            peak_memory_mb=0,  # Not tracked for inference
            avg_memory_mb=0,
            memory_efficiency=0,
            final_loss=0,
            final_accuracy=0,
            best_validation_accuracy=0,
            model_size_mb=model_size_info["quantized_size_mb"],
            compression_ratio=model_size_info["compression_ratio"],
        )
        
        logger.info(
            f"Quantized complete: {samples_per_second:.1f} samples/sec, "
            f"Model size: {result.model_size_mb:.1f} MB, "
            f"Compression: {result.compression_ratio:.1f}x"
        )
        
        return result
    
    def run_all_benchmarks(self):
        """Run all benchmarks."""
        self.setup()
        
        # Run benchmarks for each batch size
        for batch_size in self.batch_sizes:
            # Baseline
            result = self.run_baseline_benchmark(batch_size)
            self.results.append(result)
            
            # Optimized
            result = self.run_optimized_benchmark(batch_size)
            self.results.append(result)
            
            # Quantized
            for bits in [4, 8]:
                result = self.run_quantization_benchmark(batch_size, bits)
                self.results.append(result)
        
        # Save results
        self.save_results()
        self.generate_report()
        self.plot_results()
    
    def save_results(self):
        """Save benchmark results to file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save as JSON
        results_dict = [asdict(r) for r in self.results]
        json_path = self.output_dir / f"benchmark_results_{timestamp}.json"
        
        with open(json_path, "w") as f:
            json.dump(results_dict, f, indent=2)
        
        # Save as CSV
        df = pd.DataFrame(results_dict)
        csv_path = self.output_dir / f"benchmark_results_{timestamp}.csv"
        df.to_csv(csv_path, index=False)
        
        logger.info(f"Results saved to {json_path} and {csv_path}")
    
    def generate_report(self):
        """Generate benchmark report."""
        # Create comparison table
        table = Table(title="MLX Optimization Benchmark Results")
        
        table.add_column("Configuration", style="cyan")
        table.add_column("Samples/sec", style="green")
        table.add_column("Speedup", style="yellow")
        table.add_column("Memory (MB)", style="blue")
        table.add_column("Model Size (MB)", style="magenta")
        
        # Find baseline for comparison
        baseline_results = {}
        for result in self.results:
            if "baseline" in result.name:
                batch_size = result.config["batch_size"]
                baseline_results[batch_size] = result
        
        # Add rows
        for result in self.results:
            batch_size = result.config["batch_size"]
            baseline = baseline_results.get(batch_size)
            
            speedup = "1.0x"
            if baseline and baseline != result:
                speedup = f"{result.samples_per_second / baseline.samples_per_second:.2f}x"
            
            table.add_row(
                result.name,
                f"{result.samples_per_second:.1f}",
                speedup,
                f"{result.peak_memory_mb:.1f}",
                f"{result.model_size_mb:.1f}",
            )
        
        self.console.print(table)
        
        # Print summary
        self.console.print("\n[bold]Summary:[/bold]")
        
        # Calculate average improvements
        optimized_speedups = []
        for result in self.results:
            if "optimized" in result.name:
                batch_size = result.config["batch_size"]
                baseline = baseline_results.get(batch_size)
                if baseline:
                    speedup = result.samples_per_second / baseline.samples_per_second
                    optimized_speedups.append(speedup)
        
        if optimized_speedups:
            avg_speedup = np.mean(optimized_speedups)
            self.console.print(
                f"• Average optimized speedup: [green]{avg_speedup:.2f}x[/green]"
            )
        
        # Quantization compression
        quant_compressions = [r.compression_ratio for r in self.results if "quantized" in r.name]
        if quant_compressions:
            avg_compression = np.mean(quant_compressions)
            self.console.print(
                f"• Average quantization compression: [blue]{avg_compression:.1f}x[/blue]"
            )
    
    def plot_results(self):
        """Plot benchmark results."""
        # Prepare data
        df = pd.DataFrame([asdict(r) for r in self.results])
        
        # Create plots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # 1. Throughput comparison
        for batch_size in self.batch_sizes:
            batch_df = df[df["config"].apply(lambda x: x.get("batch_size") == batch_size)]
            ax1.bar(batch_df["name"], batch_df["samples_per_second"])
        ax1.set_title("Throughput Comparison")
        ax1.set_ylabel("Samples/second")
        ax1.tick_params(axis="x", rotation=45)
        
        # 2. Memory usage
        memory_df = df[df["peak_memory_mb"] > 0]
        if not memory_df.empty:
            ax2.bar(memory_df["name"], memory_df["peak_memory_mb"])
            ax2.set_title("Peak Memory Usage")
            ax2.set_ylabel("Memory (MB)")
            ax2.tick_params(axis="x", rotation=45)
        
        # 3. Model size comparison
        ax3.bar(df["name"], df["model_size_mb"])
        ax3.set_title("Model Size")
        ax3.set_ylabel("Size (MB)")
        ax3.tick_params(axis="x", rotation=45)
        
        # 4. Speedup by batch size
        speedups = []
        batch_sizes = []
        for batch_size in self.batch_sizes:
            baseline = df[(df["name"] == f"baseline_batch{batch_size}")]["samples_per_second"].values
            optimized = df[(df["name"] == f"optimized_batch{batch_size}")]["samples_per_second"].values
            
            if len(baseline) > 0 and len(optimized) > 0:
                speedup = optimized[0] / baseline[0]
                speedups.append(speedup)
                batch_sizes.append(batch_size)
        
        if speedups:
            ax4.plot(batch_sizes, speedups, "o-", linewidth=2, markersize=8)
            ax4.set_title("Optimization Speedup by Batch Size")
            ax4.set_xlabel("Batch Size")
            ax4.set_ylabel("Speedup")
            ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plot_path = self.output_dir / f"benchmark_plots_{timestamp}.png"
        plt.savefig(plot_path, dpi=150, bbox_inches="tight")
        plt.close()
        
        logger.info(f"Plots saved to {plot_path}")


def main():
    """Main benchmark function."""
    parser = argparse.ArgumentParser(description="Benchmark MLX optimizations")
    
    parser.add_argument(
        "--output_dir",
        type=str,
        default="output/benchmarks",
        help="Output directory for results",
    )
    parser.add_argument(
        "--num_steps",
        type=int,
        default=100,
        help="Number of steps per benchmark",
    )
    parser.add_argument(
        "--batch_sizes",
        type=int,
        nargs="+",
        default=[16, 32, 64],
        help="Batch sizes to test",
    )
    parser.add_argument(
        "--enable_memory_profiling",
        action="store_true",
        help="Enable memory profiling",
    )
    
    args = parser.parse_args()
    
    # Setup logging
    logger.add(
        Path(args.output_dir) / "benchmark.log",
        rotation="10 MB",
        level="INFO",
    )
    
    # Run benchmarks
    suite = MLXBenchmarkSuite(
        output_dir=args.output_dir,
        num_steps=args.num_steps,
        batch_sizes=args.batch_sizes,
        enable_memory_profiling=args.enable_memory_profiling,
    )
    
    suite.run_all_benchmarks()
    
    logger.success("Benchmarking complete!")


if __name__ == "__main__":
    main()