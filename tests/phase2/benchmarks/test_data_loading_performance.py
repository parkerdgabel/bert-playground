"""Performance benchmarks for data loading improvements in Phase 2.

This module benchmarks data loading, augmentation, and preprocessing
performance to ensure optimizations meet performance targets.
"""

import csv
import gc
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Optional
from unittest.mock import MagicMock

import mlx.core as mx
import pytest

from data.factory import create_dataset, create_dataloader
from data.augmentation.tabular import TabularAugmenter
from data.augmentation.config import AugmentationConfig, AugmentationMode
from data.loaders.mlx_loader import MLXDataLoader
from data.preprocessing.tokenizer_cache import TokenizerCache


class DataLoadingBenchmark:
    """Benchmark suite for data loading operations."""
    
    def __init__(self):
        self.results = {}
    
    def time_operation(self, name: str, operation, *args, **kwargs):
        """Time an operation and store results."""
        # Warm up
        try:
            operation(*args, **kwargs)
        except:
            pass
        
        # Clear caches
        mx.clear_caches()
        gc.collect()
        
        # Time the operation
        start_time = time.perf_counter()
        result = operation(*args, **kwargs)
        end_time = time.perf_counter()
        
        duration = end_time - start_time
        self.results[name] = {
            "duration": duration,
            "result": result
        }
        
        return result, duration


def create_test_csv(path: Path, num_rows: int, num_cols: int = 5):
    """Create a test CSV file for benchmarking."""
    with open(path, 'w', newline='') as f:
        writer = csv.writer(f)
        
        # Header
        headers = [f"feature_{i}" for i in range(num_cols - 1)] + ["label"]
        writer.writerow(headers)
        
        # Data rows
        for i in range(num_rows):
            row = [f"value_{i}_{j}" for j in range(num_cols - 1)] + [str(i % 2)]
            writer.writerow(row)


class TestDataLoadingPerformance:
    """Test data loading performance improvements."""
    
    def test_csv_loading_speed(self, tmp_path):
        """Benchmark CSV loading speed."""
        benchmark = DataLoadingBenchmark()
        
        # Test different file sizes
        file_sizes = [1000, 10000, 100000]
        
        for size in file_sizes:
            csv_path = tmp_path / f"test_{size}.csv"
            create_test_csv(csv_path, size)
            
            def load_csv():
                return create_dataset(str(csv_path))
            
            dataset, duration = benchmark.time_operation(
                f"CSV loading ({size} rows)",
                load_csv
            )
            
            rows_per_second = size / duration
            assert rows_per_second > 1000  # At least 1k rows/second
            assert dataset is not None
            assert len(dataset) == size
        
        print("\nCSV Loading Performance:")
        for name, result in benchmark.results.items():
            print(f"  {name}: {result['duration']:.3f}s")
    
    def test_dataloader_batching_speed(self, tmp_path):
        """Benchmark dataloader batching performance."""
        benchmark = DataLoadingBenchmark()
        
        # Create test dataset
        csv_path = tmp_path / "test_batching.csv"
        create_test_csv(csv_path, 10000)
        dataset = create_dataset(str(csv_path))
        
        # Test different batch sizes
        batch_sizes = [8, 16, 32, 64, 128]
        
        for batch_size in batch_sizes:
            def create_loader():
                return create_dataloader(
                    dataset,
                    batch_size=batch_size,
                    shuffle=False
                )
            
            loader, duration = benchmark.time_operation(
                f"Batching (batch_size={batch_size})",
                create_loader
            )
            
            # Test iteration speed
            def iterate_loader():
                batches = list(loader)
                return len(batches)
            
            num_batches, iter_duration = benchmark.time_operation(
                f"Iteration (batch_size={batch_size})",
                iterate_loader
            )
            
            samples_per_second = len(dataset) / iter_duration
            assert samples_per_second > 1000  # At least 1k samples/second
        
        print("\nDataLoader Batching Performance:")
        for name, result in benchmark.results.items():
            print(f"  {name}: {result['duration']:.3f}s")
    
    def test_augmentation_speed(self, tmp_path):
        """Benchmark data augmentation performance."""
        benchmark = DataLoadingBenchmark()
        
        # Create test data
        csv_path = tmp_path / "test_augmentation.csv"
        create_test_csv(csv_path, 1000)
        
        # Test different augmentation modes
        augmentation_modes = [
            AugmentationMode.NONE,
            AugmentationMode.LIGHT,
            AugmentationMode.MODERATE,
            AugmentationMode.AGGRESSIVE
        ]
        
        for mode in augmentation_modes:
            config = AugmentationConfig.from_mode(mode)
            augmenter = TabularAugmenter(config)
            
            def apply_augmentation():
                # Simulate augmenting a batch
                sample_batch = [
                    {"feature_0": "test", "feature_1": "data", "label": 0}
                    for _ in range(32)
                ]
                
                augmented = []
                for sample in sample_batch:
                    augmented.append(augmenter.augment(sample))
                
                return augmented
            
            result, duration = benchmark.time_operation(
                f"Augmentation ({mode.value})",
                apply_augmentation
            )
            
            samples_per_second = 32 / duration
            assert samples_per_second > 100  # At least 100 samples/second
        
        print("\nAugmentation Performance:")
        for name, result in benchmark.results.items():
            print(f"  {name}: {result['duration']:.3f}s")
    
    def test_tokenizer_cache_performance(self, tmp_path):
        """Benchmark tokenizer caching performance."""
        benchmark = DataLoadingBenchmark()
        
        cache_dir = tmp_path / "tokenizer_cache"
        cache = TokenizerCache(cache_dir)
        
        # Test data
        texts = [f"This is sample text number {i} for tokenization." for i in range(1000)]
        
        # Test without cache
        def tokenize_without_cache():
            results = []
            for text in texts:
                # Simulate tokenization
                tokens = text.split()
                results.append(tokens)
            return results
        
        result1, duration1 = benchmark.time_operation(
            "Tokenization without cache",
            tokenize_without_cache
        )
        
        # Test with cache (first time - cache miss)
        def tokenize_with_cache_miss():
            results = []
            for i, text in enumerate(texts):
                cache_key = f"text_{i}"
                if cache.exists(cache_key):
                    tokens = cache.load(cache_key)
                else:
                    tokens = text.split()  # Simulate tokenization
                    cache.save(cache_key, tokens)
                results.append(tokens)
            return results
        
        result2, duration2 = benchmark.time_operation(
            "Tokenization with cache (miss)",
            tokenize_with_cache_miss
        )
        
        # Test with cache (second time - cache hit)
        def tokenize_with_cache_hit():
            results = []
            for i, text in enumerate(texts):
                cache_key = f"text_{i}"
                tokens = cache.load(cache_key)
                results.append(tokens)
            return results
        
        result3, duration3 = benchmark.time_operation(
            "Tokenization with cache (hit)",
            tokenize_with_cache_hit
        )
        
        # Cache hits should be much faster
        speedup = duration1 / duration3
        assert speedup > 5  # At least 5x speedup with cache
        
        print("\nTokenizer Cache Performance:")
        for name, result in benchmark.results.items():
            print(f"  {name}: {result['duration']:.3f}s")
    
    def test_mlx_array_operations(self):
        """Benchmark MLX array operations for data processing."""
        benchmark = DataLoadingBenchmark()
        
        # Test array creation
        def create_arrays():
            arrays = []
            for _ in range(1000):
                arr = mx.random.normal((32, 128))
                arrays.append(arr)
            return arrays
        
        arrays, duration1 = benchmark.time_operation(
            "Array creation",
            create_arrays
        )
        
        # Test array concatenation
        def concatenate_arrays():
            return mx.concatenate(arrays[:100], axis=0)
        
        result, duration2 = benchmark.time_operation(
            "Array concatenation",
            concatenate_arrays
        )
        
        # Test array slicing
        def slice_arrays():
            results = []
            for arr in arrays[:100]:
                sliced = arr[:16]  # Take first 16 samples
                results.append(sliced)
            return results
        
        sliced, duration3 = benchmark.time_operation(
            "Array slicing",
            slice_arrays
        )
        
        print("\nMLX Array Operations Performance:")
        for name, result in benchmark.results.items():
            print(f"  {name}: {result['duration']:.3f}s")
    
    def test_prefetch_performance(self, tmp_path):
        """Benchmark prefetching performance."""
        benchmark = DataLoadingBenchmark()
        
        # Create test dataset
        csv_path = tmp_path / "test_prefetch.csv"
        create_test_csv(csv_path, 5000)
        dataset = create_dataset(str(csv_path))
        
        # Test without prefetching
        def iterate_without_prefetch():
            loader = create_dataloader(
                dataset,
                batch_size=32,
                prefetch_size=0
            )
            batches = list(loader)
            return len(batches)
        
        num_batches1, duration1 = benchmark.time_operation(
            "Without prefetching",
            iterate_without_prefetch
        )
        
        # Test with prefetching
        def iterate_with_prefetch():
            loader = create_dataloader(
                dataset,
                batch_size=32,
                prefetch_size=4
            )
            batches = list(loader)
            return len(batches)
        
        num_batches2, duration2 = benchmark.time_operation(
            "With prefetching",
            iterate_with_prefetch
        )
        
        # Prefetching should not slow down significantly
        slowdown = duration2 / duration1
        assert slowdown < 1.2  # Less than 20% slower
        
        print("\nPrefetching Performance:")
        for name, result in benchmark.results.items():
            print(f"  {name}: {result['duration']:.3f}s")
    
    def test_memory_efficiency(self, tmp_path):
        """Test memory efficiency of data loading."""
        benchmark = DataLoadingBenchmark()
        
        # Create large test dataset
        csv_path = tmp_path / "test_memory.csv"
        create_test_csv(csv_path, 50000, num_cols=20)
        
        def load_and_process():
            dataset = create_dataset(str(csv_path))
            loader = create_dataloader(dataset, batch_size=64)
            
            # Process first few batches
            processed = 0
            for i, batch in enumerate(loader):
                if i >= 10:  # Only process first 10 batches
                    break
                processed += len(batch)
            
            return processed
        
        processed, duration = benchmark.time_operation(
            "Memory efficient loading",
            load_and_process
        )
        
        assert processed > 0
        
        # Test memory usage doesn't grow excessively
        # This is a basic check - in practice, you'd use memory profiling tools
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()
        memory_mb = memory_info.rss / 1024 / 1024
        
        # Memory usage should be reasonable (less than 1GB for this test)
        assert memory_mb < 1000
        
        print(f"\nMemory Usage: {memory_mb:.1f} MB")
        print(f"Processing Speed: {processed / duration:.1f} samples/s")
    
    def test_concurrent_loading(self, tmp_path):
        """Test concurrent data loading performance."""
        import threading
        
        benchmark = DataLoadingBenchmark()
        
        # Create multiple test files
        file_paths = []
        for i in range(4):
            csv_path = tmp_path / f"test_concurrent_{i}.csv"
            create_test_csv(csv_path, 2500)
            file_paths.append(csv_path)
        
        def load_sequential():
            datasets = []
            for path in file_paths:
                dataset = create_dataset(str(path))
                datasets.append(dataset)
            return datasets
        
        datasets1, duration1 = benchmark.time_operation(
            "Sequential loading",
            load_sequential
        )
        
        def load_concurrent():
            datasets = [None] * len(file_paths)
            threads = []
            
            def load_file(index, path):
                datasets[index] = create_dataset(str(path))
            
            for i, path in enumerate(file_paths):
                thread = threading.Thread(target=load_file, args=(i, path))
                threads.append(thread)
                thread.start()
            
            for thread in threads:
                thread.join()
            
            return datasets
        
        datasets2, duration2 = benchmark.time_operation(
            "Concurrent loading",
            load_concurrent
        )
        
        # Concurrent loading should be faster
        speedup = duration1 / duration2
        assert speedup >= 1.0  # At least no slower
        
        print(f"\nConcurrent Loading Speedup: {speedup:.2f}x")
        print("\nConcurrent Loading Performance:")
        for name, result in benchmark.results.items():
            print(f"  {name}: {result['duration']:.3f}s")