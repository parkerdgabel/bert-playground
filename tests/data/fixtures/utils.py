"""Utility fixtures for data module testing."""

from typing import Dict, Any, List, Optional, Tuple, Iterator
from pathlib import Path
import mlx.core as mx
import pandas as pd
import numpy as np
import time
import json
import tempfile
from contextlib import contextmanager


def create_sample_dataframe(
    num_rows: int = 100,
    num_numeric: int = 5,
    num_categorical: int = 3,
    num_text: int = 1,
    add_target: bool = True,
    task_type: str = "classification",
    size: Optional[int] = None,
) -> pd.DataFrame:
    """Create a sample DataFrame for testing."""
    # Use size parameter if provided, otherwise use num_rows
    actual_rows = size if size is not None else num_rows
    
    np.random.seed(42)
    data = {}
    
    # Add numeric features
    for i in range(num_numeric):
        data[f"numeric_{i}"] = np.random.randn(actual_rows)
        
    # Add categorical features
    for i in range(num_categorical):
        data[f"categorical_{i}"] = np.random.choice(['A', 'B', 'C', 'D'], size=actual_rows)
        
    # Add text features
    for i in range(num_text):
        data[f"text_{i}"] = [f"Sample text {j}" for j in range(actual_rows)]
        
    # Add target
    if add_target:
        if task_type == "classification":
            data["target"] = np.random.randint(0, 2, size=actual_rows)
        else:  # regression
            data["target"] = np.random.randn(actual_rows)
            
    return pd.DataFrame(data)


def check_dataset_consistency(dataset: Any) -> bool:
    """Check if dataset is consistent."""
    try:
        # Check basic properties
        assert hasattr(dataset, "__len__")
        assert hasattr(dataset, "__getitem__")
        
        # Check length
        length = len(dataset)
        assert length >= 0
        
        # Check if we can get first item
        if length > 0:
            item = dataset[0]
            assert isinstance(item, dict)
            
        return True
    except Exception as e:
        print(f"Dataset consistency check failed: {e}")
        return False


def compare_batches(
    batch1: Dict[str, mx.array],
    batch2: Dict[str, mx.array],
    rtol: float = 1e-5,
    atol: float = 1e-8,
) -> Tuple[bool, Dict[str, Any]]:
    """Compare two batches for equality."""
    if set(batch1.keys()) != set(batch2.keys()):
        return False, {"error": "Different keys", "keys1": list(batch1.keys()), "keys2": list(batch2.keys())}
    
    differences = {}
    all_equal = True
    
    for key in batch1:
        arr1 = batch1[key]
        arr2 = batch2[key]
        
        if arr1.shape != arr2.shape:
            all_equal = False
            differences[key] = f"Shape mismatch: {arr1.shape} vs {arr2.shape}"
        elif not mx.allclose(arr1, arr2, rtol=rtol, atol=atol):
            all_equal = False
            max_diff = float(mx.max(mx.abs(arr1 - arr2)))
            differences[key] = f"Max difference: {max_diff}"
    
    return all_equal, differences


def create_mock_csv_data(
    path: Path,
    num_rows: int = 100,
    num_numeric: int = 5,
    num_categorical: int = 3,
    num_text: int = 1,
    has_target: bool = True,
    target_type: str = "binary",
    seed: int = 42,
) -> pd.DataFrame:
    """Create mock CSV data for testing."""
    np.random.seed(seed)
    
    data = {}
    
    # Numeric columns
    for i in range(num_numeric):
        data[f"num_feature_{i}"] = np.random.randn(num_rows)
    
    # Categorical columns
    for i in range(num_categorical):
        categories = [f"cat_{j}" for j in range(np.random.randint(2, 8))]
        data[f"cat_feature_{i}"] = np.random.choice(categories, num_rows)
    
    # Text columns
    text_samples = [
        "This is a positive review",
        "Negative sentiment here",
        "Neutral text sample",
        "Another example text",
        "Random text data",
    ]
    for i in range(num_text):
        data[f"text_feature_{i}"] = np.random.choice(text_samples, num_rows)
    
    # Target column
    if has_target:
        if target_type == "binary":
            data["target"] = np.random.randint(0, 2, num_rows)
        elif target_type == "multiclass":
            data["target"] = np.random.randint(0, 5, num_rows)
        else:  # regression
            data["target"] = np.random.randn(num_rows)
    
    df = pd.DataFrame(data)
    df.to_csv(path, index=False)
    
    return df


def create_mock_json_data(
    path: Path,
    num_samples: int = 100,
    format: str = "records",
    seed: int = 42,
) -> List[Dict[str, Any]]:
    """Create mock JSON data for testing."""
    np.random.seed(seed)
    
    data = []
    for i in range(num_samples):
        record = {
            "id": i,
            "text": f"Sample text {i} with some content",
            "score": float(np.random.randn()),
            "category": np.random.choice(["A", "B", "C"]),
            "features": [float(x) for x in np.random.randn(5)],
            "label": int(i % 2),
        }
        data.append(record)
    
    if format == "records":
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
    elif format == "lines":
        with open(path, "w") as f:
            for record in data:
                f.write(json.dumps(record) + "\n")
    
    return data


@contextmanager
def temporary_data_files(
    num_files: int = 3,
    file_type: str = "csv",
    **kwargs
) -> Iterator[List[Path]]:
    """Context manager for creating temporary data files."""
    temp_dir = Path(tempfile.mkdtemp())
    files = []
    
    try:
        for i in range(num_files):
            if file_type == "csv":
                path = temp_dir / f"data_{i}.csv"
                create_mock_csv_data(path, **kwargs)
            elif file_type == "json":
                path = temp_dir / f"data_{i}.json"
                create_mock_json_data(path, **kwargs)
            else:
                raise ValueError(f"Unknown file type: {file_type}")
            
            files.append(path)
        
        yield files
        
    finally:
        # Cleanup
        import shutil
        shutil.rmtree(temp_dir)


def profile_dataloader_performance(
    dataloader: Any,
    num_epochs: int = 3,
    warmup_epochs: int = 1,
) -> Dict[str, float]:
    """Profile data loader performance."""
    epoch_times = []
    batch_times = []
    samples_processed = 0
    
    for epoch in range(num_epochs + warmup_epochs):
        epoch_start = time.time()
        epoch_batches = 0
        
        for batch in dataloader:
            batch_start = time.time()
            
            # Simulate processing
            if "features" in batch:
                batch_size = batch["features"].shape[0]
            elif "input_ids" in batch:
                batch_size = batch["input_ids"].shape[0]
            else:
                batch_size = 1
            
            if epoch >= warmup_epochs:  # Skip warmup
                batch_times.append(time.time() - batch_start)
                samples_processed += batch_size
            
            epoch_batches += 1
        
        if epoch >= warmup_epochs:  # Skip warmup
            epoch_times.append(time.time() - epoch_start)
    
    return {
        "mean_epoch_time": np.mean(epoch_times),
        "std_epoch_time": np.std(epoch_times),
        "mean_batch_time": np.mean(batch_times),
        "std_batch_time": np.std(batch_times),
        "batches_per_epoch": epoch_batches,
        "total_samples": samples_processed,
        "samples_per_second": samples_processed / sum(epoch_times),
    }


def check_data_consistency(
    dataloader: Any,
    num_epochs: int = 2,
    check_deterministic: bool = True,
) -> Dict[str, Any]:
    """Check data loader consistency across epochs."""
    results = {
        "deterministic": True,
        "consistent_batches": True,
        "consistent_order": True,
        "batch_sizes": [],
        "epoch_samples": [],
    }
    
    first_epoch_data = []
    
    for epoch in range(num_epochs):
        epoch_data = []
        epoch_sample_count = 0
        
        for batch in dataloader:
            # Extract batch info
            if "features" in batch:
                batch_size = batch["features"].shape[0]
                batch_id = mx.mean(batch["features"]).item()  # Simple hash
            elif "input_ids" in batch:
                batch_size = batch["input_ids"].shape[0]
                batch_id = mx.mean(mx.array(batch["input_ids"], dtype=mx.float32)).item()
            else:
                batch_size = 1
                batch_id = 0
            
            epoch_data.append((batch_size, batch_id))
            epoch_sample_count += batch_size
            
            if epoch == 0:
                results["batch_sizes"].append(batch_size)
        
        results["epoch_samples"].append(epoch_sample_count)
        
        if epoch == 0:
            first_epoch_data = epoch_data
        elif check_deterministic:
            # Check if data is the same across epochs
            if len(epoch_data) != len(first_epoch_data):
                results["consistent_batches"] = False
            else:
                for i, (size1, id1) in enumerate(first_epoch_data):
                    size2, id2 = epoch_data[i]
                    if size1 != size2:
                        results["consistent_batches"] = False
                    if abs(id1 - id2) > 1e-6:  # Allow small floating point differences
                        results["consistent_order"] = False
    
    results["deterministic"] = results["consistent_batches"] and results["consistent_order"]
    
    return results


def simulate_memory_pressure(
    dataloader: Any,
    memory_limit_mb: int = 1000,
    num_iterations: int = 100,
) -> Dict[str, Any]:
    """Simulate memory pressure while using dataloader."""
    import psutil
    import os
    
    process = psutil.Process(os.getpid())
    initial_memory = process.memory_info().rss / 1024 / 1024  # MB
    
    # Allocate dummy data to increase memory pressure
    dummy_data = []
    target_memory = initial_memory + memory_limit_mb
    
    results = {
        "initial_memory_mb": initial_memory,
        "target_memory_mb": target_memory,
        "iterations_completed": 0,
        "max_memory_mb": initial_memory,
        "memory_exceeded": False,
    }
    
    try:
        for i in range(num_iterations):
            # Process batch
            batch = next(iter(dataloader))
            
            # Allocate more memory
            dummy_data.append(mx.random.normal((1000, 1000)))
            
            # Check memory
            current_memory = process.memory_info().rss / 1024 / 1024
            results["max_memory_mb"] = max(results["max_memory_mb"], current_memory)
            
            if current_memory > target_memory:
                results["memory_exceeded"] = True
                break
            
            results["iterations_completed"] = i + 1
            
    except Exception as e:
        results["error"] = str(e)
    
    # Cleanup
    del dummy_data
    mx.metal.clear_cache()
    
    return results


def validate_dataset_split(
    train_data: Any,
    val_data: Any,
    test_data: Optional[Any] = None,
    check_overlap: bool = True,
    check_distribution: bool = True,
) -> Dict[str, Any]:
    """Validate dataset splits."""
    results = {
        "train_size": len(train_data) if hasattr(train_data, "__len__") else None,
        "val_size": len(val_data) if hasattr(val_data, "__len__") else None,
        "test_size": len(test_data) if test_data and hasattr(test_data, "__len__") else None,
        "no_overlap": True,
        "balanced_distribution": True,
    }
    
    if check_overlap:
        # Simple check - compare first few samples
        try:
            train_samples = [train_data[i] for i in range(min(10, len(train_data)))]
            val_samples = [val_data[i] for i in range(min(10, len(val_data)))]
            
            # Compare hashes of samples
            train_hashes = set()
            for sample in train_samples:
                if isinstance(sample, dict) and "features" in sample:
                    hash_val = float(mx.sum(sample["features"]))
                    train_hashes.add(hash_val)
            
            for sample in val_samples:
                if isinstance(sample, dict) and "features" in sample:
                    hash_val = float(mx.sum(sample["features"]))
                    if hash_val in train_hashes:
                        results["no_overlap"] = False
                        break
        except:
            results["no_overlap"] = None  # Could not check
    
    if check_distribution:
        # Check label distribution
        try:
            train_labels = []
            val_labels = []
            
            # Collect some labels
            for i in range(min(100, len(train_data))):
                sample = train_data[i]
                if isinstance(sample, dict) and "label" in sample:
                    train_labels.append(int(sample["label"]))
            
            for i in range(min(100, len(val_data))):
                sample = val_data[i]
                if isinstance(sample, dict) and "label" in sample:
                    val_labels.append(int(sample["label"]))
            
            if train_labels and val_labels:
                # Check if distributions are roughly similar
                train_dist = np.bincount(train_labels)
                val_dist = np.bincount(val_labels)
                
                # Normalize
                train_dist = train_dist / train_dist.sum()
                val_dist = val_dist / val_dist.sum()
                
                # Check if distributions are similar (within 20%)
                max_diff = np.max(np.abs(train_dist - val_dist))
                results["balanced_distribution"] = max_diff < 0.2
                results["max_distribution_diff"] = float(max_diff)
        except:
            results["balanced_distribution"] = None  # Could not check
    
    return results


def create_data_corruption_test_cases() -> Dict[str, Any]:
    """Create test cases for data corruption handling."""
    return {
        "missing_values": {
            "features": mx.array([[1.0, float("nan"), 3.0], [4.0, 5.0, float("nan")]]),
            "labels": mx.array([0, 1]),
        },
        "infinite_values": {
            "features": mx.array([[1.0, float("inf"), 3.0], [4.0, 5.0, float("-inf")]]),
            "labels": mx.array([0, 1]),
        },
        "wrong_dimensions": {
            "features": mx.array([1.0, 2.0, 3.0]),  # 1D instead of 2D
            "labels": mx.array([0, 1]),
        },
        "mismatched_sizes": {
            "features": mx.array([[1.0, 2.0], [3.0, 4.0]]),
            "labels": mx.array([0, 1, 2]),  # 3 labels for 2 samples
        },
        "empty_batch": {
            "features": mx.array([]).reshape(0, 5),
            "labels": mx.array([]),
        },
        "single_sample": {
            "features": mx.array([[1.0, 2.0, 3.0]]),
            "labels": mx.array([0]),
        },
    }


def create_test_row(
    num_numeric: int = 5,
    num_categorical: int = 3, 
    num_text: int = 1,
    add_label: bool = True,
    seed: Optional[int] = None,
) -> Dict[str, Any]:
    """Create a single test row of data."""
    if seed is not None:
        np.random.seed(seed)
    
    row = {}
    
    # Add numeric features
    for i in range(num_numeric):
        row[f"numeric_{i}"] = np.random.randn()
    
    # Add categorical features
    for i in range(num_categorical):
        row[f"categorical_{i}"] = np.random.choice(['A', 'B', 'C', 'D'])
    
    # Add text features
    text_samples = [
        "This is a positive example",
        "Negative sentiment detected",
        "Neutral text content",
        "Another sample text",
    ]
    for i in range(num_text):
        row[f"text_{i}"] = np.random.choice(text_samples)
    
    # Add label
    if add_label:
        row["label"] = np.random.randint(0, 2)
    
    return row


def check_memory_usage(
    func: Any,
    *args,
    **kwargs
) -> Dict[str, float]:
    """Check memory usage of a function."""
    import psutil
    import os
    import gc
    
    # Force garbage collection
    gc.collect()
    
    process = psutil.Process(os.getpid())
    
    # Get initial memory
    initial_memory = process.memory_info().rss / 1024 / 1024  # MB
    
    # Run function
    start_time = time.time()
    result = func(*args, **kwargs)
    end_time = time.time()
    
    # Get final memory
    final_memory = process.memory_info().rss / 1024 / 1024  # MB
    
    # Force garbage collection again
    gc.collect()
    
    # Get memory after cleanup
    cleaned_memory = process.memory_info().rss / 1024 / 1024  # MB
    
    return {
        "initial_memory_mb": initial_memory,
        "peak_memory_mb": final_memory,
        "final_memory_mb": cleaned_memory,
        "memory_used_mb": final_memory - initial_memory,
        "memory_leaked_mb": cleaned_memory - initial_memory,
        "execution_time": end_time - start_time,
        "result": result,
    }


def measure_throughput(
    dataloader: Any,
    num_batches: int = 100,
    warmup_batches: int = 10,
) -> Dict[str, float]:
    """Measure throughput of a dataloader."""
    # Warmup
    for i, batch in enumerate(dataloader):
        if i >= warmup_batches:
            break
    
    # Measure
    batch_times = []
    samples_processed = 0
    bytes_processed = 0
    
    start_time = time.time()
    
    for i, batch in enumerate(dataloader):
        if i >= num_batches:
            break
        
        batch_start = time.time()
        
        # Count samples
        if isinstance(batch, dict):
            if "features" in batch:
                batch_size = batch["features"].shape[0]
            elif "input_ids" in batch:
                batch_size = batch["input_ids"].shape[0]
            else:
                batch_size = 1
        else:
            batch_size = 1
        
        samples_processed += batch_size
        
        # Estimate bytes (rough estimate)
        if isinstance(batch, dict):
            for key, value in batch.items():
                if hasattr(value, "nbytes"):
                    bytes_processed += value.nbytes
                elif hasattr(value, "shape"):
                    # Estimate for MLX arrays
                    bytes_processed += np.prod(value.shape) * 4  # Assume float32
        
        batch_times.append(time.time() - batch_start)
    
    total_time = time.time() - start_time
    
    return {
        "total_time": total_time,
        "num_batches": len(batch_times),
        "samples_processed": samples_processed,
        "bytes_processed": bytes_processed,
        "mean_batch_time": np.mean(batch_times),
        "std_batch_time": np.std(batch_times),
        "samples_per_second": samples_processed / total_time,
        "mb_per_second": (bytes_processed / 1024 / 1024) / total_time,
        "batches_per_second": len(batch_times) / total_time,
    }


def create_large_tensor(
    shape: Tuple[int, ...],
    dtype: mx.Dtype = mx.float32,
    fill_value: Optional[float] = None,
) -> mx.array:
    """Create a large tensor for memory testing."""
    if fill_value is not None:
        return mx.full(shape, fill_value, dtype=dtype)
    else:
        return mx.random.normal(shape, dtype=dtype)


def measure_memory_allocation_time(
    allocation_fn: Any,
    num_iterations: int = 10,
    warmup_iterations: int = 2,
) -> Dict[str, float]:
    """Measure time taken for memory allocations."""
    import gc
    
    # Warmup
    for _ in range(warmup_iterations):
        result = allocation_fn()
        del result
        gc.collect()
    
    # Measure
    times = []
    for _ in range(num_iterations):
        gc.collect()
        start = time.time()
        result = allocation_fn()
        allocation_time = time.time() - start
        times.append(allocation_time)
        del result
    
    return {
        "mean_time": np.mean(times),
        "std_time": np.std(times),
        "min_time": np.min(times),
        "max_time": np.max(times),
        "total_time": sum(times),
    }


def simulate_slow_consumer(delay: float = 0.01):
    """Simulate a slow data consumer."""
    def consume(item):
        time.sleep(delay)
        return item
    return consume


def assert_batch_structure(
    batch: Dict[str, mx.array],
    expected_keys: List[str],
    batch_size: Optional[int] = None,
) -> None:
    """Assert that a batch has the expected structure."""
    # Check keys
    assert set(batch.keys()) == set(expected_keys), \
        f"Batch keys {set(batch.keys())} don't match expected {set(expected_keys)}"
    
    # Check batch size if provided
    if batch_size is not None:
        for key, value in batch.items():
            if hasattr(value, "shape") and len(value.shape) > 0:
                assert value.shape[0] == batch_size, \
                    f"Batch size mismatch for {key}: expected {batch_size}, got {value.shape[0]}"
    
    # Check that all arrays have the same batch dimension
    batch_sizes = []
    for key, value in batch.items():
        if hasattr(value, "shape") and len(value.shape) > 0:
            batch_sizes.append(value.shape[0])
    
    if batch_sizes:
        assert all(bs == batch_sizes[0] for bs in batch_sizes), \
            f"Inconsistent batch sizes: {batch_sizes}"


def benchmark_data_pipeline(
    create_pipeline_fn: Any,
    pipeline_configs: List[Dict[str, Any]],
    num_iterations: int = 100,
) -> Dict[str, Dict[str, float]]:
    """Benchmark different data pipeline configurations."""
    results = {}
    
    for config in pipeline_configs:
        config_name = config.get("name", str(config))
        pipeline = create_pipeline_fn(**config)
        
        # Warmup
        for _ in range(10):
            _ = next(iter(pipeline))
        
        # Benchmark
        times = []
        for _ in range(num_iterations):
            start = time.time()
            _ = next(iter(pipeline))
            times.append(time.time() - start)
        
        results[config_name] = {
            "mean_time": np.mean(times),
            "std_time": np.std(times),
            "min_time": np.min(times),
            "max_time": np.max(times),
            "throughput": 1.0 / np.mean(times),
        }
    
    return results