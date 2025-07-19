"""
Pytest configuration and shared fixtures for data module tests.
"""

import pytest
import tempfile
import shutil
from pathlib import Path
from typing import Optional, Dict, Any, List, Iterator
import mlx.core as mx
import pandas as pd
import json

# Add project root to path
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from data.core.base import DatasetSpec
from data.core.metadata import CompetitionMetadata
from data.loaders.mlx_loader import MLXDataLoader, MLXLoaderConfig
from data.templates.engine import TextTemplateEngine


# Test configuration
@pytest.fixture(scope="session")
def test_config():
    """Shared test configuration."""
    return {
        "seed": 42,
        "batch_size": 4,
        "num_samples": 100,
        "num_features": 10,
        "sequence_length": 128,
        "vocab_size": 1000,
    }


# Temporary directories
@pytest.fixture
def tmp_data_dir():
    """Create temporary data directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def tmp_cache_dir():
    """Create temporary cache directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir) / "cache"


# Mock dataset implementations
class MockDataset:
    """Mock dataset for testing."""
    
    def __init__(
        self,
        num_samples: int = 100,
        num_features: int = 10,
        task_type: str = "classification",
        num_classes: int = 2,
    ):
        self.num_samples = num_samples
        self.num_features = num_features
        self.task_type = task_type
        self.num_classes = num_classes
        self._index = 0
    
    def __len__(self) -> int:
        return self.num_samples
    
    def __getitem__(self, idx: int) -> Dict[str, mx.array]:
        """Get item by index."""
        # Generate deterministic data based on index
        mx.random.seed(42 + idx)
        
        features = mx.random.normal((self.num_features,))
        
        if self.task_type == "classification":
            label = mx.array(idx % self.num_classes)
        else:  # regression
            label = mx.array(float(idx) / self.num_samples)
        
        return {
            "features": features,
            "labels": label,
            "index": mx.array(idx),
        }
    
    def __iter__(self):
        """Iterate through dataset."""
        self._index = 0
        return self
    
    def __next__(self):
        """Get next item."""
        if self._index >= self.num_samples:
            raise StopIteration
        item = self[self._index]
        self._index += 1
        return item


class MockTextDataset(MockDataset):
    """Mock text dataset for testing."""
    
    def __init__(
        self,
        num_samples: int = 100,
        vocab_size: int = 1000,
        sequence_length: int = 128,
        num_classes: int = 2,
    ):
        super().__init__(num_samples, num_features=sequence_length)
        self.vocab_size = vocab_size
        self.sequence_length = sequence_length
        self.num_classes = num_classes
    
    def __getitem__(self, idx: int) -> Dict[str, mx.array]:
        """Get text item by index."""
        mx.random.seed(42 + idx)
        
        # Generate random token IDs
        input_ids = mx.random.randint(0, self.vocab_size, (self.sequence_length,))
        attention_mask = mx.ones((self.sequence_length,))
        
        # Random padding
        padding_length = mx.random.randint(0, self.sequence_length // 2).item()
        if padding_length > 0:
            attention_mask[-padding_length:] = 0
            input_ids[-padding_length:] = 0
        
        label = mx.array(idx % self.num_classes)
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": label,
        }


# Data loader fixtures
@pytest.fixture
def mock_dataset():
    """Create mock dataset."""
    return MockDataset(num_samples=100, num_features=10)


@pytest.fixture
def mock_text_dataset():
    """Create mock text dataset."""
    return MockTextDataset(num_samples=100, vocab_size=1000)


@pytest.fixture
def mock_large_dataset():
    """Create large mock dataset for memory testing."""
    return MockDataset(num_samples=10000, num_features=100)


@pytest.fixture
def mock_dataloader():
    """Create mock data loader."""
    dataset = MockDataset(num_samples=100)
    config = MLXLoaderConfig(batch_size=4, shuffle=True)
    return MLXDataLoader(dataset, config)


# CSV data fixtures
@pytest.fixture
def create_csv_data():
    """Create CSV data for testing."""
    def _create(
        path: Path,
        num_rows: int = 100,
        num_features: int = 5,
        has_labels: bool = True,
    ):
        """Generate CSV file with synthetic data."""
        data = {
            f"feature_{i}": [float(j * i) for j in range(num_rows)]
            for i in range(num_features)
        }
        
        if has_labels:
            data["label"] = [j % 2 for j in range(num_rows)]
        
        df = pd.DataFrame(data)
        df.to_csv(path, index=False)
        return path
    
    return _create


@pytest.fixture
def create_json_data():
    """Create JSON data for testing."""
    def _create(
        path: Path,
        num_samples: int = 100,
        structure: str = "records",
    ):
        """Generate JSON file with synthetic data."""
        data = []
        for i in range(num_samples):
            record = {
                "id": i,
                "text": f"Sample text {i}",
                "label": i % 2,
                "features": [float(j) for j in range(5)],
            }
            data.append(record)
        
        if structure == "records":
            with open(path, "w") as f:
                json.dump(data, f)
        else:  # line-delimited
            with open(path, "w") as f:
                for record in data:
                    f.write(json.dumps(record) + "\n")
        
        return path
    
    return _create


# Configuration fixtures
@pytest.fixture
def loader_config():
    """Create standard loader configuration."""
    return MLXLoaderConfig(
        batch_size=4,
        shuffle=True,
        drop_last=False,
        num_workers=0,
        prefetch_size=2,
    )


@pytest.fixture
def streaming_config():
    """Create streaming loader configuration."""
    return {
        "batch_size": 32,
        "buffer_size": 1000,
        "prefetch": 4,
        "shuffle": True,
        "seed": 42,
    }


@pytest.fixture
def memory_config():
    """Create memory-optimized configuration."""
    return {
        "batch_size": 16,
        "memory_limit": 1_000_000_000,  # 1GB
        "cache_size": 100,
        "pin_memory": False,
    }


# Dataset spec fixtures
@pytest.fixture
def classification_spec():
    """Create classification dataset spec."""
    from data.core.base import CompetitionType
    return DatasetSpec(
        competition_name="test_classification",
        dataset_path="./test_data",
        competition_type=CompetitionType.BINARY_CLASSIFICATION,
        num_samples=1000,
        num_features=10,
        num_classes=2,
        target_column="label",
        numerical_columns=[f"feature_{i}" for i in range(10)],
    )


@pytest.fixture
def regression_spec():
    """Create regression dataset spec."""
    from data.core.base import CompetitionType
    return DatasetSpec(
        competition_name="test_regression",
        dataset_path="./test_data",
        competition_type=CompetitionType.REGRESSION,
        num_samples=1000,
        num_features=5,
        target_column="target",
        numerical_columns=[f"feature_{i}" for i in range(5)],
    )


# Template fixtures
@pytest.fixture
def template_engine():
    """Create template engine."""
    return TextTemplateEngine()


@pytest.fixture
def sample_tabular_data():
    """Create sample tabular data for template testing."""
    return pd.DataFrame({
        "age": [25, 30, 35, 40],
        "gender": ["M", "F", "M", "F"],
        "income": [50000, 60000, 70000, 80000],
        "label": [0, 1, 1, 0],
    })


# Utility fixtures
@pytest.fixture
def assert_batches_equal():
    """Utility for comparing batches."""
    def _assert(batch1: Dict[str, mx.array], batch2: Dict[str, mx.array]):
        """Assert two batches are equal."""
        assert set(batch1.keys()) == set(batch2.keys()), "Batch keys don't match"
        
        for key in batch1:
            assert mx.array_equal(batch1[key], batch2[key]), \
                f"Batch values for {key} don't match"
    
    return _assert


@pytest.fixture
def create_streaming_dataset():
    """Create streaming dataset."""
    class StreamingDataset:
        def __init__(self, num_samples: int = 1000):
            self.num_samples = num_samples
            self._position = 0
        
        def __iter__(self):
            self._position = 0
            return self
        
        def __next__(self):
            if self._position >= self.num_samples:
                raise StopIteration
            
            # Simulate streaming delay
            import time
            time.sleep(0.001)
            
            data = {
                "features": mx.random.normal((10,)),
                "label": mx.array(self._position % 2),
            }
            self._position += 1
            return data
    
    return StreamingDataset


# Memory profiling utilities
@pytest.fixture
def track_memory():
    """Track memory usage during test."""
    class MemoryTracker:
        def __init__(self):
            self.measurements = []
        
        def measure(self):
            """Take memory measurement."""
            import psutil
            import os
            process = psutil.Process(os.getpid())
            self.measurements.append(process.memory_info().rss)
        
        def get_peak_usage(self):
            """Get peak memory usage."""
            return max(self.measurements) if self.measurements else 0
        
        def get_delta(self):
            """Get memory usage delta."""
            if len(self.measurements) < 2:
                return 0
            return self.measurements[-1] - self.measurements[0]
    
    return MemoryTracker()


# Performance benchmarking
@pytest.fixture
def benchmark_loader():
    """Benchmark data loader performance."""
    def _benchmark(
        loader: DataLoader,
        num_epochs: int = 3,
    ) -> Dict[str, float]:
        """Benchmark loader performance."""
        import time
        
        times = []
        samples_processed = 0
        
        for epoch in range(num_epochs):
            start = time.time()
            for batch in loader:
                samples_processed += batch["features"].shape[0]
            times.append(time.time() - start)
        
        return {
            "mean_epoch_time": sum(times) / len(times),
            "total_samples": samples_processed,
            "samples_per_second": samples_processed / sum(times),
        }
    
    return _benchmark


# Error simulation fixtures
class CorruptDataset(MockDataset):
    """Dataset that produces corrupt data."""
    
    def __init__(self, corruption_rate: float = 0.1, **kwargs):
        super().__init__(**kwargs)
        self.corruption_rate = corruption_rate
    
    def __getitem__(self, idx: int) -> Dict[str, mx.array]:
        """Get item with possible corruption."""
        import random
        if random.random() < self.corruption_rate:
            raise ValueError(f"Corrupt data at index {idx}")
        return super().__getitem__(idx)


class SlowDataset(MockDataset):
    """Dataset with artificial delays."""
    
    def __init__(self, delay: float = 0.1, **kwargs):
        super().__init__(**kwargs)
        self.delay = delay
    
    def __getitem__(self, idx: int) -> Dict[str, mx.array]:
        """Get item with delay."""
        import time
        time.sleep(self.delay)
        return super().__getitem__(idx)


@pytest.fixture
def corrupt_dataset():
    """Create dataset with corrupt data."""
    return CorruptDataset(corruption_rate=0.2)


@pytest.fixture
def slow_dataset():
    """Create slow dataset."""
    return SlowDataset(delay=0.01)


# Kaggle fixtures
@pytest.fixture
def mock_kaggle_competition():
    """Mock Kaggle competition data."""
    return {
        "name": "titanic",
        "description": "Predict survival on the Titanic",
        "task_type": "binary_classification",
        "train_file": "train.csv",
        "test_file": "test.csv",
        "submission_file": "sample_submission.csv",
    }


@pytest.fixture
def create_kaggle_files(tmp_data_dir, create_csv_data):
    """Create mock Kaggle competition files."""
    def _create(competition_name: str = "titanic"):
        """Create competition files."""
        comp_dir = tmp_data_dir / competition_name
        comp_dir.mkdir(exist_ok=True)
        
        # Create train, test, and submission files
        train_path = create_csv_data(
            comp_dir / "train.csv",
            num_rows=100,
            num_features=5,
            has_labels=True,
        )
        
        test_path = create_csv_data(
            comp_dir / "test.csv",
            num_rows=50,
            num_features=5,
            has_labels=False,
        )
        
        # Create sample submission
        submission_df = pd.DataFrame({
            "PassengerId": range(50),
            "Survived": [0] * 50,
        })
        submission_df.to_csv(comp_dir / "sample_submission.csv", index=False)
        
        return comp_dir
    
    return _create


# Test markers
def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line("markers", "unit: Unit tests")
    config.addinivalue_line("markers", "integration: Integration tests")
    config.addinivalue_line("markers", "e2e: End-to-end tests")
    config.addinivalue_line("markers", "slow: Slow tests")
    config.addinivalue_line("markers", "streaming: Streaming data tests")
    config.addinivalue_line("markers", "kaggle: Kaggle-specific tests")
    config.addinivalue_line("markers", "memory: Memory-intensive tests")


# Pytest plugins
def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers based on test location."""
    for item in items:
        # Add markers based on test file location
        if "unit" in str(item.fspath):
            item.add_marker(pytest.mark.unit)
        elif "integration" in str(item.fspath):
            item.add_marker(pytest.mark.integration)
        elif "e2e" in str(item.fspath):
            item.add_marker(pytest.mark.e2e)