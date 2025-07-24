"""Unit tests for data service domain logic.

These tests verify the pure business logic of the data service
without any external dependencies or framework-specific code.
"""

import pytest
from typing import Dict, List, Any, Optional, Tuple, Iterator
from dataclasses import dataclass

from domain.data.data_service import (
    DatasetInfo,
    DataService,
    BatchProcessor,
    CacheConfig,
    DataCache,
    DataPipeline,
    DatasetIterator,
    StreamingDataService,
    DataQualityReport
)
from domain.data.data_models import (
    DatasetType,
    TaskDataType,
    DataStatistics,
    DataConfig,
    DataSplit
)


class TestDatasetInfo:
    """Test dataset information handling."""
    
    def test_basic_dataset_info(self):
        """Test basic dataset information."""
        info = DatasetInfo(
            name="test_dataset",
            dataset_type=DatasetType.TRAIN,
            task_type=TaskDataType.BINARY_CLASSIFICATION,
            num_samples=1000,
            num_features=10
        )
        
        assert info.name == "test_dataset"
        assert info.dataset_type == DatasetType.TRAIN
        assert info.task_type == TaskDataType.BINARY_CLASSIFICATION
        assert info.num_samples == 1000
        assert info.num_features == 10
        assert info.is_tokenized is False
        assert info.is_cached is False
    
    def test_dataset_info_dict(self):
        """Test converting dataset info to dictionary."""
        info = DatasetInfo(
            name="test_dataset",
            dataset_type=DatasetType.VALIDATION,
            task_type=TaskDataType.MULTICLASS_CLASSIFICATION,
            num_samples=500,
            file_path="/data/test.csv",
            size_mb=10.5,
            is_tokenized=True,
            is_cached=True
        )
        
        info_dict = info.get_info_dict()
        
        assert info_dict["name"] == "test_dataset"
        assert info_dict["type"] == "validation"
        assert info_dict["task"] == "multiclass_classification"
        assert info_dict["num_samples"] == 500
        assert info_dict["file_path"] == "/data/test.csv"
        assert info_dict["size_mb"] == 10.5
        assert info_dict["is_tokenized"] is True
        assert info_dict["is_cached"] is True
    
    def test_dataset_info_with_statistics(self):
        """Test dataset info with statistics."""
        stats = DataStatistics(
            num_samples=1000,
            num_features=5,
            label_distribution={0: 600, 1: 400}
        )
        
        info = DatasetInfo(
            name="test_dataset",
            dataset_type=DatasetType.TRAIN,
            task_type=TaskDataType.BINARY_CLASSIFICATION,
            num_samples=1000,
            statistics=stats
        )
        
        info_dict = info.get_info_dict()
        assert "statistics" in info_dict
        assert info_dict["statistics"]["num_samples"] == 1000


class TestBatchProcessor:
    """Test batch processing logic."""
    
    class MockBatchProcessor(BatchProcessor[List[int]]):
        """Mock batch processor for testing."""
        
        def process_batch(self, batch: List[Any]) -> List[int]:
            """Process batch by doubling each value."""
            return [x * 2 for x in batch]
    
    def test_batch_processor_single_batch(self):
        """Test processing a single batch."""
        processor = self.MockBatchProcessor(batch_size=3)
        examples = [1, 2, 3]
        
        results = processor.process_dataset(examples)
        
        assert len(results) == 1
        assert results[0] == [2, 4, 6]
    
    def test_batch_processor_multiple_batches(self):
        """Test processing multiple batches."""
        processor = self.MockBatchProcessor(batch_size=3)
        examples = [1, 2, 3, 4, 5, 6, 7]
        
        results = processor.process_dataset(examples)
        
        assert len(results) == 3
        assert results[0] == [2, 4, 6]
        assert results[1] == [8, 10, 12]
        assert results[2] == [14]
    
    def test_batch_processor_empty_input(self):
        """Test processing empty input."""
        processor = self.MockBatchProcessor(batch_size=3)
        examples = []
        
        results = processor.process_dataset(examples)
        
        assert len(results) == 0


class TestCacheConfig:
    """Test cache configuration."""
    
    def test_default_cache_config(self):
        """Test default cache configuration."""
        config = CacheConfig()
        
        assert config.enable_cache is True
        assert config.cache_dir == ".cache"
        assert config.cache_format == "pickle"
        assert config.compression is None
        assert config.max_cache_size_gb is None
        assert config.cache_ttl_hours is None
    
    def test_cache_key_generation(self):
        """Test cache key generation."""
        config = CacheConfig()
        
        key = config.get_cache_key("dataset_name", "config_hash_123")
        assert key == "dataset_name_config_hash_123"
    
    def test_cache_path_generation(self):
        """Test cache path generation."""
        config = CacheConfig(cache_dir="/tmp/cache", cache_format="json")
        
        path = config.get_cache_path("test_key")
        assert path == "/tmp/cache/test_key.json"
    
    def test_cache_path_with_compression(self):
        """Test cache path with compression."""
        config = CacheConfig(
            cache_dir="/tmp/cache",
            cache_format="pickle",
            compression="gzip"
        )
        
        path = config.get_cache_path("test_key")
        assert path == "/tmp/cache/test_key.pickle.gzip"


class TestDataQualityReport:
    """Test data quality reporting."""
    
    def test_basic_quality_report(self):
        """Test basic quality report."""
        report = DataQualityReport(
            total_samples=1000,
            valid_samples=950,
            missing_labels=20,
            invalid_labels=10,
            empty_texts=15,
            duplicate_samples=5
        )
        
        assert report.total_samples == 1000
        assert report.valid_samples == 950
        assert report.error_rate == 0.05
    
    def test_empty_dataset_report(self):
        """Test report for empty dataset."""
        report = DataQualityReport(total_samples=0, valid_samples=0)
        
        assert report.error_rate == 0.0
    
    def test_add_recommendations(self):
        """Test adding recommendations."""
        report = DataQualityReport(total_samples=100, valid_samples=90)
        
        report.add_recommendation("Remove duplicate samples")
        report.add_recommendation("Handle missing labels")
        
        assert len(report.recommendations) == 2
        assert "Remove duplicate samples" in report.recommendations
    
    def test_report_to_dict(self):
        """Test converting report to dictionary."""
        report = DataQualityReport(
            total_samples=1000,
            valid_samples=900,
            missing_labels=50,
            invalid_labels=30,
            empty_texts=20,
            class_imbalance_ratio=3.5,
            rare_classes=[("class_x", 5), ("class_y", 3)]
        )
        
        report.add_recommendation("Balance dataset")
        
        report_dict = report.to_dict()
        
        assert report_dict["summary"]["total_samples"] == 1000
        assert report_dict["summary"]["valid_samples"] == 900
        assert report_dict["summary"]["error_rate"] == 0.1
        assert report_dict["issues"]["missing_labels"] == 50
        assert report_dict["issues"]["invalid_labels"] == 30
        assert report_dict["warnings"]["class_imbalance_ratio"] == 3.5
        assert len(report_dict["warnings"]["rare_classes"]) == 2
        assert report_dict["recommendations"] == ["Balance dataset"]


# Mock implementations for testing abstract classes
class MockDataset:
    """Mock dataset for testing."""
    
    def __init__(self, data: List[Dict[str, Any]]):
        self.data = data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]


class MockDataLoader:
    """Mock dataloader for testing."""
    
    def __init__(self, dataset: MockDataset, batch_size: int, shuffle: bool):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
    
    def __iter__(self):
        for i in range(0, len(self.dataset), self.batch_size):
            batch = []
            for j in range(min(self.batch_size, len(self.dataset) - i)):
                batch.append(self.dataset[i + j])
            yield batch


class MockDataService(DataService[MockDataset, MockDataLoader]):
    """Mock data service for testing."""
    
    def load_dataset(
        self,
        file_path: str,
        dataset_type: DatasetType = DatasetType.TRAIN
    ) -> MockDataset:
        """Load mock dataset."""
        # Simulate loading data based on file path
        if "train" in file_path:
            data = [{"text": f"Train sample {i}", "label": i % 2} for i in range(100)]
        elif "val" in file_path:
            data = [{"text": f"Val sample {i}", "label": i % 2} for i in range(20)]
        else:
            data = [{"text": f"Test sample {i}", "label": i % 2} for i in range(30)]
        
        dataset = MockDataset(data)
        self.datasets[dataset_type] = dataset
        return dataset
    
    def create_dataloader(
        self,
        dataset: MockDataset,
        batch_size: Optional[int] = None,
        shuffle: Optional[bool] = None
    ) -> MockDataLoader:
        """Create mock dataloader."""
        batch_size = batch_size or self.config.batch_size
        shuffle = shuffle if shuffle is not None else (
            self.config.shuffle_train if dataset == self.datasets.get(DatasetType.TRAIN)
            else False
        )
        return MockDataLoader(dataset, batch_size, shuffle)
    
    def tokenize_dataset(
        self,
        dataset: MockDataset,
        tokenizer: Any
    ) -> MockDataset:
        """Mock tokenization."""
        tokenized_data = []
        for sample in dataset.data:
            tokenized = {
                "input_ids": [1, 2, 3],  # Mock token IDs
                "attention_mask": [1, 1, 1],
                "label": sample["label"]
            }
            tokenized_data.append(tokenized)
        return MockDataset(tokenized_data)
    
    def split_dataset(
        self,
        dataset: MockDataset,
        split_config: DataSplit
    ) -> Tuple[MockDataset, MockDataset, MockDataset]:
        """Mock dataset splitting."""
        total = len(dataset)
        train_size = int(total * split_config.train_ratio)
        val_size = int(total * split_config.val_ratio)
        
        train_data = dataset.data[:train_size]
        val_data = dataset.data[train_size:train_size + val_size]
        test_data = dataset.data[train_size + val_size:]
        
        return (
            MockDataset(train_data),
            MockDataset(val_data),
            MockDataset(test_data)
        )


class MockDataCache(DataCache):
    """Mock data cache for testing."""
    
    def __init__(self, config: CacheConfig):
        super().__init__(config)
        self.cache_store = {}
    
    def exists(self, cache_key: str) -> bool:
        """Check if key exists in mock cache."""
        return cache_key in self.cache_store
    
    def load(self, cache_key: str) -> Any:
        """Load from mock cache."""
        return self.cache_store.get(cache_key)
    
    def save(self, cache_key: str, data: Any) -> None:
        """Save to mock cache."""
        self.cache_store[cache_key] = data
    
    def clear(self, cache_key: Optional[str] = None) -> None:
        """Clear mock cache."""
        if cache_key:
            self.cache_store.pop(cache_key, None)
        else:
            self.cache_store.clear()


class TestDataService:
    """Test abstract data service functionality."""
    
    def test_service_initialization(self):
        """Test data service initialization."""
        config = DataConfig(batch_size=16)
        service = MockDataService(config)
        
        assert service.config.batch_size == 16
        assert len(service.datasets) == 0
        assert len(service.statistics) == 0
    
    def test_load_dataset(self):
        """Test loading datasets."""
        config = DataConfig()
        service = MockDataService(config)
        
        train_dataset = service.load_dataset("train.csv", DatasetType.TRAIN)
        val_dataset = service.load_dataset("val.csv", DatasetType.VALIDATION)
        
        assert len(train_dataset) == 100
        assert len(val_dataset) == 20
        assert DatasetType.TRAIN in service.datasets
        assert DatasetType.VALIDATION in service.datasets
    
    def test_create_dataloader(self):
        """Test dataloader creation."""
        config = DataConfig(batch_size=10, shuffle_train=True)
        service = MockDataService(config)
        
        dataset = service.load_dataset("train.csv", DatasetType.TRAIN)
        dataloader = service.create_dataloader(dataset)
        
        assert dataloader.batch_size == 10
        assert dataloader.shuffle is True
        
        # Test with custom parameters
        dataloader2 = service.create_dataloader(dataset, batch_size=5, shuffle=False)
        assert dataloader2.batch_size == 5
        assert dataloader2.shuffle is False
    
    def test_tokenize_dataset(self):
        """Test dataset tokenization."""
        config = DataConfig()
        service = MockDataService(config)
        
        dataset = service.load_dataset("train.csv")
        tokenized = service.tokenize_dataset(dataset, tokenizer=None)
        
        assert len(tokenized) == len(dataset)
        assert "input_ids" in tokenized.data[0]
        assert "attention_mask" in tokenized.data[0]
    
    def test_split_dataset(self):
        """Test dataset splitting."""
        config = DataConfig()
        service = MockDataService(config)
        
        dataset = service.load_dataset("data.csv")
        split_config = DataSplit(train_ratio=0.7, val_ratio=0.15, test_ratio=0.15)
        
        train, val, test = service.split_dataset(dataset, split_config)
        
        assert len(train) == 70  # 70% of 100
        assert len(val) == 15    # 15% of 100
        assert len(test) == 15   # 15% of 100


class TestDataPipeline:
    """Test data pipeline functionality."""
    
    def test_pipeline_without_cache(self):
        """Test pipeline without caching."""
        config = DataConfig()
        service = MockDataService(config)
        pipeline = DataPipeline(service)
        
        # Add processing steps
        pipeline.add_step("tokenize", lambda d: service.tokenize_dataset(d, None))
        
        assert len(pipeline.pipeline_steps) == 1
        assert pipeline.pipeline_steps[0][0] == "tokenize"
    
    def test_pipeline_with_cache(self):
        """Test pipeline with caching."""
        config = DataConfig()
        service = MockDataService(config)
        cache_config = CacheConfig()
        cache = MockDataCache(cache_config)
        pipeline = DataPipeline(service, cache)
        
        assert pipeline.cache is not None
        assert isinstance(pipeline.cache, MockDataCache)


class MockStreamingDataService(StreamingDataService):
    """Mock streaming data service for testing."""
    
    def stream_from_file(
        self,
        file_path: str,
        chunk_size: int = 1000
    ) -> Iterator[List[Any]]:
        """Mock file streaming."""
        # Simulate streaming by yielding chunks
        total_samples = 5000
        for i in range(0, total_samples, chunk_size):
            chunk = []
            for j in range(min(chunk_size, total_samples - i)):
                chunk.append({"id": i + j, "text": f"Sample {i + j}"})
            yield chunk
    
    def stream_from_url(
        self,
        url: str,
        chunk_size: int = 1000
    ) -> Iterator[List[Any]]:
        """Mock URL streaming."""
        # Similar to file streaming for testing
        return self.stream_from_file(url, chunk_size)


class TestStreamingDataService:
    """Test streaming data service."""
    
    def test_stream_from_file(self):
        """Test streaming from file."""
        config = DataConfig()
        service = MockStreamingDataService(config)
        
        chunks = list(service.stream_from_file("data.csv", chunk_size=1000))
        
        assert len(chunks) == 5  # 5000 samples / 1000 chunk_size
        assert len(chunks[0]) == 1000
        assert len(chunks[-1]) == 1000
        assert chunks[0][0]["id"] == 0
        assert chunks[-1][-1]["id"] == 4999
    
    def test_stream_from_url(self):
        """Test streaming from URL."""
        config = DataConfig()
        service = MockStreamingDataService(config)
        
        chunks = list(service.stream_from_url("http://example.com/data", chunk_size=2000))
        
        assert len(chunks) == 3  # 5000 samples / 2000 chunk_size
        assert len(chunks[0]) == 2000
        assert len(chunks[-1]) == 1000
    
    def test_create_streaming_dataset_from_file(self):
        """Test creating streaming dataset from file."""
        config = DataConfig()
        service = MockStreamingDataService(config)
        
        dataset_iter = service.create_streaming_dataset("data.csv", is_url=False)
        
        # Consume first few items
        items = []
        for i, item in enumerate(dataset_iter):
            items.append(item)
            if i >= 10:
                break
        
        assert len(items) == 11
        assert items[0]["id"] == 0
        assert items[10]["id"] == 10
    
    def test_create_streaming_dataset_from_url(self):
        """Test creating streaming dataset from URL."""
        config = DataConfig()
        service = MockStreamingDataService(config)
        
        dataset_iter = service.create_streaming_dataset(
            "http://example.com/data",
            is_url=True
        )
        
        # Just verify it's iterable
        first_item = next(iter(dataset_iter))
        assert "id" in first_item
        assert "text" in first_item