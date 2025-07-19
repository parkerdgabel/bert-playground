"""Unit tests for core base classes."""

import pytest
from pathlib import Path
from unittest.mock import Mock, patch

import pandas as pd
import mlx.core as mx

from data.core.base import (
    CompetitionType,
    DatasetSpec,
    KaggleDataset,
)


class TestCompetitionType:
    """Test CompetitionType enum."""
    
    def test_competition_types_exist(self):
        """Test all expected competition types exist."""
        expected_types = [
            "BINARY_CLASSIFICATION",
            "MULTICLASS_CLASSIFICATION", 
            "MULTILABEL_CLASSIFICATION",
            "REGRESSION",
            "ORDINAL_REGRESSION",
            "TIME_SERIES",
            "RANKING",
            "STRUCTURED_PREDICTION",
            "GENERATIVE",
            "UNKNOWN"
        ]
        
        for type_name in expected_types:
            assert hasattr(CompetitionType, type_name)
            
    def test_competition_type_values(self):
        """Test competition type string values."""
        assert CompetitionType.BINARY_CLASSIFICATION.value == "binary_classification"
        assert CompetitionType.MULTICLASS_CLASSIFICATION.value == "multiclass_classification"
        assert CompetitionType.REGRESSION.value == "regression"
        assert CompetitionType.UNKNOWN.value == "unknown"


class TestDatasetSpec:
    """Test DatasetSpec dataclass."""
    
    def test_basic_creation(self):
        """Test basic DatasetSpec creation."""
        spec = DatasetSpec(
            competition_name="test",
            dataset_path=Path("/tmp/test"),
            competition_type=CompetitionType.BINARY_CLASSIFICATION,
            num_samples=100,
            num_features=10,
        )
        
        assert spec.competition_name == "test"
        assert spec.dataset_path == Path("/tmp/test")
        assert spec.competition_type == CompetitionType.BINARY_CLASSIFICATION
        assert spec.num_samples == 100
        assert spec.num_features == 10
        
    def test_with_target_column(self):
        """Test DatasetSpec with target column."""
        spec = DatasetSpec(
            competition_name="titanic",
            dataset_path=Path("/tmp/titanic"),
            competition_type=CompetitionType.BINARY_CLASSIFICATION,
            num_samples=891,
            num_features=11,
            target_column="Survived",
            num_classes=2,
        )
        
        assert spec.target_column == "Survived"
        assert spec.num_classes == 2
        
    def test_with_column_classifications(self):
        """Test DatasetSpec with column type classifications."""
        spec = DatasetSpec(
            competition_name="test",
            dataset_path=Path("/tmp/test"),
            competition_type=CompetitionType.MULTICLASS_CLASSIFICATION,
            num_samples=1000,
            num_features=20,
            text_columns=["description", "title"],
            categorical_columns=["category", "type"],
            numerical_columns=["score", "rating", "count"],
        )
        
        assert spec.text_columns == ["description", "title"]
        assert spec.categorical_columns == ["category", "type"]
        assert spec.numerical_columns == ["score", "rating", "count"]
        
    def test_post_init_validation(self):
        """Test __post_init__ validation and defaults."""
        # Test binary classification auto-correction
        spec = DatasetSpec(
            competition_name="test",
            dataset_path=Path("/tmp/test"),
            competition_type=CompetitionType.BINARY_CLASSIFICATION,
            num_samples=100,
            num_features=10,
            num_classes=5,  # Wrong for binary
        )
        assert spec.num_classes == 2  # Should be corrected
        
        # Test regression auto-correction
        spec = DatasetSpec(
            competition_name="test", 
            dataset_path=Path("/tmp/test"),
            competition_type=CompetitionType.REGRESSION,
            num_samples=100,
            num_features=10,
        )
        assert spec.num_classes == 1
        
    def test_optimization_settings(self):
        """Test dataset size-based optimization settings."""
        # Large dataset
        spec = DatasetSpec(
            competition_name="large",
            dataset_path=Path("/tmp/large"),
            competition_type=CompetitionType.REGRESSION,
            num_samples=150000,
            num_features=100,
        )
        
        assert spec.recommended_batch_size <= 64
        assert spec.prefetch_size == 8
        
        # Small dataset
        spec = DatasetSpec(
            competition_name="small",
            dataset_path=Path("/tmp/small"),
            competition_type=CompetitionType.BINARY_CLASSIFICATION,
            num_samples=1000,
            num_features=10,
        )
        
        assert spec.recommended_batch_size >= 16
        assert spec.prefetch_size == 2


class MockKaggleDataset(KaggleDataset):
    """Mock implementation of KaggleDataset for testing."""
    
    def __init__(self, spec, split="train", transform=None, cache_dir=None):
        # Create mock data before calling super().__init__
        self.mock_data = pd.DataFrame({
            'feature1': [1, 2, 3, 4, 5],
            'feature2': ['A', 'B', 'C', 'D', 'E'],
            'target': [0, 1, 0, 1, 0]
        })
        super().__init__(spec, split, transform, cache_dir)
        
    def _load_data(self):
        """Mock data loading."""
        self._data = self.mock_data
        
    def _validate_data(self):
        """Mock data validation."""
        assert self._data is not None
        assert len(self._data) > 0
        
    def __getitem__(self, index):
        """Mock getitem implementation."""
        if index >= len(self._data):
            raise IndexError(f"Index {index} out of range for dataset of size {len(self._data)}")
            
        row = self._data.iloc[index]
        
        # Create sample with text representation
        text = f"feature1: {row['feature1']}, feature2: {row['feature2']}"
        
        sample = {
            'text': text,
            'input_ids': [1, 2, 3, 4],  # Mock tokenized IDs
            'attention_mask': [1, 1, 1, 1],
            'labels': row['target'] if 'target' in row else None,
            'metadata': {'index': index}
        }
        
        if self.transform:
            sample = self.transform(sample)
            
        return sample


class TestKaggleDataset:
    """Test KaggleDataset abstract base class."""
    
    @pytest.fixture
    def sample_spec(self):
        """Create sample dataset spec."""
        return DatasetSpec(
            competition_name="test",
            dataset_path=Path("/tmp/test"),
            competition_type=CompetitionType.BINARY_CLASSIFICATION,
            num_samples=5,
            num_features=2,
            target_column="target",
        )
        
    def test_dataset_creation(self, sample_spec):
        """Test basic dataset creation."""
        dataset = MockKaggleDataset(sample_spec)
        
        assert dataset.spec == sample_spec
        assert dataset.split == "train"
        assert len(dataset) == 5
        
    def test_dataset_with_split(self, sample_spec):
        """Test dataset creation with different split."""
        dataset = MockKaggleDataset(sample_spec, split="test")
        assert dataset.split == "test"
        
    def test_dataset_getitem(self, sample_spec):
        """Test dataset indexing."""
        dataset = MockKaggleDataset(sample_spec)
        
        sample = dataset[0]
        assert 'text' in sample
        assert 'input_ids' in sample
        assert 'attention_mask' in sample
        assert 'labels' in sample
        assert 'metadata' in sample
        
        # Test that text is generated
        assert "feature1: 1" in sample['text']
        assert "feature2: A" in sample['text']
        
    def test_dataset_length(self, sample_spec):
        """Test dataset length."""
        dataset = MockKaggleDataset(sample_spec)
        assert len(dataset) == 5
        
    def test_dataset_iteration(self, sample_spec):
        """Test dataset iteration."""
        dataset = MockKaggleDataset(sample_spec)
        
        samples = list(dataset)
        assert len(samples) == 5
        
        for i, sample in enumerate(samples):
            assert sample['metadata']['index'] == i
            
    def test_dataset_out_of_bounds(self, sample_spec):
        """Test dataset index out of bounds."""
        dataset = MockKaggleDataset(sample_spec)
        
        with pytest.raises(IndexError):
            dataset[10]
            
    def test_get_sample_by_id(self, sample_spec):
        """Test getting sample by ID."""
        dataset = MockKaggleDataset(sample_spec)
        
        # Test by index
        sample = dataset.get_sample_by_id("2")
        assert sample is not None
        assert sample['metadata']['index'] == 2
        
        # Test non-existent ID
        sample = dataset.get_sample_by_id("999")
        assert sample is None
        
    def test_get_batch(self, sample_spec):
        """Test batch creation."""
        dataset = MockKaggleDataset(sample_spec)
        
        batch = dataset.get_batch([0, 1, 2])
        
        assert 'input_ids' in batch
        assert 'attention_mask' in batch
        assert 'labels' in batch
        
        # Check MLX array types
        assert isinstance(batch['input_ids'], mx.array)
        assert isinstance(batch['attention_mask'], mx.array)
        assert isinstance(batch['labels'], mx.array)
        
        # Check shapes
        assert batch['input_ids'].shape[0] == 3  # Batch size
        assert batch['attention_mask'].shape[0] == 3
        assert batch['labels'].shape[0] == 3
        
    def test_collate_batch(self, sample_spec):
        """Test batch collation."""
        dataset = MockKaggleDataset(sample_spec)
        
        samples = [dataset[i] for i in range(3)]
        batch = dataset._collate_batch(samples)
        
        assert 'input_ids' in batch
        assert 'attention_mask' in batch
        assert 'labels' in batch
        assert 'text' in batch
        
        # Check that text is preserved as list
        assert isinstance(batch['text'], list)
        assert len(batch['text']) == 3
        
    def test_competition_info(self, sample_spec):
        """Test competition info retrieval."""
        dataset = MockKaggleDataset(sample_spec)
        
        info = dataset.get_competition_info()
        
        assert info['competition_name'] == "test"
        assert info['competition_type'] == "binary_classification"
        assert info['num_samples'] == 5
        assert info['num_features'] == 2
        assert info['split'] == "train"
        assert info['target_column'] == "target"
        assert info['num_classes'] == 2
        
    def test_data_statistics(self, sample_spec):
        """Test data statistics."""
        dataset = MockKaggleDataset(sample_spec)
        
        stats = dataset.get_data_statistics()
        
        assert 'total_samples' in stats
        assert 'columns' in stats
        assert 'dtypes' in stats
        assert 'memory_usage' in stats
        assert 'missing_values' in stats
        
        assert stats['total_samples'] == 5
        assert 'feature1' in stats['columns']
        assert 'feature2' in stats['columns']
        
    def test_caching_operations(self, sample_spec, test_data_dir):
        """Test caching enable/disable/clear operations."""
        dataset = MockKaggleDataset(sample_spec)
        
        # Test enabling caching
        dataset.enable_caching(test_data_dir / "cache")
        assert dataset.spec.enable_caching == True
        assert dataset.cache_dir == test_data_dir / "cache"
        
        # Test clearing cache
        dataset.clear_cache()
        assert len(dataset._sample_cache) == 0
        
    def test_mlx_device_info(self, sample_spec):
        """Test MLX device information."""
        dataset = MockKaggleDataset(sample_spec)
        
        info = dataset.get_mlx_device_info()
        
        assert 'device' in info
        assert 'unified_memory' in info
        assert 'default_stream' in info
        assert 'mlx_available' in info
        
        assert info['mlx_available'] == True
        assert info['unified_memory'] == sample_spec.use_unified_memory
        
    def test_transform_application(self, sample_spec):
        """Test transform function application."""
        def uppercase_transform(sample):
            sample['text'] = sample['text'].upper()
            return sample
            
        dataset = MockKaggleDataset(sample_spec, transform=uppercase_transform)
        
        sample = dataset[0]
        assert sample['text'].isupper()
        
    def test_sample_text_extraction(self, sample_spec):
        """Test sample text extraction."""
        dataset = MockKaggleDataset(sample_spec)
        
        text = dataset.get_sample_text(0)
        assert isinstance(text, str)
        assert "feature1: 1" in text