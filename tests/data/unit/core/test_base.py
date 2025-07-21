"""Tests for core data base classes."""

import tempfile
from pathlib import Path

import mlx.core as mx
import pytest

from data.core.base import (
    CompetitionType,
    DatasetSpec,
)
from tests.data.fixtures.configs import (
    create_dataset_spec,
    create_small_dataset_spec,
)
from tests.data.fixtures.datasets import MockKaggleDataset
from tests.data.fixtures.utils import (
    check_dataset_consistency,
    create_sample_dataframe,
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
            "UNKNOWN",
        ]

        for type_name in expected_types:
            assert hasattr(CompetitionType, type_name)

    def test_competition_type_values(self):
        """Test competition type string values."""
        assert CompetitionType.BINARY_CLASSIFICATION.value == "binary_classification"
        assert (
            CompetitionType.MULTICLASS_CLASSIFICATION.value
            == "multiclass_classification"
        )
        assert CompetitionType.REGRESSION.value == "regression"
        assert CompetitionType.UNKNOWN.value == "unknown"

    def test_competition_type_from_string(self):
        """Test creating competition type from string."""
        # This might not be implemented yet, but it's a common pattern
        for comp_type in CompetitionType:
            assert CompetitionType(comp_type.value) == comp_type


class TestDatasetSpec:
    """Test DatasetSpec dataclass."""

    def test_basic_creation(self):
        """Test basic DatasetSpec creation."""
        spec = create_dataset_spec(
            name="test",
            task_type="classification",
            num_samples=100,
            num_features=10,
            dataset_path=Path("/tmp/test"),
        )

        assert spec.competition_name == "test"
        assert spec.dataset_path == Path("/tmp/test")
        assert spec.competition_type == CompetitionType.BINARY_CLASSIFICATION
        assert spec.num_samples == 100
        assert spec.num_features == 10

    def test_with_target_column(self):
        """Test DatasetSpec with target column."""
        spec = create_dataset_spec(
            competition_name="titanic",
            dataset_path=Path("/tmp/titanic"),
            num_samples=891,
            num_features=11,
            target_column="Survived",
            num_classes=2,
        )

        assert spec.target_column == "Survived"
        assert spec.num_classes == 2

    def test_with_column_classifications(self):
        """Test DatasetSpec with column type classifications."""
        spec = create_dataset_spec(
            task_type="multiclass_classification",
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
        # Test binary classification with correct num_classes
        spec = create_dataset_spec(
            competition_type=CompetitionType.BINARY_CLASSIFICATION,
            num_classes=2,
        )
        assert spec.num_classes == 2

        # Test regression defaults
        spec = create_dataset_spec(
            task_type="regression",  # Use the string to get proper defaults
        )
        # Regression should not have num_classes set by default
        assert spec.num_classes is None or spec.num_classes == 1

    def test_optimization_settings(self):
        """Test dataset size-based optimization settings."""
        # Large dataset
        spec = create_dataset_spec(
            competition_name="large",
            num_samples=150000,
            num_features=100,
        )

        assert spec.recommended_batch_size <= 64
        assert spec.prefetch_size == 8

        # Small dataset
        spec = create_small_dataset_spec()

        assert spec.recommended_batch_size >= 16
        assert spec.prefetch_size == 2

    def test_dataset_spec_path_conversion(self):
        """Test dataset spec path conversion."""
        # Test string path conversion
        spec = DatasetSpec(
            competition_name="test",
            dataset_path="path/to/data",  # String path
            competition_type=CompetitionType.BINARY_CLASSIFICATION,
            num_samples=100,
            num_features=10,
        )

        # Should be converted to Path object
        assert isinstance(spec.dataset_path, Path)
        assert str(spec.dataset_path) == "path/to/data"

    def test_dataset_spec_serialization(self):
        """Test dataset spec serialization."""
        spec = create_dataset_spec(
            name="test",
            task_type="classification",
            num_samples=100,
            num_features=10,
            dataset_path=Path("/tmp/test"),
        )

        # Convert to dict (for saving)
        spec_dict = {
            "competition_name": spec.competition_name,
            "dataset_path": str(spec.dataset_path),
            "competition_type": spec.competition_type.value,
            "num_samples": spec.num_samples,
            "num_features": spec.num_features,
        }

        assert isinstance(spec_dict, dict)
        assert spec_dict["competition_type"] == "binary_classification"


class TestKaggleDataset:
    """Test KaggleDataset abstract base class."""

    @pytest.fixture
    def sample_spec(self):
        """Create sample dataset spec."""
        return create_dataset_spec(
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
        assert "text" in sample
        assert "input_ids" in sample
        assert "attention_mask" in sample
        assert "labels" in sample
        assert "metadata" in sample

        # Test that text is generated with some features
        assert "feature_" in sample["text"]  # Should contain feature references
        assert ":" in sample["text"]  # Should have key-value format

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
            assert sample["metadata"]["index"] == i

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
        assert sample["metadata"]["index"] == 2

        # Test non-existent ID
        sample = dataset.get_sample_by_id("999")
        assert sample is None

    def test_get_batch(self, sample_spec):
        """Test batch creation."""
        dataset = MockKaggleDataset(sample_spec)

        batch = dataset.get_batch([0, 1, 2])

        # Check what's actually in the batch
        assert "labels" in batch
        assert "text" in batch

        # Check MLX array types for numeric data
        assert isinstance(batch["labels"], mx.array)

        # Check shapes
        assert batch["labels"].shape[0] == 3  # Batch size

        # Text should remain as list
        assert isinstance(batch["text"], list)
        assert len(batch["text"]) == 3

    def test_collate_batch(self, sample_spec):
        """Test batch collation."""
        dataset = MockKaggleDataset(sample_spec)

        samples = [dataset[i] for i in range(3)]
        batch = dataset._collate_batch(samples)

        # Check keys that should be in batch
        assert "labels" in batch
        assert "text" in batch

        # Check that text is preserved as list
        assert isinstance(batch["text"], list)
        assert len(batch["text"]) == 3

        # Check labels are MLX arrays
        assert isinstance(batch["labels"], mx.array)

    def test_competition_info(self, sample_spec):
        """Test competition info retrieval."""
        dataset = MockKaggleDataset(sample_spec)

        info = dataset.get_competition_info()

        assert info["competition_name"] == "test_dataset"
        assert info["competition_type"] == "binary_classification"
        assert info["num_samples"] == 5
        assert info["num_features"] == 2
        assert info["split"] == "train"
        assert info["target_column"] == "target"
        assert info["num_classes"] == 2

    def test_data_statistics(self, sample_spec):
        """Test data statistics."""
        dataset = MockKaggleDataset(sample_spec)

        stats = dataset.get_data_statistics()

        assert "total_samples" in stats
        assert "columns" in stats
        assert "dtypes" in stats
        assert "memory_usage" in stats
        assert "missing_values" in stats

        assert stats["total_samples"] == 5
        assert "feature_0" in stats["columns"]
        assert "feature_1" in stats["columns"]

    def test_caching_operations(self, sample_spec):
        """Test caching enable/disable/clear operations."""
        with tempfile.TemporaryDirectory() as temp_dir:
            cache_dir = Path(temp_dir) / "cache"
            dataset = MockKaggleDataset(sample_spec)

            # Test enabling caching
            dataset.enable_caching(cache_dir)
            assert dataset.spec.enable_caching == True
            assert dataset.cache_dir == cache_dir

            # Test clearing cache
            dataset.clear_cache()
            assert len(dataset._sample_cache) == 0

    def test_mlx_device_info(self, sample_spec):
        """Test MLX device information."""
        dataset = MockKaggleDataset(sample_spec)

        info = dataset.get_mlx_device_info()

        assert "device" in info
        assert "unified_memory" in info
        assert "default_stream" in info
        assert "mlx_available" in info

        assert info["mlx_available"] == True
        assert info["unified_memory"] == sample_spec.use_unified_memory

    def test_transform_storage(self, sample_spec):
        """Test transform function storage."""

        def uppercase_transform(sample):
            sample["text"] = sample["text"].upper()
            return sample

        dataset = MockKaggleDataset(sample_spec, transform=uppercase_transform)

        # Check that transform is stored
        assert dataset.transform is not None
        assert callable(dataset.transform)

    def test_sample_text_extraction(self, sample_spec):
        """Test sample text extraction."""
        dataset = MockKaggleDataset(sample_spec)

        text = dataset.get_sample_text(0)
        assert isinstance(text, str)
        assert "feature_0:" in text

    def test_dataset_consistency(self, sample_spec):
        """Test dataset consistency checks."""
        dataset = MockKaggleDataset(sample_spec)

        # Check consistency
        is_consistent = check_dataset_consistency(dataset)
        assert is_consistent

    def test_with_missing_values(self):
        """Test dataset with missing values."""
        spec = create_dataset_spec(num_samples=10)
        dataset = MockKaggleDataset(spec)

        # Get sample - even if data has missing values, text should be generated
        sample = dataset[0]

        # Dataset should handle missing values gracefully
        assert "text" in sample
        assert sample["text"] is not None
        assert isinstance(sample["text"], str)


@pytest.mark.integration
class TestDatasetIntegration:
    """Integration tests for dataset functionality."""

    def test_dataset_with_real_dataframe(self):
        """Test dataset with realistic dataframe."""
        df = create_sample_dataframe(num_rows=100)

        spec = create_dataset_spec(num_samples=len(df))
        dataset = MockKaggleDataset(spec)
        dataset._data = df  # Replace mock data

        # Test that dataset handles real data
        sample = dataset[0]
        assert sample is not None
        assert "text" in sample

    def test_dataset_pipeline(self):
        """Test complete dataset pipeline."""
        # Create spec
        spec = create_dataset_spec(num_samples=5)

        # Create dataset
        dataset = MockKaggleDataset(spec)

        # Enable caching
        with tempfile.TemporaryDirectory() as temp_dir:
            dataset.enable_caching(Path(temp_dir))

            # Get samples
            samples = []
            for i in range(len(dataset)):
                sample = dataset[i]
                samples.append(sample)

            # Create batch
            batch = dataset.get_batch([0, 1, 2])

            # Verify batch
            assert batch["labels"].shape[0] == 3
            assert len(batch["text"]) == 3

            # Get statistics
            stats = dataset.get_data_statistics()
            assert stats["total_samples"] == 5
