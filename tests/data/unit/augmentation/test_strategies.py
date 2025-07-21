"""Tests for augmentation strategies."""

import mlx.core as mx
import pytest

from data.augmentation.base import FeatureMetadata, FeatureType
from data.augmentation.strategies import (
    CategoricalAugmenter,
    DateAugmenter,
    GaussianNoiseStrategy,
    MaskingStrategy,
    NumericalAugmenter,
    ScalingStrategy,
    SynonymReplacementStrategy,
    TextFeatureAugmenter,
    create_augmenter_for_type,
)


class TestNumericalAugmenter:
    """Test NumericalAugmenter class."""

    def test_basic_augmentation(self):
        """Test basic numerical augmentation."""
        meta = FeatureMetadata(
            name="value",
            feature_type=FeatureType.NUMERICAL,
            statistics={"mean": 100, "std": 20},
        )
        
        augmenter = NumericalAugmenter(
            noise_factor=0.1,
            scale_range=(0.9, 1.1),
            seed=42
        )
        
        # Test augmentation
        original = 100.0
        augmented = augmenter.augment(original, feature_metadata=meta)
        
        # Should be different but within reasonable range
        assert augmented != original
        assert 80 < augmented < 120

    def test_no_augmentation_when_disabled(self):
        """Test that augmentation can be disabled."""
        augmenter = NumericalAugmenter(noise_factor=0.0, scale_range=(1.0, 1.0))
        
        original = 100.0
        augmented = augmenter.augment(original)
        assert augmented == original

    def test_mlx_array_input(self):
        """Test augmentation with MLX array input."""
        augmenter = NumericalAugmenter(noise_factor=0.1, seed=42)
        
        original = mx.array([1.0, 2.0, 3.0])
        augmented = augmenter.augment(original)
        
        assert isinstance(augmented, mx.array)
        assert augmented.shape == original.shape


class TestCategoricalAugmenter:
    """Test CategoricalAugmenter class."""

    def test_swap_probability(self):
        """Test categorical value swapping."""
        meta = FeatureMetadata(
            name="category",
            feature_type=FeatureType.CATEGORICAL,
            domain_info={"values": ["A", "B", "C", "D"]},
        )
        
        # High swap probability for testing
        augmenter = CategoricalAugmenter(swap_probability=1.0, seed=42)
        
        original = "A"
        augmented = augmenter.augment(original, feature_metadata=meta)
        
        # Should swap to a different value
        assert augmented != original
        assert augmented in ["B", "C", "D"]

    def test_no_swap(self):
        """Test no swapping with zero probability."""
        meta = FeatureMetadata(
            name="category",
            feature_type=FeatureType.CATEGORICAL,
            domain_info={"values": ["A", "B", "C"]},
        )
        
        augmenter = CategoricalAugmenter(swap_probability=0.0)
        
        original = "B"
        augmented = augmenter.augment(original, feature_metadata=meta)
        assert augmented == original

    def test_missing_domain_info(self):
        """Test behavior when domain info is missing."""
        augmenter = CategoricalAugmenter(swap_probability=1.0)
        
        original = "X"
        augmented = augmenter.augment(original)
        assert augmented == original  # Should not change without domain info


class TestTextFeatureAugmenter:
    """Test TextFeatureAugmenter class."""

    def test_synonym_replacement(self):
        """Test text augmentation with synonym replacement."""
        augmenter = TextFeatureAugmenter(
            synonym_probability=0.5,
            mask_probability=0.0,
            seed=42
        )
        
        original = "The quick brown fox"
        augmented = augmenter.augment(original)
        
        # Should be text
        assert isinstance(augmented, str)
        assert len(augmented) > 0

    def test_masking(self):
        """Test text masking."""
        augmenter = TextFeatureAugmenter(
            synonym_probability=0.0,
            mask_probability=1.0,
            mask_token="[MASK]"
        )
        
        original = "Hello world test"
        augmented = augmenter.augment(original)
        
        # Should contain mask tokens
        assert "[MASK]" in augmented
        assert augmented.count("[MASK]") >= 1

    def test_no_augmentation(self):
        """Test no augmentation."""
        augmenter = TextFeatureAugmenter(
            synonym_probability=0.0,
            mask_probability=0.0
        )
        
        original = "Test text"
        augmented = augmenter.augment(original)
        assert augmented == original


class TestDateAugmenter:
    """Test DateAugmenter class."""

    def test_date_shifting(self):
        """Test date shifting augmentation."""
        augmenter = DateAugmenter(shift_days_range=(-7, 7), seed=42)
        
        original = "2024-01-15"
        augmented = augmenter.augment(original)
        
        # Should be a valid date string
        assert isinstance(augmented, str)
        assert len(augmented) == 10
        assert augmented != original

    def test_no_shift(self):
        """Test no date shifting."""
        augmenter = DateAugmenter(shift_days_range=(0, 0))
        
        original = "2024-01-15"
        augmented = augmenter.augment(original)
        assert augmented == original

    def test_invalid_date(self):
        """Test handling of invalid dates."""
        augmenter = DateAugmenter(shift_days_range=(-7, 7))
        
        original = "not-a-date"
        augmented = augmenter.augment(original)
        assert augmented == original  # Should return unchanged


class TestAugmentationStrategies:
    """Test individual augmentation strategies."""

    def test_gaussian_noise_strategy(self):
        """Test Gaussian noise strategy."""
        strategy = GaussianNoiseStrategy()
        config = {"noise_std": 0.1}
        
        original = 100.0
        augmented = strategy.apply(original, config)
        
        # Should add noise
        assert augmented != original
        assert isinstance(augmented, float)

    def test_scaling_strategy(self):
        """Test scaling strategy."""
        strategy = ScalingStrategy()
        config = {"scale_range": (0.8, 1.2)}
        
        original = 100.0
        augmented = strategy.apply(original, config)
        
        # Should scale the value
        assert 80 <= augmented <= 120

    def test_synonym_replacement_strategy(self):
        """Test synonym replacement strategy."""
        strategy = SynonymReplacementStrategy()
        config = {"synonym_map": {"quick": ["fast", "rapid", "swift"]}}
        
        original = "The quick brown fox"
        augmented = strategy.apply(original, config)
        
        # Should be text (might or might not change)
        assert isinstance(augmented, str)

    def test_masking_strategy(self):
        """Test masking strategy."""
        strategy = MaskingStrategy()
        config = {"mask_token": "[MASK]", "mask_probability": 0.5}
        
        original = "Hello world test example"
        augmented = strategy.apply(original, config)
        
        # Should contain some words or masks
        assert isinstance(augmented, str)
        assert len(augmented) > 0


class TestCreateAugmenterForType:
    """Test create_augmenter_for_type factory function."""

    def test_numerical_augmenter_creation(self):
        """Test creating numerical augmenter."""
        augmenter = create_augmenter_for_type(FeatureType.NUMERICAL)
        assert isinstance(augmenter, NumericalAugmenter)

    def test_categorical_augmenter_creation(self):
        """Test creating categorical augmenter."""
        augmenter = create_augmenter_for_type(FeatureType.CATEGORICAL)
        assert isinstance(augmenter, CategoricalAugmenter)

    def test_text_augmenter_creation(self):
        """Test creating text augmenter."""
        augmenter = create_augmenter_for_type(FeatureType.TEXT)
        assert isinstance(augmenter, TextFeatureAugmenter)

    def test_date_augmenter_creation(self):
        """Test creating date augmenter."""
        augmenter = create_augmenter_for_type(FeatureType.DATE)
        assert isinstance(augmenter, DateAugmenter)

    def test_unknown_type(self):
        """Test handling unknown feature type."""
        augmenter = create_augmenter_for_type(FeatureType.UNKNOWN)
        assert augmenter is None

    def test_with_config(self):
        """Test creating augmenter with config."""
        from data.augmentation.config import (
            AugmentationConfig,
            NumericalAugmentationConfig,
        )
        
        config = AugmentationConfig(
            numerical=NumericalAugmentationConfig(
                noise_factor=0.2,
                scale_range=(0.7, 1.3),
            )
        )
        
        augmenter = create_augmenter_for_type(
            FeatureType.NUMERICAL, 
            config=config,
            seed=42
        )
        
        assert isinstance(augmenter, NumericalAugmenter)
        assert augmenter.noise_factor == 0.2
        assert augmenter.scale_range == (0.7, 1.3)
        assert augmenter.seed == 42