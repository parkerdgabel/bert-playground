"""Tests for tabular augmentation."""

import mlx.core as mx
import pytest

from data.augmentation.base import FeatureMetadata, FeatureType
from data.augmentation.config import AugmentationConfig, AugmentationMode
from data.augmentation.tabular import (
    TabularAugmenter,
    TabularBERTAugmenter,
    TabularToTextAugmenter,
)


class TestTabularAugmenter:
    """Test TabularAugmenter class."""

    def test_basic_augmentation(self):
        """Test basic tabular augmentation."""
        feature_metadata = {
            "age": FeatureMetadata(
                name="age",
                feature_type=FeatureType.NUMERICAL,
                statistics={"mean": 30, "std": 10},
            ),
            "category": FeatureMetadata(
                name="category",
                feature_type=FeatureType.CATEGORICAL,
                domain_info={"values": ["A", "B", "C"]},
            ),
            "description": FeatureMetadata(
                name="description",
                feature_type=FeatureType.TEXT,
            ),
        }
        
        config = AugmentationConfig.from_mode(AugmentationMode.LIGHT)
        augmenter = TabularAugmenter(config, feature_metadata, seed=42)
        
        sample = {
            "age": 25,
            "category": "B",
            "description": "Sample text",
        }
        
        augmented = augmenter.augment(sample)
        
        # Check structure is preserved
        assert set(augmented.keys()) == set(sample.keys())
        assert isinstance(augmented["age"], (int, float))
        assert isinstance(augmented["category"], str)
        assert isinstance(augmented["description"], str)

    def test_selective_augmentation(self):
        """Test that only specified features are augmented."""
        feature_metadata = {
            "feature1": FeatureMetadata("feature1", FeatureType.NUMERICAL),
            "feature2": FeatureMetadata("feature2", FeatureType.NUMERICAL),
        }
        
        # Configure to only augment feature1
        config = AugmentationConfig.from_mode(AugmentationMode.MODERATE)
        augmenter = TabularAugmenter(
            config, 
            feature_metadata,
            features_to_augment=["feature1"],
            seed=42
        )
        
        sample = {"feature1": 100, "feature2": 200}
        augmented = augmenter.augment(sample)
        
        # feature2 should remain unchanged
        assert augmented["feature2"] == 200

    def test_unknown_features(self):
        """Test handling of unknown features."""
        feature_metadata = {
            "known": FeatureMetadata("known", FeatureType.NUMERICAL),
        }
        
        config = AugmentationConfig()
        augmenter = TabularAugmenter(config, feature_metadata)
        
        sample = {"known": 100, "unknown": "value"}
        augmented = augmenter.augment(sample)
        
        # Unknown features should pass through unchanged
        assert "unknown" in augmented
        assert augmented["unknown"] == "value"

    def test_no_augmentation_mode(self):
        """Test NONE augmentation mode."""
        feature_metadata = {
            "value": FeatureMetadata("value", FeatureType.NUMERICAL),
        }
        
        config = AugmentationConfig.from_mode(AugmentationMode.NONE)
        augmenter = TabularAugmenter(config, feature_metadata)
        
        sample = {"value": 42}
        augmented = augmenter.augment(sample)
        
        # Should be unchanged
        assert augmented == sample

    def test_batch_augmentation(self):
        """Test augmenting multiple samples."""
        feature_metadata = {
            "x": FeatureMetadata("x", FeatureType.NUMERICAL),
        }
        
        config = AugmentationConfig.from_mode(AugmentationMode.LIGHT)
        augmenter = TabularAugmenter(config, feature_metadata, seed=42)
        
        samples = [{"x": i} for i in range(5)]
        augmented_samples = augmenter.augment_batch(samples)
        
        assert len(augmented_samples) == len(samples)
        for aug in augmented_samples:
            assert "x" in aug
            assert isinstance(aug["x"], (int, float))


class TestTabularToTextAugmenter:
    """Test TabularToTextAugmenter class."""

    def test_default_text_conversion(self):
        """Test default text conversion."""
        feature_metadata = {
            "age": FeatureMetadata("age", FeatureType.NUMERICAL, importance=0.9),
            "name": FeatureMetadata("name", FeatureType.CATEGORICAL, importance=0.8),
        }
        
        augmenter = TabularToTextAugmenter(feature_metadata=feature_metadata)
        
        sample = {"age": 25, "name": "John"}
        text = augmenter.convert_to_text(sample)
        
        assert isinstance(text, str)
        assert "age" in text
        assert "25" in text
        assert "name" in text
        assert "John" in text

    def test_custom_text_converter(self):
        """Test custom text converter function."""
        def custom_converter(features):
            age = features.get("age", "unknown")
            name = features.get("name", "unknown")
            return f"{name} is {age} years old"
        
        augmenter = TabularToTextAugmenter(text_converter=custom_converter)
        
        sample = {"age": 30, "name": "Alice"}
        text = augmenter.convert_to_text(sample)
        
        assert text == "Alice is 30 years old"

    def test_augment_with_text_conversion(self):
        """Test augmentation that converts to text."""
        feature_metadata = {
            "value": FeatureMetadata("value", FeatureType.NUMERICAL),
        }
        
        config = AugmentationConfig.from_mode(AugmentationMode.LIGHT)
        augmenter = TabularToTextAugmenter(
            config=config,
            feature_metadata=feature_metadata,
        )
        
        sample = {"value": 100}
        result = augmenter.augment(sample)
        
        assert isinstance(result, str)
        assert "value" in result

    def test_augment_multiple(self):
        """Test generating multiple augmented text versions."""
        feature_metadata = {
            "x": FeatureMetadata("x", FeatureType.NUMERICAL),
            "y": FeatureMetadata("y", FeatureType.CATEGORICAL),
        }
        
        config = AugmentationConfig.from_mode(AugmentationMode.MODERATE)
        augmenter = TabularToTextAugmenter(
            config=config,
            feature_metadata=feature_metadata,
            seed=42
        )
        
        sample = {"x": 10, "y": "category_a"}
        texts = augmenter.augment_multiple(sample, num_augmentations=3)
        
        assert len(texts) == 3
        assert all(isinstance(text, str) for text in texts)
        # With augmentation, texts might be different
        assert len(set(texts)) >= 1

    def test_feature_ordering_by_importance(self):
        """Test that features are ordered by importance in text."""
        feature_metadata = {
            "low": FeatureMetadata("low", FeatureType.NUMERICAL, importance=0.1),
            "high": FeatureMetadata("high", FeatureType.NUMERICAL, importance=0.9),
            "medium": FeatureMetadata("medium", FeatureType.NUMERICAL, importance=0.5),
        }
        
        augmenter = TabularToTextAugmenter(feature_metadata=feature_metadata)
        
        sample = {"low": 1, "high": 2, "medium": 3}
        text = augmenter.convert_to_text(sample)
        
        # High importance feature should appear first
        high_pos = text.find("high")
        medium_pos = text.find("medium")
        low_pos = text.find("low")
        
        assert high_pos < medium_pos < low_pos


class TestTabularBERTAugmenter:
    """Test TabularBERTAugmenter (legacy) class."""

    def test_deprecation_warning(self):
        """Test that deprecation warning is shown."""
        import warnings
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            augmenter = TabularBERTAugmenter()
            assert len(w) == 1
            assert "deprecated" in str(w[0].message).lower()

    def test_basic_functionality(self):
        """Test basic functionality still works."""
        feature_metadata = {
            "feature1": FeatureMetadata("feature1", FeatureType.NUMERICAL),
        }
        
        augmenter = TabularBERTAugmenter(feature_metadata=feature_metadata)
        
        sample = {"feature1": 42}
        result = augmenter.augment(sample)
        
        # Should still produce text output
        assert isinstance(result, str)

    def test_augment_for_bert(self):
        """Test augment_for_bert method."""
        augmenter = TabularBERTAugmenter()
        
        sample = {"a": 1, "b": "text"}
        texts = augmenter.augment_for_bert(sample, num_augmentations=2)
        
        assert len(texts) == 2
        assert all(isinstance(text, str) for text in texts)