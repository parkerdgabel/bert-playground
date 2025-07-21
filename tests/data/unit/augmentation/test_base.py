"""Tests for base augmentation classes."""

import mlx.core as mx
import pytest

from data.augmentation.base import (
    AugmentationMode,
    BaseAugmentationStrategy,
    BaseAugmenter,
    BaseFeatureAugmenter,
    ComposeAugmenter,
    ConditionalAugmenter,
    FeatureMetadata,
    FeatureType,
)


class TestFeatureType:
    """Test FeatureType enum."""

    def test_all_types_defined(self):
        """Test all feature types are defined."""
        expected_types = {"NUMERICAL", "CATEGORICAL", "TEXT", "DATE", "BOOLEAN", "UNKNOWN"}
        actual_types = {t.name for t in FeatureType}
        assert actual_types == expected_types


class TestAugmentationMode:
    """Test AugmentationMode enum."""

    def test_all_modes_defined(self):
        """Test all augmentation modes are defined."""
        expected_modes = {"NONE", "LIGHT", "MODERATE", "HEAVY", "CUSTOM"}
        actual_modes = {m.name for m in AugmentationMode}
        assert actual_modes == expected_modes


class TestFeatureMetadata:
    """Test FeatureMetadata dataclass."""

    def test_basic_creation(self):
        """Test creating feature metadata."""
        meta = FeatureMetadata(
            name="age",
            feature_type=FeatureType.NUMERICAL,
            importance=0.8,
            statistics={"mean": 30, "std": 10},
        )
        assert meta.name == "age"
        assert meta.feature_type == FeatureType.NUMERICAL
        assert meta.importance == 0.8
        assert meta.statistics["mean"] == 30

    def test_default_values(self):
        """Test default values."""
        meta = FeatureMetadata(name="test", feature_type=FeatureType.TEXT)
        assert meta.importance == 1.0
        assert meta.statistics == {}
        assert meta.domain_info == {}
        assert meta.is_target is False


class TestBaseAugmenter:
    """Test BaseAugmenter abstract class."""

    def test_seed_handling(self):
        """Test random seed handling."""

        class DummyAugmenter(BaseAugmenter):
            def augment(self, data, **kwargs):
                return data

        # Test with seed
        aug = DummyAugmenter(seed=42)
        assert aug.seed == 42
        assert aug._rng is not None

        # Test without seed
        aug_no_seed = DummyAugmenter()
        assert aug_no_seed.seed is None
        assert aug_no_seed._rng is None

    def test_abstract_method(self):
        """Test that augment method must be implemented."""
        with pytest.raises(TypeError):
            BaseAugmenter()


class TestBaseFeatureAugmenter:
    """Test BaseFeatureAugmenter class."""

    def test_initialization(self):
        """Test initialization with feature types."""

        class NumericAugmenter(BaseFeatureAugmenter):
            def __init__(self):
                super().__init__([FeatureType.NUMERICAL])

            def augment(self, data, **kwargs):
                return data * 1.1

        aug = NumericAugmenter()
        assert aug.supported_types == [FeatureType.NUMERICAL]

    def test_can_augment(self):
        """Test can_augment method."""

        class NumericAugmenter(BaseFeatureAugmenter):
            def __init__(self):
                super().__init__([FeatureType.NUMERICAL])

            def augment(self, data, **kwargs):
                return data

        aug = NumericAugmenter()
        assert aug.can_augment(FeatureType.NUMERICAL)
        assert not aug.can_augment(FeatureType.TEXT)
        assert not aug.can_augment(FeatureType.CATEGORICAL)


class TestComposeAugmenter:
    """Test ComposeAugmenter class."""

    def test_sequential_application(self):
        """Test sequential application of augmenters."""

        class MultiplyAugmenter(BaseAugmenter):
            def __init__(self, factor):
                super().__init__()
                self.factor = factor

            def augment(self, data, **kwargs):
                return data * self.factor

        class AddAugmenter(BaseAugmenter):
            def __init__(self, value):
                super().__init__()
                self.value = value

            def augment(self, data, **kwargs):
                return data + self.value

        compose = ComposeAugmenter(
            [MultiplyAugmenter(2), AddAugmenter(10)]
        )

        result = compose.augment(5)
        assert result == 20  # (5 * 2) + 10

    def test_empty_compose(self):
        """Test compose with no augmenters."""
        compose = ComposeAugmenter([])
        result = compose.augment(42)
        assert result == 42


class TestConditionalAugmenter:
    """Test ConditionalAugmenter class."""

    def test_condition_true(self):
        """Test augmentation when condition is true."""

        class DoubleAugmenter(BaseAugmenter):
            def augment(self, data, **kwargs):
                return data * 2

        def is_positive(data):
            return data > 0

        conditional = ConditionalAugmenter(DoubleAugmenter(), is_positive)

        assert conditional.augment(5) == 10
        assert conditional.augment(-5) == -5

    def test_with_feature_metadata(self):
        """Test condition based on feature metadata."""

        class ScaleAugmenter(BaseAugmenter):
            def augment(self, data, **kwargs):
                return data * 0.9

        def is_important(data, feature_metadata=None):
            return feature_metadata and feature_metadata.importance > 0.7

        conditional = ConditionalAugmenter(ScaleAugmenter(), is_important)

        meta_important = FeatureMetadata(
            name="important", feature_type=FeatureType.NUMERICAL, importance=0.9
        )
        meta_not_important = FeatureMetadata(
            name="not_important", feature_type=FeatureType.NUMERICAL, importance=0.3
        )

        result1 = conditional.augment(100, feature_metadata=meta_important)
        assert result1 == 90

        result2 = conditional.augment(100, feature_metadata=meta_not_important)
        assert result2 == 100


class TestBaseAugmentationStrategy:
    """Test BaseAugmentationStrategy class."""

    def test_initialization(self):
        """Test strategy initialization."""

        class CustomStrategy(BaseAugmentationStrategy):
            def apply(self, data, config):
                return data * 2

        strategy = CustomStrategy("custom", [FeatureType.NUMERICAL])
        assert strategy.name == "custom"
        assert strategy.supported_types == [FeatureType.NUMERICAL]

    def test_supports_type(self):
        """Test supports_type method."""

        class TextStrategy(BaseAugmentationStrategy):
            def apply(self, data, config):
                return data.upper()

        strategy = TextStrategy("uppercase", [FeatureType.TEXT])
        assert strategy.supports_type(FeatureType.TEXT)
        assert not strategy.supports_type(FeatureType.NUMERICAL)

    def test_multiple_supported_types(self):
        """Test strategy supporting multiple types."""

        class MultiTypeStrategy(BaseAugmentationStrategy):
            def apply(self, data, config):
                return str(data)

        strategy = MultiTypeStrategy(
            "stringify", [FeatureType.NUMERICAL, FeatureType.CATEGORICAL]
        )
        assert strategy.supports_type(FeatureType.NUMERICAL)
        assert strategy.supports_type(FeatureType.CATEGORICAL)
        assert not strategy.supports_type(FeatureType.TEXT)