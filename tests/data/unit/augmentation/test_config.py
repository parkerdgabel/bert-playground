"""Tests for augmentation configuration."""

import pytest

from data.augmentation.config import (
    AugmentationConfig,
    AugmentationMode,
    BERTAugmentationConfig,
    CategoricalAugmentationConfig,
    DomainKnowledgeConfig,
    NumericalAugmentationConfig,
    TextAugmentationConfig,
)


class TestNumericalAugmentationConfig:
    """Test NumericalAugmentationConfig."""

    def test_default_values(self):
        """Test default configuration values."""
        config = NumericalAugmentationConfig()
        assert config.noise_factor == 0.1
        assert config.scale_range == (0.9, 1.1)
        assert config.clip_outliers is True
        assert config.outlier_threshold == 3.0

    def test_custom_values(self):
        """Test custom configuration values."""
        config = NumericalAugmentationConfig(
            noise_factor=0.2,
            scale_range=(0.8, 1.2),
            clip_outliers=False,
        )
        assert config.noise_factor == 0.2
        assert config.scale_range == (0.8, 1.2)
        assert config.clip_outliers is False


class TestCategoricalAugmentationConfig:
    """Test CategoricalAugmentationConfig."""

    def test_default_values(self):
        """Test default configuration values."""
        config = CategoricalAugmentationConfig()
        assert config.swap_probability == 0.1
        assert config.use_frequency_weights is False

    def test_custom_values(self):
        """Test custom configuration values."""
        config = CategoricalAugmentationConfig(
            swap_probability=0.3,
            use_frequency_weights=True,
        )
        assert config.swap_probability == 0.3
        assert config.use_frequency_weights is True


class TestTextAugmentationConfig:
    """Test TextAugmentationConfig."""

    def test_default_values(self):
        """Test default configuration values."""
        config = TextAugmentationConfig()
        assert config.synonym_probability == 0.2
        assert config.mask_probability == 0.15
        assert config.mask_token == "[MASK]"
        assert config.max_synonyms == 3

    def test_custom_values(self):
        """Test custom configuration values."""
        config = TextAugmentationConfig(
            synonym_probability=0.3,
            mask_probability=0.2,
            mask_token="<mask>",
            synonym_map={"good": ["great", "excellent"]},
        )
        assert config.synonym_probability == 0.3
        assert config.mask_probability == 0.2
        assert config.mask_token == "<mask>"
        assert "good" in config.synonym_map


class TestDomainKnowledgeConfig:
    """Test DomainKnowledgeConfig."""

    def test_default_values(self):
        """Test default configuration values."""
        config = DomainKnowledgeConfig()
        assert config.feature_relationships == {}
        assert config.business_rules == []
        assert config.value_constraints == {}
        assert config.feature_interactions == []

    def test_with_relationships(self):
        """Test configuration with relationships."""
        config = DomainKnowledgeConfig(
            feature_relationships={"age": ["income", "education"]},
            business_rules=["age >= 18", "income > 0"],
        )
        assert "age" in config.feature_relationships
        assert len(config.business_rules) == 2


class TestBERTAugmentationConfig:
    """Test BERTAugmentationConfig (legacy)."""

    def test_deprecation_warning(self):
        """Test that deprecation warning is shown."""
        import warnings
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            config = BERTAugmentationConfig()
            assert len(w) == 1
            assert "deprecated" in str(w[0].message).lower()

    def test_inheritance(self):
        """Test that it inherits from TextAugmentationConfig."""
        config = BERTAugmentationConfig(mask_probability=0.25)
        assert config.mask_probability == 0.25
        assert hasattr(config, "synonym_probability")


class TestAugmentationConfig:
    """Test main AugmentationConfig class."""

    def test_default_values(self):
        """Test default configuration values."""
        config = AugmentationConfig()
        assert config.mode == AugmentationMode.MODERATE
        assert config.enabled is True
        assert isinstance(config.numerical_config, NumericalAugmentationConfig)
        assert isinstance(config.categorical_config, CategoricalAugmentationConfig)
        assert isinstance(config.text_config, TextAugmentationConfig)
        assert config.seed == 42

    def test_custom_components(self):
        """Test custom component configurations."""
        config = AugmentationConfig(
            mode=AugmentationMode.HEAVY,
            numerical_config=NumericalAugmentationConfig(gaussian_noise_std=0.3),
            categorical_config=CategoricalAugmentationConfig(swap_prob=0.2),
            seed=123,
        )
        assert config.mode == AugmentationMode.HEAVY
        assert config.numerical_config.gaussian_noise_std == 0.3
        assert config.categorical_config.swap_prob == 0.2
        assert config.seed == 123

    def test_from_mode_none(self):
        """Test from_mode with NONE mode."""
        config = AugmentationConfig.from_mode(AugmentationMode.NONE)
        assert config.mode == AugmentationMode.NONE
        assert config.enabled is False

    def test_from_mode_light(self):
        """Test from_mode with LIGHT mode."""
        config = AugmentationConfig.from_mode(AugmentationMode.LIGHT)
        assert config.mode == AugmentationMode.LIGHT
        assert config.enabled is True
        assert config.numerical_config.gaussian_noise_std == 0.05
        assert config.categorical_config.swap_prob == 0.05
        assert config.text_config.mask_prob == 0.1

    def test_from_mode_moderate(self):
        """Test from_mode with MODERATE mode."""
        config = AugmentationConfig.from_mode(AugmentationMode.MODERATE)
        assert config.mode == AugmentationMode.MODERATE
        assert config.enabled is True
        assert config.numerical_config.gaussian_noise_std == 0.1
        assert config.categorical_config.swap_prob == 0.1
        assert config.text_config.mask_prob == 0.15

    def test_from_mode_heavy(self):
        """Test from_mode with HEAVY mode."""
        config = AugmentationConfig.from_mode(AugmentationMode.HEAVY)
        assert config.mode == AugmentationMode.HEAVY
        assert config.enabled is True
        assert config.numerical_config.gaussian_noise_std == 0.2
        assert config.categorical_config.swap_prob == 0.2
        assert config.text_config.mask_prob == 0.2

    def test_from_mode_with_overrides(self):
        """Test from_mode with custom overrides."""
        config = AugmentationConfig.from_mode(
            AugmentationMode.MODERATE,
            seed=123,
        )
        assert config.mode == AugmentationMode.MODERATE
        assert config.seed == 123


