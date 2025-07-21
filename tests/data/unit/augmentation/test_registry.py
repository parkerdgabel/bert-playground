"""Tests for augmentation registry and manager."""

import pytest

from data.augmentation.base import (
    BaseAugmentationStrategy,
    FeatureMetadata,
    FeatureType,
)
from data.augmentation.config import AugmentationConfig, AugmentationMode
from data.augmentation.registry import (
    AugmentationManager,
    AugmentationRegistry,
    AugmenterInfo,
    get_registry,
)


class MockStrategy(BaseAugmentationStrategy):
    """Mock strategy for testing."""
    
    def __init__(self, name="mock"):
        super().__init__(name, [FeatureType.NUMERICAL])
    
    def apply(self, data, config):
        return data * 2


class TestAugmenterInfo:
    """Test AugmenterInfo dataclass."""

    def test_creation(self):
        """Test creating augmenter info."""
        info = AugmenterInfo(
            name="test_augmenter",
            description="Test augmenter for unit tests",
            supported_types=[FeatureType.NUMERICAL, FeatureType.TEXT],
            parameters={
                "factor": "Multiplication factor",
                "seed": "Random seed",
            },
        )
        
        assert info.name == "test_augmenter"
        assert info.description == "Test augmenter for unit tests"
        assert len(info.supported_types) == 2
        assert "factor" in info.parameters

    def test_default_parameters(self):
        """Test default empty parameters."""
        info = AugmenterInfo(
            name="simple",
            description="Simple augmenter",
            supported_types=[FeatureType.CATEGORICAL],
        )
        assert info.parameters == {}


class TestAugmentationRegistry:
    """Test AugmentationRegistry class."""

    def test_singleton_behavior(self):
        """Test that registry is a singleton."""
        reg1 = get_registry()
        reg2 = get_registry()
        assert reg1 is reg2

    def test_register_strategy(self):
        """Test registering a strategy."""
        registry = AugmentationRegistry()
        strategy = MockStrategy("test_strategy")
        
        registry.register_strategy("test_strategy", strategy)
        
        assert "test_strategy" in registry.list_strategies()
        retrieved = registry.get_strategy("test_strategy")
        assert retrieved == strategy

    def test_register_duplicate_error(self):
        """Test error when registering duplicate strategy."""
        registry = AugmentationRegistry()
        strategy = MockStrategy()
        
        registry.register_strategy("duplicate", strategy)
        
        with pytest.raises(ValueError, match="already registered"):
            registry.register_strategy("duplicate", strategy)

    def test_get_nonexistent_strategy(self):
        """Test getting non-existent strategy."""
        registry = AugmentationRegistry()
        
        with pytest.raises(KeyError, match="not found"):
            registry.get_strategy("nonexistent")

    def test_register_augmenter(self):
        """Test registering augmenter info."""
        registry = AugmentationRegistry()
        
        info = AugmenterInfo(
            name="custom_augmenter",
            description="Custom augmenter",
            supported_types=[FeatureType.TEXT],
        )
        
        registry.register_augmenter(info)
        
        assert "custom_augmenter" in registry.list_augmenters()
        retrieved = registry.get_augmenter_info("custom_augmenter")
        assert retrieved == info

    def test_list_by_type(self):
        """Test listing strategies by feature type."""
        registry = AugmentationRegistry()
        
        # Register strategies for different types
        num_strategy = MockStrategy("num_strat")
        num_strategy.supported_types = [FeatureType.NUMERICAL]
        registry.register_strategy("num_strat", num_strategy)
        
        text_strategy = MockStrategy("text_strat")
        text_strategy.supported_types = [FeatureType.TEXT]
        registry.register_strategy("text_strat", text_strategy)
        
        multi_strategy = MockStrategy("multi_strat")
        multi_strategy.supported_types = [FeatureType.NUMERICAL, FeatureType.TEXT]
        registry.register_strategy("multi_strat", multi_strategy)
        
        # Test filtering
        num_strategies = registry.list_strategies_by_type(FeatureType.NUMERICAL)
        assert "num_strat" in num_strategies
        assert "multi_strat" in num_strategies
        assert "text_strat" not in num_strategies
        
        text_strategies = registry.list_strategies_by_type(FeatureType.TEXT)
        assert "text_strat" in text_strategies
        assert "multi_strat" in text_strategies
        assert "num_strat" not in text_strategies

    def test_default_strategies(self):
        """Test that default strategies are registered."""
        registry = get_registry()
        
        # Check some default strategies exist
        strategies = registry.list_strategies()
        assert "gaussian_noise" in strategies
        assert "scaling" in strategies
        assert "synonym_replacement" in strategies
        assert "masking" in strategies


class TestAugmentationManager:
    """Test AugmentationManager class."""

    def test_initialization(self):
        """Test manager initialization."""
        feature_metadata = {
            "feature1": FeatureMetadata("feature1", FeatureType.NUMERICAL),
            "feature2": FeatureMetadata("feature2", FeatureType.CATEGORICAL),
        }
        
        config = AugmentationConfig.from_mode(AugmentationMode.LIGHT)
        manager = AugmentationManager(feature_metadata, config)
        
        assert manager.feature_metadata == feature_metadata
        assert manager.config == config
        assert manager._augmenter is not None

    def test_augment_sample(self):
        """Test augmenting a single sample."""
        feature_metadata = {
            "value": FeatureMetadata(
                "value", 
                FeatureType.NUMERICAL,
                statistics={"mean": 100, "std": 20}
            ),
        }
        
        config = AugmentationConfig.from_mode(AugmentationMode.MODERATE)
        manager = AugmentationManager(feature_metadata, config, seed=42)
        
        sample = {"value": 100}
        augmented = manager.augment(sample)
        
        assert "value" in augmented
        assert isinstance(augmented["value"], (int, float))

    def test_augment_batch(self):
        """Test augmenting a batch of samples."""
        feature_metadata = {
            "x": FeatureMetadata("x", FeatureType.NUMERICAL),
            "y": FeatureMetadata("y", FeatureType.CATEGORICAL),
        }
        
        config = AugmentationConfig.from_mode(AugmentationMode.LIGHT)
        manager = AugmentationManager(feature_metadata, config)
        
        batch = [
            {"x": 1, "y": "A"},
            {"x": 2, "y": "B"},
            {"x": 3, "y": "C"},
        ]
        
        augmented_batch = manager.augment_batch(batch)
        
        assert len(augmented_batch) == len(batch)
        for aug in augmented_batch:
            assert "x" in aug
            assert "y" in aug

    def test_get_augmentation_stats(self):
        """Test getting augmentation statistics."""
        feature_metadata = {
            "num_feat": FeatureMetadata("num_feat", FeatureType.NUMERICAL),
            "cat_feat": FeatureMetadata("cat_feat", FeatureType.CATEGORICAL),
            "text_feat": FeatureMetadata("text_feat", FeatureType.TEXT),
        }
        
        config = AugmentationConfig.from_mode(AugmentationMode.MODERATE)
        manager = AugmentationManager(feature_metadata, config)
        
        stats = manager.get_augmentation_stats()
        
        assert "total_features" in stats
        assert stats["total_features"] == 3
        assert "features_by_type" in stats
        assert stats["features_by_type"][FeatureType.NUMERICAL] == 1
        assert stats["features_by_type"][FeatureType.CATEGORICAL] == 1
        assert stats["features_by_type"][FeatureType.TEXT] == 1
        assert "augmentation_mode" in stats
        assert stats["augmentation_mode"] == "MODERATE"
        assert "augmentation_enabled" in stats
        assert stats["augmentation_enabled"] is True

    def test_update_config(self):
        """Test updating configuration."""
        feature_metadata = {
            "feat": FeatureMetadata("feat", FeatureType.NUMERICAL),
        }
        
        manager = AugmentationManager(
            feature_metadata,
            AugmentationConfig.from_mode(AugmentationMode.LIGHT)
        )
        
        # Check initial config
        assert manager.config.mode == AugmentationMode.LIGHT
        
        # Update config
        new_config = AugmentationConfig.from_mode(AugmentationMode.HEAVY)
        manager.update_config(new_config)
        
        assert manager.config.mode == AugmentationMode.HEAVY

    def test_enable_disable(self):
        """Test enabling and disabling augmentation."""
        feature_metadata = {
            "feat": FeatureMetadata("feat", FeatureType.NUMERICAL),
        }
        
        manager = AugmentationManager(
            feature_metadata,
            AugmentationConfig.from_mode(AugmentationMode.MODERATE)
        )
        
        # Initially enabled
        assert manager.config.enabled is True
        
        # Disable
        manager.disable()
        assert manager.config.enabled is False
        
        # Re-enable
        manager.enable()
        assert manager.config.enabled is True

    def test_custom_strategies(self):
        """Test using custom strategies."""
        feature_metadata = {
            "custom_feat": FeatureMetadata("custom_feat", FeatureType.NUMERICAL),
        }
        
        # Register custom strategy
        registry = get_registry()
        custom_strategy = MockStrategy("custom_multiplier")
        registry.register_strategy("custom_multiplier", custom_strategy)
        
        # Use in manager
        config = AugmentationConfig(
            strategies={"numerical": ["custom_multiplier"]}
        )
        manager = AugmentationManager(feature_metadata, config)
        
        sample = {"custom_feat": 10}
        augmented = manager.augment(sample)
        
        # Custom strategy multiplies by 2
        assert augmented["custom_feat"] == 20