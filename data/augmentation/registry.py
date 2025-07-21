"""
Augmentation registry and manager.
"""

from dataclasses import dataclass
from typing import Any

from loguru import logger

from .base import (
    BaseAugmentationStrategy,
    BaseAugmenter,
    BaseFeatureAugmenter,
    FeatureMetadata,
    FeatureType,
)
from .strategies import (
    CategoricalAugmenter,
    DateAugmenter,
    GaussianNoiseStrategy,
    MaskingStrategy,
    NumericalAugmenter,
    ScalingStrategy,
    SynonymReplacementStrategy,
    TextFeatureAugmenter,
)


@dataclass
class AugmenterInfo:
    """Information about a registered augmenter."""

    name: str
    augmenter_class: type[BaseAugmenter]
    supported_types: list[FeatureType]
    description: str
    config_class: type | None = None


class AugmentationRegistry:
    """
    Central registry for augmentation strategies.

    Manages registration and creation of augmenters and strategies.
    """

    def __init__(self):
        """Initialize registry."""
        self._augmenters: dict[str, AugmenterInfo] = {}
        self._strategies: dict[str, BaseAugmentationStrategy] = {}
        self._feature_augmenters: dict[FeatureType, type[BaseFeatureAugmenter]] = {}

        # Register default augmenters
        self._register_defaults()

    def _register_defaults(self):
        """Register default augmenters and strategies."""
        # Feature augmenters
        self._feature_augmenters[FeatureType.NUMERICAL] = NumericalAugmenter
        self._feature_augmenters[FeatureType.CATEGORICAL] = CategoricalAugmenter
        self._feature_augmenters[FeatureType.TEXT] = TextFeatureAugmenter
        self._feature_augmenters[FeatureType.DATE] = DateAugmenter

        # Strategies
        self.register_strategy("gaussian_noise", GaussianNoiseStrategy())
        self.register_strategy("scaling", ScalingStrategy())
        self.register_strategy("synonym_replacement", SynonymReplacementStrategy())
        self.register_strategy("masking", MaskingStrategy())

    def register_augmenter(self, name: str, info: AugmenterInfo):
        """
        Register an augmenter.

        Args:
            name: Unique name for the augmenter
            info: Augmenter information
        """
        if name in self._augmenters:
            logger.warning(f"Overwriting existing augmenter: {name}")

        self._augmenters[name] = info
        logger.info(f"Registered augmenter: {name}")

    def register_strategy(self, name: str, strategy: BaseAugmentationStrategy):
        """
        Register an augmentation strategy.

        Args:
            name: Unique name for the strategy
            strategy: Strategy instance
        """
        if name in self._strategies:
            logger.warning(f"Overwriting existing strategy: {name}")

        self._strategies[name] = strategy
        logger.info(f"Registered strategy: {name}")

    def register_feature_augmenter(
        self, feature_type: FeatureType, augmenter_class: type[BaseFeatureAugmenter]
    ):
        """
        Register a feature augmenter.

        Args:
            feature_type: Feature type
            augmenter_class: Augmenter class
        """
        self._feature_augmenters[feature_type] = augmenter_class
        logger.info(f"Registered feature augmenter for {feature_type.value}")

    def get_augmenter(self, name: str, **kwargs) -> BaseAugmenter:
        """
        Get an augmenter instance by name.

        Args:
            name: Augmenter name
            **kwargs: Arguments for augmenter initialization

        Returns:
            Augmenter instance
        """
        if name not in self._augmenters:
            raise ValueError(f"Unknown augmenter: {name}")

        info = self._augmenters[name]
        return info.augmenter_class(**kwargs)

    def get_strategy(self, name: str) -> BaseAugmentationStrategy:
        """
        Get a strategy by name.

        Args:
            name: Strategy name

        Returns:
            Strategy instance
        """
        if name not in self._strategies:
            raise ValueError(f"Unknown strategy: {name}")

        return self._strategies[name]

    def get_feature_augmenter(
        self, feature_type: FeatureType, **kwargs
    ) -> BaseFeatureAugmenter:
        """
        Get a feature augmenter for the given type.

        Args:
            feature_type: Feature type
            **kwargs: Arguments for augmenter initialization

        Returns:
            Feature augmenter instance
        """
        if feature_type not in self._feature_augmenters:
            raise ValueError(f"No augmenter for type: {feature_type}")

        augmenter_class = self._feature_augmenters[feature_type]
        return augmenter_class(**kwargs)

    def list_augmenters(self) -> list[str]:
        """List all registered augmenters."""
        return list(self._augmenters.keys())

    def list_strategies(self) -> list[str]:
        """List all registered strategies."""
        return list(self._strategies.keys())

    def list_feature_types(self) -> list[FeatureType]:
        """List supported feature types."""
        return list(self._feature_augmenters.keys())


# Global registry instance
_registry = AugmentationRegistry()


# Initialize default strategies
def _initialize_default_strategies():
    """Initialize default augmentation strategies in the registry."""
    # Register built-in strategies
    _registry.register_strategy("gaussian_noise", GaussianNoiseStrategy())
    _registry.register_strategy("scaling", ScalingStrategy())
    _registry.register_strategy("synonym_replacement", SynonymReplacementStrategy())
    _registry.register_strategy("masking", MaskingStrategy())

    # Register competition-specific strategies
    try:
        from .competition_strategies import (
            CompetitionTemplateAugmenter,
            TitanicAugmenter,
        )

        _registry.register_strategy("titanic", TitanicAugmenter())
        # Note: CompetitionTemplateAugmenter requires initialization with specific parameters
        logger.debug("Registered competition-specific strategies")
    except ImportError:
        logger.debug("Competition strategies not available")


# Initialize on module load
_initialize_default_strategies()


def get_registry() -> AugmentationRegistry:
    """Get the global augmentation registry."""
    return _registry


class AugmentationManager:
    """
    Manages augmentation for datasets.

    Coordinates multiple augmenters and strategies based on
    feature metadata and configuration.
    """

    def __init__(
        self,
        feature_metadata: dict[str, FeatureMetadata],
        config: Any | None = None,
        registry: AugmentationRegistry | None = None,
    ):
        """
        Initialize manager.

        Args:
            feature_metadata: Metadata for each feature
            config: Augmentation configuration
            registry: Augmentation registry (uses global if None)
        """
        self.feature_metadata = feature_metadata
        self.config = config
        self.registry = registry or get_registry()

        # Create feature augmenters
        self._create_augmenters()

    def _create_augmenters(self):
        """Create augmenters for each feature type."""
        self.feature_augmenters = {}

        # Group features by type
        type_groups = {}
        for feature_name, metadata in self.feature_metadata.items():
            feature_type = metadata.feature_type
            if feature_type not in type_groups:
                type_groups[feature_type] = []
            type_groups[feature_type].append(feature_name)

        # Create augmenter for each type
        for feature_type, feature_names in type_groups.items():
            try:
                augmenter = self.registry.get_feature_augmenter(feature_type)
                self.feature_augmenters[feature_type] = augmenter
                logger.debug(
                    f"Created {feature_type.value} augmenter for features: {feature_names}"
                )
            except ValueError:
                logger.warning(f"No augmenter available for type: {feature_type}")

    def augment_sample(
        self, sample: dict[str, Any], strategies: list[str] | None = None
    ) -> dict[str, Any]:
        """
        Augment a single sample.

        Args:
            sample: Sample dictionary
            strategies: Optional list of strategies to apply

        Returns:
            Augmented sample
        """
        augmented = {}

        for feature_name, value in sample.items():
            metadata = self.feature_metadata.get(feature_name)
            if metadata is None:
                # No metadata, keep original
                augmented[feature_name] = value
                continue

            # Get augmenter for feature type
            augmenter = self.feature_augmenters.get(metadata.feature_type)
            if augmenter is None:
                # No augmenter, keep original
                augmented[feature_name] = value
                continue

            # Apply augmentation
            aug_value = augmenter.augment_feature(
                value, feature_name, metadata.feature_type, metadata
            )
            augmented[feature_name] = aug_value

        # Apply additional strategies if specified
        if strategies:
            for strategy_name in strategies:
                strategy = self.registry.get_strategy(strategy_name)
                augmented = self._apply_strategy(augmented, strategy)

        return augmented

    def _apply_strategy(
        self, sample: dict[str, Any], strategy: BaseAugmentationStrategy
    ) -> dict[str, Any]:
        """Apply a strategy to all applicable features."""
        augmented = {}

        for feature_name, value in sample.items():
            metadata = self.feature_metadata.get(feature_name)
            if metadata and metadata.feature_type.value in strategy.supported_types:
                aug_value = strategy.apply(value, {})
                augmented[feature_name] = aug_value
            else:
                augmented[feature_name] = value

        return augmented

    def augment_batch(
        self, batch: list[dict[str, Any]], strategies: list[str] | None = None
    ) -> list[dict[str, Any]]:
        """
        Augment a batch of samples.

        Args:
            batch: List of samples
            strategies: Optional list of strategies to apply

        Returns:
            List of augmented samples
        """
        return [self.augment_sample(sample, strategies) for sample in batch]

    def get_augmentation_stats(self) -> dict[str, Any]:
        """Get statistics about augmentation setup."""
        stats = {
            "num_features": len(self.feature_metadata),
            "feature_types": {},
            "augmenters": list(self.feature_augmenters.keys()),
        }

        # Count features by type
        for metadata in self.feature_metadata.values():
            feature_type = metadata.feature_type.value
            stats["feature_types"][feature_type] = (
                stats["feature_types"].get(feature_type, 0) + 1
            )

        return stats
