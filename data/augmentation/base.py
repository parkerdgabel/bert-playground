"""
Base classes and types for data augmentation.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import mlx.core as mx


class FeatureType(Enum):
    """Types of features for augmentation."""

    NUMERICAL = "numerical"
    CATEGORICAL = "categorical"
    TEXT = "text"
    DATE = "date"
    BOOLEAN = "boolean"
    UNKNOWN = "unknown"


class AugmentationMode(Enum):
    """Augmentation modes."""

    NONE = "none"
    LIGHT = "light"  # Minimal augmentation
    MODERATE = "moderate"  # Standard augmentation
    HEAVY = "heavy"  # Aggressive augmentation
    CUSTOM = "custom"  # User-defined settings


@dataclass
class FeatureMetadata:
    """Metadata for a feature."""

    name: str
    feature_type: FeatureType
    importance: float = 1.0  # Feature importance for augmentation
    constraints: dict[str, Any] = field(default_factory=dict)
    statistics: dict[str, float] = field(default_factory=dict)
    domain_info: dict[str, Any] = field(default_factory=dict)


@dataclass
class AugmentationResult:
    """Result of augmentation operation."""

    original: Any
    augmented: Any
    strategy_used: str
    parameters: dict[str, Any]
    success: bool = True
    error: str | None = None


class BaseAugmenter(ABC):
    """Base class for all augmenters."""

    def __init__(self, seed: int | None = None):
        """
        Initialize augmenter.

        Args:
            seed: Random seed for reproducibility
        """
        self.seed = seed
        self._rng = None
        if seed is not None:
            self.set_seed(seed)

    @abstractmethod
    def augment(self, data: Any, **kwargs) -> Any:
        """Augment a single data sample."""
        pass

    def augment_batch(self, batch: list[Any], **kwargs) -> list[Any]:
        """Augment a batch of data samples."""
        return [self.augment(sample, **kwargs) for sample in batch]

    def get_config(self) -> dict[str, Any]:
        """Get augmentation configuration."""
        return {
            "class": self.__class__.__name__,
            "seed": self.seed,
        }

    def set_seed(self, seed: int) -> None:
        """Set random seed for reproducibility."""
        self.seed = seed
        mx.random.seed(seed)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(seed={self.seed})"


class BaseFeatureAugmenter(ABC):
    """Base class for feature-specific augmenters."""

    def __init__(self, supported_types: list[FeatureType]):
        """
        Initialize feature augmenter.

        Args:
            supported_types: List of supported feature types
        """
        self.supported_types = supported_types

    @abstractmethod
    def augment_feature(
        self,
        value: Any,
        feature_name: str,
        feature_type: FeatureType,
        metadata: FeatureMetadata | None = None,
        **kwargs,
    ) -> Any:
        """
        Augment a single feature value.

        Args:
            value: Feature value to augment
            feature_name: Name of the feature
            feature_type: Type of the feature
            metadata: Optional feature metadata
            **kwargs: Additional arguments

        Returns:
            Augmented feature value
        """
        pass

    def can_augment(self, feature_type: FeatureType) -> bool:
        """Check if this augmenter can handle the feature type."""
        return feature_type in self.supported_types

    def __repr__(self) -> str:
        types = [t.value for t in self.supported_types]
        return f"{self.__class__.__name__}(types={types})"


class BaseAugmentationStrategy(ABC):
    """Base class for augmentation strategies."""

    def __init__(self, name: str, supported_types: list[FeatureType]):
        """
        Initialize strategy.

        Args:
            name: Strategy name
            supported_types: List of supported feature types
        """
        self._name = name
        self._supported_types = supported_types

    @abstractmethod
    def apply(self, data: Any, config: dict[str, Any]) -> Any:
        """Apply augmentation strategy to data."""
        pass

    @property
    def name(self) -> str:
        """Strategy name."""
        return self._name

    @property
    def supported_types(self) -> list[str]:
        """List of supported data types."""
        return [t.value for t in self._supported_types]

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}')"


class ComposeAugmenter(BaseAugmenter):
    """Compose multiple augmenters."""

    def __init__(self, augmenters: list[BaseAugmenter], apply_probability: float = 1.0):
        """
        Initialize composition.

        Args:
            augmenters: List of augmenters to compose
            apply_probability: Probability of applying augmentation
        """
        super().__init__()
        self.augmenters = augmenters
        self.apply_probability = apply_probability

    def augment(self, data: Any, **kwargs) -> Any:
        """Apply augmenters in sequence."""
        if mx.random.uniform() > self.apply_probability:
            return data

        result = data
        for augmenter in self.augmenters:
            result = augmenter.augment(result, **kwargs)

        return result

    def get_config(self) -> dict[str, Any]:
        """Get configuration."""
        config = super().get_config()
        config.update(
            {
                "augmenters": [aug.get_config() for aug in self.augmenters],
                "apply_probability": self.apply_probability,
            }
        )
        return config


class ConditionalAugmenter(BaseAugmenter):
    """Apply augmentation based on conditions."""

    def __init__(self, augmenter: BaseAugmenter, condition_fn: callable):
        """
        Initialize conditional augmenter.

        Args:
            augmenter: Base augmenter to apply
            condition_fn: Function that returns True if augmentation should be applied
        """
        super().__init__()
        self.augmenter = augmenter
        self.condition_fn = condition_fn

    def augment(self, data: Any, **kwargs) -> Any:
        """Apply augmentation if condition is met."""
        if self.condition_fn(data, **kwargs):
            return self.augmenter.augment(data, **kwargs)
        return data
