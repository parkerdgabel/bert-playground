"""
Tabular data augmentation for BERT models.
"""

from typing import Any, Callable

import mlx.core as mx
# from loguru import logger  # Domain should not depend on logging framework

from .base import (
    AugmentationResult,
    BaseAugmenter,
    FeatureMetadata,
    FeatureType,
)
from .config import AugmentationConfig
from .strategies import (
    CategoricalAugmenter,
    NumericalAugmenter,
    TextFeatureAugmenter,
)
from .text import BERTTextAugmenter


class TabularAugmenter(BaseAugmenter):
    """
    Generic augmentation for tabular data.

    This augmenter works with any tabular dataset by using
    feature metadata to determine appropriate augmentation strategies.
    """

    def __init__(
        self,
        config: AugmentationConfig | None = None,
        feature_metadata: dict[str, FeatureMetadata] | None = None,
    ):
        """
        Initialize tabular augmenter.

        Args:
            config: Augmentation configuration
            feature_metadata: Metadata for each feature
        """
        super().__init__(config.seed if config else None)
        self.config = config or AugmentationConfig()
        self.feature_metadata = feature_metadata or {}

        # Create feature augmenters
        self.feature_augmenters = {
            FeatureType.NUMERICAL: NumericalAugmenter(config.numerical_config),
            FeatureType.CATEGORICAL: CategoricalAugmenter(config.categorical_config),
            FeatureType.TEXT: TextFeatureAugmenter(config.text_config),
        }

    def augment(self, data: dict[str, Any], **kwargs) -> dict[str, Any]:
        """
        Augment tabular data.

        Args:
            data: Dictionary of feature values
            **kwargs: Additional arguments

        Returns:
            Augmented data dictionary
        """
        if not self.config.enabled:
            return data

        # Check augmentation probability
        if mx.random.uniform() > self.config.augmentation_prob:
            return data

        augmented = {}

        for feature_name, value in data.items():
            # Get feature metadata
            metadata = self.feature_metadata.get(feature_name)
            if metadata is None:
                # Infer feature type if no metadata
                feature_type = self._infer_feature_type(value)
                metadata = FeatureMetadata(name=feature_name, feature_type=feature_type)

            # Get appropriate augmenter
            augmenter = self.feature_augmenters.get(metadata.feature_type)

            if augmenter and augmenter.can_augment(metadata.feature_type):
                # Apply augmentation
                aug_value = augmenter.augment_feature(
                    value, feature_name, metadata.feature_type, metadata, **kwargs
                )
                augmented[feature_name] = aug_value
            else:
                # Keep original value
                augmented[feature_name] = value

        return augmented

    def augment_with_result(self, data: dict[str, Any], **kwargs) -> AugmentationResult:
        """Augment and return detailed result."""
        original = data.copy()
        augmented = self.augment(data, **kwargs)

        strategies_used = []
        for feature_name in data:
            if (
                feature_name in augmented
                and data[feature_name] != augmented[feature_name]
            ):
                metadata = self.feature_metadata.get(feature_name)
                if metadata:
                    strategies_used.append(
                        f"{feature_name}:{metadata.feature_type.value}"
                    )

        return AugmentationResult(
            original=original,
            augmented=augmented,
            strategy_used="tabular_augmentation",
            parameters={
                "strategies": strategies_used,
                "config": self.config.__class__.__name__,
            },
        )

    def _infer_feature_type(self, value: Any) -> FeatureType:
        """Infer feature type from value."""
        if value is None:
            return FeatureType.UNKNOWN
        elif isinstance(value, bool):
            return FeatureType.BOOLEAN
        elif isinstance(value, (int, float)):
            return FeatureType.NUMERICAL
        elif isinstance(value, str):
            # Simple heuristic for text vs categorical
            if len(value.split()) > 3:
                return FeatureType.TEXT
            else:
                return FeatureType.CATEGORICAL
        else:
            return FeatureType.UNKNOWN


class TabularToTextAugmenter(BaseAugmenter):
    """
    Augmentation for tabular data that will be converted to text.

    This augmenter applies augmentation at both the feature level
    and the text level after conversion.
    """

    def __init__(
        self,
        config: AugmentationConfig | None = None,
        feature_metadata: dict[str, FeatureMetadata] | None = None,
        text_converter: Callable | None = None,
        tokenizer: Any | None = None,
    ):
        """
        Initialize augmenter.

        Args:
            config: Augmentation configuration
            feature_metadata: Metadata for each feature
            text_converter: Function to convert features to text
            tokenizer: Tokenizer for text augmentation
        """
        super().__init__(config.seed if config else None)
        self.config = config or AugmentationConfig()

        # Tabular augmenter for feature-level augmentation
        self.tabular_augmenter = TabularAugmenter(config, feature_metadata)

        # Text augmenter for text-level augmentation
        if tokenizer:
            self.text_augmenter = BERTTextAugmenter(tokenizer, config.text_config)
        else:
            self.text_augmenter = None

        self.text_converter = text_converter or self._default_text_converter

    def augment(self, data: dict[str, Any], **kwargs) -> str:
        """
        Augment tabular data and convert to text.

        Args:
            data: Feature dictionary
            **kwargs: Additional arguments

        Returns:
            Augmented text representation
        """
        # Apply feature-level augmentation
        augmented_features = self.tabular_augmenter.augment(data, **kwargs)

        # Convert to text
        text = self.text_converter(augmented_features)

        # Apply text-level augmentation if available
        if self.text_augmenter:
            text = self.text_augmenter.augment_text(text, num_augmentations=0)[0]

        return text

    def augment_multiple(
        self, data: dict[str, Any], num_augmentations: int = 3, **kwargs
    ) -> list[str]:
        """
        Generate multiple augmented text representations.

        Args:
            data: Feature dictionary
            num_augmentations: Number of augmentations to generate
            **kwargs: Additional arguments

        Returns:
            List of augmented texts
        """
        texts = []

        for _ in range(num_augmentations):
            # Each augmentation gets different random features
            text = self.augment(data, **kwargs)
            texts.append(text)

        # Add original as well
        original_text = self.text_converter(data)
        if original_text not in texts:
            texts.insert(0, original_text)

        return texts

    def _default_text_converter(self, features: dict[str, Any]) -> str:
        """Default text conversion for features."""
        # Simple key-value format
        parts = []
        for key, value in features.items():
            if value is not None:
                parts.append(f"{key}: {value}")
        return " | ".join(parts)


# Legacy class for backward compatibility
class TabularBERTAugmenter(TabularToTextAugmenter):
    """
    Legacy augmenter for BERT models.

    This class is kept for backward compatibility but uses
    the new generic implementation internally.
    """

    def __init__(self, tokenizer=None, config=None):
        """Initialize with legacy interface."""
        if config and hasattr(config, "to_augmentation_config"):
            # Convert legacy config
            new_config = config.to_augmentation_config()
        else:
            new_config = AugmentationConfig()

        super().__init__(config=new_config, tokenizer=tokenizer)
        # logger.warning(
        #     "TabularBERTAugmenter is deprecated. Use TabularToTextAugmenter instead."
        # )

    def augment_tabular_text(self, text: str, features: dict[str, Any]) -> list[str]:
        """Legacy method for backward compatibility."""
        return self.augment_multiple(features, num_augmentations=3)
