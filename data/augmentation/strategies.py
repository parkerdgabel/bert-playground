"""
Feature-specific augmentation strategies.
"""

from typing import Any

import mlx.core as mx

from .base import (
    BaseAugmentationStrategy,
    BaseFeatureAugmenter,
    FeatureMetadata,
    FeatureType,
)
from .config import (
    CategoricalAugmentationConfig,
    NumericalAugmentationConfig,
    TextAugmentationConfig,
)


class NumericalAugmenter(BaseFeatureAugmenter):
    """Augmenter for numerical features."""

    def __init__(self, config: NumericalAugmentationConfig | None = None):
        """Initialize numerical augmenter."""
        super().__init__([FeatureType.NUMERICAL])
        self.config = config or NumericalAugmentationConfig()

    def augment_feature(
        self,
        value: Any,
        feature_name: str,
        feature_type: FeatureType,
        metadata: FeatureMetadata | None = None,
        **kwargs,
    ) -> Any:
        """Augment numerical feature value."""
        if not isinstance(value, (int, float)) or mx.isnan(value):
            return value

        # Convert to MLX array for operations
        value_mx = mx.array(float(value))
        augmented = value_mx

        # Apply Gaussian noise
        if mx.random.uniform() < self.config.apply_noise_prob:
            noise_std = self.config.gaussian_noise_std

            # Adjust noise based on feature statistics if available
            if metadata and metadata.statistics:
                feature_std = metadata.statistics.get("std", 1.0)
                noise_std *= feature_std

            noise = mx.random.normal(shape=(), scale=noise_std)
            augmented = augmented + noise

        # Apply scaling
        if mx.random.uniform() < self.config.apply_scaling_prob:
            scale = mx.random.uniform(
                low=self.config.scale_range[0], high=self.config.scale_range[1]
            )
            augmented = augmented * scale

        # Apply bounds if specified
        if self.config.min_value is not None:
            augmented = mx.maximum(augmented, mx.array(self.config.min_value))
        if self.config.max_value is not None:
            augmented = mx.minimum(augmented, mx.array(self.config.max_value))

        # Apply outlier clipping if enabled
        if self.config.clip_outliers and metadata and metadata.statistics:
            mean = metadata.statistics.get("mean", 0)
            std = metadata.statistics.get("std", 1)
            threshold = self.config.outlier_std_threshold

            lower_bound = mean - threshold * std
            upper_bound = mean + threshold * std
            augmented = mx.clip(augmented, lower_bound, upper_bound)

        return float(augmented.item())


class CategoricalAugmenter(BaseFeatureAugmenter):
    """Augmenter for categorical features."""

    def __init__(self, config: CategoricalAugmentationConfig | None = None):
        """Initialize categorical augmenter."""
        super().__init__([FeatureType.CATEGORICAL])
        self.config = config or CategoricalAugmentationConfig()

    def augment_feature(
        self,
        value: Any,
        feature_name: str,
        feature_type: FeatureType,
        metadata: FeatureMetadata | None = None,
        **kwargs,
    ) -> Any:
        """Augment categorical feature value."""
        if value is None or (isinstance(value, str) and not value):
            return value

        value_str = str(value)

        # Add unknown token
        if mx.random.uniform() < self.config.add_unknown_prob:
            return self.config.unknown_token

        # Swap with synonym if available
        if (
            self.config.use_synonyms
            and value_str in self.config.synonym_map
            and mx.random.uniform() < self.config.swap_prob
        ):
            synonyms = self.config.synonym_map[value_str]
            if synonyms:
                idx = int(mx.random.uniform(0, len(synonyms)))
                return synonyms[idx]

        # Swap with similar category if available
        if (
            self.config.swap_with_similar
            and metadata
            and metadata.domain_info.get("similar_values")
            and mx.random.uniform() < self.config.swap_prob
        ):
            similar_values = metadata.domain_info["similar_values"].get(value_str, [])
            if similar_values:
                idx = int(mx.random.uniform(0, len(similar_values)))
                return similar_values[idx]

        # Add typos if enabled
        if self.config.add_typos and mx.random.uniform() < self.config.typo_prob:
            return self._add_typo(value_str)

        return value

    def _add_typo(self, text: str) -> str:
        """Add simple typo to text."""
        if len(text) < 2:
            return text

        # Simple typo: swap two adjacent characters
        pos = int(mx.random.uniform(0, len(text) - 1))
        chars = list(text)
        chars[pos], chars[pos + 1] = chars[pos + 1], chars[pos]
        return "".join(chars)


class TextFeatureAugmenter(BaseFeatureAugmenter):
    """Augmenter for text features (not full documents)."""

    def __init__(self, config: TextAugmentationConfig | None = None):
        """Initialize text feature augmenter."""
        super().__init__([FeatureType.TEXT])
        self.config = config or TextAugmentationConfig()

    def augment_feature(
        self,
        value: Any,
        feature_name: str,
        feature_type: FeatureType,
        metadata: FeatureMetadata | None = None,
        **kwargs,
    ) -> Any:
        """Augment text feature value."""
        if not isinstance(value, str) or not value:
            return value

        # For short text features, apply simple augmentations
        words = value.split()

        # Word deletion
        if len(words) > 1 and mx.random.uniform() < self.config.delete_prob:
            idx = int(mx.random.uniform(0, len(words)))
            words.pop(idx)

        # Word masking (for BERT)
        if self.config.mask_prob > 0:
            for i in range(len(words)):
                if mx.random.uniform() < self.config.mask_prob:
                    words[i] = self.config.mask_token

        return " ".join(words)


class DateAugmenter(BaseFeatureAugmenter):
    """Augmenter for date features."""

    def __init__(self):
        """Initialize date augmenter."""
        super().__init__([FeatureType.DATE])

    def augment_feature(
        self,
        value: Any,
        feature_name: str,
        feature_type: FeatureType,
        metadata: FeatureMetadata | None = None,
        **kwargs,
    ) -> Any:
        """Augment date feature value."""
        # For now, dates are not augmented
        # Could add time shifts, format changes, etc.
        return value


class GaussianNoiseStrategy(BaseAugmentationStrategy):
    """Gaussian noise augmentation strategy."""

    def __init__(self, std: float = 0.1):
        """Initialize strategy."""
        super().__init__("gaussian_noise", [FeatureType.NUMERICAL])
        self.std = std

    def apply(self, data: Any, config: dict[str, Any]) -> Any:
        """Apply Gaussian noise."""
        if not isinstance(data, (int, float)):
            return data

        value = mx.array(float(data))
        noise = mx.random.normal(shape=(), scale=self.std)
        return float((value + noise).item())


class ScalingStrategy(BaseAugmentationStrategy):
    """Scaling augmentation strategy."""

    def __init__(self, scale_range: tuple = (0.9, 1.1)):
        """Initialize strategy."""
        super().__init__("scaling", [FeatureType.NUMERICAL])
        self.scale_range = scale_range

    def apply(self, data: Any, config: dict[str, Any]) -> Any:
        """Apply scaling."""
        if not isinstance(data, (int, float)):
            return data

        value = mx.array(float(data))
        scale = mx.random.uniform(low=self.scale_range[0], high=self.scale_range[1])
        return float((value * scale).item())


class SynonymReplacementStrategy(BaseAugmentationStrategy):
    """Synonym replacement strategy."""

    def __init__(self, synonym_map: dict[str, list[str]] = None):
        """Initialize strategy."""
        super().__init__(
            "synonym_replacement", [FeatureType.CATEGORICAL, FeatureType.TEXT]
        )
        self.synonym_map = synonym_map or {}

    def apply(self, data: Any, config: dict[str, Any]) -> Any:
        """Apply synonym replacement."""
        if not isinstance(data, str):
            return data

        if data in self.synonym_map:
            synonyms = self.synonym_map[data]
            if synonyms:
                idx = int(mx.random.uniform(0, len(synonyms)))
                return synonyms[idx]

        return data


class MaskingStrategy(BaseAugmentationStrategy):
    """Token masking strategy for text."""

    def __init__(self, mask_token: str = "[MASK]", mask_prob: float = 0.15):
        """Initialize strategy."""
        super().__init__("masking", [FeatureType.TEXT])
        self.mask_token = mask_token
        self.mask_prob = mask_prob

    def apply(self, data: Any, config: dict[str, Any]) -> Any:
        """Apply masking."""
        if not isinstance(data, str):
            return data

        words = data.split()
        for i in range(len(words)):
            if mx.random.uniform() < self.mask_prob:
                words[i] = self.mask_token

        return " ".join(words)


def create_augmenter_for_type(
    feature_type: FeatureType, config: dict[str, Any] | None = None
) -> BaseFeatureAugmenter:
    """
    Factory function to create appropriate augmenter for feature type.

    Args:
        feature_type: Type of feature
        config: Optional configuration

    Returns:
        Feature augmenter instance
    """
    if feature_type == FeatureType.NUMERICAL:
        num_config = NumericalAugmentationConfig(**config) if config else None
        return NumericalAugmenter(num_config)
    elif feature_type == FeatureType.CATEGORICAL:
        cat_config = CategoricalAugmentationConfig(**config) if config else None
        return CategoricalAugmenter(cat_config)
    elif feature_type == FeatureType.TEXT:
        text_config = TextAugmentationConfig(**config) if config else None
        return TextFeatureAugmenter(text_config)
    elif feature_type == FeatureType.DATE:
        return DateAugmenter()
    else:
        raise ValueError(f"No augmenter available for type: {feature_type}")
