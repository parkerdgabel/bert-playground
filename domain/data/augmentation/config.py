"""
Configuration for data augmentation.
"""

from dataclasses import dataclass, field
from typing import Any

from .base import AugmentationMode, FeatureMetadata


@dataclass
class NumericalAugmentationConfig:
    """Configuration for numerical feature augmentation."""

    # Noise addition
    gaussian_noise_std: float = 0.1
    apply_noise_prob: float = 0.5

    # Scaling
    scale_range: tuple = (0.9, 1.1)
    apply_scaling_prob: float = 0.3

    # Binning/discretization
    use_binning: bool = False
    bin_edges: list[float] | None = None

    # Outlier handling
    clip_outliers: bool = True
    outlier_std_threshold: float = 3.0

    # Feature-specific bounds
    min_value: float | None = None
    max_value: float | None = None


@dataclass
class CategoricalAugmentationConfig:
    """Configuration for categorical feature augmentation."""

    # Value swapping
    swap_prob: float = 0.1
    swap_with_similar: bool = True  # Swap with similar categories

    # Synonym mapping
    use_synonyms: bool = True
    synonym_map: dict[str, list[str]] = field(default_factory=dict)

    # Noise injection
    add_typos: bool = False
    typo_prob: float = 0.05

    # Unknown category handling
    unknown_token: str = "[UNKNOWN]"
    add_unknown_prob: float = 0.05


@dataclass
class TextAugmentationConfig:
    """Configuration for text feature augmentation."""

    # Token-level
    mask_prob: float = 0.15
    random_token_prob: float = 0.1
    delete_prob: float = 0.1

    # Word-level
    synonym_prob: float = 0.1
    use_wordnet: bool = False

    # Sentence-level
    sentence_shuffle_prob: float = 0.1
    sentence_drop_prob: float = 0.05

    # Advanced
    use_back_translation: bool = False
    use_paraphrasing: bool = False
    use_contextual_embeddings: bool = False

    # BERT-specific
    mask_token: str = "[MASK]"
    preserve_special_tokens: bool = True


@dataclass
class DomainKnowledgeConfig:
    """Configuration for domain-specific knowledge."""

    # Feature relationships
    correlated_features: dict[str, list[str]] = field(default_factory=dict)

    # Constraints
    feature_constraints: dict[str, dict[str, Any]] = field(default_factory=dict)

    # Business rules
    validation_rules: list[callable] = field(default_factory=list)

    # Feature importance
    feature_importance: dict[str, float] = field(default_factory=dict)


@dataclass
class AugmentationConfig:
    """Main augmentation configuration."""

    # General settings
    enabled: bool = True
    mode: AugmentationMode = AugmentationMode.MODERATE
    seed: int = 42

    # Augmentation probability
    augmentation_prob: float = 0.5  # Probability of augmenting a sample

    # Feature metadata
    feature_metadata: dict[str, FeatureMetadata] = field(default_factory=dict)

    # Type-specific configurations
    numerical_config: NumericalAugmentationConfig = field(
        default_factory=NumericalAugmentationConfig
    )
    categorical_config: CategoricalAugmentationConfig = field(
        default_factory=CategoricalAugmentationConfig
    )
    text_config: TextAugmentationConfig = field(default_factory=TextAugmentationConfig)

    # Domain knowledge
    domain_config: DomainKnowledgeConfig | None = None

    # Strategy selection
    strategy_weights: dict[str, float] = field(default_factory=dict)

    # Caching
    cache_augmented_samples: bool = True
    cache_size: int = 10000

    # MLX-specific
    use_mlx_operations: bool = True
    device: str = "cpu"  # MLX automatically uses GPU when available

    @classmethod
    def from_mode(cls, mode: AugmentationMode, **kwargs) -> "AugmentationConfig":
        """Create configuration from preset mode."""
        if mode == AugmentationMode.NONE:
            return cls(enabled=False, **kwargs)
        elif mode == AugmentationMode.LIGHT:
            return cls(
                augmentation_prob=0.3,
                numerical_config=NumericalAugmentationConfig(
                    gaussian_noise_std=0.05,
                    apply_noise_prob=0.3,
                ),
                categorical_config=CategoricalAugmentationConfig(
                    swap_prob=0.05,
                ),
                text_config=TextAugmentationConfig(
                    mask_prob=0.1,
                    random_token_prob=0.05,
                ),
                **kwargs,
            )
        elif mode == AugmentationMode.HEAVY:
            return cls(
                augmentation_prob=0.8,
                numerical_config=NumericalAugmentationConfig(
                    gaussian_noise_std=0.2,
                    apply_noise_prob=0.7,
                    apply_scaling_prob=0.5,
                ),
                categorical_config=CategoricalAugmentationConfig(
                    swap_prob=0.2,
                    add_unknown_prob=0.1,
                ),
                text_config=TextAugmentationConfig(
                    mask_prob=0.2,
                    random_token_prob=0.15,
                    sentence_shuffle_prob=0.2,
                ),
                **kwargs,
            )
        else:  # MODERATE (default)
            return cls(**kwargs)


# Legacy config for backward compatibility
@dataclass
class BERTAugmentationConfig(TextAugmentationConfig):
    """Legacy configuration for BERT text augmentation."""

    # Additional tabular-specific settings
    feature_noise_std: float = 0.1
    feature_swap_prob: float = 0.1

    def to_augmentation_config(self) -> AugmentationConfig:
        """Convert to new configuration format."""
        return AugmentationConfig(
            text_config=TextAugmentationConfig(
                mask_prob=self.mask_prob,
                random_token_prob=self.random_token_prob,
                delete_prob=self.delete_prob,
                synonym_prob=self.synonym_prob,
                sentence_shuffle_prob=self.sentence_shuffle_prob,
                use_back_translation=self.use_back_translation,
                use_paraphrasing=self.use_paraphrasing,
            ),
            numerical_config=NumericalAugmentationConfig(
                gaussian_noise_std=self.feature_noise_std,
            ),
            categorical_config=CategoricalAugmentationConfig(
                swap_prob=self.feature_swap_prob,
            ),
        )
