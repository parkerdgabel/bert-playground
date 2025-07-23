"""
Data augmentation module for BERT models.
"""

# Base classes and types
from .base import (
    AugmentationMode,
    AugmentationResult,
    BaseAugmentationStrategy,
    BaseAugmenter,
    BaseFeatureAugmenter,
    ComposeAugmenter,
    ConditionalAugmenter,
    FeatureMetadata,
    FeatureType,
)

# Competition-specific strategies
from .competition_strategies import (
    CompetitionTemplateAugmenter,
    TitanicAugmenter,
)

# Configuration
from .config import (
    AugmentationConfig,
    BERTAugmentationConfig,  # Legacy
    CategoricalAugmentationConfig,
    DomainKnowledgeConfig,
    NumericalAugmentationConfig,
    TextAugmentationConfig,
)

# Registry and Manager
from .registry import (
    AugmentationManager,
    AugmentationRegistry,
    AugmenterInfo,
    get_registry,
)

# Strategies
from .strategies import (
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
from .tabular import (
    TabularAugmenter,
    TabularBERTAugmenter,  # Legacy
    TabularToTextAugmenter,
)

# Augmenters
from .text import BERTTextAugmenter
from .tta import BERTTestTimeAugmentation

__all__ = [
    # Base types
    "FeatureType",
    "AugmentationMode",
    "FeatureMetadata",
    "AugmentationResult",
    "BaseAugmenter",
    "BaseFeatureAugmenter",
    "BaseAugmentationStrategy",
    "ComposeAugmenter",
    "ConditionalAugmenter",
    # Configuration
    "AugmentationConfig",
    "NumericalAugmentationConfig",
    "CategoricalAugmentationConfig",
    "TextAugmentationConfig",
    "DomainKnowledgeConfig",
    "BERTAugmentationConfig",
    # Augmenters
    "BERTTextAugmenter",
    "TabularAugmenter",
    "TabularToTextAugmenter",
    "TabularBERTAugmenter",  # Legacy
    "BERTTestTimeAugmentation",
    # Strategies
    "NumericalAugmenter",
    "CategoricalAugmenter",
    "TextFeatureAugmenter",
    "DateAugmenter",
    "GaussianNoiseStrategy",
    "ScalingStrategy",
    "SynonymReplacementStrategy",
    "MaskingStrategy",
    "create_augmenter_for_type",
    # Registry
    "AugmenterInfo",
    "AugmentationRegistry",
    "AugmentationManager",
    "get_registry",
    # Competition strategies
    "TitanicAugmenter",
    "CompetitionTemplateAugmenter",
]
