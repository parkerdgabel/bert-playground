"""
Modular text conversion system for MLX dataloader.
Provides flexible and extensible text conversion capabilities.
"""

from .base_converter import BaseTextConverter, TextConversionConfig
from .template_converter import TemplateConverter, TemplateConfig
from .feature_converter import FeatureConverter, FeatureConfig
from .natural_language_converter import NaturalLanguageConverter, NLConfig
from .competition_converters import (
    CompetitionConverter,
    TitanicConverter,
    SpaceshipTitanicConverter,
    get_competition_converter,
)
from .converter_factory import TextConverterFactory, register_converter
from .augmentation import TextAugmenter, AugmentationConfig

__all__ = [
    # Base classes
    "BaseTextConverter",
    "TextConversionConfig",
    
    # Converters
    "TemplateConverter",
    "TemplateConfig",
    "FeatureConverter",
    "FeatureConfig",
    "NaturalLanguageConverter",
    "NLConfig",
    "CompetitionConverter",
    "TitanicConverter",
    "SpaceshipTitanicConverter",
    
    # Factory
    "TextConverterFactory",
    "register_converter",
    "get_competition_converter",
    
    # Augmentation
    "TextAugmenter",
    "AugmentationConfig",
]