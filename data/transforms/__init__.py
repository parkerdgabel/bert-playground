"""
Data transformation pipeline for MLX dataloader system.
Provides composable transforms for data preprocessing and augmentation.
"""

from .base_transforms import (
    Transform,
    Compose,
    Lambda,
    SelectFields,
    RenameFields,
    FilterMissing,
    FillMissing,
    Normalize,
    CategoricalEncode,
    TextClean,
    RandomNoise,
    ConditionalTransform,
)

from .text_converter import (
    BaseTextConverter,
    TemplateTextConverter,
    FeatureConcatenator,
    NaturalLanguageConverter,
    MultiModalTextConverter,
    CompetitionSpecificConverter,
    create_text_converter,
)

from .preprocessing import (
    Tokenize,
    ToMLXArray,
    PadSequence,
    CreateAttentionMask,
    LabelEncode,
    FeatureExtractor,
    DataAugmentation,
)

from .augmentation import (
    TextAugmentation,
    SynonymReplacement,
    RandomInsertion,
    RandomSwap,
    RandomDeletion,
    BackTranslation,
    MixUp,
    CutMix,
    FeatureNoise,
    EasyDataAugmentation,
)

__all__ = [
    # Base transforms
    "Transform",
    "Compose",
    "Lambda",
    "SelectFields",
    "RenameFields",
    "FilterMissing",
    "FillMissing",
    "Normalize",
    "CategoricalEncode",
    "TextClean",
    "RandomNoise",
    "ConditionalTransform",
    
    # Text converters
    "BaseTextConverter",
    "TemplateTextConverter",
    "FeatureConcatenator",
    "NaturalLanguageConverter",
    "MultiModalTextConverter",
    "CompetitionSpecificConverter",
    "create_text_converter",
    
    # Preprocessing
    "Tokenize",
    "ToMLXArray",
    "PadSequence",
    "CreateAttentionMask",
    "LabelEncode",
    "FeatureExtractor",
    "DataAugmentation",
    
    # Augmentation
    "TextAugmentation",
    "SynonymReplacement",
    "RandomInsertion",
    "RandomSwap",
    "RandomDeletion",
    "BackTranslation",
    "MixUp",
    "CutMix",
    "FeatureNoise",
    "EasyDataAugmentation",
]