"""
Transform configuration for data pipelines.
"""

from typing import Dict, Any, List, Optional, Union, Type
from dataclasses import dataclass, field
from loguru import logger

from data.transforms import (
    Transform, Compose, Lambda,
    Tokenize, ToMLXArray, PadSequence,
    TextAugmentation, SynonymReplacement,
    EasyDataAugmentation,
)
from data.text_conversion import (
    BaseTextConverter,
    TextConverterFactory,
)


@dataclass
class TransformConfig:
    """Configuration for a single transform."""
    
    name: str
    params: Dict[str, Any] = field(default_factory=dict)
    enabled: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "params": self.params,
            "enabled": self.enabled,
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "TransformConfig":
        """Create from dictionary."""
        return cls(**config_dict)


@dataclass 
class TransformPipeline:
    """Configuration for transform pipeline."""
    
    # Text conversion
    text_converter: Optional[TransformConfig] = None
    
    # Preprocessing
    preprocessing: List[TransformConfig] = field(default_factory=list)
    
    # Augmentation
    augmentation: List[TransformConfig] = field(default_factory=list)
    augmentation_prob: float = 0.5
    
    # Post-processing
    postprocessing: List[TransformConfig] = field(default_factory=list)
    
    # Caching
    cache_after: Optional[str] = None  # Cache after this transform
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "text_converter": self.text_converter.to_dict() if self.text_converter else None,
            "preprocessing": [t.to_dict() for t in self.preprocessing],
            "augmentation": [t.to_dict() for t in self.augmentation],
            "augmentation_prob": self.augmentation_prob,
            "postprocessing": [t.to_dict() for t in self.postprocessing],
            "cache_after": self.cache_after,
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "TransformPipeline":
        """Create from dictionary."""
        pipeline = cls()
        
        # Text converter
        if config_dict.get("text_converter"):
            pipeline.text_converter = TransformConfig.from_dict(config_dict["text_converter"])
        
        # Preprocessing
        pipeline.preprocessing = [
            TransformConfig.from_dict(t) for t in config_dict.get("preprocessing", [])
        ]
        
        # Augmentation
        pipeline.augmentation = [
            TransformConfig.from_dict(t) for t in config_dict.get("augmentation", [])
        ]
        pipeline.augmentation_prob = config_dict.get("augmentation_prob", 0.5)
        
        # Post-processing
        pipeline.postprocessing = [
            TransformConfig.from_dict(t) for t in config_dict.get("postprocessing", [])
        ]
        
        pipeline.cache_after = config_dict.get("cache_after")
        
        return pipeline
    
    def build(self, tokenizer: Any = None) -> Transform:
        """
        Build the transform pipeline.
        
        Args:
            tokenizer: Tokenizer instance (required for tokenization transforms)
            
        Returns:
            Composed transform
        """
        transforms = []
        
        # Add text converter
        if self.text_converter and self.text_converter.enabled:
            converter = self._create_text_converter(self.text_converter)
            transforms.append(converter)
        
        # Add preprocessing
        for config in self.preprocessing:
            if config.enabled:
                transform = self._create_transform(config, tokenizer)
                transforms.append(transform)
        
        # Add augmentation (wrapped in conditional)
        if self.augmentation:
            aug_transforms = []
            for config in self.augmentation:
                if config.enabled:
                    transform = self._create_transform(config, tokenizer)
                    aug_transforms.append(transform)
            
            if aug_transforms:
                # Wrap in data augmentation transform
                from data.transforms import DataAugmentation
                aug = DataAugmentation(
                    augmentations=aug_transforms,
                    probability=self.augmentation_prob,
                    num_augmentations=1,
                )
                transforms.append(aug)
        
        # Add post-processing
        for config in self.postprocessing:
            if config.enabled:
                transform = self._create_transform(config, tokenizer)
                transforms.append(transform)
        
        # Return composed pipeline
        return Compose(transforms) if len(transforms) > 1 else transforms[0] if transforms else None
    
    def _create_text_converter(self, config: TransformConfig) -> Transform:
        """Create text converter transform."""
        # Get converter name and params
        converter_name = config.params.get("strategy", config.name)
        converter_params = {k: v for k, v in config.params.items() if k != "strategy"}
        
        # Create converter
        converter = TextConverterFactory.create(converter_name, **converter_params)
        
        # Return as transform
        return converter
    
    def _create_transform(self, config: TransformConfig, tokenizer: Any = None) -> Transform:
        """Create a transform from configuration."""
        # Map of transform names to classes
        transform_map = {
            # Basic transforms
            "lambda": Lambda,
            "compose": Compose,
            
            # Text transforms
            "tokenize": Tokenize,
            "text_augmentation": TextAugmentation,
            "synonym_replacement": SynonymReplacement,
            "eda": EasyDataAugmentation,
            
            # Array transforms
            "to_mlx_array": ToMLXArray,
            "pad_sequence": PadSequence,
        }
        
        # Get transform class
        transform_class = transform_map.get(config.name.lower())
        if transform_class is None:
            # Try to import from custom transforms
            try:
                from data.transforms import base_transforms
                transform_class = getattr(base_transforms, config.name)
            except AttributeError:
                raise ValueError(f"Unknown transform: {config.name}")
        
        # Handle special cases
        params = config.params.copy()
        
        if config.name.lower() == "tokenize" and tokenizer:
            params["tokenizer"] = tokenizer
        
        # Create transform
        return transform_class(**params)


def create_transform_pipeline(
    config: Union[Dict[str, Any], TransformPipeline],
    tokenizer: Any = None
) -> Optional[Transform]:
    """
    Create transform pipeline from configuration.
    
    Args:
        config: Pipeline configuration (dict or object)
        tokenizer: Optional tokenizer instance
        
    Returns:
        Transform pipeline or None
    """
    if isinstance(config, dict):
        pipeline = TransformPipeline.from_dict(config)
    else:
        pipeline = config
    
    return pipeline.build(tokenizer)


# Preset pipelines
PRESET_PIPELINES = {
    "basic": TransformPipeline(
        preprocessing=[
            TransformConfig("tokenize", {
                "max_length": 256,
                "padding": "max_length",
                "truncation": True,
            }),
            TransformConfig("to_mlx_array", {
                "fields": ["input_ids", "attention_mask", "label"],
            }),
        ]
    ),
    
    "augmented": TransformPipeline(
        preprocessing=[
            TransformConfig("tokenize", {
                "max_length": 256,
                "padding": "max_length",
                "truncation": True,
            }),
        ],
        augmentation=[
            TransformConfig("synonym_replacement", {
                "num_replacements": 2,
                "probability": 0.5,
            }),
            TransformConfig("eda", {
                "alpha_sr": 0.1,
                "alpha_ri": 0.1,
                "alpha_rs": 0.1,
                "alpha_rd": 0.1,
            }),
        ],
        augmentation_prob=0.3,
        postprocessing=[
            TransformConfig("to_mlx_array", {
                "fields": ["input_ids", "attention_mask", "label"],
            }),
        ]
    ),
    
    "competition": TransformPipeline(
        text_converter=TransformConfig("template", {
            "strategy": "template",
            "augment": True,
        }),
        preprocessing=[
            TransformConfig("tokenize", {
                "max_length": 256,
                "padding": "max_length",
                "truncation": True,
            }),
            TransformConfig("to_mlx_array", {
                "fields": ["input_ids", "attention_mask", "label"],
            }),
        ]
    ),
}


def get_preset_pipeline(name: str) -> TransformPipeline:
    """
    Get preset transform pipeline.
    
    Args:
        name: Preset name
        
    Returns:
        Transform pipeline
    """
    if name not in PRESET_PIPELINES:
        raise ValueError(
            f"Unknown preset: {name}. "
            f"Available: {list(PRESET_PIPELINES.keys())}"
        )
    
    # Return a copy to avoid mutations
    preset = PRESET_PIPELINES[name]
    return TransformPipeline.from_dict(preset.to_dict())