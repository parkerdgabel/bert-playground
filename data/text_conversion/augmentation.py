"""
Text augmentation utilities for converters.
"""

from typing import Dict, Any, List, Optional, Union, Tuple
from dataclasses import dataclass, field
import random
from abc import ABC, abstractmethod
from loguru import logger

from .base_converter import BaseTextConverter, TextConversionConfig


@dataclass
class AugmentationConfig:
    """Configuration for text augmentation."""
    
    # Basic settings
    probability: float = 0.5
    max_augmentations: int = 3
    seed: Optional[int] = None
    
    # Augmentation types
    enable_paraphrase: bool = True
    enable_style_variation: bool = True
    enable_detail_variation: bool = True
    enable_order_shuffling: bool = True
    
    # Paraphrase settings
    paraphrase_templates: Dict[str, List[str]] = field(default_factory=dict)
    
    # Style variations
    styles: List[str] = field(default_factory=lambda: ["formal", "casual", "technical"])
    
    # Detail levels
    detail_levels: List[str] = field(default_factory=lambda: ["concise", "standard", "detailed"])


class TextAugmenter:
    """Augments text output from converters."""
    
    def __init__(self, config: Optional[AugmentationConfig] = None):
        """
        Initialize text augmenter.
        
        Args:
            config: Augmentation configuration
        """
        self.config = config or AugmentationConfig()
        
        if self.config.seed:
            random.seed(self.config.seed)
        
        # Paraphrase patterns
        self.paraphrase_patterns = {
            "is": ["equals", "amounts to", "measures", "shows"],
            "has": ["contains", "includes", "features", "possesses"],
            "the": ["this", "that", "a", ""],
            "of": ["for", "from", "in", ""],
        }
    
    def augment(self, text: str, num_variations: int = 1) -> List[str]:
        """
        Generate augmented variations of text.
        
        Args:
            text: Original text
            num_variations: Number of variations to generate
            
        Returns:
            List of text variations (including original)
        """
        variations = [text]
        
        for _ in range(min(num_variations, self.config.max_augmentations)):
            if random.random() < self.config.probability:
                # Choose augmentation type
                augmentation_func = random.choice([
                    self._paraphrase,
                    self._vary_style,
                    self._vary_detail,
                    self._shuffle_order,
                ])
                
                try:
                    augmented = augmentation_func(text)
                    if augmented != text and augmented not in variations:
                        variations.append(augmented)
                except Exception as e:
                    logger.debug(f"Augmentation failed: {e}")
        
        return variations
    
    def _paraphrase(self, text: str) -> str:
        """Apply paraphrasing to text."""
        if not self.config.enable_paraphrase:
            return text
        
        result = text
        
        # Apply word-level substitutions
        for original, replacements in self.paraphrase_patterns.items():
            if original in result and random.random() < 0.3:
                replacement = random.choice(replacements)
                result = result.replace(f" {original} ", f" {replacement} ", 1)
        
        # Apply template-based paraphrasing
        for pattern, replacements in self.config.paraphrase_templates.items():
            if pattern in result and replacements:
                replacement = random.choice(replacements)
                result = result.replace(pattern, replacement, 1)
        
        return result.strip()
    
    def _vary_style(self, text: str) -> str:
        """Vary the style of text."""
        if not self.config.enable_style_variation or not self.config.styles:
            return text
        
        style = random.choice(self.config.styles)
        
        if style == "formal":
            # Make more formal
            replacements = {
                "don't": "do not",
                "can't": "cannot",
                "won't": "will not",
                "it's": "it is",
                "let's": "let us",
            }
            result = text
            for informal, formal in replacements.items():
                result = result.replace(informal, formal)
            return result
        
        elif style == "casual":
            # Make more casual
            replacements = {
                "do not": "don't",
                "cannot": "can't",
                "will not": "won't",
                "it is": "it's",
                "approximately": "about",
            }
            result = text
            for formal, casual in replacements.items():
                result = result.replace(formal, casual)
            return result
        
        else:
            return text
    
    def _vary_detail(self, text: str) -> str:
        """Vary the level of detail in text."""
        if not self.config.enable_detail_variation:
            return text
        
        level = random.choice(self.config.detail_levels)
        
        if level == "concise":
            # Remove some details
            sentences = text.split(". ")
            if len(sentences) > 2:
                # Keep only most important sentences
                important = sentences[:1] + sentences[-1:]
                return ". ".join(important)
        
        elif level == "detailed":
            # Add filler phrases
            additions = [
                "Notably,",
                "It should be mentioned that",
                "Additionally,",
                "Furthermore,",
            ]
            sentences = text.split(". ")
            if sentences and random.random() < 0.5:
                addition = random.choice(additions)
                idx = random.randint(1, len(sentences))
                sentences.insert(idx, addition)
                return " ".join(sentences)
        
        return text
    
    def _shuffle_order(self, text: str) -> str:
        """Shuffle the order of elements in text."""
        if not self.config.enable_order_shuffling:
            return text
        
        # Only shuffle if text has clear segments
        if ", " in text:
            segments = text.split(", ")
            if len(segments) > 2:
                # Keep first and last, shuffle middle
                middle = segments[1:-1]
                random.shuffle(middle)
                return segments[0] + ", " + ", ".join(middle) + ", " + segments[-1]
        
        return text


class AugmentedConverter(BaseTextConverter):
    """Wrapper that adds augmentation to any converter."""
    
    def __init__(
        self,
        base_converter: BaseTextConverter,
        augmenter: Optional[TextAugmenter] = None,
        num_augmentations: int = 1
    ):
        """
        Initialize augmented converter.
        
        Args:
            base_converter: Base converter to augment
            augmenter: Text augmenter instance
            num_augmentations: Number of augmentations per sample
        """
        super().__init__(base_converter.config)
        self.base_converter = base_converter
        self.augmenter = augmenter or TextAugmenter()
        self.num_augmentations = num_augmentations
    
    def convert(self, data: Dict[str, Any]) -> str:
        """Convert data with augmentation."""
        # Get base conversion
        base_text = self.base_converter.convert(data)
        
        # Apply augmentation if enabled
        if self.config.augment and self.num_augmentations > 0:
            variations = self.augmenter.augment(base_text, self.num_augmentations)
            # Return random variation (excluding original)
            if len(variations) > 1:
                return random.choice(variations[1:])
        
        return base_text


class EnsembleConverter(BaseTextConverter):
    """Ensemble of multiple converters."""
    
    def __init__(
        self,
        converters: List[BaseTextConverter],
        weights: Optional[List[float]] = None,
        aggregation: str = "random",
        config: Optional[TextConversionConfig] = None
    ):
        """
        Initialize ensemble converter.
        
        Args:
            converters: List of converters
            weights: Optional weights for weighted selection
            aggregation: Aggregation method ("random", "all", "weighted")
            config: Optional configuration
        """
        super().__init__(config)
        self.converters = converters
        self.weights = weights or [1.0] * len(converters)
        self.aggregation = aggregation
        
        # Normalize weights
        total_weight = sum(self.weights)
        self.weights = [w / total_weight for w in self.weights]
    
    def convert(self, data: Dict[str, Any]) -> str:
        """Convert using ensemble."""
        if self.aggregation == "random":
            # Random selection
            if self.config.augment:
                converter = random.choice(self.converters)
            else:
                converter = self.converters[0]
            return converter.convert(data)
        
        elif self.aggregation == "weighted":
            # Weighted random selection
            converter = random.choices(self.converters, weights=self.weights)[0]
            return converter.convert(data)
        
        elif self.aggregation == "all":
            # Concatenate all outputs
            outputs = []
            for converter in self.converters:
                output = converter.convert(data)
                outputs.append(output)
            return " ".join(outputs)
        
        else:
            raise ValueError(f"Unknown aggregation method: {self.aggregation}")


class ConditionalConverter(BaseTextConverter):
    """Converter that selects strategy based on conditions."""
    
    def __init__(
        self,
        condition_field: str,
        converters: Dict[Any, BaseTextConverter],
        default_converter: Optional[BaseTextConverter] = None,
        config: Optional[TextConversionConfig] = None
    ):
        """
        Initialize conditional converter.
        
        Args:
            condition_field: Field to check for condition
            converters: Mapping of condition values to converters
            default_converter: Default converter if no condition matches
            config: Optional configuration
        """
        super().__init__(config)
        self.condition_field = condition_field
        self.converters = converters
        self.default_converter = default_converter
    
    def convert(self, data: Dict[str, Any]) -> str:
        """Convert based on condition."""
        condition_value = data.get(self.condition_field)
        
        # Select converter
        converter = self.converters.get(condition_value, self.default_converter)
        
        if converter is None:
            raise ValueError(
                f"No converter found for condition '{condition_value}' "
                f"in field '{self.condition_field}'"
            )
        
        return converter.convert(data)


class CachedConverter(BaseTextConverter):
    """Converter with built-in caching."""
    
    def __init__(
        self,
        base_converter: BaseTextConverter,
        cache_size: int = 10000
    ):
        """
        Initialize cached converter.
        
        Args:
            base_converter: Base converter to cache
            cache_size: Maximum cache size
        """
        super().__init__(base_converter.config)
        self.base_converter = base_converter
        self.cache_size = cache_size
        self._cache = {}
    
    def convert(self, data: Dict[str, Any]) -> str:
        """Convert with caching."""
        # Generate cache key
        cache_key = self._get_cache_key(data)
        
        # Check cache
        if cache_key in self._cache:
            return self._cache[cache_key]
        
        # Convert
        result = self.base_converter.convert(data)
        
        # Update cache
        self._cache[cache_key] = result
        
        # Evict if needed
        if len(self._cache) > self.cache_size:
            # Remove oldest
            oldest = next(iter(self._cache))
            del self._cache[oldest]
        
        return result