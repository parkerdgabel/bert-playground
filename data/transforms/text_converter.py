"""
Text conversion transforms for tabular data.
Provides flexible text generation from structured data.
"""

from typing import Dict, Any, List, Optional, Union, Callable
from abc import ABC, abstractmethod
import random
from datetime import datetime
import re
import numpy as np
from loguru import logger

from .base_transforms import Transform


class BaseTextConverter(Transform, ABC):
    """Base class for text conversion."""
    
    def __init__(self, augment: bool = False, seed: Optional[int] = None):
        """
        Initialize text converter.
        
        Args:
            augment: Whether to apply augmentation
            seed: Random seed for reproducibility
        """
        self.augment = augment
        self.seed = seed
        
        if seed is not None:
            random.seed(seed)
    
    @abstractmethod
    def convert(self, data: Dict[str, Any]) -> str:
        """Convert data to text representation."""
        pass
    
    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Apply text conversion."""
        text = self.convert(data)
        result = data.copy()
        result["text"] = text
        return result


class TemplateTextConverter(BaseTextConverter):
    """Template-based text converter with augmentation support."""
    
    def __init__(
        self,
        templates: Union[str, List[str]],
        field_formatters: Optional[Dict[str, Callable]] = None,
        missing_value_text: str = "unknown",
        augment: bool = False,
        augmentation_templates: Optional[List[str]] = None,
        **kwargs
    ):
        """
        Initialize template converter.
        
        Args:
            templates: Template string(s) with {field} placeholders
            field_formatters: Custom formatters for specific fields
            missing_value_text: Text for missing values
            augment: Whether to use augmentation
            augmentation_templates: Additional templates for augmentation
            **kwargs: Additional arguments
        """
        super().__init__(augment, **kwargs)
        
        self.templates = templates if isinstance(templates, list) else [templates]
        self.augmentation_templates = augmentation_templates or []
        self.all_templates = self.templates + self.augmentation_templates
        self.field_formatters = field_formatters or {}
        self.missing_value_text = missing_value_text
    
    def convert(self, data: Dict[str, Any]) -> str:
        """Convert data to text using templates."""
        # Select template
        if self.augment and len(self.all_templates) > 1:
            template = random.choice(self.all_templates)
        else:
            template = self.templates[0]
        
        # Format fields
        formatted_data = {}
        for key, value in data.items():
            if key in self.field_formatters:
                formatted_value = self.field_formatters[key](value)
            else:
                formatted_value = self._default_formatter(value)
            formatted_data[key] = formatted_value
        
        # Fill template
        try:
            text = template.format(**formatted_data)
        except KeyError as e:
            # Handle missing fields
            missing_field = str(e).strip("'")
            formatted_data[missing_field] = self.missing_value_text
            text = template.format(**formatted_data)
        
        return text
    
    def _default_formatter(self, value: Any) -> str:
        """Default formatter for values."""
        if value is None or (isinstance(value, float) and np.isnan(value)):
            return self.missing_value_text
        elif isinstance(value, bool):
            return "yes" if value else "no"
        elif isinstance(value, (int, float)):
            if float(value).is_integer():
                return str(int(value))
            else:
                return f"{value:.2f}"
        else:
            return str(value)
    
    def __repr__(self) -> str:
        """String representation."""
        return f"TemplateTextConverter(templates={len(self.all_templates)})"


class FeatureConcatenator(BaseTextConverter):
    """Simple concatenation of features with descriptions."""
    
    def __init__(
        self,
        feature_descriptions: Dict[str, str],
        separator: str = ", ",
        prefix: str = "",
        suffix: str = "",
        skip_missing: bool = True,
        **kwargs
    ):
        """
        Initialize feature concatenator.
        
        Args:
            feature_descriptions: Mapping of field names to descriptions
            separator: Separator between features
            prefix: Text to add at the beginning
            suffix: Text to add at the end
            skip_missing: Skip missing values
            **kwargs: Additional arguments
        """
        super().__init__(**kwargs)
        
        self.feature_descriptions = feature_descriptions
        self.separator = separator
        self.prefix = prefix
        self.suffix = suffix
        self.skip_missing = skip_missing
    
    def convert(self, data: Dict[str, Any]) -> str:
        """Convert data by concatenating features."""
        parts = []
        
        if self.prefix:
            parts.append(self.prefix)
        
        for field, description in self.feature_descriptions.items():
            if field not in data:
                if not self.skip_missing:
                    parts.append(f"{description}: unknown")
                continue
            
            value = data[field]
            if value is None or (isinstance(value, float) and np.isnan(value)):
                if not self.skip_missing:
                    parts.append(f"{description}: unknown")
            else:
                parts.append(f"{description}: {value}")
        
        if self.suffix:
            parts.append(self.suffix)
        
        return self.separator.join(parts)
    
    def __repr__(self) -> str:
        """String representation."""
        return f"FeatureConcatenator(features={len(self.feature_descriptions)})"


class NaturalLanguageConverter(BaseTextConverter):
    """Convert data to natural language with context-aware formatting."""
    
    def __init__(
        self,
        intro_templates: Optional[List[str]] = None,
        feature_templates: Optional[Dict[str, List[str]]] = None,
        outro_templates: Optional[List[str]] = None,
        value_descriptions: Optional[Dict[str, Dict[Any, str]]] = None,
        numerical_binning: Optional[Dict[str, List[tuple]]] = None,
        **kwargs
    ):
        """
        Initialize natural language converter.
        
        Args:
            intro_templates: Templates for introduction
            feature_templates: Templates for each feature
            outro_templates: Templates for conclusion
            value_descriptions: Descriptions for categorical values
            numerical_binning: Binning rules for numerical features
            **kwargs: Additional arguments
        """
        super().__init__(**kwargs)
        
        self.intro_templates = intro_templates or ["The sample has the following characteristics:"]
        self.feature_templates = feature_templates or {}
        self.outro_templates = outro_templates or [""]
        self.value_descriptions = value_descriptions or {}
        self.numerical_binning = numerical_binning or {}
    
    def convert(self, data: Dict[str, Any]) -> str:
        """Convert data to natural language."""
        parts = []
        
        # Add introduction
        if self.augment:
            intro = random.choice(self.intro_templates)
        else:
            intro = self.intro_templates[0]
        if intro:
            parts.append(intro)
        
        # Add features
        for field, value in data.items():
            if field in self.feature_templates:
                # Use custom template
                templates = self.feature_templates[field]
                template = random.choice(templates) if self.augment else templates[0]
                
                # Format value
                formatted_value = self._format_value(field, value)
                
                try:
                    feature_text = template.format(value=formatted_value, **data)
                    parts.append(feature_text)
                except:
                    # Fallback to simple format
                    parts.append(f"{field} is {formatted_value}")
        
        # Add outro
        if self.augment:
            outro = random.choice(self.outro_templates)
        else:
            outro = self.outro_templates[0]
        if outro:
            parts.append(outro)
        
        return " ".join(parts)
    
    def _format_value(self, field: str, value: Any) -> str:
        """Format value based on type and configuration."""
        # Check for custom descriptions
        if field in self.value_descriptions and value in self.value_descriptions[field]:
            return self.value_descriptions[field][value]
        
        # Check for numerical binning
        if field in self.numerical_binning and isinstance(value, (int, float)):
            for (min_val, max_val), description in self.numerical_binning[field]:
                if min_val <= value < max_val:
                    return description
        
        # Default formatting
        if value is None:
            return "unknown"
        elif isinstance(value, bool):
            return "true" if value else "false"
        elif isinstance(value, (int, float)):
            return str(value)
        else:
            return str(value).lower()
    
    def __repr__(self) -> str:
        """String representation."""
        return "NaturalLanguageConverter()"


class MultiModalTextConverter(BaseTextConverter):
    """Convert data with support for different modalities (text, numerical, categorical)."""
    
    def __init__(
        self,
        text_fields: Optional[List[str]] = None,
        numerical_fields: Optional[List[str]] = None,
        categorical_fields: Optional[List[str]] = None,
        text_processor: Optional[Callable[[str], str]] = None,
        numerical_formatter: Optional[Callable[[float], str]] = None,
        categorical_formatter: Optional[Callable[[str], str]] = None,
        field_separator: str = ". ",
        **kwargs
    ):
        """
        Initialize multi-modal converter.
        
        Args:
            text_fields: Fields containing text
            numerical_fields: Fields containing numbers
            categorical_fields: Fields containing categories
            text_processor: Function to process text fields
            numerical_formatter: Function to format numbers
            categorical_formatter: Function to format categories
            field_separator: Separator between fields
            **kwargs: Additional arguments
        """
        super().__init__(**kwargs)
        
        self.text_fields = text_fields or []
        self.numerical_fields = numerical_fields or []
        self.categorical_fields = categorical_fields or []
        self.text_processor = text_processor or (lambda x: x)
        self.numerical_formatter = numerical_formatter or self._default_numerical_formatter
        self.categorical_formatter = categorical_formatter or (lambda x: str(x))
        self.field_separator = field_separator
    
    def convert(self, data: Dict[str, Any]) -> str:
        """Convert data based on field types."""
        parts = []
        
        # Process text fields
        for field in self.text_fields:
            if field in data and data[field]:
                processed = self.text_processor(str(data[field]))
                parts.append(f"{field}: {processed}")
        
        # Process numerical fields
        for field in self.numerical_fields:
            if field in data and data[field] is not None:
                formatted = self.numerical_formatter(data[field])
                parts.append(f"{field} is {formatted}")
        
        # Process categorical fields
        for field in self.categorical_fields:
            if field in data and data[field] is not None:
                formatted = self.categorical_formatter(data[field])
                parts.append(f"{field}: {formatted}")
        
        # Process remaining fields
        processed_fields = set(self.text_fields + self.numerical_fields + self.categorical_fields)
        for field, value in data.items():
            if field not in processed_fields and value is not None:
                parts.append(f"{field}: {value}")
        
        return self.field_separator.join(parts)
    
    def _default_numerical_formatter(self, value: float) -> str:
        """Default formatter for numerical values."""
        if float(value).is_integer():
            return str(int(value))
        else:
            return f"{value:.2f}"
    
    def __repr__(self) -> str:
        """String representation."""
        return "MultiModalTextConverter()"


class CompetitionSpecificConverter(BaseTextConverter):
    """Base class for competition-specific converters."""
    
    def __init__(self, competition_name: str, **kwargs):
        """
        Initialize competition-specific converter.
        
        Args:
            competition_name: Name of the competition
            **kwargs: Additional arguments
        """
        super().__init__(**kwargs)
        self.competition_name = competition_name
    
    def get_competition_context(self) -> str:
        """Get competition-specific context."""
        return f"This is data from the {self.competition_name} competition."
    
    def __repr__(self) -> str:
        """String representation."""
        return f"CompetitionSpecificConverter({self.competition_name})"


# Factory function for creating text converters
def create_text_converter(
    strategy: str = "template",
    **kwargs
) -> BaseTextConverter:
    """
    Create a text converter based on strategy.
    
    Args:
        strategy: Conversion strategy
        **kwargs: Arguments for the converter
        
    Returns:
        Text converter instance
    """
    strategies = {
        "template": TemplateTextConverter,
        "concatenate": FeatureConcatenator,
        "natural": NaturalLanguageConverter,
        "multimodal": MultiModalTextConverter,
    }
    
    if strategy not in strategies:
        raise ValueError(f"Unknown strategy: {strategy}. Available: {list(strategies.keys())}")
    
    return strategies[strategy](**kwargs)