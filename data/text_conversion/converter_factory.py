"""
Factory for creating text converters with registration system.
"""

from typing import Dict, Any, Type, Optional, Union, Callable
from pathlib import Path
import json
from loguru import logger

from .base_converter import BaseTextConverter, TextConversionConfig
from .template_converter import TemplateConverter, TemplateConfig
from .feature_converter import FeatureConverter, FeatureConfig
from .natural_language_converter import NaturalLanguageConverter, NLConfig
from .competition_converters import (
    get_competition_converter,
    COMPETITION_CONVERTERS,
)


# Global registry of converters
_CONVERTER_REGISTRY: Dict[str, Type[BaseTextConverter]] = {
    "template": TemplateConverter,
    "feature": FeatureConverter,
    "natural_language": NaturalLanguageConverter,
    "nl": NaturalLanguageConverter,  # Alias
}

# Config class registry
_CONFIG_REGISTRY: Dict[Type[BaseTextConverter], Type[TextConversionConfig]] = {
    TemplateConverter: TemplateConfig,
    FeatureConverter: FeatureConfig,
    NaturalLanguageConverter: NLConfig,
}


def register_converter(
    name: str,
    converter_class: Type[BaseTextConverter],
    config_class: Optional[Type[TextConversionConfig]] = None
) -> None:
    """
    Register a new converter type.
    
    Args:
        name: Name to register converter under
        converter_class: Converter class
        config_class: Optional config class for the converter
    """
    _CONVERTER_REGISTRY[name.lower()] = converter_class
    
    if config_class:
        _CONFIG_REGISTRY[converter_class] = config_class
    
    logger.info(f"Registered converter '{name}' -> {converter_class.__name__}")


class TextConverterFactory:
    """Factory for creating text converters."""
    
    @staticmethod
    def create(
        strategy: str,
        config: Optional[Union[Dict[str, Any], TextConversionConfig]] = None,
        **kwargs
    ) -> BaseTextConverter:
        """
        Create a text converter.
        
        Args:
            strategy: Converter strategy name
            config: Configuration dict or object
            **kwargs: Additional arguments passed to converter
            
        Returns:
            Text converter instance
        """
        # Check if it's a competition converter
        if strategy.lower() in COMPETITION_CONVERTERS:
            return get_competition_converter(strategy, config)
        
        # Get converter class
        converter_class = _CONVERTER_REGISTRY.get(strategy.lower())
        if converter_class is None:
            available = list(_CONVERTER_REGISTRY.keys()) + list(COMPETITION_CONVERTERS.keys())
            raise ValueError(
                f"Unknown converter strategy '{strategy}'. "
                f"Available: {sorted(available)}"
            )
        
        # Handle config
        if config is None:
            # Create default config
            config_class = _CONFIG_REGISTRY.get(converter_class, TextConversionConfig)
            config = config_class(**kwargs)
        elif isinstance(config, dict):
            # Create config from dict
            config_class = _CONFIG_REGISTRY.get(converter_class, TextConversionConfig)
            config = config_class(**{**config, **kwargs})
        else:
            # Already a config object
            pass
        
        return converter_class(config)
    
    @staticmethod
    def from_config(config_path: Union[str, Path]) -> BaseTextConverter:
        """
        Create converter from configuration file.
        
        Args:
            config_path: Path to configuration file
            
        Returns:
            Text converter instance
        """
        with open(config_path, "r") as f:
            config_dict = json.load(f)
        
        # Extract strategy
        strategy = config_dict.pop("strategy", "template")
        
        return TextConverterFactory.create(strategy, config_dict)
    
    @staticmethod
    def create_ensemble(
        converters: List[Union[str, BaseTextConverter]],
        weights: Optional[List[float]] = None,
        aggregation: str = "random"
    ) -> BaseTextConverter:
        """
        Create ensemble of converters.
        
        Args:
            converters: List of converter names or instances
            weights: Optional weights for weighted selection
            aggregation: How to aggregate ("random", "all", "weighted")
            
        Returns:
            Ensemble converter
        """
        from .augmentation import EnsembleConverter
        
        # Convert strings to converter instances
        converter_instances = []
        for conv in converters:
            if isinstance(conv, str):
                converter_instances.append(TextConverterFactory.create(conv))
            else:
                converter_instances.append(conv)
        
        return EnsembleConverter(
            converters=converter_instances,
            weights=weights,
            aggregation=aggregation
        )
    
    @staticmethod
    def list_available() -> Dict[str, str]:
        """
        List available converter strategies.
        
        Returns:
            Dictionary of strategy name to description
        """
        available = {}
        
        # Built-in converters
        for name, cls in _CONVERTER_REGISTRY.items():
            available[name] = cls.__doc__.strip().split("\n")[0] if cls.__doc__ else "No description"
        
        # Competition converters
        for name in COMPETITION_CONVERTERS:
            available[name] = f"Competition-specific converter for {name}"
        
        return available
    
    @staticmethod
    def get_config_class(strategy: str) -> Type[TextConversionConfig]:
        """
        Get configuration class for a strategy.
        
        Args:
            strategy: Converter strategy name
            
        Returns:
            Configuration class
        """
        # Check competition converters
        if strategy.lower() in COMPETITION_CONVERTERS:
            # Most competition converters use TemplateConfig
            return TemplateConfig
        
        # Get converter class
        converter_class = _CONVERTER_REGISTRY.get(strategy.lower())
        if converter_class is None:
            raise ValueError(f"Unknown converter strategy '{strategy}'")
        
        return _CONFIG_REGISTRY.get(converter_class, TextConversionConfig)
    
    @staticmethod
    def create_config_template(
        strategy: str,
        output_path: Optional[Union[str, Path]] = None
    ) -> Dict[str, Any]:
        """
        Create configuration template for a strategy.
        
        Args:
            strategy: Converter strategy name
            output_path: Optional path to save template
            
        Returns:
            Configuration template dict
        """
        config_class = TextConverterFactory.get_config_class(strategy)
        
        # Create default instance
        default_config = config_class()
        config_dict = default_config.to_dict()
        
        # Add strategy field
        config_dict["strategy"] = strategy
        
        # Add helpful comments (as a separate metadata dict)
        config_dict["_metadata"] = {
            "description": f"Configuration template for {strategy} converter",
            "converter_class": _CONVERTER_REGISTRY.get(strategy.lower(), "Unknown").__name__,
            "config_class": config_class.__name__,
        }
        
        # Save if path provided
        if output_path:
            with open(output_path, "w") as f:
                json.dump(config_dict, f, indent=2)
            logger.info(f"Saved config template to {output_path}")
        
        return config_dict


# Convenience function
def create_text_converter(
    strategy: str = "template",
    **kwargs
) -> BaseTextConverter:
    """
    Create a text converter (convenience function).
    
    Args:
        strategy: Converter strategy
        **kwargs: Configuration options
        
    Returns:
        Text converter instance
    """
    return TextConverterFactory.create(strategy, **kwargs)