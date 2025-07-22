"""Configuration resolver for model creation.

This module handles the resolution and validation of model configurations,
converting between different config types and applying defaults.
"""

from dataclasses import dataclass
from typing import Any, Optional, Protocol, Union

from loguru import logger

from ..bert import BertConfig, ModernBertConfig
from ..heads.base import HeadConfig


class ConfigProtocol(Protocol):
    """Protocol for model configurations."""
    
    @classmethod
    def from_dict(cls, config_dict: dict[str, Any]) -> "ConfigProtocol":
        """Create config from dictionary."""
        ...


@dataclass
class ConfigResolver:
    """Resolves and validates model configurations."""
    
    def resolve_bert_config(
        self,
        config: Union[dict[str, Any], BertConfig, ModernBertConfig, None],
        model_type: str,
        **kwargs,
    ) -> Union[BertConfig, ModernBertConfig]:
        """Resolve BERT configuration based on model type.
        
        Args:
            config: Input configuration (dict, Config object, or None)
            model_type: Type of model being created
            **kwargs: Additional configuration parameters
            
        Returns:
            Resolved configuration object
        """
        # Determine if we're creating a ModernBERT model
        is_modern = model_type in ["modernbert_core", "modernbert_with_head"]
        
        if is_modern:
            return self._resolve_modern_config(config, **kwargs)
        else:
            return self._resolve_classic_config(config, **kwargs)
    
    def _resolve_modern_config(
        self,
        config: Union[dict[str, Any], BertConfig, ModernBertConfig, None],
        **kwargs,
    ) -> ModernBertConfig:
        """Resolve ModernBERT configuration."""
        if isinstance(config, ModernBertConfig):
            return config
            
        if isinstance(config, BertConfig):
            # Convert classic to modern
            logger.info("Converting BertConfig to ModernBertConfig")
            return ModernBertConfig.from_bert_config(config)
            
        if isinstance(config, dict):
            # Create from dict, applying only valid config kwargs
            config_dict = config.copy()
            valid_keys = self._get_valid_config_keys(ModernBertConfig)
            config_kwargs = {k: v for k, v in kwargs.items() if k in valid_keys}
            config_dict.update(config_kwargs)
            return ModernBertConfig(**config_dict)
            
        # Create default config
        model_size = kwargs.pop("model_size", "base")
        valid_keys = self._get_valid_config_keys(ModernBertConfig)
        config_kwargs = {k: v for k, v in kwargs.items() if k in valid_keys}
        return ModernBertConfig(model_size=model_size, **config_kwargs)
    
    def _resolve_classic_config(
        self,
        config: Union[dict[str, Any], BertConfig, None],
        **kwargs,
    ) -> BertConfig:
        """Resolve classic BERT configuration."""
        if isinstance(config, BertConfig):
            return config
            
        if isinstance(config, dict):
            return BertConfig(**config)
            
        # Create default config
        return BertConfig(**kwargs)
    
    def resolve_head_config(
        self,
        head_config: Union[HeadConfig, dict, None],
        head_type: Optional[str],
        input_size: int,
        output_size: int,
    ) -> HeadConfig:
        """Resolve head configuration.
        
        Args:
            head_config: Head configuration (HeadConfig, dict, or None)
            head_type: Type of head to create
            input_size: Input dimension for the head
            output_size: Output dimension for the head
            
        Returns:
            Resolved HeadConfig
        """
        if isinstance(head_config, HeadConfig):
            return head_config
            
        if isinstance(head_config, dict):
            return HeadConfig(**head_config)
            
        if head_type is None:
            raise ValueError("Either head_config or head_type must be provided")
            
        # Get default config for head type
        from ..heads.base import get_default_config_for_head_type
        
        return get_default_config_for_head_type(
            head_type,
            input_size=input_size,
            output_size=output_size,
        )
    
    def _get_valid_config_keys(self, config_class: type) -> set[str]:
        """Get valid configuration keys for a config class.
        
        Args:
            config_class: Configuration class
            
        Returns:
            Set of valid key names
        """
        if hasattr(config_class, "__dataclass_fields__"):
            return {f.name for f in config_class.__dataclass_fields__.values()}
        return set()
    
    def extract_model_kwargs(
        self, 
        kwargs: dict[str, Any],
        config_class: type,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Extract config kwargs from general kwargs.
        
        Args:
            kwargs: All keyword arguments
            config_class: Configuration class to extract for
            
        Returns:
            Tuple of (config_kwargs, remaining_kwargs)
        """
        valid_keys = self._get_valid_config_keys(config_class)
        
        config_kwargs = {}
        remaining_kwargs = {}
        
        for key, value in kwargs.items():
            if key in valid_keys:
                config_kwargs[key] = value
            else:
                remaining_kwargs[key] = value
                
        return config_kwargs, remaining_kwargs