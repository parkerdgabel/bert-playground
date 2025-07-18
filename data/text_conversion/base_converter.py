"""
Base text converter interface and configuration.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass, field
import json
from pathlib import Path
from loguru import logger


@dataclass
class TextConversionConfig:
    """Configuration for text conversion."""
    
    # Basic settings
    output_field: str = "text"
    augment: bool = False
    seed: Optional[int] = None
    
    # Field handling
    include_fields: Optional[List[str]] = None
    exclude_fields: Optional[List[str]] = None
    field_mappings: Dict[str, str] = field(default_factory=dict)
    
    # Missing value handling
    missing_value_strategy: str = "default"  # "default", "skip", "error"
    missing_value_text: str = "unknown"
    
    # Formatting
    max_length: Optional[int] = None
    preserve_case: bool = True
    strip_whitespace: bool = True
    
    # Caching
    enable_cache: bool = True
    cache_size: int = 1000
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "TextConversionConfig":
        """Create config from dictionary."""
        return cls(**config_dict)
    
    @classmethod
    def from_json(cls, json_path: Union[str, Path]) -> "TextConversionConfig":
        """Load config from JSON file."""
        with open(json_path, "r") as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            "output_field": self.output_field,
            "augment": self.augment,
            "seed": self.seed,
            "include_fields": self.include_fields,
            "exclude_fields": self.exclude_fields,
            "field_mappings": self.field_mappings,
            "missing_value_strategy": self.missing_value_strategy,
            "missing_value_text": self.missing_value_text,
            "max_length": self.max_length,
            "preserve_case": self.preserve_case,
            "strip_whitespace": self.strip_whitespace,
            "enable_cache": self.enable_cache,
            "cache_size": self.cache_size,
        }


class BaseTextConverter(ABC):
    """Abstract base class for text converters."""
    
    def __init__(self, config: Optional[TextConversionConfig] = None):
        """
        Initialize text converter.
        
        Args:
            config: Text conversion configuration
        """
        self.config = config or TextConversionConfig()
        self._cache = {} if self.config.enable_cache else None
        
        # Set random seed if provided
        if self.config.seed is not None:
            import random
            import numpy as np
            random.seed(self.config.seed)
            np.random.seed(self.config.seed)
    
    @abstractmethod
    def convert(self, data: Dict[str, Any]) -> str:
        """
        Convert data dictionary to text representation.
        
        Args:
            data: Input data dictionary
            
        Returns:
            Text representation
        """
        pass
    
    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply text conversion to data.
        
        Args:
            data: Input data
            
        Returns:
            Data with added text field
        """
        # Check cache
        if self._cache is not None:
            cache_key = self._get_cache_key(data)
            if cache_key in self._cache:
                text = self._cache[cache_key]
            else:
                text = self.convert(data)
                self._update_cache(cache_key, text)
        else:
            text = self.convert(data)
        
        # Post-process text
        text = self._post_process(text)
        
        # Add to data
        result = data.copy()
        result[self.config.output_field] = text
        return result
    
    def _get_cache_key(self, data: Dict[str, Any]) -> str:
        """Generate cache key from data."""
        # Filter fields
        fields_to_use = self._get_fields_to_use(data)
        key_data = {k: v for k, v in data.items() if k in fields_to_use}
        
        # Create deterministic key
        import hashlib
        key_str = json.dumps(key_data, sort_keys=True, default=str)
        return hashlib.md5(key_str.encode()).hexdigest()
    
    def _update_cache(self, key: str, value: str) -> None:
        """Update cache with LRU eviction."""
        if self._cache is None:
            return
        
        # Add to cache
        self._cache[key] = value
        
        # Evict if over size limit
        if len(self._cache) > self.config.cache_size:
            # Remove oldest (first) item
            oldest_key = next(iter(self._cache))
            del self._cache[oldest_key]
    
    def _get_fields_to_use(self, data: Dict[str, Any]) -> List[str]:
        """Get fields to include in conversion."""
        all_fields = set(data.keys())
        
        # Apply include filter
        if self.config.include_fields:
            fields = set(self.config.include_fields) & all_fields
        else:
            fields = all_fields
        
        # Apply exclude filter
        if self.config.exclude_fields:
            fields = fields - set(self.config.exclude_fields)
        
        return list(fields)
    
    def _handle_missing_value(self, field: str) -> str:
        """Handle missing field based on strategy."""
        if self.config.missing_value_strategy == "default":
            return self.config.missing_value_text
        elif self.config.missing_value_strategy == "skip":
            return ""
        else:  # "error"
            raise KeyError(f"Missing required field: {field}")
    
    def _post_process(self, text: str) -> str:
        """Post-process generated text."""
        # Strip whitespace
        if self.config.strip_whitespace:
            text = text.strip()
            # Normalize internal whitespace
            text = " ".join(text.split())
        
        # Apply max length
        if self.config.max_length and len(text) > self.config.max_length:
            text = text[:self.config.max_length - 3] + "..."
        
        return text
    
    def format_field(self, field: str, value: Any) -> str:
        """
        Format a single field value.
        
        Args:
            field: Field name
            value: Field value
            
        Returns:
            Formatted string
        """
        if value is None:
            return self.config.missing_value_text
        
        # Handle different types
        if isinstance(value, bool):
            return "yes" if value else "no"
        elif isinstance(value, (int, float)):
            if float(value).is_integer():
                return str(int(value))
            else:
                return f"{value:.2f}"
        else:
            return str(value)
    
    def get_field_value(self, data: Dict[str, Any], field: str) -> Any:
        """
        Get field value with mapping support.
        
        Args:
            data: Data dictionary
            field: Field name
            
        Returns:
            Field value
        """
        # Check if field is mapped
        actual_field = self.config.field_mappings.get(field, field)
        
        # Get value
        if actual_field in data:
            return data[actual_field]
        else:
            return None
    
    def save_config(self, path: Union[str, Path]) -> None:
        """Save configuration to JSON file."""
        with open(path, "w") as f:
            json.dump(self.config.to_dict(), f, indent=2)
        logger.info(f"Saved text converter config to {path}")
    
    @classmethod
    def load(cls, config_path: Union[str, Path]) -> "BaseTextConverter":
        """Load converter with configuration."""
        config = TextConversionConfig.from_json(config_path)
        return cls(config)
    
    def __repr__(self) -> str:
        """String representation."""
        return f"{self.__class__.__name__}(augment={self.config.augment})"