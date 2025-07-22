"""Pre-built pipeline stages for common data processing tasks."""

from typing import Any, Dict, List, Optional, Union
import hashlib
import pickle
from pathlib import Path

import pandas as pd
from loguru import logger

from .transformers import Transformer
from ..templates import Template, get_template
from ..augmentation import get_registry as get_augmentation_registry
from ..components.validator import DataValidator, ValidationRule, ValidationLevel
from ..components.cache import DataCache


class ValidationStage(Transformer):
    """Pipeline stage for data validation."""
    
    def __init__(
        self,
        validator: Optional[DataValidator] = None,
        rules: Optional[List[ValidationRule]] = None,
        level: ValidationLevel = ValidationLevel.WARNING,
        name: str = "validation"
    ):
        """Initialize validation stage.
        
        Args:
            validator: DataValidator instance (creates new if None)
            rules: Validation rules to apply
            level: Default validation level
            name: Stage name
        """
        super().__init__(name)
        self.validator = validator or DataValidator()
        self.level = level
        
        # Add rules if provided
        if rules:
            for rule in rules:
                self.validator.add_rule(rule)
    
    def transform(self, data: Union[pd.DataFrame, Dict[str, Any], List[Dict[str, Any]]]) -> Any:
        """Validate data and return if valid.
        
        Args:
            data: Input data to validate
            
        Returns:
            Original data if valid
            
        Raises:
            ValueError: If validation fails at ERROR level
        """
        if isinstance(data, pd.DataFrame):
            is_valid, report = self.validator.validate_dataframe(data)
        elif isinstance(data, dict):
            is_valid, report = self.validator.validate_row(data)
        elif isinstance(data, list) and all(isinstance(item, dict) for item in data):
            # Validate list of rows
            is_valid = True
            reports = []
            for i, row in enumerate(data):
                row_valid, row_report = self.validator.validate_row(row)
                if not row_valid:
                    is_valid = False
                    reports.append(f"Row {i}: {row_report}")
            report = "\n".join(reports) if reports else "All rows valid"
        else:
            raise ValueError(f"Unsupported data type: {type(data)}")
        
        if not is_valid:
            logger.warning(f"Validation failed: {report}")
            if self.level == ValidationLevel.ERROR:
                raise ValueError(f"Data validation failed: {report}")
        else:
            logger.debug("Data validation passed")
        
        return data


class TemplateStage(Transformer):
    """Pipeline stage for applying templates to convert data to text."""
    
    def __init__(
        self,
        template: Union[str, Template],
        template_config: Optional[Dict[str, Any]] = None,
        column_types: Optional[Dict[str, str]] = None,
        name: str = "template"
    ):
        """Initialize template stage.
        
        Args:
            template: Template name or instance
            template_config: Configuration for template
            column_types: Column type information
            name: Stage name
        """
        super().__init__(name)
        
        if isinstance(template, str):
            from ..templates import TemplateConfig
            config = TemplateConfig(**template_config) if template_config else None
            self.template = get_template(template, config)
        else:
            self.template = template
        
        if column_types:
            self.template.set_column_types(column_types)
    
    def transform(self, data: Union[pd.DataFrame, Dict[str, Any], List[Dict[str, Any]]]) -> Union[List[str], str]:
        """Apply template to convert data to text.
        
        Args:
            data: Input data
            
        Returns:
            Text representation(s)
        """
        if isinstance(data, pd.DataFrame):
            return self.template.convert_batch(data)
        elif isinstance(data, dict):
            return self.template.convert(data)
        elif isinstance(data, list) and all(isinstance(item, dict) for item in data):
            return [self.template.convert(row) for row in data]
        else:
            raise ValueError(f"Unsupported data type: {type(data)}")


class AugmentationStage(Transformer):
    """Pipeline stage for data augmentation."""
    
    def __init__(
        self,
        augmenter: Union[str, Any],
        augmentation_config: Optional[Dict[str, Any]] = None,
        probability: float = 1.0,
        name: str = "augmentation"
    ):
        """Initialize augmentation stage.
        
        Args:
            augmenter: Augmenter name or instance
            augmentation_config: Configuration for augmenter
            probability: Probability of applying augmentation
            name: Stage name
        """
        super().__init__(name)
        
        if isinstance(augmenter, str):
            registry = get_augmentation_registry()
            self.augmenter = registry.get(augmenter)
        else:
            self.augmenter = augmenter
        
        self.probability = probability
        self.config = augmentation_config or {}
    
    def transform(self, data: Any) -> Any:
        """Apply augmentation to data.
        
        Args:
            data: Input data
            
        Returns:
            Augmented data
        """
        import random
        
        if random.random() > self.probability:
            logger.debug(f"Skipping augmentation (probability={self.probability})")
            return data
        
        if hasattr(self.augmenter, 'augment'):
            return self.augmenter.augment(data, **self.config)
        elif callable(self.augmenter):
            return self.augmenter(data, **self.config)
        else:
            raise ValueError(f"Augmenter must have 'augment' method or be callable")


class TokenizationStage(Transformer):
    """Pipeline stage for tokenization."""
    
    def __init__(
        self,
        tokenizer: Any,
        max_length: int = 512,
        padding: Union[bool, str] = True,
        truncation: bool = True,
        return_tensors: Optional[str] = None,
        name: str = "tokenization"
    ):
        """Initialize tokenization stage.
        
        Args:
            tokenizer: Tokenizer instance
            max_length: Maximum sequence length
            padding: Padding strategy
            truncation: Whether to truncate
            return_tensors: Tensor format to return
            name: Stage name
        """
        super().__init__(name)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.padding = padding
        self.truncation = truncation
        self.return_tensors = return_tensors
    
    def transform(self, data: Union[str, List[str], Dict[str, Any]]) -> Dict[str, Any]:
        """Tokenize input data.
        
        Args:
            data: Text data to tokenize
            
        Returns:
            Tokenized data
        """
        if isinstance(data, dict) and "text" in data:
            # Extract text from dict
            texts = data["text"]
        elif isinstance(data, (str, list)):
            texts = data
        else:
            raise ValueError(f"Unsupported data type for tokenization: {type(data)}")
        
        # Tokenize
        tokenized = self.tokenizer(
            texts,
            max_length=self.max_length,
            padding=self.padding,
            truncation=self.truncation,
            return_tensors=self.return_tensors
        )
        
        # If input was dict, preserve other fields
        if isinstance(data, dict):
            result = data.copy()
            result.update(tokenized)
            return result
        
        return tokenized


class CacheStage(Transformer):
    """Pipeline stage for caching intermediate results."""
    
    def __init__(
        self,
        cache: Optional[DataCache] = None,
        cache_dir: Optional[Union[str, Path]] = None,
        cache_key_func: Optional[Callable[[Any], str]] = None,
        ttl: Optional[int] = None,
        name: str = "cache"
    ):
        """Initialize cache stage.
        
        Args:
            cache: DataCache instance (creates new if None)
            cache_dir: Directory for cache storage
            cache_key_func: Function to generate cache keys
            ttl: Time-to-live for cache entries (seconds)
            name: Stage name
        """
        super().__init__(name)
        
        if cache is None:
            from ..components.cache import DiskCache
            cache_dir = Path(cache_dir) if cache_dir else Path("data/cache/pipeline")
            self.cache = DiskCache(cache_dir)
        else:
            self.cache = cache
        
        self.cache_key_func = cache_key_func or self._default_cache_key
        self.ttl = ttl
    
    def _default_cache_key(self, data: Any) -> str:
        """Generate default cache key."""
        # Convert data to bytes for hashing
        if isinstance(data, pd.DataFrame):
            data_bytes = pickle.dumps(data.to_dict())
        elif isinstance(data, (dict, list)):
            data_bytes = pickle.dumps(data)
        else:
            data_bytes = str(data).encode()
        
        # Generate hash
        return hashlib.md5(data_bytes).hexdigest()
    
    def transform(self, data: Any) -> Any:
        """Check cache and return cached result if available.
        
        Args:
            data: Input data
            
        Returns:
            Cached or original data
        """
        cache_key = self.cache_key_func(data)
        
        # Check cache
        cached = self.cache.get(cache_key)
        if cached is not None:
            logger.debug(f"Cache hit for key: {cache_key}")
            return cached
        
        logger.debug(f"Cache miss for key: {cache_key}")
        # Store in cache for next stage to use
        self._pending_cache = (cache_key, data)
        
        return data
    
    def cache_result(self, result: Any) -> None:
        """Cache the result from a subsequent stage.
        
        Args:
            result: Result to cache
        """
        if hasattr(self, '_pending_cache'):
            cache_key, _ = self._pending_cache
            self.cache.set(cache_key, result, ttl=self.ttl)
            logger.debug(f"Cached result for key: {cache_key}")
            delattr(self, '_pending_cache')