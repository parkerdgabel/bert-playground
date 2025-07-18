"""
Base transformation classes for the data pipeline.
Provides composable transforms that can be chained together.
"""

from typing import Any, Dict, List, Optional, Callable, Union
from abc import ABC, abstractmethod
import numpy as np
from loguru import logger
import random
import re


class Transform(ABC):
    """Abstract base class for all transforms."""
    
    @abstractmethod
    def __call__(self, data: Any) -> Any:
        """Apply transform to data."""
        pass
    
    def __repr__(self) -> str:
        """String representation."""
        return f"{self.__class__.__name__}()"


class Compose:
    """Compose multiple transforms together."""
    
    def __init__(self, transforms: List[Transform]):
        """
        Initialize composed transform.
        
        Args:
            transforms: List of transforms to apply in order
        """
        self.transforms = transforms
    
    def __call__(self, data: Any) -> Any:
        """Apply all transforms in sequence."""
        for transform in self.transforms:
            data = transform(data)
        return data
    
    def __repr__(self) -> str:
        """String representation."""
        transform_strs = [str(t) for t in self.transforms]
        return f"Compose({', '.join(transform_strs)})"
    
    def append(self, transform: Transform) -> None:
        """Add transform to the end of the pipeline."""
        self.transforms.append(transform)
    
    def prepend(self, transform: Transform) -> None:
        """Add transform to the beginning of the pipeline."""
        self.transforms.insert(0, transform)


class Lambda(Transform):
    """Apply a custom function as a transform."""
    
    def __init__(self, func: Callable[[Any], Any], name: Optional[str] = None):
        """
        Initialize lambda transform.
        
        Args:
            func: Function to apply
            name: Optional name for the transform
        """
        self.func = func
        self.name = name or func.__name__
    
    def __call__(self, data: Any) -> Any:
        """Apply function to data."""
        return self.func(data)
    
    def __repr__(self) -> str:
        """String representation."""
        return f"Lambda({self.name})"


class SelectFields(Transform):
    """Select specific fields from a dictionary."""
    
    def __init__(self, fields: List[str], allow_missing: bool = False):
        """
        Initialize field selector.
        
        Args:
            fields: Fields to select
            allow_missing: Whether to allow missing fields
        """
        self.fields = fields
        self.allow_missing = allow_missing
    
    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Select specified fields."""
        result = {}
        for field in self.fields:
            if field in data:
                result[field] = data[field]
            elif not self.allow_missing:
                raise KeyError(f"Field '{field}' not found in data")
        return result
    
    def __repr__(self) -> str:
        """String representation."""
        return f"SelectFields({self.fields})"


class RenameFields(Transform):
    """Rename fields in a dictionary."""
    
    def __init__(self, mapping: Dict[str, str]):
        """
        Initialize field renamer.
        
        Args:
            mapping: Mapping from old names to new names
        """
        self.mapping = mapping
    
    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Rename fields."""
        result = data.copy()
        for old_name, new_name in self.mapping.items():
            if old_name in result:
                result[new_name] = result.pop(old_name)
        return result
    
    def __repr__(self) -> str:
        """String representation."""
        return f"RenameFields({self.mapping})"


class FilterMissing(Transform):
    """Filter out samples with missing values."""
    
    def __init__(self, fields: Optional[List[str]] = None, allow_empty: bool = False):
        """
        Initialize missing value filter.
        
        Args:
            fields: Fields to check (None = check all)
            allow_empty: Whether to allow empty strings
        """
        self.fields = fields
        self.allow_empty = allow_empty
    
    def __call__(self, data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Filter samples with missing values."""
        fields_to_check = self.fields or list(data.keys())
        
        for field in fields_to_check:
            if field not in data:
                return None
            
            value = data[field]
            if value is None:
                return None
            
            if not self.allow_empty and isinstance(value, str) and value.strip() == "":
                return None
            
            if isinstance(value, float) and np.isnan(value):
                return None
        
        return data


class FillMissing(Transform):
    """Fill missing values with defaults."""
    
    def __init__(
        self,
        defaults: Dict[str, Any],
        fill_empty: bool = True,
        fill_nan: bool = True
    ):
        """
        Initialize missing value filler.
        
        Args:
            defaults: Default values for fields
            fill_empty: Whether to fill empty strings
            fill_nan: Whether to fill NaN values
        """
        self.defaults = defaults
        self.fill_empty = fill_empty
        self.fill_nan = fill_nan
    
    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Fill missing values."""
        result = data.copy()
        
        for field, default in self.defaults.items():
            if field not in result or result[field] is None:
                result[field] = default
            elif self.fill_empty and isinstance(result[field], str) and result[field].strip() == "":
                result[field] = default
            elif self.fill_nan and isinstance(result[field], float) and np.isnan(result[field]):
                result[field] = default
        
        return result
    
    def __repr__(self) -> str:
        """String representation."""
        return f"FillMissing(defaults={self.defaults})"


class Normalize(Transform):
    """Normalize numerical values."""
    
    def __init__(
        self,
        fields: List[str],
        means: Optional[Dict[str, float]] = None,
        stds: Optional[Dict[str, float]] = None,
        min_vals: Optional[Dict[str, float]] = None,
        max_vals: Optional[Dict[str, float]] = None,
        method: str = "standard"
    ):
        """
        Initialize normalizer.
        
        Args:
            fields: Fields to normalize
            means: Mean values for standard normalization
            stds: Standard deviation values
            min_vals: Minimum values for minmax normalization
            max_vals: Maximum values for minmax normalization
            method: Normalization method ('standard' or 'minmax')
        """
        self.fields = fields
        self.means = means or {}
        self.stds = stds or {}
        self.min_vals = min_vals or {}
        self.max_vals = max_vals or {}
        self.method = method
    
    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize specified fields."""
        result = data.copy()
        
        for field in self.fields:
            if field not in result:
                continue
            
            value = result[field]
            if not isinstance(value, (int, float)):
                continue
            
            if self.method == "standard":
                if field in self.means and field in self.stds:
                    result[field] = (value - self.means[field]) / (self.stds[field] + 1e-8)
            elif self.method == "minmax":
                if field in self.min_vals and field in self.max_vals:
                    range_val = self.max_vals[field] - self.min_vals[field]
                    if range_val > 0:
                        result[field] = (value - self.min_vals[field]) / range_val
        
        return result
    
    def __repr__(self) -> str:
        """String representation."""
        return f"Normalize(fields={self.fields}, method={self.method})"


class CategoricalEncode(Transform):
    """Encode categorical variables."""
    
    def __init__(
        self,
        fields: List[str],
        mappings: Optional[Dict[str, Dict[str, int]]] = None,
        unknown_value: int = -1
    ):
        """
        Initialize categorical encoder.
        
        Args:
            fields: Fields to encode
            mappings: Predefined mappings for each field
            unknown_value: Value for unknown categories
        """
        self.fields = fields
        self.mappings = mappings or {}
        self.unknown_value = unknown_value
    
    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Encode categorical fields."""
        result = data.copy()
        
        for field in self.fields:
            if field not in result:
                continue
            
            value = result[field]
            if field in self.mappings:
                result[field] = self.mappings[field].get(value, self.unknown_value)
            else:
                # Simple hash-based encoding for unknown mappings
                if value is not None:
                    result[field] = hash(str(value)) % 1000000
                else:
                    result[field] = self.unknown_value
        
        return result
    
    def __repr__(self) -> str:
        """String representation."""
        return f"CategoricalEncode(fields={self.fields})"


class TextClean(Transform):
    """Clean text data."""
    
    def __init__(
        self,
        fields: List[str],
        lowercase: bool = True,
        remove_punctuation: bool = False,
        remove_numbers: bool = False,
        remove_extra_spaces: bool = True,
        custom_patterns: Optional[List[tuple]] = None
    ):
        """
        Initialize text cleaner.
        
        Args:
            fields: Fields to clean
            lowercase: Convert to lowercase
            remove_punctuation: Remove punctuation
            remove_numbers: Remove numbers
            remove_extra_spaces: Remove extra whitespace
            custom_patterns: List of (pattern, replacement) tuples
        """
        self.fields = fields
        self.lowercase = lowercase
        self.remove_punctuation = remove_punctuation
        self.remove_numbers = remove_numbers
        self.remove_extra_spaces = remove_extra_spaces
        self.custom_patterns = custom_patterns or []
    
    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Clean text fields."""
        result = data.copy()
        
        for field in self.fields:
            if field not in result or not isinstance(result[field], str):
                continue
            
            text = result[field]
            
            # Apply cleaning operations
            if self.lowercase:
                text = text.lower()
            
            if self.remove_punctuation:
                text = re.sub(r'[^\w\s]', ' ', text)
            
            if self.remove_numbers:
                text = re.sub(r'\d+', '', text)
            
            # Apply custom patterns
            for pattern, replacement in self.custom_patterns:
                text = re.sub(pattern, replacement, text)
            
            if self.remove_extra_spaces:
                text = ' '.join(text.split())
            
            result[field] = text
        
        return result
    
    def __repr__(self) -> str:
        """String representation."""
        return f"TextClean(fields={self.fields})"


class RandomNoise(Transform):
    """Add random noise to numerical values."""
    
    def __init__(
        self,
        fields: List[str],
        noise_std: float = 0.1,
        probability: float = 1.0,
        seed: Optional[int] = None
    ):
        """
        Initialize random noise transform.
        
        Args:
            fields: Fields to add noise to
            noise_std: Standard deviation of noise
            probability: Probability of applying noise
            seed: Random seed
        """
        self.fields = fields
        self.noise_std = noise_std
        self.probability = probability
        self.seed = seed
        
        if seed is not None:
            np.random.seed(seed)
    
    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Add noise to specified fields."""
        if random.random() > self.probability:
            return data
        
        result = data.copy()
        
        for field in self.fields:
            if field not in result:
                continue
            
            value = result[field]
            if isinstance(value, (int, float)):
                noise = np.random.normal(0, self.noise_std)
                result[field] = value + noise
            elif isinstance(value, (list, np.ndarray)):
                noise = np.random.normal(0, self.noise_std, size=len(value))
                result[field] = value + noise
        
        return result
    
    def __repr__(self) -> str:
        """String representation."""
        return f"RandomNoise(fields={self.fields}, std={self.noise_std})"


class ConditionalTransform(Transform):
    """Apply transform conditionally based on a predicate."""
    
    def __init__(
        self,
        condition: Callable[[Dict[str, Any]], bool],
        transform: Transform,
        else_transform: Optional[Transform] = None
    ):
        """
        Initialize conditional transform.
        
        Args:
            condition: Condition function
            transform: Transform to apply if condition is true
            else_transform: Transform to apply if condition is false
        """
        self.condition = condition
        self.transform = transform
        self.else_transform = else_transform
    
    def __call__(self, data: Dict[str, Any]) -> Any:
        """Apply transform conditionally."""
        if self.condition(data):
            return self.transform(data)
        elif self.else_transform:
            return self.else_transform(data)
        else:
            return data
    
    def __repr__(self) -> str:
        """String representation."""
        return f"ConditionalTransform(if={self.condition.__name__}, then={self.transform})"