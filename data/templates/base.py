"""Base template class and configuration for text conversion templates."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import pandas as pd
from loguru import logger


@dataclass
class TemplateConfig:
    """Configuration for templates."""
    
    # Template behavior
    include_null_values: bool = False
    null_representation: str = "missing"
    
    # Formatting options
    decimal_places: int = 2
    use_scientific_notation: bool = False
    scientific_threshold: float = 1e6
    
    # Column filtering
    include_columns: Optional[List[str]] = None
    exclude_columns: Optional[List[str]] = None
    
    # Text processing
    max_text_length: Optional[int] = None
    truncate_indicator: str = "..."
    
    # Special handling
    categorical_prefix: str = ""
    numerical_prefix: str = ""
    text_prefix: str = ""
    
    # Template-specific options
    custom_options: Dict[str, Any] = field(default_factory=dict)


class Template(ABC):
    """Abstract base class for data-to-text templates."""
    
    def __init__(self, config: Optional[TemplateConfig] = None):
        """Initialize template with configuration.
        
        Args:
            config: Template configuration
        """
        self.config = config or TemplateConfig()
        self._column_types: Dict[str, str] = {}
        logger.debug(f"Initialized {self.__class__.__name__} template")
    
    @abstractmethod
    def convert(self, data: Dict[str, Any]) -> str:
        """Convert a data row to text representation.
        
        Args:
            data: Dictionary containing row data
            
        Returns:
            Text representation of the data
        """
        pass
    
    def convert_batch(self, data: pd.DataFrame) -> List[str]:
        """Convert a batch of data rows to text representations.
        
        Args:
            data: DataFrame containing multiple rows
            
        Returns:
            List of text representations
        """
        texts = []
        for _, row in data.iterrows():
            text = self.convert(row.to_dict())
            texts.append(text)
        return texts
    
    def set_column_types(self, column_types: Dict[str, str]) -> None:
        """Set column type information for better formatting.
        
        Args:
            column_types: Dictionary mapping column names to types
                         ('text', 'categorical', 'numerical')
        """
        self._column_types = column_types
        logger.debug(f"Set column types for {len(column_types)} columns")
    
    def filter_columns(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Filter columns based on configuration.
        
        Args:
            data: Original data dictionary
            
        Returns:
            Filtered data dictionary
        """
        filtered = {}
        
        for key, value in data.items():
            # Skip if in exclude list
            if self.config.exclude_columns and key in self.config.exclude_columns:
                continue
            
            # Skip if not in include list (when specified)
            if self.config.include_columns and key not in self.config.include_columns:
                continue
            
            # Skip null values if configured
            if pd.isna(value) and not self.config.include_null_values:
                continue
            
            filtered[key] = value
        
        return filtered
    
    def format_value(self, value: Any, column_name: str) -> str:
        """Format a single value based on its type and column info.
        
        Args:
            value: Value to format
            column_name: Name of the column
            
        Returns:
            Formatted string representation
        """
        # Handle null values
        if pd.isna(value):
            return self.config.null_representation
        
        # Get column type
        col_type = self._column_types.get(column_name, "unknown")
        
        # Format based on type
        if col_type == "numerical" or isinstance(value, (int, float)):
            return self._format_numerical(value)
        elif col_type == "text" or isinstance(value, str):
            return self._format_text(value)
        elif col_type == "categorical":
            return self._format_categorical(value)
        else:
            return str(value)
    
    def _format_numerical(self, value: float) -> str:
        """Format numerical values."""
        if isinstance(value, int):
            return str(value)
        
        # Use scientific notation for large numbers
        if self.config.use_scientific_notation and abs(value) >= self.config.scientific_threshold:
            return f"{value:.{self.config.decimal_places}e}"
        
        # Regular decimal formatting
        return f"{value:.{self.config.decimal_places}f}"
    
    def _format_text(self, value: str) -> str:
        """Format text values."""
        text = str(value).strip()
        
        # Truncate if needed
        if self.config.max_text_length and len(text) > self.config.max_text_length:
            text = text[:self.config.max_text_length - len(self.config.truncate_indicator)]
            text += self.config.truncate_indicator
        
        return text
    
    def _format_categorical(self, value: Any) -> str:
        """Format categorical values."""
        return str(value).strip()
    
    def get_prefix(self, column_name: str) -> str:
        """Get prefix for a column based on its type.
        
        Args:
            column_name: Name of the column
            
        Returns:
            Prefix string
        """
        col_type = self._column_types.get(column_name, "unknown")
        
        if col_type == "numerical":
            return self.config.numerical_prefix
        elif col_type == "text":
            return self.config.text_prefix
        elif col_type == "categorical":
            return self.config.categorical_prefix
        
        return ""
    
    @property
    def name(self) -> str:
        """Get template name."""
        return self.__class__.__name__.replace("Template", "").lower()
    
    @property
    def description(self) -> str:
        """Get template description."""
        return self.__class__.__doc__ or f"{self.name} template"
    
    def validate(self, data: Dict[str, Any]) -> bool:
        """Validate that data is suitable for this template.
        
        Args:
            data: Data to validate
            
        Returns:
            True if data is valid for this template
        """
        # Basic validation - can be overridden by subclasses
        if not isinstance(data, dict):
            return False
        
        # Check required columns if specified
        if self.config.include_columns:
            for col in self.config.include_columns:
                if col not in data:
                    logger.warning(f"Required column '{col}' not found in data")
                    return False
        
        return True