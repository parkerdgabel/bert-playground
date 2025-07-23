"""Built-in templates for common data-to-text conversion patterns."""

import json
from typing import Any, Dict, Optional

from .base import Template, TemplateConfig
from .registry import register_template


class KeyValueTemplate(Template):
    """Simple key-value pair template (key: value | key: value)."""
    
    def convert(self, data: Dict[str, Any]) -> str:
        """Convert data to key-value format."""
        filtered_data = self.filter_columns(data)
        parts = []
        
        for key, value in filtered_data.items():
            formatted_value = self.format_value(value, key)
            prefix = self.get_prefix(key)
            
            if prefix:
                parts.append(f"{prefix}{key}: {formatted_value}")
            else:
                parts.append(f"{key}: {formatted_value}")
        
        separator = self.config.custom_options.get("separator", " | ")
        return separator.join(parts)


class NaturalLanguageTemplate(Template):
    """Natural language template that creates sentences from data."""
    
    def convert(self, data: Dict[str, Any]) -> str:
        """Convert data to natural language format."""
        filtered_data = self.filter_columns(data)
        sentences = []
        
        # Group by column type
        numerical_items = []
        categorical_items = []
        text_items = []
        
        for key, value in filtered_data.items():
            if pd.isna(value):
                continue
                
            col_type = self._column_types.get(key, "unknown")
            formatted_value = self.format_value(value, key)
            
            if col_type == "numerical":
                numerical_items.append(f"The {key} is {formatted_value}")
            elif col_type == "categorical":
                categorical_items.append(f"The {key} is {formatted_value}")
            elif col_type == "text":
                text_items.append(f"For {key}: {formatted_value}")
            else:
                sentences.append(f"The {key} has value {formatted_value}")
        
        # Combine into sentences
        if numerical_items:
            sentences.append(". ".join(numerical_items) + ".")
        
        if categorical_items:
            sentences.append(". ".join(categorical_items) + ".")
        
        if text_items:
            sentences.extend([item + "." for item in text_items])
        
        return " ".join(sentences)


class DescriptiveTemplate(Template):
    """Descriptive template with context and narrative structure."""
    
    def convert(self, data: Dict[str, Any]) -> str:
        """Convert data to descriptive format."""
        filtered_data = self.filter_columns(data)
        
        # Start with introduction if provided
        intro = self.config.custom_options.get("introduction", "This record contains the following information:")
        parts = [intro]
        
        # Group by importance (if specified)
        primary_cols = self.config.custom_options.get("primary_columns", [])
        secondary_cols = self.config.custom_options.get("secondary_columns", [])
        
        # Process primary columns first
        if primary_cols:
            primary_parts = []
            for col in primary_cols:
                if col in filtered_data:
                    value = filtered_data[col]
                    formatted = self.format_value(value, col)
                    primary_parts.append(f"{col} is {formatted}")
            
            if primary_parts:
                parts.append("Key attributes: " + ", ".join(primary_parts) + ".")
        
        # Process remaining columns
        remaining_parts = []
        for key, value in filtered_data.items():
            if key in primary_cols or key in secondary_cols:
                continue
            
            formatted = self.format_value(value, key)
            col_type = self._column_types.get(key, "unknown")
            
            if col_type == "text" and len(str(value)) > 50:
                remaining_parts.append(f"The {key} states: '{formatted}'")
            else:
                remaining_parts.append(f"{key}: {formatted}")
        
        if remaining_parts:
            parts.append("Additional details include " + ", ".join(remaining_parts) + ".")
        
        # Add secondary columns at the end
        if secondary_cols:
            secondary_parts = []
            for col in secondary_cols:
                if col in filtered_data:
                    value = filtered_data[col]
                    formatted = self.format_value(value, col)
                    secondary_parts.append(f"{col}: {formatted}")
            
            if secondary_parts:
                parts.append("Other information: " + ", ".join(secondary_parts) + ".")
        
        return " ".join(parts)


class CustomTemplate(Template):
    """Custom template using user-provided format string."""
    
    def __init__(self, config: Optional[TemplateConfig] = None, template_string: Optional[str] = None):
        """Initialize with custom template string.
        
        Args:
            config: Template configuration
            template_string: Format string with {column_name} placeholders
        """
        super().__init__(config)
        self.template_string = template_string or self.config.custom_options.get(
            "template_string", 
            "Record with {column1} and {column2}"
        )
    
    def convert(self, data: Dict[str, Any]) -> str:
        """Convert data using custom template string."""
        filtered_data = self.filter_columns(data)
        
        # Start with template string
        result = self.template_string
        
        # Replace placeholders
        for key, value in filtered_data.items():
            placeholder = f"{{{key}}}"
            if placeholder in result:
                formatted = self.format_value(value, key)
                result = result.replace(placeholder, formatted)
        
        # Handle any remaining placeholders
        import re
        remaining = re.findall(r'\{(\w+)\}', result)
        for key in remaining:
            result = result.replace(f"{{{key}}}", self.config.null_representation)
        
        return result


class MarkdownTableTemplate(Template):
    """Template that formats data as a markdown table row."""
    
    def convert(self, data: Dict[str, Any]) -> str:
        """Convert data to markdown table format."""
        filtered_data = self.filter_columns(data)
        
        # Get column order
        column_order = self.config.custom_options.get("column_order", list(filtered_data.keys()))
        
        # Create header if requested
        include_header = self.config.custom_options.get("include_header", True)
        parts = []
        
        if include_header:
            header = "| " + " | ".join(column_order) + " |"
            separator = "| " + " | ".join(["---"] * len(column_order)) + " |"
            parts.extend([header, separator])
        
        # Create data row
        values = []
        for col in column_order:
            if col in filtered_data:
                value = filtered_data[col]
                formatted = self.format_value(value, col)
                values.append(formatted)
            else:
                values.append(self.config.null_representation)
        
        row = "| " + " | ".join(values) + " |"
        parts.append(row)
        
        return "\n".join(parts)


class JSONTemplate(Template):
    """Template that formats data as JSON."""
    
    def convert(self, data: Dict[str, Any]) -> str:
        """Convert data to JSON format."""
        filtered_data = self.filter_columns(data)
        
        # Convert values to JSON-serializable format
        json_data = {}
        for key, value in filtered_data.items():
            if pd.isna(value):
                json_data[key] = None
            elif isinstance(value, (int, float, str, bool)):
                json_data[key] = value
            else:
                json_data[key] = str(value)
        
        # Format options
        indent = self.config.custom_options.get("indent", 2)
        compact = self.config.custom_options.get("compact", False)
        
        if compact:
            return json.dumps(json_data, separators=(',', ':'))
        else:
            return json.dumps(json_data, indent=indent)


# Register built-in templates
register_template("keyvalue", KeyValueTemplate)
register_template("natural", NaturalLanguageTemplate)
register_template("descriptive", DescriptiveTemplate)
register_template("custom", CustomTemplate)
register_template("markdown", MarkdownTableTemplate)
register_template("json", JSONTemplate)


# Import pandas for null checking
import pandas as pd