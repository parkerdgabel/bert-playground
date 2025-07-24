"""Template system for converting structured data to text representations.

This module provides a flexible template system for converting tabular and structured
data into text formats suitable for BERT processing.
"""

from .base import Template, TemplateConfig
from .registry import TemplateRegistry, get_template_registry
from .builtin import (
    KeyValueTemplate,
    NaturalLanguageTemplate,
    DescriptiveTemplate,
    CustomTemplate,
    MarkdownTableTemplate,
    JSONTemplate,
)

__all__ = [
    # Base classes
    "Template",
    "TemplateConfig",
    # Registry
    "TemplateRegistry",
    "get_template_registry",
    # Built-in templates
    "KeyValueTemplate",
    "NaturalLanguageTemplate",
    "DescriptiveTemplate",
    "CustomTemplate",
    "MarkdownTableTemplate",
    "JSONTemplate",
]