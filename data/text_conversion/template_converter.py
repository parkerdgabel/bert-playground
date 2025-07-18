"""
Template-based text converter with advanced features.
"""

from typing import Dict, Any, List, Optional, Union, Callable
from dataclasses import dataclass, field
import random
import re
from string import Template
from loguru import logger

from .base_converter import BaseTextConverter, TextConversionConfig


@dataclass
class TemplateConfig(TextConversionConfig):
    """Configuration for template-based conversion."""
    
    # Template settings
    templates: List[str] = field(default_factory=list)
    augmentation_templates: List[str] = field(default_factory=list)
    
    # Field formatters
    field_formatters: Dict[str, Callable[[Any], str]] = field(default_factory=dict)
    field_descriptions: Dict[str, str] = field(default_factory=dict)
    
    # Advanced formatting
    use_safe_substitution: bool = True
    template_engine: str = "python"  # "python", "jinja2", "custom"
    
    # Conditional templates
    conditional_templates: Dict[str, List[str]] = field(default_factory=dict)
    condition_field: Optional[str] = None


class TemplateConverter(BaseTextConverter):
    """Template-based text converter with augmentation support."""
    
    def __init__(self, config: Optional[TemplateConfig] = None):
        """
        Initialize template converter.
        
        Args:
            config: Template configuration
        """
        if config is None:
            config = TemplateConfig()
        super().__init__(config)
        
        self.config: TemplateConfig = config
        self._compiled_templates = self._compile_templates()
    
    def _compile_templates(self) -> List[Template]:
        """Compile templates for efficiency."""
        all_templates = self.config.templates + self.config.augmentation_templates
        
        if not all_templates:
            # Default template if none provided
            all_templates = ["Sample with features: $features"]
        
        compiled = []
        for template_str in all_templates:
            if self.config.template_engine == "python":
                # Use Python string.Template
                compiled.append(Template(template_str))
            else:
                # For now, just store as string
                compiled.append(template_str)
        
        return compiled
    
    def convert(self, data: Dict[str, Any]) -> str:
        """Convert data to text using templates."""
        # Select appropriate template
        template = self._select_template(data)
        
        # Prepare template variables
        template_vars = self._prepare_template_vars(data)
        
        # Render template
        if self.config.template_engine == "python":
            if isinstance(template, Template):
                if self.config.use_safe_substitution:
                    text = template.safe_substitute(**template_vars)
                else:
                    text = template.substitute(**template_vars)
            else:
                # String template
                text = template.format(**template_vars)
        elif self.config.template_engine == "jinja2":
            # Optional Jinja2 support
            text = self._render_jinja2(template, template_vars)
        else:
            # Custom template engine
            text = self._render_custom(template, template_vars)
        
        return text
    
    def _select_template(self, data: Dict[str, Any]) -> Union[Template, str]:
        """Select appropriate template based on data and configuration."""
        # Check for conditional templates
        if self.config.conditional_templates and self.config.condition_field:
            condition_value = data.get(self.config.condition_field)
            if condition_value in self.config.conditional_templates:
                templates = self.config.conditional_templates[condition_value]
                if templates:
                    template_str = random.choice(templates) if self.config.augment else templates[0]
                    return Template(template_str) if self.config.template_engine == "python" else template_str
        
        # Select from regular templates
        if self.config.augment and len(self._compiled_templates) > 1:
            return random.choice(self._compiled_templates)
        else:
            return self._compiled_templates[0]
    
    def _prepare_template_vars(self, data: Dict[str, Any]) -> Dict[str, str]:
        """Prepare variables for template substitution."""
        template_vars = {}
        
        # Get fields to use
        fields_to_use = self._get_fields_to_use(data)
        
        # Process each field
        for field in fields_to_use:
            value = self.get_field_value(data, field)
            
            # Apply custom formatter if available
            if field in self.config.field_formatters:
                try:
                    formatted_value = self.config.field_formatters[field](value)
                except Exception as e:
                    logger.warning(f"Error in custom formatter for {field}: {e}")
                    formatted_value = self.format_field(field, value)
            else:
                formatted_value = self.format_field(field, value)
            
            # Add to template vars
            template_vars[field] = formatted_value
            
            # Add field description if available
            if field in self.config.field_descriptions:
                template_vars[f"{field}_desc"] = self.config.field_descriptions[field]
        
        # Add computed variables
        template_vars.update(self._compute_additional_vars(data, template_vars))
        
        return template_vars
    
    def _compute_additional_vars(self, data: Dict[str, Any], template_vars: Dict[str, str]) -> Dict[str, str]:
        """Compute additional template variables."""
        additional_vars = {}
        
        # Add feature summary
        features = []
        for field, value in template_vars.items():
            if not field.endswith("_desc") and value != self.config.missing_value_text:
                if field in self.config.field_descriptions:
                    features.append(f"{self.config.field_descriptions[field]}: {value}")
                else:
                    features.append(f"{field}: {value}")
        
        additional_vars["features"] = ", ".join(features)
        additional_vars["feature_count"] = str(len(features))
        
        # Add any/all indicators
        additional_vars["has_missing"] = "yes" if self.config.missing_value_text in template_vars.values() else "no"
        
        return additional_vars
    
    def _render_jinja2(self, template: str, variables: Dict[str, str]) -> str:
        """Render template using Jinja2 (optional dependency)."""
        try:
            from jinja2 import Template as Jinja2Template
            jinja_template = Jinja2Template(template)
            return jinja_template.render(**variables)
        except ImportError:
            logger.warning("Jinja2 not installed, falling back to Python templates")
            return Template(template).safe_substitute(**variables)
    
    def _render_custom(self, template: str, variables: Dict[str, str]) -> str:
        """Render template using custom engine."""
        # Simple regex-based replacement as example
        text = template
        for key, value in variables.items():
            # Replace {{key}} style placeholders
            text = re.sub(rf"\{{\{{{key}\}}\}}", value, text)
            # Replace ${key} style placeholders
            text = re.sub(rf"\$\{{{key}\}}", value, text)
        return text
    
    @classmethod
    def from_templates(
        cls,
        templates: List[str],
        augmentation_templates: Optional[List[str]] = None,
        **kwargs
    ) -> "TemplateConverter":
        """
        Create converter from template strings.
        
        Args:
            templates: List of template strings
            augmentation_templates: Additional templates for augmentation
            **kwargs: Additional configuration options
            
        Returns:
            TemplateConverter instance
        """
        config = TemplateConfig(
            templates=templates,
            augmentation_templates=augmentation_templates or [],
            **kwargs
        )
        return cls(config)
    
    def add_field_formatter(self, field: str, formatter: Callable[[Any], str]) -> None:
        """
        Add custom formatter for a field.
        
        Args:
            field: Field name
            formatter: Formatting function
        """
        self.config.field_formatters[field] = formatter
    
    def add_field_description(self, field: str, description: str) -> None:
        """
        Add description for a field.
        
        Args:
            field: Field name
            description: Field description
        """
        self.config.field_descriptions[field] = description
    
    def __repr__(self) -> str:
        """String representation."""
        return f"TemplateConverter(templates={len(self._compiled_templates)})"