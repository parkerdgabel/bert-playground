"""Template registry for managing and discovering templates."""

from typing import Dict, List, Optional, Type

from loguru import logger

from .base import Template, TemplateConfig


class TemplateRegistry:
    """Registry for managing templates."""
    
    def __init__(self):
        """Initialize the template registry."""
        self._templates: Dict[str, Type[Template]] = {}
        self._instances: Dict[str, Template] = {}
        logger.debug("Initialized template registry")
    
    def register(self, name: str, template_class: Type[Template]) -> None:
        """Register a template class.
        
        Args:
            name: Name to register the template under
            template_class: Template class to register
        """
        if not issubclass(template_class, Template):
            raise ValueError(f"{template_class} must be a subclass of Template")
        
        self._templates[name] = template_class
        logger.debug(f"Registered template '{name}': {template_class.__name__}")
    
    def unregister(self, name: str) -> None:
        """Unregister a template.
        
        Args:
            name: Name of template to unregister
        """
        if name in self._templates:
            del self._templates[name]
            if name in self._instances:
                del self._instances[name]
            logger.debug(f"Unregistered template '{name}'")
    
    def get(self, name: str, config: Optional[TemplateConfig] = None) -> Template:
        """Get a template instance by name.
        
        Args:
            name: Name of the template
            config: Configuration for the template
            
        Returns:
            Template instance
        """
        if name not in self._templates:
            raise ValueError(f"Template '{name}' not found. Available: {self.list_templates()}")
        
        # Create instance if needed or config provided
        if name not in self._instances or config is not None:
            template_class = self._templates[name]
            instance = template_class(config)
            if config is None:
                # Only cache instances with default config
                self._instances[name] = instance
            return instance
        
        return self._instances[name]
    
    def create(self, name: str, config: Optional[TemplateConfig] = None) -> Template:
        """Create a new template instance (alias for get).
        
        Args:
            name: Name of the template
            config: Configuration for the template
            
        Returns:
            Template instance
        """
        return self.get(name, config)
    
    def list_templates(self) -> List[str]:
        """List all registered template names.
        
        Returns:
            List of template names
        """
        return list(self._templates.keys())
    
    def get_template_info(self, name: str) -> Dict[str, str]:
        """Get information about a template.
        
        Args:
            name: Name of the template
            
        Returns:
            Dictionary with template information
        """
        if name not in self._templates:
            raise ValueError(f"Template '{name}' not found")
        
        template_class = self._templates[name]
        instance = self.get(name)
        
        return {
            "name": name,
            "class": template_class.__name__,
            "module": template_class.__module__,
            "description": instance.description,
        }
    
    def clear(self) -> None:
        """Clear all registered templates."""
        self._templates.clear()
        self._instances.clear()
        logger.debug("Cleared template registry")


# Global registry instance
_registry = TemplateRegistry()


def get_template_registry() -> TemplateRegistry:
    """Get the global template registry.
    
    Returns:
        Global template registry instance
    """
    return _registry


def register_template(name: str, template_class: Type[Template]) -> None:
    """Register a template in the global registry.
    
    Args:
        name: Name to register the template under
        template_class: Template class to register
    """
    _registry.register(name, template_class)


def get_template(name: str, config: Optional[TemplateConfig] = None) -> Template:
    """Get a template from the global registry.
    
    Args:
        name: Name of the template
        config: Configuration for the template
        
    Returns:
        Template instance
    """
    return _registry.get(name, config)