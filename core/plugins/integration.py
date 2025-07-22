"""Integration layer for the unified plugin system.

This module provides:
- Migration from old plugin system to new
- Integration with existing CLI commands
- Backwards compatibility
"""

import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Type, Union

from loguru import logger

# Avoid circular import - import these later when needed
# from cli.plugins.registry import ComponentRegistry as OldRegistry  
# from cli.plugins.loader import PluginLoader as OldLoader
from core.di import Container, get_container

from .base import Plugin, PluginContext
from .config import PluginConfig, load_plugin_config
from .registry import PluginRegistry


class PluginSystemIntegration:
    """Integrates the unified plugin system with existing k-bert components."""
    
    def __init__(self, container: Optional[Container] = None):
        """Initialize integration.
        
        Args:
            container: DI container
        """
        self.container = container or get_container()
        self.new_registry = PluginRegistry(container=self.container)
        self.old_registry = None  # Lazy load to avoid circular imports
        self.old_loader = None    # Lazy load to avoid circular imports
    
    def migrate_from_old_system(
        self,
        project_root: Optional[Union[str, Path]] = None
    ) -> Dict[str, Plugin]:
        """Migrate plugins from the old system to the new unified system.
        
        Args:
            project_root: Project root directory
            
        Returns:
            Dictionary of migrated plugins
        """
        # Lazy import to avoid circular dependencies
        if self.old_registry is None:
            from cli.plugins.registry import ComponentRegistry
            from cli.plugins.loader import PluginLoader
            self.old_registry = ComponentRegistry()
            self.old_loader = PluginLoader()
        
        migrated = {}
        
        # Load using old system first
        if project_root:
            old_results = self.old_loader.load_project_plugins(override=True)
            logger.info(f"Old system loaded plugins from: {list(old_results.keys())}")
        
        # Get old components
        old_components = self.old_registry.list_components()
        
        for component_type, component_names in old_components.items():
            for name in component_names:
                try:
                    # Get old component instance
                    old_component = self.old_registry.get(component_type, name)
                    
                    # Create wrapper plugin
                    wrapper = self._create_wrapper_plugin(old_component, component_type)
                    
                    # Register with new system
                    self.new_registry.register(
                        plugin=wrapper,
                        name=f"{component_type}_{name}",
                        category=component_type,
                        initialize=False,  # Don't auto-initialize migrated plugins
                    )
                    
                    migrated[f"{component_type}_{name}"] = wrapper
                    
                except Exception as e:
                    logger.error(f"Failed to migrate {component_type}.{name}: {e}")
        
        logger.info(f"Migrated {len(migrated)} plugins from old system")
        return migrated
    
    def setup_unified_system(
        self,
        project_root: Optional[Union[str, Path]] = None,
        config_file: Optional[Union[str, Path]] = None,
        config_dict: Optional[Dict[str, Any]] = None,
        migrate_old: bool = True,
    ) -> PluginRegistry:
        """Set up the unified plugin system.
        
        Args:
            project_root: Project root directory
            config_file: Configuration file path
            config_dict: Configuration dictionary
            migrate_old: Whether to migrate from old system
            
        Returns:
            Configured plugin registry
        """
        # Load configuration
        plugin_config = load_plugin_config(
            project_root=project_root,
            config_file=config_file,
            config_dict=config_dict,
        )
        
        # Migrate old system if requested
        if migrate_old:
            self.migrate_from_old_system(project_root)
        
        # Load new plugins
        new_plugins = self.new_registry.load_and_register(
            project_root=project_root,
            config=plugin_config.to_dict(),
            validate=plugin_config.schema.validate_on_load,
            initialize=plugin_config.schema.auto_initialize,
        )
        
        logger.info(f"Loaded {len(new_plugins)} new plugins")
        
        # Auto-start if configured
        if plugin_config.schema.auto_start:
            self.new_registry.start_all()
        
        return self.new_registry
    
    def get_integrated_registry(self) -> PluginRegistry:
        """Get the integrated plugin registry.
        
        Returns:
            Plugin registry with both old and new plugins
        """
        return self.new_registry
    
    def _create_wrapper_plugin(self, old_component: Any, component_type: str) -> Plugin:
        """Create a wrapper plugin for an old component.
        
        Args:
            old_component: Old component instance
            component_type: Component type
            
        Returns:
            Wrapper plugin
        """
        from .base import PluginBase, PluginMetadata
        
        class LegacyPluginWrapper(PluginBase):
            """Wrapper for legacy components."""
            
            def __init__(self, component: Any, comp_type: str):
                super().__init__()
                self.component = component
                self.comp_type = comp_type
            
            def _create_metadata(self) -> PluginMetadata:
                """Create metadata for legacy component."""
                # Handle mock objects and missing attributes gracefully
                name = self.component.__class__.__name__
                if hasattr(self.component, "name") and isinstance(self.component.name, str):
                    name = self.component.name
                
                version = "legacy"
                if hasattr(self.component, "version") and isinstance(self.component.version, str):
                    version = self.component.version
                
                return PluginMetadata(
                    name=name,
                    version=version,
                    description=f"Legacy {self.comp_type} component",
                    category=self.comp_type,
                    tags=["legacy", "migrated"],
                )
            
            def _initialize(self, context: PluginContext) -> None:
                """Initialize legacy component."""
                # Most legacy components don't need explicit initialization
                pass
            
            def get_component(self):
                """Get the wrapped legacy component."""
                return self.component
        
        return LegacyPluginWrapper(old_component, component_type)


# Global integration instance
_integration: Optional[PluginSystemIntegration] = None


def get_integration() -> PluginSystemIntegration:
    """Get the global integration instance.
    
    Returns:
        PluginSystemIntegration instance
    """
    global _integration
    if _integration is None:
        _integration = PluginSystemIntegration()
    return _integration


def setup_plugin_system(
    project_root: Optional[Union[str, Path]] = None,
    config_file: Optional[Union[str, Path]] = None,
    config_dict: Optional[Dict[str, Any]] = None,
    migrate_old: bool = True,
) -> PluginRegistry:
    """Set up the unified plugin system.
    
    This is the main entry point for setting up plugins in k-bert.
    
    Args:
        project_root: Project root directory
        config_file: Configuration file path
        config_dict: Configuration dictionary
        migrate_old: Whether to migrate from old system
        
    Returns:
        Configured plugin registry
    """
    integration = get_integration()
    return integration.setup_unified_system(
        project_root=project_root,
        config_file=config_file,
        config_dict=config_dict,
        migrate_old=migrate_old,
    )


def ensure_plugins_loaded(project_root: Optional[Union[str, Path]] = None) -> None:
    """Ensure plugins are loaded for a project.
    
    This function is idempotent and can be called multiple times.
    
    Args:
        project_root: Project root directory
    """
    # Check if already set up
    integration = get_integration()
    registry = integration.get_integrated_registry()
    
    existing_plugins = registry.list_plugins()
    if existing_plugins:
        logger.debug(f"Plugins already loaded: {len(existing_plugins)}")
        return
    
    # Set up the system
    setup_plugin_system(project_root=project_root)


# Backwards compatibility functions
def get_component_registry():
    """Get component registry (backwards compatibility).
    
    Returns:
        ComponentRegistry instance
        
    Deprecated:
        Use get_integration().get_integrated_registry() instead
    """
    warnings.warn(
        "get_component_registry() is deprecated. Use the unified plugin system instead.",
        DeprecationWarning,
        stacklevel=2
    )
    
    integration = get_integration()
    # Lazy load old registry
    if integration.old_registry is None:
        from cli.plugins.registry import ComponentRegistry
        integration.old_registry = ComponentRegistry()
    return integration.old_registry


def load_project_plugins(project_root: Optional[Union[str, Path]] = None):
    """Load project plugins (backwards compatibility).
    
    Args:
        project_root: Project root directory
        
    Deprecated:
        Use setup_plugin_system() instead
    """
    warnings.warn(
        "load_project_plugins() is deprecated. Use setup_plugin_system() instead.",
        DeprecationWarning,
        stacklevel=2
    )
    
    return setup_plugin_system(project_root=project_root)