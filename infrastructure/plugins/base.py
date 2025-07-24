"""Base classes and protocols for the k-bert plugin system.

This module defines the core abstractions for plugins including:
- Plugin metadata and lifecycle
- Base plugin class with lifecycle hooks
- Plugin context for dependency injection
- Plugin state management
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Protocol, Type, TypeVar

from loguru import logger
from pydantic import BaseModel, Field

from infrastructure.di import Container

T = TypeVar("T", bound="Plugin")


class PluginState(Enum):
    """Plugin lifecycle states."""
    
    DISCOVERED = auto()  # Plugin has been discovered but not loaded
    VALIDATED = auto()   # Plugin has been validated
    LOADED = auto()      # Plugin class has been loaded
    INITIALIZED = auto() # Plugin instance has been initialized
    RUNNING = auto()     # Plugin is actively running
    STOPPED = auto()     # Plugin has been stopped
    FAILED = auto()      # Plugin has failed
    CLEANED_UP = auto()  # Plugin has been cleaned up


class PluginMetadata(BaseModel):
    """Metadata for a plugin."""
    
    name: str = Field(..., description="Unique plugin name")
    version: str = Field("1.0.0", description="Plugin version")
    description: Optional[str] = Field(None, description="Plugin description")
    author: Optional[str] = Field(None, description="Plugin author")
    email: Optional[str] = Field(None, description="Author email")
    tags: List[str] = Field(default_factory=list, description="Plugin tags")
    requirements: List[str] = Field(default_factory=list, description="Plugin requirements")
    category: Optional[str] = Field(None, description="Plugin category (model, data, etc.)")
    
    # Dependencies
    depends_on: List[str] = Field(default_factory=list, description="Other plugins this depends on")
    conflicts_with: List[str] = Field(default_factory=list, description="Plugins this conflicts with")
    
    # Capabilities
    provides: List[str] = Field(default_factory=list, description="Capabilities provided")
    consumes: List[str] = Field(default_factory=list, description="Capabilities consumed")
    
    class Config:
        extra = "allow"  # Allow additional fields


@dataclass
class PluginContext:
    """Context passed to plugins during lifecycle operations."""
    
    container: Container
    config: Dict[str, Any] = field(default_factory=dict)
    metadata: Optional[PluginMetadata] = None
    state: PluginState = PluginState.DISCOVERED
    parent_context: Optional["PluginContext"] = None
    
    def get_config(self, key: str, default: Any = None) -> Any:
        """Get configuration value with fallback to parent context."""
        if key in self.config:
            return self.config[key]
        if self.parent_context:
            return self.parent_context.get_config(key, default)
        return default
    
    def resolve(self, service_type: Type[T]) -> T:
        """Resolve a service from the DI container."""
        return self.container.resolve(service_type)
    
    def create_child(self, **kwargs) -> "PluginContext":
        """Create a child context."""
        child_config = self.config.copy()
        child_config.update(kwargs.get("config", {}))
        
        return PluginContext(
            container=kwargs.get("container", self.container.create_child()),
            config=child_config,
            metadata=kwargs.get("metadata", self.metadata),
            state=kwargs.get("state", self.state),
            parent_context=self,
        )


class PluginError(Exception):
    """Base exception for plugin-related errors."""
    
    def __init__(self, message: str, plugin_name: Optional[str] = None, cause: Optional[Exception] = None):
        super().__init__(message)
        self.plugin_name = plugin_name
        self.cause = cause


class Plugin(Protocol):
    """Protocol defining the plugin contract."""
    
    @property
    def metadata(self) -> PluginMetadata:
        """Get plugin metadata."""
        ...
    
    @property
    def state(self) -> PluginState:
        """Get current plugin state."""
        ...
    
    def validate(self, context: PluginContext) -> None:
        """Validate plugin configuration and dependencies.
        
        Args:
            context: Plugin context
            
        Raises:
            PluginError: If validation fails
        """
        ...
    
    def initialize(self, context: PluginContext) -> None:
        """Initialize the plugin.
        
        Args:
            context: Plugin context
            
        Raises:
            PluginError: If initialization fails
        """
        ...
    
    def start(self, context: PluginContext) -> None:
        """Start the plugin.
        
        Args:
            context: Plugin context
            
        Raises:
            PluginError: If startup fails
        """
        ...
    
    def stop(self, context: PluginContext) -> None:
        """Stop the plugin.
        
        Args:
            context: Plugin context
        """
        ...
    
    def cleanup(self, context: PluginContext) -> None:
        """Clean up plugin resources.
        
        Args:
            context: Plugin context
        """
        ...


class PluginLifecycle(ABC):
    """Abstract base class for plugin lifecycle management."""
    
    def __init__(self):
        self._state = PluginState.DISCOVERED
        self._metadata: Optional[PluginMetadata] = None
    
    @property
    def state(self) -> PluginState:
        """Get current plugin state."""
        return self._state
    
    @property
    def metadata(self) -> PluginMetadata:
        """Get plugin metadata."""
        if self._metadata is None:
            self._metadata = self._create_metadata()
        return self._metadata
    
    @abstractmethod
    def _create_metadata(self) -> PluginMetadata:
        """Create plugin metadata.
        
        Returns:
            Plugin metadata
        """
        pass
    
    def validate(self, context: PluginContext) -> None:
        """Validate plugin configuration and dependencies.
        
        Args:
            context: Plugin context
            
        Raises:
            PluginError: If validation fails
        """
        logger.debug(f"Validating plugin: {self.metadata.name}")
        self._validate(context)
        self._state = PluginState.VALIDATED
    
    def _validate(self, context: PluginContext) -> None:
        """Override to implement validation logic."""
        pass
    
    def initialize(self, context: PluginContext) -> None:
        """Initialize the plugin.
        
        Args:
            context: Plugin context
            
        Raises:
            PluginError: If initialization fails
        """
        logger.debug(f"Initializing plugin: {self.metadata.name}")
        
        if self._state not in (PluginState.VALIDATED, PluginState.LOADED):
            raise PluginError(
                f"Cannot initialize plugin in state {self._state}",
                plugin_name=self.metadata.name
            )
        
        try:
            self._initialize(context)
            self._state = PluginState.INITIALIZED
        except Exception as e:
            self._state = PluginState.FAILED
            raise PluginError(
                f"Failed to initialize plugin: {e}",
                plugin_name=self.metadata.name,
                cause=e
            )
    
    @abstractmethod
    def _initialize(self, context: PluginContext) -> None:
        """Override to implement initialization logic."""
        pass
    
    def start(self, context: PluginContext) -> None:
        """Start the plugin.
        
        Args:
            context: Plugin context
            
        Raises:
            PluginError: If startup fails
        """
        logger.debug(f"Starting plugin: {self.metadata.name}")
        
        if self._state != PluginState.INITIALIZED:
            raise PluginError(
                f"Cannot start plugin in state {self._state}",
                plugin_name=self.metadata.name
            )
        
        try:
            self._start(context)
            self._state = PluginState.RUNNING
        except Exception as e:
            self._state = PluginState.FAILED
            raise PluginError(
                f"Failed to start plugin: {e}",
                plugin_name=self.metadata.name,
                cause=e
            )
    
    def _start(self, context: PluginContext) -> None:
        """Override to implement startup logic."""
        pass
    
    def stop(self, context: PluginContext) -> None:
        """Stop the plugin.
        
        Args:
            context: Plugin context
        """
        logger.debug(f"Stopping plugin: {self.metadata.name}")
        
        if self._state != PluginState.RUNNING:
            return
        
        try:
            self._stop(context)
            self._state = PluginState.STOPPED
        except Exception as e:
            logger.error(f"Error stopping plugin {self.metadata.name}: {e}")
            self._state = PluginState.FAILED
    
    def _stop(self, context: PluginContext) -> None:
        """Override to implement stop logic."""
        pass
    
    def cleanup(self, context: PluginContext) -> None:
        """Clean up plugin resources.
        
        Args:
            context: Plugin context
        """
        logger.debug(f"Cleaning up plugin: {self.metadata.name}")
        
        try:
            self._cleanup(context)
            self._state = PluginState.CLEANED_UP
        except Exception as e:
            logger.error(f"Error cleaning up plugin {self.metadata.name}: {e}")
    
    def _cleanup(self, context: PluginContext) -> None:
        """Override to implement cleanup logic."""
        pass


class PluginBase(PluginLifecycle):
    """Base class for concrete plugin implementations."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize plugin with configuration.
        
        Args:
            config: Plugin-specific configuration
        """
        super().__init__()
        self.config = config or {}
    
    def _create_metadata(self) -> PluginMetadata:
        """Create default metadata from class attributes."""
        return PluginMetadata(
            name=getattr(self, "NAME", self.__class__.__name__),
            version=getattr(self, "VERSION", "1.0.0"),
            description=getattr(self, "DESCRIPTION", self.__class__.__doc__),
            author=getattr(self, "AUTHOR", None),
            email=getattr(self, "EMAIL", None),
            tags=getattr(self, "TAGS", []),
            requirements=getattr(self, "REQUIREMENTS", []),
            category=getattr(self, "CATEGORY", None),
            depends_on=getattr(self, "DEPENDS_ON", []),
            conflicts_with=getattr(self, "CONFLICTS_WITH", []),
            provides=getattr(self, "PROVIDES", []),
            consumes=getattr(self, "CONSUMES", []),
        )
    
    def _initialize(self, context: PluginContext) -> None:
        """Default initialization - can be overridden."""
        pass