"""Secondary configuration port - Configuration services that the application depends on.

This port defines the configuration interface that the application core uses
to load and manage configuration. It's a driven port implemented by adapters
for different configuration sources (YAML, JSON, environment variables, etc.).
"""

from pathlib import Path
from typing import Any, Protocol, TypeVar, runtime_checkable, Callable, Optional

from typing_extensions import TypeAlias
from infrastructure.di import port

# Type aliases
ConfigValue: TypeAlias = Any
ConfigDict: TypeAlias = dict[str, Any]
ConfigPath: TypeAlias = str | Path

T = TypeVar("T")


@port()
@runtime_checkable
class ConfigurationProvider(Protocol):
    """Secondary port for configuration management.
    
    This interface is implemented by adapters for specific configuration
    sources and formats. The application core depends on this for all
    configuration needs.
    """

    def load(
        self,
        path: ConfigPath,
        environment: Optional[str] = None,
        overrides: Optional[ConfigDict] = None
    ) -> ConfigDict:
        """Load configuration from a source.
        
        Args:
            path: Configuration path/identifier
            environment: Optional environment name
            overrides: Optional configuration overrides
            
        Returns:
            Configuration dictionary
            
        Raises:
            FileNotFoundError: If configuration not found
            ValueError: If configuration is invalid
        """
        ...

    def save(
        self,
        config: ConfigDict,
        path: ConfigPath,
        format: Optional[str] = None
    ) -> None:
        """Save configuration to a destination.
        
        Args:
            config: Configuration to save
            path: Save path
            format: Optional format hint
        """
        ...

    def validate(
        self,
        config: ConfigDict,
        schema: type[T] | dict[str, Any] | None = None
    ) -> T | ConfigDict:
        """Validate configuration against a schema.
        
        Args:
            config: Configuration to validate
            schema: Optional schema (Pydantic model or dict schema)
            
        Returns:
            Validated configuration
            
        Raises:
            ValueError: If validation fails
        """
        ...

    def merge(
        self,
        *configs: ConfigDict,
        deep: bool = True
    ) -> ConfigDict:
        """Merge multiple configurations.
        
        Args:
            *configs: Configurations to merge (later overrides earlier)
            deep: Whether to deep merge nested dicts
            
        Returns:
            Merged configuration
        """
        ...

    def get(
        self,
        config: ConfigDict,
        key: str,
        default: Any = None,
        required: bool = False
    ) -> ConfigValue:
        """Get a configuration value by key.
        
        Args:
            config: Configuration dict
            key: Dot-separated key path (e.g., "model.hidden_size")
            default: Default value if key not found
            required: Whether to raise if key not found
            
        Returns:
            Configuration value
            
        Raises:
            KeyError: If required=True and key not found
        """
        ...

    def set(
        self,
        config: ConfigDict,
        key: str,
        value: ConfigValue
    ) -> ConfigDict:
        """Set a configuration value by key.
        
        Args:
            config: Configuration dict
            key: Dot-separated key path
            value: Value to set
            
        Returns:
            Updated configuration
        """
        ...

    def expand_vars(
        self,
        config: ConfigDict,
        env_vars: dict[str, str] | None = None
    ) -> ConfigDict:
        """Expand environment variables in configuration.
        
        Args:
            config: Configuration with potential variables
            env_vars: Optional environment variables to use
            
        Returns:
            Configuration with expanded variables
        """
        ...

    def to_flat(
        self,
        config: ConfigDict,
        separator: str = "."
    ) -> dict[str, ConfigValue]:
        """Flatten nested configuration to flat key-value pairs.
        
        Args:
            config: Nested configuration
            separator: Key separator
            
        Returns:
            Flat configuration
        """
        ...

    def from_flat(
        self,
        flat_config: dict[str, ConfigValue],
        separator: str = "."
    ) -> ConfigDict:
        """Reconstruct nested configuration from flat key-value pairs.
        
        Args:
            flat_config: Flat configuration
            separator: Key separator
            
        Returns:
            Nested configuration
        """
        ...

    def watch(
        self,
        path: ConfigPath,
        callback: Callable[[ConfigDict], None],
        interval: float = 1.0
    ) -> None:
        """Watch configuration file for changes.
        
        Args:
            path: Configuration path to watch
            callback: Function to call on changes
            interval: Check interval in seconds
        """
        ...

    def diff(
        self,
        config1: ConfigDict,
        config2: ConfigDict
    ) -> dict[str, tuple[Any, Any]]:
        """Compare two configurations.
        
        Args:
            config1: First configuration
            config2: Second configuration
            
        Returns:
            Dictionary of differences (key -> (value1, value2))
        """
        ...


@port()
@runtime_checkable
class ConfigRegistry(Protocol):
    """Registry for managing multiple configuration sources.
    
    This allows the application to work with configurations from
    multiple sources with priority-based merging.
    """

    def register_source(
        self,
        name: str,
        provider: ConfigurationProvider,
        priority: int = 0
    ) -> None:
        """Register a configuration source.
        
        Args:
            name: Source name
            provider: Configuration provider
            priority: Source priority (higher = higher precedence)
        """
        ...

    def unregister_source(self, name: str) -> None:
        """Unregister a configuration source.
        
        Args:
            name: Source name to remove
        """
        ...

    def load_all(
        self,
        environment: Optional[str] = None
    ) -> ConfigDict:
        """Load and merge all registered configurations.
        
        Args:
            environment: Optional environment name
            
        Returns:
            Merged configuration from all sources
        """
        ...

    def get_source(self, name: str) -> Optional[ConfigurationProvider]:
        """Get a specific configuration source.
        
        Args:
            name: Source name
            
        Returns:
            Configuration provider or None
        """
        ...

    def list_sources(self) -> list[tuple[str, int]]:
        """List all registered sources.
        
        Returns:
            List of (name, priority) tuples
        """
        ...

    def reload(self) -> ConfigDict:
        """Reload all configuration sources.
        
        Returns:
            Updated merged configuration
        """
        ...

    def set_environment(self, environment: str) -> None:
        """Set the active environment.
        
        Args:
            environment: Environment name
        """
        ...

    def get_environment(self) -> Optional[str]:
        """Get the active environment.
        
        Returns:
            Environment name or None
        """
        ...