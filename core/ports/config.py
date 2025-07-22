"""Configuration provider port interface.

This port abstracts configuration management, allowing the core domain
to be independent of specific configuration formats and sources.
"""

from pathlib import Path
from typing import Any, Protocol, TypeVar, runtime_checkable

from typing_extensions import TypeAlias

# Type aliases
ConfigValue: TypeAlias = Any
ConfigDict: TypeAlias = dict[str, Any]
ConfigPath: TypeAlias = str | Path

T = TypeVar("T")


@runtime_checkable
class ConfigurationProvider(Protocol):
    """Port for configuration management."""

    def load(
        self,
        path: ConfigPath,
        environment: str | None = None,
        overrides: ConfigDict | None = None
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
        format: str | None = None
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


@runtime_checkable
class ConfigRegistry(Protocol):
    """Registry for managing multiple configuration sources."""

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
        environment: str | None = None
    ) -> ConfigDict:
        """Load and merge all registered configurations.
        
        Args:
            environment: Optional environment name
            
        Returns:
            Merged configuration from all sources
        """
        ...

    def get_source(self, name: str) -> ConfigurationProvider | None:
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