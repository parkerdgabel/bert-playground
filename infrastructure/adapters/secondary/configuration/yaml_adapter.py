"""YAML configuration adapter implementation."""

import os
from pathlib import Path
from typing import Any, TypeVar

import yaml
from pydantic import BaseModel, ValidationError

from infrastructure.di import adapter, Scope
from application.ports.secondary.configuration import (
    ConfigDict,
    ConfigPath,
    ConfigRegistry,
    ConfigValue,
    ConfigurationProvider,
)

T = TypeVar("T")


@adapter(ConfigurationProvider, scope=Scope.SINGLETON)
class YamlConfigurationAdapter:
    """YAML implementation of the ConfigurationProvider port."""

    def load(
        self,
        path: ConfigPath,
        environment: str | None = None,
        overrides: ConfigDict | None = None
    ) -> ConfigDict:
        """Load configuration from a YAML file."""
        path = Path(path)
        
        if not path.exists():
            raise FileNotFoundError(f"Configuration file not found: {path}")
        
        with open(path, "r") as f:
            config = yaml.safe_load(f) or {}
        
        # Apply environment-specific overrides
        if environment and "environments" in config:
            env_config = config.get("environments", {}).get(environment, {})
            config = self.merge(config, env_config)
            # Remove environments section after merging
            config.pop("environments", None)
        
        # Apply provided overrides
        if overrides:
            config = self.merge(config, overrides)
        
        # Expand environment variables
        config = self.expand_vars(config)
        
        return config

    def save(
        self,
        config: ConfigDict,
        path: ConfigPath,
        format: str | None = None
    ) -> None:
        """Save configuration to a YAML file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, "w") as f:
            yaml.dump(
                config,
                f,
                default_flow_style=False,
                sort_keys=False,
                indent=2
            )

    def validate(
        self,
        config: ConfigDict,
        schema: type[T] | dict[str, Any] | None = None
    ) -> T | ConfigDict:
        """Validate configuration against a schema."""
        if schema is None:
            return config
        
        # If schema is a Pydantic model
        if isinstance(schema, type) and issubclass(schema, BaseModel):
            try:
                return schema(**config)
            except ValidationError as e:
                raise ValueError(f"Configuration validation failed: {e}")
        
        # If schema is a dict, perform basic type checking
        if isinstance(schema, dict):
            validated = {}
            for key, expected_type in schema.items():
                if key in config:
                    value = config[key]
                    if not isinstance(value, expected_type):
                        raise ValueError(
                            f"Configuration key '{key}' expected type "
                            f"{expected_type.__name__}, got {type(value).__name__}"
                        )
                    validated[key] = value
            return validated
        
        return config

    def merge(
        self,
        *configs: ConfigDict,
        deep: bool = True
    ) -> ConfigDict:
        """Merge multiple configurations."""
        if not configs:
            return {}
        
        result = configs[0].copy()
        
        for config in configs[1:]:
            if deep:
                result = self._deep_merge(result, config)
            else:
                result.update(config)
        
        return result

    def _deep_merge(self, base: ConfigDict, override: ConfigDict) -> ConfigDict:
        """Deep merge two dictionaries."""
        result = base.copy()
        
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value
        
        return result

    def get(
        self,
        config: ConfigDict,
        key: str,
        default: Any = None,
        required: bool = False
    ) -> ConfigValue:
        """Get a configuration value by key."""
        keys = key.split(".")
        value = config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                if required:
                    raise KeyError(f"Required configuration key not found: {key}")
                return default
        
        return value

    def set(
        self,
        config: ConfigDict,
        key: str,
        value: ConfigValue
    ) -> ConfigDict:
        """Set a configuration value by key."""
        keys = key.split(".")
        result = config.copy()
        current = result
        
        # Navigate to the parent of the target key
        for k in keys[:-1]:
            if k not in current:
                current[k] = {}
            current = current[k]
        
        # Set the final value
        current[keys[-1]] = value
        
        return result

    def expand_vars(
        self,
        config: ConfigDict,
        env_vars: dict[str, str] | None = None
    ) -> ConfigDict:
        """Expand environment variables in configuration."""
        env_vars = env_vars or os.environ
        
        def expand_value(value: Any) -> Any:
            if isinstance(value, str):
                # Expand ${VAR} and $VAR patterns
                for var_name, var_value in env_vars.items():
                    value = value.replace(f"${{{var_name}}}", var_value)
                    value = value.replace(f"${var_name}", var_value)
                return value
            elif isinstance(value, dict):
                return {k: expand_value(v) for k, v in value.items()}
            elif isinstance(value, list):
                return [expand_value(v) for v in value]
            else:
                return value
        
        return expand_value(config)

    def to_flat(
        self,
        config: ConfigDict,
        separator: str = "."
    ) -> dict[str, ConfigValue]:
        """Flatten nested configuration to flat key-value pairs."""
        flat = {}
        
        def flatten(obj: Any, prefix: str = "") -> None:
            if isinstance(obj, dict):
                for key, value in obj.items():
                    new_key = f"{prefix}{separator}{key}" if prefix else key
                    flatten(value, new_key)
            elif isinstance(obj, list):
                for i, value in enumerate(obj):
                    new_key = f"{prefix}[{i}]"
                    flatten(value, new_key)
            else:
                flat[prefix] = obj
        
        flatten(config)
        return flat

    def from_flat(
        self,
        flat_config: dict[str, ConfigValue],
        separator: str = "."
    ) -> ConfigDict:
        """Reconstruct nested configuration from flat key-value pairs."""
        result: ConfigDict = {}
        
        for flat_key, value in flat_config.items():
            # Handle array indices
            parts = []
            current_part = ""
            in_bracket = False
            
            for char in flat_key:
                if char == "[":
                    if current_part:
                        parts.append(current_part)
                        current_part = ""
                    in_bracket = True
                elif char == "]":
                    if current_part:
                        parts.append(int(current_part))
                        current_part = ""
                    in_bracket = False
                elif char == separator and not in_bracket:
                    if current_part:
                        parts.append(current_part)
                        current_part = ""
                else:
                    current_part += char
            
            if current_part:
                parts.append(current_part)
            
            # Build nested structure
            current = result
            for i, part in enumerate(parts[:-1]):
                if isinstance(part, int):
                    # Handle list index
                    if not isinstance(current, list):
                        raise ValueError(f"Expected list at {parts[:i]}")
                    while len(current) <= part:
                        current.append(None)
                    if current[part] is None:
                        # Determine if next part is int (list) or str (dict)
                        next_part = parts[i + 1]
                        current[part] = [] if isinstance(next_part, int) else {}
                    current = current[part]
                else:
                    # Handle dict key
                    if part not in current:
                        # Determine if next part is int (list) or str (dict)
                        next_part = parts[i + 1]
                        current[part] = [] if isinstance(next_part, int) else {}
                    current = current[part]
            
            # Set final value
            final_part = parts[-1]
            if isinstance(final_part, int):
                while len(current) <= final_part:
                    current.append(None)
                current[final_part] = value
            else:
                current[final_part] = value
        
        return result


class ConfigRegistryImpl:
    """Implementation of ConfigRegistry for managing multiple configuration sources."""

    def __init__(self):
        """Initialize the registry."""
        self._sources: dict[str, tuple[ConfigurationProvider, int]] = {}

    def register_source(
        self,
        name: str,
        provider: ConfigurationProvider,
        priority: int = 0
    ) -> None:
        """Register a configuration source."""
        self._sources[name] = (provider, priority)

    def unregister_source(self, name: str) -> None:
        """Unregister a configuration source."""
        if name in self._sources:
            del self._sources[name]

    def load_all(
        self,
        environment: str | None = None
    ) -> ConfigDict:
        """Load and merge all registered configurations."""
        # Sort sources by priority (higher priority last)
        sorted_sources = sorted(
            self._sources.items(),
            key=lambda x: x[1][1]
        )
        
        # Load and merge configurations
        result = {}
        yaml_adapter = YAMLConfigAdapter()  # For merging
        
        for name, (provider, _) in sorted_sources:
            try:
                config = provider.load("", environment=environment)
                result = yaml_adapter.merge(result, config)
            except FileNotFoundError:
                # Skip missing sources
                pass
        
        return result

    def get_source(self, name: str) -> ConfigurationProvider | None:
        """Get a specific configuration source."""
        if name in self._sources:
            return self._sources[name][0]
        return None

    def list_sources(self) -> list[tuple[str, int]]:
        """List all registered sources."""
        return [(name, priority) for name, (_, priority) in self._sources.items()]