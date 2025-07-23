"""Plugin configuration support.

This module provides:
- Plugin configuration schema
- Configuration loading and validation
- Configuration merging and inheritance
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import yaml
from loguru import logger
from pydantic import BaseModel, Field, validator


class PluginConfigSchema(BaseModel):
    """Schema for plugin configuration."""
    
    # Plugin selection
    enabled: List[str] = Field(
        default_factory=list,
        description="List of enabled plugins"
    )
    disabled: List[str] = Field(
        default_factory=list,
        description="List of explicitly disabled plugins"
    )
    
    # Plugin paths
    paths: List[str] = Field(
        default_factory=list,
        description="Additional paths to search for plugins"
    )
    
    # Plugin configuration
    configs: Dict[str, Dict[str, Any]] = Field(
        default_factory=dict,
        description="Plugin-specific configurations"
    )
    
    # Discovery settings
    discover_project: bool = Field(
        True,
        description="Whether to discover plugins from project"
    )
    discover_entry_points: bool = Field(
        True,
        description="Whether to discover plugins from entry points"
    )
    
    # Validation settings
    validate_on_load: bool = Field(
        True,
        description="Whether to validate plugins on load"
    )
    fail_on_validation_error: bool = Field(
        False,
        description="Whether to fail if any plugin validation fails"
    )
    
    # Initialization settings
    auto_initialize: bool = Field(
        True,
        description="Whether to automatically initialize plugins"
    )
    auto_start: bool = Field(
        False,
        description="Whether to automatically start plugins"
    )
    
    # Categories to load
    categories: Optional[List[str]] = Field(
        None,
        description="Specific categories to load (None means all)"
    )
    
    @validator("paths")
    def expand_paths(cls, v: List[str]) -> List[str]:
        """Expand paths to absolute paths."""
        expanded = []
        for path in v:
            expanded_path = Path(path).expanduser().absolute()
            expanded.append(str(expanded_path))
        return expanded
    
    class Config:
        extra = "allow"  # Allow additional fields


class PluginConfig:
    """Manages plugin configuration."""
    
    def __init__(self, config: Optional[Union[Dict[str, Any], PluginConfigSchema]] = None):
        """Initialize plugin configuration.
        
        Args:
            config: Configuration dict or schema
        """
        if config is None:
            self.schema = PluginConfigSchema()
        elif isinstance(config, PluginConfigSchema):
            self.schema = config
        else:
            self.schema = PluginConfigSchema(**config)
    
    @classmethod
    def from_file(cls, path: Union[str, Path]) -> "PluginConfig":
        """Load configuration from file.
        
        Args:
            path: Path to configuration file
            
        Returns:
            PluginConfig instance
        """
        path = Path(path)
        
        if not path.exists():
            raise FileNotFoundError(f"Configuration file not found: {path}")
        
        # Load based on extension
        if path.suffix in (".yaml", ".yml"):
            with open(path) as f:
                data = yaml.safe_load(f) or {}
        elif path.suffix == ".json":
            import json
            with open(path) as f:
                data = json.load(f)
        else:
            raise ValueError(f"Unsupported configuration format: {path.suffix}")
        
        # Extract plugin configuration
        plugin_config = data.get("plugins", {})
        
        return cls(plugin_config)
    
    @classmethod
    def from_project(cls, project_root: Union[str, Path]) -> "PluginConfig":
        """Load configuration from project.
        
        Args:
            project_root: Project root directory
            
        Returns:
            PluginConfig instance
        """
        project_root = Path(project_root)
        
        # Look for k-bert.yaml
        config_file = project_root / "k-bert.yaml"
        if config_file.exists():
            return cls.from_file(config_file)
        
        # Look for pyproject.toml
        pyproject_file = project_root / "pyproject.toml"
        if pyproject_file.exists():
            try:
                import toml
                with open(pyproject_file) as f:
                    data = toml.load(f)
                
                plugin_config = data.get("tool", {}).get("k-bert", {}).get("plugins", {})
                return cls(plugin_config)
            except Exception as e:
                logger.debug(f"Could not load plugin config from pyproject.toml: {e}")
        
        # Return default configuration
        return cls()
    
    def merge(self, other: "PluginConfig") -> "PluginConfig":
        """Merge with another configuration.
        
        Args:
            other: Other configuration
            
        Returns:
            New merged configuration
        """
        # Convert to dicts
        self_dict = self.schema.dict()
        other_dict = other.schema.dict()
        
        # Merge configurations
        merged = self._deep_merge(self_dict, other_dict)
        
        return PluginConfig(merged)
    
    def get_plugin_config(self, plugin_name: str) -> Dict[str, Any]:
        """Get configuration for a specific plugin.
        
        Args:
            plugin_name: Plugin name
            
        Returns:
            Plugin configuration
        """
        return self.schema.configs.get(plugin_name, {})
    
    def is_enabled(self, plugin_name: str) -> bool:
        """Check if a plugin is enabled.
        
        Args:
            plugin_name: Plugin name
            
        Returns:
            True if enabled
        """
        # Explicitly disabled
        if plugin_name in self.schema.disabled:
            return False
        
        # If enabled list is specified, must be in it
        if self.schema.enabled:
            return plugin_name in self.schema.enabled
        
        # Default to enabled
        return True
    
    def should_load_category(self, category: str) -> bool:
        """Check if a category should be loaded.
        
        Args:
            category: Category name
            
        Returns:
            True if should load
        """
        if self.schema.categories is None:
            return True
        return category in self.schema.categories
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary.
        
        Returns:
            Configuration dictionary
        """
        return self.schema.dict()
    
    def save(self, path: Union[str, Path]) -> None:
        """Save configuration to file.
        
        Args:
            path: Path to save to
        """
        path = Path(path)
        data = {"plugins": self.to_dict()}
        
        if path.suffix in (".yaml", ".yml"):
            with open(path, "w") as f:
                yaml.safe_dump(data, f, default_flow_style=False)
        elif path.suffix == ".json":
            import json
            with open(path, "w") as f:
                json.dump(data, f, indent=2)
        else:
            raise ValueError(f"Unsupported configuration format: {path.suffix}")
    
    def _deep_merge(self, dict1: Dict[str, Any], dict2: Dict[str, Any]) -> Dict[str, Any]:
        """Deep merge two dictionaries.
        
        Args:
            dict1: First dictionary
            dict2: Second dictionary (takes precedence)
            
        Returns:
            Merged dictionary
        """
        result = dict1.copy()
        
        for key, value in dict2.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            elif key in result and isinstance(result[key], list) and isinstance(value, list):
                # For lists, extend rather than replace
                result[key] = list(set(result[key] + value))
            else:
                result[key] = value
        
        return result


def load_plugin_config(
    project_root: Optional[Union[str, Path]] = None,
    config_file: Optional[Union[str, Path]] = None,
    config_dict: Optional[Dict[str, Any]] = None,
) -> PluginConfig:
    """Load plugin configuration from various sources.
    
    Args:
        project_root: Project root directory
        config_file: Specific configuration file
        config_dict: Configuration dictionary
        
    Returns:
        PluginConfig instance
    """
    configs = []
    
    # Load from project if specified
    if project_root:
        try:
            project_config = PluginConfig.from_project(project_root)
            configs.append(project_config)
        except Exception as e:
            logger.debug(f"Could not load project config: {e}")
    
    # Load from file if specified
    if config_file:
        file_config = PluginConfig.from_file(config_file)
        configs.append(file_config)
    
    # Load from dict if specified
    if config_dict:
        dict_config = PluginConfig(config_dict)
        configs.append(dict_config)
    
    # Merge all configurations
    if not configs:
        return PluginConfig()
    
    result = configs[0]
    for config in configs[1:]:
        result = result.merge(config)
    
    return result