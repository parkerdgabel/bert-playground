"""YAML and JSON configuration loader with advanced features.

This module provides a unified configuration loading system that supports:
- YAML and JSON formats
- Environment variable substitution
- File includes
- Schema validation
- Type conversion for dataclasses and enums
"""

import json
import os
import re
from pathlib import Path
from typing import Any, Dict, Optional, Type, TypeVar, Union

import yaml
from loguru import logger

T = TypeVar("T")


class ConfigurationError(Exception):
    """Configuration loading error."""
    pass


class EnvVarLoader(yaml.SafeLoader):
    """YAML loader with environment variable substitution."""
    pass


class IncludeLoader(yaml.SafeLoader):
    """YAML loader with file inclusion support."""
    
    def __init__(self, stream):
        self._root = Path(stream.name).parent if hasattr(stream, 'name') else Path.cwd()
        super().__init__(stream)


def env_var_constructor(loader: yaml.SafeLoader, node: yaml.ScalarNode) -> str:
    """Handle environment variable substitution in YAML.
    
    Supports syntax: ${VAR_NAME:default_value}
    """
    value = loader.construct_scalar(node)
    pattern = re.compile(r'\$\{([^}]+)\}')
    
    def replacer(match):
        env_var = match.group(1)
        if ':' in env_var:
            var_name, default = env_var.split(':', 1)
            return os.environ.get(var_name, default)
        return os.environ.get(env_var, match.group(0))
    
    return pattern.sub(replacer, value)


def include_constructor(loader: IncludeLoader, node: yaml.ScalarNode) -> Dict[str, Any]:
    """Handle file includes in YAML.
    
    Supports syntax: !include path/to/file.yaml
    """
    include_path = loader.construct_scalar(node)
    file_path = loader._root / include_path
    
    if not file_path.exists():
        raise ConfigurationError(f"Include file not found: {file_path}")
    
    with open(file_path, 'r') as f:
        return yaml.load(f, Loader=IncludeLoader)


def path_constructor(loader: yaml.SafeLoader, node: yaml.ScalarNode) -> Path:
    """Handle path construction in YAML.
    
    Supports syntax: !path path/to/file
    """
    path_str = loader.construct_scalar(node)
    # Expand environment variables in paths
    path_str = os.path.expandvars(path_str)
    path_str = os.path.expanduser(path_str)
    return Path(path_str)


# Register custom constructors
yaml.add_implicit_resolver('!env_var', re.compile(r'.*\$\{.*\}.*'), None, EnvVarLoader)
yaml.add_constructor('!env_var', env_var_constructor, EnvVarLoader)
yaml.add_constructor('!include', include_constructor, IncludeLoader)
yaml.add_constructor('!path', path_constructor, yaml.SafeLoader)
yaml.add_constructor('!path', path_constructor, EnvVarLoader)
yaml.add_constructor('!path', path_constructor, IncludeLoader)


class ConfigLoader:
    """Unified configuration loader for YAML and JSON formats."""
    
    @staticmethod
    def detect_format(file_path: Union[str, Path]) -> str:
        """Detect configuration format from file extension."""
        path = Path(file_path)
        extension = path.suffix.lower()
        
        if extension in ['.yaml', '.yml']:
            return 'yaml'
        elif extension == '.json':
            return 'json'
        else:
            raise ConfigurationError(
                f"Unsupported configuration format: {extension}. "
                "Supported formats: .yaml, .yml, .json"
            )
    
    @classmethod
    def load(
        cls,
        file_path: Union[str, Path],
        format: Optional[str] = None,
        schema: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Load configuration from file.
        
        Args:
            file_path: Path to configuration file
            format: Force specific format ('yaml' or 'json'), auto-detect if None
            schema: Optional schema for validation
            
        Returns:
            Configuration dictionary
        """
        path = Path(file_path)
        
        if not path.exists():
            raise ConfigurationError(f"Configuration file not found: {path}")
        
        # Detect format if not specified
        if format is None:
            format = cls.detect_format(path)
        
        # Load configuration
        try:
            with open(path, 'r') as f:
                if format == 'yaml':
                    # Use appropriate loader based on features needed
                    if '!include' in f.read():
                        f.seek(0)
                        config = yaml.load(f, Loader=IncludeLoader)
                    else:
                        f.seek(0)
                        config = yaml.load(f, Loader=EnvVarLoader)
                elif format == 'json':
                    config = json.load(f)
                else:
                    raise ConfigurationError(f"Unsupported format: {format}")
        except Exception as e:
            raise ConfigurationError(f"Failed to load configuration: {e}")
        
        # Validate against schema if provided
        if schema:
            cls.validate(config, schema)
        
        logger.info(f"Loaded {format.upper()} configuration from {path}")
        return config
    
    @classmethod
    def save(
        cls,
        config: Dict[str, Any],
        file_path: Union[str, Path],
        format: Optional[str] = None,
        create_backup: bool = True,
    ) -> None:
        """Save configuration to file.
        
        Args:
            config: Configuration dictionary
            file_path: Path to save configuration
            format: Output format ('yaml' or 'json'), auto-detect if None
            create_backup: Create backup of existing file
        """
        path = Path(file_path)
        
        # Detect format if not specified
        if format is None:
            format = cls.detect_format(path)
        
        # Create backup if file exists
        if create_backup and path.exists():
            backup_path = path.with_suffix(path.suffix + '.bak')
            path.rename(backup_path)
            logger.info(f"Created backup: {backup_path}")
        
        # Ensure directory exists
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save configuration
        try:
            with open(path, 'w') as f:
                if format == 'yaml':
                    yaml.dump(
                        config,
                        f,
                        default_flow_style=False,
                        sort_keys=False,
                        allow_unicode=True,
                        width=120,
                    )
                elif format == 'json':
                    json.dump(config, f, indent=2, default=str)
                else:
                    raise ConfigurationError(f"Unsupported format: {format}")
        except Exception as e:
            raise ConfigurationError(f"Failed to save configuration: {e}")
        
        logger.info(f"Saved {format.upper()} configuration to {path}")
    
    @staticmethod
    def validate(config: Dict[str, Any], schema: Dict[str, Any]) -> None:
        """Validate configuration against schema.
        
        This is a simple validation that checks required fields and types.
        Can be extended with more sophisticated schema validation.
        """
        def check_schema(data: Dict[str, Any], schema: Dict[str, Any], path: str = "") -> None:
            for key, spec in schema.items():
                current_path = f"{path}.{key}" if path else key
                
                # Check if required
                if isinstance(spec, dict) and spec.get('required', False) and key not in data:
                    raise ConfigurationError(f"Required field missing: {current_path}")
                
                if key in data:
                    value = data[key]
                    
                    # Check type
                    if isinstance(spec, dict) and 'type' in spec:
                        expected_type = spec['type']
                        if not isinstance(value, expected_type):
                            raise ConfigurationError(
                                f"Type mismatch at {current_path}: "
                                f"expected {expected_type.__name__}, got {type(value).__name__}"
                            )
                    
                    # Recursively check nested objects
                    if isinstance(spec, dict) and 'properties' in spec and isinstance(value, dict):
                        check_schema(value, spec['properties'], current_path)
        
        check_schema(config, schema)
    
    @classmethod
    def merge_configs(
        cls,
        *configs: Dict[str, Any],
        deep: bool = True,
    ) -> Dict[str, Any]:
        """Merge multiple configurations.
        
        Later configurations override earlier ones.
        
        Args:
            configs: Configuration dictionaries to merge
            deep: Perform deep merge for nested dictionaries
            
        Returns:
            Merged configuration
        """
        result = {}
        
        for config in configs:
            if deep:
                result = cls._deep_merge(result, config)
            else:
                result.update(config)
        
        return result
    
    @staticmethod
    def _deep_merge(base: Dict[str, Any], update: Dict[str, Any]) -> Dict[str, Any]:
        """Deep merge two dictionaries."""
        result = base.copy()
        
        for key, value in update.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = ConfigLoader._deep_merge(result[key], value)
            else:
                result[key] = value
        
        return result
    
    @classmethod
    def convert_json_to_yaml(
        cls,
        json_path: Union[str, Path],
        yaml_path: Optional[Union[str, Path]] = None,
    ) -> Path:
        """Convert JSON configuration to YAML format.
        
        Args:
            json_path: Path to JSON configuration
            yaml_path: Output YAML path (auto-generate if None)
            
        Returns:
            Path to created YAML file
        """
        json_path = Path(json_path)
        
        if yaml_path is None:
            yaml_path = json_path.with_suffix('.yaml')
        else:
            yaml_path = Path(yaml_path)
        
        # Load JSON
        config = cls.load(json_path, format='json')
        
        # Save as YAML
        cls.save(config, yaml_path, format='yaml', create_backup=False)
        
        logger.info(f"Converted {json_path} to {yaml_path}")
        return yaml_path