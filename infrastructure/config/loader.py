"""Configuration loading utilities.

This module provides utilities for loading configuration from various sources:
- YAML files
- JSON files  
- Environment variables
- Command line arguments
"""

import json
import yaml
from pathlib import Path
from typing import Any, Dict, List, Union

try:
    from yaml import CLoader as YamlLoader
except ImportError:
    from yaml import Loader as YamlLoader


class ConfigurationLoader:
    """Utility class for loading configuration from various sources."""
    
    def load_yaml(self, path: Union[str, Path]) -> Dict[str, Any]:
        """Load configuration from YAML file.
        
        Args:
            path: Path to YAML file
            
        Returns:
            Configuration dictionary
            
        Raises:
            FileNotFoundError: If file doesn't exist
            yaml.YAMLError: If YAML is invalid
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Configuration file not found: {path}")
            
        try:
            with open(path, 'r', encoding='utf-8') as f:
                content = yaml.load(f, Loader=YamlLoader) or {}
                
            if not isinstance(content, dict):
                raise ValueError(f"Configuration file must contain a dictionary: {path}")
                
            return content
            
        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML in {path}: {e}")
            
    def load_json(self, path: Union[str, Path]) -> Dict[str, Any]:
        """Load configuration from JSON file.
        
        Args:
            path: Path to JSON file
            
        Returns:
            Configuration dictionary
            
        Raises:
            FileNotFoundError: If file doesn't exist
            json.JSONDecodeError: If JSON is invalid
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Configuration file not found: {path}")
            
        try:
            with open(path, 'r', encoding='utf-8') as f:
                content = json.load(f) or {}
                
            if not isinstance(content, dict):
                raise ValueError(f"Configuration file must contain a dictionary: {path}")
                
            return content
            
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in {path}: {e}")
            
    def merge_configs(self, base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        """Merge two configuration dictionaries with deep merging.
        
        The override dict takes precedence over the base dict.
        Nested dictionaries are merged recursively.
        
        Args:
            base: Base configuration
            override: Override configuration
            
        Returns:
            Merged configuration
        """
        import copy
        result = copy.deepcopy(base)
        
        for key, value in override.items():
            if (key in result and 
                isinstance(result[key], dict) and 
                isinstance(value, dict)):
                # Recursively merge nested dictionaries
                result[key] = self.merge_configs(result[key], value)
            else:
                # Override the value
                result[key] = copy.deepcopy(value)
                
        return result
        
    def load_multiple(self, paths: List[Union[str, Path]]) -> Dict[str, Any]:
        """Load and merge multiple configuration files.
        
        Files are loaded in order, with later files overriding earlier ones.
        
        Args:
            paths: List of configuration file paths
            
        Returns:
            Merged configuration
        """
        result = {}
        
        for path in paths:
            path = Path(path)
            if not path.exists():
                continue
                
            if path.suffix.lower() in ('.yaml', '.yml'):
                config = self.load_yaml(path)
            elif path.suffix.lower() == '.json':
                config = self.load_json(path)
            else:
                raise ValueError(f"Unsupported configuration file format: {path}")
                
            result = self.merge_configs(result, config)
            
        return result
        
    def save_yaml(self, config: Dict[str, Any], path: Union[str, Path]) -> None:
        """Save configuration to YAML file.
        
        Args:
            config: Configuration to save
            path: Path to save to
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'w', encoding='utf-8') as f:
            yaml.dump(
                config, 
                f, 
                default_flow_style=False,
                indent=2,
                sort_keys=False
            )
            
    def save_json(self, config: Dict[str, Any], path: Union[str, Path]) -> None:
        """Save configuration to JSON file.
        
        Args:
            config: Configuration to save
            path: Path to save to
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, sort_keys=False)
            
    def resolve_paths(self, config: Dict[str, Any], base_path: Path) -> Dict[str, Any]:
        """Resolve relative paths in configuration relative to base path.
        
        Args:
            config: Configuration with potentially relative paths
            base_path: Base path for resolution
            
        Returns:
            Configuration with resolved paths
        """
        import copy
        result = copy.deepcopy(config)
        
        def _resolve_recursive(obj: Any) -> Any:
            if isinstance(obj, dict):
                return {k: _resolve_recursive(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [_resolve_recursive(item) for item in obj]
            elif isinstance(obj, str):
                # Check if this looks like a path
                if ('/' in obj or '\\' in obj) and not obj.startswith(('http://', 'https://')):
                    path = Path(obj)
                    if not path.is_absolute():
                        return str(base_path / path)
                return obj
            else:
                return obj
                
        return _resolve_recursive(result)
        
    def expand_variables(self, config: Dict[str, Any], variables: Dict[str, str]) -> Dict[str, Any]:
        """Expand variables in configuration strings.
        
        Variables are specified as ${VAR_NAME} in configuration values.
        
        Args:
            config: Configuration with variables
            variables: Variable name -> value mapping
            
        Returns:
            Configuration with variables expanded
        """
        import copy
        import re
        result = copy.deepcopy(config)
        
        var_pattern = re.compile(r'\$\{([^}]+)\}')
        
        def _expand_recursive(obj: Any) -> Any:
            if isinstance(obj, dict):
                return {k: _expand_recursive(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [_expand_recursive(item) for item in obj]
            elif isinstance(obj, str):
                def replace_var(match):
                    var_name = match.group(1)
                    return variables.get(var_name, match.group(0))
                return var_pattern.sub(replace_var, obj)
            else:
                return obj
                
        return _expand_recursive(result)