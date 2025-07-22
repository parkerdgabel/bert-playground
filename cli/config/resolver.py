"""Configuration resolver component for k-bert CLI.

This module handles resolving environment variables, paths, and other
dynamic values in configurations.
"""

import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol, Union
from abc import abstractmethod

from loguru import logger


class ConfigResolverProtocol(Protocol):
    """Protocol for configuration resolvers."""
    
    @abstractmethod
    def resolve(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Resolve all dynamic values in configuration.
        
        Args:
            config: Configuration dictionary
            
        Returns:
            Resolved configuration dictionary
        """
        ...
    
    @abstractmethod
    def resolve_value(self, value: Any) -> Any:
        """Resolve a single configuration value.
        
        Args:
            value: Value to resolve
            
        Returns:
            Resolved value
        """
        ...


class ConfigResolver:
    """Resolves environment variables and paths in configurations."""
    
    def __init__(self, base_path: Optional[Path] = None):
        """Initialize resolver.
        
        Args:
            base_path: Base path for relative path resolution
        """
        self.base_path = base_path or Path.cwd()
        
        # Patterns for different types of variables
        self.env_var_pattern = re.compile(r'\$\{([^}]+)\}|\$([A-Z_][A-Z0-9_]*)')
        self.home_pattern = re.compile(r'^~')
        self.relative_path_pattern = re.compile(r'^\./|^\.\.')
    
    def resolve(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Resolve all dynamic values in configuration.
        
        Args:
            config: Configuration dictionary
            
        Returns:
            Resolved configuration dictionary
        """
        return self._resolve_recursive(config)
    
    def resolve_value(self, value: Any) -> Any:
        """Resolve a single configuration value.
        
        Args:
            value: Value to resolve
            
        Returns:
            Resolved value
        """
        if isinstance(value, str):
            return self._resolve_string(value)
        elif isinstance(value, dict):
            return self._resolve_recursive(value)
        elif isinstance(value, list):
            return [self.resolve_value(v) for v in value]
        else:
            return value
    
    def _resolve_recursive(self, obj: Any) -> Any:
        """Recursively resolve values in a nested structure.
        
        Args:
            obj: Object to resolve (dict, list, or value)
            
        Returns:
            Resolved object
        """
        if isinstance(obj, dict):
            return {key: self._resolve_recursive(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._resolve_recursive(item) for item in obj]
        elif isinstance(obj, str):
            return self._resolve_string(obj)
        else:
            return obj
    
    def _resolve_string(self, value: str) -> Union[str, Path]:
        """Resolve a string value.
        
        Args:
            value: String to resolve
            
        Returns:
            Resolved string or Path object
        """
        # First, expand environment variables
        resolved = self._expand_env_vars(value)
        
        # Then, resolve paths
        resolved = self._resolve_path(resolved)
        
        return resolved
    
    def _expand_env_vars(self, value: str) -> str:
        """Expand environment variables in a string.
        
        Supports both ${VAR} and $VAR formats.
        
        Args:
            value: String containing environment variables
            
        Returns:
            String with environment variables expanded
        """
        def replacer(match):
            # Get variable name from either group
            var_name = match.group(1) or match.group(2)
            
            # Handle special cases
            if ":-" in var_name:
                # Default value syntax: ${VAR:-default}
                var_part, default_part = var_name.split(":-", 1)
                return os.environ.get(var_part, default_part)
            else:
                # Regular variable
                env_value = os.environ.get(var_name)
                if env_value is None:
                    logger.warning(f"Environment variable '{var_name}' not found")
                    return match.group(0)  # Return original if not found
                return env_value
        
        return self.env_var_pattern.sub(replacer, value)
    
    def _resolve_path(self, value: str) -> Union[str, Path]:
        """Resolve path-like strings.
        
        Args:
            value: String that might be a path
            
        Returns:
            Resolved path or original string
        """
        # Check if this looks like a path
        if not self._is_path_like(value):
            return value
        
        # Expand home directory
        if self.home_pattern.match(value):
            value = os.path.expanduser(value)
        
        # Convert to Path object
        path = Path(value)
        
        # Resolve relative paths
        if not path.is_absolute():
            if self.relative_path_pattern.match(value):
                # Explicit relative path (./something or ../something)
                path = self.base_path / path
            else:
                # Implicit relative path - keep as is for now
                # This allows for paths relative to working directory
                pass
        
        # Return as Path object for path-like values
        return path
    
    def _is_path_like(self, value: str) -> bool:
        """Check if a string looks like a path.
        
        Args:
            value: String to check
            
        Returns:
            True if the string appears to be a path
        """
        # Skip certain values that should not be treated as paths
        skip_patterns = [
            ":/",  # URLs (http://, file://, etc.)
            ".",   # Version numbers (1.0, 2.0)
        ]
        
        # Check skip patterns
        for pattern in skip_patterns:
            if pattern in value and not value.startswith("./") and not value.startswith("../"):
                return False
        
        # Skip if it looks like a model name or package name
        if "/" in value and not value.startswith("/") and not value.startswith("./"):
            # Could be a HuggingFace model like "answerdotai/ModernBERT-base"
            parts = value.split("/")
            if len(parts) == 2 and not any(p.startswith(".") for p in parts):
                return False
        
        # Common path indicators
        path_indicators = [
            "/",  # Unix path separator (but not in middle of string)
            "\\",  # Windows path separator  
            "~",  # Home directory
        ]
        
        # Check for explicit path indicators
        if value.startswith("/") or value.startswith("~") or value.startswith("./") or value.startswith("../"):
            return True
        
        # Check for file extensions with path separators
        if "/" in value or "\\" in value:
            if "." in value.split("/")[-1] and value.split(".")[-1] in ["yaml", "yml", "json", "csv", "txt", "log"]:
                return True
        
        # Check for known path-related keywords in config keys
        # (but not in values like "file://...")
        path_keywords = ["_path", "_dir", "_file", "_folder"]
        if any(keyword in value.lower() for keyword in path_keywords):
            return True
        
        return False


class EnvironmentConfigResolver(ConfigResolver):
    """Resolver with enhanced environment variable support."""
    
    def __init__(self, base_path: Optional[Path] = None, env_prefix: str = "K_BERT_"):
        """Initialize environment resolver.
        
        Args:
            base_path: Base path for relative path resolution
            env_prefix: Prefix for environment variables
        """
        super().__init__(base_path)
        self.env_prefix = env_prefix
    
    def get_env_overrides(self) -> Dict[str, Any]:
        """Get configuration overrides from environment variables.
        
        Returns:
            Dictionary of configuration overrides
        """
        overrides = {}
        
        for key, value in os.environ.items():
            if key.startswith(self.env_prefix):
                # Remove prefix and convert to lowercase
                config_key = key[len(self.env_prefix):].lower()
                
                # Convert underscores to dots for nested keys
                config_key = config_key.replace("__", ".")
                
                # Parse value
                parsed_value = self._parse_env_value(value)
                
                # Set in overrides dict
                self._set_nested_value(overrides, config_key, parsed_value)
        
        return overrides
    
    def _parse_env_value(self, value: str) -> Any:
        """Parse environment variable value to appropriate type.
        
        Args:
            value: String value from environment
            
        Returns:
            Parsed value
        """
        # Try to parse as boolean
        if value.lower() in ["true", "false"]:
            return value.lower() == "true"
        
        # Try to parse as int
        try:
            return int(value)
        except ValueError:
            pass
        
        # Try to parse as float
        try:
            return float(value)
        except ValueError:
            pass
        
        # Return as string
        return value
    
    def _set_nested_value(self, config: Dict[str, Any], key: str, value: Any) -> None:
        """Set a value in nested configuration using dot notation.
        
        Args:
            config: Configuration dictionary to modify
            key: Dot-separated key path
            value: Value to set
        """
        parts = key.split(".")
        current = config
        
        # Navigate to the parent of the final key
        for part in parts[:-1]:
            if part not in current:
                current[part] = {}
            current = current[part]
        
        # Set the final value
        current[parts[-1]] = value


class TemplateConfigResolver(ConfigResolver):
    """Resolver with template variable support."""
    
    def __init__(self, base_path: Optional[Path] = None):
        """Initialize template resolver."""
        super().__init__(base_path)
        
        # Template pattern: {{variable}}
        self.template_pattern = re.compile(r'\{\{([^}]+)\}\}')
        
        # Predefined template variables
        self.template_vars = {
            "project_root": str(self.base_path),
            "home": str(Path.home()),
            "cwd": str(Path.cwd()),
            "date": self._get_date_string(),
        }
    
    def add_template_var(self, name: str, value: str) -> None:
        """Add a custom template variable.
        
        Args:
            name: Variable name
            value: Variable value
        """
        self.template_vars[name] = value
    
    def _resolve_string(self, value: str) -> Union[str, Path]:
        """Resolve a string value with template support.
        
        Args:
            value: String to resolve
            
        Returns:
            Resolved string or Path object
        """
        # First resolve templates
        value = self._expand_templates(value)
        
        # Then do normal resolution
        return super()._resolve_string(value)
    
    def _expand_templates(self, value: str) -> str:
        """Expand template variables in a string.
        
        Args:
            value: String containing template variables
            
        Returns:
            String with templates expanded
        """
        def replacer(match):
            var_name = match.group(1).strip()
            
            if var_name in self.template_vars:
                return self.template_vars[var_name]
            else:
                logger.warning(f"Unknown template variable: {var_name}")
                return match.group(0)
        
        return self.template_pattern.sub(replacer, value)
    
    def _get_date_string(self) -> str:
        """Get current date string for templates.
        
        Returns:
            Date string in YYYY-MM-DD format
        """
        from datetime import datetime
        return datetime.now().strftime("%Y-%m-%d")