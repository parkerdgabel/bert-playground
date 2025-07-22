"""Configuration manager for k-bert CLI.

This module provides the main configuration management functionality,
including loading from files, environment variables, and merging configurations.
"""

import os
import yaml
import json
from pathlib import Path
from typing import Any, Dict, Optional, Union, List
from functools import lru_cache

from loguru import logger

from .schemas import KBertConfig, ProjectConfig
from .defaults import get_default_config, get_competition_defaults
from .validators import validate_config, ConfigValidationError


class ConfigManager:
    """Manages k-bert configuration loading and merging."""
    
    # Configuration file search paths
    CONFIG_SEARCH_PATHS = [
        Path("k-bert.yaml"),
        Path("k-bert.yml"),
        Path("k-bert.json"),
        Path("pyproject.toml"),  # In [tool.k-bert] section
        Path(".k-bert/config.yaml"),
        Path(".k-bert/config.yml"),
        Path(".k-bert/config.json"),
    ]
    
    # User configuration path
    USER_CONFIG_PATH = Path("~/.k-bert/config.yaml").expanduser()
    
    def __init__(self):
        """Initialize configuration manager."""
        self._cache: Dict[str, Any] = {}
        self._user_config: Optional[KBertConfig] = None
        self._project_config: Optional[ProjectConfig] = None
        self._merged_config: Optional[KBertConfig] = None
    
    @property
    @lru_cache(maxsize=1)
    def user_config_path(self) -> Path:
        """Get user configuration path."""
        return self.USER_CONFIG_PATH
    
    @property
    @lru_cache(maxsize=1)
    def project_config_path(self) -> Optional[Path]:
        """Find project configuration path."""
        for path in self.CONFIG_SEARCH_PATHS:
            if path.exists():
                return path
        return None
    
    def load_user_config(self, force_reload: bool = False) -> KBertConfig:
        """Load user configuration.
        
        Args:
            force_reload: Force reload even if cached
            
        Returns:
            User configuration
        """
        if not force_reload and self._user_config is not None:
            return self._user_config
        
        if not self.user_config_path.exists():
            logger.debug("No user configuration found, using defaults")
            self._user_config = get_default_config()
            return self._user_config
        
        try:
            config_dict = self._load_file(self.user_config_path)
            self._user_config = KBertConfig.from_dict(config_dict)
            logger.debug(f"Loaded user configuration from {self.user_config_path}")
        except Exception as e:
            logger.warning(f"Failed to load user configuration: {e}")
            self._user_config = get_default_config()
        
        return self._user_config
    
    def load_project_config(self, path: Optional[Path] = None) -> Optional[ProjectConfig]:
        """Load project configuration.
        
        Args:
            path: Specific path to load from (optional)
            
        Returns:
            Project configuration if found
        """
        if path is None:
            path = self.project_config_path
        
        if path is None or not path.exists():
            return None
        
        try:
            config_dict = self._load_file(path)
            
            # Handle pyproject.toml special case
            if path.name == "pyproject.toml":
                if "tool" in config_dict and "k-bert" in config_dict["tool"]:
                    config_dict = config_dict["tool"]["k-bert"]
                else:
                    return None
            
            # Add name if not present
            if "name" not in config_dict:
                config_dict["name"] = path.parent.name
            
            self._project_config = ProjectConfig(**config_dict)
            logger.debug(f"Loaded project configuration from {path}")
        except Exception as e:
            logger.warning(f"Failed to load project configuration: {e}")
            return None
        
        return self._project_config
    
    def get_merged_config(
        self,
        cli_overrides: Optional[Dict[str, Any]] = None,
        project_path: Optional[Path] = None,
        validate: bool = True,
    ) -> KBertConfig:
        """Get merged configuration from all sources.
        
        Configuration priority (highest to lowest):
        1. CLI arguments
        2. Environment variables
        3. Project configuration
        4. User configuration
        5. System defaults
        
        Args:
            cli_overrides: CLI argument overrides
            project_path: Specific project config path
            validate: Whether to validate the final config
            
        Returns:
            Merged configuration
        """
        # Start with defaults
        config = get_default_config()
        
        # Merge user configuration
        user_config = self.load_user_config()
        config = config.merge(user_config)
        
        # Merge project configuration if available
        project_config = self.load_project_config(project_path)
        if project_config:
            # Convert project config sections to dict and merge
            project_dict = {}
            for field in ["kaggle", "models", "training", "mlflow", "data", "logging"]:
                value = getattr(project_config, field, None)
                if value is not None:
                    project_dict[field] = value.model_dump(exclude_none=True)
            
            config = config.merge(project_dict)
            
            # Apply competition-specific defaults if specified
            if project_config.competition:
                comp_defaults = get_competition_defaults(project_config.competition)
                config = config.merge(comp_defaults)
        
        # Apply environment variables (handled by Pydantic)
        config = self._apply_env_vars(config)
        
        # Apply CLI overrides
        if cli_overrides:
            config = config.merge(cli_overrides)
        
        # Validate if requested
        if validate:
            try:
                validate_config(config)
            except ConfigValidationError as e:
                logger.error(f"Configuration validation failed: {e}")
                raise
        
        self._merged_config = config
        return config
    
    def save_user_config(self, config: KBertConfig) -> None:
        """Save user configuration.
        
        Args:
            config: Configuration to save
        """
        # Ensure directory exists
        self.user_config_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert to dict and remove None values
        config_dict = config.to_dict()
        
        # Convert Path objects to strings for YAML serialization
        config_dict = self._convert_paths_to_strings(config_dict)
        
        # Save as YAML
        with open(self.user_config_path, "w") as f:
            yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)
        
        logger.info(f"Saved user configuration to {self.user_config_path}")
        
        # Clear cache
        self._user_config = None
    
    def save_project_config(
        self, 
        config: Union[ProjectConfig, Dict[str, Any]], 
        path: Optional[Path] = None
    ) -> None:
        """Save project configuration.
        
        Args:
            config: Configuration to save
            path: Path to save to (defaults to k-bert.yaml)
        """
        if path is None:
            path = Path("k-bert.yaml")
        
        # Convert to dict if needed
        if isinstance(config, ProjectConfig):
            config_dict = config.model_dump(exclude_none=True)
        else:
            config_dict = config
        
        # Determine format from extension
        if path.suffix in [".yaml", ".yml"]:
            with open(path, "w") as f:
                yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)
        elif path.suffix == ".json":
            with open(path, "w") as f:
                json.dump(config_dict, f, indent=2)
        elif path.name == "pyproject.toml":
            # For pyproject.toml, we need to be careful not to overwrite other content
            import toml
            
            if path.exists():
                with open(path, "r") as f:
                    pyproject = toml.load(f)
            else:
                pyproject = {}
            
            if "tool" not in pyproject:
                pyproject["tool"] = {}
            
            pyproject["tool"]["k-bert"] = config_dict
            
            with open(path, "w") as f:
                toml.dump(pyproject, f)
        else:
            raise ValueError(f"Unsupported configuration format: {path.suffix}")
        
        logger.info(f"Saved project configuration to {path}")
        
        # Clear cache
        self._project_config = None
    
    def init_user_config(self, interactive: bool = True) -> KBertConfig:
        """Initialize user configuration.
        
        Args:
            interactive: Whether to prompt for values
            
        Returns:
            Initialized configuration
        """
        if self.user_config_path.exists():
            logger.warning(f"User configuration already exists at {self.user_config_path}")
            if interactive:
                from rich.prompt import Confirm
                if not Confirm.ask("Overwrite existing configuration?"):
                    return self.load_user_config()
        
        config = get_default_config()
        
        if interactive:
            from rich.prompt import Prompt, Confirm
            from rich.console import Console
            
            console = Console()
            console.print("\n[bold]Welcome to k-bert![/bold] Let's set up your configuration.\n")
            
            # Kaggle credentials
            username = Prompt.ask("Kaggle username", default="")
            if username:
                config.kaggle.username = username
                
                key = Prompt.ask("Kaggle API key (will be stored securely)", password=True, default="")
                if key:
                    config.kaggle.key = key
            
            # Default model
            model = Prompt.ask(
                "Default model",
                default=config.models.default_model,
                choices=[
                    "answerdotai/ModernBERT-base",
                    "answerdotai/ModernBERT-large",
                    "bert-base-uncased",
                ]
            )
            config.models.default_model = model
            
            # Output directory
            output_dir = Prompt.ask("Default output directory", default=str(config.training.output_dir))
            config.training.output_dir = Path(output_dir)
            
            # MLflow tracking
            enable_mlflow = Confirm.ask("Enable MLflow tracking?", default=True)
            config.mlflow.auto_log = enable_mlflow
            
            console.print(f"\n[green]Configuration saved to {self.user_config_path}[/green]")
        
        self.save_user_config(config)
        return config
    
    def get_value(self, key: str, default: Any = None) -> Any:
        """Get a configuration value by key.
        
        Args:
            key: Dot-separated key path (e.g., "kaggle.username")
            default: Default value if not found
            
        Returns:
            Configuration value
        """
        config = self.get_merged_config()
        
        # Navigate through nested keys
        value = config
        for part in key.split("."):
            if hasattr(value, part):
                value = getattr(value, part)
            elif isinstance(value, dict) and part in value:
                value = value[part]
            else:
                return default
        
        return value
    
    def set_value(self, key: str, value: Any, save: bool = True) -> None:
        """Set a configuration value.
        
        Args:
            key: Dot-separated key path (e.g., "kaggle.username")
            value: Value to set
            save: Whether to save immediately
        """
        config = self.load_user_config()
        
        # Navigate to the parent object
        parts = key.split(".")
        parent = config
        for part in parts[:-1]:
            if hasattr(parent, part):
                parent = getattr(parent, part)
            else:
                raise ValueError(f"Invalid configuration key: {key}")
        
        # Set the value
        final_key = parts[-1]
        if hasattr(parent, final_key):
            setattr(parent, final_key, value)
        else:
            raise ValueError(f"Invalid configuration key: {key}")
        
        if save:
            self.save_user_config(config)
    
    def _load_file(self, path: Path) -> Dict[str, Any]:
        """Load configuration from file.
        
        Args:
            path: Path to configuration file
            
        Returns:
            Configuration dictionary
        """
        if path.suffix in [".yaml", ".yml"]:
            with open(path, "r") as f:
                return yaml.safe_load(f) or {}
        elif path.suffix == ".json":
            with open(path, "r") as f:
                return json.load(f)
        elif path.name == "pyproject.toml":
            import toml
            with open(path, "r") as f:
                return toml.load(f)
        else:
            raise ValueError(f"Unsupported configuration format: {path.suffix}")
    
    def _apply_env_vars(self, config: KBertConfig) -> KBertConfig:
        """Apply environment variable overrides.
        
        Args:
            config: Base configuration
            
        Returns:
            Configuration with env vars applied
        """
        # Pydantic handles most env var loading, but we can add custom logic here
        # For example, expanding ${VAR} references in string values
        
        config_dict = config.to_dict()
        self._expand_env_vars(config_dict)
        
        return KBertConfig.from_dict(config_dict)
    
    def _expand_env_vars(self, obj: Any) -> Any:
        """Recursively expand environment variables in configuration.
        
        Args:
            obj: Configuration object (dict, list, or value)
            
        Returns:
            Object with env vars expanded
        """
        if isinstance(obj, dict):
            for key, value in obj.items():
                obj[key] = self._expand_env_vars(value)
        elif isinstance(obj, list):
            for i, value in enumerate(obj):
                obj[i] = self._expand_env_vars(value)
        elif isinstance(obj, str):
            # Expand ${VAR} and $VAR patterns
            import re
            
            def replacer(match):
                var_name = match.group(1)
                return os.environ.get(var_name, match.group(0))
            
            obj = re.sub(r'\$\{([^}]+)\}', replacer, obj)
            obj = re.sub(r'\$([A-Z_][A-Z0-9_]*)', replacer, obj)
        
        return obj
    
    def list_all_values(self) -> Dict[str, Any]:
        """Get all configuration values as a flat dictionary.
        
        Returns:
            Flat dictionary of all configuration values
        """
        config = self.get_merged_config()
        return self._flatten_config(config.to_dict())
    
    def list_settings(self) -> Dict[str, Any]:
        """List all settings (alias for list_all_values).
        
        Returns:
            Flat dictionary of all configuration values
        """
        return self.list_all_values()
    
    def validate_project_config(self, path: Optional[Path] = None) -> List[str]:
        """Validate project configuration.
        
        Args:
            path: Path to project config file (optional)
            
        Returns:
            List of validation errors (empty if valid)
        """
        try:
            config = self.load_project_config(path)
            if config is None:
                return ["No project configuration found"]
            
            # Basic validation - check required fields
            errors = []
            
            if not config.name:
                errors.append("Project name is required")
            
            if not config.models or not config.models.default_model:
                errors.append("Default model must be specified")
            
            if not config.data:
                errors.append("Data configuration is required")
            
            return errors
        except Exception as e:
            return [f"Failed to load configuration: {str(e)}"]
    
    def _flatten_config(self, obj: Any, prefix: str = "") -> Dict[str, Any]:
        """Flatten nested configuration to dot-notation keys.
        
        Args:
            obj: Configuration object
            prefix: Current key prefix
            
        Returns:
            Flat dictionary
        """
        result = {}
        
        if isinstance(obj, dict):
            for key, value in obj.items():
                new_prefix = f"{prefix}.{key}" if prefix else key
                if isinstance(value, (dict, list)):
                    result.update(self._flatten_config(value, new_prefix))
                else:
                    result[new_prefix] = value
        elif isinstance(obj, list):
            for i, value in enumerate(obj):
                new_prefix = f"{prefix}[{i}]"
                if isinstance(value, (dict, list)):
                    result.update(self._flatten_config(value, new_prefix))
                else:
                    result[new_prefix] = value
        else:
            result[prefix] = obj
        
        return result
    
    def _convert_paths_to_strings(self, obj: Any) -> Any:
        """Convert Path objects to strings for YAML serialization.
        
        Args:
            obj: Configuration object
            
        Returns:
            Object with paths converted to strings
        """
        if isinstance(obj, Path):
            return str(obj)
        elif isinstance(obj, dict):
            return {k: self._convert_paths_to_strings(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_paths_to_strings(v) for v in obj]
        else:
            return obj


# Global configuration manager instance
_config_manager = ConfigManager()


# Convenience functions
def get_config(**kwargs) -> KBertConfig:
    """Get merged configuration."""
    return _config_manager.get_merged_config(**kwargs)


def get_value(key: str, default: Any = None) -> Any:
    """Get configuration value."""
    return _config_manager.get_value(key, default)


def set_value(key: str, value: Any, save: bool = True) -> None:
    """Set configuration value."""
    _config_manager.set_value(key, value, save)