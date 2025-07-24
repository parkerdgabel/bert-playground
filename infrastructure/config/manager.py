"""Configuration management system for the infrastructure layer.

This module provides centralized configuration management that supports:
- Hierarchical configuration loading (user, project, command-level)
- Environment variable overrides
- Configuration validation and schema checking
- Adapter-specific configuration with easy swapping
"""

import os
from pathlib import Path
from typing import Any, Dict, Optional, Type, Union

from .loader import ConfigurationLoader
from .validator import ConfigurationValidator, ConfigurationValidationError


class ConfigurationManager:
    """Central configuration manager for the infrastructure layer.
    
    Supports hierarchical configuration loading:
    1. User config: ~/.k-bert/config.yaml
    2. Project config: ./k-bert.yaml
    3. Command config: --config file.yaml
    4. Environment variables
    5. CLI arguments
    """
    
    def __init__(
        self,
        user_config_path: Optional[Path] = None,
        project_config_path: Optional[Path] = None,
        command_config_path: Optional[Path] = None,
    ):
        """Initialize configuration manager.
        
        Args:
            user_config_path: Path to user config file
            project_config_path: Path to project config file  
            command_config_path: Path to command-specific config file
        """
        self.loader = ConfigurationLoader()
        self.validator = ConfigurationValidator()
        
        # Default paths
        self.user_config_path = user_config_path or self._get_default_user_config_path()
        self.project_config_path = project_config_path or self._get_default_project_config_path()
        self.command_config_path = command_config_path
        
        self._config_cache: Optional[Dict[str, Any]] = None
        self._adapter_configs: Dict[str, Dict[str, Any]] = {}
        
    def _get_default_user_config_path(self) -> Path:
        """Get default user config path."""
        config_dir = Path.home() / ".k-bert"
        config_dir.mkdir(exist_ok=True)
        return config_dir / "config.yaml"
        
    def _get_default_project_config_path(self) -> Path:
        """Get default project config path."""
        return Path.cwd() / "k-bert.yaml"
        
    def load_configuration(self) -> Dict[str, Any]:
        """Load hierarchical configuration.
        
        Returns:
            Merged configuration dictionary
        """
        if self._config_cache is not None:
            return self._config_cache
            
        config = {}
        
        # 1. Load user config (lowest priority)
        if self.user_config_path.exists():
            user_config = self.loader.load_yaml(self.user_config_path)
            config = self.loader.merge_configs(config, user_config)
            
        # 2. Load project config
        if self.project_config_path.exists():
            project_config = self.loader.load_yaml(self.project_config_path)
            config = self.loader.merge_configs(config, project_config)
            
        # 3. Load command-specific config
        if self.command_config_path and self.command_config_path.exists():
            command_config = self.loader.load_yaml(self.command_config_path)
            config = self.loader.merge_configs(config, command_config)
            
        # 4. Apply environment variable overrides
        config = self._apply_env_overrides(config)
        
        # 5. Validate final configuration
        self.validator.validate(config)
        
        # Cache the result
        self._config_cache = config
        return config
        
    def _apply_env_overrides(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Apply environment variable overrides.
        
        Environment variables with K_BERT_ prefix override config values.
        Examples:
        - K_BERT_MODELS_BATCH_SIZE=32 -> config.models.batch_size = 32
        - K_BERT_TRAINING_LEARNING_RATE=0.001 -> config.training.learning_rate = 0.001
        
        Args:
            config: Base configuration
            
        Returns:
            Configuration with environment overrides applied
        """
        import copy
        config = copy.deepcopy(config)
        
        for key, value in os.environ.items():
            if not key.startswith("K_BERT_"):
                continue
                
            # Convert K_BERT_FOO_BAR to ["foo", "bar"]
            config_path = key[7:].lower().split("_")
            
            # Navigate to the config section
            current = config
            for path_part in config_path[:-1]:
                if path_part not in current:
                    current[path_part] = {}
                current = current[path_part]
                
            # Set the final value (with type conversion)
            final_key = config_path[-1]
            current[final_key] = self._convert_env_value(value)
            
        return config
        
    def _convert_env_value(self, value: str) -> Union[str, int, float, bool]:
        """Convert environment variable string to appropriate type.
        
        Args:
            value: Environment variable string value
            
        Returns:
            Converted value
        """
        # Try boolean
        if value.lower() in ("true", "false"):
            return value.lower() == "true"
            
        # Try integer
        try:
            return int(value)
        except ValueError:
            pass
            
        # Try float
        try:
            return float(value)
        except ValueError:
            pass
            
        # Return as string
        return value
        
    def get_config(self, key: str, default: Any = None) -> Any:
        """Get configuration value by dot-notation key.
        
        Args:
            key: Configuration key (e.g., "models.batch_size")
            default: Default value if key not found
            
        Returns:
            Configuration value
        """
        config = self.load_configuration()
        
        # Navigate the config using dot notation
        current = config
        for part in key.split("."):
            if isinstance(current, dict) and part in current:
                current = current[part]
            else:
                return default
                
        return current
        
    def get_adapter_config(self, adapter_type: str) -> Dict[str, Any]:
        """Get configuration for a specific adapter type.
        
        This allows easy swapping of adapters by changing configuration.
        
        Args:
            adapter_type: Type of adapter (e.g., "monitoring", "storage", "compute")
            
        Returns:
            Adapter-specific configuration
        """
        if adapter_type in self._adapter_configs:
            return self._adapter_configs[adapter_type]
            
        config = self.load_configuration()
        adapters_config = config.get("adapters", {})
        adapter_config = adapters_config.get(adapter_type, {})
        
        self._adapter_configs[adapter_type] = adapter_config
        return adapter_config
        
    def get_adapter_implementation(self, adapter_type: str, default: str) -> str:
        """Get the implementation class name for an adapter.
        
        This enables easy adapter switching via configuration.
        
        Args:
            adapter_type: Type of adapter 
            default: Default implementation if not configured
            
        Returns:
            Implementation class name
        """
        adapter_config = self.get_adapter_config(adapter_type)
        return adapter_config.get("implementation", default)
        
    def get(self, key: str, default: Any = None) -> Any:
        """Get a configuration value by key.
        
        Args:
            key: Configuration key (dot notation, e.g., "training.learning_rate")
            default: Default value if key not found
            
        Returns:
            Configuration value or default
        """
        config = self.load_configuration()
        
        # Navigate through nested config
        current = config
        parts = key.split(".")
        
        for part in parts:
            if isinstance(current, dict) and part in current:
                current = current[part]
            else:
                return default
                
        return current
        
    def set_config(self, key: str, value: Any) -> None:
        """Set a configuration value at runtime.
        
        Args:
            key: Configuration key (dot notation)
            value: Value to set
        """
        config = self.load_configuration()
        
        # Navigate to parent
        current = config
        parts = key.split(".")
        for part in parts[:-1]:
            if part not in current:
                current[part] = {}
            current = current[part]
            
        # Set final value
        current[parts[-1]] = value
        
        # Invalidate cache
        self._config_cache = None
        
    def reload_configuration(self) -> Dict[str, Any]:
        """Reload configuration from files.
        
        Returns:
            Reloaded configuration
        """
        self._config_cache = None
        self._adapter_configs.clear()
        return self.load_configuration()
        
    def get_logging_config(self) -> Dict[str, Any]:
        """Get logging-specific configuration.
        
        Returns:
            Logging configuration
        """
        return self.get_config("logging", {
            "level": "INFO",
            "format": "{time} | {level} | {name} | {message}",
            "rotation": "10 MB",
            "retention": "7 days",
        })
        
    def get_mlflow_config(self) -> Dict[str, Any]:
        """Get MLflow-specific configuration.
        
        Returns:
            MLflow configuration
        """
        return self.get_config("mlflow", {
            "tracking_uri": "./output/mlruns",
            "experiment_name": "k-bert-training",
            "auto_log": True,
        })
        
    def get_model_config(self) -> Dict[str, Any]:
        """Get model-specific configuration.
        
        Returns:
            Model configuration
        """
        return self.get_config("models", {
            "type": "modernbert_with_head",
            "batch_size": 32,
            "max_length": 512,
        })
        
    def get_training_config(self) -> Dict[str, Any]:
        """Get training-specific configuration.
        
        Returns:
            Training configuration
        """
        return self.get_config("training", {
            "epochs": 10,
            "learning_rate": 2e-5,
            "weight_decay": 0.01,
            "warmup_steps": 100,
        })
        
    def validate_adapter_config(self, adapter_type: str, schema: Type) -> bool:
        """Validate adapter configuration against a schema.
        
        Args:
            adapter_type: Type of adapter
            schema: Pydantic schema class
            
        Returns:
            True if valid
        """
        adapter_config = self.get_adapter_config(adapter_type)
        return self.validator.validate_schema(adapter_config, schema)