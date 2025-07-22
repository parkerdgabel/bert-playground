"""Configuration merger component for k-bert CLI.

This module handles merging configurations from multiple sources following
a hierarchical priority system.
"""

from typing import Any, Dict, List, Optional, Protocol, Union
from abc import abstractmethod
from copy import deepcopy

from loguru import logger


class ConfigMergerProtocol(Protocol):
    """Protocol for configuration mergers."""
    
    @abstractmethod
    def merge(self, *configs: Dict[str, Any]) -> Dict[str, Any]:
        """Merge multiple configurations into one.
        
        Args:
            *configs: Configuration dictionaries to merge (in priority order)
            
        Returns:
            Merged configuration dictionary
        """
        ...
    
    @abstractmethod
    def merge_with_priority(
        self,
        base: Dict[str, Any],
        overrides: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Merge configurations with explicit priority order.
        
        Args:
            base: Base configuration
            overrides: List of override configs (later = higher priority)
            
        Returns:
            Merged configuration
        """
        ...


class ConfigMerger:
    """Handles merging configurations from multiple sources."""
    
    def __init__(self, deep_merge: bool = True):
        """Initialize merger.
        
        Args:
            deep_merge: Whether to perform deep merging of nested structures
        """
        self.deep_merge = deep_merge
    
    def merge(self, *configs: Dict[str, Any]) -> Dict[str, Any]:
        """Merge multiple configurations into one.
        
        Later configurations override earlier ones.
        
        Args:
            *configs: Configuration dictionaries to merge
            
        Returns:
            Merged configuration dictionary
        """
        if not configs:
            return {}
        
        # Start with a deep copy of the first config
        result = deepcopy(configs[0]) if configs[0] else {}
        
        # Merge each subsequent config
        for config in configs[1:]:
            if config:
                result = self._merge_dicts(result, config)
        
        return result
    
    def merge_with_priority(
        self,
        base: Dict[str, Any],
        overrides: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Merge configurations with explicit priority order.
        
        Args:
            base: Base configuration
            overrides: Override configs in priority order
            
        Returns:
            Merged configuration
        """
        configs = [base] + overrides
        return self.merge(*configs)
    
    def _merge_dicts(self, base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        """Merge two dictionaries.
        
        Args:
            base: Base dictionary
            override: Override dictionary
            
        Returns:
            Merged dictionary
        """
        result = deepcopy(base)
        
        for key, value in override.items():
            if key in result and self.deep_merge:
                # Handle nested merging
                if isinstance(result[key], dict) and isinstance(value, dict):
                    result[key] = self._merge_dicts(result[key], value)
                elif isinstance(result[key], list) and isinstance(value, list):
                    # For lists, we replace rather than extend by default
                    result[key] = deepcopy(value)
                else:
                    # Direct replacement for non-dict/list values
                    result[key] = deepcopy(value)
            else:
                # Key doesn't exist or not deep merging
                result[key] = deepcopy(value)
        
        return result


class HierarchicalConfigMerger(ConfigMerger):
    """Merger that respects configuration hierarchy and special rules."""
    
    def __init__(self):
        """Initialize hierarchical merger."""
        super().__init__(deep_merge=True)
        
        # Define merge strategies for specific keys
        self.merge_strategies = {
            # Lists that should be extended rather than replaced
            "plugins": "extend",
            "augmenters": "extend",
            "callbacks": "extend",
            
            # Values that should not be overridden if already set
            "kaggle.username": "keep_first",
            "kaggle.key": "keep_first",
            
            # Special merge logic
            "environment": "merge_env",
        }
    
    def _merge_dicts(self, base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        """Merge dictionaries with hierarchical rules.
        
        Args:
            base: Base dictionary
            override: Override dictionary
            
        Returns:
            Merged dictionary
        """
        result = deepcopy(base)
        
        for key, value in override.items():
            # Check for special merge strategies
            strategy = self._get_merge_strategy(key)
            
            if strategy == "keep_first" and key in result:
                # Don't override if already set
                logger.debug(f"Keeping existing value for {key}")
                continue
            elif strategy == "extend" and key in result:
                # Extend lists instead of replacing
                if isinstance(result[key], list) and isinstance(value, list):
                    result[key] = result[key] + [v for v in value if v not in result[key]]
                    continue
            elif strategy == "merge_env" and key in result:
                # Special environment variable merging
                if isinstance(result[key], dict) and isinstance(value, dict):
                    result[key] = self._merge_environment_vars(result[key], value)
                    continue
            
            # Default merging behavior
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._merge_dicts(result[key], value)
            else:
                result[key] = deepcopy(value)
        
        return result
    
    def _get_merge_strategy(self, key: str) -> Optional[str]:
        """Get merge strategy for a key.
        
        Args:
            key: Configuration key
            
        Returns:
            Merge strategy name or None
        """
        # Direct match
        if key in self.merge_strategies:
            return self.merge_strategies[key]
        
        # Check for nested keys (e.g., "kaggle.username")
        for strategy_key, strategy in self.merge_strategies.items():
            if "." in strategy_key and key == strategy_key.split(".")[-1]:
                # This is a simplified check - in production you'd want full path matching
                return strategy
        
        return None
    
    def _merge_environment_vars(
        self,
        base_env: Dict[str, str],
        override_env: Dict[str, str]
    ) -> Dict[str, str]:
        """Merge environment variables with special handling.
        
        Args:
            base_env: Base environment variables
            override_env: Override environment variables
            
        Returns:
            Merged environment variables
        """
        result = base_env.copy()
        
        for key, value in override_env.items():
            if key in result and key.endswith("_PATH"):
                # For PATH-like variables, prepend rather than replace
                result[key] = f"{value}:{result[key]}"
            else:
                result[key] = value
        
        return result


class CliConfigMerger(HierarchicalConfigMerger):
    """Specialized merger for CLI configuration with priority rules.
    
    Priority order (highest to lowest):
    1. CLI arguments
    2. Environment variables
    3. Project configuration
    4. User configuration
    5. System defaults
    """
    
    def merge_cli_config(
        self,
        defaults: Dict[str, Any],
        user_config: Optional[Dict[str, Any]] = None,
        project_config: Optional[Dict[str, Any]] = None,
        env_vars: Optional[Dict[str, Any]] = None,
        cli_args: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Merge CLI configuration following priority rules.
        
        Args:
            defaults: System default configuration
            user_config: User configuration (~/.k-bert/config.yaml)
            project_config: Project configuration (k-bert.yaml)
            env_vars: Environment variable overrides
            cli_args: CLI argument overrides
            
        Returns:
            Merged configuration
        """
        # Build configuration in priority order
        configs = [defaults]
        
        if user_config:
            logger.debug("Merging user configuration")
            configs.append(user_config)
        
        if project_config:
            logger.debug("Merging project configuration")
            configs.append(project_config)
        
        if env_vars:
            logger.debug("Merging environment variables")
            configs.append(env_vars)
        
        if cli_args:
            logger.debug("Merging CLI arguments")
            # Filter out None values from CLI args
            filtered_cli_args = {k: v for k, v in cli_args.items() if v is not None}
            if filtered_cli_args:
                configs.append(filtered_cli_args)
        
        # Merge all configurations
        result = self.merge(*configs)
        
        logger.debug(f"Configuration merged from {len(configs)} sources")
        return result
    
    def extract_cli_overrides(self, cli_args: Dict[str, Any]) -> Dict[str, Any]:
        """Extract configuration overrides from CLI arguments.
        
        Converts flat CLI arguments to nested configuration structure.
        
        Args:
            cli_args: Flat dictionary of CLI arguments
            
        Returns:
            Nested configuration dictionary
        """
        config = {}
        
        for key, value in cli_args.items():
            if value is None:
                continue
            
            # Convert underscore to dot notation
            # e.g., "model_name" -> "model.name"
            if "_" in key:
                parts = key.split("_", 1)
                if parts[0] in ["model", "training", "data", "mlflow", "logging"]:
                    nested_key = f"{parts[0]}.{parts[1]}"
                    self._set_nested_value(config, nested_key, value)
                    continue
            
            # Handle known direct mappings
            direct_mappings = {
                "batch_size": "training.batch_size",
                "epochs": "training.epochs",
                "learning_rate": "training.learning_rate",
                "model": "models.default_model",
                "output_dir": "training.output_dir",
            }
            
            if key in direct_mappings:
                self._set_nested_value(config, direct_mappings[key], value)
            else:
                # Keep as-is if no mapping
                config[key] = value
        
        return config
    
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