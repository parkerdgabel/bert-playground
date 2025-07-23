"""Configuration loader for CLI.

This module provides utilities for loading and merging configuration from various sources.
"""

from pathlib import Path
from typing import Optional, Dict, Any, List
import yaml
from loguru import logger


class ConfigurationLoader:
    """Loads and merges configuration from multiple sources."""
    
    @staticmethod
    def find_project_config() -> Optional[Path]:
        """Find project configuration file in current directory.
        
        Returns:
            Path to config file if found, None otherwise
        """
        config_names = ["k-bert.yaml", "k-bert.yml", ".k-bert.yaml"]
        cwd = Path.cwd()
        
        for name in config_names:
            config_path = cwd / name
            if config_path.exists():
                logger.debug(f"Found project config: {config_path}")
                return config_path
                
        return None
        
    @staticmethod
    def find_user_config() -> Optional[Path]:
        """Find user configuration file.
        
        Returns:
            Path to user config file if found, None otherwise
        """
        user_config = Path.home() / ".k-bert" / "config.yaml"
        if user_config.exists():
            logger.debug(f"Found user config: {user_config}")
            return user_config
        return None
        
    @staticmethod
    def load_yaml_config(path: Path) -> Dict[str, Any]:
        """Load configuration from YAML file.
        
        Args:
            path: Path to YAML file
            
        Returns:
            Configuration dictionary
            
        Raises:
            ValueError: If file cannot be loaded
        """
        try:
            with open(path, 'r') as f:
                config = yaml.safe_load(f) or {}
                logger.debug(f"Loaded config from {path}")
                return config
        except Exception as e:
            raise ValueError(f"Failed to load config from {path}: {e}")
            
    @staticmethod
    def merge_configs(configs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Merge multiple configuration dictionaries.
        
        Later configs override earlier ones.
        
        Args:
            configs: List of config dictionaries
            
        Returns:
            Merged configuration
        """
        result = {}
        
        for config in configs:
            result = ConfigurationLoader._deep_merge(result, config)
            
        return result
        
    @staticmethod
    def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        """Deep merge two dictionaries.
        
        Args:
            base: Base dictionary
            override: Override dictionary
            
        Returns:
            Merged dictionary
        """
        result = base.copy()
        
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = ConfigurationLoader._deep_merge(result[key], value)
            else:
                result[key] = value
                
        return result
        
    @staticmethod
    def apply_cli_overrides(
        config: Dict[str, Any],
        train_data: Optional[Path] = None,
        val_data: Optional[Path] = None,
        test_data: Optional[Path] = None,
        epochs: Optional[int] = None,
        batch_size: Optional[int] = None,
        learning_rate: Optional[float] = None,
        output_dir: Optional[Path] = None,
        model_path: Optional[Path] = None,
        checkpoint: Optional[Path] = None,
    ) -> Dict[str, Any]:
        """Apply CLI argument overrides to configuration.
        
        Args:
            config: Base configuration
            train_data: Training data path override
            val_data: Validation data path override
            test_data: Test data path override
            epochs: Number of epochs override
            batch_size: Batch size override
            learning_rate: Learning rate override
            output_dir: Output directory override
            model_path: Model path override
            checkpoint: Checkpoint path override
            
        Returns:
            Updated configuration
        """
        # Make a copy to avoid modifying the original
        config = config.copy()
        
        # Ensure sections exist
        if "data" not in config:
            config["data"] = {}
        if "training" not in config:
            config["training"] = {}
        if "models" not in config:
            config["models"] = {}
            
        # Apply overrides
        if train_data:
            config["data"]["train_path"] = str(train_data)
        if val_data:
            config["data"]["val_path"] = str(val_data)
        if test_data:
            config["data"]["test_path"] = str(test_data)
        if epochs is not None:
            config["training"]["epochs"] = epochs
        if batch_size is not None:
            config["data"]["batch_size"] = batch_size
        if learning_rate is not None:
            config["training"]["learning_rate"] = learning_rate
        if output_dir:
            config["training"]["output_dir"] = str(output_dir)
        if model_path:
            config["models"]["model_path"] = str(model_path)
        if checkpoint:
            config["training"]["resume_from_checkpoint"] = str(checkpoint)
            
        return config
        
    @staticmethod
    def validate_config(config: Dict[str, Any], command: str) -> List[str]:
        """Validate configuration for a specific command.
        
        Args:
            config: Configuration to validate
            command: Command name (train, evaluate, predict)
            
        Returns:
            List of validation errors (empty if valid)
        """
        errors = []
        
        if command == "train":
            # Training requires data paths
            if not config.get("data", {}).get("train_path"):
                errors.append("Training data path not specified (data.train_path)")
                
        elif command == "evaluate":
            # Evaluation requires model and data
            if not config.get("models", {}).get("model_path") and \
               not config.get("training", {}).get("resume_from_checkpoint"):
                errors.append("Model path or checkpoint not specified")
            if not config.get("data", {}).get("val_path") and \
               not config.get("data", {}).get("test_path"):
                errors.append("Evaluation data path not specified")
                
        elif command == "predict":
            # Prediction requires model and data
            if not config.get("models", {}).get("model_path") and \
               not config.get("training", {}).get("resume_from_checkpoint"):
                errors.append("Model path or checkpoint not specified")
            if not config.get("data", {}).get("test_path"):
                errors.append("Test data path not specified")
                
        return errors