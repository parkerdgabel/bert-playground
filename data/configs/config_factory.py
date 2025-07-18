"""
Configuration factory for creating and managing dataloader configs.
"""

from typing import Dict, Any, Optional, Union, List, Type
from pathlib import Path
import json
import yaml
from copy import deepcopy
from loguru import logger

from .base_config import (
    BaseConfig,
    DataLoaderConfig,
    DatasetConfig,
    ExperimentConfig,
    CompetitionConfig,
    OptimizationProfile,
)
from .transform_config import TransformPipeline, get_preset_pipeline
from .cache_config import CacheConfig, get_preset_cache_config


class ConfigFactory:
    """Factory for creating and managing configurations."""
    
    @staticmethod
    def create_dataloader_config(
        preset: Optional[str] = None,
        **kwargs
    ) -> DataLoaderConfig:
        """
        Create dataloader configuration.
        
        Args:
            preset: Optional preset name
            **kwargs: Configuration overrides
            
        Returns:
            DataLoader configuration
        """
        if preset:
            config = ConfigFactory.get_preset_config(preset)
        else:
            config = DataLoaderConfig()
        
        # Apply overrides
        for key, value in kwargs.items():
            if hasattr(config, key):
                setattr(config, key, value)
        
        return config
    
    @staticmethod
    def create_dataset_config(
        data_path: Optional[Union[str, Path]] = None,
        **kwargs
    ) -> DatasetConfig:
        """
        Create dataset configuration.
        
        Args:
            data_path: Path to data
            **kwargs: Configuration options
            
        Returns:
            Dataset configuration
        """
        config = DatasetConfig(data_path=data_path)
        
        # Apply options
        for key, value in kwargs.items():
            if hasattr(config, key):
                setattr(config, key, value)
        
        return config
    
    @staticmethod
    def create_experiment_config(
        name: str,
        dataset_config: Optional[Union[Dict, DatasetConfig]] = None,
        dataloader_config: Optional[Union[Dict, DataLoaderConfig]] = None,
        **kwargs
    ) -> ExperimentConfig:
        """
        Create experiment configuration.
        
        Args:
            name: Experiment name
            dataset_config: Dataset configuration
            dataloader_config: DataLoader configuration
            **kwargs: Additional options
            
        Returns:
            Experiment configuration
        """
        config = ExperimentConfig(name=name)
        
        # Set dataset config
        if dataset_config:
            if isinstance(dataset_config, dict):
                config.dataset = DatasetConfig.from_dict(dataset_config)
            else:
                config.dataset = dataset_config
        
        # Set dataloader config
        if dataloader_config:
            if isinstance(dataloader_config, dict):
                config.dataloader = DataLoaderConfig.from_dict(dataloader_config)
            else:
                config.dataloader = dataloader_config
        
        # Apply additional options
        for key, value in kwargs.items():
            if hasattr(config, key):
                setattr(config, key, value)
        
        return config
    
    @staticmethod
    def create_competition_config(
        competition: str,
        data_path: Optional[Union[str, Path]] = None,
        **kwargs
    ) -> CompetitionConfig:
        """
        Create competition-specific configuration.
        
        Args:
            competition: Competition name
            data_path: Path to competition data
            **kwargs: Additional options
            
        Returns:
            Competition configuration
        """
        config = CompetitionConfig(
            name=f"{competition}_experiment",
            competition_name=competition,
        )
        
        # Set up for competition
        config.setup_for_competition(competition)
        
        # Set data path
        if data_path:
            config.dataset.data_path = data_path
        
        # Apply overrides
        for key, value in kwargs.items():
            if hasattr(config, key):
                setattr(config, key, value)
            elif hasattr(config.dataset, key):
                setattr(config.dataset, key, value)
            elif hasattr(config.dataloader, key):
                setattr(config.dataloader, key, value)
        
        return config
    
    @staticmethod
    def get_preset_config(name: str) -> DataLoaderConfig:
        """
        Get preset dataloader configuration.
        
        Args:
            name: Preset name
            
        Returns:
            DataLoader configuration
        """
        presets = {
            "fast": DataLoaderConfig(
                batch_size=64,
                num_workers=8,
                prefetch_size=8,
                optimization_profile="speed",
            ),
            
            "balanced": DataLoaderConfig(
                batch_size=32,
                num_workers=4,
                prefetch_size=4,
                optimization_profile="balanced",
            ),
            
            "memory_efficient": DataLoaderConfig(
                batch_size=16,
                num_workers=2,
                prefetch_size=2,
                optimization_profile="memory",
            ),
            
            "debug": DataLoaderConfig(
                batch_size=8,
                num_workers=0,
                prefetch_size=1,
                shuffle=False,
                optimization_profile="debug",
            ),
            
            "competition": DataLoaderConfig(
                batch_size=32,
                num_workers=4,
                prefetch_size=4,
                augment=True,
                enable_cache=True,
                cache_tokenized=True,
            ),
        }
        
        if name not in presets:
            raise ValueError(
                f"Unknown preset: {name}. "
                f"Available: {list(presets.keys())}"
            )
        
        # Return a copy
        config = presets[name]
        return DataLoaderConfig.from_dict(config.to_dict())
    
    @staticmethod
    def from_file(path: Union[str, Path]) -> BaseConfig:
        """
        Load configuration from file.
        
        Args:
            path: Path to configuration file
            
        Returns:
            Configuration object
        """
        path = Path(path)
        
        if not path.exists():
            raise FileNotFoundError(f"Configuration file not found: {path}")
        
        # Load based on extension
        if path.suffix == ".json":
            with open(path, "r") as f:
                config_dict = json.load(f)
        elif path.suffix in [".yaml", ".yml"]:
            with open(path, "r") as f:
                config_dict = yaml.safe_load(f)
        else:
            raise ValueError(f"Unsupported config format: {path.suffix}")
        
        # Determine config type
        config_type = config_dict.get("_type", "dataloader")
        
        if config_type == "dataset":
            return DatasetConfig.from_dict(config_dict)
        elif config_type == "dataloader":
            return DataLoaderConfig.from_dict(config_dict)
        elif config_type == "experiment":
            return ExperimentConfig.from_dict(config_dict)
        elif config_type == "competition":
            return CompetitionConfig.from_dict(config_dict)
        else:
            # Default to dataloader
            return DataLoaderConfig.from_dict(config_dict)


def load_config(path: Union[str, Path]) -> BaseConfig:
    """
    Load configuration from file.
    
    Args:
        path: Path to configuration file
        
    Returns:
        Configuration object
    """
    return ConfigFactory.from_file(path)


def save_config(config: BaseConfig, path: Union[str, Path]) -> None:
    """
    Save configuration to file.
    
    Args:
        config: Configuration object
        path: Path to save to
    """
    path = Path(path)
    
    # Add type information
    config_dict = config.to_dict()
    config_dict["_type"] = config.__class__.__name__.lower().replace("config", "")
    
    # Save based on extension
    if path.suffix == ".json":
        with open(path, "w") as f:
            json.dump(config_dict, f, indent=2)
    elif path.suffix in [".yaml", ".yml"]:
        with open(path, "w") as f:
            yaml.dump(config_dict, f, default_flow_style=False)
    else:
        # Default to JSON
        path = path.with_suffix(".json")
        with open(path, "w") as f:
            json.dump(config_dict, f, indent=2)
    
    logger.info(f"Saved configuration to {path}")


def validate_config(config: BaseConfig) -> List[str]:
    """
    Validate configuration.
    
    Args:
        config: Configuration to validate
        
    Returns:
        List of validation errors
    """
    return config.validate()


def merge_configs(
    base: BaseConfig,
    override: Union[BaseConfig, Dict[str, Any]],
    deep: bool = True
) -> BaseConfig:
    """
    Merge configurations.
    
    Args:
        base: Base configuration
        override: Override configuration
        deep: Whether to do deep merge
        
    Returns:
        Merged configuration
    """
    # Convert to dicts
    base_dict = base.to_dict()
    
    if isinstance(override, BaseConfig):
        override_dict = override.to_dict()
    else:
        override_dict = override
    
    # Merge
    if deep:
        merged = deep_merge(base_dict, override_dict)
    else:
        merged = {**base_dict, **override_dict}
    
    # Create new config
    return base.__class__.from_dict(merged)


def deep_merge(base: Dict, override: Dict) -> Dict:
    """Deep merge two dictionaries."""
    result = deepcopy(base)
    
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = deepcopy(value)
    
    return result


# Configuration templates
def create_config_template(
    config_type: str = "dataloader",
    output_path: Optional[Union[str, Path]] = None
) -> Dict[str, Any]:
    """
    Create configuration template.
    
    Args:
        config_type: Type of configuration
        output_path: Optional path to save template
        
    Returns:
        Configuration template
    """
    templates = {
        "dataset": DatasetConfig(),
        "dataloader": DataLoaderConfig(),
        "experiment": ExperimentConfig(),
        "competition": CompetitionConfig(),
        "transform": TransformPipeline(),
        "cache": CacheConfig(),
    }
    
    if config_type not in templates:
        raise ValueError(
            f"Unknown config type: {config_type}. "
            f"Available: {list(templates.keys())}"
        )
    
    config = templates[config_type]
    template = config.to_dict()
    
    # Add helpful comments
    template["_description"] = f"Template for {config_type} configuration"
    template["_type"] = config_type
    
    # Save if requested
    if output_path:
        path = Path(output_path)
        if path.suffix == ".yaml" or path.suffix == ".yml":
            with open(path, "w") as f:
                yaml.dump(template, f, default_flow_style=False)
        else:
            with open(path, "w") as f:
                json.dump(template, f, indent=2)
        logger.info(f"Saved template to {path}")
    
    return template