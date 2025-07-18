"""
Preset configurations for common use cases.
"""

from typing import Dict, Any, Optional, List
from copy import deepcopy
from loguru import logger

from .base_config import DataLoaderConfig, DatasetConfig, ExperimentConfig
from .transform_config import TransformPipeline, TransformConfig
from .cache_config import CacheConfig


# Preset dataloader configurations
DATALOADER_PRESETS = {
    "default": {
        "batch_size": 32,
        "shuffle": True,
        "num_workers": 4,
        "prefetch_size": 4,
        "max_length": 256,
        "optimization_profile": "balanced",
    },
    
    "fast_training": {
        "batch_size": 64,
        "shuffle": True,
        "num_workers": 8,
        "prefetch_size": 8,
        "buffer_size": 2000,
        "max_length": 256,
        "optimization_profile": "speed",
        "enable_cache": True,
        "cache_tokenized": True,
        "persistent_workers": True,
    },
    
    "memory_constrained": {
        "batch_size": 16,
        "shuffle": True,
        "num_workers": 2,
        "prefetch_size": 2,
        "buffer_size": 500,
        "max_length": 128,
        "optimization_profile": "memory",
        "enable_cache": False,
        "persistent_workers": False,
    },
    
    "inference": {
        "batch_size": 32,
        "shuffle": False,
        "num_workers": 4,
        "prefetch_size": 4,
        "drop_last": False,
        "optimization_profile": "balanced",
        "enable_cache": True,
    },
    
    "debug": {
        "batch_size": 4,
        "shuffle": False,
        "num_workers": 0,
        "prefetch_size": 1,
        "buffer_size": 100,
        "max_length": 128,
        "optimization_profile": "debug",
        "enable_cache": False,
    },
}


# Preset dataset configurations
DATASET_PRESETS = {
    "default": {
        "data_format": "csv",
        "train_split": 0.8,
        "val_split": 0.1,
        "test_split": 0.1,
        "stratify": True,
        "shuffle": True,
        "filter_empty": True,
    },
    
    "kaggle_competition": {
        "data_format": "csv",
        "train_split": 0.9,
        "val_split": 0.1,
        "test_split": 0.0,
        "stratify": True,
        "shuffle": True,
        "filter_empty": True,
    },
    
    "cross_validation": {
        "data_format": "csv",
        "train_split": 1.0,
        "val_split": 0.0,
        "test_split": 0.0,
        "stratify": False,
        "shuffle": False,
        "filter_empty": True,
    },
    
    "large_dataset": {
        "data_format": "parquet",
        "train_split": 0.8,
        "val_split": 0.1,
        "test_split": 0.1,
        "stratify": False,
        "shuffle": True,
        "lazy_loading": True,
        "chunk_size": 10000,
    },
}


# Preset transform pipelines
TRANSFORM_PRESETS = {
    "basic_nlp": {
        "preprocessing": [
            {
                "name": "tokenize",
                "params": {
                    "max_length": 256,
                    "padding": "max_length",
                    "truncation": True,
                },
            },
            {
                "name": "to_mlx_array",
                "params": {
                    "fields": ["input_ids", "attention_mask", "label"],
                },
            },
        ],
    },
    
    "augmented_nlp": {
        "preprocessing": [
            {
                "name": "tokenize",
                "params": {
                    "max_length": 256,
                    "padding": "max_length",
                    "truncation": True,
                },
            },
        ],
        "augmentation": [
            {
                "name": "eda",
                "params": {
                    "alpha_sr": 0.1,
                    "alpha_ri": 0.1,
                    "alpha_rs": 0.1,
                    "alpha_rd": 0.1,
                },
            },
        ],
        "augmentation_prob": 0.3,
        "postprocessing": [
            {
                "name": "to_mlx_array",
                "params": {
                    "fields": ["input_ids", "attention_mask", "label"],
                },
            },
        ],
    },
    
    "tabular_to_text": {
        "text_converter": {
            "name": "template",
            "params": {
                "strategy": "template",
                "augment": True,
            },
        },
        "preprocessing": [
            {
                "name": "tokenize",
                "params": {
                    "max_length": 256,
                    "padding": "max_length",
                    "truncation": True,
                },
            },
            {
                "name": "to_mlx_array",
                "params": {
                    "fields": ["input_ids", "attention_mask", "label"],
                },
            },
        ],
    },
}


# Preset cache configurations
CACHE_PRESETS = {
    "disabled": {
        "enabled": False,
    },
    
    "minimal": {
        "cache_type": "memory",
        "memory_max_items": 100,
        "cache_tokenized": True,
        "cache_converted_text": False,
    },
    
    "standard": {
        "cache_type": "disk",
        "disk_max_size_mb": 1000,
        "cache_tokenized": True,
        "cache_converted_text": True,
    },
    
    "aggressive": {
        "cache_type": "hybrid",
        "memory_max_items": 1000,
        "disk_max_size_mb": 5000,
        "cache_tokenized": True,
        "cache_converted_text": True,
        "cache_transformed": True,
        "dataset_cache_enabled": True,
    },
}


# Competition-specific presets
COMPETITION_PRESETS = {
    "titanic": {
        "dataset": {
            "label_column": "survived",
            "id_column": "passengerid",
            "data_format": "csv",
            "stratify": True,
        },
        "dataloader": {
            "text_converter": "titanic",
            "batch_size": 32,
            "max_length": 256,
            "augment": True,
        },
        "transforms": TRANSFORM_PRESETS["tabular_to_text"],
        "cache": CACHE_PRESETS["standard"],
    },
    
    "spaceship-titanic": {
        "dataset": {
            "label_column": "transported",
            "id_column": "passengerid",
            "data_format": "csv",
            "stratify": True,
        },
        "dataloader": {
            "text_converter": "spaceship-titanic",
            "batch_size": 32,
            "max_length": 256,
            "augment": True,
        },
        "transforms": TRANSFORM_PRESETS["tabular_to_text"],
        "cache": CACHE_PRESETS["standard"],
    },
}


# Main preset registry
PRESET_CONFIGS = {
    "dataloader": DATALOADER_PRESETS,
    "dataset": DATASET_PRESETS,
    "transform": TRANSFORM_PRESETS,
    "cache": CACHE_PRESETS,
    "competition": COMPETITION_PRESETS,
}


def get_preset_config(
    category: str,
    name: str,
    override: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Get preset configuration.
    
    Args:
        category: Category of preset (dataloader, dataset, etc.)
        name: Preset name
        override: Optional overrides to apply
        
    Returns:
        Configuration dictionary
    """
    if category not in PRESET_CONFIGS:
        raise ValueError(
            f"Unknown category: {category}. "
            f"Available: {list(PRESET_CONFIGS.keys())}"
        )
    
    category_presets = PRESET_CONFIGS[category]
    if name not in category_presets:
        raise ValueError(
            f"Unknown preset '{name}' in category '{category}'. "
            f"Available: {list(category_presets.keys())}"
        )
    
    # Get preset (make a copy)
    preset = deepcopy(category_presets[name])
    
    # Apply overrides
    if override:
        preset = merge_dicts(preset, override)
    
    return preset


def merge_dicts(base: Dict, override: Dict) -> Dict:
    """Recursively merge dictionaries."""
    result = deepcopy(base)
    
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = merge_dicts(result[key], value)
        else:
            result[key] = value
    
    return result


def register_preset(
    category: str,
    name: str,
    config: Dict[str, Any]
) -> None:
    """
    Register a new preset configuration.
    
    Args:
        category: Category to register under
        name: Preset name
        config: Configuration dictionary
    """
    if category not in PRESET_CONFIGS:
        PRESET_CONFIGS[category] = {}
    
    PRESET_CONFIGS[category][name] = config
    logger.info(f"Registered preset '{name}' in category '{category}'")


def list_presets(category: Optional[str] = None) -> Dict[str, List[str]]:
    """
    List available presets.
    
    Args:
        category: Optional category to filter by
        
    Returns:
        Dictionary of category -> preset names
    """
    if category:
        if category not in PRESET_CONFIGS:
            return {}
        return {category: list(PRESET_CONFIGS[category].keys())}
    else:
        return {
            cat: list(presets.keys())
            for cat, presets in PRESET_CONFIGS.items()
        }


def create_competition_config(
    competition: str,
    data_path: Optional[str] = None,
    **overrides
) -> Dict[str, Any]:
    """
    Create configuration for a competition.
    
    Args:
        competition: Competition name
        data_path: Optional data path
        **overrides: Configuration overrides
        
    Returns:
        Complete configuration
    """
    if competition not in COMPETITION_PRESETS:
        raise ValueError(
            f"Unknown competition: {competition}. "
            f"Available: {list(COMPETITION_PRESETS.keys())}"
        )
    
    config = deepcopy(COMPETITION_PRESETS[competition])
    
    # Set data path if provided
    if data_path:
        if "dataset" not in config:
            config["dataset"] = {}
        config["dataset"]["data_path"] = data_path
    
    # Apply overrides
    for key, value in overrides.items():
        if isinstance(value, dict) and key in config and isinstance(config[key], dict):
            config[key] = merge_dicts(config[key], value)
        else:
            config[key] = value
    
    return config