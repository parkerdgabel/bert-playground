"""
Unified configuration system for MLX dataloaders.
"""

from .base_config import (
    BaseConfig,
    DataLoaderConfig,
    DatasetConfig,
    OptimizationProfile,
    get_optimization_profile,
)

from .transform_config import (
    TransformConfig,
    TransformPipeline,
    create_transform_pipeline,
)

from .cache_config import (
    CacheConfig,
    create_cache,
)

from .config_factory import (
    ConfigFactory,
    load_config,
    save_config,
    validate_config,
    merge_configs,
)

from .presets import (
    PRESET_CONFIGS,
    get_preset_config,
    register_preset,
)

__all__ = [
    # Base configs
    "BaseConfig",
    "DataLoaderConfig",
    "DatasetConfig",
    "OptimizationProfile",
    "get_optimization_profile",
    
    # Transform configs
    "TransformConfig",
    "TransformPipeline",
    "create_transform_pipeline",
    
    # Cache configs
    "CacheConfig",
    "create_cache",
    
    # Factory functions
    "ConfigFactory",
    "load_config",
    "save_config",
    "validate_config",
    "merge_configs",
    
    # Presets
    "PRESET_CONFIGS",
    "get_preset_config",
    "register_preset",
]