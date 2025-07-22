"""Configuration module for k-bert CLI.

This module provides configuration management functionality including
schemas, defaults, validation, and the main configuration manager.
"""

from .config_manager import (
    ConfigManager,
    get_config,
    get_value,
    set_value,
)
from .defaults import (
    get_default_config,
    get_competition_defaults,
    get_preset_config,
    COMPETITION_PROFILES,
    MODEL_PRESETS,
    TRAINING_PRESETS,
    LORA_PRESETS,
    AUGMENTATION_PRESETS,
)
from .schemas import (
    KBertConfig,
    KaggleConfig,
    ModelConfig,
    TrainingConfig,
    MLflowConfig,
    DataConfig,
    LoggingConfig,
    ProjectConfig,
    TrainingRunConfig,
    CompetitionConfig,
)
from .validators import (
    validate_config,
    validate_competition_config,
    validate_path,
    validate_cli_overrides,
    ConfigValidationError,
)

__all__ = [
    # Manager
    "ConfigManager",
    "get_config",
    "get_value",
    "set_value",
    # Schemas
    "KBertConfig",
    "KaggleConfig",
    "ModelConfig",
    "TrainingConfig",
    "MLflowConfig",
    "DataConfig",
    "LoggingConfig",
    "ProjectConfig",
    "TrainingRunConfig",
    "CompetitionConfig",
    # Defaults
    "get_default_config",
    "get_competition_defaults",
    "get_preset_config",
    "COMPETITION_PROFILES",
    "MODEL_PRESETS",
    "TRAINING_PRESETS",
    "LORA_PRESETS",
    "AUGMENTATION_PRESETS",
    # Validators
    "validate_config",
    "validate_competition_config",
    "validate_path",
    "validate_cli_overrides",
    "ConfigValidationError",
]