"""
Simplified Kaggle Competition Factory - Works with existing TrainingConfig.

This module provides intelligent factory functions that automatically configure
the existing TrainingConfig for different types of Kaggle competitions.
"""

import pandas as pd
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple, Union
from enum import Enum
from loguru import logger

from training.config import TrainingConfig, OptimizationLevel, LossFunction
from models.classification.kaggle_heads import KAGGLE_HEAD_REGISTRY


class CompetitionType(Enum):
    """Types of Kaggle competitions."""
    BINARY_CLASSIFICATION = "binary_classification"
    MULTICLASS_CLASSIFICATION = "multiclass_classification"
    MULTILABEL_CLASSIFICATION = "multilabel_classification"
    REGRESSION = "regression"
    UNKNOWN = "unknown"


def create_competition_config(
    competition_name: str,
    data_path: str,
    **overrides
) -> TrainingConfig:
    """
    Create an optimized TrainingConfig for a Kaggle competition.
    
    Args:
        competition_name: Name of the competition
        data_path: Path to training data
        **overrides: Configuration overrides
        
    Returns:
        Optimized TrainingConfig
    """
    config = TrainingConfig()
    
    # Set basic paths and metadata
    config.train_path = data_path
    config.competition_name = competition_name
    config.experiment_name = f"{competition_name}_experiment"
    
    # Apply known competition presets
    if competition_name.lower() == "titanic":
        _configure_titanic(config)
    elif competition_name.lower() == "spaceship-titanic":
        _configure_spaceship_titanic(config)
    else:
        # Generic classification setup
        config.dataloader.text_converter = "template"
        config.competition_type = "classification"
    
    # Apply overrides
    for key, value in overrides.items():
        _apply_config_override(config, key, value)
    
    logger.info(f"Created config for competition: {competition_name}")
    return config


def _configure_titanic(config: TrainingConfig):
    """Configure for Titanic competition."""
    config.target_column = "survived"
    config.num_labels = 2
    config.task_type = "classification"
    config.head_type = "standard"
    config.competition_type = "classification"
    
    # Dataloader settings
    config.dataloader.text_converter = "titanic"
    config.dataloader.enable_text_augmentation = True
    config.dataloader.enable_caching = True
    
    # Training settings
    config.epochs = 5
    config.learning_rate = 2e-5
    config.batch_size = 32


def _configure_spaceship_titanic(config: TrainingConfig):
    """Configure for Spaceship Titanic competition."""
    config.target_column = "transported"
    config.num_labels = 2
    config.task_type = "classification"
    config.head_type = "standard"
    config.competition_type = "classification"
    
    # Dataloader settings
    config.dataloader.text_converter = "spaceship-titanic"
    config.dataloader.enable_text_augmentation = True
    config.dataloader.enable_caching = True
    
    # Training settings
    config.epochs = 5
    config.learning_rate = 1e-5
    config.batch_size = 32


def _apply_config_override(config: TrainingConfig, key: str, value: Any):
    """Apply a configuration override."""
    if "." in key:
        # Handle nested keys like "dataloader.batch_size"
        section, field = key.split(".", 1)
        if hasattr(config, section):
            section_obj = getattr(config, section)
            if hasattr(section_obj, field):
                setattr(section_obj, field, value)
    else:
        if hasattr(config, key):
            setattr(config, key, value)


def auto_configure_from_data(data_path: str) -> TrainingConfig:
    """
    Auto-configure based on dataset analysis.
    
    Args:
        data_path: Path to dataset
        
    Returns:
        Auto-configured TrainingConfig
    """
    config = TrainingConfig()
    config.train_path = data_path
    
    try:
        # Load a sample of the data
        df = pd.read_csv(data_path).head(1000)
        
        # Try to detect target column
        target_col = _detect_target_column(df)
        if target_col:
            config.target_column = target_col
            
            # Analyze target to determine task type
            competition_type, num_classes = _analyze_target(df, target_col)
            config.competition_type = competition_type.value
            config.num_labels = num_classes
            
            if competition_type == CompetitionType.BINARY_CLASSIFICATION:
                config.task_type = "classification"
                config.head_type = "standard"
            elif competition_type == CompetitionType.MULTICLASS_CLASSIFICATION:
                config.task_type = "classification"
                config.head_type = "standard"
            elif competition_type == CompetitionType.REGRESSION:
                config.task_type = "regression"
                config.loss_function = LossFunction.CROSS_ENTROPY  # Will need MSE for regression
        
        # Check if there are text columns that might benefit from text conversion
        text_cols = _find_text_columns(df)
        if text_cols:
            config.dataloader.text_converter = "template"
            config.dataloader.enable_text_augmentation = True
        
        logger.info(f"Auto-configured from data analysis")
        
    except Exception as e:
        logger.warning(f"Could not analyze data: {e}, using default config")
    
    return config


def _detect_target_column(df: pd.DataFrame) -> Optional[str]:
    """Detect the most likely target column."""
    # Common target column names
    common_targets = [
        'target', 'label', 'y', 'class', 'outcome', 'result',
        'survived', 'transported', 'price', 'score', 'rating'
    ]
    
    # Check for exact matches (case insensitive)
    df_lower_cols = [col.lower() for col in df.columns]
    for target in common_targets:
        if target in df_lower_cols:
            idx = df_lower_cols.index(target)
            return df.columns[idx]
    
    return None


def _analyze_target(df: pd.DataFrame, target_column: str) -> Tuple[CompetitionType, Optional[int]]:
    """Analyze the target variable to determine competition type."""
    target = df[target_column].dropna()
    
    # Check if it's numeric
    if pd.api.types.is_numeric_dtype(target):
        unique_vals = target.nunique()
        
        if unique_vals == 2:
            return CompetitionType.BINARY_CLASSIFICATION, 2
        elif unique_vals <= 50 and target.dtype in ['int64', 'int32']:
            return CompetitionType.MULTICLASS_CLASSIFICATION, unique_vals
        else:
            return CompetitionType.REGRESSION, None
    else:
        # Categorical target
        unique_vals = target.nunique()
        if unique_vals == 2:
            return CompetitionType.BINARY_CLASSIFICATION, 2
        else:
            return CompetitionType.MULTICLASS_CLASSIFICATION, unique_vals


def _find_text_columns(df: pd.DataFrame) -> List[str]:
    """Find columns that contain text data."""
    text_columns = []
    
    for col in df.columns:
        if df[col].dtype == 'object':
            # Check average string length
            avg_length = df[col].astype(str).str.len().mean()
            if avg_length > 10:  # Likely text rather than categorical
                text_columns.append(col)
    
    return text_columns


def create_development_config(data_path: str, **overrides) -> TrainingConfig:
    """Create a fast development configuration."""
    config = TrainingConfig()
    config.train_path = data_path
    config.optimization_level = OptimizationLevel.DEVELOPMENT
    config.epochs = 1
    config.batch_size = 8
    config.dataloader.optimization_profile = "debug"
    config.experiment_name = "development_test"
    
    # Apply overrides
    for key, value in overrides.items():
        _apply_config_override(config, key, value)
    
    return config


def create_production_config(
    competition_name: str,
    data_path: str,
    **overrides
) -> TrainingConfig:
    """Create a production-optimized configuration."""
    config = create_competition_config(competition_name, data_path)
    config.optimization_level = OptimizationLevel.PRODUCTION
    config.epochs = 10
    config.dataloader.optimization_profile = "speed"
    config.dataloader.enable_caching = True
    
    # Apply overrides
    for key, value in overrides.items():
        _apply_config_override(config, key, value)
    
    return config

def create_kaggle_head_config(
    head_type: str,
    competition_data: str,
    **overrides
) -> TrainingConfig:
    """
    Create configuration optimized for specific Kaggle head types.
    
    Args:
        head_type: Type of head ("time_series", "ranking", "contrastive", etc.)
        competition_data: Path to competition data
        **overrides: Configuration overrides
        
    Returns:
        Optimized TrainingConfig for the head type
    """
    if head_type not in KAGGLE_HEAD_REGISTRY:
        raise ValueError(f"Unknown head type: {head_type}. Available: {list(KAGGLE_HEAD_REGISTRY.keys())}")
    
    config = TrainingConfig()
    config.train_path = competition_data
    config.head_type = head_type
    config.task_type = head_type
    
    # Optimize based on head type
    if head_type == "time_series":
        config.epochs = 15  # Time series often needs more epochs
        config.learning_rate = 1e-4
        config.batch_size = 16  # Smaller batches for sequence data
        config.dataloader.optimization_profile = "memory"
        
    elif head_type == "ranking":
        config.epochs = 8
        config.learning_rate = 2e-5
        config.batch_size = 32
        config.dataloader.optimization_profile = "balanced"
        
    elif head_type == "contrastive":
        config.epochs = 20  # Contrastive learning often needs more epochs
        config.learning_rate = 1e-4
        config.batch_size = 64  # Larger batches for contrastive learning
        config.dataloader.optimization_profile = "speed"
        
    elif head_type == "multi_task":
        config.epochs = 10
        config.learning_rate = 2e-5
        config.batch_size = 32
        config.dataloader.optimization_profile = "balanced"
        
    elif head_type == "metric_learning":
        config.epochs = 15
        config.learning_rate = 1e-4
        config.batch_size = 32
        config.dataloader.optimization_profile = "balanced"
    
    # Apply overrides
    for key, value in overrides.items():
        _apply_config_override(config, key, value)
    
    logger.info(f"Created {head_type} head configuration")
    return config


def list_available_kaggle_heads() -> List[str]:
    """List all available Kaggle-specific head types."""
    return list(KAGGLE_HEAD_REGISTRY.keys())


def get_head_type_description(head_type: str) -> str:
    """Get description of what a head type is used for."""
    descriptions = {
        "time_series": "Sequential/temporal data with LSTM and attention mechanisms",
        "ranking": "Learning-to-rank for recommendation and ranking tasks",
        "contrastive": "Similarity learning and retrieval tasks",
        "multi_task": "Multiple objectives/targets in a single model",
        "metric_learning": "Learning embeddings with specific distance properties",
    }
    return descriptions.get(head_type, f"Kaggle-specific head: {head_type}")
EOF < /dev/null