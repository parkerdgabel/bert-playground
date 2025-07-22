"""Default configurations for k-bert CLI.

This module provides default configuration values and competition profiles
for common Kaggle competitions.
"""

from pathlib import Path
from typing import Dict, Any

from .schemas import (
    KBertConfig,
    KaggleConfig,
    ModelConfig,
    TrainingConfig,
    MLflowConfig,
    DataConfig,
    LoggingConfig,
    CompetitionConfig,
)


def get_default_config() -> KBertConfig:
    """Get the default k-bert configuration."""
    return KBertConfig(
        version="1.0",
        kaggle=KaggleConfig(
            username=None,
            key=None,
            default_competition=None,
            auto_download=True,
            submission_message="Submitted by k-bert",
            competitions_dir=Path("./competitions"),
        ),
        models=ModelConfig(
            default_model="answerdotai/ModernBERT-base",
            cache_dir=Path("~/.k-bert/models").expanduser(),
            use_mlx_embeddings=True,
            default_architecture="modernbert",
            use_lora=False,
            lora_preset="balanced",
        ),
        training=TrainingConfig(
            default_batch_size=32,
            default_epochs=5,
            default_learning_rate=2e-5,
            output_dir=Path("./outputs"),
            save_best_only=True,
            early_stopping_patience=3,
            gradient_accumulation_steps=1,
            warmup_ratio=0.1,
            max_grad_norm=1.0,
            seed=42,
            mixed_precision=True,
            use_compilation=True,
        ),
        mlflow=MLflowConfig(
            tracking_uri="file://~/.k-bert/mlruns",
            default_experiment="k-bert-experiments",
            auto_log=True,
            log_models=True,
            log_metrics=True,
        ),
        data=DataConfig(
            cache_dir=Path("~/.k-bert/cache").expanduser(),
            max_length=256,
            use_pretokenized=True,
            augmentation_mode="moderate",
            num_workers=4,
            prefetch_size=4,
            mlx_prefetch_size=None,
            tokenizer_backend="auto",
        ),
        logging=LoggingConfig(
            level="INFO",
            format="structured",
            file_output=True,
            file_dir=Path("~/.k-bert/logs").expanduser(),
            rotation="500 MB",
            retention="30 days",
        ),
    )


# Competition profiles for common Kaggle competitions
COMPETITION_PROFILES: Dict[str, CompetitionConfig] = {
    "titanic": CompetitionConfig(
        name="titanic",
        type="binary_classification",
        metrics=["accuracy", "f1"],
        data_dir=Path("competitions/titanic"),
        train_file="train.csv",
        test_file="test.csv",
        sample_submission_file="gender_submission.csv",
        target_column="Survived",
        id_column="PassengerId",
        text_columns=["Name", "Ticket"],
        feature_columns=[
            "Pclass", "Sex", "Age", "SibSp", "Parch", 
            "Fare", "Cabin", "Embarked"
        ],
        recommended_models=["answerdotai/ModernBERT-base"],
        recommended_batch_size=32,
        recommended_max_length=256,
    ),
    
    "house-prices": CompetitionConfig(
        name="house-prices-advanced-regression-techniques",
        type="regression",
        metrics=["rmse", "mae"],
        data_dir=Path("competitions/house-prices"),
        train_file="train.csv",
        test_file="test.csv",
        sample_submission_file="sample_submission.csv",
        target_column="SalePrice",
        id_column="Id",
        text_columns=None,
        feature_columns=None,  # Too many to list
        recommended_models=["answerdotai/ModernBERT-base"],
        recommended_batch_size=16,
        recommended_max_length=512,
    ),
    
    "digit-recognizer": CompetitionConfig(
        name="digit-recognizer",
        type="multiclass_classification",
        metrics=["accuracy"],
        data_dir=Path("competitions/digit-recognizer"),
        train_file="train.csv",
        test_file="test.csv",
        sample_submission_file="sample_submission.csv",
        target_column="label",
        id_column=None,
        text_columns=None,
        feature_columns=None,  # Pixel values
        recommended_models=["answerdotai/ModernBERT-base"],
        recommended_batch_size=64,
        recommended_max_length=128,
    ),
    
    "nlp-getting-started": CompetitionConfig(
        name="nlp-getting-started",
        type="binary_classification",
        metrics=["f1"],
        data_dir=Path("competitions/nlp-getting-started"),
        train_file="train.csv",
        test_file="test.csv",
        sample_submission_file="sample_submission.csv",
        target_column="target",
        id_column="id",
        text_columns=["text"],
        feature_columns=["keyword", "location"],
        recommended_models=["answerdotai/ModernBERT-base"],
        recommended_batch_size=32,
        recommended_max_length=128,
    ),
}


# Model architecture presets
MODEL_PRESETS = {
    "modernbert-base": {
        "model_name": "answerdotai/ModernBERT-base",
        "architecture": "modernbert",
        "hidden_size": 768,
        "num_layers": 12,
        "num_heads": 12,
        "max_length": 8192,
    },
    "modernbert-large": {
        "model_name": "answerdotai/ModernBERT-large",
        "architecture": "modernbert",
        "hidden_size": 1024,
        "num_layers": 24,
        "num_heads": 16,
        "max_length": 8192,
    },
    "bert-base": {
        "model_name": "bert-base-uncased",
        "architecture": "bert",
        "hidden_size": 768,
        "num_layers": 12,
        "num_heads": 12,
        "max_length": 512,
    },
    "neobert": {
        "model_name": "neobert-250m",
        "architecture": "neobert",
        "hidden_size": 768,
        "num_layers": 28,
        "num_heads": 12,
        "max_length": 4096,
    },
}


# Training presets for different scenarios
TRAINING_PRESETS = {
    "quick": {
        "epochs": 1,
        "batch_size": 32,
        "learning_rate": 2e-5,
        "warmup_ratio": 0.1,
        "save_best_only": False,
        "early_stopping_patience": 0,
    },
    "standard": {
        "epochs": 5,
        "batch_size": 32,
        "learning_rate": 2e-5,
        "warmup_ratio": 0.1,
        "save_best_only": True,
        "early_stopping_patience": 3,
    },
    "thorough": {
        "epochs": 10,
        "batch_size": 16,
        "learning_rate": 1e-5,
        "warmup_ratio": 0.2,
        "gradient_accumulation_steps": 2,
        "save_best_only": True,
        "early_stopping_patience": 5,
    },
    "competition": {
        "epochs": 20,
        "batch_size": 32,
        "learning_rate": 2e-5,
        "warmup_ratio": 0.1,
        "gradient_accumulation_steps": 2,
        "save_best_only": True,
        "early_stopping_patience": 5,
        "use_lora": True,
    },
}


# LoRA presets
LORA_PRESETS = {
    "minimal": {
        "r": 4,
        "alpha": 8,
        "dropout": 0.05,
        "target_modules": ["query", "value"],
    },
    "balanced": {
        "r": 8,
        "alpha": 16,
        "dropout": 0.1,
        "target_modules": ["query", "value", "key", "dense"],
    },
    "aggressive": {
        "r": 16,
        "alpha": 32,
        "dropout": 0.1,
        "target_modules": ["query", "value", "key", "dense", "intermediate", "output"],
    },
}


# Data augmentation presets
AUGMENTATION_PRESETS = {
    "none": {
        "enabled": False,
    },
    "light": {
        "enabled": True,
        "augmentation_prob": 0.2,
        "numerical_noise": 0.05,
        "categorical_swap_prob": 0.1,
        "text_aug_prob": 0.1,
    },
    "moderate": {
        "enabled": True,
        "augmentation_prob": 0.5,
        "numerical_noise": 0.1,
        "categorical_swap_prob": 0.2,
        "text_aug_prob": 0.2,
    },
    "heavy": {
        "enabled": True,
        "augmentation_prob": 0.8,
        "numerical_noise": 0.15,
        "categorical_swap_prob": 0.3,
        "text_aug_prob": 0.3,
    },
}


def get_competition_defaults(competition_name: str) -> Dict[str, Any]:
    """Get default configuration for a specific competition.
    
    Args:
        competition_name: Name of the competition
        
    Returns:
        Dictionary with competition-specific defaults
    """
    if competition_name not in COMPETITION_PROFILES:
        return {}
    
    profile = COMPETITION_PROFILES[competition_name]
    
    return {
        "data": {
            "max_length": profile.recommended_max_length or 256,
        },
        "training": {
            "default_batch_size": profile.recommended_batch_size or 32,
        },
        "models": {
            "default_model": (
                profile.recommended_models[0] 
                if profile.recommended_models 
                else "answerdotai/ModernBERT-base"
            ),
        },
    }


def get_preset_config(preset_type: str, preset_name: str) -> Dict[str, Any]:
    """Get a preset configuration.
    
    Args:
        preset_type: Type of preset (model, training, lora, augmentation)
        preset_name: Name of the preset
        
    Returns:
        Dictionary with preset configuration
    """
    preset_maps = {
        "model": MODEL_PRESETS,
        "training": TRAINING_PRESETS,
        "lora": LORA_PRESETS,
        "augmentation": AUGMENTATION_PRESETS,
    }
    
    if preset_type not in preset_maps:
        raise ValueError(f"Unknown preset type: {preset_type}")
    
    preset_map = preset_maps[preset_type]
    if preset_name not in preset_map:
        raise ValueError(
            f"Unknown {preset_type} preset: {preset_name}. "
            f"Available: {list(preset_map.keys())}"
        )
    
    return preset_map[preset_name]