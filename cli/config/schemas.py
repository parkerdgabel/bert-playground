"""Configuration schemas for k-bert CLI.

This module defines the configuration structure and validation schemas
for the k-bert CLI tool using Pydantic for robust validation.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class KaggleConfig(BaseModel):
    """Kaggle API configuration."""
    
    username: Optional[str] = Field(None, description="Kaggle username")
    key: Optional[str] = Field(None, description="Kaggle API key")
    default_competition: Optional[str] = Field(None, description="Default competition name")
    auto_download: bool = Field(True, description="Automatically download competition data")
    submission_message: str = Field("Submitted by k-bert", description="Default submission message")
    competitions_dir: Path = Field(Path("./competitions"), description="Directory for competition data")


class ModelConfig(BaseModel):
    """Model configuration."""
    
    default_model: str = Field("answerdotai/ModernBERT-base", description="Default model name")
    cache_dir: Path = Field(Path("~/.k-bert/models").expanduser(), description="Model cache directory")
    use_mlx_embeddings: bool = Field(True, description="Use MLX embeddings by default")
    default_architecture: str = Field("modernbert", description="Default architecture type")
    use_lora: bool = Field(False, description="Use LoRA adaptation by default")
    lora_preset: str = Field("balanced", description="Default LoRA preset")


class TrainingConfig(BaseModel):
    """Training configuration."""
    
    default_batch_size: int = Field(32, description="Default batch size", ge=1)
    default_epochs: int = Field(5, description="Default number of epochs", ge=1)
    default_learning_rate: float = Field(2e-5, description="Default learning rate", gt=0)
    output_dir: Path = Field(Path("./outputs"), description="Default output directory")
    save_best_only: bool = Field(True, description="Save only best model")
    early_stopping_patience: int = Field(3, description="Early stopping patience", ge=0)
    gradient_accumulation_steps: int = Field(1, description="Gradient accumulation steps", ge=1)
    warmup_ratio: float = Field(0.1, description="Warmup ratio", ge=0, le=1)
    max_grad_norm: float = Field(1.0, description="Max gradient norm", gt=0)
    seed: int = Field(42, description="Random seed")
    mixed_precision: bool = Field(True, description="Use mixed precision training")
    use_compilation: bool = Field(True, description="Use MLX compilation")


class MLflowConfig(BaseModel):
    """MLflow configuration."""
    
    tracking_uri: str = Field("file://~/.k-bert/mlruns", description="MLflow tracking URI")
    default_experiment: str = Field("k-bert-experiments", description="Default experiment name")
    auto_log: bool = Field(True, description="Enable automatic logging")
    log_models: bool = Field(True, description="Log models to MLflow")
    log_metrics: bool = Field(True, description="Log metrics to MLflow")


class DataConfig(BaseModel):
    """Data processing configuration."""
    
    cache_dir: Path = Field(Path("~/.k-bert/cache").expanduser(), description="Data cache directory")
    max_length: int = Field(256, description="Maximum sequence length", ge=1)
    use_pretokenized: bool = Field(True, description="Use pre-tokenized data")
    augmentation_mode: str = Field("moderate", description="Augmentation mode")
    num_workers: int = Field(4, description="Number of data loading workers", ge=0)
    prefetch_size: int = Field(4, description="Prefetch size", ge=0)
    mlx_prefetch_size: Optional[int] = Field(None, description="MLX-specific prefetch size")
    tokenizer_backend: str = Field("auto", description="Tokenizer backend")
    
    @field_validator("augmentation_mode")
    @classmethod
    def validate_augmentation_mode(cls, v: str) -> str:
        """Validate augmentation mode."""
        valid_modes = ["none", "light", "moderate", "heavy"]
        if v not in valid_modes:
            raise ValueError(f"Invalid augmentation mode. Must be one of: {valid_modes}")
        return v


class LoggingConfig(BaseModel):
    """Logging configuration."""
    
    level: str = Field("INFO", description="Logging level")
    format: str = Field("structured", description="Log format (structured or simple)")
    file_output: bool = Field(True, description="Enable file logging")
    file_dir: Path = Field(Path("~/.k-bert/logs").expanduser(), description="Log file directory")
    rotation: str = Field("500 MB", description="Log file rotation")
    retention: str = Field("30 days", description="Log file retention")
    
    @field_validator("level")
    @classmethod
    def validate_level(cls, v: str) -> str:
        """Validate logging level."""
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        v_upper = v.upper()
        if v_upper not in valid_levels:
            raise ValueError(f"Invalid logging level. Must be one of: {valid_levels}")
        return v_upper


class ProjectConfig(BaseModel):
    """Project-level configuration."""
    
    name: str = Field(..., description="Project name")
    competition: Optional[str] = Field(None, description="Competition name")
    description: Optional[str] = Field(None, description="Project description")
    version: str = Field("1.0", description="Configuration version")
    
    # Override configurations
    kaggle: Optional[KaggleConfig] = None
    models: Optional[ModelConfig] = None
    training: Optional[TrainingConfig] = None
    mlflow: Optional[MLflowConfig] = None
    data: Optional[DataConfig] = None
    logging: Optional[LoggingConfig] = None
    
    # Project-specific settings
    experiments: Optional[List[Dict[str, Any]]] = Field(None, description="Experiment configurations")
    pipelines: Optional[Dict[str, Any]] = Field(None, description="Pipeline definitions")


class KBertConfig(BaseSettings):
    """Main k-bert configuration.
    
    This is the root configuration that combines all sub-configurations
    and handles environment variable loading.
    """
    
    model_config = SettingsConfigDict(
        env_prefix="K_BERT_",
        env_nested_delimiter="__",
        case_sensitive=False,
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )
    
    version: str = Field("1.0", description="Configuration version")
    kaggle: KaggleConfig = Field(default_factory=KaggleConfig)
    models: ModelConfig = Field(default_factory=ModelConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    mlflow: MLflowConfig = Field(default_factory=MLflowConfig)
    data: DataConfig = Field(default_factory=DataConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "KBertConfig":
        """Create configuration from dictionary."""
        return cls(**data)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return self.model_dump(exclude_none=True)
    
    def merge(self, other: Union["KBertConfig", Dict[str, Any]]) -> "KBertConfig":
        """Merge with another configuration."""
        if isinstance(other, dict):
            other_dict = other
        else:
            other_dict = other.to_dict()
        
        current_dict = self.to_dict()
        merged = self._deep_merge(current_dict, other_dict)
        return self.__class__.from_dict(merged)
    
    @staticmethod
    def _deep_merge(base: Dict[str, Any], update: Dict[str, Any]) -> Dict[str, Any]:
        """Deep merge two dictionaries."""
        result = base.copy()
        
        for key, value in update.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = KBertConfig._deep_merge(result[key], value)
            else:
                result[key] = value
        
        return result


# Configuration validation schemas for specific use cases
class TrainingRunConfig(BaseModel):
    """Configuration for a single training run."""
    
    name: str = Field(..., description="Run name")
    model: str = Field(..., description="Model name or path")
    data: Dict[str, Path] = Field(..., description="Data paths (train, val, test)")
    
    # Training parameters
    epochs: int = Field(..., ge=1)
    batch_size: int = Field(..., ge=1)
    learning_rate: float = Field(..., gt=0)
    
    # Optional overrides
    model_config: Optional[Dict[str, Any]] = None
    training_config: Optional[Dict[str, Any]] = None
    data_config: Optional[Dict[str, Any]] = None
    
    # Experiment tracking
    experiment_name: Optional[str] = None
    tags: Optional[Dict[str, str]] = None
    
    @field_validator("data")
    @classmethod
    def validate_data_paths(cls, v: Dict[str, Path]) -> Dict[str, Path]:
        """Validate that required data paths exist."""
        if "train" not in v:
            raise ValueError("Training data path is required")
        
        # Convert to Path objects and validate existence
        result = {}
        for key, path in v.items():
            path_obj = Path(path)
            if key in ["train", "val"] and not path_obj.exists():
                raise ValueError(f"{key} data path does not exist: {path}")
            result[key] = path_obj
        
        return result


class CompetitionConfig(BaseModel):
    """Configuration for a Kaggle competition."""
    
    name: str = Field(..., description="Competition name")
    type: str = Field(..., description="Competition type")
    metrics: List[str] = Field(..., description="Evaluation metrics")
    
    # Data configuration
    data_dir: Path = Field(..., description="Data directory")
    train_file: str = Field("train.csv", description="Training file name")
    test_file: str = Field("test.csv", description="Test file name")
    sample_submission_file: Optional[str] = Field("sample_submission.csv", description="Sample submission file")
    
    # Competition-specific settings
    target_column: Optional[str] = None
    id_column: Optional[str] = None
    text_columns: Optional[List[str]] = None
    feature_columns: Optional[List[str]] = None
    
    # Model recommendations
    recommended_models: Optional[List[str]] = None
    recommended_batch_size: Optional[int] = None
    recommended_max_length: Optional[int] = None
    
    @field_validator("type")
    @classmethod
    def validate_competition_type(cls, v: str) -> str:
        """Validate competition type."""
        valid_types = [
            "binary_classification",
            "multiclass_classification",
            "multilabel_classification",
            "regression",
            "ordinal_regression",
            "time_series",
            "ranking",
        ]
        if v not in valid_types:
            raise ValueError(f"Invalid competition type. Must be one of: {valid_types}")
        return v