"""
Base configuration classes for MLX dataloaders.
"""

from typing import Dict, Any, Optional, List, Union, Callable
from dataclasses import dataclass, field, asdict
from pathlib import Path
import json
from enum import Enum
from loguru import logger


class OptimizationProfile(Enum):
    """Predefined optimization profiles."""
    SPEED = "speed"
    MEMORY = "memory"
    BALANCED = "balanced"
    DEBUG = "debug"


@dataclass
class BaseConfig:
    """Base configuration class."""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "BaseConfig":
        """Create from dictionary."""
        return cls(**config_dict)
    
    def save(self, path: Union[str, Path]) -> None:
        """Save configuration to JSON file."""
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, path: Union[str, Path]) -> "BaseConfig":
        """Load configuration from JSON file."""
        with open(path, "r") as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)
    
    def validate(self) -> List[str]:
        """
        Validate configuration.
        
        Returns:
            List of validation errors (empty if valid)
        """
        return []


@dataclass
class DatasetConfig(BaseConfig):
    """Configuration for dataset."""
    
    # Data source
    data_path: Optional[Union[str, Path]] = None
    data_format: str = "csv"  # csv, json, parquet, custom
    
    # Data loading
    columns: Optional[List[str]] = None
    label_column: Optional[str] = None
    text_column: Optional[str] = None
    id_column: Optional[str] = None
    
    # Splitting
    train_split: float = 0.8
    val_split: float = 0.1
    test_split: float = 0.1
    stratify: bool = True
    shuffle: bool = True
    random_seed: int = 42
    
    # Filtering
    max_samples: Optional[int] = None
    filter_empty: bool = True
    filter_conditions: Dict[str, Any] = field(default_factory=dict)
    
    # Memory optimization
    lazy_loading: bool = False
    chunk_size: Optional[int] = None
    
    def validate(self) -> List[str]:
        """Validate dataset configuration."""
        errors = []
        
        # Check splits
        total_split = self.train_split + self.val_split + self.test_split
        if abs(total_split - 1.0) > 0.001:
            errors.append(f"Splits must sum to 1.0, got {total_split}")
        
        # Check data format
        valid_formats = ["csv", "json", "parquet", "custom"]
        if self.data_format not in valid_formats:
            errors.append(f"Invalid data format: {self.data_format}")
        
        return errors


@dataclass
class DataLoaderConfig(BaseConfig):
    """Configuration for data loader."""
    
    # Batching
    batch_size: int = 32
    shuffle: bool = True
    drop_last: bool = False
    
    # Tokenization
    tokenizer_name: str = "answerdotai/ModernBERT-base"
    tokenizer_backend: str = "huggingface"  # huggingface, mlx
    max_length: int = 256
    padding: Union[bool, str] = "max_length"
    truncation: bool = True
    return_attention_mask: bool = True
    
    # Text conversion
    text_converter: str = "template"  # template, feature, natural_language, competition
    text_converter_config: Dict[str, Any] = field(default_factory=dict)
    
    # Transforms
    transforms: List[Dict[str, Any]] = field(default_factory=list)
    augment: bool = False
    augmentation_prob: float = 0.5
    
    # Caching
    enable_cache: bool = True
    cache_dir: Optional[str] = None
    cache_tokenized: bool = True
    cache_converted_text: bool = True
    
    # MLX optimization
    prefetch_size: int = 4
    num_workers: int = 4
    buffer_size: int = 1000
    
    # Optimization profile
    optimization_profile: Optional[str] = None
    
    # Advanced settings
    pin_memory: bool = False
    persistent_workers: bool = True
    
    def validate(self) -> List[str]:
        """Validate dataloader configuration."""
        errors = []
        
        # Check batch size
        if self.batch_size <= 0:
            errors.append(f"Batch size must be positive, got {self.batch_size}")
        
        # Check max length
        if self.max_length <= 0:
            errors.append(f"Max length must be positive, got {self.max_length}")
        
        # Check workers
        if self.num_workers < 0:
            errors.append(f"Number of workers must be non-negative, got {self.num_workers}")
        
        return errors
    
    def apply_optimization_profile(self, profile: Union[str, OptimizationProfile]) -> None:
        """Apply optimization profile to configuration."""
        if isinstance(profile, str):
            profile = OptimizationProfile(profile)
        
        if profile == OptimizationProfile.SPEED:
            self.prefetch_size = 8
            self.num_workers = 8
            self.buffer_size = 2000
            self.enable_cache = True
            self.cache_tokenized = True
            self.persistent_workers = True
        
        elif profile == OptimizationProfile.MEMORY:
            self.prefetch_size = 2
            self.num_workers = 2
            self.buffer_size = 500
            self.batch_size = min(self.batch_size, 16)
            self.enable_cache = False
            self.persistent_workers = False
        
        elif profile == OptimizationProfile.BALANCED:
            self.prefetch_size = 4
            self.num_workers = 4
            self.buffer_size = 1000
            self.enable_cache = True
            self.cache_tokenized = True
            self.persistent_workers = True
        
        elif profile == OptimizationProfile.DEBUG:
            self.prefetch_size = 1
            self.num_workers = 0
            self.buffer_size = 100
            self.batch_size = min(self.batch_size, 8)
            self.shuffle = False
        
        self.optimization_profile = profile.value


def get_optimization_profile(name: str) -> Dict[str, Any]:
    """
    Get settings for an optimization profile.
    
    Args:
        name: Profile name
        
    Returns:
        Dictionary of settings
    """
    config = DataLoaderConfig()
    config.apply_optimization_profile(name)
    return config.to_dict()


@dataclass
class ExperimentConfig(BaseConfig):
    """Configuration for an entire experiment."""
    
    # Experiment info
    name: str = "mlx_experiment"
    description: str = ""
    tags: List[str] = field(default_factory=list)
    
    # Component configs
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    dataloader: DataLoaderConfig = field(default_factory=DataLoaderConfig)
    
    # Model settings (optional, for reference)
    model_config: Dict[str, Any] = field(default_factory=dict)
    
    # Training settings (optional, for reference)
    training_config: Dict[str, Any] = field(default_factory=dict)
    
    # Output settings
    output_dir: str = "./output"
    save_preprocessed: bool = False
    
    def validate(self) -> List[str]:
        """Validate experiment configuration."""
        errors = []
        
        # Validate sub-configs
        errors.extend(self.dataset.validate())
        errors.extend(self.dataloader.validate())
        
        # Check name
        if not self.name:
            errors.append("Experiment name is required")
        
        return errors


@dataclass
class CompetitionConfig(ExperimentConfig):
    """Configuration for competition-specific experiments."""
    
    # Competition info
    competition_name: str = ""
    competition_type: str = "classification"  # classification, regression, etc.
    
    # Submission settings
    submission_format: Dict[str, Any] = field(default_factory=dict)
    test_data_path: Optional[str] = None
    
    # Competition-specific settings
    competition_params: Dict[str, Any] = field(default_factory=dict)
    
    def setup_for_competition(self, competition: str) -> None:
        """
        Set up configuration for a specific competition.
        
        Args:
            competition: Competition name
        """
        self.competition_name = competition
        
        # Apply competition-specific defaults
        if competition.lower() == "titanic":
            self.dataset.label_column = "survived"
            self.dataset.id_column = "passengerid"
            self.dataloader.text_converter = "titanic"
            self.competition_type = "classification"
        
        elif competition.lower() == "spaceship-titanic":
            self.dataset.label_column = "transported"
            self.dataset.id_column = "passengerid"
            self.dataloader.text_converter = "spaceship-titanic"
            self.competition_type = "classification"