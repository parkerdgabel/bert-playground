"""
Unified Configuration System for BERT Kaggle Playground.

This module provides a comprehensive configuration system that unifies the existing
TrainingConfig with the modular dataloader's ExperimentConfig, creating a seamless
interface for configuring the entire training pipeline.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Union, List
from pathlib import Path
import json
from enum import Enum

from training.config import (
    TrainingConfig, 
    OptimizationLevel, 
    LearningRateSchedule, 
    OptimizerType, 
    LossFunction,
    MemoryConfig,
    MLXOptimizationConfig,
    MonitoringConfig,
    CheckpointConfig,
    EvaluationConfig,
    AdvancedFeatures
)
from data.configs import ConfigFactory, DataLoaderConfig, DatasetConfig
from data.configs.base_config import ExperimentConfig


@dataclass
class DataPipelineConfig:
    """Configuration for the data pipeline integration."""
    
    # Data loading strategy
    use_modular_dataloader: bool = True
    fallback_to_legacy: bool = True
    
    # Competition-specific settings
    competition_name: Optional[str] = None
    competition_type: str = "classification"  # classification, regression, ranking, etc.
    auto_detect_competition: bool = True
    
    # Text conversion settings
    text_converter: Optional[str] = None  # Auto-detected if None
    text_conversion_config: Dict[str, Any] = field(default_factory=dict)
    enable_text_augmentation: bool = True
    augmentation_probability: float = 0.5
    
    # Caching configuration
    enable_caching: bool = True
    cache_dir: Optional[str] = None  # Auto-generated if None
    cache_tokenized: bool = True
    cache_converted_text: bool = True
    
    # Data splitting
    train_split: float = 0.8
    val_split: float = 0.1
    test_split: float = 0.1
    stratify: bool = True
    
    # Performance optimization
    optimization_profile: str = "balanced"  # speed, memory, balanced, debug
    prefetch_size: int = 4
    num_workers: int = 4
    buffer_size: int = 1000


@dataclass
class ModelConfig:
    """Enhanced model configuration with head attachment capabilities."""
    
    # Base model settings
    model_name: str = "answerdotai/ModernBERT-base"
    model_type: str = "bert_with_head"  # bert_core, bert_with_head
    max_length: int = 256
    
    # Task configuration
    task_type: str = "classification"  # classification, regression, multilabel, etc.
    num_labels: Optional[int] = None  # Auto-detected from data
    
    # Head configuration
    head_type: str = "standard"  # standard, multilabel, ordinal, hierarchical, ensemble
    head_config: Dict[str, Any] = field(default_factory=dict)
    pooling_strategy: str = "cls"  # cls, mean, max, attention, weighted, learned
    
    # Custom head registry
    custom_head_path: Optional[str] = None
    head_registry_config: Dict[str, Any] = field(default_factory=dict)
    
    # MLX embeddings support
    use_mlx_embeddings: bool = False
    mlx_model_path: Optional[str] = None
    tokenizer_backend: str = "huggingface"  # huggingface, mlx


@dataclass 
class UnifiedConfig:
    """
    Unified configuration that combines TrainingConfig with modular dataloader capabilities.
    
    This class provides a seamless interface for configuring the entire BERT Kaggle
    playground training pipeline, from data loading to model training and evaluation.
    """
    
    # Core training configuration
    training: TrainingConfig = field(default_factory=TrainingConfig)
    
    # Data pipeline configuration
    data_pipeline: DataPipelineConfig = field(default_factory=DataPipelineConfig)
    
    # Enhanced model configuration
    model: ModelConfig = field(default_factory=ModelConfig)
    
    # Experiment metadata
    experiment_name: str = "bert_kaggle_experiment"
    description: str = ""
    tags: List[str] = field(default_factory=list)
    
    # Output settings
    output_dir: str = "output"
    save_preprocessed: bool = False
    
    def __post_init__(self):
        """Post-initialization setup and validation."""
        # Sync some fields between configs for compatibility
        self._sync_configurations()
        
        # Apply auto-configuration if requested
        if self.data_pipeline.auto_detect_competition and self.data_pipeline.competition_name:
            self._apply_competition_presets()
    
    def _sync_configurations(self):
        """Synchronize related fields between different configuration sections."""
        # Sync basic training parameters
        if hasattr(self.training, 'batch_size') and hasattr(self.data_pipeline, 'optimization_profile'):
            # Update batch size based on optimization profile if not explicitly set
            profile_batch_sizes = {
                "speed": 64,
                "balanced": 32, 
                "memory": 16,
                "debug": 8
            }
            if self.data_pipeline.optimization_profile in profile_batch_sizes:
                if self.training.batch_size == 32:  # Default value
                    self.training.batch_size = profile_batch_sizes[self.data_pipeline.optimization_profile]
        
        # Sync model parameters
        if self.training.max_length != self.model.max_length:
            # Use the model config as source of truth
            self.training.max_length = self.model.max_length
        
        # Sync model name
        if self.training.model_name != self.model.model_name:
            self.training.model_name = self.model.model_name
    
    def _apply_competition_presets(self):
        """Apply competition-specific presets and optimizations."""
        if not self.data_pipeline.competition_name:
            return
        
        competition = self.data_pipeline.competition_name.lower()
        
        # Apply known competition presets
        if competition in ["titanic", "spaceship-titanic"]:
            self.data_pipeline.text_converter = competition
            self.data_pipeline.competition_type = "classification"
            self.model.task_type = "classification"
            self.model.num_labels = 2
            
        # Add more competition presets as needed
        # TODO: Implement auto-detection based on data characteristics
    
    def to_experiment_config(self) -> ExperimentConfig:
        """Convert to modular dataloader's ExperimentConfig format."""
        # Create dataset config
        dataset_config = DatasetConfig(
            data_path=self.training.train_path or "",
            label_column=self.training.target_column or "label",
            train_split=self.data_pipeline.train_split,
            val_split=self.data_pipeline.val_split,
            test_split=self.data_pipeline.test_split,
            stratify=self.data_pipeline.stratify,
        )
        
        # Create dataloader config
        dataloader_config = DataLoaderConfig(
            batch_size=self.training.batch_size,
            max_length=self.model.max_length,
            tokenizer_name=self.model.model_name,
            tokenizer_backend=self.model.tokenizer_backend,
            text_converter=self.data_pipeline.text_converter or "",
            text_converter_config=self.data_pipeline.text_conversion_config,
            augment=self.data_pipeline.enable_text_augmentation,
            augmentation_prob=self.data_pipeline.augmentation_probability,
            enable_cache=self.data_pipeline.enable_caching,
            cache_dir=self.data_pipeline.cache_dir,
            cache_tokenized=self.data_pipeline.cache_tokenized,
            cache_converted_text=self.data_pipeline.cache_converted_text,
            optimization_profile=self.data_pipeline.optimization_profile,
            prefetch_size=self.data_pipeline.prefetch_size,
            num_workers=self.data_pipeline.num_workers,
            buffer_size=self.data_pipeline.buffer_size,
        )
        
        # Create experiment config
        return ExperimentConfig(
            name=self.experiment_name,
            description=self.description,
            tags=self.tags,
            dataset=dataset_config,
            dataloader=dataloader_config,
            output_dir=self.output_dir,
            save_preprocessed=self.save_preprocessed,
            competition_name=self.data_pipeline.competition_name,
            competition_type=self.data_pipeline.competition_type,
        )
    
    def to_training_config(self) -> TrainingConfig:
        """Extract the TrainingConfig for use with the trainer."""
        # Update training config with current values
        self.training.experiment_name = self.experiment_name
        self.training.output_dir = self.output_dir
        self.training.max_length = self.model.max_length
        self.training.model_name = self.model.model_name
        self.training.model_type = self.model.model_type
        self.training.num_labels = self.model.num_labels
        
        return self.training
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        def convert_value(value):
            if hasattr(value, "to_dict"):
                return value.to_dict()
            elif isinstance(value, Enum):
                return value.value
            elif hasattr(value, "__dict__"):
                return {k: convert_value(v) for k, v in value.__dict__.items()}
            elif isinstance(value, (list, tuple)):
                return [convert_value(item) for item in value]
            else:
                return value
        
        return {
            key: convert_value(value)
            for key, value in self.__dict__.items()
            if not key.startswith("_")
        }
    
    def save(self, config_path: Union[str, Path]) -> None:
        """Save unified configuration to file."""
        config_path = Path(config_path)
        config_dict = self.to_dict()
        
        with open(config_path, "w") as f:
            json.dump(config_dict, f, indent=2, default=str)
    
    @classmethod
    def load(cls, config_path: Union[str, Path]) -> "UnifiedConfig":
        """Load unified configuration from file."""
        config_path = Path(config_path)
        
        with open(config_path) as f:
            config_dict = json.load(f)
        
        return cls.from_dict(config_dict)
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "UnifiedConfig":
        """Create UnifiedConfig from dictionary."""
        # Reconstruct sub-configurations
        training_dict = config_dict.get("training", {})
        if training_dict:
            config_dict["training"] = TrainingConfig.from_dict(training_dict)
        
        data_pipeline_dict = config_dict.get("data_pipeline", {})
        if data_pipeline_dict:
            config_dict["data_pipeline"] = DataPipelineConfig(**data_pipeline_dict)
        
        model_dict = config_dict.get("model", {})
        if model_dict:
            config_dict["model"] = ModelConfig(**model_dict)
        
        return cls(**config_dict)
    
    @classmethod
    def from_training_config(cls, training_config: TrainingConfig) -> "UnifiedConfig":
        """Create UnifiedConfig from existing TrainingConfig."""
        unified = cls()
        unified.training = training_config
        
        # Sync fields
        unified.experiment_name = training_config.experiment_name or "bert_kaggle_experiment"
        unified.output_dir = training_config.output_dir
        unified.model.model_name = training_config.model_name
        unified.model.model_type = training_config.model_type
        unified.model.max_length = training_config.max_length
        unified.model.num_labels = training_config.num_labels
        
        return unified
    
    @classmethod
    def from_experiment_config(cls, experiment_config: ExperimentConfig) -> "UnifiedConfig":
        """Create UnifiedConfig from modular dataloader's ExperimentConfig."""
        unified = cls()
        
        # Set experiment metadata
        unified.experiment_name = experiment_config.name
        unified.description = experiment_config.description
        unified.tags = experiment_config.tags
        unified.output_dir = experiment_config.output_dir
        
        # Set data pipeline config
        unified.data_pipeline.competition_name = getattr(experiment_config, 'competition_name', None)
        unified.data_pipeline.competition_type = getattr(experiment_config, 'competition_type', 'classification')
        unified.data_pipeline.text_converter = experiment_config.dataloader.text_converter
        unified.data_pipeline.text_conversion_config = experiment_config.dataloader.text_converter_config
        unified.data_pipeline.enable_text_augmentation = experiment_config.dataloader.augment
        unified.data_pipeline.augmentation_probability = experiment_config.dataloader.augmentation_prob
        unified.data_pipeline.enable_caching = experiment_config.dataloader.enable_cache
        unified.data_pipeline.cache_dir = experiment_config.dataloader.cache_dir
        unified.data_pipeline.train_split = experiment_config.dataset.train_split
        unified.data_pipeline.val_split = experiment_config.dataset.val_split
        unified.data_pipeline.test_split = experiment_config.dataset.test_split
        unified.data_pipeline.stratify = experiment_config.dataset.stratify
        
        # Set model config
        unified.model.model_name = experiment_config.dataloader.tokenizer_name
        unified.model.max_length = experiment_config.dataloader.max_length
        unified.model.tokenizer_backend = experiment_config.dataloader.tokenizer_backend
        
        # Update training config
        unified.training.batch_size = experiment_config.dataloader.batch_size
        unified.training.max_length = experiment_config.dataloader.max_length
        unified.training.model_name = experiment_config.dataloader.tokenizer_name
        unified.training.train_path = experiment_config.dataset.data_path
        unified.training.target_column = experiment_config.dataset.label_column
        
        return unified
    
    @classmethod
    def for_competition(
        cls,
        competition: str,
        data_path: str,
        **overrides
    ) -> "UnifiedConfig":
        """Create optimized configuration for a specific Kaggle competition."""
        config = cls()
        
        # Set competition basics
        config.data_pipeline.competition_name = competition
        config.data_pipeline.auto_detect_competition = True
        config.training.train_path = data_path
        config.experiment_name = f"{competition}_experiment"
        
        # Apply overrides
        for key, value in overrides.items():
            if "." in key:
                # Handle nested keys like "training.batch_size"
                section, field = key.split(".", 1)
                if hasattr(config, section):
                    section_obj = getattr(config, section)
                    if hasattr(section_obj, field):
                        setattr(section_obj, field, value)
            else:
                if hasattr(config, key):
                    setattr(config, key, value)
        
        # Trigger post-init processing
        config.__post_init__()
        
        return config
    
    def create_modular_dataloader(self, split: str = "train"):
        """Create a ModularMLXDataLoader instance from this configuration."""
        from data.modular_mlx_dataloader import ModularMLXDataLoader
        
        experiment_config = self.to_experiment_config()
        return ModularMLXDataLoader(experiment_config, split=split)
    
    def get_optimization_recommendations(self) -> Dict[str, Any]:
        """Get optimization recommendations based on current configuration."""
        recommendations = {}
        
        # Memory optimization
        if self.training.batch_size > 64:
            recommendations["memory"] = "Consider reducing batch size or enabling gradient accumulation"
        
        # Performance optimization 
        if self.model.max_length > 512:
            recommendations["performance"] = "Long sequences may impact training speed"
        
        # Caching recommendations
        if not self.data_pipeline.enable_caching:
            recommendations["caching"] = "Enable caching for faster data loading in subsequent runs"
        
        return recommendations


# Factory functions for easy configuration creation
def create_competition_config(
    competition: str,
    data_path: str,
    **kwargs
) -> UnifiedConfig:
    """Create a competition-optimized configuration."""
    return UnifiedConfig.for_competition(competition, data_path, **kwargs)


def create_development_config(
    data_path: str,
    **kwargs
) -> UnifiedConfig:
    """Create a development-optimized configuration."""
    config = UnifiedConfig()
    config.training.optimization_level = OptimizationLevel.DEVELOPMENT
    config.training.train_path = data_path
    config.training.epochs = 1
    config.training.batch_size = 8
    config.data_pipeline.optimization_profile = "debug"
    config.experiment_name = "development_test"
    
    # Apply overrides
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)
    
    return config


def create_production_config(
    competition: str,
    data_path: str,
    **kwargs
) -> UnifiedConfig:
    """Create a production-optimized configuration."""
    config = UnifiedConfig.for_competition(competition, data_path)
    config.training.optimization_level = OptimizationLevel.PRODUCTION
    config.training.epochs = 10
    config.data_pipeline.optimization_profile = "speed"
    config.data_pipeline.enable_caching = True
    config.model.use_mlx_embeddings = True  # Use native MLX for best performance
    
    # Apply overrides
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)
    
    return config