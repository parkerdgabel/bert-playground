"""Configuration fixtures for data module testing."""

from pathlib import Path
from typing import Dict, Any, Optional, List
import json

from data.loaders.mlx_loader import MLXLoaderConfig
from data.core.metadata import CompetitionMetadata
from data.core.base import DatasetSpec, CompetitionType
from data.templates.engine import TemplateConfig


def create_loader_config(
    batch_size: int = 32,
    shuffle: bool = True,
    drop_last: bool = False,
    num_workers: int = 0,
    prefetch_size: int = 2,
    **kwargs
) -> MLXLoaderConfig:
    """Create data loader configuration with sensible defaults."""
    config_dict = {
        "batch_size": batch_size,
        "shuffle": shuffle,
        "drop_last": drop_last,
        "num_workers": num_workers,
        "prefetch_size": prefetch_size,
        "seed": kwargs.get("seed", 42),
        "pin_memory": kwargs.get("pin_memory", False),
        "persistent_workers": kwargs.get("persistent_workers", False),
    }
    
    # Add any additional kwargs
    for key, value in kwargs.items():
        if key not in config_dict:
            config_dict[key] = value
    
    return MLXLoaderConfig(**config_dict)


def create_streaming_config(
    batch_size: int = 32,
    buffer_size: int = 1000,
    prefetch: int = 4,
    shuffle: bool = True,
    **kwargs
) -> Dict[str, Any]:
    """Create streaming data configuration."""
    return {
        "batch_size": batch_size,
        "buffer_size": buffer_size,
        "prefetch": prefetch,
        "shuffle": shuffle,
        "seed": kwargs.get("seed", 42),
        "drop_incomplete": kwargs.get("drop_incomplete", False),
        "num_parallel_calls": kwargs.get("num_parallel_calls", 4),
        "deterministic": kwargs.get("deterministic", True),
    }


def create_memory_config(
    memory_limit: int = 1_000_000_000,  # 1GB
    cache_size: int = 100,
    batch_size: int = 16,
    **kwargs
) -> Dict[str, Any]:
    """Create memory-optimized configuration."""
    return {
        "batch_size": batch_size,
        "memory_limit": memory_limit,
        "cache_size": cache_size,
        "pin_memory": kwargs.get("pin_memory", False),
        "num_workers": kwargs.get("num_workers", 0),  # Single process for memory control
        "prefetch_factor": kwargs.get("prefetch_factor", 2),
        "persistent_workers": False,  # Disable to save memory
        "drop_last": kwargs.get("drop_last", True),  # Drop incomplete batches
    }


def create_small_dataset_spec(**kwargs) -> DatasetSpec:
    """Create small dataset spec for quick testing."""
    defaults = {
        "competition_name": "test_small",
        "dataset_path": "./test_data",
        "competition_type": CompetitionType.BINARY_CLASSIFICATION,
        "num_samples": 100,
        "num_features": 5,
        "num_classes": 2,
        "target_column": "target",
        "recommended_batch_size": 8,
        "num_workers": 0,  # No multiprocessing for tests
    }
    defaults.update(kwargs)
    return DatasetSpec(**defaults)


def create_dataset_spec(
    name: str = "test_dataset",
    task_type: str = "classification",
    num_samples: int = 1000,
    num_features: int = 10,
    **kwargs
) -> DatasetSpec:
    """Create dataset specification."""
    # Map task type string to CompetitionType enum
    competition_type_map = {
        "classification": CompetitionType.BINARY_CLASSIFICATION,
        "binary_classification": CompetitionType.BINARY_CLASSIFICATION,
        "multiclass_classification": CompetitionType.MULTICLASS_CLASSIFICATION,
        "regression": CompetitionType.REGRESSION,
        "time_series": CompetitionType.TIME_SERIES,
    }
    competition_type = competition_type_map.get(task_type, CompetitionType.UNKNOWN)
    
    # Create feature columns - use provided ones or generate defaults
    if "numerical_columns" in kwargs:
        numerical_columns = kwargs["numerical_columns"]
    else:
        num_numeric = kwargs.get("num_numeric_features", num_features)
        numerical_columns = [f"numeric_{i}" for i in range(num_numeric)]
    
    if "categorical_columns" in kwargs:
        categorical_columns = kwargs["categorical_columns"]
    else:
        num_categorical = kwargs.get("num_categorical_features", 0)
        categorical_columns = [f"categorical_{i}" for i in range(num_categorical)]
    
    if "text_columns" in kwargs:
        text_columns = kwargs["text_columns"]
    else:
        num_text = kwargs.get("num_text_features", 0)
        text_columns = [f"text_{i}" for i in range(num_text)]
    
    spec = DatasetSpec(
        competition_name=name,
        dataset_path=kwargs.get("dataset_path", Path("/tmp/test_data")),
        competition_type=competition_type,
        num_samples=num_samples,
        num_features=num_features,
        target_column=kwargs.get("target_column", "target"),
        text_columns=text_columns,
        categorical_columns=categorical_columns,
        numerical_columns=numerical_columns,
        num_classes=kwargs.get("num_classes", 2 if "classification" in task_type else None),
    )
    
    return spec


def create_kaggle_config(
    competition_name: str = "titanic",
    data_dir: Optional[Path] = None,
    **kwargs
) -> Dict[str, Any]:
    """Create Kaggle-specific configuration."""
    if data_dir is None:
        data_dir = Path("/tmp/kaggle_data")
    
    return {
        "competition_name": competition_name,
        "data_dir": str(data_dir),
        "train_file": kwargs.get("train_file", "train.csv"),
        "test_file": kwargs.get("test_file", "test.csv"),
        "submission_file": kwargs.get("submission_file", "sample_submission.csv"),
        "id_column": kwargs.get("id_column", "PassengerId"),
        "target_column": kwargs.get("target_column", "Survived"),
        "task_type": kwargs.get("task_type", "binary_classification"),
        "eval_metric": kwargs.get("eval_metric", "accuracy"),
        "use_cv": kwargs.get("use_cv", True),
        "cv_folds": kwargs.get("cv_folds", 5),
    }


def create_template_config(
    template_type: str = "default",
    max_length: int = 512,
    include_feature_names: bool = True,
    **kwargs
) -> TemplateConfig:
    """Create template engine configuration."""
    return TemplateConfig(
        template_type=template_type,
        max_length=max_length,
        include_feature_names=include_feature_names,
        separator=kwargs.get("separator", ", "),
        missing_value_token=kwargs.get("missing_value_token", "[MISSING]"),
        numeric_precision=kwargs.get("numeric_precision", 2),
        categorical_unknown_token=kwargs.get("categorical_unknown_token", "[UNK]"),
        text_preprocessing=kwargs.get("text_preprocessing", ["lowercase", "strip"]),
    )


def create_augmentation_config(
    augmentation_types: Optional[List[str]] = None,
    augmentation_prob: float = 0.5,
    **kwargs
) -> Dict[str, Any]:
    """Create data augmentation configuration."""
    if augmentation_types is None:
        augmentation_types = ["synonym", "deletion", "swap"]
    
    return {
        "augmentation_types": augmentation_types,
        "augmentation_prob": augmentation_prob,
        "synonym_prob": kwargs.get("synonym_prob", 0.3),
        "deletion_prob": kwargs.get("deletion_prob", 0.2),
        "swap_prob": kwargs.get("swap_prob", 0.2),
        "max_augmentations": kwargs.get("max_augmentations", 3),
        "preserve_length": kwargs.get("preserve_length", True),
        "seed": kwargs.get("seed", 42),
    }


def create_cache_config(
    cache_dir: Optional[Path] = None,
    cache_size_limit: int = 10_000_000_000,  # 10GB
    eviction_policy: str = "lru",
    **kwargs
) -> Dict[str, Any]:
    """Create cache configuration."""
    if cache_dir is None:
        cache_dir = Path("/tmp/data_cache")
    
    return {
        "cache_dir": str(cache_dir),
        "cache_size_limit": cache_size_limit,
        "eviction_policy": eviction_policy,
        "compression": kwargs.get("compression", "gzip"),
        "compression_level": kwargs.get("compression_level", 6),
        "ttl_seconds": kwargs.get("ttl_seconds", 3600),  # 1 hour
        "enable_memory_cache": kwargs.get("enable_memory_cache", True),
        "memory_cache_size": kwargs.get("memory_cache_size", 100),
    }


def create_preprocessing_config(
    normalize_numeric: bool = True,
    encode_categorical: str = "onehot",
    handle_missing: str = "mean",
    **kwargs
) -> Dict[str, Any]:
    """Create preprocessing configuration."""
    return {
        "normalize_numeric": normalize_numeric,
        "normalization_method": kwargs.get("normalization_method", "standard"),
        "encode_categorical": encode_categorical,
        "max_categories": kwargs.get("max_categories", 100),
        "handle_missing": handle_missing,
        "missing_numeric_value": kwargs.get("missing_numeric_value", 0.0),
        "missing_categorical_value": kwargs.get("missing_categorical_value", "missing"),
        "text_vectorizer": kwargs.get("text_vectorizer", "tfidf"),
        "text_max_features": kwargs.get("text_max_features", 10000),
        "remove_outliers": kwargs.get("remove_outliers", False),
        "outlier_method": kwargs.get("outlier_method", "iqr"),
        "outlier_threshold": kwargs.get("outlier_threshold", 3.0),
    }


def create_loader_config_variations() -> Dict[str, MLXLoaderConfig]:
    """Create various loader configuration variations for testing."""
    return {
        "default": create_loader_config(),
        "large_batch": create_loader_config(batch_size=256),
        "small_batch": create_loader_config(batch_size=1),
        "no_shuffle": create_loader_config(shuffle=False),
        "drop_last": create_loader_config(drop_last=True),
        "multi_worker": create_loader_config(num_workers=4),
        "prefetch": create_loader_config(prefetch_size=8),
        "memory_pinned": create_loader_config(pin_memory=True),
    }


def create_invalid_config() -> Dict[str, Any]:
    """Create invalid configuration for error testing."""
    return {
        "batch_size": -1,  # Invalid negative batch size
        "num_workers": -1,  # Invalid negative workers
        "prefetch_size": 0,  # Invalid zero prefetch
        "shuffle": "yes",  # Invalid type (should be bool)
        "drop_last": 1,  # Invalid type (should be bool)
    }


def create_edge_case_configs() -> Dict[str, Any]:
    """Create edge case configurations for testing."""
    return {
        "single_sample_batch": create_loader_config(batch_size=1, drop_last=False),
        "huge_batch": create_loader_config(batch_size=10000),
        "many_workers": create_loader_config(num_workers=32),
        "no_prefetch": create_loader_config(prefetch_size=0),
        "deterministic": create_streaming_config(deterministic=True, shuffle=False),
        "max_memory": create_memory_config(memory_limit=100_000_000_000),  # 100GB
        "min_memory": create_memory_config(memory_limit=1000),  # 1KB
    }


def save_config_to_file(config: Any, file_path: Path, format: str = "json"):
    """Save configuration to file for testing."""
    file_path.parent.mkdir(parents=True, exist_ok=True)
    
    if hasattr(config, "to_dict"):
        config_dict = config.to_dict()
    elif hasattr(config, "__dict__"):
        config_dict = config.__dict__
    else:
        config_dict = dict(config)
    
    if format == "json":
        with open(file_path, "w") as f:
            json.dump(config_dict, f, indent=2)
    else:
        raise ValueError(f"Unknown format: {format}")


def create_mlx_loader_config(
    batch_size: int = 32,
    shuffle: bool = True,
    drop_last: bool = False,
    num_workers: int = 0,
    prefetch_size: int = 2,
    pin_memory: bool = False,
    persistent_workers: bool = False,
    seed: int = 42,
    **kwargs
) -> MLXLoaderConfig:
    """Create MLX data loader configuration."""
    return MLXLoaderConfig(
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers,
        prefetch_size=prefetch_size,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        seed=seed,
        **kwargs
    )


def load_config_from_file(file_path: Path, config_class: Optional[type] = None) -> Any:
    """Load configuration from file."""
    with open(file_path, "r") as f:
        config_dict = json.load(f)
    
    if config_class:
        return config_class(**config_dict)
    else:
        return config_dict