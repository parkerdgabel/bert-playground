"""Data loading and processing modules for MLX-based Kaggle competitions."""

# Core MLX DataLoader
from .mlx_dataloader import (
    KaggleDataLoader,
    create_kaggle_dataloader,
    create_titanic_dataloader,
)

# Dataset specifications and registry
from .datasets import (
    DatasetSpec,
    ProblemType,
    dataset_registry,
    get_dataset_spec,
    register_dataset,
    list_datasets,
)

# Text generation utilities
from .text_generation import (
    TextGenerator,
    TabularTextGenerator,
    TitanicTextGenerator,
    get_text_generator,
)

__all__ = [
    # Core DataLoader
    "KaggleDataLoader",
    "create_kaggle_dataloader",
    "create_titanic_dataloader",
    # Dataset management
    "DatasetSpec",
    "ProblemType",
    "dataset_registry",
    "get_dataset_spec",
    "register_dataset",
    "list_datasets",
    # Text generation
    "TextGenerator",
    "TabularTextGenerator", 
    "TitanicTextGenerator",
    "get_text_generator",
]