"""Data factory adapter implementing domain ports.

This module provides the infrastructure implementation of data factory services,
bridging between the domain layer and the concrete data implementations.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional

from loguru import logger

from infrastructure.di import factory
from application.data.interfaces import DatasetFactory as DatasetFactoryPort
from application.data.data_models import DataConfig, DatasetType
from ..tokenizer.mlx.tokenizer_adapter import MLXTokenizer
from .loaders.mlx_loader import MLXDataLoader, MLXLoaderConfig

# Import the original factory functions for backward compatibility
import sys
sys.path.append(str(Path(__file__).parent.parent.parent.parent / "data"))
try:
    from data.factory import (
        create_dataset,
        create_dataloader, 
        create_data_pipeline,
        CSVDataset
    )
except ImportError:
    logger.warning("Could not import original data factory functions")
    # Provide stub functions for now
    def create_dataset(*args, **kwargs):
        raise NotImplementedError("Original create_dataset not available")
    
    def create_dataloader(*args, **kwargs):
        raise NotImplementedError("Original create_dataloader not available")
    
    def create_data_pipeline(*args, **kwargs):
        raise NotImplementedError("Original create_data_pipeline not available")
    
    class CSVDataset:
        def __init__(self, *args, **kwargs):
            raise NotImplementedError("Original CSVDataset not available")


@factory(Any)  # Produces various dataset types
class DatasetFactory(DatasetFactoryPort):
    """Concrete implementation of dataset factory port."""
    
    def create_text_dataset(
        self,
        data_path: Path,
        config: DataConfig,
        dataset_type: DatasetType = DatasetType.TRAIN
    ) -> Any:
        """Create a text dataset."""
        logger.info(f"Creating text dataset from {data_path}")
        
        return create_dataset(
            data_path=data_path,
            dataset_type="csv",
            text_column=config.text_column,
            label_column=config.label_column,
            split=dataset_type.value,
        )
    
    def create_tabular_dataset(
        self,
        data_path: Path,
        config: DataConfig,
        dataset_type: DatasetType = DatasetType.TRAIN
    ) -> Any:
        """Create a tabular dataset."""
        logger.info(f"Creating tabular dataset from {data_path}")
        
        # For tabular data, we'll use the augmentation system to convert to text
        return create_dataset(
            data_path=data_path,
            dataset_type="csv",
            text_column=None,  # Will be converted from tabular
            label_column=config.label_column,
            split=dataset_type.value,
        )
    
    def create_tokenized_dataset(
        self,
        dataset: Any,
        tokenizer: Any,
        config: DataConfig
    ) -> Any:
        """Create a tokenized dataset from raw dataset."""
        logger.info("Creating tokenized dataset")
        
        # This would typically involve pre-tokenization and caching
        # For now, return the original dataset with tokenizer attached
        dataset.tokenizer = tokenizer
        return dataset


@factory(MLXDataLoader)
class MLXDataLoaderFactory:
    """Factory for creating MLX data loaders."
    
    def create_dataloader(
        self,
        dataset: Any,
        batch_size: int = 32,
        shuffle: bool = True,
        **kwargs
    ) -> MLXDataLoader:
        """Create an MLX data loader."""
        logger.info(f"Creating MLX dataloader with batch_size={batch_size}")
        
        # Create loader config
        config = MLXLoaderConfig(
            batch_size=batch_size,
            shuffle=shuffle,
            **{k: v for k, v in kwargs.items() if hasattr(MLXLoaderConfig, k)}
        )
        
        # Create tokenizer if needed
        tokenizer = kwargs.get('tokenizer')
        if tokenizer is None:
            tokenizer = MLXTokenizer()
        
        return MLXDataLoader(
            dataset=dataset,
            config=config,
            tokenizer=tokenizer
        )


@factory(dict)
class DataPipelineFactory:
    """Factory for creating complete data pipelines."""
    
    def create_pipeline(
        self,
        train_path: Path,
        val_path: Optional[Path] = None,
        test_path: Optional[Path] = None,
        config: DataConfig = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Create a complete data pipeline."""
        logger.info("Creating data pipeline")
        
        if config is None:
            # Create default config
            config = DataConfig(
                task_type=kwargs.get('task_type', 'text_classification'),
                batch_size=kwargs.get('batch_size', 32),
            )
        
        return create_data_pipeline(
            train_path=train_path,
            val_path=val_path,
            test_path=test_path,
            batch_size=config.batch_size,
            **kwargs
        )


# Backward compatibility - export the original factory class
# This allows existing code to continue working
__all__ = [
    "DatasetFactory",
    "MLXDataLoaderFactory", 
    "DataPipelineFactory",
    "create_dataset",
    "create_dataloader",
    "create_data_pipeline",
    "CSVDataset",
]