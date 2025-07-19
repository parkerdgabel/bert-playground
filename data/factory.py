"""Factory functions for creating datasets and data loaders.

This module provides convenient factory functions to create datasets and data loaders
for the training pipeline, following the same pattern as the models and training factories.
"""

from pathlib import Path
from typing import Dict, Optional, Union, Any
import pandas as pd
from loguru import logger

from .core import (
    CompetitionType,
    DatasetSpec,
    KaggleDataset,
)
from .loaders import MLXDataLoader, MLXLoaderConfig
from .templates import TextTemplateEngine


class CSVDataset(KaggleDataset):
    """Simple CSV dataset implementation for Kaggle competitions."""
    
    def __init__(
        self,
        csv_path: Union[str, Path],
        spec: Optional[DatasetSpec] = None,
        text_column: Optional[str] = None,
        label_column: Optional[str] = None,
        text_converter: Optional[Any] = None,
        **kwargs
    ):
        """Initialize CSV dataset.
        
        Args:
            csv_path: Path to CSV file
            spec: Dataset specification
            text_column: Column containing text data
            label_column: Column containing labels
            text_converter: Optional text converter for tabular data
            **kwargs: Additional arguments for parent class
        """
        self.csv_path = Path(csv_path)
        self.text_column = text_column
        self.label_column = label_column
        self.text_converter = text_converter
        
        # Create default spec if not provided
        if spec is None:
            spec = self._create_default_spec()
            
        super().__init__(spec, **kwargs)
    
    def _create_default_spec(self) -> DatasetSpec:
        """Create a default dataset specification."""
        # Try to infer from filename
        filename = self.csv_path.stem.lower()
        
        if "titanic" in filename:
            competition_type = CompetitionType.BINARY_CLASSIFICATION
            num_classes = 2
            competition_name = "titanic"
        elif "train" in filename or "test" in filename:
            competition_type = CompetitionType.UNKNOWN
            num_classes = None
            competition_name = "unknown"
        else:
            competition_type = CompetitionType.UNKNOWN
            num_classes = None
            competition_name = filename
            
        return DatasetSpec(
            competition_name=competition_name,
            dataset_path=self.csv_path,
            competition_type=competition_type,
            num_samples=0,  # Will be updated in _load_data
            num_features=0,  # Will be updated in _load_data
            target_column=self.label_column,
            num_classes=num_classes,
        )
    
    def _load_data(self) -> None:
        """Load data from CSV file."""
        if not self.csv_path.exists():
            raise FileNotFoundError(f"CSV file not found: {self.csv_path}")
            
        # Load data
        self._data = pd.read_csv(self.csv_path)
        
        # Update spec with actual data info
        self.spec.num_samples = len(self._data)
        self.spec.num_features = len(self._data.columns)
        
        # Identify columns if not specified
        if self.text_column is None:
            # Look for text-like columns
            text_candidates = []
            for col in self._data.columns:
                if self._data[col].dtype == 'object':
                    avg_length = self._data[col].astype(str).str.len().mean()
                    if avg_length > 20:  # Likely text column
                        text_candidates.append(col)
            
            if text_candidates:
                self.text_column = text_candidates[0]
                logger.info(f"Auto-detected text column: {self.text_column}")
        
        # Identify label column if not specified
        if self.label_column is None and self.split == "train":
            # Common label column names
            label_candidates = ['label', 'labels', 'target', 'class', 'Survived', 
                              'Label', 'Target', 'Class', 'y']
            for col in label_candidates:
                if col in self._data.columns:
                    self.label_column = col
                    self.spec.target_column = col
                    logger.info(f"Auto-detected label column: {self.label_column}")
                    break
    
    def _validate_data(self) -> None:
        """Validate the loaded data."""
        if self._data is None:
            raise ValueError("Data not loaded")
            
        # Check if required columns exist
        if self.text_column and self.text_column not in self._data.columns:
            # If no text column, we'll convert the entire row to text
            logger.warning(f"Text column '{self.text_column}' not found. Will use row-to-text conversion.")
            self.text_column = None
            
        if self.label_column and self.label_column not in self._data.columns:
            if self.split == "train":
                raise ValueError(f"Label column '{self.label_column}' not found in data")
    
    def __getitem__(self, index: int) -> Dict[str, Any]:
        """Get a single sample by index."""
        if index < 0 or index >= len(self._data):
            raise IndexError(f"Index {index} out of range for dataset of size {len(self._data)}")
            
        # Get row data
        row = self._data.iloc[index]
        sample = {}
        
        # Handle text data
        if self.text_column and self.text_column in row:
            sample['text'] = str(row[self.text_column])
        else:
            # Convert entire row to text if no text column
            if self.text_converter:
                sample['text'] = self.text_converter.convert(row.to_dict())
            else:
                # Simple conversion: key-value pairs
                text_parts = []
                for col, val in row.items():
                    if col != self.label_column and pd.notna(val):
                        text_parts.append(f"{col}: {val}")
                sample['text'] = " | ".join(text_parts)
        
        # Handle labels
        if self.label_column and self.label_column in row:
            label_value = row[self.label_column]
            if pd.notna(label_value):
                sample['labels'] = int(label_value) if self.spec.competition_type in [
                    CompetitionType.BINARY_CLASSIFICATION,
                    CompetitionType.MULTICLASS_CLASSIFICATION
                ] else float(label_value)
        
        # Add metadata
        sample['metadata'] = {
            'index': index,
            'split': self.split,
        }
        
        # Add ID if available
        id_columns = ['id', 'Id', 'ID', 'PassengerId', 'test_id']
        for col in id_columns:
            if col in row:
                sample['metadata']['id'] = row[col]
                break
                
        return sample


def create_dataset(
    data_path: Union[str, Path],
    dataset_type: str = "csv",
    competition_name: Optional[str] = None,
    text_column: Optional[str] = None,
    label_column: Optional[str] = None,
    split: str = "train",
    **kwargs
) -> KaggleDataset:
    """Create a dataset instance.
    
    Args:
        data_path: Path to data file
        dataset_type: Type of dataset (currently only "csv" supported)
        competition_name: Name of the competition
        text_column: Column containing text data
        label_column: Column containing labels
        split: Dataset split ("train", "val", "test")
        **kwargs: Additional arguments for dataset
        
    Returns:
        Dataset instance
    """
    data_path = Path(data_path)
    
    if dataset_type == "csv":
        # Create text converter if needed
        text_converter = None
        if text_column is None:
            from .templates import TextTemplateEngine
            text_converter = TextTemplateEngine()
            
        dataset = CSVDataset(
            csv_path=data_path,
            text_column=text_column,
            label_column=label_column,
            text_converter=text_converter,
            split=split,
            **kwargs
        )
        
        # Update competition name if provided
        if competition_name:
            dataset.spec.competition_name = competition_name
            
        return dataset
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")


def create_dataloader(
    dataset: Optional[KaggleDataset] = None,
    data_path: Optional[Union[str, Path]] = None,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 4,
    prefetch_size: int = 4,
    tokenizer=None,
    max_length: int = 512,
    **kwargs
) -> MLXDataLoader:
    """Create a data loader instance.
    
    Args:
        dataset: Dataset instance (if not provided, will create from data_path)
        data_path: Path to data file (used if dataset not provided)
        batch_size: Batch size
        shuffle: Whether to shuffle data
        num_workers: Number of worker threads
        prefetch_size: Number of batches to prefetch
        tokenizer: Tokenizer for text processing
        max_length: Maximum sequence length
        **kwargs: Additional arguments
        
    Returns:
        DataLoader instance
    """
    # Create dataset if not provided
    if dataset is None:
        if data_path is None:
            raise ValueError("Either dataset or data_path must be provided")
        dataset = create_dataset(data_path, **kwargs)
    
    # Create loader config
    config = MLXLoaderConfig(
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        prefetch_size=prefetch_size,
        max_length=max_length,
        **{k: v for k, v in kwargs.items() if hasattr(MLXLoaderConfig, k)}
    )
    
    # Create loader
    loader = MLXDataLoader(
        dataset=dataset,
        config=config,
        tokenizer=tokenizer,
    )
    
    return loader


def create_data_pipeline(
    train_path: Union[str, Path],
    val_path: Optional[Union[str, Path]] = None,
    test_path: Optional[Union[str, Path]] = None,
    competition_name: Optional[str] = None,
    batch_size: int = 32,
    eval_batch_size: Optional[int] = None,
    tokenizer=None,
    **kwargs
) -> Dict[str, MLXDataLoader]:
    """Create a complete data pipeline with train/val/test loaders.
    
    Args:
        train_path: Path to training data
        val_path: Path to validation data (optional)
        test_path: Path to test data (optional)
        competition_name: Name of the competition
        batch_size: Training batch size
        eval_batch_size: Evaluation batch size (defaults to 2*batch_size)
        tokenizer: Tokenizer for text processing
        **kwargs: Additional arguments
        
    Returns:
        Dictionary with 'train', 'val', 'test' loaders (as available)
    """
    if eval_batch_size is None:
        eval_batch_size = batch_size * 2
        
    loaders = {}
    
    # Create train loader
    loaders['train'] = create_dataloader(
        data_path=train_path,
        competition_name=competition_name,
        batch_size=batch_size,
        shuffle=True,
        split="train",
        tokenizer=tokenizer,
        **kwargs
    )
    
    # Create validation loader if path provided
    if val_path:
        loaders['val'] = create_dataloader(
            data_path=val_path,
            competition_name=competition_name,
            batch_size=eval_batch_size,
            shuffle=False,
            split="val",
            tokenizer=tokenizer,
            **kwargs
        )
    
    # Create test loader if path provided
    if test_path:
        loaders['test'] = create_dataloader(
            data_path=test_path,
            competition_name=competition_name,
            batch_size=eval_batch_size,
            shuffle=False,
            split="test",
            tokenizer=tokenizer,
            **kwargs
        )
    
    logger.info(f"Created data pipeline with loaders: {list(loaders.keys())}")
    
    return loaders