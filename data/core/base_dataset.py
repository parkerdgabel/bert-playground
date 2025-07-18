"""
Base dataset implementation for MLX dataloader system.
Provides common functionality for all dataset types.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Any, List, Optional, Union, Callable
import pandas as pd
import numpy as np
from loguru import logger
import hashlib
import json

from .interfaces import Dataset, Transform, TextConverter, DataSpec


class BaseDataset(ABC):
    """
    Abstract base class for all datasets.
    Provides common functionality like caching, transforms, and metadata.
    """
    
    def __init__(
        self,
        data_path: Union[str, Path],
        spec: Optional[DataSpec] = None,
        transforms: Optional[List[Transform]] = None,
        text_converter: Optional[TextConverter] = None,
        cache_dir: Optional[Union[str, Path]] = None,
        **kwargs
    ):
        """
        Initialize base dataset.
        
        Args:
            data_path: Path to the data file
            spec: Dataset specification
            transforms: List of transforms to apply
            text_converter: Text converter for tabular data
            cache_dir: Directory for caching processed data
            **kwargs: Additional arguments for subclasses
        """
        self.data_path = Path(data_path)
        self.spec = spec
        self.transforms = transforms or []
        self.text_converter = text_converter
        self.cache_dir = Path(cache_dir) if cache_dir else None
        
        # Initialize data storage
        self._data: Optional[pd.DataFrame] = None
        self._metadata: Dict[str, Any] = {}
        self._cache: Dict[str, Any] = {}
        
        # Load and process data
        self._load_data(**kwargs)
        self._compute_metadata()
    
    @abstractmethod
    def _load_data(self, **kwargs) -> None:
        """Load data from file. Must be implemented by subclasses."""
        pass
    
    def _compute_metadata(self) -> None:
        """Compute dataset metadata."""
        if self._data is None:
            return
        
        self._metadata = {
            "num_samples": len(self._data),
            "columns": list(self._data.columns),
            "dtypes": {col: str(dtype) for col, dtype in self._data.dtypes.items()},
            "shape": self._data.shape,
            "memory_usage": self._data.memory_usage(deep=True).sum(),
        }
        
        # Add spec information if available
        if self.spec:
            self._metadata["spec"] = self.spec.to_dict()
        
        # Compute basic statistics for numerical columns
        numeric_cols = self._data.select_dtypes(include=[np.number]).columns.tolist()
        if numeric_cols:
            self._metadata["numeric_stats"] = self._data[numeric_cols].describe().to_dict()
        
        # Count unique values for categorical columns
        categorical_cols = self._data.select_dtypes(include=['object', 'category']).columns.tolist()
        if categorical_cols:
            self._metadata["categorical_stats"] = {
                col: self._data[col].nunique() for col in categorical_cols
            }
        
        logger.debug(f"Computed metadata for dataset: {self._metadata['num_samples']} samples")
    
    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self._data) if self._data is not None else 0
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get a single item from the dataset.
        
        Args:
            idx: Index of the item
            
        Returns:
            Dictionary containing the processed sample
        """
        if self._data is None:
            raise RuntimeError("Dataset not loaded")
        
        if idx < 0 or idx >= len(self._data):
            raise IndexError(f"Index {idx} out of range for dataset of size {len(self._data)}")
        
        # Get raw sample
        sample = self._data.iloc[idx].to_dict()
        
        # Apply text conversion if available
        if self.text_converter:
            sample = self._convert_to_text(sample)
        
        # Apply transforms
        for transform in self.transforms:
            sample = transform(sample)
        
        return sample
    
    def _convert_to_text(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """Convert sample to text representation."""
        if not self.text_converter:
            return sample
        
        # Check cache first
        cache_key = self._get_cache_key(sample)
        if cache_key in self._cache:
            sample["text"] = self._cache[cache_key]
            return sample
        
        # Convert to text
        text = self.text_converter.convert(sample)
        sample["text"] = text
        
        # Cache the result
        self._cache[cache_key] = text
        
        return sample
    
    def _get_cache_key(self, sample: Dict[str, Any]) -> str:
        """Generate cache key for a sample."""
        # Create a deterministic string representation
        key_data = json.dumps(sample, sort_keys=True)
        return hashlib.md5(key_data.encode()).hexdigest()
    
    def get_metadata(self) -> Dict[str, Any]:
        """Return dataset metadata."""
        return self._metadata.copy()
    
    def add_transform(self, transform: Transform) -> None:
        """Add a transform to the pipeline."""
        self.transforms.append(transform)
    
    def set_text_converter(self, converter: TextConverter) -> None:
        """Set the text converter."""
        self.text_converter = converter
        # Clear text cache when converter changes
        self._cache.clear()
    
    def clear_cache(self) -> None:
        """Clear all cached data."""
        self._cache.clear()
    
    def get_batch(self, indices: List[int]) -> List[Dict[str, Any]]:
        """
        Get multiple items efficiently.
        
        Args:
            indices: List of indices
            
        Returns:
            List of processed samples
        """
        return [self[idx] for idx in indices]
    
    def split(
        self,
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        test_ratio: float = 0.1,
        stratify_column: Optional[str] = None,
        random_state: int = 42,
    ) -> Dict[str, "BaseDataset"]:
        """
        Split dataset into train/val/test sets.
        
        Args:
            train_ratio: Ratio for training set
            val_ratio: Ratio for validation set
            test_ratio: Ratio for test set
            stratify_column: Column to use for stratified split
            random_state: Random seed
            
        Returns:
            Dictionary with 'train', 'val', 'test' datasets
        """
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1"
        
        from sklearn.model_selection import train_test_split
        
        indices = np.arange(len(self._data))
        
        # Stratify if column is provided
        stratify = None
        if stratify_column and stratify_column in self._data.columns:
            stratify = self._data[stratify_column].values
        
        # First split: train+val vs test
        train_val_idx, test_idx = train_test_split(
            indices,
            test_size=test_ratio,
            stratify=stratify,
            random_state=random_state,
        )
        
        # Second split: train vs val
        if val_ratio > 0:
            val_size = val_ratio / (train_ratio + val_ratio)
            stratify_train_val = stratify[train_val_idx] if stratify is not None else None
            
            train_idx, val_idx = train_test_split(
                train_val_idx,
                test_size=val_size,
                stratify=stratify_train_val,
                random_state=random_state,
            )
        else:
            train_idx = train_val_idx
            val_idx = np.array([], dtype=int)
        
        # Create subset datasets
        splits = {}
        
        if len(train_idx) > 0:
            splits["train"] = self._create_subset(train_idx)
        
        if len(val_idx) > 0:
            splits["val"] = self._create_subset(val_idx)
        
        if len(test_idx) > 0:
            splits["test"] = self._create_subset(test_idx)
        
        return splits
    
    def _create_subset(self, indices: np.ndarray) -> "BaseDataset":
        """Create a subset of the dataset."""
        subset = self.__class__.__new__(self.__class__)
        
        # Copy attributes
        subset.data_path = self.data_path
        subset.spec = self.spec
        subset.transforms = self.transforms.copy()
        subset.text_converter = self.text_converter
        subset.cache_dir = self.cache_dir
        
        # Create subset data
        subset._data = self._data.iloc[indices].reset_index(drop=True)
        subset._metadata = {}
        subset._cache = {}
        
        # Recompute metadata
        subset._compute_metadata()
        
        return subset
    
    def filter(self, condition: Callable[[pd.Series], bool]) -> "BaseDataset":
        """
        Filter dataset based on a condition.
        
        Args:
            condition: Function that takes a row and returns True to keep it
            
        Returns:
            Filtered dataset
        """
        mask = self._data.apply(condition, axis=1)
        indices = np.where(mask)[0]
        return self._create_subset(indices)
    
    def save_processed(self, path: Union[str, Path]) -> None:
        """Save processed dataset to file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save data
        data_path = path.with_suffix(".parquet")
        self._data.to_parquet(data_path)
        
        # Save metadata
        metadata_path = path.with_suffix(".json")
        with open(metadata_path, "w") as f:
            json.dump(self.get_metadata(), f, indent=2)
        
        logger.info(f"Saved processed dataset to {path}")
    
    @classmethod
    def load_processed(
        cls,
        path: Union[str, Path],
        **kwargs
    ) -> "BaseDataset":
        """Load a previously saved processed dataset."""
        path = Path(path)
        
        # Load data
        data_path = path.with_suffix(".parquet")
        data = pd.read_parquet(data_path)
        
        # Create dataset instance
        dataset = cls.__new__(cls)
        dataset.data_path = path
        dataset.spec = None
        dataset.transforms = []
        dataset.text_converter = None
        dataset.cache_dir = None
        dataset._data = data
        dataset._cache = {}
        
        # Load metadata
        metadata_path = path.with_suffix(".json")
        if metadata_path.exists():
            with open(metadata_path, "r") as f:
                dataset._metadata = json.load(f)
                
                # Restore spec if available
                if "spec" in dataset._metadata:
                    dataset.spec = DataSpec.from_dict(dataset._metadata["spec"])
        else:
            dataset._compute_metadata()
        
        return dataset