"""MLX dataset wrapper for efficient data handling."""

from typing import Any, Dict, Optional, List, Tuple
import json
import pickle
from pathlib import Path
import mlx.core as mx
import numpy as np

from domain.entities.dataset import Dataset, DatasetSplit


class MLXDatasetWrapper:
    """Wrapper for Dataset entities to provide MLX-optimized access."""
    
    def __init__(self, dataset: Dataset):
        """Initialize MLX dataset wrapper.
        
        Args:
            dataset: The domain Dataset entity
        """
        self.dataset = dataset
        self._cache: Dict[int, Any] = {}
        self._pretokenized_path: Optional[Path] = None
        
        # Check for pre-tokenized data
        self._check_pretokenized_data()
    
    @property
    def size(self) -> int:
        """Get dataset size."""
        return self.dataset.size
    
    @property
    def split(self) -> DatasetSplit:
        """Get dataset split."""
        return self.dataset.split
    
    def get_sample(self, idx: int) -> Any:
        """Get a single sample by index.
        
        Args:
            idx: Sample index
            
        Returns:
            Sample data (format depends on dataset)
        """
        # Check cache first
        if idx in self._cache:
            return self._cache[idx]
        
        # Load from pre-tokenized if available
        if self._pretokenized_path is not None:
            sample = self._load_pretokenized_sample(idx)
        else:
            # Load from dataset features
            sample = self._load_raw_sample(idx)
        
        # Cache if small dataset
        if self.size < 10000:
            self._cache[idx] = sample
        
        return sample
    
    def get_batch_samples(self, indices: List[int]) -> List[Any]:
        """Get multiple samples efficiently.
        
        Args:
            indices: List of sample indices
            
        Returns:
            List of samples
        """
        return [self.get_sample(idx) for idx in indices]
    
    def save_pretokenized(self, path: Path) -> None:
        """Save dataset in pre-tokenized format for fast loading.
        
        Args:
            path: Directory to save pre-tokenized data
        """
        path.mkdir(parents=True, exist_ok=True)
        
        # Save metadata
        metadata = {
            "name": self.dataset.name,
            "split": self.dataset.split.value,
            "size": self.dataset.size,
            "features": self.dataset.features,
            "num_classes": self.dataset.num_classes,
        }
        
        with open(path / "metadata.json", "w") as f:
            json.dump(metadata, f)
        
        # Save samples in batches for efficiency
        batch_size = 1000
        for batch_idx in range(0, self.size, batch_size):
            batch_samples = []
            
            for idx in range(batch_idx, min(batch_idx + batch_size, self.size)):
                sample = self.get_sample(idx)
                batch_samples.append(sample)
            
            # Save batch
            batch_path = path / f"batch_{batch_idx // batch_size}.pkl"
            with open(batch_path, "wb") as f:
                pickle.dump(batch_samples, f)
    
    def load_pretokenized(self, path: Path) -> None:
        """Load pre-tokenized data.
        
        Args:
            path: Directory containing pre-tokenized data
        """
        if not path.exists():
            raise ValueError(f"Pre-tokenized data not found at {path}")
        
        self._pretokenized_path = path
        
        # Load metadata
        with open(path / "metadata.json", "r") as f:
            metadata = json.load(f)
        
        # Verify compatibility
        if metadata["size"] != self.dataset.size:
            raise ValueError("Pre-tokenized data size mismatch")
    
    # Private helper methods
    
    def _check_pretokenized_data(self) -> None:
        """Check if pre-tokenized data exists."""
        # Look for pre-tokenized data in standard locations
        cache_dir = Path(".cache") / "datasets" / self.dataset.name / self.dataset.split.value
        if cache_dir.exists() and (cache_dir / "metadata.json").exists():
            try:
                self.load_pretokenized(cache_dir)
            except Exception:
                # Ignore errors and fall back to raw loading
                pass
    
    def _load_pretokenized_sample(self, idx: int) -> Any:
        """Load sample from pre-tokenized data."""
        if self._pretokenized_path is None:
            raise RuntimeError("No pre-tokenized data loaded")
        
        # Determine batch file
        batch_size = 1000
        batch_idx = idx // batch_size
        sample_idx = idx % batch_size
        
        batch_path = self._pretokenized_path / f"batch_{batch_idx}.pkl"
        
        # Load batch
        with open(batch_path, "rb") as f:
            batch_samples = pickle.load(f)
        
        return batch_samples[sample_idx]
    
    def _load_raw_sample(self, idx: int) -> Any:
        """Load sample from raw dataset features."""
        # This is a placeholder - actual implementation would depend on
        # how the dataset stores its data
        features = self.dataset.features
        
        sample = {}
        for feature_name, feature_data in features.items():
            if isinstance(feature_data, list) and idx < len(feature_data):
                sample[feature_name] = feature_data[idx]
            elif hasattr(feature_data, "__getitem__"):
                sample[feature_name] = feature_data[idx]
        
        return sample


class MLXIterableDataset:
    """Iterable dataset for streaming large datasets."""
    
    def __init__(
        self,
        dataset: Dataset,
        transform: Optional[callable] = None
    ):
        """Initialize iterable dataset.
        
        Args:
            dataset: The domain Dataset entity
            transform: Optional transform to apply to samples
        """
        self.dataset = dataset
        self.transform = transform
        self._iter_idx = 0
    
    def __iter__(self):
        """Iterate through dataset."""
        self._iter_idx = 0
        return self
    
    def __next__(self):
        """Get next sample."""
        if self._iter_idx >= self.dataset.size:
            raise StopIteration
        
        # Get sample (implementation depends on dataset source)
        sample = self._get_sample_at(self._iter_idx)
        
        # Apply transform if provided
        if self.transform is not None:
            sample = self.transform(sample)
        
        self._iter_idx += 1
        return sample
    
    def _get_sample_at(self, idx: int) -> Any:
        """Get sample at specific index."""
        # Placeholder - actual implementation depends on dataset source
        return {"index": idx}