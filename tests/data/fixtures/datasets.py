"""Dataset fixtures for data module testing."""

from typing import Dict, List, Optional, Any, Iterator
import mlx.core as mx
import pandas as pd
import numpy as np
from pathlib import Path
import json

from data.core.base import DatasetSpec, CompetitionType, KaggleDataset
from data.core.metadata import CompetitionMetadata


class MockKaggleDataset(KaggleDataset):
    """Mock KaggleDataset for testing."""
    
    def __init__(
        self,
        spec: DatasetSpec,
        size: Optional[int] = None,
        cache_dir: Optional[Path] = None,
        split: str = "train",
        transform=None,
        **kwargs
    ):
        """Initialize mock dataset with synthetic data."""
        # Set attributes before calling parent init
        # Use size from spec if not provided
        self.size = size if size is not None else spec.num_samples
        self._generated = False
        self.transform = transform
        
        # Call parent init which will call _load_data and _validate_data
        super().__init__(spec, split=split, cache_dir=cache_dir, **kwargs)
        
    def _generate_data(self):
        """Generate synthetic data based on competition type."""
        if self._generated:
            return
            
        np.random.seed(42)
        
        # Generate features
        num_features = self.spec.num_features or 10
        data = {
            f"feature_{i}": np.random.randn(self.size) 
            for i in range(num_features)
        }
        
        # Add categorical features
        if self.spec.categorical_columns:
            for col in self.spec.categorical_columns:
                data[col] = np.random.choice(['A', 'B', 'C'], size=self.size)
                
        # Add text features  
        if self.spec.text_columns:
            for col in self.spec.text_columns:
                data[col] = [f"Text sample {i}" for i in range(self.size)]
                
        # Generate target based on competition type
        if self.spec.competition_type == CompetitionType.BINARY_CLASSIFICATION:
            data[self.spec.target_column or 'target'] = np.random.randint(0, 2, size=self.size)
        elif self.spec.competition_type == CompetitionType.MULTICLASS_CLASSIFICATION:
            num_classes = self.spec.num_classes or 3
            data[self.spec.target_column or 'target'] = np.random.randint(0, num_classes, size=self.size)
        elif self.spec.competition_type == CompetitionType.REGRESSION:
            data[self.spec.target_column or 'target'] = np.random.randn(self.size)
            
        self._data = pd.DataFrame(data)
        self._generated = True
        
    def __len__(self) -> int:
        """Return dataset size."""
        return self.size
        
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get item by index."""
        if not self._generated:
            self._generate_data()
            
        if idx >= self.size:
            raise IndexError(f"Index {idx} out of range for dataset of size {self.size}")
            
        row = self._data.iloc[idx]
        row_dict = row.to_dict()
        
        # Create text representation
        text_parts = []
        for col, val in row_dict.items():
            if col != self.spec.target_column:
                text_parts.append(f"{col}: {val}")
        text = " ".join(text_parts)
        
        # Extract label
        label = row_dict.get(self.spec.target_column, 0)
        
        # Return expected structure
        # For testing, return pre-tokenized dummy data
        max_length = 128
        sample = {
            "text": text,
            "input_ids": np.random.randint(1, 1000, size=max_length),  # Dummy token IDs
            "attention_mask": np.ones(max_length, dtype=np.int32),  # All tokens are valid
            "labels": label,
            "metadata": {
                "index": idx,
                "raw_data": row_dict,
            }
        }
        
        # Apply transform if provided
        if self.transform:
            sample = self.transform(sample)
            
        return sample
        
    def get_dataframe(self) -> pd.DataFrame:
        """Get full dataset as DataFrame."""
        if not self._generated:
            self._generate_data()
        return self._data.copy()
    
    def _load_data(self) -> None:
        """Load data - for mock, just generate it."""
        if not self._generated:
            self._generate_data()
        # Store in parent class's _data attribute
        self._data = self._data
    
    def _validate_data(self) -> None:
        """Validate data - for mock, just check it's not empty."""
        if self._data is None or len(self._data) == 0:
            raise ValueError("Mock data is empty")


class SyntheticTabularDataset:
    """Synthetic tabular dataset for testing."""
    
    def __init__(
        self,
        num_samples: int = 1000,
        num_numeric_features: int = 5,
        num_categorical_features: int = 3,
        num_text_features: int = 2,
        task_type: str = "classification",
        num_classes: int = 2,
        seed: int = 42,
    ):
        self.num_samples = num_samples
        self.num_numeric_features = num_numeric_features
        self.num_categorical_features = num_categorical_features
        self.num_text_features = num_text_features
        self.task_type = task_type
        self.num_classes = num_classes
        self.seed = seed
        
        # Generate data
        np.random.seed(seed)
        self.data = self._generate_data()
        self.metadata = self._create_metadata()
    
    def _generate_data(self) -> pd.DataFrame:
        """Generate synthetic data."""
        data = {}
        
        # Numeric features
        for i in range(self.num_numeric_features):
            data[f"numeric_{i}"] = np.random.randn(self.num_samples)
        
        # Categorical features
        for i in range(self.num_categorical_features):
            categories = [f"cat_{j}" for j in range(np.random.randint(2, 10))]
            data[f"categorical_{i}"] = np.random.choice(categories, self.num_samples)
        
        # Text features
        text_samples = [
            "The weather is nice today",
            "I love machine learning",
            "This is a test sentence",
            "Data science is fascinating",
            "Python is a great language",
        ]
        for i in range(self.num_text_features):
            data[f"text_{i}"] = np.random.choice(text_samples, self.num_samples)
        
        # Target variable
        if self.task_type == "classification":
            data["target"] = np.random.randint(0, self.num_classes, self.num_samples)
        else:  # regression
            data["target"] = np.random.randn(self.num_samples)
        
        return pd.DataFrame(data)
    
    def _create_metadata(self) -> dict:
        """Create dataset metadata."""
        feature_types = {}
        
        for col in self.data.columns:
            if col == "target":
                continue
            elif col.startswith("numeric_"):
                feature_types[col] = "numeric"
            elif col.startswith("categorical_"):
                feature_types[col] = "categorical"
            elif col.startswith("text_"):
                feature_types[col] = "text"
        
        return {
            "name": "synthetic_tabular",
            "task_type": self.task_type,
            "num_samples": self.num_samples,
            "feature_names": list(feature_types.keys()),
            "feature_types": feature_types,
            "target_name": "target",
            "num_classes": self.num_classes if self.task_type == "classification" else None,
        }
    
    def to_dataframe(self) -> pd.DataFrame:
        """Get data as pandas DataFrame."""
        return self.data.copy()
    
    def save(self, path: Path):
        """Save dataset to file."""
        self.data.to_csv(path / "data.csv", index=False)
        with open(path / "metadata.json", "w") as f:
            json.dump(self.metadata, f, indent=2)


class SyntheticTextDataset:
    """Synthetic text dataset for testing."""
    
    def __init__(
        self,
        num_samples: int = 1000,
        vocab_size: int = 10000,
        sequence_length: int = 128,
        task_type: str = "classification",
        num_classes: int = 2,
        seed: int = 42,
    ):
        self.num_samples = num_samples
        self.vocab_size = vocab_size
        self.sequence_length = sequence_length
        self.task_type = task_type
        self.num_classes = num_classes
        self.seed = seed
        
        np.random.seed(seed)
        mx.random.seed(seed)
    
    def __len__(self) -> int:
        return self.num_samples
    
    def __getitem__(self, idx: int) -> Dict[str, mx.array]:
        """Get item by index."""
        # Ensure deterministic generation based on index
        mx.random.seed(self.seed + idx)
        
        # Generate token IDs
        input_ids = mx.random.randint(1, self.vocab_size, (self.sequence_length,))
        
        # Generate attention mask with some padding
        seq_len = mx.random.randint(self.sequence_length // 2, self.sequence_length).item()
        attention_mask = mx.zeros((self.sequence_length,))
        attention_mask[:seq_len] = 1
        
        # Generate label
        if self.task_type == "classification":
            label = mx.array(idx % self.num_classes)
        else:  # regression
            label = mx.array(float(idx) / self.num_samples)
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": label,
        }
    
    def create_metadata(self) -> dict:
        """Create dataset metadata."""
        return {
            "name": "synthetic_text",
            "task_type": self.task_type,
            "num_samples": self.num_samples,
            "feature_names": ["input_ids", "attention_mask"],
            "feature_types": {
                "input_ids": "text",
                "attention_mask": "numeric",
            },
            "target_name": "labels",
            "num_classes": self.num_classes if self.task_type == "classification" else None,
        }


class ImbalancedDataset:
    """Imbalanced dataset for testing."""
    
    def __init__(
        self,
        num_samples: int = 1000,
        imbalance_ratio: float = 0.1,  # minority class ratio
        num_features: int = 10,
        seed: int = 42,
    ):
        self.num_samples = num_samples
        self.imbalance_ratio = imbalance_ratio
        self.num_features = num_features
        self.seed = seed
        
        np.random.seed(seed)
        self.data = self._generate_imbalanced_data()
    
    def _generate_imbalanced_data(self) -> pd.DataFrame:
        """Generate imbalanced dataset."""
        # Calculate class distribution
        num_minority = int(self.num_samples * self.imbalance_ratio)
        num_majority = self.num_samples - num_minority
        
        # Generate features
        features = []
        labels = []
        
        # Majority class (label 0)
        for _ in range(num_majority):
            # Majority class centered around 0
            feat = np.random.randn(self.num_features)
            features.append(feat)
            labels.append(0)
        
        # Minority class (label 1)
        for _ in range(num_minority):
            # Minority class centered around 2
            feat = np.random.randn(self.num_features) + 2
            features.append(feat)
            labels.append(1)
        
        # Shuffle
        indices = np.random.permutation(self.num_samples)
        features = np.array(features)[indices]
        labels = np.array(labels)[indices]
        
        # Create DataFrame
        data = pd.DataFrame(
            features,
            columns=[f"feature_{i}" for i in range(self.num_features)]
        )
        data["target"] = labels
        
        return data
    
    def get_class_weights(self) -> Dict[int, float]:
        """Calculate class weights for balanced training."""
        class_counts = self.data["target"].value_counts()
        total = len(self.data)
        
        weights = {}
        for cls, count in class_counts.items():
            weights[cls] = total / (len(class_counts) * count)
        
        return weights


class StreamingDataset:
    """Streaming dataset for testing streaming functionality."""
    
    def __init__(
        self,
        spec_or_num_samples = None,
        num_samples: int = 10000,
        chunk_size: int = 100,
        delay: float = 0.01,  # Simulate network delay
        num_features: int = 10,
        seed: int = 42,
        size: Optional[int] = None,
        **kwargs,
    ):
        # Handle different calling patterns
        if size is not None:
            self.num_samples = size
        elif spec_or_num_samples is not None:
            if isinstance(spec_or_num_samples, int):
                self.num_samples = spec_or_num_samples
            elif hasattr(spec_or_num_samples, 'num_samples'):
                self.num_samples = spec_or_num_samples.num_samples
            else:
                self.num_samples = num_samples
        else:
            self.num_samples = num_samples
            
        self.chunk_size = chunk_size
        self.delay = delay
        self.num_features = num_features
        self.seed = seed
        self._position = 0
        self.transform = kwargs.get('transform', None)
    
    def __len__(self) -> int:
        """Return dataset size."""
        return self.num_samples
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get a single item from the dataset."""
        if idx >= self.num_samples:
            raise IndexError(f"Index {idx} out of range for dataset of size {self.num_samples}")
        
        mx.random.seed(self.seed + idx)
        
        # Generate sample
        sample = {
            "input_ids": mx.random.randint(0, 1000, (50,)),
            "attention_mask": mx.ones((50,), dtype=mx.int32),
            "labels": mx.array(idx % 2),
            "features": mx.random.normal((self.num_features,)),
        }
        
        # Apply transform if provided
        if self.transform:
            sample = self.transform(sample)
            
        return sample
    
    def __iter__(self) -> Iterator[Dict[str, mx.array]]:
        """Iterate through streaming data."""
        self._position = 0
        return self
    
    def __next__(self) -> Dict[str, mx.array]:
        """Get next chunk of data."""
        if self._position >= self.num_samples:
            raise StopIteration
        
        # Simulate streaming delay
        import time
        time.sleep(self.delay)
        
        # Generate chunk
        chunk_size = min(self.chunk_size, self.num_samples - self._position)
        mx.random.seed(self.seed + self._position)
        
        features = mx.random.normal((chunk_size, self.num_features))
        labels = mx.random.randint(0, 2, (chunk_size,))
        
        self._position += chunk_size
        
        return {
            "features": features,
            "labels": labels,
            "chunk_id": mx.array(self._position // self.chunk_size),
        }
    
    def reset(self):
        """Reset stream position."""
        self._position = 0


class MultiModalDataset:
    """Multi-modal dataset with mixed data types."""
    
    def __init__(
        self,
        num_samples: int = 1000,
        image_shape: tuple = (3, 32, 32),
        text_length: int = 50,
        num_tabular_features: int = 10,
        seed: int = 42,
    ):
        self.num_samples = num_samples
        self.image_shape = image_shape
        self.text_length = text_length
        self.num_tabular_features = num_tabular_features
        self.seed = seed
    
    def __len__(self) -> int:
        return self.num_samples
    
    def __getitem__(self, idx: int) -> Dict[str, mx.array]:
        """Get multi-modal sample."""
        mx.random.seed(self.seed + idx)
        
        # Image data
        image = mx.random.uniform(0, 1, self.image_shape)
        
        # Text data
        text_ids = mx.random.randint(1, 1000, (self.text_length,))
        
        # Tabular data
        tabular = mx.random.normal((self.num_tabular_features,))
        
        # Label (binary classification)
        label = mx.array(idx % 2)
        
        return {
            "image": image,
            "text_ids": text_ids,
            "tabular_features": tabular,
            "label": label,
        }


class CorruptDataset:
    """Dataset that produces corrupt data for error testing."""
    
    def __init__(
        self,
        num_samples: int = 100,
        corruption_rate: float = 0.2,
        corruption_types: List[str] = ["missing", "nan", "wrong_shape"],
        num_features: int = 10,
        seed: int = 42,
    ):
        self.num_samples = num_samples
        self.corruption_rate = corruption_rate
        self.corruption_types = corruption_types
        self.num_features = num_features
        self.seed = seed
    
    def __len__(self) -> int:
        return self.num_samples
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get item with possible corruption."""
        np.random.seed(self.seed + idx)
        
        # Decide if this sample should be corrupt
        if np.random.random() < self.corruption_rate:
            corruption_type = np.random.choice(self.corruption_types)
            
            if corruption_type == "missing":
                # Return incomplete data
                return {"features": mx.random.normal((self.num_features,))}
            
            elif corruption_type == "nan":
                # Return data with NaN
                features = mx.random.normal((self.num_features,))
                features = features / 0  # Force NaN
                return {
                    "features": features,
                    "labels": mx.array(0),
                }
            
            elif corruption_type == "wrong_shape":
                # Return wrong shaped data
                return {
                    "features": mx.random.normal((self.num_features + 5,)),
                    "labels": mx.array([0, 1]),  # Wrong shape for single sample
                }
            
            elif corruption_type == "exception":
                # Raise exception
                raise ValueError(f"Corrupt data at index {idx}")
        
        # Return normal data
        return {
            "features": mx.random.normal((self.num_features,)),
            "labels": mx.array(idx % 2),
        }


class FaultyDataset:
    """Dataset that randomly fails for testing error handling."""
    
    def __init__(
        self,
        spec: DatasetSpec,
        error_rate: float = 0.1,
        **kwargs
    ):
        """Initialize faulty dataset."""
        self.spec = spec
        self.error_rate = error_rate
        self.num_samples = spec.num_samples
        
    def __len__(self) -> int:
        """Return dataset size."""
        return self.num_samples
        
    def __getitem__(self, idx: int) -> Optional[Dict[str, mx.array]]:
        """Get item with random failures."""
        if np.random.random() < self.error_rate:
            raise RuntimeError(f"Simulated error at index {idx}")
            
        # Return normal data
        return {
            "input_ids": mx.random.randint(0, 1000, (128,)),
            "attention_mask": mx.ones((128,)),
            "labels": mx.array(idx % 2),
        }


class LargeDataset:
    """Large dataset for memory and performance testing."""
    
    def __init__(
        self,
        num_samples: int = 1000000,
        feature_dim: int = 1000,
        sparse: bool = False,
        sparsity: float = 0.9,
        seed: int = 42,
    ):
        self.num_samples = num_samples
        self.feature_dim = feature_dim
        self.sparse = sparse
        self.sparsity = sparsity
        self.seed = seed
    
    def __len__(self) -> int:
        return self.num_samples
    
    def __getitem__(self, idx: int) -> Dict[str, mx.array]:
        """Get large sample."""
        mx.random.seed(self.seed + idx)
        
        if self.sparse:
            # Generate sparse features
            features = mx.random.normal((self.feature_dim,))
            mask = mx.random.uniform((self.feature_dim,)) > self.sparsity
            features = features * mask
        else:
            # Dense features
            features = mx.random.normal((self.feature_dim,))
        
        # Simple label based on sum of features
        label = mx.array(int(mx.sum(features) > 0))
        
        return {
            "features": features,
            "label": label,
        }


# Factory functions
def create_kaggle_like_dataset(
    competition_name: str = "titanic",
    num_samples: int = 1000,
    num_features: int = 10,
    task_type: str = "binary_classification",
    **kwargs
) -> MockKaggleDataset:
    """Create a Kaggle-like dataset for testing."""
    # Map task type to CompetitionType enum
    from data.core.base import CompetitionType
    
    competition_type_map = {
        "binary_classification": CompetitionType.BINARY_CLASSIFICATION,
        "multiclass_classification": CompetitionType.MULTICLASS_CLASSIFICATION,
        "regression": CompetitionType.REGRESSION,
        "time_series": CompetitionType.TIME_SERIES,
    }
    
    competition_type = competition_type_map.get(task_type, CompetitionType.BINARY_CLASSIFICATION)
    
    # Create dataset spec
    spec = DatasetSpec(
        competition_name=competition_name,
        dataset_path=kwargs.get("dataset_path", Path("/tmp/test_data")),
        competition_type=competition_type,
        num_samples=num_samples,
        num_features=num_features,
        target_column=kwargs.get("target_column", "target"),
        text_columns=kwargs.get("text_columns", []),
        categorical_columns=kwargs.get("categorical_columns", ["cat_1", "cat_2"]),
        numerical_columns=[f"num_{i}" for i in range(num_features - 2)],  # Minus categorical columns
        num_classes=kwargs.get("num_classes", 2 if "classification" in task_type else None),
    )
    
    # Remove size from kwargs if it exists to avoid duplicate
    kwargs_filtered = {k: v for k, v in kwargs.items() if k not in ['size', 'cache_dir']}
    
    return MockKaggleDataset(
        spec=spec,
        size=num_samples,
        cache_dir=kwargs.get("cache_dir"),
        **kwargs_filtered
    )


def create_classification_dataset(
    num_samples: int = 1000,
    num_classes: int = 2,
    dataset_type: str = "tabular",
    **kwargs
) -> Any:
    """Create classification dataset of specified type."""
    if dataset_type == "tabular":
        return SyntheticTabularDataset(
            num_samples=num_samples,
            task_type="classification",
            num_classes=num_classes,
            **kwargs
        )
    elif dataset_type == "text":
        return SyntheticTextDataset(
            num_samples=num_samples,
            task_type="classification",
            num_classes=num_classes,
            **kwargs
        )
    elif dataset_type == "imbalanced":
        return ImbalancedDataset(
            num_samples=num_samples,
            **kwargs
        )
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")


def create_regression_dataset(
    num_samples: int = 1000,
    dataset_type: str = "tabular",
    **kwargs
) -> Any:
    """Create regression dataset of specified type."""
    if dataset_type == "tabular":
        return SyntheticTabularDataset(
            num_samples=num_samples,
            task_type="regression",
            **kwargs
        )
    elif dataset_type == "text":
        return SyntheticTextDataset(
            num_samples=num_samples,
            task_type="regression",
            **kwargs
        )
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")


def create_edge_case_datasets() -> Dict[str, Any]:
    """Create datasets for edge case testing."""
    return {
        "empty": type("EmptyDataset", (), {"__len__": lambda self: 0}),
        "single_sample": SyntheticTabularDataset(num_samples=1),
        "large_features": SyntheticTabularDataset(
            num_samples=10,
            num_numeric_features=1000,
        ),
        "no_features": SyntheticTabularDataset(
            num_samples=100,
            num_numeric_features=0,
            num_categorical_features=0,
            num_text_features=0,
        ),
        "all_same_label": type(
            "AllSameLabel",
            (),
            {
                "__len__": lambda self: 100,
                "__getitem__": lambda self, idx: {
                    "features": mx.random.normal((10,)),
                    "label": mx.array(0),  # Always same label
                },
            },
        )(),
    }