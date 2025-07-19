"""Dataset fixtures for data module testing."""

from typing import Dict, List, Optional, Any, Iterator
import mlx.core as mx
import pandas as pd
import numpy as np
from pathlib import Path
import json

from data.core.interfaces import Dataset
from data.core.metadata import DatasetMetadata, FeatureType


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
    
    def _create_metadata(self) -> DatasetMetadata:
        """Create dataset metadata."""
        feature_types = {}
        
        for col in self.data.columns:
            if col == "target":
                continue
            elif col.startswith("numeric_"):
                feature_types[col] = FeatureType.NUMERIC
            elif col.startswith("categorical_"):
                feature_types[col] = FeatureType.CATEGORICAL
            elif col.startswith("text_"):
                feature_types[col] = FeatureType.TEXT
        
        return DatasetMetadata(
            name="synthetic_tabular",
            task_type=self.task_type,
            num_samples=self.num_samples,
            feature_names=list(feature_types.keys()),
            feature_types=feature_types,
            target_name="target",
            num_classes=self.num_classes if self.task_type == "classification" else None,
        )
    
    def to_dataframe(self) -> pd.DataFrame:
        """Get data as pandas DataFrame."""
        return self.data.copy()
    
    def save(self, path: Path):
        """Save dataset to file."""
        self.data.to_csv(path / "data.csv", index=False)
        with open(path / "metadata.json", "w") as f:
            json.dump(self.metadata.to_dict(), f, indent=2)


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
    
    def create_metadata(self) -> DatasetMetadata:
        """Create dataset metadata."""
        return DatasetMetadata(
            name="synthetic_text",
            task_type=self.task_type,
            num_samples=self.num_samples,
            feature_names=["input_ids", "attention_mask"],
            feature_types={
                "input_ids": FeatureType.TEXT,
                "attention_mask": FeatureType.NUMERIC,
            },
            target_name="labels",
            num_classes=self.num_classes if self.task_type == "classification" else None,
        )


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
        num_samples: int = 10000,
        chunk_size: int = 100,
        delay: float = 0.01,  # Simulate network delay
        num_features: int = 10,
        seed: int = 42,
    ):
        self.num_samples = num_samples
        self.chunk_size = chunk_size
        self.delay = delay
        self.num_features = num_features
        self.seed = seed
        self._position = 0
    
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