"""Dataset fixtures for testing."""

import mlx.core as mx
from typing import Dict, List, Optional, Tuple, Iterator
import numpy as np


class SyntheticDataLoader:
    """Synthetic data loader for testing."""
    
    def __init__(
        self,
        num_samples: int,
        batch_size: int,
        input_dim: int = 10,
        output_dim: int = 2,
        task_type: str = "classification",
        seed: int = 42,
        infinite: bool = False,
    ):
        self.num_samples = num_samples
        self.batch_size = batch_size
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.task_type = task_type
        self.seed = seed
        self.infinite = infinite
        
        # Generate synthetic data
        np.random.seed(seed)
        self.X = mx.array(np.random.randn(num_samples, input_dim).astype(np.float32))
        
        if task_type == "classification":
            self.y = mx.array(np.random.randint(0, output_dim, size=num_samples))
        else:  # regression
            self.y = mx.array(np.random.randn(num_samples).astype(np.float32))
        
        self._index = 0
    
    def __iter__(self) -> Iterator[Dict[str, mx.array]]:
        self._index = 0
        return self
    
    def __next__(self) -> Dict[str, mx.array]:
        if not self.infinite and self._index >= len(self):
            raise StopIteration
        
        start_idx = (self._index * self.batch_size) % self.num_samples
        end_idx = min(start_idx + self.batch_size, self.num_samples)
        
        if end_idx == self.num_samples and not self.infinite:
            # Last batch might be smaller
            batch_X = self.X[start_idx:end_idx]
            batch_y = self.y[start_idx:end_idx]
        else:
            # Handle wrap-around for infinite loader
            if end_idx > self.num_samples:
                indices = list(range(start_idx, self.num_samples)) + list(range(0, end_idx - self.num_samples))
                batch_X = self.X[indices]
                batch_y = self.y[indices]
            else:
                batch_X = self.X[start_idx:end_idx]
                batch_y = self.y[start_idx:end_idx]
        
        self._index += 1
        
        key = "labels" if self.task_type == "classification" else "targets"
        return {"input": batch_X, key: batch_y}
    
    def __len__(self) -> int:
        return (self.num_samples + self.batch_size - 1) // self.batch_size


class ImbalancedDataLoader(SyntheticDataLoader):
    """Imbalanced dataset for testing."""
    
    def __init__(self, num_samples: int = 1000, batch_size: int = 32, imbalance_ratio: float = 0.9, **kwargs):
        super().__init__(num_samples, batch_size, output_dim=2, task_type="classification", **kwargs)
        
        # Create imbalanced labels
        num_positive = int(num_samples * (1 - imbalance_ratio))
        labels = np.zeros(num_samples, dtype=np.int32)
        labels[:num_positive] = 1
        np.random.shuffle(labels)
        self.y = mx.array(labels)


class NoisyDataLoader(SyntheticDataLoader):
    """Dataset with label noise for testing."""
    
    def __init__(self, num_samples: int = 1000, batch_size: int = 32, noise_rate: float = 0.1, **kwargs):
        super().__init__(num_samples, batch_size, **kwargs)
        
        # Add label noise
        if self.task_type == "classification":
            num_noisy = int(num_samples * noise_rate)
            noisy_indices = np.random.choice(num_samples, num_noisy, replace=False)
            labels = self.y.tolist()
            for idx in noisy_indices:
                labels[idx] = (labels[idx] + 1) % self.output_dim
            self.y = mx.array(labels)


class VariableLengthDataLoader:
    """Data loader with variable length sequences for testing."""
    
    def __init__(
        self,
        num_samples: int = 100,
        batch_size: int = 4,
        min_length: int = 5,
        max_length: int = 20,
        vocab_size: int = 100,
    ):
        self.num_samples = num_samples
        self.batch_size = batch_size
        self.min_length = min_length
        self.max_length = max_length
        self.vocab_size = vocab_size
        
        # Generate variable length sequences
        self.sequences = []
        self.labels = []
        
        for _ in range(num_samples):
            length = np.random.randint(min_length, max_length + 1)
            seq = np.random.randint(0, vocab_size, size=length)
            label = np.random.randint(0, 2)
            self.sequences.append(seq)
            self.labels.append(label)
        
        self._index = 0
    
    def __iter__(self):
        self._index = 0
        return self
    
    def __next__(self) -> Dict[str, mx.array]:
        if self._index >= len(self):
            raise StopIteration
        
        batch_sequences = []
        batch_labels = []
        batch_size = min(self.batch_size, len(self) - self._index)
        
        for i in range(batch_size):
            idx = self._index * self.batch_size + i
            if idx >= self.num_samples:
                break
            batch_sequences.append(self.sequences[idx])
            batch_labels.append(self.labels[idx])
        
        # Pad sequences to same length
        max_len = max(len(seq) for seq in batch_sequences)
        padded_sequences = []
        attention_mask = []
        
        for seq in batch_sequences:
            pad_len = max_len - len(seq)
            padded_seq = np.pad(seq, (0, pad_len), constant_values=0)
            mask = np.ones(len(seq), dtype=np.float32)
            mask = np.pad(mask, (0, pad_len), constant_values=0)
            padded_sequences.append(padded_seq)
            attention_mask.append(mask)
        
        self._index += 1
        
        return {
            "input_ids": mx.array(padded_sequences),
            "attention_mask": mx.array(attention_mask),
            "labels": mx.array(batch_labels),
        }
    
    def __len__(self) -> int:
        return (self.num_samples + self.batch_size - 1) // self.batch_size


class EmptyDataLoader:
    """Empty data loader for edge case testing."""
    
    def __iter__(self):
        return self
    
    def __next__(self):
        raise StopIteration
    
    def __len__(self):
        return 0


class SingleBatchDataLoader:
    """Data loader with only one batch."""
    
    def __init__(self, batch_size: int = 4, input_dim: int = 10):
        self.batch_size = batch_size
        self.input_dim = input_dim
        self._yielded = False
    
    def __iter__(self):
        self._yielded = False
        return self
    
    def __next__(self):
        if self._yielded:
            raise StopIteration
        
        self._yielded = True
        return {
            "input": mx.random.normal((self.batch_size, self.input_dim)),
            "labels": mx.random.randint(0, 2, (self.batch_size,)),
        }
    
    def __len__(self):
        return 1


# Factory functions
def create_test_dataloader(
    dataset_type: str = "synthetic",
    num_samples: int = 100,
    batch_size: int = 4,
    **kwargs
) -> Iterator[Dict[str, mx.array]]:
    """Create test data loader by type."""
    loaders = {
        "synthetic": SyntheticDataLoader,
        "imbalanced": ImbalancedDataLoader,
        "noisy": NoisyDataLoader,
        "variable_length": VariableLengthDataLoader,
        "empty": EmptyDataLoader,
        "single_batch": SingleBatchDataLoader,
    }
    
    if dataset_type not in loaders:
        raise ValueError(f"Unknown dataset type: {dataset_type}")
    
    loader_class = loaders[dataset_type]
    
    # Handle special cases
    if dataset_type in ["empty", "single_batch"]:
        return loader_class(batch_size=batch_size, **kwargs)
    
    return loader_class(num_samples=num_samples, batch_size=batch_size, **kwargs)


def create_cv_splits(
    num_samples: int = 1000,
    num_folds: int = 5,
    stratified: bool = True,
    labels: Optional[mx.array] = None,
) -> List[Tuple[mx.array, mx.array]]:
    """Create cross-validation splits for testing."""
    indices = mx.arange(num_samples)
    
    if stratified and labels is not None:
        # Simple stratified split
        splits = []
        unique_labels = mx.unique(labels)
        
        for fold in range(num_folds):
            train_indices = []
            val_indices = []
            
            for label in unique_labels:
                label_indices = mx.where(labels == label)[0]
                n_samples = len(label_indices)
                fold_size = n_samples // num_folds
                
                start = fold * fold_size
                end = start + fold_size if fold < num_folds - 1 else n_samples
                
                val_idx = label_indices[start:end]
                train_idx = mx.concatenate([label_indices[:start], label_indices[end:]])
                
                val_indices.append(val_idx)
                train_indices.append(train_idx)
            
            train_indices = mx.concatenate(train_indices)
            val_indices = mx.concatenate(val_indices)
            splits.append((train_indices, val_indices))
    else:
        # Simple k-fold split
        fold_size = num_samples // num_folds
        splits = []
        
        for fold in range(num_folds):
            start = fold * fold_size
            end = start + fold_size if fold < num_folds - 1 else num_samples
            
            val_indices = indices[start:end]
            train_indices = mx.concatenate([indices[:start], indices[end:]])
            splits.append((train_indices, val_indices))
    
    return splits