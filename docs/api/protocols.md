# Core Protocols and Interfaces API Reference

This document provides a comprehensive reference for all protocols and interfaces used throughout the MLX BERT project. These protocols define the contracts that implementations must follow.

## Table of Contents

- [Data Module Protocols](#data-module-protocols)
- [Model Module Protocols](#model-module-protocols)
- [Training Module Protocols](#training-module-protocols)
- [Common Types](#common-types)

## Data Module Protocols

### `data.core.interfaces`

#### Dataset Protocol

```python
class Dataset(Protocol):
    """Protocol for all dataset implementations."""
    
    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        ...
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get a single sample from the dataset.
        
        Args:
            idx: Index of the sample
            
        Returns:
            Dictionary with at least 'input_ids' and 'labels' keys
        """
        ...
```

#### DataLoader Protocol

```python
class DataLoader(Protocol):
    """Protocol for data loaders."""
    
    def __iter__(self) -> Iterator[Batch]:
        """Iterate over batches of data."""
        ...
    
    def __len__(self) -> int:
        """Return the number of batches."""
        ...
    
    @property
    def batch_size(self) -> int:
        """Return the batch size."""
        ...
```

#### TextConverter Protocol

```python
class TextConverter(Protocol):
    """Protocol for converting data to text."""
    
    def convert(self, row: Dict[str, Any]) -> str:
        """
        Convert a data row to text representation.
        
        Args:
            row: Dictionary containing data fields
            
        Returns:
            Text representation of the data
        """
        ...
    
    def fit(self, data: pd.DataFrame) -> None:
        """
        Fit the converter on training data (optional).
        
        Args:
            data: Training dataframe
        """
        ...
```

#### Tokenizer Protocol

```python
class Tokenizer(Protocol):
    """Protocol for tokenizers."""
    
    def __call__(
        self,
        text: Union[str, List[str]],
        max_length: Optional[int] = None,
        padding: Union[bool, str] = True,
        truncation: bool = True,
        return_tensors: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Tokenize input text.
        
        Args:
            text: Input text or list of texts
            max_length: Maximum sequence length
            padding: Padding strategy
            truncation: Whether to truncate
            return_tensors: Output tensor format
            
        Returns:
            Dictionary with 'input_ids' and 'attention_mask'
        """
        ...
    
    @property
    def vocab_size(self) -> int:
        """Return vocabulary size."""
        ...
```

#### Cache Protocol

```python
class Cache(Protocol):
    """Protocol for caching mechanisms."""
    
    def get(self, key: str) -> Optional[Any]:
        """Retrieve item from cache."""
        ...
    
    def put(self, key: str, value: Any) -> None:
        """Store item in cache."""
        ...
    
    def clear(self) -> None:
        """Clear all cached items."""
        ...
    
    def size(self) -> int:
        """Return number of cached items."""
        ...
```

#### StreamBuilder Protocol

```python
class StreamBuilder(Protocol):
    """Protocol for building data streams."""
    
    def add_source(self, source: Any) -> "StreamBuilder":
        """Add a data source."""
        ...
    
    def add_transform(self, transform: Callable) -> "StreamBuilder":
        """Add a transformation."""
        ...
    
    def build(self) -> DataStream:
        """Build the stream."""
        ...
```

#### BatchProcessor Protocol

```python
class BatchProcessor(Protocol):
    """Protocol for batch processing."""
    
    def process(self, batch: List[Any]) -> Batch:
        """
        Process a list of samples into a batch.
        
        Args:
            batch: List of samples
            
        Returns:
            Processed batch with MLX arrays
        """
        ...
```

### Data Types

```python
# Type aliases used in data module
Sample = Dict[str, Union[mx.array, np.ndarray, List, Any]]
Batch = Dict[str, mx.array]
DataStream = Iterator[Batch]
Transform = Callable[[Any], Any]
```

## Model Module Protocols

### `models.core.protocols`

#### Model Protocol

```python
class Model(Protocol):
    """Protocol for all model implementations."""
    
    def __call__(
        self,
        input_ids: mx.array,
        attention_mask: Optional[mx.array] = None,
        token_type_ids: Optional[mx.array] = None,
        **kwargs
    ) -> ModelOutput:
        """
        Forward pass through the model.
        
        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            token_type_ids: Token type IDs for BERT
            **kwargs: Additional arguments
            
        Returns:
            Model output with logits/predictions
        """
        ...
    
    def parameters(self) -> Dict[str, mx.array]:
        """Return model parameters."""
        ...
    
    def save(self, path: str) -> None:
        """Save model to path."""
        ...
    
    def load(self, path: str) -> None:
        """Load model from path."""
        ...
```

#### Head Protocol

```python
class Head(Protocol):
    """Protocol for task-specific heads."""
    
    def __call__(
        self,
        hidden_states: mx.array,
        labels: Optional[mx.array] = None,
        **kwargs
    ) -> HeadOutput:
        """
        Process hidden states through the head.
        
        Args:
            hidden_states: Encoder output
            labels: Target labels for loss computation
            **kwargs: Additional arguments
            
        Returns:
            Head output with predictions and optional loss
        """
        ...
    
    def compute_metrics(
        self,
        predictions: mx.array,
        labels: mx.array
    ) -> Dict[str, float]:
        """Compute task-specific metrics."""
        ...
```

#### Adapter Protocol

```python
class Adapter(Protocol):
    """Protocol for model adapters (e.g., LoRA)."""
    
    def apply(self, model: Model) -> Model:
        """Apply adapter to model."""
        ...
    
    def merge(self) -> None:
        """Merge adapter weights into base model."""
        ...
    
    def save_adapter(self, path: str) -> None:
        """Save only adapter weights."""
        ...
    
    def load_adapter(self, path: str) -> None:
        """Load adapter weights."""
        ...
```

### Model Output Types

```python
@dataclass
class ModelOutput:
    """Base class for model outputs."""
    logits: Optional[mx.array] = None
    hidden_states: Optional[Tuple[mx.array, ...]] = None
    attentions: Optional[Tuple[mx.array, ...]] = None
    
@dataclass
class HeadOutput:
    """Base class for head outputs."""
    predictions: mx.array
    loss: Optional[mx.array] = None
    metrics: Optional[Dict[str, float]] = None
```

## Training Module Protocols

### `training.core.protocols`

#### Trainer Protocol

```python
class Trainer(Protocol):
    """Protocol for trainer implementations."""
    
    def train(
        self,
        train_dataloader: DataLoader,
        val_dataloader: Optional[DataLoader] = None,
        num_epochs: Optional[int] = None,
    ) -> TrainingResult:
        """
        Train the model.
        
        Args:
            train_dataloader: Training data
            val_dataloader: Validation data
            num_epochs: Number of epochs
            
        Returns:
            Training result with metrics
        """
        ...
    
    def evaluate(
        self,
        dataloader: DataLoader,
        prefix: str = "eval"
    ) -> Dict[str, float]:
        """Evaluate model on dataset."""
        ...
    
    def predict(
        self,
        dataloader: DataLoader,
        return_logits: bool = False
    ) -> mx.array:
        """Generate predictions."""
        ...
```

#### Optimizer Protocol

```python
class Optimizer(Protocol):
    """Protocol for optimizers."""
    
    def update(
        self,
        model: Model,
        gradients: Dict[str, mx.array]
    ) -> None:
        """
        Update model parameters with gradients.
        
        Args:
            model: Model to update
            gradients: Parameter gradients
        """
        ...
    
    @property
    def learning_rate(self) -> float:
        """Current learning rate."""
        ...
    
    def state_dict(self) -> Dict[str, Any]:
        """Return optimizer state."""
        ...
    
    def load_state_dict(self, state: Dict[str, Any]) -> None:
        """Load optimizer state."""
        ...
```

#### LRScheduler Protocol

```python
class LRScheduler(Protocol):
    """Protocol for learning rate schedulers."""
    
    def step(self, metrics: Optional[float] = None) -> None:
        """
        Update learning rate.
        
        Args:
            metrics: Current metric value (for ReduceLROnPlateau)
        """
        ...
    
    def get_last_lr(self) -> float:
        """Get current learning rate."""
        ...
    
    def state_dict(self) -> Dict[str, Any]:
        """Return scheduler state."""
        ...
```

#### TrainingHook Protocol

```python
class TrainingHook(Protocol):
    """Protocol for training callbacks/hooks."""
    
    def on_train_begin(self, trainer: Trainer, state: TrainingState) -> None:
        """Called at the beginning of training."""
        ...
    
    def on_epoch_begin(self, trainer: Trainer, state: TrainingState) -> None:
        """Called at the beginning of each epoch."""
        ...
    
    def on_step_begin(self, trainer: Trainer, state: TrainingState) -> None:
        """Called before each training step."""
        ...
    
    def on_step_end(self, trainer: Trainer, state: TrainingState) -> None:
        """Called after each training step."""
        ...
    
    def on_epoch_end(self, trainer: Trainer, state: TrainingState) -> None:
        """Called at the end of each epoch."""
        ...
    
    def on_train_end(self, trainer: Trainer, state: TrainingState) -> None:
        """Called at the end of training."""
        ...
```

### Training Types

```python
@dataclass
class TrainingState:
    """Training state information."""
    epoch: int = 0
    global_step: int = 0
    train_loss: float = 0.0
    val_loss: Optional[float] = None
    metrics: Dict[str, float] = field(default_factory=dict)
    best_metric: Optional[float] = None
    early_stopping_counter: int = 0
    should_stop: bool = False

@dataclass
class TrainingResult:
    """Training result."""
    final_loss: float
    best_metric: float
    metrics_history: Dict[str, List[float]]
    model_path: str
    training_time: float
```

## Common Types

### Type Aliases

```python
# Common type aliases used across modules
from typing import TypeVar, Union, Dict, Any, Optional

# MLX array type
Array = mx.array

# Generic tensor type (MLX or NumPy)
Tensor = Union[mx.array, np.ndarray]

# Model parameters
Parameters = Dict[str, Array]

# Gradients
Gradients = Dict[str, Array]

# Generic numeric type
Number = Union[int, float]

# Competition types
CompetitionType = Literal[
    "binary_classification",
    "multiclass_classification",
    "multilabel_classification",
    "regression",
    "ordinal_regression",
    "time_series_regression"
]
```

### Utility Protocols

```python
class Serializable(Protocol):
    """Protocol for serializable objects."""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        ...
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Serializable":
        """Create from dictionary."""
        ...

class Configurable(Protocol):
    """Protocol for configurable objects."""
    
    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "Configurable":
        """Create from configuration."""
        ...
    
    def get_config(self) -> Dict[str, Any]:
        """Get configuration."""
        ...
```

## Usage Examples

### Implementing a Custom Dataset

```python
from data.core.interfaces import Dataset
import mlx.core as mx

class MyDataset:
    """Custom dataset implementation."""
    
    def __init__(self, data_path: str):
        self.data = self._load_data(data_path)
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, mx.array]:
        sample = self.data[idx]
        return {
            "input_ids": mx.array(sample["tokens"]),
            "attention_mask": mx.array(sample["mask"]),
            "labels": mx.array(sample["label"])
        }
```

### Implementing a Custom Head

```python
from models.core.protocols import Head, HeadOutput
import mlx.core as mx
import mlx.nn as nn

class CustomHead(nn.Module):
    """Custom task head implementation."""
    
    def __init__(self, hidden_size: int, num_classes: int):
        super().__init__()
        self.classifier = nn.Linear(hidden_size, num_classes)
    
    def __call__(
        self,
        hidden_states: mx.array,
        labels: Optional[mx.array] = None
    ) -> HeadOutput:
        logits = self.classifier(hidden_states)
        
        loss = None
        if labels is not None:
            loss = mx.mean(
                nn.losses.cross_entropy(logits, labels)
            )
        
        return HeadOutput(
            predictions=logits,
            loss=loss
        )
```

### Implementing a Custom Callback

```python
from training.core.protocols import TrainingHook, TrainingState, Trainer

class LearningRateLogger:
    """Log learning rate at each step."""
    
    def on_step_end(self, trainer: Trainer, state: TrainingState) -> None:
        lr = trainer.optimizer.learning_rate
        print(f"Step {state.global_step}: LR = {lr:.6f}")
```

## Best Practices

1. **Type Hints**: Always use type hints when implementing protocols
2. **Documentation**: Document all methods clearly
3. **Error Handling**: Raise appropriate exceptions for invalid inputs
4. **Validation**: Validate inputs in implementations
5. **Testing**: Write tests to verify protocol compliance

## Protocol Extensions

Protocols can be extended for specific use cases:

```python
class StreamingDataset(Dataset, Protocol):
    """Extended dataset protocol for streaming."""
    
    def prefetch(self, n: int) -> None:
        """Prefetch n samples."""
        ...
    
    def set_epoch(self, epoch: int) -> None:
        """Set epoch for shuffling."""
        ...
```

This allows for more specialized implementations while maintaining compatibility with the base protocol.