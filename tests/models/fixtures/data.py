"""Data fixtures for model testing."""

from typing import Dict, Optional, Tuple, List
import mlx.core as mx
import numpy as np


def create_test_batch(
    batch_size: int = 4,
    seq_length: int = 128,
    vocab_size: int = 30522,
    include_labels: bool = True,
    num_classes: int = 2,
    seed: Optional[int] = 42,
) -> Dict[str, mx.array]:
    """Create a test batch for model testing."""
    if seed is not None:
        mx.random.seed(seed)
    
    # Generate input IDs
    input_ids = mx.random.randint(0, vocab_size, (batch_size, seq_length))
    
    # Generate attention mask (some sequences have padding)
    attention_mask = mx.ones((batch_size, seq_length))
    for i in range(batch_size):
        # Randomly add padding to some sequences
        if mx.random.uniform() > 0.5:
            padding_length = mx.random.randint(1, seq_length // 4).item()
            attention_mask[i, -padding_length:] = 0
            input_ids[i, -padding_length:] = 0  # Use 0 as padding token
    
    # Generate token type IDs
    token_type_ids = mx.zeros((batch_size, seq_length), dtype=mx.int32)
    
    batch = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "token_type_ids": token_type_ids,
    }
    
    # Add labels if requested
    if include_labels:
        if num_classes == 1:  # Regression
            labels = mx.random.normal((batch_size, 1))
        else:  # Classification
            labels = mx.random.randint(0, num_classes, (batch_size,))
        batch["labels"] = labels
    
    return batch


def create_variable_length_batch(
    batch_size: int = 4,
    min_length: int = 32,
    max_length: int = 128,
    vocab_size: int = 30522,
    pad_to_max: bool = True,
) -> Dict[str, mx.array]:
    """Create batch with variable length sequences."""
    mx.random.seed(42)
    
    # Generate random lengths for each sequence
    lengths = mx.random.randint(min_length, max_length + 1, (batch_size,))
    
    if pad_to_max:
        # Pad all sequences to max_length
        input_ids = mx.zeros((batch_size, max_length), dtype=mx.int32)
        attention_mask = mx.zeros((batch_size, max_length))
        
        for i in range(batch_size):
            seq_len = lengths[i].item()
            input_ids[i, :seq_len] = mx.random.randint(1, vocab_size, (seq_len,))
            attention_mask[i, :seq_len] = 1
    else:
        # Return list of variable length sequences
        sequences = []
        masks = []
        for i in range(batch_size):
            seq_len = lengths[i].item()
            seq = mx.random.randint(1, vocab_size, (seq_len,))
            mask = mx.ones((seq_len,))
            sequences.append(seq)
            masks.append(mask)
        
        return {
            "sequences": sequences,
            "attention_masks": masks,
            "lengths": lengths,
        }
    
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "lengths": lengths,
    }


def create_attention_mask(
    batch_size: int = 4,
    seq_length: int = 128,
    mask_type: str = "padding",
    mask_ratio: float = 0.15,
) -> mx.array:
    """Create different types of attention masks."""
    if mask_type == "padding":
        # Padding mask (1s followed by 0s)
        mask = mx.ones((batch_size, seq_length))
        for i in range(batch_size):
            if mx.random.uniform() > 0.3:  # 70% of sequences have padding
                padding_length = int(seq_length * mx.random.uniform(0.1, 0.5).item())
                mask[i, -padding_length:] = 0
        return mask
    
    elif mask_type == "random":
        # Random masking (like BERT MLM)
        mask = mx.random.uniform((batch_size, seq_length)) > mask_ratio
        return mask.astype(mx.float32)
    
    elif mask_type == "causal":
        # Causal mask for autoregressive models
        mask = mx.tril(mx.ones((seq_length, seq_length)))
        return mx.broadcast_to(mask[None, None, :, :], (batch_size, 1, seq_length, seq_length))
    
    elif mask_type == "block":
        # Block masking (contiguous spans)
        mask = mx.ones((batch_size, seq_length))
        for i in range(batch_size):
            num_blocks = mx.random.randint(1, 4).item()
            for _ in range(num_blocks):
                block_size = mx.random.randint(5, 20).item()
                start = mx.random.randint(0, max(1, seq_length - block_size)).item()
                mask[i, start:start + block_size] = 0
        return mask
    
    else:
        raise ValueError(f"Unknown mask type: {mask_type}")


def create_position_ids(
    batch_size: int = 4,
    seq_length: int = 128,
    position_type: str = "absolute",
) -> mx.array:
    """Create position IDs for testing."""
    if position_type == "absolute":
        # Standard absolute positions
        position_ids = mx.arange(seq_length)[None, :]
        return mx.broadcast_to(position_ids, (batch_size, seq_length))
    
    elif position_type == "relative":
        # Relative positions (for testing relative position embeddings)
        position_ids = []
        for i in range(batch_size):
            # Create relative positions from a random starting point
            start = mx.random.randint(0, 100).item()
            ids = mx.arange(start, start + seq_length)
            position_ids.append(ids)
        return mx.stack(position_ids)
    
    elif position_type == "random":
        # Random positions (for testing robustness)
        return mx.random.randint(0, seq_length, (batch_size, seq_length))
    
    else:
        raise ValueError(f"Unknown position type: {position_type}")


def create_embeddings(
    batch_size: int = 4,
    seq_length: int = 128,
    hidden_size: int = 768,
    embedding_type: str = "random",
) -> mx.array:
    """Create embeddings for testing."""
    if embedding_type == "random":
        # Random embeddings
        return mx.random.normal((batch_size, seq_length, hidden_size))
    
    elif embedding_type == "zero":
        # Zero embeddings
        return mx.zeros((batch_size, seq_length, hidden_size))
    
    elif embedding_type == "one":
        # One embeddings
        return mx.ones((batch_size, seq_length, hidden_size))
    
    elif embedding_type == "position":
        # Position-based embeddings (for testing position encoding)
        embeddings = mx.zeros((batch_size, seq_length, hidden_size))
        for i in range(seq_length):
            # Create position-specific pattern
            embeddings[:, i, :] = mx.sin(mx.array(i) / 10000 ** (mx.arange(hidden_size) / hidden_size))
        return embeddings
    
    else:
        raise ValueError(f"Unknown embedding type: {embedding_type}")


def create_classification_targets(
    batch_size: int = 4,
    num_classes: int = 2,
    target_type: str = "random",
    class_weights: Optional[List[float]] = None,
) -> mx.array:
    """Create classification targets."""
    if target_type == "random":
        # Random labels
        return mx.random.randint(0, num_classes, (batch_size,))
    
    elif target_type == "balanced":
        # Balanced labels
        labels = []
        for i in range(batch_size):
            labels.append(i % num_classes)
        return mx.array(labels)
    
    elif target_type == "imbalanced":
        # Imbalanced labels (majority class is 0)
        labels = mx.zeros((batch_size,), dtype=mx.int32)
        # Make ~20% positive class
        num_positive = max(1, int(batch_size * 0.2))
        positive_indices = mx.random.choice(batch_size, num_positive, replace=False)
        labels[positive_indices] = 1
        return labels
    
    elif target_type == "weighted":
        # Weighted random sampling
        if class_weights is None:
            class_weights = [1.0] * num_classes
        
        # Normalize weights
        total_weight = sum(class_weights)
        probs = [w / total_weight for w in class_weights]
        
        # Sample according to weights
        labels = []
        for _ in range(batch_size):
            r = mx.random.uniform().item()
            cumsum = 0
            for i, p in enumerate(probs):
                cumsum += p
                if r <= cumsum:
                    labels.append(i)
                    break
        
        return mx.array(labels)
    
    else:
        raise ValueError(f"Unknown target type: {target_type}")


def create_regression_targets(
    batch_size: int = 4,
    output_dim: int = 1,
    target_type: str = "random",
    range_min: float = -1.0,
    range_max: float = 1.0,
) -> mx.array:
    """Create regression targets."""
    if target_type == "random":
        # Random targets in range
        return mx.random.uniform(
            low=range_min,
            high=range_max,
            shape=(batch_size, output_dim)
        )
    
    elif target_type == "linear":
        # Linear targets (for testing linear relationships)
        x = mx.linspace(range_min, range_max, batch_size)
        if output_dim == 1:
            return x.reshape(-1, 1)
        else:
            # Multiple linear functions
            targets = []
            for i in range(output_dim):
                slope = (i + 1) / output_dim
                targets.append(slope * x + i * 0.1)
            return mx.stack(targets, axis=1)
    
    elif target_type == "sine":
        # Sinusoidal targets (for testing non-linear relationships)
        x = mx.linspace(0, 2 * np.pi, batch_size)
        if output_dim == 1:
            return mx.sin(x).reshape(-1, 1)
        else:
            # Multiple sine waves with different frequencies
            targets = []
            for i in range(output_dim):
                freq = (i + 1)
                targets.append(mx.sin(freq * x))
            return mx.stack(targets, axis=1)
    
    else:
        raise ValueError(f"Unknown target type: {target_type}")


def create_model_inputs(
    model_type: str = "bert",
    batch_size: int = 4,
    **kwargs
) -> Dict[str, mx.array]:
    """Create inputs specific to model type."""
    if model_type == "bert":
        return create_test_batch(batch_size=batch_size, **kwargs)
    
    elif model_type == "classifier":
        # Simple classifier expects flat features
        input_dim = kwargs.get("input_dim", 768)
        return {
            "features": mx.random.normal((batch_size, input_dim)),
            "labels": create_classification_targets(
                batch_size,
                kwargs.get("num_classes", 2)
            ),
        }
    
    elif model_type == "regressor":
        # Regression model expects flat features
        input_dim = kwargs.get("input_dim", 768)
        output_dim = kwargs.get("output_dim", 1)
        return {
            "features": mx.random.normal((batch_size, input_dim)),
            "targets": create_regression_targets(batch_size, output_dim),
        }
    
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def create_edge_case_data() -> Dict[str, Dict[str, mx.array]]:
    """Create edge case data for testing model robustness."""
    return {
        "empty_batch": {
            "input_ids": mx.zeros((0, 128), dtype=mx.int32),
            "attention_mask": mx.zeros((0, 128)),
        },
        "single_token": {
            "input_ids": mx.array([[101]]),  # Single [CLS] token
            "attention_mask": mx.array([[1]]),
        },
        "max_length": {
            "input_ids": mx.ones((1, 512), dtype=mx.int32),
            "attention_mask": mx.ones((1, 512)),
        },
        "all_padding": {
            "input_ids": mx.zeros((4, 128), dtype=mx.int32),
            "attention_mask": mx.zeros((4, 128)),
        },
        "very_long": {
            "input_ids": mx.ones((1, 8192), dtype=mx.int32),
            "attention_mask": mx.ones((1, 8192)),
        },
    }