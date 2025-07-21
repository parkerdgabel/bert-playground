# Data Augmentation Guide

## Overview

The data augmentation module provides a flexible, dataset-agnostic framework for augmenting tabular and text data for BERT training. It supports multiple augmentation strategies based on feature types and is fully integrated with MLX for optimal performance on Apple Silicon.

## Key Features

- **Generic Framework**: Works with any dataset without hardcoding
- **Feature-Type Aware**: Different strategies for numerical, categorical, text, and date features
- **MLX Native**: All operations use MLX for unified memory and GPU acceleration
- **Extensible**: Easy to add custom augmentation strategies
- **Integrated**: Seamless integration with the data pipeline and BERT models

## Architecture

```
augmentation/
├── base.py              # Base classes and types
├── config.py            # Configuration system
├── strategies.py        # Feature-specific strategies
├── text.py             # BERT text augmentation
├── tabular.py          # Tabular data augmentation
├── tta.py              # Test-time augmentation
└── registry.py         # Strategy registry and manager
```

## Quick Start

### Basic Usage

```python
from bert_playground.data.augmentation import (
    AugmentationConfig,
    AugmentationMode,
    FeatureMetadata,
    FeatureType,
    TabularAugmenter,
)

# Define your features
feature_metadata = {
    "age": FeatureMetadata(
        name="age",
        feature_type=FeatureType.NUMERICAL,
        statistics={"mean": 30, "std": 10}
    ),
    "category": FeatureMetadata(
        name="category",
        feature_type=FeatureType.CATEGORICAL
    ),
}

# Create configuration
config = AugmentationConfig.from_mode(AugmentationMode.MODERATE)

# Create augmenter
augmenter = TabularAugmenter(config, feature_metadata)

# Augment data
sample = {"age": 25, "category": "A"}
augmented = augmenter.augment(sample)
```

### Integration with Datasets

```python
from bert_playground.data.core.base_dataset import BaseDataset
from bert_playground.data.augmentation import TabularToTextAugmenter

# Create dataset with augmentation
dataset = YourDataset(
    data_path="data.csv",
    augmenter=augmenter,
    augment_during_training=True
)

# Augmentation happens automatically during training
for sample in dataset:
    # Sample is augmented if in training mode
    pass

# Disable for validation
dataset.set_training_mode(False)
```

## Feature Types and Strategies

### Numerical Features

```python
config = NumericalAugmentationConfig(
    gaussian_noise_std=0.1,      # Standard deviation for noise
    apply_noise_prob=0.5,        # Probability of adding noise
    scale_range=(0.9, 1.1),      # Range for scaling
    apply_scaling_prob=0.3,      # Probability of scaling
    clip_outliers=True,          # Clip to prevent outliers
    outlier_std_threshold=3.0    # Threshold for outliers
)
```

**Strategies:**
- Gaussian noise addition
- Random scaling
- Outlier clipping
- Feature-specific bounds

### Categorical Features

```python
config = CategoricalAugmentationConfig(
    swap_prob=0.1,               # Probability of swapping values
    swap_with_similar=True,      # Swap with similar categories
    use_synonyms=True,           # Use synonym mappings
    add_unknown_prob=0.05,       # Probability of unknown token
    add_typos=False,             # Add typographical errors
)
```

**Strategies:**
- Value swapping
- Synonym replacement
- Unknown token injection
- Typo generation

### Text Features

```python
config = TextAugmentationConfig(
    mask_prob=0.15,              # BERT masking probability
    random_token_prob=0.1,       # Random token replacement
    delete_prob=0.1,             # Token deletion
    sentence_shuffle_prob=0.1,   # Sentence reordering
    use_back_translation=False,  # Back-translation (requires models)
)
```

**Strategies:**
- Token masking (BERT-style)
- Random token replacement
- Token deletion
- Sentence shuffling
- Synonym replacement

## Augmentation Modes

Pre-configured modes for common use cases:

```python
# No augmentation
config = AugmentationConfig.from_mode(AugmentationMode.NONE)

# Light augmentation (testing/development)
config = AugmentationConfig.from_mode(AugmentationMode.LIGHT)

# Moderate augmentation (default)
config = AugmentationConfig.from_mode(AugmentationMode.MODERATE)

# Heavy augmentation (aggressive)
config = AugmentationConfig.from_mode(AugmentationMode.HEAVY)

# Custom configuration
config = AugmentationConfig.from_mode(AugmentationMode.CUSTOM, 
    augmentation_prob=0.7,
    numerical_config=NumericalAugmentationConfig(gaussian_noise_std=0.2)
)
```

## BERT-Specific Features

### Text Augmentation for BERT

```python
from bert_playground.data.augmentation import BERTTextAugmenter

augmenter = BERTTextAugmenter(tokenizer)

# Single augmentation
text = "The quick brown fox"
augmented = augmenter.augment(text)

# Multiple augmentations
augmented_texts = augmenter.augment_text(text, num_augmentations=5)
```

### Tabular-to-Text with Augmentation

```python
from bert_playground.data.augmentation import TabularToTextAugmenter

# Custom text converter
def my_converter(features):
    return f"Customer age: {features['age']}, Type: {features['type']}"

augmenter = TabularToTextAugmenter(
    config=config,
    feature_metadata=metadata,
    text_converter=my_converter,
    tokenizer=tokenizer  # For text-level augmentation
)

# Augment and convert to text
text = augmenter.augment({"age": 25, "type": "premium"})
```

### Test-Time Augmentation (TTA)

```python
from bert_playground.data.augmentation import BERTTestTimeAugmentation

tta = BERTTestTimeAugmentation(text_augmenter, num_augmentations=5)

# Generate predictions with TTA
predictions = tta.predict_with_tta(model, dataloader)

# Combine predictions
combined = tta.combine_predictions(predictions_list, method="mean")
```

## Advanced Usage

### Custom Augmentation Strategies

```python
from bert_playground.data.augmentation import BaseAugmentationStrategy

class MyCustomStrategy(BaseAugmentationStrategy):
    def __init__(self):
        super().__init__("my_strategy", [FeatureType.NUMERICAL])
    
    def apply(self, data, config):
        # Your augmentation logic
        return augmented_data

# Register strategy
registry = get_registry()
registry.register_strategy("my_strategy", MyCustomStrategy())
```

### Augmentation Manager

```python
from bert_playground.data.augmentation import AugmentationManager

# Create manager with feature metadata
manager = AugmentationManager(
    feature_metadata=metadata,
    config=config
)

# Augment samples with specific strategies
augmented = manager.augment_sample(
    sample,
    strategies=["gaussian_noise", "synonym_replacement"]
)

# Get statistics
stats = manager.get_augmentation_stats()
```

### Domain-Specific Configuration

```python
from bert_playground.data.augmentation import DomainKnowledgeConfig

domain_config = DomainKnowledgeConfig(
    # Define feature relationships
    correlated_features={
        "income": ["age", "education"],
        "price": ["quality", "brand"]
    },
    
    # Add constraints
    feature_constraints={
        "age": {"min": 0, "max": 120},
        "probability": {"min": 0.0, "max": 1.0}
    },
    
    # Feature importance for augmentation
    feature_importance={
        "critical_feature": 0.1,  # Less augmentation
        "robust_feature": 0.9     # More augmentation
    }
)

config = AugmentationConfig(domain_config=domain_config)
```

## Performance Considerations

### MLX Optimization

All augmentation operations use MLX for optimal performance:

```python
# MLX random operations
noise = mx.random.normal(shape=(), scale=std)

# MLX array operations
augmented = mx.clip(value + noise, min_val, max_val)

# Efficient batch processing
batch_augmented = augmenter.augment_batch(batch)
```

### Caching

Enable caching for expensive augmentations:

```python
config = AugmentationConfig(
    cache_augmented_samples=True,
    cache_size=10000
)
```

### Memory Management

For large datasets, control augmentation probability:

```python
config = AugmentationConfig(
    augmentation_prob=0.3,  # Only augment 30% of samples
    mode=AugmentationMode.LIGHT  # Use lighter augmentation
)
```

## Integration with Training

### With BaseDataset

```python
class MyDataset(BaseDataset):
    def __init__(self, *args, **kwargs):
        # Create augmenter
        augmenter = self._create_augmenter()
        
        super().__init__(
            *args,
            augmenter=augmenter,
            augment_during_training=True,
            **kwargs
        )
    
    def _create_augmenter(self):
        # Define feature metadata
        metadata = self._get_feature_metadata()
        
        # Create config
        config = AugmentationConfig.from_mode(
            AugmentationMode.MODERATE
        )
        
        return TabularAugmenter(config, metadata)
```

### With DataLoader

```python
# Training loader with augmentation
train_dataset.set_training_mode(True)
train_loader = MLXDataLoader(train_dataset, ...)

# Validation loader without augmentation
val_dataset.set_training_mode(False)
val_loader = MLXDataLoader(val_dataset, ...)
```

## Best Practices

1. **Define Feature Metadata**: Always specify feature types and statistics
2. **Start Light**: Begin with light augmentation and increase gradually
3. **Validate Impact**: Monitor model performance with/without augmentation
4. **Feature-Specific**: Use appropriate strategies for each feature type
5. **Preserve Semantics**: Ensure augmentations maintain data meaning
6. **Test Augmentations**: Visualize augmented samples before training

## Troubleshooting

### Common Issues

**No augmentation happening:**
- Check `augment_during_training` is True
- Verify `is_training` mode is set
- Ensure augmentation probability > 0

**Too aggressive augmentation:**
- Reduce augmentation probability
- Use lighter augmentation mode
- Adjust feature-specific parameters

**Memory issues:**
- Disable caching
- Reduce batch size
- Use streaming augmentation

**Type errors:**
- Verify feature metadata matches data
- Check augmenter compatibility with features
- Ensure MLX arrays are properly converted

## Examples

See `examples/augmentation_demo.py` for complete working examples of:
- Basic feature augmentation
- Tabular-to-text augmentation
- BERT-specific augmentation
- Custom strategies
- Integration with datasets

## Future Enhancements

- Learned augmentation policies
- Adversarial augmentation
- Cross-feature augmentation
- AutoAugment integration
- Multi-modal augmentation