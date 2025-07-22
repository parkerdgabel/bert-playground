# K-BERT Project Template

This is a base template for k-bert projects with custom components.

## Project Structure

```
.
├── k-bert.yaml           # Project configuration
├── src/
│   ├── heads/           # Custom BERT heads
│   │   └── custom_head.py
│   ├── augmenters/      # Data augmentation strategies
│   │   └── custom_augmenter.py
│   ├── features/        # Feature extractors
│   │   └── feature_extractors.py
│   ├── models/          # Custom model architectures
│   ├── metrics/         # Custom evaluation metrics
│   └── utils/           # Utility functions
├── data/
│   ├── raw/            # Original data
│   ├── processed/      # Preprocessed data
│   └── submissions/    # Competition submissions
├── configs/
│   ├── models/         # Model configurations
│   ├── training/       # Training configurations
│   └── experiments/    # Experiment configurations
├── notebooks/          # Jupyter notebooks
└── outputs/            # Training outputs
```

## Custom Components

### Custom Heads

The project includes example custom BERT heads in `src/heads/`:

- **CustomBinaryHead**: Binary classification with hidden layer
- **CustomMulticlassHead**: Multiclass classification with label smoothing

To create your own head:

1. Inherit from `HeadPlugin`
2. Implement required methods
3. Register with `@register_component`

### Data Augmenters

Example augmenters in `src/augmenters/`:

- **DomainSpecificAugmenter**: Text augmentation techniques
- **TabularDataAugmenter**: Augmentation for structured data

### Feature Extractors

Example extractors in `src/features/`:

- **TextStatisticsExtractor**: Extract text statistics
- **TemporalFeatureExtractor**: Extract time-based features

## Usage

### Training with Custom Components

```bash
# Train using project configuration
k-bert run

# Train with specific experiment
k-bert run --experiment full_pipeline

# Override configuration
k-bert train --train data/train.csv --config k-bert.yaml
```

### Using Custom Components Programmatically

```python
from k_bert.plugins import get_component

# Load custom head
head = get_component("head", "CustomBinaryHead", config={
    "hidden_size": 768,
    "dropout_prob": 0.1
})

# Load augmenter
augmenter = get_component("augmenter", "DomainSpecificAugmenter")

# Apply augmentation
augmented_data = augmenter.augment({"text": "Example text"})
```

### Adding New Components

1. Create a new Python file in the appropriate directory
2. Implement your component class inheriting from the base plugin
3. Register it with `@register_component`
4. Update `k-bert.yaml` to use your component

Example:

```python
from k_bert.plugins import HeadPlugin, register_component

@register_component(name="my_custom_head")
class MyCustomHead(HeadPlugin):
    def __init__(self, config=None):
        super().__init__(config)
        # Initialize your head
    
    def __call__(self, hidden_states, attention_mask=None, **kwargs):
        # Implement forward pass
        pass
    
    def compute_loss(self, logits, labels, **kwargs):
        # Implement loss computation
        pass
```

## Configuration

The `k-bert.yaml` file controls:

- Plugin directories
- Default model settings
- Component configurations
- Training parameters
- Experiment definitions

You can override any setting via CLI arguments or environment variables.

## Best Practices

1. **Version Control**: Track your custom components in git
2. **Testing**: Write tests for your components
3. **Documentation**: Document component parameters and behavior
4. **Modularity**: Keep components focused and reusable
5. **Configuration**: Use configuration files for reproducibility

## Next Steps

1. Customize the example components for your use case
2. Add domain-specific augmentation strategies
3. Create task-specific heads
4. Implement custom metrics
5. Run experiments to find optimal configurations