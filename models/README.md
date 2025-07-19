# MLX BERT Models Module

This module provides state-of-the-art BERT implementations optimized for Apple Silicon using the MLX framework. It includes classic BERT, ModernBERT from Answer.AI, and a variety of task-specific heads for classification and regression tasks.

## Table of Contents

- [Architecture Overview](#architecture-overview)
- [Model Variants](#model-variants)
- [Task Heads](#task-heads)
- [LoRA Adapters](#lora-adapters)
- [Usage Examples](#usage-examples)
- [Model Factory](#model-factory)
- [Configuration](#configuration)
- [MLX Optimizations](#mlx-optimizations)

## Architecture Overview

The models module follows a modular, composable architecture:

```
models/
├── bert/                    # Core BERT implementations
│   ├── core.py             # Classic BERT encoder
│   ├── modernbert_config.py # ModernBERT configuration
│   └── layers/             # Attention, embeddings, feedforward
├── heads/                  # Task-specific output layers
│   ├── classification.py   # Binary/multiclass/multilabel heads
│   └── regression.py       # Regression/ordinal heads
├── lora/                   # LoRA adapter implementation
│   ├── adapter.py          # Core LoRA functionality
│   └── config.py           # LoRA configuration
└── factory.py              # Unified model creation

```

## Model Variants

### 1. Classic BERT (`BertCore`)

Standard BERT implementation following the original paper:

```python
from models.bert import BertCore, BertConfig

# Create BERT-base configuration
config = BertConfig(
    vocab_size=30522,
    hidden_size=768,
    num_hidden_layers=12,
    num_attention_heads=12,
    intermediate_size=3072,
    max_position_embeddings=512
)

# Initialize model
bert = BertCore(config)
```

**Features:**
- Multi-head self-attention with proper masking
- GELU activation in feed-forward layers
- Layer normalization and dropout
- Support for token type embeddings
- Attention weight extraction for interpretability

### 2. ModernBERT (`ModernBertCore`)

Answer.AI's 2024 architecture with significant improvements:

```python
from models.bert import ModernBertCore, ModernBertConfig

# Create ModernBERT configuration
config = ModernBertConfig(
    vocab_size=50265,
    hidden_size=768,
    num_hidden_layers=22,
    num_attention_heads=12,
    intermediate_size=2688,
    max_position_embeddings=8192,
    rope_theta=160000,
    attention_dropout=0.0,
    alternating_attention=True
)

# Initialize model
modern_bert = ModernBertCore(config)
```

**Key Improvements:**
- **RoPE** (Rotary Position Embeddings) instead of learned positions
- **GeGLU/SwiGLU** activation functions for better performance
- **Alternating attention**: Local sliding window + global attention
- **Pre-normalization** with optional RMSNorm
- **8192 sequence length** support
- **No bias terms** for efficiency
- **Flash attention** compatible

### 3. neoBERT Configuration

Efficient 250M parameter variant:

```python
from models.factory import create_model

# Create neoBERT
neo_bert = create_model("neobert", num_labels=2)
```

**Specifications:**
- 28 layers (deeper than BERT-base)
- SwiGLU activation
- RoPE embeddings
- 4096 context length
- Optimized for efficiency

## Task Heads

The module provides specialized heads for different tasks:

### Classification Heads

#### Binary Classification
```python
from models.heads import BinaryClassificationHead

head = BinaryClassificationHead(
    hidden_size=768,
    hidden_dropout_prob=0.1,
    use_focal_loss=True,  # For imbalanced datasets
    focal_gamma=2.0,
    temperature_scaling=1.0  # For calibration
)
```

#### Multiclass Classification
```python
from models.heads import MulticlassClassificationHead

head = MulticlassClassificationHead(
    hidden_size=768,
    num_labels=10,
    label_smoothing=0.1,  # Regularization
    use_multiclass_focal=False
)
```

#### Multilabel Classification
```python
from models.heads import MultilabelClassificationHead

head = MultilabelClassificationHead(
    hidden_size=768,
    num_labels=20,
    pos_weight=None,  # For class imbalance
    adaptive_threshold=True  # Learn optimal thresholds
)
```

### Regression Heads

#### Standard Regression
```python
from models.heads import RegressionHead

head = RegressionHead(
    hidden_size=768,
    loss_type="mse",  # mse, mae, huber
    uncertainty_estimation=True  # Predict uncertainty
)
```

#### Ordinal Regression
```python
from models.heads import OrdinalRegressionHead

head = OrdinalRegressionHead(
    hidden_size=768,
    num_classes=5,  # Ordinal categories
    use_cumulative_link=True
)
```

#### Time Series Regression
```python
from models.heads import TimeSeriesRegressionHead

head = TimeSeriesRegressionHead(
    hidden_size=768,
    forecast_horizon=24,
    use_temporal_features=True
)
```

## LoRA Adapters

Low-Rank Adaptation for efficient fine-tuning:

### Basic Usage
```python
from models.lora import LoRAConfig, LoRAAdapter

# Configure LoRA
lora_config = LoRAConfig(
    r=8,  # Rank
    alpha=16,  # Scaling factor
    dropout=0.1,
    target_modules=["query", "value"]  # Which layers to adapt
)

# Apply to model
lora_adapter = LoRAAdapter(bert, lora_config)
```

### Presets
```python
from models.lora import create_lora_config

# Efficient: r=4, minimal parameters
config = create_lora_config("efficient")

# Balanced: r=8, good trade-off
config = create_lora_config("balanced")

# Expressive: r=16, maximum flexibility
config = create_lora_config("expressive")

# QLoRA: 4-bit quantization + LoRA
config = create_lora_config("qlora_memory")
```

### Advanced Features
- **DoRA**: Weight-decomposed LoRA for better performance
- **RSLoRA**: Rank-stabilized scaling
- **Layer-specific ranks**: Different ranks for different layers
- **Adapter merging**: Fuse adapters back into base model

## Usage Examples

### Complete Training Example
```python
from models.factory import create_bert_with_head
from training import create_trainer
import mlx.core as mx

# Create model with classification head
model = create_bert_with_head(
    model_type="modernbert",
    head_type="multiclass",
    num_labels=3,
    use_lora=True,
    lora_preset="balanced"
)

# Initialize trainer
trainer = create_trainer(
    model=model,
    config="configs/production.yaml"
)

# Train
trainer.train(train_data, val_data)
```

### Kaggle Competition Example
```python
from models.factory import create_kaggle_model

# Auto-configured for Titanic competition
model = create_kaggle_model(
    competition="titanic",
    use_lora=True,
    ensemble_size=3  # Create ensemble
)
```

### Custom Model Creation
```python
from models.factory import ModelFactory
from models.bert import BertConfig
from models.heads import HeadConfig

# Create custom configuration
bert_config = BertConfig(
    hidden_size=1024,
    num_hidden_layers=24,
    num_attention_heads=16
)

head_config = HeadConfig(
    head_type="binary",
    use_focal_loss=True,
    hidden_dropout_prob=0.2
)

# Build model
factory = ModelFactory()
model = factory.create_model(
    bert_config=bert_config,
    head_config=head_config
)
```

## Model Factory

The factory provides multiple levels of abstraction:

### High-Level API
```python
from models.factory import create_model

# Quick model creation
model = create_model("bert", num_labels=2)
model = create_model("modernbert", head_type="regression")
model = create_model("neobert", use_lora=True)
```

### Competition-Specific Models
```python
from models.factory import create_kaggle_model

# Optimized for specific competitions
model = create_kaggle_model("titanic")  # Binary classification
model = create_kaggle_model("house-prices")  # Regression
model = create_kaggle_model("nlp-disaster")  # Text classification
```

### Ensemble Creation
```python
from models.factory import create_ensemble

# Create ensemble of models
ensemble = create_ensemble(
    base_model="modernbert",
    num_models=5,
    diversity_strategy="different_seeds"
)
```

## Configuration

### Model Configuration
```yaml
model:
  architecture: modernbert
  hidden_size: 768
  num_layers: 12
  num_heads: 12
  max_length: 512
  
head:
  type: multiclass
  num_labels: 3
  dropout: 0.1
  
lora:
  enabled: true
  preset: balanced
  target_modules: ["query", "value", "key"]
```

### Loading from HuggingFace
```python
from models.factory import load_from_huggingface

# Load pretrained model
model = load_from_huggingface(
    "answerdotai/ModernBERT-base",
    add_lora=True,
    task="classification"
)
```

## MLX Optimizations

The implementation leverages MLX features throughout:

### Memory Efficiency
- **Unified memory**: Seamless CPU/GPU data movement
- **Lazy evaluation**: Computation only when needed
- **Gradient checkpointing**: Trade compute for memory

### Performance Optimizations
- **Fused operations**: Combined QKV projections
- **Optimized attention**: Memory-efficient implementation
- **Native operations**: Uses MLX primitives throughout

### Quantization Support
```python
from models.quantization_utils import quantize_model

# 4-bit quantization
quantized_model = quantize_model(
    model,
    bits=4,
    group_size=128
)

# 8-bit quantization
quantized_model = quantize_model(
    model,
    bits=8,
    quantize_embeddings=False
)
```

## Best Practices

1. **Model Selection**:
   - Use ModernBERT for best performance
   - Use Classic BERT for compatibility
   - Use neoBERT for efficiency

2. **LoRA Fine-tuning**:
   - Start with "balanced" preset
   - Target attention modules first
   - Use QLoRA for memory constraints

3. **Head Selection**:
   - Use focal loss for imbalanced data
   - Enable label smoothing for regularization
   - Consider ensemble heads for competitions

4. **Memory Management**:
   - Use gradient accumulation for large models
   - Enable gradient checkpointing if needed
   - Consider quantization for deployment

## Extending the Module

### Adding New Model Variants
1. Create configuration in `bert/config.py`
2. Implement model in `bert/core.py`
3. Register in factory

### Adding New Heads
1. Inherit from `BaseHead`
2. Implement `forward` and loss computation
3. Add to head registry

### Custom LoRA Strategies
1. Extend `LoRAAdapter`
2. Implement custom decomposition
3. Register preset in config

## Troubleshooting

### Common Issues

**Out of Memory**:
- Reduce batch size
- Enable gradient accumulation
- Use LoRA or quantization

**Slow Training**:
- Check batch size (use powers of 2)
- Enable prefetching in data loader
- Use mixed precision (automatic in MLX)

**Poor Performance**:
- Try different model variants
- Adjust learning rate
- Enable data augmentation

For more details, see the individual module documentation and implementation files.