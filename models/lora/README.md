# LoRA (Low-Rank Adaptation) for BERT Models

This module provides efficient fine-tuning capabilities for BERT models using LoRA and QLoRA techniques, optimized for MLX and designed specifically for Kaggle competitions.

## Overview

LoRA (Low-Rank Adaptation) enables efficient fine-tuning of large language models by adding small trainable rank decomposition matrices to frozen pretrained weights. This approach:

- **Reduces trainable parameters by 90-99%**
- **Maintains model quality** while using less memory
- **Enables training on consumer hardware**
- **Supports multiple adapters** for ensemble methods

## Quick Start

### Basic LoRA Fine-tuning

```python
from models import create_bert_with_lora

# Create BERT model with LoRA adapters
model, lora_adapter = create_bert_with_lora(
    head_type="binary_classification",
    lora_preset="balanced",  # r=8, balanced performance
    num_labels=2,
    freeze_bert=True,  # Recommended with LoRA
)

# Train only LoRA parameters (much faster!)
# ... your training code ...
```

### Memory-Efficient QLoRA

```python
from models import create_qlora_model

# Create 4-bit quantized model with LoRA
model, lora_adapter = create_qlora_model(
    model_type="modernbert_with_head",
    qlora_preset="qlora_memory",
    head_type="multiclass_classification",
    num_labels=10,
)
```

### Kaggle Competition Auto-Configuration

```python
from models import create_kaggle_lora_model

# Auto-configures LoRA based on competition type
model, lora_adapter = create_kaggle_lora_model(
    competition_type="binary_classification",
    data_path="data/train.csv",  # Optional: analyzes data
    auto_select_preset=True,  # Auto-selects best config
)
```

## Available Presets

| Preset | Rank (r) | Target Modules | Use Case |
|--------|----------|----------------|----------|
| `efficient` | 4 | Q, V only | Maximum efficiency, minimal resources |
| `balanced` | 8 | Q, K, V, O | Good balance of performance/efficiency |
| `expressive` | 16 | All attention + FFN | Maximum expressivity |
| `qlora_memory` | 4 | Q, V only | 4-bit models, extreme memory savings |
| `qlora_quality` | 8 | Q, K, V, O | 4-bit with better quality |

## Custom Configuration

```python
from models import LoRAConfig

# Create custom LoRA configuration
config = LoRAConfig(
    r=32,  # Higher rank for more parameters
    alpha=64,  # Scaling factor (typically alpha = 2*r)
    dropout=0.1,
    target_modules={"query", "key", "value", "dense"},
    use_rslora=True,  # Rank-Stabilized LoRA
    use_dora=False,   # Weight-Decomposed LoRA
)

# Use with any model
model, adapter = create_model_with_lora(
    model_type="bert_with_head",
    lora_config=config,
)
```

## Multi-Adapter Support

```python
from models import create_multi_adapter_model

# Create model with multiple adapters
model, manager = create_multi_adapter_model(
    adapter_configs={
        "task1": "efficient",
        "task2": "balanced",
        "ensemble": "expressive",
    }
)

# Switch between adapters
manager.activate_adapter("task1")
# ... train on task 1 ...

manager.activate_adapter("task2")
# ... train on task 2 ...
```

## Saving and Loading

```python
# Save only LoRA weights (much smaller)
lora_state = lora_adapter.get_lora_state_dict()
import safetensors.mlx
safetensors.mlx.save_file(lora_state, "lora_adapter.safetensors")

# Load LoRA weights
loaded_state = safetensors.mlx.load_file("lora_adapter.safetensors")
lora_adapter.load_lora_state_dict(loaded_state)

# Merge for deployment (no adapter overhead)
lora_adapter.merge_adapters()
```

## Architecture Details

### LoRA Mathematics

The LoRA adaptation works by decomposing weight updates:
```
W' = W + ΔW = W + BA * (α/r)
```
Where:
- `W` is the frozen pretrained weight
- `B` is the up-projection matrix (r × d_out)
- `A` is the down-projection matrix (d_in × r)
- `α` is the scaling factor
- `r` is the rank (typically 4-64)

### Target Modules

By default, LoRA targets these linear layers:
- **Attention**: query, key, value, output projections
- **FFN**: intermediate and output dense layers
- **Heads**: Can optionally target classification heads

### Memory Savings

For a BERT-base model:
- **Full fine-tuning**: ~110M parameters
- **LoRA (r=8)**: ~1-2M parameters (98% reduction)
- **QLoRA (4-bit + r=8)**: ~30MB model + 1-2M LoRA params

## Performance Tips

1. **Rank Selection**:
   - Small datasets: Use r=4-8
   - Large datasets: Use r=16-32
   - Start with r=8 and adjust

2. **Learning Rate**:
   - Use higher LR than full fine-tuning
   - Typical: 1e-4 to 5e-4
   - Can use different LR for LoRA vs base

3. **Target Modules**:
   - Minimum: Query and Value projections
   - Balanced: All attention projections
   - Maximum: Attention + FFN layers

4. **Batch Size**:
   - LoRA allows larger batch sizes
   - Memory saved can be used for batching

## Competition-Specific Recommendations

### Binary Classification (Titanic, etc.)
```python
model, adapter = create_bert_with_lora(
    head_type="binary_classification", 
    lora_preset="balanced",
    num_labels=2,
)
```

### Multi-class (MNIST, etc.)
```python
model, adapter = create_bert_with_lora(
    head_type="multiclass_classification",
    lora_preset="expressive",  # More parameters for complex tasks
    num_labels=num_classes,
)
```

### Regression (House Prices, etc.)
```python
model, adapter = create_bert_with_lora(
    head_type="regression",
    lora_preset="efficient",  # Regression often needs fewer params
    num_labels=1,
)
```

### Large Datasets (>100k samples)
```python
# Use QLoRA for memory efficiency
model, adapter = create_qlora_model(
    qlora_preset="qlora_memory",
    head_type=task_type,
)
```

## Advanced Features

### Layer-Specific Configuration
```python
config = LoRAConfig(
    r=8,
    layer_specific_config={
        "bert.encoder.layer.11.attention": {"r": 16},  # Higher rank for last layer
        "bert.encoder.layer.0.attention": {"r": 4},    # Lower rank for first layer
    }
)
```

### DoRA (Weight-Decomposed LoRA)
```python
config = LoRAConfig(
    r=8,
    use_dora=True,  # Decomposes weights into magnitude and direction
)
```

### RSLoRA (Rank-Stabilized LoRA)
```python
config = LoRAConfig(
    r=16,
    use_rslora=True,  # Better scaling for higher ranks
)
```

## Troubleshooting

**Issue**: Out of memory during training
- **Solution**: Use QLoRA or reduce rank

**Issue**: Poor performance with LoRA
- **Solution**: Increase rank or target more modules

**Issue**: Slow training
- **Solution**: Ensure base model is frozen, check target modules

**Issue**: Can't load saved adapter
- **Solution**: Ensure config matches original, check module names

## References

- [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)
- [QLoRA: Efficient Finetuning of Quantized LLMs](https://arxiv.org/abs/2305.14314)
- [DoRA: Weight-Decomposed Low-Rank Adaptation](https://arxiv.org/abs/2402.09353)