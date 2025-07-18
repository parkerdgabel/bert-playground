# HuggingFace Hub Integration

This document describes the new HuggingFace Hub integration features added to the BERT Kaggle models implementation.

## Features

### 🌐 HuggingFace Hub Model Loading

You can now load MLX-native BERT models directly from the HuggingFace Hub:

```python
from models.bert.core import BertCore

# Load an MLX-native BERT model from HuggingFace Hub
model = BertCore.from_pretrained("mlx-community/bert-base-uncased")

# Use with factory
from models.factory import create_model
model = create_model("bert_core", model_name="mlx-community/bert-base-uncased")
```

### 📁 Local Model Loading

Enhanced local model loading with HuggingFace config compatibility:

```python
# Load from local directory with HuggingFace config format
model = BertCore.from_pretrained("./path/to/local/model")

# Works with both MLX and HuggingFace config formats
```

### ⚙️ Configuration Compatibility

Full compatibility with HuggingFace configuration format:

```python
from models.bert.config import BertConfig

# Convert HuggingFace config to MLX format
hf_config = {...}  # Standard HuggingFace config
mlx_config = BertConfig.from_hf_config(hf_config)

# Convert MLX config to HuggingFace format
hf_config = mlx_config.to_hf_config()
```

## Supported Models

The integration specifically supports **MLX-native BERT models** from the HuggingFace Hub, particularly from the `mlx-community` organization.

### Example Models

- `mlx-community/bert-base-uncased`
- `mlx-community/bert-large-uncased`
- `mlx-community/distilbert-base-uncased`
- Any other MLX-native BERT model on the Hub

## Requirements

To use HuggingFace Hub downloading, install the optional dependency:

```bash
pip install huggingface_hub
```

## Technical Details

### Model ID Detection

The system automatically detects HuggingFace model IDs using the pattern `organization/model-name`:

```python
from models.bert.core import _is_hub_model_id

# Returns True for Hub model IDs
_is_hub_model_id("mlx-community/bert-base-uncased")  # True

# Returns False for local paths
_is_hub_model_id("./local_model")  # False
```

### File Format Support

- **Preferred**: `.safetensors` format (MLX-native)
- **Config**: `config.json` (both HuggingFace and MLX formats)
- **Fallback**: `.bin` files (with warning)

### Configuration Mapping

The system maps HuggingFace configuration fields to MLX equivalents:

| HuggingFace Field | MLX Field |
|------------------|-----------|
| `hidden_size` | `hidden_size` |
| `num_hidden_layers` | `num_hidden_layers` |
| `num_attention_heads` | `num_attention_heads` |
| `intermediate_size` | `intermediate_size` |
| `vocab_size` | `vocab_size` |
| `max_position_embeddings` | `max_position_embeddings` |
| `type_vocab_size` | `type_vocab_size` |
| `layer_norm_eps` | `layer_norm_eps` |

## Usage Examples

### Basic Usage

```python
from models.bert.core import BertCore

# Load model from Hub
model = BertCore.from_pretrained("mlx-community/bert-base-uncased")

# Create input tensors
import mlx.core as mx
input_ids = mx.ones((2, 10), dtype=mx.int32)
attention_mask = mx.ones((2, 10), dtype=mx.int32)

# Forward pass
output = model(input_ids, attention_mask)
print(f"Output shape: {output.last_hidden_state.shape}")
```

### With Classification Head

```python
from models.factory import create_model

# Create model with classification head
model = create_model(
    "bert_with_head",
    model_name="mlx-community/bert-base-uncased",
    head_type="binary_classification"
)

# Use for classification
predictions = model(input_ids, attention_mask)
print(f"Logits shape: {predictions['logits'].shape}")
```

### Custom Cache Directory

```python
# Use custom cache directory
model = BertCore.from_pretrained(
    "mlx-community/bert-base-uncased",
    cache_dir="./my_model_cache"
)
```

## Error Handling

The system includes robust error handling:

- **Missing `huggingface_hub`**: Clear error message with installation instructions
- **Download failures**: Informative error messages
- **Missing weight files**: Graceful fallback to random initialization
- **Config errors**: Fallback to default configuration

## Backward Compatibility

All existing functionality remains unchanged:

- Local model loading still works
- MLX-native configurations still work
- All existing heads and factory functions work
- No breaking changes to existing APIs

## Testing

Run the comprehensive test suite:

```bash
uv run python test_huggingface_integration.py
```

This tests:
- Model ID detection
- Configuration compatibility
- Local model loading
- Factory integration
- Hub model simulation

## Future Enhancements

Potential future improvements:

1. **Weight Conversion**: Automatic conversion from PyTorch BERT models
2. **More Architectures**: Support for other transformer architectures
3. **Quantization**: Integration with MLX quantization features
4. **Caching**: More sophisticated caching strategies