"""
ModernBERT Configuration Attributes Reference
=============================================

This script demonstrates all available configuration attributes for the
answerdotai/ModernBERT-base model from HuggingFace.
"""

from transformers import AutoConfig

# Load the ModernBERT configuration
config = AutoConfig.from_pretrained("answerdotai/ModernBERT-base")

print("ModernBERT Configuration Attributes")
print("=" * 50)
print()

# Core Model Architecture Parameters
print("CORE MODEL ARCHITECTURE:")
print(f"  vocab_size: {config.vocab_size} - Vocabulary size of the model")
print(f"  hidden_size: {config.hidden_size} - Dimension of hidden representations")
print(f"  intermediate_size: {config.intermediate_size} - Dimension of MLP representations")
print(f"  num_hidden_layers: {config.num_hidden_layers} - Number of Transformer decoder layers")
print(f"  num_attention_heads: {config.num_attention_heads} - Number of attention heads per layer")
print(f"  hidden_activation: {config.hidden_activation} - Non-linear activation function")
print(f"  max_position_embeddings: {config.max_position_embeddings} - Maximum sequence length")
print()

# Initialization Parameters
print("INITIALIZATION PARAMETERS:")
print(f"  initializer_range: {config.initializer_range} - Std dev for weight initialization")
print(f"  initializer_cutoff_factor: {config.initializer_cutoff_factor} - Truncated normal cutoff")
print()

# Normalization Parameters
print("NORMALIZATION PARAMETERS:")
print(f"  norm_eps: {config.norm_eps} - Epsilon for RMS normalization layers")
print(f"  norm_bias: {config.norm_bias} - Whether to use bias in normalization layers")
print()

# Token IDs
print("SPECIAL TOKEN IDS:")
print(f"  pad_token_id: {config.pad_token_id} - ID for padding token")
print(f"  eos_token_id: {config.eos_token_id} - ID for end-of-sequence token")
print(f"  bos_token_id: {config.bos_token_id} - ID for beginning-of-sequence token")
print()

# Attention Configuration
print("ATTENTION CONFIGURATION:")
print(f"  attention_bias: {config.attention_bias} - Whether to use bias in attention projections")
print(f"  attention_dropout: {config.attention_dropout} - Dropout for attention probabilities")
print(f"  global_attn_every_n_layers: {config.global_attn_every_n_layers} - Frequency of global attention layers")
print(f"  local_attention: {config.local_attention} - Local attention window size")
print()

# RoPE (Rotary Position Embedding) Parameters
print("ROPE PARAMETERS:")
print(f"  global_rope_theta: {config.global_rope_theta} - Base period for global RoPE embeddings")
print(f"  local_rope_theta: {config.local_rope_theta} - Base period for local RoPE embeddings")
print()

# Dropout Parameters
print("DROPOUT PARAMETERS:")
print(f"  embedding_dropout: {config.embedding_dropout} - Dropout for embeddings")
print(f"  mlp_dropout: {config.mlp_dropout} - Dropout for MLP layers")
print(f"  classifier_dropout: {config.classifier_dropout} - Dropout for classifier")
print()

# MLP Configuration
print("MLP CONFIGURATION:")
print(f"  mlp_bias: {config.mlp_bias} - Whether to use bias in MLP layers")
print(f"  classifier_bias: {config.classifier_bias} - Whether to use bias in classifier")
print(f"  classifier_activation: {config.classifier_activation} - Activation for classifier")
print()

# Additional Parameters
print("ADDITIONAL PARAMETERS:")
print(f"  sparse_pred_ignore_index: {config.sparse_pred_ignore_index} - Index to ignore for sparse prediction")
print(f"  reference_compile: {getattr(config, 'reference_compile', 'Not set')} - Whether to compile layers")
print(f"  repad_logits_with_grad: {getattr(config, 'repad_logits_with_grad', False)} - Track gradients when repadding")
print()

# Model Type and Architecture
print("MODEL TYPE AND ARCHITECTURE:")
print(f"  model_type: {config.model_type}")
print(f"  architectures: {config.architectures}")
print(f"  torch_dtype: {config.torch_dtype}")
print(f"  transformers_version: {config.transformers_version}")
print()

# All attributes (for completeness)
print("ALL CONFIG ATTRIBUTES:")
print("-" * 50)
all_attrs = sorted([attr for attr in dir(config) if not attr.startswith('_')])
for attr in all_attrs:
    if hasattr(config, attr) and not callable(getattr(config, attr)):
        print(f"  {attr}: {getattr(config, attr)}")