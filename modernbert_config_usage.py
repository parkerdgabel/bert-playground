"""
ModernBERT Configuration Usage Examples
======================================

This script demonstrates how to use and customize ModernBERT configuration.
"""

from transformers import AutoConfig, AutoModel

# Example 1: Load default configuration
print("Example 1: Loading default ModernBERT configuration")
print("-" * 50)
config = AutoConfig.from_pretrained("answerdotai/ModernBERT-base")
print(f"Default hidden size: {config.hidden_size}")
print(f"Default num layers: {config.num_hidden_layers}")
print(f"Default max sequence length: {config.max_position_embeddings}")
print()

# Example 2: Create custom configuration
print("Example 2: Creating custom ModernBERT configuration")
print("-" * 50)
custom_config = AutoConfig.from_pretrained("answerdotai/ModernBERT-base")

# Modify configuration for a smaller model
custom_config.num_hidden_layers = 12  # Reduce layers from 22 to 12
custom_config.hidden_size = 512  # Reduce hidden size from 768 to 512
custom_config.intermediate_size = 768  # Reduce intermediate size
custom_config.num_attention_heads = 8  # Reduce attention heads
custom_config.max_position_embeddings = 4096  # Reduce max sequence length

print(f"Custom hidden size: {custom_config.hidden_size}")
print(f"Custom num layers: {custom_config.num_hidden_layers}")
print(f"Custom max sequence length: {custom_config.max_position_embeddings}")
print()

# Example 3: Key configuration attributes for different use cases
print("Example 3: Configuration for different use cases")
print("-" * 50)

# For long-context processing
print("Long-context configuration:")
long_context_config = AutoConfig.from_pretrained("answerdotai/ModernBERT-base")
print(f"  Max position embeddings: {long_context_config.max_position_embeddings}")
print(f"  Local attention window: {long_context_config.local_attention}")
print(f"  Global attention every N layers: {long_context_config.global_attn_every_n_layers}")
print(f"  Global RoPE theta: {long_context_config.global_rope_theta}")
print(f"  Local RoPE theta: {long_context_config.local_rope_theta}")
print()

# For efficient inference
print("Efficient inference configuration:")
print(f"  Attention dropout: {config.attention_dropout}")
print(f"  Embedding dropout: {config.embedding_dropout}")
print(f"  MLP dropout: {config.mlp_dropout}")
print(f"  Attention bias: {config.attention_bias}")
print(f"  MLP bias: {config.mlp_bias}")
print()

# Example 4: Access all configuration as dictionary
print("Example 4: Configuration as dictionary")
print("-" * 50)
config_dict = config.to_dict()
print(f"Total configuration parameters: {len(config_dict)}")
print("First 10 parameters:")
for i, (key, value) in enumerate(config_dict.items()):
    if i < 10:
        print(f"  {key}: {value}")
print()

# Example 5: Important attributes for fine-tuning
print("Example 5: Key attributes for fine-tuning")
print("-" * 50)
print(f"Vocabulary size: {config.vocab_size}")
print(f"Hidden activation: {config.hidden_activation}")
print(f"Classifier activation: {config.classifier_activation}")
print(f"Classifier dropout: {config.classifier_dropout}")
print(f"Initializer range: {config.initializer_range}")
print(f"Norm epsilon: {config.norm_eps}")
print()

# Example 6: Special tokens
print("Example 6: Special token IDs")
print("-" * 50)
print(f"PAD token ID: {config.pad_token_id}")
print(f"BOS token ID: {config.bos_token_id}")
print(f"EOS token ID: {config.eos_token_id}")
print()

# Note about model instantiation
print("Note: To instantiate a model with custom config:")
print("model = AutoModel.from_config(custom_config)")
print("This creates a randomly initialized model with your configuration.")