"""
MLX Embeddings Configuration

Configuration utilities for mlx-embeddings integration, including model mappings
and default configurations.
"""

from typing import Dict, Optional
from dataclasses import dataclass, field


# Mapping of HuggingFace model names to mlx-community equivalents
MODEL_NAME_MAPPING = {
    # ModernBERT models
    "answerdotai/ModernBERT-base": "mlx-community/answerdotai-ModernBERT-base-4bit",
    "answerdotai/ModernBERT-large": "mlx-community/answerdotai-ModernBERT-large-4bit",
    "nomic-ai/modernbert-embed-base": "mlx-community/nomicai-modernbert-embed-base-4bit",
    "tasksource/ModernBERT-base-embed": "mlx-community/tasksource-ModernBERT-base-embed-4bit",
    
    # Standard BERT models
    "bert-base-uncased": "mlx-community/bert-base-uncased-4bit",
    "bert-base-cased": "mlx-community/bert-base-cased-4bit",
    "bert-large-uncased": "mlx-community/bert-large-uncased-4bit",
    
    # Sentence transformers
    "sentence-transformers/all-MiniLM-L6-v2": "mlx-community/all-MiniLM-L6-v2-4bit",
    "sentence-transformers/all-MiniLM-L12-v2": "mlx-community/all-MiniLM-L12-v2-4bit",
    "sentence-transformers/all-mpnet-base-v2": "mlx-community/all-mpnet-base-v2-4bit",
    
    # RoBERTa models
    "roberta-base": "mlx-community/roberta-base-4bit",
    "roberta-large": "mlx-community/roberta-large-4bit",
}


@dataclass
class MLXEmbeddingsConfig:
    """Configuration for MLX embeddings integration."""
    
    # Model configuration
    model_name: str = "answerdotai/ModernBERT-base"
    use_mlx_embeddings: bool = True
    backend: str = "auto"  # "auto", "mlx", "huggingface"
    
    # Model architecture
    hidden_size: int = 768
    num_hidden_layers: int = 12
    num_attention_heads: int = 12
    intermediate_size: int = 3072
    max_position_embeddings: int = 8192
    
    # Tokenizer configuration
    vocab_size: int = 50368
    type_vocab_size: int = 2
    pad_token_id: int = 0
    mask_token_id: int = 103
    
    # Training configuration
    pooling_strategy: str = "mean"  # "mean", "cls", "max"
    normalize_embeddings: bool = True
    freeze_embeddings: bool = False
    dropout_rate: float = 0.1
    
    # Performance configuration
    use_4bit: bool = True
    batch_size: int = 32
    max_length: int = 512
    
    # Cache configuration
    cache_dir: Optional[str] = None
    
    # Additional model kwargs
    model_kwargs: Dict = field(default_factory=dict)
    
    def get_mlx_model_name(self) -> str:
        """Get the MLX community model name."""
        # Check if already an mlx-community model
        if self.model_name.startswith("mlx-community/"):
            return self.model_name
        
        # Try to map from HuggingFace name
        return MODEL_NAME_MAPPING.get(self.model_name, self.model_name)
    
    def to_dict(self) -> Dict:
        """Convert config to dictionary."""
        return {
            "model_name": self.model_name,
            "use_mlx_embeddings": self.use_mlx_embeddings,
            "backend": self.backend,
            "hidden_size": self.hidden_size,
            "num_hidden_layers": self.num_hidden_layers,
            "num_attention_heads": self.num_attention_heads,
            "intermediate_size": self.intermediate_size,
            "max_position_embeddings": self.max_position_embeddings,
            "vocab_size": self.vocab_size,
            "type_vocab_size": self.type_vocab_size,
            "pad_token_id": self.pad_token_id,
            "mask_token_id": self.mask_token_id,
            "pooling_strategy": self.pooling_strategy,
            "normalize_embeddings": self.normalize_embeddings,
            "freeze_embeddings": self.freeze_embeddings,
            "dropout_rate": self.dropout_rate,
            "use_4bit": self.use_4bit,
            "batch_size": self.batch_size,
            "max_length": self.max_length,
            "cache_dir": self.cache_dir,
            "model_kwargs": self.model_kwargs,
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict) -> "MLXEmbeddingsConfig":
        """Create config from dictionary."""
        return cls(**config_dict)
    
    @classmethod
    def from_model_name(cls, model_name: str, **kwargs) -> "MLXEmbeddingsConfig":
        """
        Create config from model name with appropriate defaults.
        
        Args:
            model_name: Model name (HuggingFace or mlx-community)
            **kwargs: Additional configuration overrides
            
        Returns:
            MLXEmbeddingsConfig instance
        """
        config = cls(model_name=model_name)
        
        # Set appropriate defaults based on model
        if "large" in model_name.lower():
            config.hidden_size = 1024
            config.num_hidden_layers = 24
            config.num_attention_heads = 16
            config.intermediate_size = 4096
        elif "base" in model_name.lower():
            config.hidden_size = 768
            config.num_hidden_layers = 12
            config.num_attention_heads = 12
            config.intermediate_size = 3072
        elif "small" in model_name.lower() or "mini" in model_name.lower():
            config.hidden_size = 384
            config.num_hidden_layers = 6
            config.num_attention_heads = 12
            config.intermediate_size = 1536
        
        # Apply overrides
        for key, value in kwargs.items():
            if hasattr(config, key):
                setattr(config, key, value)
        
        return config


# Default configurations for common models
DEFAULT_CONFIGS = {
    "modernbert-base": MLXEmbeddingsConfig(
        model_name="answerdotai/ModernBERT-base",
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        max_position_embeddings=8192,
    ),
    "modernbert-large": MLXEmbeddingsConfig(
        model_name="answerdotai/ModernBERT-large",
        hidden_size=1024,
        num_hidden_layers=24,
        num_attention_heads=16,
        max_position_embeddings=8192,
    ),
    "bert-base": MLXEmbeddingsConfig(
        model_name="bert-base-uncased",
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        max_position_embeddings=512,
    ),
    "minilm": MLXEmbeddingsConfig(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        hidden_size=384,
        num_hidden_layers=6,
        num_attention_heads=12,
        max_position_embeddings=512,
    ),
}


def get_default_config(model_type: str) -> MLXEmbeddingsConfig:
    """Get default configuration for a model type."""
    if model_type in DEFAULT_CONFIGS:
        return DEFAULT_CONFIGS[model_type]
    
    # Try to infer from model type
    if "modernbert" in model_type.lower():
        if "large" in model_type.lower():
            return DEFAULT_CONFIGS["modernbert-large"]
        return DEFAULT_CONFIGS["modernbert-base"]
    elif "bert" in model_type.lower():
        return DEFAULT_CONFIGS["bert-base"]
    elif "minilm" in model_type.lower():
        return DEFAULT_CONFIGS["minilm"]
    
    # Return base config as default
    return MLXEmbeddingsConfig()