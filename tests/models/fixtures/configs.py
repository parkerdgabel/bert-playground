"""Configuration fixtures for model testing."""

import json
from pathlib import Path
from typing import Any

from models.bert.config import BertConfig
from models.bert.modernbert_config import ModernBertConfig
from models.heads.config import ClassificationConfig, RegressionConfig
from models.lora.config import LoRAConfig


def create_bert_config(
    vocab_size: int = 30522,
    hidden_size: int = 768,
    num_hidden_layers: int = 12,
    num_attention_heads: int = 12,
    **kwargs,
) -> BertConfig:
    """Create BERT configuration with sensible defaults."""
    config_dict = {
        "vocab_size": vocab_size,
        "hidden_size": hidden_size,
        "num_hidden_layers": num_hidden_layers,
        "num_attention_heads": num_attention_heads,
        "intermediate_size": kwargs.get("intermediate_size", hidden_size * 4),
        "hidden_act": kwargs.get("hidden_act", "gelu"),
        "hidden_dropout_prob": kwargs.get("hidden_dropout_prob", 0.1),
        "attention_probs_dropout_prob": kwargs.get("attention_probs_dropout_prob", 0.1),
        "max_position_embeddings": kwargs.get("max_position_embeddings", 512),
        "type_vocab_size": kwargs.get("type_vocab_size", 2),
        "initializer_range": kwargs.get("initializer_range", 0.02),
        "layer_norm_eps": kwargs.get("layer_norm_eps", 1e-12),
    }

    # Override with any additional kwargs
    for key, value in kwargs.items():
        if key not in config_dict:
            config_dict[key] = value

    return BertConfig(**config_dict)


def create_small_bert_config(**kwargs) -> BertConfig:
    """Create small BERT configuration for fast testing."""
    return create_bert_config(
        vocab_size=1000,
        hidden_size=128,
        num_hidden_layers=2,
        num_attention_heads=2,
        intermediate_size=256,
        max_position_embeddings=128,
        **kwargs,
    )


def create_tiny_bert_config(**kwargs) -> BertConfig:
    """Create tiny BERT configuration for unit tests."""
    return create_bert_config(
        vocab_size=100,
        hidden_size=32,
        num_hidden_layers=1,
        num_attention_heads=1,
        intermediate_size=64,
        max_position_embeddings=32,
        **kwargs,
    )


def create_modernbert_config(
    vocab_size: int = 30522,
    hidden_size: int = 768,
    num_hidden_layers: int = 12,
    num_attention_heads: int = 12,
    **kwargs,
) -> ModernBertConfig:
    """Create ModernBERT configuration with sensible defaults."""
    config_dict = {
        "vocab_size": vocab_size,
        "hidden_size": hidden_size,
        "num_hidden_layers": num_hidden_layers,
        "num_attention_heads": num_attention_heads,
        "intermediate_size": kwargs.get("intermediate_size", hidden_size * 4),
        "hidden_act": kwargs.get("hidden_act", "gelu"),
        "hidden_dropout_prob": kwargs.get(
            "hidden_dropout_prob", 0.0
        ),  # ModernBERT default
        "attention_probs_dropout_prob": kwargs.get("attention_probs_dropout_prob", 0.0),
        "max_position_embeddings": kwargs.get("max_position_embeddings", 8192),
        "type_vocab_size": kwargs.get("type_vocab_size", 2),
        "initializer_range": kwargs.get("initializer_range", 0.02),
        "layer_norm_eps": kwargs.get("layer_norm_eps", 1e-5),
        "rope_base": kwargs.get("rope_base", 10000.0),
        "attention_bias": kwargs.get("attention_bias", False),
        "deterministic_flash_attn": kwargs.get("deterministic_flash_attn", True),
        "sliding_window": kwargs.get("sliding_window"),
        "global_attn_every_n_layers": kwargs.get("global_attn_every_n_layers", 1),
    }

    return ModernBertConfig(**config_dict)


def create_small_modernbert_config(**kwargs) -> ModernBertConfig:
    """Create small ModernBERT configuration for fast testing."""
    return create_modernbert_config(
        vocab_size=1000,
        hidden_size=128,
        num_hidden_layers=2,
        num_attention_heads=2,
        intermediate_size=256,
        max_position_embeddings=256,
        **kwargs,
    )


def create_classification_config(
    hidden_size: int = 768,
    num_labels: int = 2,
    dropout_prob: float = 0.1,
    pooling_type: str = "cls",
    **kwargs,
) -> ClassificationConfig:
    """Create classification head configuration."""
    config_dict = {
        "input_size": hidden_size,
        "output_size": num_labels,
        "num_classes": num_labels,
        "dropout_prob": dropout_prob,
        "pooling_type": pooling_type,
        "activation": kwargs.get("hidden_act", "tanh"),
        "head_type": "classification",
    }

    return ClassificationConfig(**config_dict)


def create_regression_config(
    hidden_size: int = 768,
    dropout_prob: float = 0.1,
    pooling_type: str = "mean",
    **kwargs,
) -> RegressionConfig:
    """Create regression head configuration."""
    config_dict = {
        "input_size": hidden_size,
        "output_size": kwargs.get("output_dim", 1),
        "dropout_prob": dropout_prob,
        "pooling_type": pooling_type,
        "activation": kwargs.get("hidden_act", "tanh"),
        "head_type": "regression",
    }

    return RegressionConfig(**config_dict)


def create_lora_config(
    rank: int = 8,
    alpha: int = 16,
    dropout: float = 0.1,
    target_modules: list | None = None,
    **kwargs,
) -> LoRAConfig:
    """Create LoRA adapter configuration."""
    if target_modules is None:
        target_modules = ["query", "value"]

    config_dict = {
        "rank": rank,
        "alpha": alpha,
        "dropout": dropout,
        "target_modules": target_modules,
        "modules_to_save": kwargs.get("modules_to_save", ["classifier"]),
        "inference_mode": kwargs.get("inference_mode", False),
    }

    return LoRAConfig(**config_dict)


def create_model_config_variations() -> dict[str, Any]:
    """Create various model configuration variations for testing."""
    return {
        # BERT variations
        "bert_base": create_bert_config(),
        "bert_small": create_small_bert_config(),
        "bert_tiny": create_tiny_bert_config(),
        "bert_large": create_bert_config(
            hidden_size=1024,
            num_hidden_layers=24,
            num_attention_heads=16,
        ),
        # ModernBERT variations
        "modernbert_base": create_modernbert_config(),
        "modernbert_small": create_small_modernbert_config(),
        "modernbert_long_context": create_modernbert_config(
            max_position_embeddings=16384,
            sliding_window=512,
        ),
        # Head variations
        "binary_classification": create_classification_config(num_labels=2),
        "multiclass_classification": create_classification_config(num_labels=10),
        "regression": create_regression_config(),
        "regression_multidim": create_regression_config(output_dim=5),
        # LoRA variations
        "lora_small": create_lora_config(rank=4, alpha=8),
        "lora_standard": create_lora_config(rank=8, alpha=16),
        "lora_large": create_lora_config(rank=16, alpha=32),
        "lora_all_layers": create_lora_config(
            target_modules=["query", "key", "value", "output"]
        ),
    }


def create_invalid_bert_config() -> dict[str, Any]:
    """Create invalid BERT configuration for error testing."""
    return {
        "vocab_size": -100,  # Invalid negative vocab size
        "hidden_size": 0,  # Invalid zero hidden size
        "num_hidden_layers": -1,  # Invalid negative layers
        "num_attention_heads": 3,  # Not divisible by hidden_size
        "hidden_dropout_prob": 1.5,  # Invalid probability > 1
    }


def create_edge_case_configs() -> dict[str, BertConfig]:
    """Create edge case configurations for testing."""
    return {
        "single_layer": create_bert_config(num_hidden_layers=1),
        "single_head": create_bert_config(
            hidden_size=64,
            num_attention_heads=1,
        ),
        "no_dropout": create_bert_config(
            hidden_dropout_prob=0.0,
            attention_probs_dropout_prob=0.0,
        ),
        "max_dropout": create_bert_config(
            hidden_dropout_prob=0.9,
            attention_probs_dropout_prob=0.9,
        ),
        "short_context": create_bert_config(max_position_embeddings=32),
        "long_context": create_bert_config(max_position_embeddings=4096),
    }


def save_config_to_file(config: Any, file_path: Path, format: str = "json"):
    """Save model configuration to file for testing."""
    file_path.parent.mkdir(parents=True, exist_ok=True)

    if hasattr(config, "to_dict"):
        config_dict = config.to_dict()
    else:
        config_dict = dict(config)

    if format == "json":
        with open(file_path, "w") as f:
            json.dump(config_dict, f, indent=2)
    else:
        raise ValueError(f"Unknown format: {format}")


def load_config_from_file(file_path: Path, config_class: type) -> Any:
    """Load model configuration from file."""
    with open(file_path) as f:
        config_dict = json.load(f)

    return config_class(**config_dict)
