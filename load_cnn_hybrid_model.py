#!/usr/bin/env python3
"""Correctly load the CNN hybrid model with the proper hidden size."""

import mlx.core as mx
from pathlib import Path
import json
from models.modernbert_cnn_hybrid import CNNEnhancedModernBERT, CNNHybridConfig
from loguru import logger


def load_cnn_hybrid_checkpoint(checkpoint_path: str) -> CNNEnhancedModernBERT:
    """Load a CNN hybrid model from checkpoint, fixing config issues."""

    checkpoint_path = Path(checkpoint_path)

    # Load config
    config_path = checkpoint_path / "config.json"
    with open(config_path, "r") as f:
        config = json.load(f)

    logger.info(f"Original config hidden_size: {config['hidden_size']}")

    # Fix the hidden_size mismatch
    # The model was actually trained with hidden_size=768, not 512
    config["hidden_size"] = 768

    logger.info(f"Corrected hidden_size to: {config['hidden_size']}")

    # Create model with corrected config
    model_config = CNNHybridConfig(**config)
    model = CNNEnhancedModernBERT(model_config)

    # Load weights
    model_path = checkpoint_path / "model.safetensors"
    logger.info(f"Loading weights from: {model_path}")

    try:
        model.load_weights(str(model_path))
        logger.success("✅ Weights loaded successfully!")
    except Exception as e:
        logger.error(f"Failed to load weights: {e}")
        raise

    return model


def verify_model(model: CNNEnhancedModernBERT):
    """Verify the loaded model structure."""
    logger.info("\nVerifying model structure:")

    # Check parameters
    params = model.parameters()

    def count_params(d):
        """Count parameters in nested dict."""
        count = 0
        for k, v in d.items():
            if isinstance(v, dict):
                count += count_params(v)
            elif isinstance(v, mx.array):
                count += v.size
        return count

    total_params = count_params(params)
    logger.info(f"Total parameters: {total_params:,}")

    # Test forward pass
    logger.info("\nTesting forward pass...")
    batch_size = 2
    seq_length = 128

    # Create dummy input
    input_ids = mx.random.randint(0, 50000, (batch_size, seq_length))
    attention_mask = mx.ones((batch_size, seq_length))

    try:
        output = model(input_ids, attention_mask)
        if isinstance(output, dict):
            logger.info(f"Output is a dictionary with keys: {list(output.keys())}")
            if "logits" in output:
                logger.info(f"Logits shape: {output['logits'].shape}")
        elif hasattr(output, "shape"):
            logger.info(f"Output shape: {output.shape}")
        else:
            logger.info(f"Output type: {type(output)}")
        logger.success("✅ Forward pass successful!")
    except Exception as e:
        logger.error(f"Forward pass failed: {e}")
        raise


def main():
    """Main function."""
    checkpoint_path = "output/cnn_hybrid_lion/checkpoint-1675"

    logger.info(f"Loading CNN hybrid model from: {checkpoint_path}")

    # Load the model
    model = load_cnn_hybrid_checkpoint(checkpoint_path)

    # Verify it works
    verify_model(model)

    # Show usage example
    print("\n" + "=" * 60)
    print("Example usage in your code:")
    print("=" * 60)
    print("""
from load_cnn_hybrid_model import load_cnn_hybrid_checkpoint

# Load the model
model = load_cnn_hybrid_checkpoint('output/cnn_hybrid_lion/checkpoint-1675')

# Use for inference
output = model(input_ids, attention_mask)
predictions = mx.argmax(output, axis=-1)
""")


if __name__ == "__main__":
    main()
