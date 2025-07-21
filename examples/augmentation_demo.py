#!/usr/bin/env python3
"""
Demonstration of the data augmentation module.

This script shows how to use the generic augmentation framework
with any dataset for BERT training.
"""

import mlx.core as mx

# Import augmentation components
from bert_playground.data.augmentation import (
    # Configuration
    AugmentationConfig,
    # Manager
    AugmentationManager,
    AugmentationMode,
    BERTTextAugmenter,
    FeatureMetadata,
    FeatureType,
    # Augmenters
    TabularAugmenter,
    TabularToTextAugmenter,
    get_registry,
)
from loguru import logger


def demonstrate_basic_augmentation():
    """Show basic feature augmentation."""
    logger.info("\n=== Basic Feature Augmentation Demo ===")

    # Define feature metadata
    feature_metadata = {
        "age": FeatureMetadata(
            name="age",
            feature_type=FeatureType.NUMERICAL,
            statistics={"mean": 30, "std": 15, "min": 0, "max": 100},
        ),
        "income": FeatureMetadata(
            name="income",
            feature_type=FeatureType.NUMERICAL,
            statistics={"mean": 50000, "std": 20000},
        ),
        "category": FeatureMetadata(
            name="category",
            feature_type=FeatureType.CATEGORICAL,
            domain_info={"values": ["A", "B", "C"]},
        ),
        "description": FeatureMetadata(
            name="description", feature_type=FeatureType.TEXT
        ),
    }

    # Create augmentation config
    config = AugmentationConfig.from_mode(
        AugmentationMode.MODERATE, feature_metadata=feature_metadata
    )

    # Create augmenter
    augmenter = TabularAugmenter(config, feature_metadata)

    # Example data
    sample = {
        "age": 25,
        "income": 45000,
        "category": "B",
        "description": "This is a sample description",
    }

    logger.info(f"Original sample: {sample}")

    # Augment multiple times to see variations
    for i in range(3):
        augmented = augmenter.augment(sample)
        logger.info(f"Augmented {i + 1}: {augmented}")


def demonstrate_tabular_to_text_augmentation():
    """Show augmentation for tabular-to-text conversion."""
    logger.info("\n=== Tabular-to-Text Augmentation Demo ===")

    # Feature metadata for a hypothetical customer dataset
    feature_metadata = {
        "customer_age": FeatureMetadata(
            name="customer_age",
            feature_type=FeatureType.NUMERICAL,
            importance=0.8,
            statistics={"mean": 35, "std": 12},
        ),
        "purchase_amount": FeatureMetadata(
            name="purchase_amount",
            feature_type=FeatureType.NUMERICAL,
            importance=0.9,
            statistics={"mean": 100, "std": 50},
        ),
        "membership": FeatureMetadata(
            name="membership", feature_type=FeatureType.CATEGORICAL, importance=0.6
        ),
        "last_visit": FeatureMetadata(name="last_visit", feature_type=FeatureType.DATE),
    }

    # Create config with light augmentation
    config = AugmentationConfig.from_mode(AugmentationMode.LIGHT)

    # Custom text converter
    def customer_text_converter(features):
        age = features.get("customer_age", "unknown")
        amount = features.get("purchase_amount", 0)
        membership = features.get("membership", "standard")

        return f"A {age}-year-old {membership} member made a ${amount:.2f} purchase"

    # Create augmenter
    augmenter = TabularToTextAugmenter(
        config=config,
        feature_metadata=feature_metadata,
        text_converter=customer_text_converter,
    )

    # Example customer data
    customer = {
        "customer_age": 42,
        "purchase_amount": 125.50,
        "membership": "premium",
        "last_visit": "2024-01-15",
    }

    logger.info(f"Original customer: {customer}")

    # Generate multiple augmented texts
    texts = augmenter.augment_multiple(customer, num_augmentations=5)
    for i, text in enumerate(texts):
        logger.info(f"Text {i + 1}: {text}")


def demonstrate_augmentation_manager():
    """Show the augmentation manager in action."""
    logger.info("\n=== Augmentation Manager Demo ===")

    # Feature metadata for a competition dataset
    feature_metadata = {
        "feature1": FeatureMetadata("feature1", FeatureType.NUMERICAL),
        "feature2": FeatureMetadata("feature2", FeatureType.NUMERICAL),
        "feature3": FeatureMetadata("feature3", FeatureType.CATEGORICAL),
        "feature4": FeatureMetadata("feature4", FeatureType.TEXT),
    }

    # Create manager
    manager = AugmentationManager(
        feature_metadata=feature_metadata,
        config=AugmentationConfig.from_mode(AugmentationMode.MODERATE),
    )

    # Get augmentation statistics
    stats = manager.get_augmentation_stats()
    logger.info(f"Augmentation stats: {stats}")

    # Batch of samples
    batch = [
        {"feature1": 10, "feature2": 20, "feature3": "cat_a", "feature4": "text one"},
        {"feature1": 15, "feature2": 25, "feature3": "cat_b", "feature4": "text two"},
        {"feature1": 20, "feature2": 30, "feature3": "cat_c", "feature4": "text three"},
    ]

    # Augment batch
    augmented_batch = manager.augment_batch(batch)

    logger.info("\nBatch augmentation results:")
    for i, (orig, aug) in enumerate(zip(batch, augmented_batch, strict=False)):
        logger.info(f"Sample {i + 1}:")
        logger.info(f"  Original: {orig}")
        logger.info(f"  Augmented: {aug}")


def demonstrate_bert_specific_augmentation():
    """Show BERT-specific text augmentation."""
    logger.info("\n=== BERT-Specific Augmentation Demo ===")

    # Mock tokenizer for demo
    class MockTokenizer:
        def __init__(self):
            self.vocab_size = 30000
            self.mask_token = "[MASK]"
            self.mask_token_id = 103
            self.pad_token_id = 0
            self.cls_token_id = 101
            self.sep_token_id = 102

        def tokenize(self, text):
            return text.split()

        def convert_tokens_to_string(self, tokens):
            return " ".join(tokens)

        def convert_ids_to_tokens(self, ids):
            return [f"token_{id}" for id in ids]

    tokenizer = MockTokenizer()

    # Create BERT text augmenter
    augmenter = BERTTextAugmenter(tokenizer)

    # Example text
    text = "The quick brown fox jumps over the lazy dog"
    logger.info(f"Original text: {text}")

    # Generate augmented versions
    augmented_texts = augmenter.augment_text(text, num_augmentations=5)
    for i, aug_text in enumerate(augmented_texts):
        logger.info(f"Augmented {i + 1}: {aug_text}")


def demonstrate_custom_strategies():
    """Show how to use custom augmentation strategies."""
    logger.info("\n=== Custom Augmentation Strategies Demo ===")

    # Get the registry
    registry = get_registry()

    # List available strategies
    logger.info(f"Available strategies: {registry.list_strategies()}")

    # Create custom strategy
    from bert_playground.data.augmentation import BaseAugmentationStrategy

    class CustomNoiseStrategy(BaseAugmentationStrategy):
        def __init__(self):
            super().__init__("custom_noise", [FeatureType.NUMERICAL])

        def apply(self, data, config):
            if isinstance(data, (int, float)):
                # Add custom noise pattern
                noise = mx.random.uniform(-0.05, 0.05) * data
                return float(data + noise.item())
            return data

    # Register custom strategy
    registry.register_strategy("custom_noise", CustomNoiseStrategy())

    # Use in augmentation
    sample = {"value": 100.0}
    strategy = registry.get_strategy("custom_noise")
    augmented = strategy.apply(sample["value"], {})

    logger.info(f"Original value: {sample['value']}")
    logger.info(f"With custom noise: {augmented}")


def main():
    """Run all demonstrations."""
    logger.info("Data Augmentation Module Demonstration")
    logger.info("=" * 50)

    # Set random seed for reproducibility
    mx.random.seed(42)

    demonstrate_basic_augmentation()
    demonstrate_tabular_to_text_augmentation()
    demonstrate_augmentation_manager()
    demonstrate_bert_specific_augmentation()
    demonstrate_custom_strategies()

    logger.info("\n" + "=" * 50)
    logger.info("Summary:")
    logger.info("- Generic augmentation works with any dataset")
    logger.info("- Feature-specific strategies based on data types")
    logger.info("- Easy integration with BERT models")
    logger.info("- Extensible with custom strategies")
    logger.info("- MLX-native operations for performance")


if __name__ == "__main__":
    main()
