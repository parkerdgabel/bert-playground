#!/usr/bin/env python3
"""
Simple demonstration of BERT enhancements for Kaggle competitions.

This script shows the key features in action:
1. Multi-stage training (frozen → partial → full fine-tuning)
2. Layer-wise learning rates
3. Text augmentation
4. Ensemble predictions
"""

from bert_playground.data.augmentation import BERTTextAugmenter
from bert_playground.models.bert.utils import BERTLayerManager
from bert_playground.models.ensemble import BERTEnsembleConfig, BERTEnsembleModel
from bert_playground.models.factory import create_model
from bert_playground.training.strategies.multi_stage import (
    BERTTrainingStrategy,
    MultiStageBERTTrainer,
    TrainingStage,
)
from loguru import logger


def demonstrate_multi_stage_training():
    """Show multi-stage BERT training in action."""
    logger.info("\n=== Multi-Stage BERT Training Demo ===")

    # Create a BERT model
    model = create_model(
        "modernbert_with_head", head_type="binary_classification", num_labels=2
    )

    # Configure multi-stage strategy
    strategy = BERTTrainingStrategy(
        frozen_epochs=1,  # Train only head for 1 epoch
        partial_unfreeze_epochs=2,  # Unfreeze top 4 layers for 2 epochs
        full_finetune_epochs=2,  # Fine-tune all layers for 2 epochs
        num_layers_to_unfreeze=4,
        head_lr_multiplier=10.0,  # Head learns 10x faster
        layer_lr_decay=0.95,  # Each layer gets 5% lower LR
    )

    # Create multi-stage trainer
    trainer = MultiStageBERTTrainer(model, strategy)

    # Simulate training stages
    for epoch in range(5):
        stage = trainer.get_current_stage(epoch)
        stage_info = trainer.get_stage_info()

        logger.info(f"\nEpoch {epoch + 1}:")
        logger.info(f"  Stage: {stage.value}")
        logger.info(f"  Trainable layers: {stage_info['trainable_layers']}")
        logger.info(f"  Total layers: {stage_info['total_layers']}")

        # Show which parts are trainable
        if stage == TrainingStage.FROZEN_BERT:
            logger.info("  → Training only the classification head")
        elif stage == TrainingStage.PARTIAL_UNFREEZE:
            logger.info("  → Training head + top 4 BERT layers")
        else:
            logger.info("  → Training entire model")


def demonstrate_layer_wise_learning_rates():
    """Show layer-wise learning rate scheduling."""
    logger.info("\n=== Layer-wise Learning Rates Demo ===")

    # Create model and layer manager
    model = create_model("bert_with_head", head_type="binary_classification")
    layer_manager = BERTLayerManager(model)

    # Get parameter groups with layer-wise LRs
    param_groups = layer_manager.get_parameter_groups(
        base_lr=2e-5, layer_lr_decay=0.95, head_lr_multiplier=10.0
    )

    # Display learning rates
    logger.info("\nLearning rates by layer:")
    for group in param_groups[:5]:  # Show first 5 groups
        logger.info(f"  {group['name']}: LR = {group['lr']:.2e}")

    if len(param_groups) > 5:
        logger.info(f"  ... and {len(param_groups) - 5} more groups")


def demonstrate_text_augmentation():
    """Show text augmentation capabilities."""
    logger.info("\n=== Text Augmentation Demo ===")

    augmenter = BERTTextAugmenter()

    # Example text
    text = "The passenger was a 35-year-old male traveling in first class."

    logger.info(f"\nOriginal text: {text}")
    logger.info("\nAugmentation methods:")

    # Token masking
    masked = augmenter.mask_tokens([text], mask_prob=0.15)
    logger.info(f"  Token masking: {masked[0]}")

    # Synonym replacement (simulated)
    logger.info(
        "  Synonym replacement: The traveler was a 35-year-old gentleman traveling in first class."
    )

    # Sentence shuffling for longer texts
    logger.info("  Sentence shuffling: (for multi-sentence texts)")

    # Back-translation (simulated)
    logger.info(
        "  Back-translation: A 35-year-old man was traveling in the first class."
    )


def demonstrate_ensemble():
    """Show ensemble model capabilities."""
    logger.info("\n=== BERT Ensemble Demo ===")

    # Configure ensemble
    config = BERTEnsembleConfig(
        model_types=["bert"],
        random_seeds=[42, 123, 456],  # 3 different initializations
        dropout_rates=[0.1, 0.2],  # 2 dropout variations
        ensemble_method="weighted_average",
    )

    ensemble = BERTEnsembleModel(config)

    # Create diverse models
    logger.info("\nCreating ensemble models:")
    models = ensemble.create_diverse_models(
        base_config={}, head_type="binary_classification", num_labels=2
    )

    logger.info(f"\nEnsemble contains {len(models)} models:")
    for i, config in enumerate(ensemble.model_configs):
        logger.info(
            f"  Model {i + 1}: seed={config['seed']}, dropout={config['dropout']}"
        )

    logger.info("\nEnsemble will combine predictions using weighted averaging")
    logger.info("Weights can be updated based on CV performance")


def main():
    """Run all demonstrations."""
    logger.info("BERT Enhancements for Kaggle Competitions")
    logger.info("=" * 50)

    demonstrate_multi_stage_training()
    demonstrate_layer_wise_learning_rates()
    demonstrate_text_augmentation()
    demonstrate_ensemble()

    logger.info("\n" + "=" * 50)
    logger.info("Summary of Enhancements:")
    logger.info("=" * 50)
    logger.info("""
1. Multi-stage Training:
   - Gradually unfreeze BERT layers for stable training
   - Prevents catastrophic forgetting
   - Allows aggressive learning rates for task head

2. Layer-wise Learning Rates:
   - Lower layers (general features) learn slower
   - Higher layers (task-specific) learn faster
   - Classification head learns fastest

3. Text Augmentation:
   - Token masking for robustness
   - Synonym replacement for diversity
   - Back-translation for paraphrasing
   - Sentence shuffling for longer texts

4. Ensemble Methods:
   - Multiple random seeds for diversity
   - Different dropout rates
   - Weighted averaging based on CV performance
   - Checkpoint averaging for single models

These enhancements work together to improve BERT performance on Kaggle competitions!
""")


if __name__ == "__main__":
    main()
