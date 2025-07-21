#!/usr/bin/env python3
"""
Example script showcasing all BERT enhancements for Kaggle competitions.

This script demonstrates:
1. Multi-stage BERT training with gradual unfreezing
2. Layer-wise learning rate scheduling
3. BERT ensemble training
4. Text augmentation for tabular data
5. Adversarial validation
6. Pseudo-labeling with confidence scoring
7. Test-time augmentation (TTA)
8. Cross-validation with OOF predictions
"""

import argparse
from pathlib import Path

# Import our enhanced components
from bert_playground.cli.utils.model_setup import create_model_and_tokenizer
from bert_playground.data.kaggle.titanic import TitanicDataset
from bert_playground.data.loaders import create_data_loader
from bert_playground.data.templates import EnhancedTabularToTextConverter
from bert_playground.training.core.config import (
    CheckpointConfig,
    OptimizerConfig,
    TrainingConfig,
)
from bert_playground.training.kaggle import (
    KaggleTrainer,
    KaggleTrainerConfig,
    get_competition_config,
)
from loguru import logger


def main():
    parser = argparse.ArgumentParser(
        description="BERT training for Kaggle competitions"
    )
    parser.add_argument(
        "--competition", type=str, default="titanic", help="Competition name"
    )
    parser.add_argument(
        "--train-csv", type=str, required=True, help="Path to training CSV"
    )
    parser.add_argument("--test-csv", type=str, help="Path to test CSV")
    parser.add_argument(
        "--model-type", type=str, default="modernbert", choices=["bert", "modernbert"]
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--cv-folds", type=int, default=5, help="Number of CV folds")
    parser.add_argument(
        "--enable-ensemble", action="store_true", help="Enable model ensemble"
    )
    parser.add_argument(
        "--enable-pseudo-labeling", action="store_true", help="Enable pseudo-labeling"
    )
    parser.add_argument(
        "--enable-tta", action="store_true", help="Enable test-time augmentation"
    )
    parser.add_argument("--output-dir", type=str, default="output/kaggle_bert")
    args = parser.parse_args()

    # Setup logging
    logger.info(f"Starting BERT training for {args.competition} competition")

    # 1. Create model and tokenizer
    logger.info(f"Creating {args.model_type} model...")
    model, tokenizer = create_model_and_tokenizer(
        model_name=f"{args.model_type}_with_head",
        head_type="binary_classification",  # For Titanic
        num_labels=2,
        use_pretrained=False,  # Start from scratch for demo
    )

    # 2. Load and prepare data with enhanced text templates
    logger.info("Loading data with enhanced text templates...")

    # Create text converter with multiple template styles
    text_converter = EnhancedTabularToTextConverter(
        template_style="analytical",  # Options: narrative, analytical, comparative, qa
        include_statistics=True,
        use_feature_descriptions=True,
    )

    # Create datasets
    train_dataset = TitanicDataset(
        csv_path=args.train_csv,
        tokenizer=tokenizer,
        max_length=256,
        is_training=True,
        text_converter=text_converter,
    )

    test_dataset = None
    if args.test_csv:
        test_dataset = TitanicDataset(
            csv_path=args.test_csv,
            tokenizer=tokenizer,
            max_length=256,
            is_training=False,
            text_converter=text_converter,
        )

    # Create data loaders
    train_loader = create_data_loader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        prefetch_size=4,
    )

    test_loader = None
    if test_dataset:
        test_loader = create_data_loader(
            test_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=4,
            prefetch_size=4,
        )

    # 3. Configure training with all enhancements
    logger.info("Configuring Kaggle trainer with BERT enhancements...")

    # Get competition-specific config
    kaggle_config = get_competition_config(args.competition)
    kaggle_config.cv_folds = args.cv_folds
    kaggle_config.enable_ensemble = args.enable_ensemble
    kaggle_config.enable_tta = args.enable_tta
    kaggle_config.tta_iterations = 5
    kaggle_config.pseudo_labeling_iterations = 3 if args.enable_pseudo_labeling else 0

    # Create full training config
    config = KaggleTrainerConfig(
        training=TrainingConfig(
            num_epochs=args.epochs,
            batch_size=args.batch_size,
            gradient_accumulation_steps=1,
            mixed_precision=True,
            gradient_clip_val=1.0,
            log_interval=10,
            eval_interval=50,
            save_interval=100,
            save_total_limit=3,
            use_compilation=True,  # MLX compilation
        ),
        optimizer=OptimizerConfig(
            type="adamw",
            learning_rate=2e-5,
            weight_decay=0.01,
            betas=[0.9, 0.999],
            eps=1e-8,
        ),
        checkpoint=CheckpointConfig(
            enable_checkpointing=True,
            save_best_model=True,
            save_best_only=True,  # Save space
            best_model_metric="val_accuracy",
            best_model_mode="max",
        ),
        kaggle=kaggle_config,
        environment={
            "output_dir": Path(args.output_dir),
            "run_name": f"{args.competition}_{args.model_type}_enhanced",
        },
    )

    # 4. Initialize trainer with BERT strategies
    trainer = KaggleTrainer(
        model=model,
        config=config,
        test_dataloader=test_loader,
        enable_bert_strategies=True,  # Enable all BERT enhancements
    )

    # 5. Perform adversarial validation if test data available
    if test_loader and args.enable_pseudo_labeling:
        logger.info("Running adversarial validation...")
        trainer.validate_with_adversarial(train_loader, test_loader)

    # 6. Train with cross-validation and multi-stage BERT training
    logger.info("Starting enhanced BERT training...")
    result = trainer.train(train_loader)

    # 7. Apply pseudo-labeling if enabled and test data available
    if args.enable_pseudo_labeling and test_loader:
        logger.info("Applying pseudo-labeling to test data...")
        augmented_loader = trainer.apply_pseudo_labeling(test_loader)

        # Retrain with augmented data
        logger.info("Retraining with pseudo-labeled data...")
        trainer.train(augmented_loader)

    # 8. Generate final predictions with ensemble and TTA
    if test_loader:
        logger.info("Generating final predictions...")
        predictions = trainer.generate_test_predictions()

        # Create submission
        submission_path = trainer.create_submission(
            submission_name=f"{args.competition}_bert_enhanced"
        )
        logger.info(f"Submission saved to: {submission_path}")

    # 9. Display final results
    logger.info("\n" + "=" * 50)
    logger.info("Training Complete!")
    logger.info("=" * 50)

    if hasattr(result, "cv_mean"):
        logger.info(f"CV Score: {result['cv_mean']:.4f} ± {result['cv_std']:.4f}")

    logger.info("\nFeatures used:")
    logger.info("✓ Multi-stage BERT training with gradual unfreezing")
    logger.info("✓ Layer-wise learning rate scheduling")
    if args.enable_ensemble:
        logger.info("✓ BERT ensemble with multiple models")
    logger.info("✓ Enhanced tabular-to-text templates")
    if args.enable_pseudo_labeling:
        logger.info("✓ Pseudo-labeling with BERT confidence scoring")
    if args.enable_tta:
        logger.info("✓ Test-time augmentation")
    logger.info("✓ Cross-validation with OOF predictions")

    logger.info("\nCheck the output directory for:")
    logger.info(f"- Model checkpoints: {args.output_dir}/checkpoints/")
    logger.info(f"- OOF predictions: {args.output_dir}/submissions/oof_predictions.npy")
    logger.info(
        f"- Test predictions: {args.output_dir}/submissions/test_predictions.npy"
    )
    logger.info(f"- Submission file: {args.output_dir}/submissions/")


if __name__ == "__main__":
    main()
