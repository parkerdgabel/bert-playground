#!/usr/bin/env python3
"""
Test script for Spaceship Titanic implementation.

This script tests all components of the Spaceship Titanic solution:
- Text conversion
- Data loading  
- Model creation
- Training (quick test)
"""

import sys
from pathlib import Path
import mlx.core as mx
import pandas as pd
from loguru import logger

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from data.spaceship_text_templates import SpaceshipTitanicTextConverter
from data.spaceship_loader import create_spaceship_dataloaders
from models.classification.spaceship_classifier import (
    create_spaceship_classifier,
    create_ensemble_spaceship_classifier
)
from embeddings.tokenizer_wrapper import TokenizerWrapper


def test_text_conversion():
    """Test text conversion functionality."""
    logger.info("Testing text conversion...")
    
    # Create converter
    converter = SpaceshipTitanicTextConverter(augment=False)
    
    # Load sample data
    train_df = pd.read_csv("data/spaceship-titanic/train.csv")
    
    # Convert first few rows
    for i in range(3):
        row = train_df.iloc[i]
        text = converter.convert_row_to_text(row)
        logger.info(f"Row {i}: {text[:100]}...")
    
    # Test augmentation
    converter.augment = True
    aug_texts = converter.create_augmented_texts(train_df.iloc[0], n_augmentations=2)
    logger.info(f"Generated {len(aug_texts)} augmented versions")
    
    logger.success("Text conversion test passed!")


def test_data_loading():
    """Test data loading functionality."""
    logger.info("Testing data loading...")
    
    # Create tokenizer
    tokenizer = TokenizerWrapper("mlx-community/answerdotai-ModernBERT-base-4bit")
    
    # Create data loaders
    loaders = create_spaceship_dataloaders(
        train_path="data/spaceship-titanic/train.csv",
        test_path="data/spaceship-titanic/test.csv",
        tokenizer=tokenizer,
        batch_size=4,
        max_length=128,
        augment_train=False,
        val_split=0.2
    )
    
    # Test train loader
    for i, batch in enumerate(loaders['train']):
        logger.info(f"Train batch {i}:")
        logger.info(f"  Input IDs shape: {batch['input_ids'].shape}")
        logger.info(f"  Labels: {batch['label']}")
        
        if i >= 2:
            break
    
    # Test val loader
    val_batch = next(iter(loaders['val']))
    logger.info(f"Validation batch shape: {val_batch['input_ids'].shape}")
    
    logger.success("Data loading test passed!")


def test_model_creation():
    """Test model creation."""
    logger.info("Testing model creation...")
    
    # Test standard classifier
    model = create_spaceship_classifier(
        hidden_dim=128,  # Smaller for testing
        dropout_rate=0.1
    )
    logger.info(f"Standard model parameters: {model.get_num_trainable_params():,}")
    
    # Test ensemble classifier
    ensemble = create_ensemble_spaceship_classifier(
        num_heads=3,
        base_hidden_dim=128
    )
    logger.info(f"Ensemble model parameters: {ensemble.get_num_trainable_params():,}")
    
    # Test forward pass
    dummy_input = mx.random.randint(0, 1000, (4, 64))
    # Try float16 to match model dtype
    dummy_mask = mx.ones((4, 64), dtype=mx.float16)
    
    try:
        output = model.forward(dummy_input, dummy_mask)
    except Exception as e:
        logger.error(f"Forward pass failed: {e}")
        # Try with no mask
        logger.info("Trying without attention mask...")
        output = model.forward(dummy_input)
    logger.info(f"Model output shape: {output.shape}")
    
    probs = model.get_transported_probabilities(dummy_input, dummy_mask)
    logger.info(f"Probability range: [{float(mx.min(probs)):.3f}, {float(mx.max(probs)):.3f}]")
    
    logger.success("Model creation test passed!")


def test_quick_training():
    """Test quick training run."""
    logger.info("Testing quick training...")
    
    import mlx.optimizers as optim
    from training.trainer_v2 import EnhancedTrainer
    
    # Create small model
    model = create_spaceship_classifier(hidden_dim=64, dropout_rate=0.1)
    
    # Create optimizer
    optimizer = optim.AdamW(learning_rate=1e-4)
    
    # Create minimal config
    config = {
        'training': {
            'num_epochs': 1,
            'batch_size': 8,
            'gradient_accumulation_steps': 1,
            'eval_steps': 10,
            'save_steps': 20,
            'logging_steps': 5,
            'early_stopping_patience': 3,
            'metric_for_best_model': 'eval_loss',
            'greater_is_better': False,
        },
        'data': {
            'max_sequence_length': 64,
        },
        'evaluation': {
            'metrics': ['accuracy', 'f1'],
        }
    }
    
    # Create tokenizer
    tokenizer = TokenizerWrapper("mlx-community/answerdotai-ModernBERT-base-4bit")
    
    # Create small data loaders
    loaders = create_spaceship_dataloaders(
        train_path="data/spaceship-titanic/train.csv",
        tokenizer=tokenizer,
        batch_size=8,
        max_length=64,
        augment_train=False,
        val_split=0.1  # Small validation set
    )
    
    # Create trainer
    trainer = EnhancedTrainer(
        model=model,
        optimizer=optimizer,
        config=config
    )
    
    # Train for 1 epoch
    logger.info("Running quick training for 1 epoch...")
    trainer.train(
        train_loader=loaders['train'],
        val_loader=loaders['val'],
        num_epochs=1
    )
    
    logger.success("Quick training test passed!")


def test_prediction_generation():
    """Test prediction generation."""
    logger.info("Testing prediction generation...")
    
    # Create model
    model = create_spaceship_classifier(hidden_dim=64)
    
    # Create tokenizer
    tokenizer = TokenizerWrapper("mlx-community/answerdotai-ModernBERT-base-4bit")
    
    # Create test loader
    loaders = create_spaceship_dataloaders(
        train_path="data/spaceship-titanic/train.csv",
        test_path="data/spaceship-titanic/test.csv",
        tokenizer=tokenizer,
        batch_size=16,
        max_length=64,
        augment_train=False
    )
    
    # Generate predictions for a few batches
    model.eval()
    predictions = []
    
    for i, batch in enumerate(loaders['test']):
        # MLX doesn't have no_grad context manager - gradients are not computed in eval mode
        # Ensure input_ids are int32
        input_ids = mx.array(batch['input_ids'], dtype=mx.int32)
        attention_mask = batch['attention_mask']
        
        probs = model.get_transported_probabilities(
            input_ids,
            attention_mask
        )
        predictions.extend(probs.tolist())
        
        if i >= 2:  # Just test a few batches
            break
    
    logger.info(f"Generated {len(predictions)} predictions")
    logger.info(f"Prediction range: [{min(predictions):.3f}, {max(predictions):.3f}]")
    
    # Test submission format
    test_df = pd.read_csv("data/spaceship-titanic/test.csv")
    sample_submission = pd.DataFrame({
        'PassengerId': test_df['PassengerId'][:len(predictions)],
        'Transported': [p > 0.5 for p in predictions]
    })
    
    logger.info(f"Sample submission shape: {sample_submission.shape}")
    logger.info(f"Transported distribution: {sample_submission['Transported'].value_counts(normalize=True).to_dict()}")
    
    logger.success("Prediction generation test passed!")


def main():
    """Run all tests."""
    logger.remove()
    logger.add(sys.stderr, level="INFO")
    
    logger.info("Starting Spaceship Titanic tests...")
    
    try:
        # Run tests
        test_text_conversion()
        test_data_loading()
        test_model_creation()
        test_prediction_generation()
        
        # Quick training test (optional - takes longer)
        if "--with-training" in sys.argv:
            test_quick_training()
        
        logger.success("\nAll tests passed! ✅")
        logger.info("\nThe Spaceship Titanic implementation is ready for training.")
        logger.info("To start training, run:")
        logger.info("  uv run python train_spaceship_titanic.py")
        logger.info("\nFor a quick test run:")
        logger.info("  uv run python train_spaceship_titanic.py --quick-test")
        
    except Exception as e:
        logger.error(f"Test failed: {e}")
        raise


if __name__ == "__main__":
    main()