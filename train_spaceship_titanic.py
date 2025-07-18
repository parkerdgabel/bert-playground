#!/usr/bin/env python3
"""
Training script for Spaceship Titanic competition.

This script trains a BERT-based classifier to predict whether passengers
were transported to an alternate dimension.
"""

import os
import sys
from pathlib import Path
from typing import Dict, Optional
import yaml
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from loguru import logger
import pandas as pd
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from models.classification.spaceship_classifier import (
    create_spaceship_classifier,
    create_ensemble_spaceship_classifier,
    create_advanced_spaceship_classifier
)
from data.spaceship_loader import create_spaceship_dataloaders
from embeddings.tokenizer_wrapper import TokenizerWrapper
from training.trainer_v2 import EnhancedTrainer
from utils.mlflow_utils import setup_mlflow_tracking
from utils.kaggle_integration import KaggleIntegration


def load_config(config_path: str) -> Dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def create_model(config: Dict):
    """Create model based on configuration."""
    model_config = config['model']
    
    # Check if ensemble is enabled
    if config.get('ensemble', {}).get('enabled', False):
        logger.info("Creating ensemble classifier")
        model = create_ensemble_spaceship_classifier(
            model_name=model_config['name'],
            num_heads=config['ensemble']['num_heads'],
            ensemble_method=config['ensemble']['ensemble_method'],
            base_hidden_dim=model_config['hidden_dim'],
            freeze_embeddings=model_config.get('freeze_embeddings', False),
            label_smoothing=model_config.get('label_smoothing', 0.0)
        )
    # Check if multi-task is enabled
    elif config.get('multi_task', {}).get('enabled', False):
        logger.info("Creating multi-task classifier")
        model = create_advanced_spaceship_classifier(
            model_name=model_config['name'],
            use_auxiliary_tasks=True,
            freeze_embeddings=model_config.get('freeze_embeddings', False),
            label_smoothing=model_config.get('label_smoothing', 0.0)
        )
    else:
        logger.info("Creating standard classifier")
        model = create_spaceship_classifier(
            model_name=model_config['name'],
            hidden_dim=model_config['hidden_dim'],
            dropout_rate=model_config['dropout_rate'],
            pooling_type=model_config['pooling_type'],
            activation=model_config['activation'],
            use_layer_norm=model_config['use_layer_norm'],
            freeze_embeddings=model_config.get('freeze_embeddings', False),
            label_smoothing=model_config.get('label_smoothing', 0.0)
        )
    
    return model


def create_optimizer(model, config: Dict):
    """Create optimizer based on configuration."""
    train_config = config['training']
    
    optimizer = optim.AdamW(
        learning_rate=train_config['learning_rate'],
        betas=[train_config['adam_beta1'], train_config['adam_beta2']],
        eps=train_config['adam_epsilon'],
        weight_decay=train_config['weight_decay']
    )
    
    return optimizer


def generate_predictions(model, test_loader, tokenizer, config: Dict):
    """Generate predictions for test set."""
    logger.info("Generating predictions for test set...")
    
    model.eval()
    predictions = []
    passenger_ids = []
    
    for batch in test_loader:
        # Get predictions
        with mx.no_grad():
            if hasattr(model, 'get_transported_probabilities'):
                probs = model.get_transported_probabilities(
                    batch['input_ids'],
                    batch['attention_mask']
                )
            else:
                # Generic classifier
                probs = model.predict_proba(
                    batch['input_ids'],
                    batch['attention_mask']
                )
                probs = probs[:, 1]  # Get transported probability
        
        predictions.extend(probs.tolist())
        
        # Get passenger IDs if available
        if 'passenger_id' in batch:
            passenger_ids.extend(batch['passenger_id'])
    
    # Create submission DataFrame
    if not passenger_ids:
        # Load test data to get passenger IDs
        test_df = pd.read_csv(config['data']['test_file'])
        passenger_ids = test_df['PassengerId'].tolist()
    
    # Apply threshold
    threshold = config.get('evaluation', {}).get('threshold', 0.5)
    binary_predictions = [p > threshold for p in predictions]
    
    submission_df = pd.DataFrame({
        'PassengerId': passenger_ids[:len(predictions)],
        'Transported': binary_predictions
    })
    
    return submission_df, predictions


def main(config_path: str = "configs/spaceship_titanic.yaml"):
    """Main training function."""
    # Load configuration
    config = load_config(config_path)
    
    # Set random seed
    mx.random.seed(config.get('seed', 42))
    
    # Setup logging
    logger.remove()
    log_level = config.get('logging', {}).get('level', 'INFO')
    logger.add(sys.stderr, level=log_level)
    
    if config.get('logging', {}).get('log_to_file', False):
        log_file = config['logging'].get('log_file', 'training.log')
        logger.add(log_file, level=log_level)
    
    # Setup MLflow
    if config.get('mlflow', {}).get('enabled', True):
        mlflow_config = config['mlflow']
        setup_mlflow_tracking(
            experiment_name=mlflow_config['experiment_name'],
            tracking_uri=mlflow_config.get('tracking_uri', './mlruns')
        )
        
        import mlflow
        mlflow.start_run(run_name=mlflow_config.get('run_name'))
        mlflow.log_params({
            'model_name': config['model']['name'],
            'task_type': config['model']['task_type'],
            'pooling_type': config['model']['pooling_type'],
            'hidden_dim': config['model']['hidden_dim'],
            'batch_size': config['training']['batch_size'],
            'learning_rate': config['training']['learning_rate'],
            'num_epochs': config['training']['num_epochs'],
        })
        
        # Log tags
        for tag_key, tag_value in mlflow_config.get('tags', {}).items():
            mlflow.set_tag(tag_key, tag_value)
    
    # Create tokenizer
    logger.info("Initializing tokenizer...")
    tokenizer = TokenizerWrapper(
        model_name=config['model']['name'],
        backend=config['data'].get('tokenizer_backend', 'mlx')
    )
    
    # Create data loaders
    logger.info("Creating data loaders...")
    data_config = config['data']
    loaders = create_spaceship_dataloaders(
        train_path=data_config['train_file'],
        test_path=data_config.get('test_file'),
        tokenizer=tokenizer,
        batch_size=config['training']['batch_size'],
        max_length=data_config['max_sequence_length'],
        augment_train=data_config.get('augmentation', {}).get('enabled', False),
        n_augmentations=data_config.get('augmentation', {}).get('n_augmentations', 2),
        num_workers=data_config.get('num_workers', 4),
        prefetch_size=data_config.get('prefetch_size', 4),
        val_split=data_config.get('val_split', 0.2),
        cache_dir=data_config.get('cache_dir')
    )
    
    # Create model
    logger.info("Creating model...")
    model = create_model(config)
    logger.info(f"Model has {model.get_num_trainable_params():,} trainable parameters")
    
    # Create optimizer
    optimizer = create_optimizer(model, config)
    
    # Create trainer
    logger.info("Creating trainer...")
    trainer = EnhancedTrainer(
        model=model,
        optimizer=optimizer,
        config=config
    )
    
    # Train model
    logger.info("Starting training...")
    trainer.train(
        train_loader=loaders['train'],
        val_loader=loaders['val'],
        num_epochs=config['training']['num_epochs']
    )
    
    # Generate predictions if test data available
    if 'test' in loaders and config.get('submission', {}).get('generate', True):
        submission_df, probabilities = generate_predictions(
            model, loaders['test'], tokenizer, config
        )
        
        # Save submission
        submission_dir = Path(config['submission']['output_dir'])
        submission_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        submission_path = submission_dir / f"submission_{timestamp}.csv"
        submission_df.to_csv(submission_path, index=False)
        logger.info(f"Saved submission to {submission_path}")
        
        # Log to MLflow
        if mlflow.active_run():
            mlflow.log_artifact(str(submission_path))
            mlflow.log_metric("test_predictions_mean", float(mx.mean(mx.array(probabilities))))
    
    # Submit to Kaggle if configured
    if config.get('submission', {}).get('auto_submit', False):
        kaggle = KaggleIntegration()
        
        # Get validation score from trainer
        val_score = trainer.best_val_score
        
        # Format submission message
        message_template = config['submission'].get(
            'submission_message_template',
            'MLX-BERT submission'
        )
        message = message_template.format(
            model_type=config['model']['task_type'],
            val_accuracy=val_score
        )
        
        # Submit
        result = kaggle.submit_predictions(
            competition_id='spaceship-titanic',
            submission_file=submission_path,
            message=message
        )
        
        logger.info(f"Submitted to Kaggle: {result}")
    
    # End MLflow run
    if mlflow.active_run():
        mlflow.end_run()
    
    logger.info("Training completed!")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train Spaceship Titanic classifier")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/spaceship_titanic.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--quick-test",
        action="store_true",
        help="Run quick test with minimal epochs"
    )
    
    args = parser.parse_args()
    
    # Override config for quick test
    if args.quick_test:
        config = load_config(args.config)
        if 'quick_test' in config:
            # Merge quick test settings
            config['training'].update(config['quick_test'].get('training', {}))
            config['data'].update(config['quick_test'].get('data', {}))
            config['ensemble'] = config['quick_test'].get('ensemble', config.get('ensemble', {}))
            config['multi_task'] = config['quick_test'].get('multi_task', config.get('multi_task', {}))
    
    main(args.config)