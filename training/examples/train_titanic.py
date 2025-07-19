"""
Example: Training a model for Titanic competition using the new declarative training module.
"""

from pathlib import Path
import mlx.core as mx
import mlx.nn as nn

# Import from our new training module
from training import create_trainer, KaggleTrainerConfig
from training.kaggle import CompetitionType
from models import create_model
from data import create_dataloader


def main():
    """Train a model for the Titanic competition."""
    
    # 1. Create model
    model = create_model(
        model_type="bert_with_head",
        head_type="binary_classification",
        num_labels=2,
    )
    
    # 2. Create configuration
    config = KaggleTrainerConfig(
        # Kaggle settings
        kaggle={
            "competition_name": "titanic",
            "competition_type": CompetitionType.BINARY_CLASSIFICATION,
            "competition_metric": "accuracy",
            "cv_folds": 5,
            "enable_ensemble": True,
            "enable_tta": True,
            "auto_submit": False,  # Set to True to auto-submit
        },
        # Optimizer settings
        optimizer={
            "type": "adamw",
            "learning_rate": 2e-5,
            "weight_decay": 0.01,
        },
        # Scheduler settings
        scheduler={
            "type": "cosine",
            "warmup_ratio": 0.1,
        },
        # Data settings
        data={
            "batch_size": 32,
            "num_workers": 8,
            "augment_train": True,
        },
        # Training settings
        training={
            "num_epochs": 10,
            "eval_strategy": "epoch",
            "save_best_only": True,
            "early_stopping": True,
            "early_stopping_patience": 3,
            "report_to": ["mlflow", "tensorboard"],
        },
        # Environment settings
        environment={
            "output_dir": Path("output/titanic"),
            "experiment_name": "titanic-modernbert",
            "seed": 42,
        },
    )
    
    # 3. Create data loaders
    train_loader = create_dataloader(
        dataset_name="titanic",
        split="train",
        batch_size=config.data.batch_size,
        shuffle=True,
    )
    
    test_loader = create_dataloader(
        dataset_name="titanic",
        split="test",
        batch_size=config.data.eval_batch_size,
        shuffle=False,
    )
    
    # 4. Create trainer
    trainer = create_trainer(
        model=model,
        trainer_type="kaggle",
        config=config,
        test_dataloader=test_loader,
    )
    
    # 5. Train with cross-validation
    cv_results = trainer.train_with_cv(
        train_dataloader=train_loader,
    )
    
    print(f"\nCV Score: {cv_results['cv_mean']:.4f} ± {cv_results['cv_std']:.4f}")
    
    # 6. Create submission
    submission_path = trainer.create_submission()
    print(f"Created submission: {submission_path}")
    
    # 7. Optional: Submit to Kaggle (if auto_submit is False)
    # trainer.submit_to_kaggle(submission_path)


def train_with_config_file():
    """Example using YAML configuration file."""
    
    # Create model
    model = create_model("titanic-bert")
    
    # Create data loaders
    train_loader = create_dataloader("titanic", "train")
    test_loader = create_dataloader("titanic", "test")
    
    # Create trainer from YAML config
    trainer = create_trainer(
        model=model,
        config="configs/titanic.yaml",
        trainer_type="kaggle",
        test_dataloader=test_loader,
    )
    
    # Train
    trainer.train_with_cv(train_loader)


def quick_test():
    """Quick test with minimal configuration."""
    
    # Use factory convenience function
    from training.factory import create_kaggle_trainer
    
    # Create model
    model = create_model("bert-binary", num_labels=2)
    
    # Create data loaders
    train_loader = create_dataloader("titanic", "train", batch_size=8)
    test_loader = create_dataloader("titanic", "test", batch_size=8)
    
    # Create trainer with preset
    trainer = create_kaggle_trainer(
        model=model,
        competition="titanic",
        test_dataloader=test_loader,
        # Override some settings
        num_epochs=1,
        cv_folds=2,
    )
    
    # Quick training
    trainer.train_with_cv(train_loader)


if __name__ == "__main__":
    # Run the main training
    main()
    
    # Or run quick test
    # quick_test()
    
    # Or use config file
    # train_with_config_file()