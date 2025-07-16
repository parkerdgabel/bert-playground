#!/usr/bin/env python3
import argparse
from pathlib import Path

from training.trainer import create_trainer
from kaggle.titanic.submission import create_submission


def main():
    parser = argparse.ArgumentParser(description="Train ModernBERT on Titanic dataset")
    
    # Data arguments
    parser.add_argument(
        "--train_path",
        type=str,
        default="kaggle/titanic/data/train.csv",
        help="Path to training data"
    )
    parser.add_argument(
        "--val_path",
        type=str,
        default=None,
        help="Path to validation data (optional)"
    )
    parser.add_argument(
        "--test_path",
        type=str,
        default="kaggle/titanic/data/test.csv",
        help="Path to test data for submission"
    )
    
    # Model arguments
    parser.add_argument(
        "--model_name",
        type=str,
        default="answerdotai/ModernBERT-base",
        help="Pretrained model name from HuggingFace"
    )
    
    # Training arguments
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Training batch size"
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=2e-5,
        help="Learning rate"
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=3,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./output",
        help="Directory to save checkpoints"
    )
    
    # Action arguments
    parser.add_argument(
        "--do_train",
        action="store_true",
        help="Run training"
    )
    parser.add_argument(
        "--do_predict",
        action="store_true",
        help="Generate predictions"
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default=None,
        help="Path to checkpoint for prediction"
    )
    
    args = parser.parse_args()
    
    if args.do_train:
        print("Starting training...")
        print(f"Model: {args.model_name}")
        print(f"Batch size: {args.batch_size}")
        print(f"Learning rate: {args.learning_rate}")
        print(f"Epochs: {args.num_epochs}")
        
        # Create trainer
        trainer = create_trainer(
            train_path=args.train_path,
            val_path=args.val_path,
            model_name=args.model_name,
            learning_rate=args.learning_rate,
            batch_size=args.batch_size,
            num_epochs=args.num_epochs,
            output_dir=args.output_dir
        )
        
        # Train model
        trainer.train(args.num_epochs)
        
        # Set checkpoint path for prediction
        if args.do_predict and not args.checkpoint_path:
            args.checkpoint_path = str(Path(args.output_dir) / "best_model")
    
    if args.do_predict:
        if not args.checkpoint_path:
            raise ValueError("Must provide --checkpoint_path for prediction")
        
        print(f"\nGenerating predictions from {args.checkpoint_path}...")
        
        # Create submission
        create_submission(
            test_path=args.test_path,
            checkpoint_path=args.checkpoint_path,
            output_path="submission.csv"
        )


if __name__ == "__main__":
    main()