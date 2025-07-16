#!/usr/bin/env python3
"""Production training runner with optimal hyperparameters for Titanic dataset."""

import argparse
import subprocess
import sys
from pathlib import Path
from datetime import datetime


CONFIGS = {
    "quick": {
        "description": "Quick training for testing (1 epoch, small batch)",
        "batch_size": 16,
        "num_epochs": 1,
        "learning_rate": 2e-5,
        "warmup_steps": 50,
        "eval_steps": 50,
    },
    "standard": {
        "description": "Standard training (5 epochs, balanced settings)",
        "batch_size": 32,
        "num_epochs": 5,
        "learning_rate": 2e-5,
        "warmup_steps": 100,
        "eval_steps": 50,
    },
    "thorough": {
        "description": "Thorough training (10 epochs, smaller batch for stability)",
        "batch_size": 16,
        "num_epochs": 10,
        "learning_rate": 1e-5,
        "warmup_steps": 200,
        "eval_steps": 100,
    },
    "experimental": {
        "description": "Experimental settings (15 epochs, very small LR)",
        "batch_size": 8,
        "num_epochs": 15,
        "learning_rate": 5e-6,
        "warmup_steps": 300,
        "eval_steps": 150,
    }
}


def main():
    parser = argparse.ArgumentParser(description="Production training runner for Titanic ModernBERT")
    parser.add_argument(
        "--config",
        choices=list(CONFIGS.keys()),
        default="standard",
        help="Training configuration to use"
    )
    parser.add_argument(
        "--experiment-name",
        default="titanic_modernbert",
        help="MLflow experiment name"
    )
    parser.add_argument(
        "--enable-mlflow",
        action="store_true",
        help="Enable MLflow tracking"
    )
    parser.add_argument(
        "--predict",
        action="store_true",
        help="Run prediction after training"
    )
    args = parser.parse_args()
    
    config = CONFIGS[args.config]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"./output/{args.config}_{timestamp}"
    
    print(f"\n{'='*60}")
    print(f"Production Training: {args.config.upper()} Configuration")
    print(f"{'='*60}")
    print(f"Description: {config['description']}")
    print(f"Dataset: Titanic (891 training samples, ~2673 with augmentation)")
    print(f"Configuration:")
    for key, value in config.items():
        if key != "description":
            print(f"  {key}: {value}")
    print(f"Output directory: {output_dir}")
    print(f"MLflow tracking: {'Enabled' if args.enable_mlflow else 'Disabled'}")
    print(f"{'='*60}\n")
    
    # Calculate expected training time
    samples_with_aug = 2673
    steps_per_epoch = samples_with_aug // config["batch_size"]
    total_steps = steps_per_epoch * config["num_epochs"]
    print(f"Expected training steps: {total_steps} ({steps_per_epoch} per epoch)")
    print(f"Estimated time: {total_steps * 0.5:.1f} - {total_steps * 1.5:.1f} seconds\n")
    
    # Build command
    cmd = [
        "uv", "run", "python", "train_titanic_v2.py",
        "--train_path", "data/titanic/train.csv",
        "--val_path", "data/titanic/val.csv",
        "--model_name", "answerdotai/ModernBERT-base",
        "--batch_size", str(config["batch_size"]),
        "--learning_rate", str(config["learning_rate"]),
        "--num_epochs", str(config["num_epochs"]),
        "--output_dir", output_dir,
        "--experiment_name", args.experiment_name,
        "--run_name", f"{args.config}_{timestamp}",
        "--log_level", "INFO",
        "--do_train"
    ]
    
    if not args.enable_mlflow:
        cmd.append("--disable_mlflow")
    
    # Run training
    print("Starting training...")
    result = subprocess.run(cmd, capture_output=False, text=True)
    
    if result.returncode != 0:
        print(f"\nTraining failed with exit code {result.returncode}")
        sys.exit(1)
    
    print("\nTraining completed successfully!")
    
    # Run prediction if requested
    if args.predict:
        print("\nRunning predictions on test set...")
        pred_cmd = [
            "uv", "run", "python", "train_titanic_v2.py",
            "--test_path", "data/titanic/test.csv",
            "--checkpoint_path", f"{output_dir}/best_model_accuracy",
            "--output_dir", output_dir,
            "--do_predict"
        ]
        
        pred_result = subprocess.run(pred_cmd, capture_output=False, text=True)
        
        if pred_result.returncode == 0:
            print(f"\nPredictions saved to: {output_dir}/submission.csv")
        else:
            print("\nPrediction failed!")
    
    # Print summary
    history_path = Path(output_dir) / "training_history.json"
    if history_path.exists():
        import json
        with open(history_path) as f:
            history = json.load(f)
        
        print(f"\n{'='*60}")
        print("Training Summary")
        print(f"{'='*60}")
        
        if history["train_loss"]:
            print(f"Final train loss: {history['train_loss'][-1]:.4f}")
            print(f"Final train accuracy: {history['train_accuracy'][-1]:.4f}")
        
        if history["val_loss"]:
            best_val_loss = min(history["val_loss"])
            best_val_acc = max(history["val_accuracy"])
            best_val_auc = max(history["val_auc"]) if history["val_auc"] else 0
            
            print(f"\nBest validation metrics:")
            print(f"  Loss: {best_val_loss:.4f}")
            print(f"  Accuracy: {best_val_acc:.4f}")
            print(f"  AUC: {best_val_auc:.4f}")
        
        print(f"\nOutput directory: {output_dir}")
        print(f"{'='*60}")


if __name__ == "__main__":
    main()