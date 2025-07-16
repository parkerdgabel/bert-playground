#!/usr/bin/env python3
"""Quick test script to verify training works."""

import sys
sys.path.append('.')

from training.trainer_v2 import create_trainer_v2

def main():
    print("Creating trainer with minimal configuration...")
    
    # Create trainer with very small configuration
    trainer = create_trainer_v2(
        train_path="data/titanic/train.csv",
        val_path="data/titanic/val.csv",
        model_name="answerdotai/ModernBERT-base",
        learning_rate=2e-5,
        batch_size=8,  # Small batch size
        num_epochs=1,
        output_dir="./output_test",
        experiment_name="titanic_test",
        enable_mlflow=False  # Disable MLflow for speed
    )
    
    # Override max steps for quick test
    trainer.max_steps = 10
    trainer.eval_steps = 5
    trainer.save_steps = 10
    
    print("\nStarting minimal training (10 steps)...")
    trainer.train(num_epochs=1)
    
    print("\nTraining complete! Check output_test directory for results.")
    
    # Print final metrics
    if trainer.training_history['train_loss']:
        print(f"\nFinal training loss: {trainer.training_history['train_loss'][-1]:.4f}")
        print(f"Final training accuracy: {trainer.training_history['train_accuracy'][-1]:.4f}")
    
    if trainer.training_history['val_loss']:
        print(f"Final validation loss: {trainer.training_history['val_loss'][-1]:.4f}")
        print(f"Final validation accuracy: {trainer.training_history['val_accuracy'][-1]:.4f}")


if __name__ == "__main__":
    main()