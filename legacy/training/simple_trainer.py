"""Simplified trainer for debugging MLX issues."""
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from typing import Dict, Tuple
import time
from tqdm import tqdm

from models.modernbert_mlx import create_model
from models.classification_head import TitanicClassifier
from data.titanic_loader import TitanicDataPipeline


def loss_fn(model, inputs, labels):
    """Compute loss for the model."""
    outputs = model(
        input_ids=inputs['input_ids'],
        attention_mask=inputs['attention_mask'],
        labels=labels
    )
    return outputs['loss']


def eval_fn(model, inputs):
    """Forward pass without loss."""
    return model(
        input_ids=inputs['input_ids'],
        attention_mask=inputs['attention_mask']
    )


def train_simple(
    train_path: str = "kaggle/titanic/data/train.csv",
    num_epochs: int = 1,
    batch_size: int = 2,
    learning_rate: float = 1e-5
):
    print("Creating simple trainer...")
    
    # Create data loader
    train_loader = TitanicDataPipeline(
        data_path=train_path,
        batch_size=batch_size,
        is_training=True,
        augment=False  # Disable augmentation for simplicity
    )
    
    # Create model
    bert_model = create_model("answerdotai/ModernBERT-base")
    model = TitanicClassifier(bert_model)
    
    # Initialize optimizer
    optimizer = optim.SGD(learning_rate=learning_rate)
    
    # Training loop
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        
        total_loss = 0
        num_batches = 0
        
        # Create data generator
        data_gen = train_loader.get_dataloader()()
        
        for batch_idx, batch in enumerate(data_gen):
            if batch_idx >= train_loader.get_num_batches():
                break
                
            # Prepare inputs
            inputs = {
                'input_ids': batch['input_ids'],
                'attention_mask': batch['attention_mask']
            }
            labels = batch['labels'].squeeze()
            
            # Create loss function that captures batch data
            def batch_loss_fn(params):
                # Apply parameters to model
                model.update(params)
                return loss_fn(model, inputs, labels)
            
            # Get current parameters
            params = model.parameters()
            
            # Compute loss and gradients
            loss_value, grads = mx.value_and_grad(batch_loss_fn)(params)
            
            # Update parameters
            optimizer.update(model, grads)
            
            # Compute accuracy
            outputs = eval_fn(model, inputs)
            predictions = mx.argmax(outputs['logits'], axis=-1)
            accuracy = mx.mean(predictions == labels)
            
            total_loss += float(loss_value)
            num_batches += 1
            
            print(f"Batch {batch_idx + 1}: Loss = {float(loss_value):.4f}, Accuracy = {float(accuracy):.4f}")
        
        avg_loss = total_loss / num_batches
        print(f"Epoch {epoch + 1} - Average Loss: {avg_loss:.4f}")
    
    print("\nTraining completed!")
    return model


if __name__ == "__main__":
    # Run simple training
    model = train_simple(num_epochs=1, batch_size=2)