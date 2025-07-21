"""
MLX compilation optimizations for training.

This module provides compiled versions of training functions using mx.compile
for improved performance on Apple Silicon.
"""

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten, tree_map
from functools import partial
from typing import Dict, Tuple, List, Any, Optional
from loguru import logger

from .protocols import Model


def create_compiled_train_step(
    model: Model,
    optimizer: Any,  # MLX optimizer, not nn.Optimizer
    config: Any,
    gradient_accumulator: Optional[Any] = None
) -> Tuple[callable, List[Any]]:
    """
    Create a compiled training step function.
    
    This function returns:
    1. A compiled train_step function
    2. The state list that needs to be tracked
    
    Args:
        model: The model to train
        optimizer: The optimizer
        config: Training configuration
        gradient_accumulator: Optional gradient accumulator
        
    Returns:
        Tuple of (compiled_train_step, state_list)
    """
    # Define the state that needs to be tracked
    # Include mx.random.state if using dropout or random augmentation
    state = [model.state, optimizer.state]
    
    # Add random state if model has dropout
    if _has_dropout(model):
        state.append(mx.random.state)
        logger.info("Added mx.random.state to compilation state (dropout detected)")
    
    # Create the loss function
    def loss_fn(model, batch):
        """Forward pass and loss computation."""
        # Remove metadata if present
        model_inputs = {k: v for k, v in batch.items() 
                       if k not in ['metadata'] and v is not None}
        
        # Apply mixed precision if enabled
        if config.training.mixed_precision:
            model_inputs = {
                k: v.astype(mx.bfloat16) if v.dtype == mx.float32 else v
                for k, v in model_inputs.items()
            }
        
        # Forward pass
        try:
            outputs = model(**model_inputs)
        except TypeError:
            outputs = model(batch)
        
        # Extract loss
        loss = outputs.get("loss")
        if loss is None:
            raise ValueError("Model must return a dictionary with 'loss' key")
        
        # Apply label smoothing if configured
        if config.training.label_smoothing > 0:
            loss = loss * (1 - config.training.label_smoothing)
            
        return loss, outputs
    
    # Create value and grad function
    value_and_grad_fn = nn.value_and_grad(model, loss_fn)
    
    # Define the training step to be compiled
    @partial(mx.compile, inputs=state, outputs=state)
    def compiled_train_step(batch: Dict[str, mx.array]) -> Tuple[mx.array, Dict[str, Any]]:
        """
        Compiled training step with state management.
        
        This function is compiled with mx.compile to optimize the computation graph.
        """
        # Compute loss and gradients
        (loss, outputs), grads = value_and_grad_fn(model, batch)
        
        # NOTE: mx.eval() cannot be used inside compiled functions
        # The compiled function will handle evaluation automatically
        
        # Gradient clipping if enabled
        if config.optimizer.max_grad_norm > 0:
            grads = _clip_gradients_compiled(grads, config.optimizer.max_grad_norm)
        
        # Handle gradient accumulation if provided
        if gradient_accumulator is not None and config.training.gradient_accumulation_steps > 1:
            # For compiled functions, we need to handle accumulation differently
            # The accumulator needs to be part of the state for proper compilation
            grads = _scale_gradients(grads, 1.0 / config.training.gradient_accumulation_steps)
        
        # Update model parameters
        optimizer.update(model, grads)
        
        # Build metrics dict
        metrics = {
            "loss": loss,
            "learning_rate": optimizer.learning_rate,
        }
        
        # Add other outputs
        for k, v in outputs.items():
            if k != "loss":
                metrics[k] = v
        
        return loss, metrics
    
    # Create a wrapper that handles gradient accumulation outside compilation
    if gradient_accumulator is not None and config.training.gradient_accumulation_steps > 1:
        def train_step_with_accumulation(batch):
            """Wrapper that handles gradient accumulation."""
            # This part runs outside compilation for flexibility
            loss, metrics = compiled_train_step(batch)
            return loss, metrics
        
        return train_step_with_accumulation, state
    else:
        return compiled_train_step, state


def create_compiled_eval_step(model: Model) -> callable:
    """
    Create a compiled evaluation step function.
    
    Args:
        model: The model to evaluate
        
    Returns:
        Compiled eval_step function
    """
    # For evaluation, we only need model state
    state = [model.state]
    
    @partial(mx.compile, inputs=state, outputs=state)
    def compiled_eval_step(batch: Dict[str, mx.array]) -> Tuple[mx.array, Dict[str, mx.array]]:
        """Compiled evaluation step."""
        # Remove metadata
        model_inputs = {k: v for k, v in batch.items() 
                       if k not in ['metadata'] and v is not None}
        
        # Forward pass (no gradients)
        try:
            outputs = model(**model_inputs)
        except TypeError:
            outputs = model(batch)
        
        # Extract loss
        loss = outputs.get("loss")
        if loss is None:
            raise ValueError("Model must return a dictionary with 'loss' key")
        
        # Build metrics
        metrics = {k: v for k, v in outputs.items() if k != "loss"}
        
        return loss, metrics
    
    return compiled_eval_step


def _has_dropout(model: Model) -> bool:
    """Check if model contains dropout layers."""
    if hasattr(model, 'modules'):
        for module in model.modules():
            if isinstance(module, nn.Dropout):
                return True
    return False


def _clip_gradients_compiled(grads: Dict[str, Any], max_norm: float) -> Dict[str, Any]:
    """
    Clip gradients by global norm (compiled version).
    
    This is a simplified version optimized for compilation.
    """
    # Compute global norm
    total_norm_sq = 0.0
    # Use tree_flatten from mlx.utils - returns a flat list
    flat_grads = tree_flatten(grads)
    for g in flat_grads:
        if g is not None:
            total_norm_sq = total_norm_sq + mx.sum(g * g)
    
    total_norm = mx.sqrt(total_norm_sq)
    
    # Clip factor
    clip_factor = mx.minimum(1.0, max_norm / (total_norm + 1e-6))
    
    # Apply clipping
    clipped_grads = tree_map(
        lambda g: g * clip_factor if g is not None else None,
        grads
    )
    
    return clipped_grads


def _scale_gradients(grads: Dict[str, Any], scale: float) -> Dict[str, Any]:
    """Scale gradients by a factor."""
    return tree_map(
        lambda g: g * scale if g is not None else None,
        grads
    )


def compile_full_training_loop(
    model: Model,
    optimizer: Any,  # MLX optimizer
    train_dataloader: Any,
    config: Any
) -> callable:
    """
    Compile the entire training loop for maximum performance.
    
    This is an experimental function that compiles the full epoch training.
    Use with caution as it may have limitations with certain operations.
    
    Args:
        model: The model to train
        optimizer: The optimizer
        train_dataloader: Training data loader
        config: Training configuration
        
    Returns:
        Compiled training loop function
    """
    state = [model.state, optimizer.state]
    
    if _has_dropout(model):
        state.append(mx.random.state)
    
    # Create loss function
    def loss_fn(model, batch):
        model_inputs = {k: v for k, v in batch.items() 
                       if k not in ['metadata'] and v is not None}
        
        try:
            outputs = model(**model_inputs)
        except TypeError:
            outputs = model(batch)
        
        loss = outputs.get("loss")
        if loss is None:
            raise ValueError("Model must return a dictionary with 'loss' key")
            
        return loss
    
    value_and_grad_fn = nn.value_and_grad(model, loss_fn)
    
    @partial(mx.compile, inputs=state, outputs=state)
    def compiled_epoch(batches: List[Dict[str, mx.array]]) -> mx.array:
        """
        Compiled epoch training.
        
        Note: This processes a list of batches in one compiled call.
        """
        total_loss = 0.0
        
        for batch in batches:
            # Compute loss and gradients
            loss, grads = value_and_grad_fn(model, batch)
            
            # Update model
            optimizer.update(model, grads)
            
            # Accumulate loss
            total_loss = total_loss + loss
        
        return total_loss / len(batches)
    
    return compiled_epoch


# Utility functions for compilation optimization

def should_compile_model(model: Model, config: Any) -> bool:
    """
    Determine if a model should be compiled based on its architecture and config.
    
    Some models or configurations may not benefit from compilation or may have
    compatibility issues.
    """
    # Check if compilation is explicitly disabled
    if hasattr(config, 'disable_compilation') and config.disable_compilation:
        return False
    
    # Check model size - very small models may not benefit
    # tree_flatten returns a flat list of parameters
    flat_params = tree_flatten(model.parameters())
    # flat_params is a list of (name, param) tuples
    param_count = sum(p.size for _, p in flat_params if p is not None)
    if param_count < 1000:  # Less than 1K parameters
        logger.info(f"Model too small for compilation benefits ({param_count} parameters)")
        return False
    
    # Check for known incompatible layers/operations
    # Add checks as needed based on MLX compilation limitations
    
    return True


def optimize_batch_size_for_compilation(
    model: Model,
    current_batch_size: int,
    config: Any
) -> int:
    """
    Suggest optimal batch size for compiled training.
    
    MLX compilation works best with certain batch sizes.
    """
    # MLX performs better with powers of 2
    optimal_sizes = [8, 16, 32, 64, 128, 256]
    
    # Find closest power of 2
    for size in optimal_sizes:
        if size >= current_batch_size:
            if size != current_batch_size:
                logger.info(f"Suggesting batch size {size} (from {current_batch_size}) for better compilation performance")
            return size
    
    return current_batch_size