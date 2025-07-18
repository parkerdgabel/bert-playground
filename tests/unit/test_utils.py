"""Test utilities for model testing."""

import mlx.core as mx
import mlx.nn as nn
from typing import Tuple, Dict, Any, Optional
import numpy as np


def create_dummy_inputs(
    batch_size: int = 2,
    seq_length: int = 16,
    vocab_size: int = 1000,
    include_labels: bool = False,
    num_labels: int = 2,
    task_type: str = "classification"
) -> Dict[str, mx.array]:
    """Create dummy inputs for model testing.
    
    Args:
        batch_size: Batch size
        seq_length: Sequence length
        vocab_size: Vocabulary size
        include_labels: Whether to include labels
        num_labels: Number of labels for classification
        task_type: Task type (classification, regression, multilabel)
        
    Returns:
        Dictionary of input tensors
    """
    inputs = {
        "input_ids": mx.random.randint(0, vocab_size, (batch_size, seq_length)),
        "attention_mask": mx.ones((batch_size, seq_length))
    }
    
    if include_labels:
        if task_type == "classification":
            inputs["labels"] = mx.random.randint(0, num_labels, (batch_size,))
        elif task_type == "regression":
            inputs["labels"] = mx.random.normal((batch_size, 1))
        elif task_type == "multilabel":
            inputs["labels"] = mx.random.randint(0, 2, (batch_size, num_labels))
        elif task_type == "ordinal":
            inputs["labels"] = mx.random.randint(0, num_labels, (batch_size,))
    
    return inputs


def assert_tensor_shape(tensor: mx.array, expected_shape: Tuple[int, ...], name: str = "tensor"):
    """Assert tensor has expected shape."""
    actual_shape = tuple(tensor.shape)
    assert actual_shape == expected_shape, f"{name} shape mismatch: expected {expected_shape}, got {actual_shape}"


def assert_all_finite(tensor: mx.array, name: str = "tensor"):
    """Assert tensor contains no inf or nan values."""
    assert mx.all(mx.isfinite(tensor)), f"{name} contains inf or nan values"


def count_parameters(model: nn.Module, trainable_only: bool = True) -> int:
    """Count model parameters.
    
    Args:
        model: Model to count parameters for
        trainable_only: Whether to count only trainable parameters
        
    Returns:
        Number of parameters
    """
    total = 0
    for name, param in model.named_parameters():
        if trainable_only and hasattr(param, 'stop_gradient') and param.stop_gradient:
            continue
        total += param.size
    return total


def check_gradient_flow(model: nn.Module, loss: mx.array) -> Dict[str, bool]:
    """Check if gradients flow through model layers.
    
    Args:
        model: Model to check
        loss: Loss to compute gradients from
        
    Returns:
        Dictionary mapping parameter names to whether gradients exist
    """
    # Compute gradients
    grads = mx.grad(lambda: loss)(model)
    
    gradient_flow = {}
    for name, param in model.named_parameters():
        if name in grads:
            grad = grads[name]
            has_gradient = grad is not None and mx.any(grad != 0)
            gradient_flow[name] = has_gradient
        else:
            gradient_flow[name] = False
    
    return gradient_flow


def create_simple_bert_config():
    """Create a simple BERT config for testing."""
    from models.bert import BertConfig
    
    return BertConfig(
        hidden_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=256,
        vocab_size=1000,
        max_position_embeddings=512,
        type_vocab_size=2
    )


def create_simple_modernbert_config():
    """Create a simple ModernBERT config for testing."""
    from models.bert import ModernBertConfig
    
    return ModernBertConfig(
        hidden_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=256,
        vocab_size=1000,
        max_position_embeddings=512,
        model_size="small"
    )


def create_simple_lora_config():
    """Create a simple LoRA config for testing."""
    from models.lora import LoRAConfig
    
    return LoRAConfig(
        r=4,
        alpha=8,
        dropout=0.1,
        target_modules={"query", "value"}
    )


def create_simple_qlora_config():
    """Create a simple QLoRA config for testing."""
    from models.lora import QLoRAConfig
    
    return QLoRAConfig(
        r=4,
        alpha=4,
        dropout=0.1,
        target_modules={"query", "value"},
        bnb_4bit_compute_dtype="float16",
        bnb_4bit_quant_type="nf4"
    )


class ModelTestCase:
    """Base class for model testing with common assertions."""
    
    @staticmethod
    def assert_forward_pass(model: nn.Module, inputs: Dict[str, mx.array], 
                          expected_output_shape: Optional[Tuple[int, ...]] = None):
        """Assert model forward pass works correctly."""
        # Forward pass
        outputs = model(**inputs)
        
        # Check outputs exist
        assert outputs is not None, "Model returned None"
        
        # Check output shape if provided
        if expected_output_shape is not None:
            if hasattr(outputs, 'shape'):
                assert_tensor_shape(outputs, expected_output_shape, "output")
            elif hasattr(outputs, 'logits'):
                assert_tensor_shape(outputs.logits, expected_output_shape, "logits")
        
        # Check for NaN/Inf
        if hasattr(outputs, 'last_hidden_state'):
            assert_all_finite(outputs.last_hidden_state, "last_hidden_state")
        if hasattr(outputs, 'logits'):
            assert_all_finite(outputs.logits, "logits")
        if hasattr(outputs, 'loss') and outputs.loss is not None:
            assert_all_finite(outputs.loss, "loss")
        
        return outputs
    
    @staticmethod
    def assert_backward_pass(model: nn.Module, loss: mx.array):
        """Assert gradients flow correctly through model."""
        # Check loss is scalar
        assert loss.shape == (), f"Loss should be scalar, got shape {loss.shape}"
        
        # Check gradients
        gradient_flow = check_gradient_flow(model, loss)
        
        # At least some parameters should have gradients
        has_any_gradient = any(gradient_flow.values())
        assert has_any_gradient, "No gradients found in any parameters"
        
        return gradient_flow