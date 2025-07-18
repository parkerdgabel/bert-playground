"""Example: Using LoRA for efficient BERT fine-tuning on Kaggle competitions.

This example demonstrates how to use LoRA (Low-Rank Adaptation) to efficiently
fine-tune BERT models for Kaggle competitions with minimal computational resources.
"""

import mlx.core as mx
import mlx.nn as nn
from pathlib import Path
from loguru import logger

# Import model creation functions
from models import (
    create_bert_with_lora,
    create_modernbert_with_lora,
    create_qlora_model,
    create_kaggle_lora_model,
    create_multi_adapter_model,
    LoRAConfig,
    QLoRAConfig,
    KAGGLE_LORA_PRESETS,
)


def example_1_basic_lora():
    """Example 1: Basic LoRA fine-tuning for binary classification."""
    logger.info("Example 1: Basic LoRA fine-tuning")
    
    # Create BERT model with LoRA adapters for binary classification
    model, lora_adapter = create_bert_with_lora(
        head_type="binary_classification",
        lora_preset="balanced",  # Use balanced preset (r=8)
        num_labels=2,
        freeze_bert=True,  # Freeze base BERT (recommended)
    )
    
    # The model now has LoRA adapters injected
    logger.info(f"Model created with LoRA adapters")
    
    # You can train only the LoRA parameters (much more efficient)
    trainable_params = sum(p.size for p in model.parameters() if not p.stop_gradient)
    total_params = sum(p.size for p in model.parameters())
    logger.info(f"Trainable parameters: {trainable_params:,} / {total_params:,} ({trainable_params/total_params*100:.1f}%)")
    
    return model, lora_adapter


def example_2_custom_lora_config():
    """Example 2: Custom LoRA configuration for specific needs."""
    logger.info("\nExample 2: Custom LoRA configuration")
    
    # Create custom LoRA config
    custom_config = LoRAConfig(
        r=16,  # Higher rank for more expressivity
        alpha=32,  # Scaling factor
        dropout=0.1,
        target_modules={"query", "value"},  # Only target Q and V projections
        use_rslora=True,  # Use Rank-Stabilized LoRA
    )
    
    # Create model with custom config
    model, lora_adapter = create_bert_with_lora(
        head_type="multiclass_classification",
        lora_config=custom_config,
        num_labels=5,
    )
    
    logger.info(f"Model created with custom LoRA config (r={custom_config.r})")
    
    return model, lora_adapter


def example_3_qlora_memory_efficient():
    """Example 3: QLoRA for memory-efficient fine-tuning of large models."""
    logger.info("\nExample 3: QLoRA for memory efficiency")
    
    # Create QLoRA model (4-bit base + fp16 LoRA adapters)
    model, lora_adapter = create_qlora_model(
        model_type="modernbert_with_head",
        qlora_preset="qlora_memory",
        head_type="binary_classification",
        num_labels=2,
        quantize_base=True,  # Quantize to 4-bit
    )
    
    logger.info("Created QLoRA model with 4-bit quantized base")
    
    # QLoRA uses much less memory
    if hasattr(lora_adapter.lora_modules.get(list(lora_adapter.lora_modules.keys())[0], None), 'memory_footprint'):
        first_module = list(lora_adapter.lora_modules.values())[0]
        memory_info = first_module.memory_footprint
        logger.info(f"Memory savings: {memory_info['compression_ratio']:.1f}x compression")
    
    return model, lora_adapter


def example_4_kaggle_competition_specific():
    """Example 4: Auto-configured LoRA for specific Kaggle competition."""
    logger.info("\nExample 4: Kaggle competition-specific LoRA")
    
    # Create LoRA model optimized for a specific competition type
    model, lora_adapter = create_kaggle_lora_model(
        competition_type="binary_classification",
        data_path=None,  # Would analyze data if provided
        lora_preset=None,  # Auto-selects based on competition
        auto_select_preset=True,
        num_labels=2,
    )
    
    logger.info("Created auto-configured LoRA model for binary classification")
    
    # Show which preset was selected
    logger.info(f"Auto-selected config: r={lora_adapter.config.r}, alpha={lora_adapter.config.alpha}")
    
    return model, lora_adapter


def example_5_multi_adapter_ensemble():
    """Example 5: Multiple LoRA adapters for ensemble or multi-task learning."""
    logger.info("\nExample 5: Multi-adapter for ensemble learning")
    
    # Create base model with multiple LoRA adapters
    from models import create_model
    base_model = create_model("bert_with_head", head_type="binary_classification")
    
    # Define multiple adapter configs
    adapter_configs = {
        "adapter_1": "efficient",  # r=4 for fast training
        "adapter_2": "balanced",   # r=8 for balance
        "adapter_3": "expressive", # r=16 for expressivity
    }
    
    # Create multi-adapter model
    model, adapter_manager = create_multi_adapter_model(
        base_model=base_model,
        adapter_configs=adapter_configs,
    )
    
    logger.info(f"Created model with {len(adapter_configs)} LoRA adapters")
    
    # You can train different adapters on different data splits
    # or use them for ensemble predictions
    
    # Activate specific adapter
    adapter_manager.activate_adapter("adapter_1")
    logger.info("Activated adapter_1 for training")
    
    return model, adapter_manager


def example_6_lora_for_different_tasks():
    """Example 6: Using LoRA presets for different Kaggle tasks."""
    logger.info("\nExample 6: LoRA presets for different tasks")
    
    # Show available presets
    logger.info("Available LoRA presets:")
    for name, config in KAGGLE_LORA_PRESETS.items():
        logger.info(f"  - {name}: r={config.r}, targets={len(config.target_modules)} modules")
    
    # Create models for different tasks
    tasks = [
        ("titanic", "binary_classification", "balanced"),
        ("house_prices", "regression", "efficient"),
        ("digit_recognizer", "multiclass_classification", "expressive"),
    ]
    
    models = {}
    for task_name, task_type, preset in tasks:
        model, adapter = create_kaggle_lora_model(
            competition_type=task_type,
            lora_preset=preset,
            num_labels=2 if task_type == "binary_classification" else 10,
        )
        models[task_name] = (model, adapter)
        logger.info(f"Created {task_name} model with {preset} preset")
    
    return models


def example_7_lora_adapter_management():
    """Example 7: Managing LoRA adapters (save, load, merge)."""
    logger.info("\nExample 7: LoRA adapter management")
    
    # Create model with LoRA
    model, lora_adapter = create_bert_with_lora(
        head_type="binary_classification",
        lora_preset="balanced",
    )
    
    # Get LoRA-only state dict (much smaller than full model)
    lora_state = lora_adapter.get_lora_state_dict()
    logger.info(f"LoRA state dict has {len(lora_state)} tensors")
    
    # Save LoRA weights (would use safetensors in practice)
    # import safetensors.mlx
    # safetensors.mlx.save_file(lora_state, "lora_weights.safetensors")
    
    # Remove adapters (restore original model)
    lora_adapter.remove_adapters(restore_original=True)
    logger.info("Removed LoRA adapters")
    
    # Re-inject adapters
    lora_adapter.inject_adapters()
    logger.info("Re-injected LoRA adapters")
    
    # Load saved weights
    # loaded_state = safetensors.mlx.load_file("lora_weights.safetensors")
    # lora_adapter.load_lora_state_dict(loaded_state)
    
    # Merge adapters for inference (creates new model with merged weights)
    # This is useful for deployment when you don't need adapter flexibility
    merge_status = lora_adapter.merge_adapters()
    logger.info(f"Merged {sum(merge_status.values())} adapters into base model")
    
    return model, lora_adapter


def main():
    """Run all examples."""
    logger.info("LoRA for Kaggle Competitions - Examples\n")
    
    # Run examples
    example_1_basic_lora()
    example_2_custom_lora_config()
    example_3_qlora_memory_efficient()
    example_4_kaggle_competition_specific()
    example_5_multi_adapter_ensemble()
    example_6_lora_for_different_tasks()
    example_7_lora_adapter_management()
    
    logger.info("\nAll examples completed!")
    logger.info("\nKey takeaways:")
    logger.info("1. LoRA reduces trainable parameters by 90-99%")
    logger.info("2. QLoRA enables fine-tuning with 4-bit models")
    logger.info("3. Different presets optimize for different scenarios")
    logger.info("4. Multi-adapter support enables ensemble methods")
    logger.info("5. LoRA weights can be saved/loaded separately")


if __name__ == "__main__":
    main()