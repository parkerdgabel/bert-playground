"""Checkpoint management for loading and saving model states."""

import json
from pathlib import Path
from typing import Any

from core.bootstrap import get_service
from core.ports.compute import ComputeBackend, Module
from loguru import logger

from utils.logging_utils import bind_context, log_timing


class CheckpointManager:
    """Manager for loading and saving model checkpoints."""

    def __init__(self):
        self.compute_backend = get_service(ComputeBackend)

    def load_model_from_checkpoint(self, checkpoint_path: str | Path) -> Module:
        """Load a model from a checkpoint directory.
        
        Args:
            checkpoint_path: Path to checkpoint directory
            
        Returns:
            Loaded model
            
        Raises:
            ValueError: If checkpoint is invalid or missing
        """
        checkpoint_path = Path(checkpoint_path)
        log = bind_context(checkpoint=str(checkpoint_path))
        
        with log_timing("load_model_from_checkpoint", checkpoint=str(checkpoint_path)):
            log.info(f"Loading model from checkpoint: {checkpoint_path}")

            # Look for training configuration
            training_config = self._load_training_config(checkpoint_path)
            
            # Load weights and infer model architecture
            weights_path = checkpoint_path / "model.safetensors"
            if not weights_path.exists():
                raise ValueError(f"No model.safetensors found in {checkpoint_path}")

            weights = self.compute_backend.load_weights(str(weights_path))
            model_type = self._infer_model_type(weights)
            
            # Import here to avoid circular dependency
            from ..factory import create_model
            
            # Create model with appropriate architecture
            model = create_model(
                model_type=model_type,
                head_type="binary_classification",
                num_labels=2,
                model_size=training_config.get("model_type", "base"),
            )

            # Load weights into model using tree_unflatten
            unflattened_weights = self.compute_backend.tree_unflatten(list(weights.items()))
            model.update(unflattened_weights)
            log.info(f"Loaded weights from {weights_path}")

            return model

    def save_model_checkpoint(
        self, 
        model: Module, 
        checkpoint_path: str | Path,
        metadata: dict[str, Any] | None = None
    ) -> None:
        """Save a model checkpoint with metadata.
        
        Args:
            model: Model to save
            checkpoint_path: Path to save checkpoint
            metadata: Optional metadata to save with checkpoint
        """
        checkpoint_path = Path(checkpoint_path)
        checkpoint_path.mkdir(parents=True, exist_ok=True)
        
        log = bind_context(checkpoint=str(checkpoint_path))
        
        with log_timing("save_model_checkpoint", checkpoint=str(checkpoint_path)):
            log.info(f"Saving model checkpoint to: {checkpoint_path}")
            
            # Save model weights
            weights_path = checkpoint_path / "model.safetensors"
            model_weights = dict(model.parameters())
            self.compute_backend.save_arrays(str(weights_path), model_weights)
            
            # Save metadata
            if metadata:
                metadata_path = checkpoint_path / "metadata.json"
                with open(metadata_path, "w") as f:
                    json.dump(metadata, f, indent=2)
                    
            log.info(f"Saved model checkpoint to {checkpoint_path}")

    def _load_training_config(self, checkpoint_path: Path) -> dict[str, Any]:
        """Load training configuration from checkpoint or parent directories.
        
        Args:
            checkpoint_path: Path to checkpoint directory
            
        Returns:
            Training configuration dictionary
        """
        # Try to find training configuration in parent directories
        training_config_path = None
        current_path = checkpoint_path
        for _ in range(3):  # Search up to 3 levels up
            current_path = current_path.parent
            potential_config = current_path / "training_config.json"
            if potential_config.exists():
                training_config_path = potential_config
                break

        # Load training config if available
        if training_config_path:
            with open(training_config_path) as f:
                training_config = json.load(f)
            logger.info(f"Found training config at {training_config_path}")
        else:
            # Use defaults
            training_config = {"model": "answerdotai/ModernBERT-base", "model_type": "base"}
            logger.warning("No training config found, using defaults")
            
        return training_config

    def _infer_model_type(self, weights: dict[str, Any]) -> str:
        """Infer model architecture from weight keys.
        
        Args:
            weights: Model weights dictionary
            
        Returns:
            Model type string
        """
        weight_keys = list(weights.keys())

        # Infer model type from weight keys by checking the number of encoder layers
        encoder_layer_keys = [k for k in weight_keys if "encoder_layers" in k]
        if encoder_layer_keys:
            # Extract layer numbers
            layer_numbers = set()
            for key in encoder_layer_keys:
                parts = key.split('.')
                for i, part in enumerate(parts):
                    if part == "encoder_layers" and i + 1 < len(parts):
                        try:
                            layer_num = int(parts[i + 1])
                            layer_numbers.add(layer_num)
                        except ValueError:
                            pass
            
            max_layer = max(layer_numbers) if layer_numbers else 0
            
            if max_layer > 11:  # ModernBERT has 22 layers (0-21)
                model_type = "modernbert_with_head"
                logger.info(f"Detected ModernBERT architecture (found {max_layer + 1} layers)")
            else:  # Classic BERT has 12 layers (0-11)
                model_type = "bert_with_head"
                logger.info(f"Detected classic BERT architecture (found {max_layer + 1} layers)")
        else:
            # Default to classic BERT for backward compatibility
            model_type = "bert_with_head"
            logger.warning(
                "Could not determine architecture from weights, defaulting to classic BERT"
            )
            
        return model_type

    def load_pretrained_weights(self, model: Module, weights_path: str | Path) -> None:
        """Load pretrained weights into a model.
        
        Args:
            model: Model to load weights into
            weights_path: Path to weights file
        """
        log = bind_context(weights_path=str(weights_path))
        
        with log_timing("load_pretrained_weights", path=str(weights_path)):
            log.info(f"Loading pretrained weights from {weights_path}")
            
            weights = self.compute_backend.load_weights(str(weights_path))
            unflattened_weights = self.compute_backend.tree_unflatten(list(weights.items()))
            model.update(unflattened_weights)
            
            log.info(f"Loaded pretrained weights from {weights_path}")