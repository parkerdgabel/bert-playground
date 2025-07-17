"""
Migration Utilities for MLX Embeddings

Provides utilities for migrating existing checkpoints and configurations
to work with the mlx-embeddings backend.
"""

from typing import Dict, Optional, Union, Any
from pathlib import Path
import json
import shutil
from loguru import logger

from embeddings.config import MODEL_NAME_MAPPING, MLXEmbeddingsConfig
from embeddings.tokenizer_wrapper import TokenizerWrapper


class CheckpointMigrator:
    """Handles migration of existing checkpoints to MLX embeddings format."""
    
    def __init__(self, checkpoint_path: Union[str, Path]):
        """
        Initialize checkpoint migrator.
        
        Args:
            checkpoint_path: Path to existing checkpoint directory
        """
        self.checkpoint_path = Path(checkpoint_path)
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        self.config_path = self.checkpoint_path / "config.json"
        self.weights_path = self.checkpoint_path / "model.safetensors"
        
        # Load existing configuration
        self.original_config = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from checkpoint."""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config not found: {self.config_path}")
        
        with open(self.config_path) as f:
            return json.load(f)
    
    def migrate_to_mlx_embeddings(
        self,
        output_path: Optional[Union[str, Path]] = None,
        model_name_override: Optional[str] = None,
    ) -> Path:
        """
        Migrate checkpoint to use MLX embeddings.
        
        Args:
            output_path: Path for migrated checkpoint (default: adds _mlx suffix)
            model_name_override: Override model name mapping
            
        Returns:
            Path to migrated checkpoint
        """
        # Determine output path
        if output_path is None:
            output_path = self.checkpoint_path.parent / f"{self.checkpoint_path.name}_mlx"
        output_path = Path(output_path)
        
        # Create output directory
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Migrate configuration
        migrated_config = self._migrate_config(model_name_override)
        
        # Save migrated config
        with open(output_path / "config.json", "w") as f:
            json.dump(migrated_config, f, indent=2)
        
        # Copy weights if they exist
        if self.weights_path.exists():
            shutil.copy2(self.weights_path, output_path / "model.safetensors")
            logger.info(f"Copied weights to {output_path}")
        
        # Create migration info
        migration_info = {
            "original_checkpoint": str(self.checkpoint_path),
            "original_model": self.original_config.get("model_name", "unknown"),
            "migrated_model": migrated_config["model_name"],
            "migration_type": "mlx_embeddings",
            "tokenizer_backend": "mlx",
        }
        
        with open(output_path / "migration_info.json", "w") as f:
            json.dump(migration_info, f, indent=2)
        
        logger.info(f"Successfully migrated checkpoint to: {output_path}")
        return output_path
    
    def _migrate_config(self, model_name_override: Optional[str] = None) -> Dict[str, Any]:
        """
        Migrate configuration to MLX embeddings format.
        
        Args:
            model_name_override: Override model name mapping
            
        Returns:
            Migrated configuration
        """
        config = self.original_config.copy()
        
        # Get original model name
        original_model = config.get("model_name", config.get("model", "answerdotai/ModernBERT-base"))
        
        # Map to MLX model name
        if model_name_override:
            mlx_model_name = model_name_override
        else:
            mlx_model_name = MODEL_NAME_MAPPING.get(original_model, original_model)
        
        # Update configuration
        config["model_name"] = mlx_model_name
        config["use_mlx_embeddings"] = True
        config["tokenizer_backend"] = "mlx"
        config["original_model_name"] = original_model
        
        # Add MLX-specific settings
        if "hidden_size" not in config:
            # Infer from model name
            if "large" in mlx_model_name.lower():
                config["hidden_size"] = 1024
            elif "small" in mlx_model_name.lower() or "mini" in mlx_model_name.lower():
                config["hidden_size"] = 384
            else:
                config["hidden_size"] = 768
        
        return config
    
    def validate_migration(self, migrated_path: Union[str, Path]) -> bool:
        """
        Validate that migration was successful.
        
        Args:
            migrated_path: Path to migrated checkpoint
            
        Returns:
            True if validation passes
        """
        migrated_path = Path(migrated_path)
        
        # Check required files exist
        required_files = ["config.json", "migration_info.json"]
        for file in required_files:
            if not (migrated_path / file).exists():
                logger.error(f"Missing required file: {file}")
                return False
        
        # Load and validate config
        try:
            with open(migrated_path / "config.json") as f:
                config = json.load(f)
            
            if not config.get("use_mlx_embeddings"):
                logger.error("Configuration not set for MLX embeddings")
                return False
            
            # Try to create tokenizer with new config
            tokenizer = TokenizerWrapper(
                model_name=config["model_name"],
                backend="mlx"
            )
            
            logger.info("Migration validation passed")
            return True
            
        except Exception as e:
            logger.error(f"Validation failed: {e}")
            return False


def migrate_checkpoint(
    checkpoint_path: Union[str, Path],
    output_path: Optional[Union[str, Path]] = None,
    validate: bool = True,
) -> Path:
    """
    Convenience function to migrate a checkpoint to MLX embeddings.
    
    Args:
        checkpoint_path: Path to existing checkpoint
        output_path: Output path for migrated checkpoint
        validate: Whether to validate migration
        
    Returns:
        Path to migrated checkpoint
    """
    migrator = CheckpointMigrator(checkpoint_path)
    migrated_path = migrator.migrate_to_mlx_embeddings(output_path)
    
    if validate:
        if not migrator.validate_migration(migrated_path):
            raise RuntimeError("Migration validation failed")
    
    return migrated_path


def create_mlx_config_from_huggingface(
    model_name: str,
    num_labels: Optional[int] = None,
    **kwargs
) -> MLXEmbeddingsConfig:
    """
    Create MLX embeddings configuration from HuggingFace model name.
    
    Args:
        model_name: HuggingFace model name
        num_labels: Number of classification labels
        **kwargs: Additional configuration parameters
        
    Returns:
        MLXEmbeddingsConfig instance
    """
    # Map to MLX model name
    mlx_model_name = MODEL_NAME_MAPPING.get(model_name, model_name)
    
    # Create config
    config = MLXEmbeddingsConfig.from_model_name(
        mlx_model_name,
        **kwargs
    )
    
    # Set original model name for reference
    config.model_kwargs["original_model_name"] = model_name
    
    return config


def check_mlx_embeddings_compatibility(model_name: str) -> Dict[str, Any]:
    """
    Check if a model is compatible with MLX embeddings.
    
    Args:
        model_name: Model name to check
        
    Returns:
        Dictionary with compatibility information
    """
    info = {
        "model_name": model_name,
        "has_mlx_mapping": model_name in MODEL_NAME_MAPPING,
        "mlx_model_name": MODEL_NAME_MAPPING.get(model_name, None),
        "supported": False,
        "notes": []
    }
    
    # Check if model has direct mapping
    if info["has_mlx_mapping"]:
        info["supported"] = True
        info["notes"].append("Direct MLX community model available")
    
    # Check if it's already an MLX model
    elif model_name.startswith("mlx-community/"):
        info["supported"] = True
        info["mlx_model_name"] = model_name
        info["notes"].append("Already an MLX community model")
    
    # Check if it's a BERT-based model
    elif any(bert_type in model_name.lower() for bert_type in ["bert", "roberta"]):
        info["supported"] = True
        info["notes"].append("BERT-based model, may work with generic MLX BERT")
    
    else:
        info["notes"].append("Model may not be compatible with MLX embeddings")
    
    return info