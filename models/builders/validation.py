"""Validation service for model creation.

This module provides validation for models, ensuring they meet
requirements and have expected properties.
"""

from dataclasses import dataclass
from typing import Any, Optional

import mlx.nn as nn
from loguru import logger


@dataclass
class ValidationService:
    """Service for validating models and configurations."""
    
    def validate_model(self, model: nn.Module, model_type: str) -> None:
        """Validate a core model.
        
        Args:
            model: Model to validate
            model_type: Type of model for context
            
        Raises:
            ValueError: If validation fails
        """
        # Check model is a valid MLX module
        if not isinstance(model, nn.Module):
            raise ValueError(f"Model must be an MLX Module, got {type(model)}")
        
        # Check for expected methods based on model type
        if "core" in model_type:
            self._validate_core_model(model)
        elif "with_head" in model_type:
            self._validate_model_with_head(model)
            
        logger.debug(f"Model validation passed for {model_type}")
    
    def _validate_core_model(self, model: nn.Module) -> None:
        """Validate a core BERT model."""
        # Check for expected attributes
        # Note: BertCore uses encoder_layers (list) instead of encoder
        required_attrs = ["embeddings"]
        
        for attr in required_attrs:
            if not hasattr(model, attr):
                raise ValueError(f"Core model missing required attribute: {attr}")
        
        # Check for encoder layers (either encoder or encoder_layers)
        if not hasattr(model, "encoder") and not hasattr(model, "encoder_layers"):
            raise ValueError("Core model must have either 'encoder' or 'encoder_layers' attribute")
        
        # Check for expected methods
        if hasattr(model, "get_hidden_size"):
            hidden_size = model.get_hidden_size()
            if not isinstance(hidden_size, int) or hidden_size <= 0:
                raise ValueError(f"Invalid hidden size: {hidden_size}")
    
    def validate_model_with_head(
        self,
        model: nn.Module,
        head_type: Optional[str] = None,
    ) -> None:
        """Validate a model with attached head.
        
        Args:
            model: Model to validate
            head_type: Expected head type
            
        Raises:
            ValueError: If validation fails
        """
        # Check for bert and head components
        if not hasattr(model, "bert"):
            raise ValueError("Model with head must have 'bert' attribute")
            
        if not hasattr(model, "head"):
            raise ValueError("Model with head must have 'head' attribute")
        
        # Validate bert component
        self._validate_core_model(model.bert)
        
        # Validate head component
        self._validate_head(model.head, head_type)
        
        # Check freeze settings if present
        if hasattr(model, "freeze_bert"):
            if not isinstance(model.freeze_bert, bool):
                logger.warning("freeze_bert attribute is not a boolean")
    
    def _validate_head(self, head: nn.Module, expected_type: Optional[str]) -> None:
        """Validate a task head."""
        if not isinstance(head, nn.Module):
            raise ValueError(f"Head must be an MLX Module, got {type(head)}")
        
        # Check for forward method
        if not hasattr(head, "__call__"):
            raise ValueError("Head must be callable")
        
        logger.debug(f"Head validation passed for type: {expected_type}")
    
    def validate_config(self, config: Any, config_type: type) -> None:
        """Validate a configuration object.
        
        Args:
            config: Configuration to validate
            config_type: Expected configuration type
            
        Raises:
            ValueError: If validation fails
        """
        if not isinstance(config, config_type):
            raise ValueError(
                f"Config must be of type {config_type.__name__}, got {type(config)}"
            )
        
        # Validate specific config values
        if hasattr(config, "hidden_size"):
            if config.hidden_size <= 0:
                raise ValueError(f"Invalid hidden_size: {config.hidden_size}")
                
        if hasattr(config, "num_hidden_layers"):
            if config.num_hidden_layers <= 0:
                raise ValueError(f"Invalid num_hidden_layers: {config.num_hidden_layers}")
    
    def validate_weights_compatibility(
        self,
        model: nn.Module,
        weights: dict[str, Any],
    ) -> bool:
        """Check if weights are compatible with model.
        
        Args:
            model: Model to check against
            weights: Weights dictionary
            
        Returns:
            True if compatible, False otherwise
        """
        model_params = dict(model.parameters())
        
        # Check if all weight keys exist in model
        missing_keys = []
        for key in weights:
            if key not in model_params:
                missing_keys.append(key)
        
        if missing_keys:
            logger.warning(f"Missing keys in model: {missing_keys[:5]}...")
            
        # Check shape compatibility for common keys
        shape_mismatches = []
        for key in weights:
            if key in model_params:
                weight_shape = weights[key].shape
                param_shape = model_params[key].shape
                if weight_shape != param_shape:
                    shape_mismatches.append(
                        f"{key}: weight {weight_shape} vs param {param_shape}"
                    )
        
        if shape_mismatches:
            logger.warning(f"Shape mismatches: {shape_mismatches[:5]}...")
            
        return len(missing_keys) == 0 and len(shape_mismatches) == 0
    
    def validate_training_config(self, config: dict[str, Any]) -> None:
        """Validate training configuration.
        
        Args:
            config: Training configuration dictionary
            
        Raises:
            ValueError: If validation fails
        """
        required_keys = ["batch_size", "learning_rate", "num_epochs"]
        
        for key in required_keys:
            if key not in config:
                raise ValueError(f"Missing required training config key: {key}")
        
        # Validate values
        if config["batch_size"] <= 0:
            raise ValueError(f"Invalid batch_size: {config['batch_size']}")
            
        if config["learning_rate"] <= 0:
            raise ValueError(f"Invalid learning_rate: {config['learning_rate']}")
            
        if config["num_epochs"] <= 0:
            raise ValueError(f"Invalid num_epochs: {config['num_epochs']}")