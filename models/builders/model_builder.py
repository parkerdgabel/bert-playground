"""Core model builder for BERT and ModernBERT architectures.

This module handles the instantiation of base models and composed models,
delegating configuration and head creation to specialized components.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Union

import mlx.nn as nn
from loguru import logger

from ..bert import (
    BertConfig,
    BertCore,
    BertWithHead,
    ModernBertConfig,
    create_bert_core,
    create_bert_with_head,
    create_modernbert_core,
)
from .config_resolver import ConfigResolver
from .head_factory import HeadFactory
from .validation import ValidationService


@dataclass
class ModelBuilder:
    """Builder for creating BERT-based models."""
    
    config_resolver: ConfigResolver
    head_factory: HeadFactory
    validation_service: ValidationService
    
    def build_core(
        self,
        model_type: str,
        config: Union[dict[str, Any], BertConfig, ModernBertConfig, None] = None,
        **kwargs,
    ) -> nn.Module:
        """Build a core BERT model without head.
        
        Args:
            model_type: Type of core model ("bert_core" or "modernbert_core")
            config: Model configuration
            **kwargs: Additional model parameters
            
        Returns:
            Core model instance
        """
        # Resolve configuration
        config = self.config_resolver.resolve_bert_config(config, model_type, **kwargs)
        
        if model_type == "bert_core":
            model = create_bert_core(config=config, **kwargs)
            logger.info("Created Classic BertCore model")
        elif model_type == "modernbert_core":
            model = create_modernbert_core(config=config, **kwargs)
            logger.info("Created ModernBertCore model")
        else:
            raise ValueError(f"Unknown core model type: {model_type}")
            
        # Validate model
        self.validation_service.validate_model(model, model_type)
        
        return model
    
    def build_with_head(
        self,
        model_type: str,
        config: Union[dict[str, Any], BertConfig, ModernBertConfig, None] = None,
        head_type: Optional[str] = None,
        head_config: Optional[Any] = None,
        num_labels: int = 2,
        freeze_bert: bool = False,
        freeze_bert_layers: Optional[int] = None,
        **kwargs,
    ) -> BertWithHead:
        """Build a BERT model with attached head.
        
        Args:
            model_type: Type of model ("bert_with_head" or "modernbert_with_head")
            config: Model configuration
            head_type: Type of head to attach
            head_config: Head configuration
            num_labels: Number of output labels
            freeze_bert: Whether to freeze BERT parameters
            freeze_bert_layers: Number of BERT layers to freeze
            **kwargs: Additional parameters
            
        Returns:
            Model with attached head
        """
        # Resolve configuration
        config = self.config_resolver.resolve_bert_config(config, model_type, **kwargs)
        
        if model_type == "bert_with_head":
            # Create Classic BERT with head
            model = create_bert_with_head(
                bert_config=config,
                head_config=head_config,
                head_type=head_type,
                num_labels=num_labels,
                freeze_bert=freeze_bert,
                freeze_bert_layers=freeze_bert_layers,
                **kwargs,
            )
            logger.info(f"Created BertWithHead model (head_type: {head_type})")
            
        elif model_type == "modernbert_with_head":
            # Create ModernBERT core first
            modernbert_core = create_modernbert_core(config=config, **kwargs)
            
            # Create head
            head = self._create_head_for_core(
                modernbert_core,
                head_type,
                head_config,
                num_labels,
            )
            
            # Compose model
            model = BertWithHead(
                bert=modernbert_core,
                head=head,
                freeze_bert=freeze_bert,
                freeze_bert_layers=freeze_bert_layers,
            )
            logger.info(f"Created ModernBertWithHead model (head_type: {head_type})")
            
        else:
            raise ValueError(f"Unknown model type: {model_type}")
            
        # Validate composed model
        self.validation_service.validate_model_with_head(model, head_type)
        
        return model
    
    def _create_head_for_core(
        self,
        core_model: nn.Module,
        head_type: Optional[str],
        head_config: Optional[Any],
        num_labels: int,
    ) -> nn.Module:
        """Create a head for a core model.
        
        Args:
            core_model: Core BERT model
            head_type: Type of head to create
            head_config: Head configuration
            num_labels: Number of output labels
            
        Returns:
            Initialized head
        """
        # Get hidden size from core model
        if hasattr(core_model, "get_hidden_size"):
            hidden_size = core_model.get_hidden_size()
        else:
            # Fallback to config if available
            if hasattr(core_model, "config"):
                hidden_size = core_model.config.hidden_size
            else:
                raise ValueError("Cannot determine hidden size from core model")
        
        # Resolve head config if needed
        if head_config is None:
            head_config = self.config_resolver.resolve_head_config(
                head_config=None,
                head_type=head_type,
                input_size=hidden_size,
                output_size=num_labels,
            )
        elif isinstance(head_config, dict):
            # Convert dict to HeadConfig
            from ..heads.base import HeadConfig
            head_config = HeadConfig(**head_config)
        
        # Create head using factory
        return self.head_factory.create_head(
            head_type=head_type or head_config.head_type,
            head_config=head_config,
            input_size=hidden_size,
            output_size=num_labels,
        )
    
    def load_pretrained_weights(
        self,
        model: nn.Module,
        weights_path: Union[str, Path],
    ) -> None:
        """Load pretrained weights into a model.
        
        Args:
            model: Model to load weights into
            weights_path: Path to weights file or directory
        """
        weights_path = Path(weights_path)
        import mlx.core as mx
        
        if weights_path.is_dir():
            # Load from directory
            safetensors_path = weights_path / "model.safetensors"
            if safetensors_path.exists():
                weights = mx.load(str(safetensors_path))
                model.load_weights(list(weights.items()))
                logger.info(f"Loaded weights from {safetensors_path}")
            else:
                raise ValueError(f"No model.safetensors found in {weights_path}")
                
        elif weights_path.suffix == ".safetensors":
            # Load safetensors file directly
            weights = mx.load(str(weights_path))
            model.load_weights(list(weights.items()))
            logger.info(f"Loaded weights from {weights_path}")
            
        else:
            raise ValueError(f"Unsupported weights format: {weights_path}")
    
    def get_parameter_count(self, model: nn.Module) -> dict[str, int]:
        """Get parameter count breakdown for a model.
        
        Args:
            model: Model to analyze
            
        Returns:
            Dictionary with parameter counts by component
        """
        breakdown = {}
        
        if hasattr(model, "bert") and hasattr(model, "head"):
            # BertWithHead model
            if hasattr(model.bert, "num_parameters"):
                breakdown["bert"] = model.bert.num_parameters()
            if hasattr(model.head, "num_parameters"):
                breakdown["head"] = model.head.num_parameters()
        
        # Total count
        breakdown["total"] = model.num_parameters() if hasattr(model, "num_parameters") else 0
        
        return breakdown