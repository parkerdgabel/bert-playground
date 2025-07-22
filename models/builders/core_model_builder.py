"""Core model builder for creating BERT and ModernBERT models."""

from typing import Any

from core.bootstrap import get_service
from core.ports.compute import ComputeBackend, Module
from loguru import logger

from ..bert import BertConfig, ModernBertConfig, create_bert_core, create_modernbert_core


class CoreModelBuilder:
    """Builder for core BERT and ModernBERT models."""

    def __init__(self):
        self.compute_backend = get_service(ComputeBackend)

    def build_bert_core(
        self, config: dict[str, Any] | BertConfig | None = None, **kwargs
    ) -> Module:
        """Build a BERT core model.
        
        Args:
            config: Model configuration
            **kwargs: Additional configuration parameters
            
        Returns:
            BERT core model
        """
        logger.info("Building BERT core model")
        return create_bert_core(config=config, **kwargs)

    def build_modernbert_core(
        self, 
        config: dict[str, Any] | ModernBertConfig | None = None, 
        model_size: str = "base",
        **kwargs
    ) -> Module:
        """Build a ModernBERT core model.
        
        Args:
            config: Model configuration
            model_size: Model size (base or large)
            **kwargs: Additional configuration parameters
            
        Returns:
            ModernBERT core model
        """
        logger.info(f"Building ModernBERT core model ({model_size})")
        return create_modernbert_core(config=config, model_size=model_size, **kwargs)

    def build_core_model(self, model_type: str, config: Any = None, **kwargs) -> Module:
        """Build a core model of the specified type.
        
        Args:
            model_type: Type of model ("bert_core" or "modernbert_core")
            config: Model configuration
            **kwargs: Additional parameters
            
        Returns:
            Core model
            
        Raises:
            ValueError: If model_type is not supported
        """
        if model_type == "bert_core":
            return self.build_bert_core(config=config, **kwargs)
        elif model_type == "modernbert_core":
            return self.build_modernbert_core(config=config, **kwargs)
        else:
            raise ValueError(f"Unsupported core model type: {model_type}")