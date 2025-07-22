"""Builder for creating BERT models with task-specific heads."""

from typing import Any

from core.bootstrap import get_service
from core.ports.compute import ComputeBackend, Module
from loguru import logger

from ..bert import BertConfig, ModernBertConfig, create_bert_with_head, BertWithHead, ModernBertCore, create_modernbert_core
from ..heads.base import HeadConfig


class ModelWithHeadBuilder:
    """Builder for BERT models with task-specific heads."""

    def __init__(self):
        self.compute_backend = get_service(ComputeBackend)

    def build_bert_with_head(
        self,
        config: dict[str, Any] | BertConfig | None = None,
        head_type: str = "binary_classification",
        head_config: HeadConfig | dict | None = None,
        num_labels: int = 2,
        freeze_bert: bool = False,
        freeze_bert_layers: list[int] | None = None,
        **kwargs
    ) -> Module:
        """Build a BERT model with task-specific head.
        
        Args:
            config: BERT model configuration
            head_type: Type of head to attach
            head_config: Head configuration
            num_labels: Number of output labels
            freeze_bert: Whether to freeze BERT parameters
            freeze_bert_layers: Specific layers to freeze
            **kwargs: Additional parameters
            
        Returns:
            BERT model with head
        """
        logger.info(f"Building BERT with {head_type} head ({num_labels} labels)")
        
        return create_bert_with_head(
            config=config,
            head_type=head_type,
            head_config=head_config,
            num_labels=num_labels,
            freeze_bert=freeze_bert,
            freeze_bert_layers=freeze_bert_layers,
            **kwargs
        )

    def build_modernbert_with_head(
        self,
        config: dict[str, Any] | ModernBertConfig | None = None,
        head_type: str = "binary_classification",
        head_config: HeadConfig | dict | None = None,
        model_size: str = "base",
        num_labels: int = 2,
        freeze_bert: bool = False,
        freeze_bert_layers: list[int] | None = None,
        **kwargs
    ) -> Module:
        """Build a ModernBERT model with task-specific head.
        
        Args:
            config: ModernBERT model configuration
            head_type: Type of head to attach
            head_config: Head configuration
            model_size: Model size (base or large)
            num_labels: Number of output labels
            freeze_bert: Whether to freeze BERT parameters
            freeze_bert_layers: Specific layers to freeze
            **kwargs: Additional parameters
            
        Returns:
            ModernBERT model with head
        """
        logger.info(f"Building ModernBERT {model_size} with {head_type} head ({num_labels} labels)")
        
        # Create ModernBERT core first
        modernbert_core = create_modernbert_core(config=config, model_size=model_size, **kwargs)
        
        # Create head using the head factory (need to get it from DI)
        from core.bootstrap import get_service
        try:
            # Try to get head factory from DI container
            from models.builders.head_factory import HeadFactory
            head_factory = get_service(HeadFactory)
            
            # Get hidden size from core model
            if hasattr(modernbert_core, "get_hidden_size"):
                hidden_size = modernbert_core.get_hidden_size()
            else:
                # Fallback to config
                if hasattr(modernbert_core, "config"):
                    hidden_size = modernbert_core.config.hidden_size
                else:
                    # Default hidden size for ModernBERT base
                    hidden_size = 768 if model_size == "base" else 1024
            
            # Create head
            head = head_factory.create_head(
                head_type=head_type,
                head_config=head_config,
                input_size=hidden_size,
                output_size=num_labels,
            )
        except Exception:
            # Fallback: create basic head manually
            from ..heads.factory import create_head
            head = create_head(
                head_type, 
                input_size=modernbert_core.config.hidden_size if hasattr(modernbert_core, "config") else 768,
                output_size=num_labels
            )
        
        # Compose model using BertWithHead
        model = BertWithHead(
            bert=modernbert_core,
            head=head,
            freeze_bert=freeze_bert,
            freeze_bert_layers=freeze_bert_layers,
        )
        
        return model

    def build_model_with_head(
        self,
        model_type: str,
        config: Any = None,
        head_type: str = "binary_classification",
        head_config: Any = None,
        **kwargs
    ) -> Module:
        """Build a model with head of the specified type.
        
        Args:
            model_type: Type of model ("bert_with_head" or "modernbert_with_head")
            config: Model configuration
            head_type: Type of head to attach
            head_config: Head configuration
            **kwargs: Additional parameters
            
        Returns:
            Model with head
            
        Raises:
            ValueError: If model_type is not supported
        """
        if model_type == "bert_with_head":
            return self.build_bert_with_head(
                config=config,
                head_type=head_type,
                head_config=head_config,
                **kwargs
            )
        elif model_type == "modernbert_with_head":
            return self.build_modernbert_with_head(
                config=config,
                head_type=head_type,
                head_config=head_config,
                **kwargs
            )
        else:
            raise ValueError(f"Unsupported model with head type: {model_type}")