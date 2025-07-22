"""Port-based model factory using hexagonal architecture.

This module demonstrates how to use the port interfaces to create models
in a framework-agnostic way.
"""

from pathlib import Path
from typing import Any, Literal

import mlx.nn as nn
from loguru import logger

from core.factory import get_context
from core.ports import ComputeBackend, MonitoringService, StorageService
from core.ports.compute import NeuralOps
from core.ports.storage import ModelStorageService

# Import model configs
from .bert import BertConfig, ModernBertConfig
from .heads.base import HeadConfig

ModelType = Literal[
    "bert_core",
    "bert_with_head",
    "modernbert_core",
    "modernbert_with_head",
]


class PortBasedModelFactory:
    """Model factory that uses port interfaces for all external dependencies."""

    def __init__(
        self,
        compute: ComputeBackend | None = None,
        neural_ops: NeuralOps | None = None,
        storage: StorageService | None = None,
        model_storage: ModelStorageService | None = None,
        monitoring: MonitoringService | None = None,
    ):
        """Initialize factory with port implementations.
        
        If ports are not provided, they will be obtained from the global context.
        """
        context = get_context()
        
        self.compute = compute or context.compute
        self.neural_ops = neural_ops or context.neural_ops
        self.storage = storage or context.storage
        self.model_storage = model_storage or context.model_storage
        self.monitoring = monitoring or context.monitoring

    def create_model(
        self,
        model_type: ModelType,
        config: dict[str, Any] | BertConfig | ModernBertConfig | None = None,
        pretrained_path: str | Path | None = None,
        head_type: str | None = None,
        head_config: HeadConfig | dict | None = None,
        **kwargs,
    ) -> Any:
        """Create a model using port interfaces.
        
        This method demonstrates how model creation can be abstracted
        from specific framework implementations.
        """
        # Log model creation
        self.monitoring.info(
            f"Creating model: {model_type}",
            model_type=model_type,
            head_type=head_type,
        )
        
        # Timer for performance monitoring
        with self.monitoring.timer("model_creation", tags={"model_type": model_type}):
            # Create model based on type
            if model_type == "bert_core":
                model = self._create_bert_core(config, **kwargs)
            elif model_type == "bert_with_head":
                model = self._create_bert_with_head(
                    config, head_type, head_config, **kwargs
                )
            elif model_type == "modernbert_core":
                model = self._create_modernbert_core(config, **kwargs)
            elif model_type == "modernbert_with_head":
                model = self._create_modernbert_with_head(
                    config, head_type, head_config, **kwargs
                )
            else:
                raise ValueError(f"Unknown model type: {model_type}")
            
            # Load pretrained weights if provided
            if pretrained_path:
                self._load_pretrained_weights(model, pretrained_path)
            
            # Log model statistics
            param_count = self._count_parameters(model)
            self.monitoring.metric(
                "model.parameters",
                param_count,
                tags={"model_type": model_type}
            )
            self.monitoring.info(
                f"Model created with {param_count:,} parameters"
            )
            
            return model

    def _create_bert_core(
        self,
        config: dict[str, Any] | BertConfig | None,
        **kwargs
    ) -> Any:
        """Create BERT core model."""
        # Import here to avoid circular dependencies
        from .bert import BertCore, create_bert_core
        
        # For now, delegate to existing implementation
        # In future, this would use compute port for framework-agnostic creation
        return create_bert_core(config=config, **kwargs)

    def _create_bert_with_head(
        self,
        config: dict[str, Any] | BertConfig | None,
        head_type: str | None,
        head_config: HeadConfig | dict | None,
        **kwargs
    ) -> Any:
        """Create BERT with head."""
        from .bert import create_bert_with_head
        
        return create_bert_with_head(
            config=config,
            head_type=head_type,
            head_config=head_config,
            **kwargs
        )

    def _create_modernbert_core(
        self,
        config: dict[str, Any] | ModernBertConfig | None,
        **kwargs
    ) -> Any:
        """Create ModernBERT core model."""
        from .bert import create_modernbert_core
        
        return create_modernbert_core(config=config, **kwargs)

    def _create_modernbert_with_head(
        self,
        config: dict[str, Any] | ModernBertConfig | None,
        head_type: str | None,
        head_config: HeadConfig | dict | None,
        **kwargs
    ) -> Any:
        """Create ModernBERT with head."""
        from .bert import BertWithHead, create_modernbert_core
        from .heads import create_head
        
        # Create core model
        bert_core = create_modernbert_core(config=config, **kwargs)
        
        # Create head
        num_labels = kwargs.get("num_labels", 2)
        head = create_head(
            head_type or "binary_classification",
            bert_core.config.hidden_size,
            num_labels,
            head_config
        )
        
        # Combine into BertWithHead
        return BertWithHead(
            bert_core,
            head,
            freeze_bert=kwargs.get("freeze_bert", False),
            freeze_bert_layers=kwargs.get("freeze_bert_layers"),
        )

    def _load_pretrained_weights(
        self,
        model: Any,
        path: str | Path
    ) -> None:
        """Load pretrained weights using storage port."""
        path = Path(path)
        
        with self.monitoring.timer("load_pretrained_weights"):
            if path.is_dir():
                # Load from directory using model storage
                loaded_model, metadata = self.model_storage.load_model(
                    path,
                    model_class=type(model),
                    load_optimizer=False,
                    load_metrics=False
                )
                
                # Update model weights
                if hasattr(model, "update"):
                    model.update(loaded_model.parameters())
                
                self.monitoring.info(
                    f"Loaded pretrained weights from {path}",
                    metadata=metadata
                )
            else:
                # Load from single file
                weights = self.storage.load(path)
                
                if hasattr(model, "update"):
                    model.update(weights)
                
                self.monitoring.info(f"Loaded pretrained weights from {path}")

    def _count_parameters(self, model: Any) -> int:
        """Count model parameters."""
        if hasattr(model, "num_parameters"):
            return model.num_parameters()
        elif hasattr(model, "parameters"):
            # Count manually
            return sum(p.size for p in model.parameters())
        else:
            return 0

    def save_model(
        self,
        model: Any,
        path: Path,
        include_optimizer: bool = True,
        include_metrics: bool = True,
        metadata: dict[str, Any] | None = None
    ) -> None:
        """Save model using storage port."""
        with self.monitoring.timer("save_model"):
            self.model_storage.save_model(
                model,
                path,
                include_optimizer=include_optimizer,
                include_metrics=include_metrics,
                metadata=metadata
            )
            
            self.monitoring.info(
                f"Saved model to {path}",
                include_optimizer=include_optimizer,
                include_metrics=include_metrics
            )

    def load_model(
        self,
        path: Path,
        model_class: type[Any] | None = None
    ) -> tuple[Any, dict[str, Any] | None]:
        """Load model using storage port."""
        with self.monitoring.timer("load_model"):
            model, metadata = self.model_storage.load_model(
                path,
                model_class=model_class,
                load_optimizer=True,
                load_metrics=True
            )
            
            self.monitoring.info(
                f"Loaded model from {path}",
                metadata=metadata
            )
            
            return model, metadata

    def create_from_checkpoint(
        self,
        checkpoint_path: Path
    ) -> Any:
        """Create and load model from checkpoint."""
        checkpoint_path = Path(checkpoint_path)
        
        with self.monitoring.span("create_from_checkpoint") as span:
            span.set_tag("checkpoint", str(checkpoint_path))
            
            # Load checkpoint data
            checkpoint = self.model_storage.load_checkpoint(checkpoint_path)
            
            # Extract model configuration
            model_config = checkpoint.get("model_config", {})
            model_type = checkpoint.get("model_type", "bert_with_head")
            
            # Create model
            model = self.create_model(
                model_type=model_type,
                config=model_config,
                **checkpoint.get("model_kwargs", {})
            )
            
            # Load weights
            if "model_state" in checkpoint:
                if hasattr(model, "update"):
                    model.update(checkpoint["model_state"])
            
            span.set_status("success")
            return model


# Global factory instance
_port_based_factory: PortBasedModelFactory | None = None


def get_port_based_factory() -> PortBasedModelFactory:
    """Get or create the global port-based factory."""
    global _port_based_factory
    if _port_based_factory is None:
        _port_based_factory = PortBasedModelFactory()
    return _port_based_factory


def create_model_with_ports(
    model_type: ModelType,
    **kwargs
) -> Any:
    """Create a model using the port-based factory.
    
    This is a convenience function that uses the global factory.
    """
    factory = get_port_based_factory()
    return factory.create_model(model_type, **kwargs)