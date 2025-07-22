"""Main factory facade that delegates to focused builders using hexagonal architecture.

This factory maintains API compatibility while internally using focused builders
following the Single Responsibility Principle and Dependency Injection.
"""

from pathlib import Path
from typing import Any, Literal

from loguru import logger

from core.bootstrap import get_service
from core.ports.compute import Module
from data.core.base import CompetitionType
from utils.logging_utils import bind_context, catch_and_log

from .bert import BertConfig, ModernBertConfig
from .builders import (
    CheckpointManager,
    CompetitionAnalysis,
    CompetitionBuilder,
    CoreModelBuilder,
    LoRABuilder,
    ModelWithHeadBuilder,
)
from .heads.base import HeadConfig
from .lora import LoRAAdapter

ModelType = Literal[
    "bert_core",
    "bert_with_head", 
    "modernbert_core",
    "modernbert_with_head",
]


class ModelFactory:
    """Main model factory using hexagonal architecture and focused builders."""

    def __init__(self):
        """Initialize with lazy builders to avoid circular imports."""
        self._core_builder = None
        self._head_builder = None
        self._lora_builder = None
        self._checkpoint_manager = None
        self._competition_builder = None

    @property
    def core_builder(self):
        """Lazy initialization of core builder."""
        if self._core_builder is None:
            self._core_builder = CoreModelBuilder()
        return self._core_builder

    @property
    def head_builder(self):
        """Lazy initialization of head builder."""
        if self._head_builder is None:
            self._head_builder = ModelWithHeadBuilder()
        return self._head_builder

    @property
    def lora_builder(self):
        """Lazy initialization of LoRA builder."""
        if self._lora_builder is None:
            self._lora_builder = LoRABuilder()
        return self._lora_builder

    @property
    def checkpoint_manager(self):
        """Lazy initialization of checkpoint manager."""
        if self._checkpoint_manager is None:
            self._checkpoint_manager = CheckpointManager()
        return self._checkpoint_manager

    @property
    def competition_builder(self):
        """Lazy initialization of competition builder."""
        if self._competition_builder is None:
            self._competition_builder = CompetitionBuilder()
        return self._competition_builder

    @catch_and_log(ValueError, "Model creation failed", reraise=True)
    def create_model(
        self,
        model_type: ModelType = "bert_with_head",
        config: dict[str, Any] | BertConfig | ModernBertConfig | None = None,
        pretrained_path: str | Path | None = None,
        head_type: str | None = None,
        head_config: HeadConfig | dict | None = None,
        **kwargs,
    ) -> Module:
        """Create a model using focused builders.

        Args:
            model_type: Type of model to create
            config: Model configuration
            pretrained_path: Path to pretrained weights
            head_type: Type of head to attach (for models with heads)
            head_config: Head configuration
            **kwargs: Additional arguments

        Returns:
            Initialized model
        """
        log = bind_context(model_type=model_type, head_type=head_type)
        log.info(f"Creating model: {model_type}")

        # Delegate to appropriate builder
        if model_type in ["bert_core", "modernbert_core"]:
            model = self.core_builder.build_core_model(model_type, config, **kwargs)
        elif model_type in ["bert_with_head", "modernbert_with_head"]:
            model = self.head_builder.build_model_with_head(
                model_type, config, head_type, head_config, **kwargs
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")

        # Load pretrained weights if provided
        if pretrained_path:
            self.checkpoint_manager.load_pretrained_weights(model, pretrained_path)

        # Log model statistics
        if hasattr(model, "num_parameters"):
            param_count = model.num_parameters()
            log.info(f"Model created with {param_count:,} parameters")

        return model

    def create_model_with_lora(
        self,
        base_model: Module | None = None,
        model_type: ModelType | None = None,
        lora_config: Any = None,
        inject_lora: bool = True,
        verbose: bool = False,
        **model_kwargs,
    ) -> tuple[Module, LoRAAdapter]:
        """Create a model with LoRA adapters."""
        # Create base model if not provided
        if base_model is None:
            if model_type is None:
                model_type = "bert_with_head"
            base_model = self.create_model(model_type, **model_kwargs)

        return self.lora_builder.build_lora_model(
            base_model, lora_config, inject_lora, verbose
        )

    def create_model_from_checkpoint(self, checkpoint_path: str | Path) -> Module:
        """Create and load a model from a checkpoint."""
        return self.checkpoint_manager.load_model_from_checkpoint(checkpoint_path)

    def create_kaggle_classifier(
        self,
        task_type: str,
        num_classes: int | None = None,
        **kwargs,
    ) -> Module:
        """Create classifier optimized for Kaggle competitions."""
        return self.competition_builder.build_kaggle_classifier(
            task_type, num_classes, **kwargs
        )

    def create_competition_classifier(
        self,
        data_path: str,
        target_column: str,
        model_name: str = "answerdotai/ModernBERT-base",
        auto_optimize: bool = True,
        **kwargs,
    ) -> tuple[Module, CompetitionAnalysis]:
        """Analyze dataset and create optimized classifier."""
        return self.competition_builder.build_competition_classifier(
            data_path, target_column, model_name, auto_optimize, **kwargs
        )

    def create_kaggle_lora_model(
        self,
        competition_type: str | CompetitionType,
        data_path: str | None = None,
        lora_preset: str | None = None,
        auto_select_preset: bool = True,
        **kwargs,
    ) -> tuple[Module, LoRAAdapter]:
        """Create optimized LoRA model for Kaggle competition."""
        return self.competition_builder.build_kaggle_lora_model(
            competition_type, data_path, lora_preset, auto_select_preset, **kwargs
        )

    def analyze_competition_dataset(
        self, data_path: str, target_column: str
    ) -> CompetitionAnalysis:
        """Analyze a competition dataset."""
        return self.competition_builder.analyze_competition_dataset(data_path, target_column)


# Global factory instance
_factory = ModelFactory()

# Convenience functions that delegate to the factory
def create_model(*args, **kwargs) -> Module:
    """Create a model using the global factory."""
    return _factory.create_model(*args, **kwargs)

def create_model_with_lora(*args, **kwargs) -> tuple[Module, LoRAAdapter]:
    """Create a model with LoRA using the global factory.""" 
    return _factory.create_model_with_lora(*args, **kwargs)

def create_model_from_checkpoint(*args, **kwargs) -> Module:
    """Create a model from checkpoint using the global factory."""
    return _factory.create_model_from_checkpoint(*args, **kwargs)

def create_kaggle_classifier(*args, **kwargs) -> Module:
    """Create a Kaggle classifier using the global factory."""
    return _factory.create_kaggle_classifier(*args, **kwargs)

def create_competition_classifier(*args, **kwargs) -> tuple[Module, CompetitionAnalysis]:
    """Create a competition classifier using the global factory."""
    return _factory.create_competition_classifier(*args, **kwargs)

def create_kaggle_lora_model(*args, **kwargs) -> tuple[Module, LoRAAdapter]:
    """Create a Kaggle LoRA model using the global factory."""
    return _factory.create_kaggle_lora_model(*args, **kwargs)

def analyze_competition_dataset(*args, **kwargs) -> CompetitionAnalysis:
    """Analyze a competition dataset using the global factory."""
    return _factory.analyze_competition_dataset(*args, **kwargs)