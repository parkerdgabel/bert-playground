"""Model registry for managing available model types.

This module provides a registry pattern for model types, allowing for
easy extension and discovery of available models.
"""

from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from loguru import logger


ModelFactory = Callable[..., Any]


@dataclass
class ModelInfo:
    """Information about a registered model."""
    
    name: str
    factory: ModelFactory
    description: str = ""
    category: str = "general"
    tags: list[str] = field(default_factory=list)


@dataclass
class ModelRegistry:
    """Registry for available model types."""
    
    _models: dict[str, ModelInfo] = field(default_factory=dict)
    
    def register(
        self,
        name: str,
        factory: ModelFactory,
        description: str = "",
        category: str = "general",
        tags: Optional[list[str]] = None,
    ) -> None:
        """Register a new model type.
        
        Args:
            name: Unique name for the model
            factory: Factory function to create the model
            description: Human-readable description
            category: Model category (e.g., "core", "classification", "lora")
            tags: Optional tags for filtering
        """
        if name in self._models:
            logger.warning(f"Overwriting existing model: {name}")
            
        self._models[name] = ModelInfo(
            name=name,
            factory=factory,
            description=description,
            category=category,
            tags=tags or [],
        )
        
        logger.debug(f"Registered model: {name} (category: {category})")
    
    def unregister(self, name: str) -> None:
        """Unregister a model type.
        
        Args:
            name: Name of model to unregister
        """
        if name in self._models:
            del self._models[name]
            logger.debug(f"Unregistered model: {name}")
        else:
            logger.warning(f"Model not found: {name}")
    
    def get(self, name: str) -> Optional[ModelInfo]:
        """Get model info by name.
        
        Args:
            name: Model name
            
        Returns:
            ModelInfo if found, None otherwise
        """
        return self._models.get(name)
    
    def create(self, name: str, **kwargs) -> Any:
        """Create a model by name.
        
        Args:
            name: Model name
            **kwargs: Arguments for model factory
            
        Returns:
            Created model instance
            
        Raises:
            ValueError: If model not found
        """
        model_info = self.get(name)
        if model_info is None:
            available = ", ".join(self.list_models())
            raise ValueError(
                f"Unknown model: {name}. Available models: {available}"
            )
            
        return model_info.factory(**kwargs)
    
    def list_models(self, category: Optional[str] = None) -> list[str]:
        """List available model names.
        
        Args:
            category: Optional category filter
            
        Returns:
            List of model names
        """
        if category is None:
            return list(self._models.keys())
            
        return [
            name for name, info in self._models.items()
            if info.category == category
        ]
    
    def list_categories(self) -> list[str]:
        """List available model categories.
        
        Returns:
            List of unique categories
        """
        categories = {info.category for info in self._models.values()}
        return sorted(categories)
    
    def search_by_tags(self, tags: list[str]) -> list[str]:
        """Search models by tags.
        
        Args:
            tags: Tags to search for (models must have all tags)
            
        Returns:
            List of model names matching all tags
        """
        results = []
        for name, info in self._models.items():
            if all(tag in info.tags for tag in tags):
                results.append(name)
        return results
    
    def get_info(self) -> dict[str, dict[str, Any]]:
        """Get information about all registered models.
        
        Returns:
            Dictionary mapping model names to their info
        """
        return {
            name: {
                "description": info.description,
                "category": info.category,
                "tags": info.tags,
            }
            for name, info in self._models.items()
        }
    
    def register_defaults(self, model_builder: Any) -> None:
        """Register default model types.
        
        Args:
            model_builder: ModelBuilder instance to use for factories
        """
        # Core models
        self.register(
            "bert-core",
            lambda **kwargs: model_builder.build_core("bert_core", **kwargs),
            description="Classic BERT core model",
            category="core",
            tags=["bert", "core"],
        )
        
        self.register(
            "modernbert-core",
            lambda **kwargs: model_builder.build_core("modernbert_core", **kwargs),
            description="ModernBERT core model with RoPE and GeGLU",
            category="core",
            tags=["modernbert", "core"],
        )
        
        # Classification models
        self.register(
            "bert-binary",
            lambda **kwargs: model_builder.build_with_head(
                "bert_with_head",
                head_type="binary_classification",
                **kwargs
            ),
            description="BERT for binary classification",
            category="classification",
            tags=["bert", "classification", "binary"],
        )
        
        self.register(
            "bert-multiclass",
            lambda **kwargs: model_builder.build_with_head(
                "bert_with_head",
                head_type="multiclass_classification",
                **kwargs
            ),
            description="BERT for multi-class classification",
            category="classification",
            tags=["bert", "classification", "multiclass"],
        )
        
        self.register(
            "bert-regression",
            lambda **kwargs: model_builder.build_with_head(
                "bert_with_head",
                head_type="regression",
                **kwargs
            ),
            description="BERT for regression tasks",
            category="regression",
            tags=["bert", "regression"],
        )
        
        # ModernBERT variants
        self.register(
            "modernbert-binary",
            lambda **kwargs: model_builder.build_with_head(
                "modernbert_with_head",
                head_type="binary_classification",
                **kwargs
            ),
            description="ModernBERT for binary classification",
            category="classification",
            tags=["modernbert", "classification", "binary"],
        )
        
        self.register(
            "modernbert-multiclass",
            lambda **kwargs: model_builder.build_with_head(
                "modernbert_with_head",
                head_type="multiclass_classification",
                **kwargs
            ),
            description="ModernBERT for multi-class classification",
            category="classification",
            tags=["modernbert", "classification", "multiclass"],
        )
        
        logger.info(f"Registered {len(self._models)} default models")