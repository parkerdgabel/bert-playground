"""Data loading and processing modules for MLX-based BERT training."""

# New universal system
from .dataset_spec import (
    KaggleDatasetSpec,
    ProblemType,
    FeatureType,
    OptimizationProfile,
    get_dataset_spec,
    register_dataset_spec,
    TITANIC_SPEC,
)

from .mlx_streaming import (
    MLXStreamConfig,
    MLXCSVStreamer,
    MLXDataPipeline,
    create_mlx_pipeline,
)

from .universal_loader import (
    UniversalKaggleLoader,
    UniversalTextGenerator,
    TextGenerationStrategy,
    create_universal_loader,
    create_titanic_loader,
)

# Legacy components (still available)
from .unified_loader import (
    UnifiedTitanicDataPipeline,
    OptimizationLevel,
)

from .enhanced_unified_loader import EnhancedUnifiedDataPipeline

from .text_templates import TitanicTextTemplates

__all__ = [
    # New universal system
    "KaggleDatasetSpec",
    "ProblemType", 
    "FeatureType",
    "OptimizationProfile",
    "get_dataset_spec",
    "register_dataset_spec",
    "TITANIC_SPEC",
    "MLXStreamConfig",
    "MLXCSVStreamer", 
    "MLXDataPipeline",
    "create_mlx_pipeline",
    "UniversalKaggleLoader",
    "UniversalTextGenerator",
    "TextGenerationStrategy",
    "create_universal_loader",
    "create_titanic_loader",
    # Legacy components
    "UnifiedTitanicDataPipeline",
    "EnhancedUnifiedDataPipeline",
    "OptimizationLevel",
    "TitanicTextTemplates",
]