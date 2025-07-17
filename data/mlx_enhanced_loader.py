"""Compatibility wrapper for mlx_enhanced_loader.

This module redirects to the enhanced unified loader for backward compatibility.
"""

import warnings
from .enhanced_unified_loader import EnhancedUnifiedDataPipeline


class MLXEnhancedDataPipeline(EnhancedUnifiedDataPipeline):
    """Compatibility wrapper for MLXEnhancedDataPipeline.
    
    This class is deprecated. Use EnhancedUnifiedDataPipeline instead.
    """
    
    def __init__(self, *args, **kwargs):
        warnings.warn(
            "MLXEnhancedDataPipeline is deprecated. "
            "Use EnhancedUnifiedDataPipeline from data.enhanced_unified_loader instead.",
            DeprecationWarning,
            stacklevel=2
        )
        
        # Map old parameter names to new ones
        if "tokenizer" in kwargs:
            # Old API passed tokenizer instance, new API uses tokenizer name
            tokenizer = kwargs.pop("tokenizer")
            kwargs["tokenizer_name"] = tokenizer.name_or_path
        
        if "enable_augmentation" in kwargs:
            kwargs["augment"] = kwargs.pop("enable_augmentation")
        
        if "memory_map" in kwargs:
            # Memory mapping is not yet implemented
            kwargs.pop("memory_map")
        
        super().__init__(*args, **kwargs)