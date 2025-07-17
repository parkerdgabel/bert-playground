"""Universal Kaggle data loader for any competition dataset.

This module provides a unified interface for loading and processing
any Kaggle dataset using the modular architecture with optimized
MLX-Data streaming and automatic feature detection.
"""

import mlx.core as mx
import mlx.data as dx
from typing import Dict, List, Optional, Callable, Any, Union, Iterator, Tuple
import pandas as pd
import numpy as np
from pathlib import Path
from loguru import logger
import json
import random
from enum import Enum

from .dataset_spec import (
    KaggleDatasetSpec, 
    ProblemType, 
    FeatureType, 
    OptimizationProfile,
    get_dataset_spec,
)
from .mlx_streaming import MLXDataPipeline, MLXStreamConfig, create_mlx_pipeline
from .text_templates import TitanicTextTemplates


class TextGenerationStrategy(Enum):
    """Strategies for generating text from tabular data."""
    TEMPLATE_BASED = "template_based"
    FEATURE_CONCATENATION = "feature_concatenation"
    NARRATIVE = "narrative"
    STRUCTURED = "structured"
    AUTO = "auto"


class UniversalTextGenerator:
    """Universal text generator for any Kaggle dataset."""
    
    def __init__(
        self,
        dataset_spec: KaggleDatasetSpec,
        strategy: TextGenerationStrategy = TextGenerationStrategy.AUTO,
    ):
        self.dataset_spec = dataset_spec
        self.strategy = strategy
        
        # Auto-select strategy if needed
        if self.strategy == TextGenerationStrategy.AUTO:
            self.strategy = self._auto_select_strategy()
        
        logger.info(f"Initialized text generator with strategy: {self.strategy}")
    
    def _auto_select_strategy(self) -> TextGenerationStrategy:
        """Auto-select the best text generation strategy."""
        # Use template-based for known datasets
        if self.dataset_spec.name == "titanic":
            return TextGenerationStrategy.TEMPLATE_BASED
        
        # Use feature concatenation for datasets with many categorical features
        if len(self.dataset_spec.categorical_columns) > 5:
            return TextGenerationStrategy.FEATURE_CONCATENATION
        
        # Use narrative for datasets with existing text columns
        if len(self.dataset_spec.text_columns) > 0:
            return TextGenerationStrategy.NARRATIVE
        
        # Default to structured approach
        return TextGenerationStrategy.STRUCTURED
    
    def generate_text(self, row_dict: Dict[str, Any]) -> str:
        """Generate text representation from a data row."""
        if self.strategy == TextGenerationStrategy.TEMPLATE_BASED:
            return self._generate_template_based(row_dict)
        elif self.strategy == TextGenerationStrategy.FEATURE_CONCATENATION:
            return self._generate_feature_concatenation(row_dict)
        elif self.strategy == TextGenerationStrategy.NARRATIVE:
            return self._generate_narrative(row_dict)
        elif self.strategy == TextGenerationStrategy.STRUCTURED:
            return self._generate_structured(row_dict)
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")
    
    def _generate_template_based(self, row_dict: Dict[str, Any]) -> str:
        """Generate text using dataset-specific templates."""
        if self.dataset_spec.name == "titanic":
            # Use the existing Titanic template system
            template_generator = TitanicTextTemplates()
            return template_generator.row_to_text(row_dict)
        else:
            # Fallback to structured approach for unknown datasets
            return self._generate_structured(row_dict)
    
    def _generate_feature_concatenation(self, row_dict: Dict[str, Any]) -> str:
        """Generate text by concatenating features with descriptions."""
        parts = []
        
        # Process categorical features
        for col in self.dataset_spec.categorical_columns:
            if col in row_dict and pd.notna(row_dict[col]):
                parts.append(f"{col}: {row_dict[col]}")
        
        # Process numerical features
        for col in self.dataset_spec.numerical_columns:
            if col in row_dict and pd.notna(row_dict[col]):
                value = row_dict[col]
                if isinstance(value, float):
                    parts.append(f"{col}: {value:.2f}")
                else:
                    parts.append(f"{col}: {value}")
        
        # Process text features
        for col in self.dataset_spec.text_columns:
            if col in row_dict and pd.notna(row_dict[col]):
                text_value = str(row_dict[col])[:100]  # Truncate long text
                parts.append(f"{col}: {text_value}")
        
        return " | ".join(parts) if parts else "No data available"
    
    def _generate_narrative(self, row_dict: Dict[str, Any]) -> str:
        """Generate narrative text incorporating existing text columns."""
        narrative_parts = []
        
        # Start with existing text columns
        for col in self.dataset_spec.text_columns:
            if col in row_dict and pd.notna(row_dict[col]):
                narrative_parts.append(str(row_dict[col]))
        
        # Add contextual information from other features
        context_parts = []
        
        # Add key categorical features
        key_categoricals = self.dataset_spec.categorical_columns[:3]
        for col in key_categoricals:
            if col in row_dict and pd.notna(row_dict[col]):
                context_parts.append(f"{col} is {row_dict[col]}")
        
        # Add key numerical features
        key_numericals = self.dataset_spec.numerical_columns[:2]
        for col in key_numericals:
            if col in row_dict and pd.notna(row_dict[col]):
                value = row_dict[col]
                if isinstance(value, float):
                    context_parts.append(f"{col} is {value:.2f}")
                else:
                    context_parts.append(f"{col} is {value}")
        
        if context_parts:
            context = "Additional context: " + ", ".join(context_parts)
            narrative_parts.append(context)
        
        return " ".join(narrative_parts) if narrative_parts else "No narrative available"
    
    def _generate_structured(self, row_dict: Dict[str, Any]) -> str:
        """Generate structured text description."""
        parts = []
        
        # Create a structured description
        parts.append("Data record:")
        
        # Process features by importance (categorical first, then numerical)
        all_features = (
            self.dataset_spec.categorical_columns + 
            self.dataset_spec.numerical_columns +
            self.dataset_spec.text_columns
        )
        
        feature_descriptions = []
        for col in all_features[:10]:  # Limit to top 10 features
            if col in row_dict and pd.notna(row_dict[col]):
                value = row_dict[col]
                if isinstance(value, float):
                    feature_descriptions.append(f"{col}={value:.2f}")
                elif isinstance(value, str) and len(value) > 50:
                    # Truncate long strings
                    feature_descriptions.append(f"{col}={value[:47]}...")
                else:
                    feature_descriptions.append(f"{col}={value}")
        
        if feature_descriptions:
            parts.append(", ".join(feature_descriptions))
        else:
            parts.append("No features available")
        
        return " ".join(parts)


class UniversalKaggleLoader:
    """Universal data loader for any Kaggle competition dataset."""
    
    def __init__(
        self,
        train_path: Optional[str] = None,
        test_path: Optional[str] = None,
        val_path: Optional[str] = None,
        dataset_spec: Optional[KaggleDatasetSpec] = None,
        target_column: Optional[str] = None,
        # Model parameters
        tokenizer_name: str = "answerdotai/ModernBERT-base",
        max_length: int = 256,
        batch_size: int = 32,
        # Text generation
        text_strategy: TextGenerationStrategy = TextGenerationStrategy.AUTO,
        augment: bool = True,
        # Optimization
        optimization_profile: Optional[OptimizationProfile] = None,
        **mlx_config_kwargs
    ):
        """Initialize universal Kaggle loader.
        
        Args:
            train_path: Path to training CSV file
            test_path: Path to test CSV file (optional)
            val_path: Path to validation CSV file (optional)
            dataset_spec: Dataset specification (auto-detected if None)
            target_column: Target column name (required if dataset_spec is None)
            tokenizer_name: Tokenizer to use
            max_length: Maximum sequence length
            batch_size: Batch size
            text_strategy: Text generation strategy
            augment: Whether to apply data augmentation
            optimization_profile: Optimization profile (auto-detected if None)
            **mlx_config_kwargs: Additional MLX configuration parameters
        """
        self.train_path = train_path
        self.test_path = test_path
        self.val_path = val_path
        self.tokenizer_name = tokenizer_name
        self.max_length = max_length
        self.batch_size = batch_size
        self.text_strategy = text_strategy
        self.augment = augment
        
        # Auto-detect or validate dataset specification
        if dataset_spec is None:
            if train_path is None:
                raise ValueError("Either train_path or dataset_spec must be provided")
            if target_column is None:
                raise ValueError("target_column is required when dataset_spec is None")
            
            logger.info(f"Auto-detecting dataset specification from: {train_path}")
            self.dataset_spec = KaggleDatasetSpec.from_csv_analysis(train_path, target_column)
        else:
            self.dataset_spec = dataset_spec
        
        # Override optimization profile if provided
        if optimization_profile is not None:
            self.dataset_spec.optimization_profile = optimization_profile
        
        logger.info(f"Dataset: {self.dataset_spec.name}")
        logger.info(f"Problem type: {self.dataset_spec.problem_type}")
        logger.info(f"Optimization profile: {self.dataset_spec.optimization_profile}")
        
        # Create MLX stream configuration
        self.mlx_config = MLXStreamConfig(
            optimization_profile=self.dataset_spec.optimization_profile,
            batch_size=batch_size,
            max_length=max_length,
            **mlx_config_kwargs
        )
        
        # Initialize text generator
        self.text_generator = UniversalTextGenerator(
            dataset_spec=self.dataset_spec,
            strategy=text_strategy,
        )
        
        # Initialize data pipelines
        self._init_data_pipelines()
        
        logger.info("UniversalKaggleLoader initialization complete")
    
    def _init_data_pipelines(self):
        """Initialize MLX data pipelines."""
        self.train_pipeline = None
        self.val_pipeline = None
        self.test_pipeline = None
        
        # Create text transformation function
        def text_transform_fn(sample: Dict[str, Any]) -> Dict[str, Any]:
            """Transform a sample by adding text representation."""
            try:
                text = self.text_generator.generate_text(sample)
                
                # Apply augmentation if enabled and this is training data
                if self.augment and hasattr(self, '_is_training_transform') and self._is_training_transform:
                    text = self._augment_text(text)
                
                # Add text to sample
                sample_copy = sample.copy()
                sample_copy['text'] = text
                return sample_copy
                
            except Exception as e:
                logger.warning(f"Text generation failed for sample: {e}")
                sample_copy = sample.copy()
                sample_copy['text'] = "Error generating text"
                return sample_copy
        
        # Store transform function
        self.text_transform_fn = text_transform_fn
        
        # Initialize training pipeline
        if self.train_path:
            logger.info("Initializing training pipeline")
            self._is_training_transform = True
            self.train_pipeline = create_mlx_pipeline(
                csv_path=self.train_path,
                dataset_spec=self.dataset_spec,
                tokenizer_name=self.tokenizer_name,
                config=self.mlx_config,
                text_transform_fn=text_transform_fn,
            )
        
        # Initialize validation pipeline
        if self.val_path:
            logger.info("Initializing validation pipeline")
            self._is_training_transform = False
            val_config = MLXStreamConfig(
                optimization_profile=self.dataset_spec.optimization_profile,
                batch_size=self.batch_size,
                max_length=self.max_length,
                # Disable some optimizations for validation
                enable_dynamic_batching=False,
            )
            self.val_pipeline = create_mlx_pipeline(
                csv_path=self.val_path,
                dataset_spec=self.dataset_spec,
                tokenizer_name=self.tokenizer_name,
                config=val_config,
                text_transform_fn=text_transform_fn,
            )
        
        # Initialize test pipeline
        if self.test_path:
            logger.info("Initializing test pipeline")
            self._is_training_transform = False
            test_config = MLXStreamConfig(
                optimization_profile=self.dataset_spec.optimization_profile,
                batch_size=self.batch_size,
                max_length=self.max_length,
                # Disable optimizations that might affect reproducibility
                enable_dynamic_batching=False,
                prefetch_size=1,
            )
            
            # Create test spec without target column
            test_spec = KaggleDatasetSpec(
                name=self.dataset_spec.name + "_test",
                problem_type=self.dataset_spec.problem_type,
                target_column="",  # No target in test data
                categorical_columns=self.dataset_spec.categorical_columns,
                numerical_columns=self.dataset_spec.numerical_columns,
                text_columns=self.dataset_spec.text_columns,
                datetime_columns=self.dataset_spec.datetime_columns,
                id_columns=self.dataset_spec.id_columns,
                optimization_profile=self.dataset_spec.optimization_profile,
            )
            
            self.test_pipeline = create_mlx_pipeline(
                csv_path=self.test_path,
                dataset_spec=test_spec,
                tokenizer_name=self.tokenizer_name,
                config=test_config,
                text_transform_fn=text_transform_fn,
            )
    
    def _augment_text(self, text: str) -> str:
        """Apply text augmentation for training data."""
        if not self.augment:
            return text
        
        # Simple augmentation strategies
        augmentations = [
            text,  # Original text
            f"Training sample: {text}",
            f"Dataset entry: {text}",
            f"Competition data: {text}",
        ]
        
        # Add dataset-specific augmentations
        if self.dataset_spec.name == "titanic":
            augmentations.extend([
                f"Titanic passenger information: {text}",
                f"Historical record from 1912: {text}",
            ])
        
        return random.choice(augmentations)
    
    def get_train_loader(self) -> Iterator[Dict[str, mx.array]]:
        """Get training data iterator."""
        if self.train_pipeline is None:
            raise ValueError("No training data provided")
        
        self._is_training_transform = True
        return self.train_pipeline.get_stream_iterator(is_training=True)
    
    def get_val_loader(self) -> Iterator[Dict[str, mx.array]]:
        """Get validation data iterator."""
        if self.val_pipeline is None:
            # Use training data for validation if no separate validation set
            if self.train_pipeline is None:
                raise ValueError("No validation or training data provided")
            logger.info("Using training data for validation (no separate val set)")
            self._is_training_transform = False
            return self.train_pipeline.get_stream_iterator(is_training=False)
        
        self._is_training_transform = False
        return self.val_pipeline.get_stream_iterator(is_training=False)
    
    def get_test_loader(self) -> Iterator[Dict[str, mx.array]]:
        """Get test data iterator."""
        if self.test_pipeline is None:
            raise ValueError("No test data provided")
        
        self._is_training_transform = False
        return self.test_pipeline.get_stream_iterator(is_training=False)
    
    def get_dataset_info(self) -> Dict[str, Any]:
        """Get comprehensive dataset information."""
        info = {
            "dataset_spec": self.dataset_spec.to_dict(),
            "paths": {
                "train": self.train_path,
                "val": self.val_path,
                "test": self.test_path,
            },
            "configuration": {
                "tokenizer": self.tokenizer_name,
                "max_length": self.max_length,
                "batch_size": self.batch_size,
                "text_strategy": self.text_strategy.value,
                "augment": self.augment,
                "optimization_profile": self.dataset_spec.optimization_profile.value,
            },
            "mlx_config": {
                "prefetch_size": self.mlx_config.prefetch_size,
                "num_threads": self.mlx_config.num_threads,
                "buffer_size": self.mlx_config.buffer_size,
                "enable_dynamic_batching": self.mlx_config.enable_dynamic_batching,
                "enable_shape_optimization": self.mlx_config.enable_shape_optimization,
            }
        }
        
        # Add pipeline info if available
        if self.train_pipeline:
            info["train_pipeline"] = self.train_pipeline.get_dataset_info()
        
        return info
    
    def save_config(self, config_path: str) -> None:
        """Save loader configuration to file."""
        config = {
            "dataset_spec": self.dataset_spec.to_dict(),
            "train_path": self.train_path,
            "val_path": self.val_path,
            "test_path": self.test_path,
            "tokenizer_name": self.tokenizer_name,
            "max_length": self.max_length,
            "batch_size": self.batch_size,
            "text_strategy": self.text_strategy.value,
            "augment": bool(self.augment),  # Ensure it's a Python bool
        }
        
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        logger.info(f"Configuration saved to: {config_path}")
    
    @classmethod
    def from_config(cls, config_path: str, **override_kwargs) -> "UniversalKaggleLoader":
        """Load configuration from file."""
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        # Restore dataset spec
        dataset_spec = KaggleDatasetSpec.from_dict(config["dataset_spec"])
        
        # Override with any provided kwargs
        config.update(override_kwargs)
        config["dataset_spec"] = dataset_spec
        
        return cls(**config)


# Convenience functions
def create_universal_loader(
    train_path: str,
    target_column: str,
    test_path: Optional[str] = None,
    val_path: Optional[str] = None,
    **kwargs
) -> UniversalKaggleLoader:
    """Convenience function to create universal loader with auto-detection."""
    return UniversalKaggleLoader(
        train_path=train_path,
        test_path=test_path,
        val_path=val_path,
        target_column=target_column,
        **kwargs
    )


def create_titanic_loader(
    train_path: str = "data/titanic/train.csv",
    test_path: Optional[str] = "data/titanic/test.csv",
    val_path: Optional[str] = None,
    **kwargs
) -> UniversalKaggleLoader:
    """Convenience function to create Titanic loader."""
    from .dataset_spec import TITANIC_SPEC
    
    return UniversalKaggleLoader(
        train_path=train_path,
        test_path=test_path,
        val_path=val_path,
        dataset_spec=TITANIC_SPEC,
        **kwargs
    )