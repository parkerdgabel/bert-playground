"""Universal Kaggle data loader for any competition dataset.

This module provides the SINGLE, unified interface for loading and processing
any Kaggle dataset. It combines all the best features from the previous
multiple dataloader implementations into one comprehensive solution.

Features:
- Protocol compliance with MLXTrainer
- Advanced caching (text conversion + tokenization)
- MLX-native streaming with optimization profiles
- Automatic dataset detection and text generation
- Support for all Kaggle competition types
- Modular configuration system integration
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
import hashlib
import pickle
import time
from sklearn.model_selection import train_test_split

from .dataset_spec import (
    KaggleDatasetSpec, 
    ProblemType, 
    FeatureType, 
    OptimizationProfile,
    get_dataset_spec,
)
from .mlx_streaming import MLXDataPipeline, MLXStreamConfig, create_mlx_pipeline
from .text_templates import TitanicTextTemplates

# Import text conversion and config systems for advanced features
try:
    from .text_conversion import TextConverterFactory
    from .configs import ConfigFactory, DataLoaderConfig, DatasetConfig
    from .configs.base_config import ExperimentConfig
    from .utils.caching import DiskCache, TokenizationCache
    ADVANCED_FEATURES_AVAILABLE = True
except ImportError:
    ADVANCED_FEATURES_AVAILABLE = False
    logger.warning("Advanced features not available - using basic implementation")

# Import tokenizer wrapper for backend flexibility
try:
    from embeddings.tokenizer_wrapper import TokenizerWrapper
    USE_TOKENIZER_WRAPPER = True
except ImportError:
    USE_TOKENIZER_WRAPPER = False
    from transformers import AutoTokenizer
    logger.warning("TokenizerWrapper not available, using HuggingFace tokenizer")


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
    """Universal data loader for any Kaggle competition dataset.
    
    This is the SINGLE dataloader for the entire codebase. It combines:
    - Protocol compliance with MLXTrainer (from ModularMLXDataLoader)
    - Advanced caching and configuration (from ModularMLXDataLoader)
    - Universal dataset support (from UniversalKaggleLoader)
    - MLX-native streaming (from MLXDataPipeline)
    - Flexible text generation (from UniversalTextGenerator)
    """
    
    def __init__(
        self,
        # Data paths
        train_path: Optional[str] = None,
        test_path: Optional[str] = None,
        val_path: Optional[str] = None,
        
        # Dataset configuration
        dataset_spec: Optional[KaggleDatasetSpec] = None,
        target_column: Optional[str] = None,
        
        # Configuration object support (from ModularMLXDataLoader)
        config: Optional[Union["ExperimentConfig", "TrainingConfig", str, Path]] = None,
        data: Optional[pd.DataFrame] = None,
        split: str = "train",
        
        # Model parameters
        tokenizer_name: str = "answerdotai/ModernBERT-base",
        max_length: int = 256,
        batch_size: int = 32,
        
        # Text generation
        text_strategy: TextGenerationStrategy = TextGenerationStrategy.AUTO,
        augment: bool = True,
        
        # Advanced features (from ModularMLXDataLoader)
        enable_cache: bool = True,
        cache_dir: Optional[str] = None,
        cache_text_conversion: bool = True,
        cache_tokenization: bool = True,
        
        # Tokenizer backend
        tokenizer_backend: str = "auto",
        
        # MLX streaming configuration
        shuffle: bool = True,
        shuffle_buffer_size: int = 1000,
        prefetch_size: int = 4,
        num_workers: int = 4,
        drop_last: bool = False,
        
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
            config: Configuration object, file path, or preset name (ModularMLXDataLoader compatibility)
            data: Optional pre-loaded data (ModularMLXDataLoader compatibility)
            split: Data split to use ("train", "val", "test")
            tokenizer_name: Tokenizer to use
            max_length: Maximum sequence length
            batch_size: Batch size
            text_strategy: Text generation strategy
            augment: Whether to apply data augmentation
            enable_cache: Enable caching system
            cache_dir: Cache directory path
            cache_text_conversion: Cache text conversion results
            cache_tokenization: Cache tokenization results
            tokenizer_backend: Tokenizer backend ("auto", "mlx", "huggingface")
            shuffle: Whether to shuffle data
            shuffle_buffer_size: Shuffle buffer size
            prefetch_size: Prefetch size for MLX streaming
            num_workers: Number of worker threads
            drop_last: Drop incomplete batches
            optimization_profile: Optimization profile (auto-detected if None)
            **mlx_config_kwargs: Additional MLX configuration parameters
        """
        # Store configuration
        self.train_path = train_path
        self.test_path = test_path
        self.val_path = val_path
        self.tokenizer_name = tokenizer_name
        self.max_length = max_length
        self.batch_size = batch_size
        self.text_strategy = text_strategy
        self.augment = augment
        self.split = split
        self.data = data
        
        # Advanced features
        self.enable_cache = enable_cache
        self.cache_dir = cache_dir
        self.cache_text_conversion = cache_text_conversion
        self.cache_tokenization = cache_tokenization
        self.tokenizer_backend = tokenizer_backend
        
        # MLX streaming config
        self.shuffle = shuffle
        self.shuffle_buffer_size = shuffle_buffer_size
        self.prefetch_size = prefetch_size
        self.num_workers = num_workers
        self.drop_last = drop_last
        
        # Handle configuration object (ModularMLXDataLoader compatibility)
        if config is not None:
            self._process_config_object(config)
        
        # Initialize caches
        self.caches = {}
        self._initialize_caches()
        
        # Auto-detect or validate dataset specification
        if dataset_spec is None:
            if train_path is None and data is None:
                raise ValueError("Either train_path, data, or dataset_spec must be provided")
            if target_column is None:
                raise ValueError("target_column is required when dataset_spec is None")
            
            # Use provided data or load from path
            if data is not None:
                self.dataset_spec = KaggleDatasetSpec.from_dataframe_analysis(data, target_column)
            else:
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
        
        # Load data if not provided
        if self.data is None:
            self._load_data()
        
        # Initialize components
        self._initialize_tokenizer()
        self._initialize_text_generator()
        
        # Create MLX stream configuration
        self.mlx_config = MLXStreamConfig(
            optimization_profile=self.dataset_spec.optimization_profile,
            batch_size=batch_size,
            max_length=max_length,
            prefetch_size=prefetch_size,
            num_threads=num_workers,
            buffer_size=shuffle_buffer_size,
            enable_dynamic_batching=mlx_config_kwargs.get('enable_dynamic_batching', True),
            **mlx_config_kwargs
        )
        
        # Initialize data pipelines
        self._init_data_pipelines()
        
        # Protocol-required attributes (for MLXTrainer compatibility)
        self.dataset_spec_dict = self._create_dataset_spec()
        
        logger.info("UniversalKaggleLoader initialization complete")
        logger.info(f"Protocol compliance: {'enabled' if hasattr(self, 'dataset_spec_dict') else 'disabled'}")
        logger.info(f"Advanced caching: {'enabled' if self.enable_cache else 'disabled'}")
        logger.info(f"Batch count: {len(self)} batches")
    
    def _process_config_object(self, config):
        """Process configuration object for ModularMLXDataLoader compatibility."""
        if isinstance(config, (str, Path)):
            if ADVANCED_FEATURES_AVAILABLE and Path(config).exists():
                # Load from file
                config = ConfigFactory.from_file(config)
            else:
                # Assume it's a preset name
                logger.warning(f"Config file {config} not found or ConfigFactory not available")
                return
        
        # Extract settings from config object
        if hasattr(config, 'dataloader'):
            # Training config format
            dl_config = config.dataloader
            self.batch_size = getattr(dl_config, 'batch_size', self.batch_size)
            self.max_length = getattr(dl_config, 'max_length', self.max_length)
            self.tokenizer_name = getattr(dl_config, 'tokenizer_name', self.tokenizer_name)
            self.tokenizer_backend = getattr(dl_config, 'tokenizer_backend', self.tokenizer_backend)
            self.enable_cache = getattr(dl_config, 'enable_cache', self.enable_cache)
            self.cache_dir = getattr(dl_config, 'cache_dir', self.cache_dir)
            self.augment = getattr(dl_config, 'augment', self.augment)
            self.prefetch_size = getattr(dl_config, 'prefetch_size', self.prefetch_size)
            self.num_workers = getattr(dl_config, 'num_workers', self.num_workers)
            
            # Dataset config
            if hasattr(config, 'dataset'):
                ds_config = config.dataset
                if hasattr(ds_config, 'data_path') and not self.train_path:
                    self.train_path = ds_config.data_path
                if hasattr(ds_config, 'label_column'):
                    self.target_column = ds_config.label_column
    
    def _initialize_caches(self):
        """Initialize caching system."""
        if not self.enable_cache:
            return
            
        if not ADVANCED_FEATURES_AVAILABLE:
            logger.warning("Advanced caching not available")
            return
        
        cache_dir = Path(self.cache_dir or "./cache")
        
        # Text conversion cache
        if self.cache_text_conversion:
            self.caches['text'] = DiskCache(
                cache_dir=cache_dir / "text",
                max_size_mb=500,
            )
        
        # Tokenization cache
        if self.cache_tokenization:
            self.caches['tokenization'] = TokenizationCache(
                cache_dir=cache_dir / "tokenization",
                tokenizer_name=self.tokenizer_name,
            )
        
        logger.info(f"Initialized caches: {list(self.caches.keys())}")
    
    def _load_data(self):
        """Load data from file path."""
        if self.train_path:
            data_path = Path(self.train_path)
            if not data_path.exists():
                raise FileNotFoundError(f"Data file not found: {data_path}")
            
            if data_path.suffix.lower() == '.csv':
                self.data = pd.read_csv(data_path)
            elif data_path.suffix.lower() == '.json':
                self.data = pd.read_json(data_path)
            elif data_path.suffix.lower() == '.parquet':
                self.data = pd.read_parquet(data_path)
            else:
                raise ValueError(f"Unsupported file format: {data_path.suffix}")
            
            # Normalize column names
            self.data.columns = self.data.columns.str.lower()
            
            logger.info(f"Loaded {len(self.data)} samples from {data_path}")
        else:
            logger.info("No data path provided, using pre-loaded data")
    
    def _initialize_tokenizer(self):
        """Initialize tokenizer with backend flexibility."""
        if USE_TOKENIZER_WRAPPER:
            self.tokenizer = TokenizerWrapper(
                model_name=self.tokenizer_name,
                backend=self.tokenizer_backend,
            )
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name)
        
        logger.info(f"Initialized tokenizer: {self.tokenizer_name} (backend: {self.tokenizer_backend})")
    
    def _initialize_text_generator(self):
        """Initialize text generator."""
        self.text_generator = UniversalTextGenerator(
            dataset_spec=self.dataset_spec,
            strategy=self.text_strategy,
        )
    
    def _create_dataset_spec(self) -> Dict[str, Any]:
        """Create dataset specification for trainer (protocol compliance)."""
        return {
            "name": self.dataset_spec.name,
            "competition_name": getattr(self.dataset_spec, 'competition_name', 'unknown'),
            "competition_type": getattr(self.dataset_spec, 'competition_type', 'classification'),
            "num_samples": len(self.data) if self.data is not None else 0,
            "num_classes": self._get_num_classes(),
            "label_column": self.dataset_spec.target_column,
            "features": list(self.data.columns) if self.data is not None else [],
            "split": self.split,
        }
    
    def _get_num_classes(self) -> int:
        """Get number of classes for classification tasks."""
        if self.data is not None and self.dataset_spec.target_column in self.data.columns:
            return len(self.data[self.dataset_spec.target_column].unique())
        return 2  # Default binary classification
    
    def __len__(self) -> int:
        """Return number of batches (protocol compliance)."""
        if self.data is None:
            return 0
        
        num_samples = len(self.data)
        if self.drop_last:
            return num_samples // self.batch_size
        else:
            return (num_samples + self.batch_size - 1) // self.batch_size
    
    def __iter__(self) -> Iterator[Dict[str, mx.array]]:
        """Iterate over batches (protocol compliance)."""
        if self.data is None:
            return
        
        # Convert data to records for easier processing
        data_records = self.data.to_dict('records')
        
        # Shuffle if configured
        if self.shuffle and self.split == "train":
            random.shuffle(data_records)
        
        # Yield batches
        for i in range(0, len(data_records), self.batch_size):
            batch_data = data_records[i:i + self.batch_size]
            
            # Skip incomplete batch if drop_last is True
            if self.drop_last and len(batch_data) < self.batch_size:
                continue
            
            # Process batch
            try:
                batch = self._process_batch(batch_data)
                yield batch
            except Exception as e:
                logger.error(f"Error processing batch {i//self.batch_size}: {e}")
                continue
    
    def _process_batch(self, batch_data: List[Dict[str, Any]]) -> Dict[str, mx.array]:
        """Process a batch of data through the text conversion and tokenization pipeline."""
        batch_texts = []
        batch_labels = []
        
        # Process each sample in the batch
        for sample in batch_data:
            # Get or create sample ID for caching
            sample_id = str(sample.get('id', hash(str(sample))))
            
            # Text conversion with caching
            if 'text' in self.caches:
                cached_text = self.caches['text'].get(sample_id)
                if cached_text:
                    text = cached_text
                else:
                    text = self.text_generator.generate_text(sample)
                    self.caches['text'].set(sample_id, text)
            else:
                # Convert without cache
                text = self.text_generator.generate_text(sample)
            
            # Apply augmentation if enabled and this is training data
            if self.augment and self.split == "train":
                text = self._augment_text(text)
            
            batch_texts.append(text)
            
            # Extract label
            label = sample.get(self.dataset_spec.target_column, 0)
            batch_labels.append(label)
        
        # Tokenize batch
        if 'tokenization' in self.caches:
            # Use tokenization cache
            encodings = []
            for text in batch_texts:
                cached = self.caches['tokenization'].get_tokenized(
                    text=text,
                    max_length=self.max_length,
                    padding="max_length",
                    truncation=True,
                )
                if cached is not None:
                    encodings.append(cached)
                else:
                    # Tokenize and cache
                    if USE_TOKENIZER_WRAPPER:
                        encoding = self.tokenizer.batch_encode_plus(
                            [text],
                            max_length=self.max_length,
                            padding="max_length",
                            truncation=True,
                            return_attention_mask=True,
                        )
                        encoding_dict = {k: v[0] for k, v in encoding.items()}
                    else:
                        encoding = self.tokenizer(
                            text,
                            max_length=self.max_length,
                            padding="max_length",
                            truncation=True,
                            return_tensors="np"
                        )
                        encoding_dict = {k: v[0] for k, v in encoding.items()}
                    
                    # Cache it
                    self.caches['tokenization'].set_tokenized(
                        text=text,
                        max_length=self.max_length,
                        padding="max_length",
                        truncation=True,
                        encoding=encoding_dict,
                    )
                    encodings.append(encoding_dict)
        else:
            # Tokenize without cache
            if USE_TOKENIZER_WRAPPER:
                batch_encoding = self.tokenizer.batch_encode_plus(
                    batch_texts,
                    max_length=self.max_length,
                    padding="max_length",
                    truncation=True,
                    return_attention_mask=True,
                )
                # Convert to list of dicts
                encodings = [
                    {k: v[i] for k, v in batch_encoding.items()}
                    for i in range(len(batch_texts))
                ]
            else:
                encodings = []
                for text in batch_texts:
                    encoding = self.tokenizer(
                        text,
                        max_length=self.max_length,
                        padding="max_length",
                        truncation=True,
                        return_tensors="np"
                    )
                    encodings.append({k: v[0] for k, v in encoding.items()})
        
        # Create batch arrays
        input_ids = mx.array([e['input_ids'] for e in encodings])
        attention_mask = mx.array([e['attention_mask'] for e in encodings], dtype=mx.float32)
        labels = mx.array(batch_labels, dtype=mx.int32)
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels,
        }
    
    def _init_data_pipelines(self):
        """Initialize MLX data pipelines (for backward compatibility)."""
        # For backward compatibility, create pipelines when requested
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
        
        # Initialize pipelines lazily when needed
        logger.info("Data pipelines will be initialized lazily when requested")
    
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
    
    def _prepare_data_splits(self):
        """Prepare data splits based on configuration (ModularMLXDataLoader compatibility)."""
        if self.data is None:
            return
        
        # Apply filtering if needed
        if hasattr(self.dataset_spec, 'filter_empty'):
            initial_len = len(self.data)
            # Remove rows where label column is null
            if self.dataset_spec.target_column in self.data.columns:
                self.data = self.data.dropna(subset=[self.dataset_spec.target_column])
            # Remove completely empty rows
            self.data = self.data.dropna(how='all')
            
            if len(self.data) < initial_len:
                logger.info(f"Filtered data: removed {initial_len - len(self.data)} empty samples")
        
        # Reset index
        self.data = self.data.reset_index(drop=True)
    
    def get_train_loader(self) -> Iterator[Dict[str, mx.array]]:
        """Get training data iterator."""
        if self.train_path is None and self.data is None:
            raise ValueError("No training data provided")
        
        # For primary implementation, return self as iterator
        self.split = "train"
        if self.data is None:
            self._load_data()
        
        return iter(self)
    
    def get_val_loader(self) -> Iterator[Dict[str, mx.array]]:
        """Get validation data iterator."""
        if self.val_path:
            # Create separate loader for validation
            val_loader = UniversalKaggleLoader(
                train_path=self.val_path,
                dataset_spec=self.dataset_spec,
                tokenizer_name=self.tokenizer_name,
                max_length=self.max_length,
                batch_size=self.batch_size,
                text_strategy=self.text_strategy,
                augment=False,  # No augmentation for validation
                split="val",
                shuffle=False,
                **{k: v for k, v in self.__dict__.items() if k.startswith('cache_') or k in ['tokenizer_backend', 'enable_cache']}
            )
            return iter(val_loader)
        else:
            # Use training data for validation if no separate validation set
            logger.info("Using training data for validation (no separate val set)")
            self.split = "val"
            if self.data is None:
                self._load_data()
            return iter(self)
    
    def get_test_loader(self) -> Iterator[Dict[str, mx.array]]:
        """Get test data iterator."""
        if self.test_path is None:
            raise ValueError("No test data provided")
        
        # Create separate loader for test
        test_loader = UniversalKaggleLoader(
            train_path=self.test_path,
            dataset_spec=self.dataset_spec,
            tokenizer_name=self.tokenizer_name,
            max_length=self.max_length,
            batch_size=self.batch_size,
            text_strategy=self.text_strategy,
            augment=False,  # No augmentation for test
            split="test",
            shuffle=False,
            **{k: v for k, v in self.__dict__.items() if k.startswith('cache_') or k in ['tokenizer_backend', 'enable_cache']}
        )
        return iter(test_loader)
    
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
    
    @classmethod
    def from_config_file(cls, config_path: Union[str, Path], split: str = "train", **overrides):
        """Create dataloader from configuration file (ModularMLXDataLoader compatibility)."""
        return cls(config=config_path, split=split, **overrides)
    
    @classmethod
    def from_competition(
        cls, 
        competition: str, 
        data_path: str, 
        split: str = "train", 
        **overrides
    ):
        """Create dataloader for a specific competition (ModularMLXDataLoader compatibility)."""
        return cls(
            train_path=data_path,
            split=split,
            **overrides
        )
    
    def create_stream(self, is_training: bool = True) -> Iterator[Dict[str, mx.array]]:
        """Create stream iterator (backward compatibility)."""
        self.split = "train" if is_training else "val"
        return iter(self)
    
    def get_sample_batch(self) -> Dict[str, mx.array]:
        """Get a single batch for testing/debugging."""
        return next(iter(self))
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache usage statistics."""
        stats = {}
        for name, cache in self.caches.items():
            if hasattr(cache, 'cache'):
                stats[name] = {
                    "items": len(cache.cache),
                    "hit_rate": getattr(cache, 'hit_rate', 0.0)
                }
        return stats


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
    try:
        from .dataset_spec import TITANIC_SPEC
        return UniversalKaggleLoader(
            train_path=train_path,
            test_path=test_path,
            val_path=val_path,
            dataset_spec=TITANIC_SPEC,
            **kwargs
        )
    except ImportError:
        logger.warning("TITANIC_SPEC not available, using auto-detection")
        return UniversalKaggleLoader(
            train_path=train_path,
            test_path=test_path,
            val_path=val_path,
            target_column="survived",
            **kwargs
        )


# Legacy compatibility functions
def create_kaggle_dataloader(
    dataset_name: str,
    csv_path: str,
    tokenizer_name: str = "answerdotai/ModernBERT-base",
    **kwargs
) -> UniversalKaggleLoader:
    """Legacy compatibility function."""
    logger.warning("create_kaggle_dataloader is deprecated. Use UniversalKaggleLoader directly.")
    return UniversalKaggleLoader(
        train_path=csv_path,
        tokenizer_name=tokenizer_name,
        **kwargs
    )


def create_titanic_dataloader(
    train_path: str = "data/titanic/train.csv",
    test_path: Optional[str] = "data/titanic/test.csv",
    val_path: Optional[str] = None,
    **kwargs
) -> UniversalKaggleLoader:
    """Legacy compatibility function for Titanic dataloader."""
    logger.warning("create_titanic_dataloader is deprecated. Use create_titanic_loader instead.")
    return create_titanic_loader(
        train_path=train_path,
        test_path=test_path,
        val_path=val_path,
        **kwargs
    )


# Main dataloader class alias for backward compatibility
KaggleDataLoader = UniversalKaggleLoader
ModularMLXDataLoader = UniversalKaggleLoader