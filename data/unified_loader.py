"""Unified data loader for MLX with configurable optimization levels."""

import mlx.core as mx
import mlx.data as dx
from typing import Dict, List, Optional, Tuple, Iterator, Union
import pandas as pd
import numpy as np
from pathlib import Path
from loguru import logger
from transformers import AutoTokenizer
import random
from enum import Enum

from .text_templates import TitanicTextTemplates


class OptimizationLevel(Enum):
    """Data loader optimization levels."""

    BASIC = "basic"  # Simple batching, no optimization
    STANDARD = "standard"  # Moderate optimization with prefetching
    OPTIMIZED = "optimized"  # Full optimization with MLX-Data streams
    AUTO = "auto"  # Automatically choose based on dataset size


class UnifiedTitanicDataPipeline:
    """Unified data pipeline with configurable optimization levels."""

    def __init__(
        self,
        data_path: str,
        tokenizer_name: str = "answerdotai/ModernBERT-base",
        max_length: int = 256,
        batch_size: int = 32,
        is_training: bool = True,
        augment: bool = True,
        # Optimization parameters
        optimization_level: Union[str, OptimizationLevel] = OptimizationLevel.AUTO,
        prefetch_size: int = 4,
        num_threads: int = 4,
        cache_size: int = 1000,
        use_mlx_data: Optional[bool] = None,
        pre_tokenize: Optional[bool] = None,
    ):
        self.data_path = Path(data_path)
        self.max_length = max_length
        self.batch_size = batch_size
        self.is_training = is_training
        self.augment = augment and is_training

        # Parse optimization level
        if isinstance(optimization_level, str):
            optimization_level = OptimizationLevel(optimization_level)
        self.optimization_level = optimization_level

        # Initialize tokenizer
        logger.info(f"Loading tokenizer: {tokenizer_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Initialize text template converter
        self.text_converter = TitanicTextTemplates()

        # Load data first
        logger.info(f"Loading data from {self.data_path}")
        self.df = pd.read_csv(self.data_path)

        # Determine optimization settings
        self._configure_optimization(
            optimization_level,
            use_mlx_data,
            pre_tokenize,
            prefetch_size,
            num_threads,
            cache_size,
        )

        # Prepare data after optimization settings are configured
        self.prepare_data()

        # Initialize data pipeline based on optimization level
        if self.use_mlx_data:
            self._initialize_mlx_stream()
        else:
            self._initialize_basic_pipeline()

        logger.info(
            f"Initialized UnifiedTitanicDataPipeline: {len(self.texts)} samples, "
            f"optimization={self.optimization_level.value}, "
            f"mlx_data={self.use_mlx_data}, pre_tokenize={self.pre_tokenize}"
        )

    def _configure_optimization(
        self,
        level: OptimizationLevel,
        use_mlx_data: Optional[bool],
        pre_tokenize: Optional[bool],
        prefetch_size: int,
        num_threads: int,
        cache_size: int,
    ):
        """Configure optimization settings based on level."""
        if level == OptimizationLevel.AUTO:
            # Auto-detect based on dataset size
            dataset_size = len(self.df)
            if dataset_size < 1000:
                level = OptimizationLevel.BASIC
            elif dataset_size < 10000:
                level = OptimizationLevel.STANDARD
            else:
                level = OptimizationLevel.OPTIMIZED
            logger.info(f"Auto-selected optimization level: {level.value}")

        # Set defaults based on optimization level
        if level == OptimizationLevel.BASIC:
            self.use_mlx_data = use_mlx_data if use_mlx_data is not None else False
            self.pre_tokenize = pre_tokenize if pre_tokenize is not None else False
            self.prefetch_size = 1
            self.num_threads = 1
            self.cache_size = 0
        elif level == OptimizationLevel.STANDARD:
            self.use_mlx_data = use_mlx_data if use_mlx_data is not None else True
            self.pre_tokenize = pre_tokenize if pre_tokenize is not None else False
            self.prefetch_size = prefetch_size
            self.num_threads = min(num_threads, 4)
            self.cache_size = min(cache_size, 500)
        else:  # OPTIMIZED
            self.use_mlx_data = use_mlx_data if use_mlx_data is not None else True
            self.pre_tokenize = pre_tokenize if pre_tokenize is not None else True
            self.prefetch_size = prefetch_size
            self.num_threads = num_threads
            self.cache_size = cache_size

        # Pre-tokenization cache
        self.tokenized_cache = {} if self.pre_tokenize else None

    def prepare_data(self):
        """Prepare data with preprocessing."""
        logger.info("Preparing data...")

        # Log missing values
        missing_stats = self.df.isnull().sum()
        if missing_stats.any():
            logger.debug("Missing values before preprocessing:")
            for col, count in missing_stats[missing_stats > 0].items():
                logger.debug(f"  {col}: {count} ({count / len(self.df) * 100:.1f}%)")

        # Fill missing values
        self.df["Age"] = self.df["Age"].fillna(self.df["Age"].median())
        self.df["Fare"] = self.df["Fare"].fillna(self.df["Fare"].median())
        self.df["Embarked"] = self.df["Embarked"].fillna("S")
        self.df["Cabin"] = self.df["Cabin"].fillna("Unknown")

        # Convert to text representations
        self.texts = []
        self.labels = []

        logger.info("Converting tabular data to text representations...")
        for idx, (_, row) in enumerate(self.df.iterrows()):
            row_dict = row.to_dict()

            if self.augment:
                # Generate multiple variations
                for _ in range(3):
                    text = self.text_converter.row_to_text(row_dict)
                    self.texts.append(text)
                    if "Survived" in row_dict:
                        self.labels.append(int(row["Survived"]))
                    else:
                        self.labels.append(-1)  # No label for test data
            else:
                text = self.text_converter.row_to_text(row_dict)
                self.texts.append(text)
                if "Survived" in row_dict:
                    self.labels.append(int(row["Survived"]))
                else:
                    self.labels.append(-1)

            if idx < 3:  # Log examples
                logger.debug(f"Example {idx + 1}: {text[:100]}...")

        # Convert labels to numpy array
        self.labels = np.array(self.labels, dtype=np.int32)

        # Log label distribution
        if self.is_training:
            unique, counts = np.unique(self.labels, return_counts=True)
            logger.info("Label distribution:")
            for label, count in zip(unique, counts):
                logger.info(
                    f"  Class {label}: {count} ({count / len(self.labels) * 100:.1f}%)"
                )

        # Pre-tokenize if enabled
        if self.pre_tokenize:
            self._pretokenize_all()

    def _pretokenize_all(self):
        """Pre-tokenize all texts for efficiency."""
        logger.info("Pre-tokenizing texts...")
        batch_size = 100

        for i in range(0, len(self.texts), batch_size):
            batch_texts = self.texts[i : i + batch_size]
            batch_tokens = self.tokenizer(
                batch_texts,
                padding="max_length",
                truncation=True,
                max_length=self.max_length,
                return_tensors="np",
            )

            for j, idx in enumerate(range(i, min(i + batch_size, len(self.texts)))):
                self.tokenized_cache[idx] = {
                    "input_ids": batch_tokens["input_ids"][j],
                    "attention_mask": batch_tokens["attention_mask"][j],
                }

    def tokenize_function(self, text: Union[str, int]) -> Dict[str, mx.array]:
        """Tokenize a single text or retrieve from cache."""
        if isinstance(text, int) and self.tokenized_cache:
            # Retrieve from cache
            cached = self.tokenized_cache[text]
            return {
                "input_ids": mx.array(cached["input_ids"]),
                "attention_mask": mx.array(cached["attention_mask"]),
            }
        else:
            # Tokenize on the fly
            if isinstance(text, int):
                text = self.texts[text]

            tokens = self.tokenizer(
                text,
                padding="max_length",
                truncation=True,
                max_length=self.max_length,
                return_tensors="np",
            )

            return {
                "input_ids": mx.array(tokens["input_ids"][0]),
                "attention_mask": mx.array(tokens["attention_mask"][0]),
            }

    def _initialize_basic_pipeline(self):
        """Initialize basic data pipeline without MLX-Data."""
        self.indices = list(range(len(self.texts)))
        if self.is_training:
            random.shuffle(self.indices)

    def _initialize_mlx_stream(self):
        """Initialize MLX-Data stream pipeline."""
        # Create dataset from indices
        indices = list(range(len(self.texts)))

        # Create MLX dataset
        dataset = dx.buffer_from_vector(indices)

        if self.is_training:
            dataset = dataset.shuffle(buffer_size=len(indices))

        # Batch the dataset
        dataset = dataset.batch(self.batch_size)

        # Apply tokenization
        if self.pre_tokenize:
            dataset = dataset.map(
                lambda batch: self._process_cached_batch(batch),
                num_workers=self.num_threads,
            )
        else:
            dataset = dataset.map(
                lambda batch: self._process_batch(batch), num_workers=self.num_threads
            )

        # Prefetch for performance
        if self.prefetch_size > 0:
            dataset = dataset.prefetch(self.prefetch_size)

        self.stream = dataset.to_stream()

    def _process_batch(self, indices: List[int]) -> Dict[str, mx.array]:
        """Process a batch of indices into tokenized inputs."""
        texts = [self.texts[i] for i in indices]
        # Convert labels to Python int to avoid numpy int32 issues
        labels = mx.array([int(self.labels[i]) for i in indices])

        # Tokenize batch
        tokens = self.tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="np",
        )

        return {
            "input_ids": mx.array(tokens["input_ids"]),
            "attention_mask": mx.array(tokens["attention_mask"]),
            "labels": labels,
        }

    def _process_cached_batch(self, indices: List[int]) -> Dict[str, mx.array]:
        """Process a batch using cached tokens."""
        input_ids = []
        attention_masks = []

        for i in indices:
            cached = self.tokenized_cache[i]
            input_ids.append(cached["input_ids"])
            attention_masks.append(cached["attention_mask"])

        return {
            "input_ids": mx.array(np.stack(input_ids)),
            "attention_mask": mx.array(np.stack(attention_masks)),
            "labels": mx.array([int(self.labels[i]) for i in indices]),
        }

    def get_dataloader(self) -> Iterator[Dict[str, mx.array]]:
        """Get data loader iterator."""
        if self.use_mlx_data:
            # Reset stream for each epoch
            self.stream.reset()
            for batch in self.stream:
                yield batch
        else:
            # Basic iteration
            for i in range(0, len(self.indices), self.batch_size):
                batch_indices = self.indices[i : i + self.batch_size]

                if self.pre_tokenize:
                    yield self._process_cached_batch(batch_indices)
                else:
                    yield self._process_batch(batch_indices)

    def __iter__(self):
        """Make the pipeline iterable."""
        return self.get_dataloader()

    def __len__(self):
        """Return number of batches."""
        return (len(self.texts) + self.batch_size - 1) // self.batch_size

    def get_num_batches(self) -> int:
        """Get total number of batches."""
        return len(self)


# Factory function for backward compatibility
def create_unified_dataloaders(
    train_path: str,
    val_path: str,
    tokenizer_name: str = "answerdotai/ModernBERT-base",
    max_length: int = 256,
    batch_size: int = 32,
    augment: bool = True,
    optimization_level: str = "auto",
    **kwargs,
) -> Tuple[UnifiedTitanicDataPipeline, UnifiedTitanicDataPipeline]:
    """Create train and validation data loaders with unified pipeline."""
    train_loader = UnifiedTitanicDataPipeline(
        train_path,
        tokenizer_name=tokenizer_name,
        max_length=max_length,
        batch_size=batch_size,
        is_training=True,
        augment=augment,
        optimization_level=optimization_level,
        **kwargs,
    )

    val_loader = UnifiedTitanicDataPipeline(
        val_path,
        tokenizer_name=tokenizer_name,
        max_length=max_length,
        batch_size=batch_size,
        is_training=False,
        augment=False,
        optimization_level=optimization_level,
        **kwargs,
    )

    return train_loader, val_loader


# Backward compatibility imports
TitanicDataPipeline = UnifiedTitanicDataPipeline
OptimizedTitanicDataPipeline = UnifiedTitanicDataPipeline
create_data_loaders = create_unified_dataloaders
create_optimized_dataloaders = create_unified_dataloaders
