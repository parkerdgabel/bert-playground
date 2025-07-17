"""MLX-Data streaming utilities for efficient data loading.

This module provides optimized MLX-Data streaming implementations
that fully utilize MLX-Data capabilities for maximum performance.
"""

import mlx.core as mx
import mlx.data as dx
from typing import Dict, List, Optional, Callable, Any, Union, Iterator
import pandas as pd
import numpy as np
from pathlib import Path
from loguru import logger
from transformers import AutoTokenizer
import json

from .dataset_spec import KaggleDatasetSpec, FeatureType, OptimizationProfile


class MLXStreamConfig:
    """Configuration for MLX-Data streaming optimization."""
    
    def __init__(
        self,
        optimization_profile: OptimizationProfile,
        batch_size: int = 32,
        max_length: int = 256,
        prefetch_size: int = 4,
        num_threads: int = 4,
        buffer_size: int = 1000,
        shuffle_buffer_size: Optional[int] = None,
        enable_dynamic_batching: bool = True,
        enable_shape_optimization: bool = True,
    ):
        self.optimization_profile = optimization_profile
        self.batch_size = batch_size
        self.max_length = max_length
        self.prefetch_size = prefetch_size
        self.num_threads = num_threads
        self.buffer_size = buffer_size
        self.shuffle_buffer_size = shuffle_buffer_size or buffer_size
        self.enable_dynamic_batching = enable_dynamic_batching
        self.enable_shape_optimization = enable_shape_optimization
        
        # Adjust settings based on optimization profile
        self._adjust_for_profile()
    
    def _adjust_for_profile(self):
        """Adjust settings based on optimization profile."""
        if self.optimization_profile == OptimizationProfile.DEVELOPMENT:
            # Minimal optimization for fast iteration
            self.prefetch_size = 2
            self.num_threads = 2
            self.buffer_size = 100
            self.shuffle_buffer_size = 100
            self.enable_dynamic_batching = False
            self.enable_shape_optimization = False
            
        elif self.optimization_profile == OptimizationProfile.PRODUCTION:
            # Balanced optimization
            self.prefetch_size = max(4, self.prefetch_size)
            self.num_threads = min(6, max(4, self.num_threads))
            self.buffer_size = max(500, self.buffer_size)
            self.shuffle_buffer_size = max(500, self.shuffle_buffer_size)
            
        elif self.optimization_profile == OptimizationProfile.COMPETITION:
            # Maximum optimization
            self.prefetch_size = max(8, self.prefetch_size)
            self.num_threads = max(8, self.num_threads)
            self.buffer_size = max(2000, self.buffer_size)
            self.shuffle_buffer_size = max(2000, self.shuffle_buffer_size)
            self.enable_dynamic_batching = True
            self.enable_shape_optimization = True


class MLXCSVStreamer:
    """High-performance CSV streaming using MLX-Data."""
    
    def __init__(
        self,
        csv_path: str,
        dataset_spec: KaggleDatasetSpec,
        tokenizer_name: str = "answerdotai/ModernBERT-base",
        config: Optional[MLXStreamConfig] = None,
    ):
        self.csv_path = Path(csv_path)
        self.dataset_spec = dataset_spec
        self.tokenizer_name = tokenizer_name
        self.config = config or MLXStreamConfig(dataset_spec.optimization_profile)
        
        # Initialize tokenizer
        logger.info(f"Loading tokenizer: {tokenizer_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Determine columns to load
        self._determine_columns()
        
        logger.info(f"Initialized MLXCSVStreamer for {self.csv_path}")
        logger.info(f"Optimization profile: {self.config.optimization_profile}")
        logger.info(f"Columns to load: {self.columns_to_load}")
    
    def _determine_columns(self):
        """Determine which columns to load from CSV."""
        # Always include target column if it exists
        self.columns_to_load = []
        
        # Add target column if this is training data
        if (self.dataset_spec.target_column and 
            self.dataset_spec.target_column not in self.dataset_spec.id_columns):
            self.columns_to_load.append(self.dataset_spec.target_column)
        
        # Add feature columns (exclude ID columns for efficiency)
        feature_columns = [
            col for col, ftype in self.dataset_spec.feature_types.items()
            if ftype != FeatureType.ID and ftype != FeatureType.TARGET
        ]
        self.columns_to_load.extend(feature_columns)
        
        # Ensure we have at least some columns
        if not self.columns_to_load:
            logger.warning("No feature columns specified, will load all columns")
            self.columns_to_load = None  # Load all columns
    
    def create_stream(
        self,
        is_training: bool = True,
        text_transform_fn: Optional[Callable] = None,
    ) -> dx.Stream:
        """Create optimized MLX-Data stream from CSV."""
        logger.info(f"Creating MLX stream (training={is_training})")
        
        try:
            # Use MLX-Data's native CSV reading for better performance
            if self.columns_to_load:
                # Load specific columns
                logger.debug(f"Loading columns: {self.columns_to_load}")
                stream = dx.stream_csv_reader(
                    str(self.csv_path),
                    columns=self.columns_to_load,
                    delimiter=',',
                    has_header=True,
                )
            else:
                # Load all columns
                stream = dx.stream_csv_reader(
                    str(self.csv_path),
                    delimiter=',',
                    has_header=True,
                )
            
            logger.info("Successfully created CSV stream")
            
        except Exception as e:
            logger.warning(f"Direct CSV streaming failed: {e}")
            logger.info("Falling back to pandas-based streaming")
            
            # Fallback to pandas-based approach
            stream = self._create_pandas_stream()
        
        # Apply data transformations
        if text_transform_fn:
            logger.info("Applying text transformation")
            stream = stream.sample_transform(text_transform_fn)
        
        # Apply tokenization
        logger.info("Applying tokenization")
        stream = stream.sample_transform(self._tokenize_sample)
        
        # Apply optimizations based on configuration
        stream = self._apply_optimizations(stream, is_training)
        
        return stream
    
    def _create_pandas_stream(self) -> dx.Stream:
        """Fallback method using pandas for CSV reading."""
        # Read CSV with pandas
        logger.info("Loading CSV with pandas")
        df = pd.read_csv(self.csv_path, usecols=self.columns_to_load)
        
        # Convert to records
        records = df.to_dict('records')
        
        # Create MLX stream from records
        return dx.buffer_from_vector(records)
    
    def _tokenize_sample(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """Tokenize a single sample."""
        # For now, assume we have a 'text' field
        # This should be customized based on the text transformation
        text = sample.get('text', '')
        
        if not text:
            # Create default text if none provided
            text = "Missing text data"
        
        # Tokenize
        tokens = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.config.max_length,
            return_tensors="np",
        )
        
        result = {
            "input_ids": tokens["input_ids"][0],
            "attention_mask": tokens["attention_mask"][0],
        }
        
        # Add label if available
        target_col = self.dataset_spec.target_column
        if target_col and target_col in sample:
            result["labels"] = int(sample[target_col])
        
        return result
    
    def _apply_optimizations(self, stream: dx.Stream, is_training: bool) -> dx.Stream:
        """Apply MLX-Data optimizations to the stream."""
        # Shuffling for training
        if is_training:
            logger.info(f"Adding shuffle with buffer size: {self.config.shuffle_buffer_size}")
            stream = stream.shuffle(buffer_size=self.config.shuffle_buffer_size)
        
        # Shape-based filtering if enabled
        if self.config.enable_shape_optimization:
            logger.info("Applying shape-based optimizations")
            # Filter out extremely short or long sequences
            # This is sample-dependent and would need customization
            pass
        
        # Buffering for performance
        if self.config.buffer_size > 0:
            logger.info(f"Adding buffer: {self.config.buffer_size}")
            stream = stream.buffered(self.config.buffer_size)
        
        # Batching
        if self.config.enable_dynamic_batching:
            logger.info(f"Using dynamic batching: max_batch_size={self.config.batch_size}")
            # Dynamic batching based on token count
            max_tokens = self.config.max_length * self.config.batch_size
            try:
                stream = stream.dynamic_batch(
                    max_batch_size=self.config.batch_size,
                    max_tokens=max_tokens,
                )
            except Exception as e:
                logger.warning(f"Dynamic batching failed: {e}, falling back to fixed batching")
                stream = stream.batch(self.config.batch_size)
        else:
            logger.info(f"Using fixed batching: batch_size={self.config.batch_size}")
            stream = stream.batch(self.config.batch_size)
        
        # Prefetching for performance
        if self.config.prefetch_size > 0:
            logger.info(f"Adding prefetch: size={self.config.prefetch_size}, threads={self.config.num_threads}")
            stream = stream.prefetch(
                prefetch_size=self.config.prefetch_size,
                num_threads=self.config.num_threads,
            )
        
        return stream


class MLXDataPipeline:
    """Complete data pipeline using optimized MLX-Data streaming."""
    
    def __init__(
        self,
        csv_path: str,
        dataset_spec: KaggleDatasetSpec,
        tokenizer_name: str = "answerdotai/ModernBERT-base",
        config: Optional[MLXStreamConfig] = None,
        text_transform_fn: Optional[Callable] = None,
    ):
        self.csv_path = csv_path
        self.dataset_spec = dataset_spec
        self.tokenizer_name = tokenizer_name
        self.config = config or MLXStreamConfig(dataset_spec.optimization_profile)
        self.text_transform_fn = text_transform_fn
        
        # Create the streamer
        self.streamer = MLXCSVStreamer(
            csv_path=csv_path,
            dataset_spec=dataset_spec,
            tokenizer_name=tokenizer_name,
            config=config,
        )
        
        self._train_stream = None
        self._eval_stream = None
    
    def get_train_stream(self) -> dx.Stream:
        """Get training data stream."""
        if self._train_stream is None:
            logger.info("Creating training stream")
            self._train_stream = self.streamer.create_stream(
                is_training=True,
                text_transform_fn=self.text_transform_fn,
            )
        return self._train_stream
    
    def get_eval_stream(self) -> dx.Stream:
        """Get evaluation data stream."""
        if self._eval_stream is None:
            logger.info("Creating evaluation stream")
            self._eval_stream = self.streamer.create_stream(
                is_training=False,
                text_transform_fn=self.text_transform_fn,
            )
        return self._eval_stream
    
    def get_stream_iterator(self, is_training: bool = True) -> Iterator[Dict[str, mx.array]]:
        """Get iterator over batches."""
        stream = self.get_train_stream() if is_training else self.get_eval_stream()
        
        for batch in stream:
            # Convert to MLX arrays
            mlx_batch = {}
            for key, value in batch.items():
                if isinstance(value, (list, np.ndarray)):
                    mlx_batch[key] = mx.array(value)
                else:
                    mlx_batch[key] = value
            
            yield mlx_batch
    
    def get_dataset_info(self) -> Dict[str, Any]:
        """Get information about the dataset."""
        return {
            "csv_path": str(self.csv_path),
            "dataset_spec": self.dataset_spec.to_dict(),
            "config": {
                "optimization_profile": self.config.optimization_profile.value,
                "batch_size": self.config.batch_size,
                "max_length": self.config.max_length,
                "prefetch_size": self.config.prefetch_size,
                "num_threads": self.config.num_threads,
                "enable_dynamic_batching": self.config.enable_dynamic_batching,
            },
            "tokenizer": self.tokenizer_name,
        }


def create_mlx_pipeline(
    csv_path: str,
    dataset_spec: Optional[KaggleDatasetSpec] = None,
    target_column: Optional[str] = None,
    **kwargs
) -> MLXDataPipeline:
    """Convenience function to create MLX data pipeline."""
    if dataset_spec is None:
        if target_column is None:
            raise ValueError("Either dataset_spec or target_column must be provided")
        dataset_spec = KaggleDatasetSpec.from_csv_analysis(csv_path, target_column)
    
    return MLXDataPipeline(
        csv_path=csv_path,
        dataset_spec=dataset_spec,
        **kwargs
    )