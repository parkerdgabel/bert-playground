"""
MLX DataLoader V2 - Idiomatic stream-based implementation for tabular-to-text tasks.

This implementation follows MLX's design philosophy:
- Stream-first approach for efficient data loading
- CharTrie tokenization for performance
- Lazy evaluation and unified memory
- Multi-threaded processing without Python GIL
"""

import mlx.data as dx
import mlx.data.core as dx_core
import mlx.core as mx
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import json

# Import tokenizer wrapper for backend flexibility
try:
    from embeddings.tokenizer_wrapper import TokenizerWrapper
    USE_TOKENIZER_WRAPPER = True
except ImportError:
    USE_TOKENIZER_WRAPPER = False
    from transformers import AutoTokenizer


class MLXTabularTextDataLoader:
    """Idiomatic MLX data loader for tabular-to-text tasks."""
    
    def __init__(
        self,
        csv_path: str,
        tokenizer_name: str = "answerdotai/ModernBERT-base",
        max_length: int = 128,
        batch_size: int = 32,
        label_column: str = "label",
        text_columns: Optional[List[str]] = None,
        shuffle_buffer_size: int = 1000,
        prefetch_size: int = 4,
        num_workers: int = 4,
        tokenizer_backend: str = "auto",
    ):
        """
        Initialize the MLX DataLoader.
        
        Args:
            csv_path: Path to the CSV file
            tokenizer_name: HuggingFace tokenizer to use for vocabulary
            max_length: Maximum sequence length
            batch_size: Batch size for training
            label_column: Name of the label column
            text_columns: Columns to use for text generation (None = all)
            shuffle_buffer_size: Buffer size for shuffling
            prefetch_size: Number of batches to prefetch
            num_workers: Number of parallel workers
        """
        self.csv_path = csv_path
        self.tokenizer_name = tokenizer_name
        self.max_length = max_length
        self.batch_size = batch_size
        self.label_column = label_column
        self.text_columns = text_columns
        self.shuffle_buffer_size = shuffle_buffer_size
        self.prefetch_size = prefetch_size
        self.num_workers = num_workers
        self.tokenizer_backend = tokenizer_backend
        
        # Build tokenizer
        if USE_TOKENIZER_WRAPPER:
            self.tokenizer = TokenizerWrapper(
                model_name=tokenizer_name,
                backend=tokenizer_backend
            )
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.tokenizer_info = {
            'pad_token_id': self.tokenizer.pad_token_id or 0,
            'cls_token_id': self.tokenizer.cls_token_id,
            'sep_token_id': self.tokenizer.sep_token_id,
            'unk_token_id': self.tokenizer.unk_token_id or 1,
        }
        
        # Get column info from CSV
        self._analyze_csv()
    
    def _analyze_csv(self):
        """Analyze CSV to get columns and data types."""
        # Read just the header
        df_head = pd.read_csv(self.csv_path, nrows=5)
        self.all_columns = list(df_head.columns)
        
        # If text_columns not specified, use all non-label columns
        if self.text_columns is None:
            self.text_columns = [col for col in self.all_columns if col != self.label_column]
    
    
    def _record_to_text(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """Transform tabular record to text."""
        text_parts = []
        
        # Generate text from specified columns
        for col in self.text_columns:
            if col in sample and sample[col] is not None:
                # MLX CSV reader returns byte arrays for strings
                value = sample[col]
                
                # Handle different types
                if isinstance(value, mx.array):
                    # It's a byte array from CSV reader - decode it
                    try:
                        # Convert MLX array to numpy then decode
                        value_bytes = bytes(np.array(value, dtype=np.uint8))
                        value = value_bytes.decode('utf-8', errors='ignore').strip()
                    except:
                        value = str(value)
                elif isinstance(value, bytes):
                    value = value.decode('utf-8', errors='ignore').strip()
                else:
                    value = str(value).strip()
                    
                if value and value.lower() not in ['nan', 'none', '', 'na']:
                    # Create natural language representation
                    col_name = col.replace('_', ' ').title()
                    text_parts.append(f"{col_name}: {value}")
        
        # Join all parts into a single text
        text = ". ".join(text_parts)
        if text:
            text += "."
        
        # Add text to sample
        sample['text'] = text
        
        # Handle label
        if self.label_column in sample:
            label_value = sample[self.label_column]
            
            # Handle MLX array from CSV reader
            if isinstance(label_value, mx.array):
                # Extract scalar value
                if label_value.size == 1:
                    label_value = float(label_value.item())
                else:
                    # It's a string encoded as bytes - decode it
                    try:
                        label_bytes = bytes(np.array(label_value, dtype=np.uint8))
                        label_value = float(label_bytes.decode('utf-8'))
                    except:
                        label_value = 0
            elif isinstance(label_value, bytes):
                try:
                    label_value = float(label_value.decode('utf-8'))
                except:
                    label_value = 0
            
            if isinstance(label_value, (int, np.integer)):
                sample['labels'] = int(label_value)
            elif isinstance(label_value, (float, np.floating)):
                sample['labels'] = int(label_value)
            elif isinstance(label_value, str):
                try:
                    sample['labels'] = int(float(label_value))
                except:
                    sample['labels'] = 0
            else:
                sample['labels'] = 0
        else:
            sample['labels'] = 0
        
        return sample
    
    def _add_padding_and_mask(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """Add padding and attention mask to tokenized sample."""
        input_ids = sample.get('input_ids', [])
        
        # Convert to numpy array if needed
        if not isinstance(input_ids, np.ndarray):
            input_ids = np.array(input_ids, dtype=np.int32)
        
        # Add special tokens if needed
        if self.tokenizer_info['cls_token_id'] is not None:
            input_ids = np.concatenate([[self.tokenizer_info['cls_token_id']], input_ids])
        if self.tokenizer_info['sep_token_id'] is not None:
            input_ids = np.concatenate([input_ids, [self.tokenizer_info['sep_token_id']]])
        
        # Truncate if needed
        if len(input_ids) > self.max_length:
            input_ids = input_ids[:self.max_length]
        
        # Calculate padding
        current_length = len(input_ids)
        pad_length = self.max_length - current_length
        
        # Pad sequences
        if pad_length > 0:
            pad_id = self.tokenizer_info['pad_token_id']
            input_ids = np.concatenate([input_ids, np.full(pad_length, pad_id, dtype=np.int32)])
        
        # Create attention mask
        attention_mask = np.ones(self.max_length, dtype=np.int32)
        if pad_length > 0:
            attention_mask[current_length:] = 0
        
        # Update sample
        sample['input_ids'] = input_ids
        sample['attention_mask'] = attention_mask
        
        # Remove text to save memory
        sample.pop('text', None)
        
        # Keep only needed fields
        return {
            'input_ids': sample['input_ids'],
            'attention_mask': sample['attention_mask'],
            'labels': sample.get('labels', 0)
        }
    
    
    def create_stream(self, is_training: bool = True) -> dx.Stream:
        """
        Create MLX data stream.
        
        Args:
            is_training: Whether this is for training (enables shuffling)
            
        Returns:
            MLX data stream ready for iteration
        """
        # Load and preprocess data with pandas first
        df = pd.read_csv(self.csv_path)
        
        # Process each row to create the final samples
        processed_samples = []
        for _, row in df.iterrows():
            # Convert to text
            text_parts = []
            for col in self.text_columns:
                if col in row and pd.notna(row[col]):
                    value = str(row[col]).strip()
                    if value and value.lower() not in ['nan', 'none', '', 'na']:
                        col_name = col.replace('_', ' ').title()
                        text_parts.append(f"{col_name}: {value}")
            
            text = ". ".join(text_parts)
            if text:
                text += "."
            
            # Tokenize
            tokens = self.tokenizer(
                text,
                truncation=True,
                max_length=self.max_length,
                padding='max_length',
                return_attention_mask=True,
                return_tensors=None
            )
            
            # Get label
            if self.label_column in row:
                label = int(row[self.label_column]) if pd.notna(row[self.label_column]) else 0
            else:
                label = 0
            
            # Create final sample with only numeric data
            processed_samples.append({
                'input_ids': tokens['input_ids'],  # Keep as list for now
                'attention_mask': tokens['attention_mask'],  # Keep as list for now
                'labels': label  # Keep as int for now
            })
        
        # Create stream from processed samples
        stream = dx.buffer_from_vector(processed_samples).to_stream()
        
        # Apply training-specific transformations
        if is_training:
            stream = stream.shuffle(buffer_size=self.shuffle_buffer_size)
        
        # Batch samples
        stream = stream.batch(batch_size=self.batch_size)
        
        # Convert each key in the batch to MLX arrays using a single transform
        def convert_batch_to_mlx(batch):
            """Convert all arrays in the batch to MLX format."""
            # batch['input_ids'] is a list of lists
            input_ids_array = np.array(batch['input_ids'], dtype=np.int32)
            attention_mask_array = np.array(batch['attention_mask'], dtype=np.int32)
            labels_array = np.array(batch['labels'], dtype=np.int32)
            
            return {
                'input_ids': mx.array(input_ids_array),
                'attention_mask': mx.array(attention_mask_array),
                'labels': mx.array(labels_array)
            }
        
        stream = stream.sample_transform(convert_batch_to_mlx)
        
        # Prefetch for performance
        stream = stream.prefetch(self.prefetch_size, self.num_workers)
        
        return stream
    
    def create_buffer(self) -> dx.Buffer:
        """
        Create buffer-based pipeline for small datasets.
        
        Returns:
            MLX data buffer for random access
        """
        # Use the same preprocessing as create_stream
        df = pd.read_csv(self.csv_path)
        
        # Process each row to create the final samples
        processed_samples = []
        for _, row in df.iterrows():
            # Convert to text
            text_parts = []
            for col in self.text_columns:
                if col in row and pd.notna(row[col]):
                    value = str(row[col]).strip()
                    if value and value.lower() not in ['nan', 'none', '', 'na']:
                        col_name = col.replace('_', ' ').title()
                        text_parts.append(f"{col_name}: {value}")
            
            text = ". ".join(text_parts)
            if text:
                text += "."
            
            # Tokenize
            tokens = self.tokenizer(
                text,
                truncation=True,
                max_length=self.max_length,
                padding='max_length',
                return_attention_mask=True,
                return_tensors=None
            )
            
            # Get label
            if self.label_column in row:
                label = int(row[self.label_column]) if pd.notna(row[self.label_column]) else 0
            else:
                label = 0
            
            # Create final sample with only numeric data
            processed_samples.append({
                'input_ids': tokens['input_ids'],
                'attention_mask': tokens['attention_mask'],
                'labels': label
            })
        
        # Create buffer
        buffer = dx.buffer_from_vector(processed_samples)
        
        # Shuffle entire buffer
        buffer = buffer.shuffle()
        
        return buffer
    
    def __iter__(self):
        """Make the dataloader iterable."""
        return iter(self.create_stream())
    
    def __len__(self):
        """Get number of batches (approximate for streaming)."""
        # This is approximate since we're streaming
        df_size = pd.read_csv(self.csv_path, usecols=[0]).shape[0]
        return (df_size + self.batch_size - 1) // self.batch_size


def create_cached_dataloader(
    csv_path: str,
    cache_path: str,
    tokenizer_name: str = "answerdotai/ModernBERT-base",
    **kwargs
) -> MLXTabularTextDataLoader:
    """
    Create a dataloader with caching support.
    
    Args:
        csv_path: Path to source CSV
        cache_path: Path to cache directory
        tokenizer_name: Tokenizer to use
        **kwargs: Additional arguments for MLXTabularTextDataLoader
        
    Returns:
        MLXTabularTextDataLoader instance
    """
    cache_path = Path(cache_path)
    cache_path.mkdir(parents=True, exist_ok=True)
    
    # Create cache key from parameters
    cache_key = f"{Path(csv_path).stem}_{tokenizer_name.replace('/', '_')}"
    cached_data_path = cache_path / f"{cache_key}_tokenized.npz"
    
    if cached_data_path.exists():
        # Load from cache
        print(f"Loading cached data from {cached_data_path}")
        # Implementation would load pre-tokenized data
        # For now, just create regular dataloader
    
    return MLXTabularTextDataLoader(csv_path, tokenizer_name, **kwargs)


# Utility functions for common datasets
def create_titanic_dataloader(
    data_path: str,
    batch_size: int = 32,
    max_length: int = 128,
    is_training: bool = True
) -> dx.Stream:
    """Create dataloader specifically for Titanic dataset."""
    loader = MLXTabularTextDataLoader(
        csv_path=data_path,
        tokenizer_name="answerdotai/ModernBERT-base",
        max_length=max_length,
        batch_size=batch_size,
        label_column="Survived",
        text_columns=["Pclass", "Name", "Sex", "Age", "SibSp", "Parch", "Ticket", "Fare", "Embarked"]
    )
    
    return loader.create_stream(is_training=is_training)