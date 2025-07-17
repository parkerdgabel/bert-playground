"""
MLX-native DataLoader for Kaggle tabular-to-text problems.

This module provides an efficient, stream-based data loading solution
optimized for Apple Silicon using MLX's native data pipeline.
"""

import mlx.data as dx
import mlx.core as mx
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Any, Iterator
from loguru import logger

# Import tokenizer wrapper for backend flexibility
try:
    from embeddings.tokenizer_wrapper import TokenizerWrapper
    USE_TOKENIZER_WRAPPER = True
except ImportError:
    USE_TOKENIZER_WRAPPER = False
    from transformers import AutoTokenizer


class KaggleDataLoader:
    """Efficient MLX-native data loader for Kaggle competitions.
    
    This loader converts tabular data to text and provides efficient
    streaming with MLX's data pipeline for optimal performance on
    Apple Silicon.
    """
    
    def __init__(
        self,
        csv_path: str,
        tokenizer_name: str = "answerdotai/ModernBERT-base",
        max_length: int = 128,
        batch_size: int = 32,
        label_column: str = "label",
        text_columns: Optional[List[str]] = None,
        shuffle: bool = True,
        shuffle_buffer_size: int = 1000,
        prefetch_size: int = 4,
        num_workers: int = 4,
        text_template: Optional[str] = None,
        tokenizer_backend: str = "auto",
    ):
        """
        Initialize the Kaggle DataLoader.
        
        Args:
            csv_path: Path to the CSV file
            tokenizer_name: HuggingFace tokenizer to use
            max_length: Maximum sequence length
            batch_size: Batch size for training
            label_column: Name of the label column
            text_columns: Columns to use for text generation (None = all except label)
            shuffle: Whether to shuffle the data
            shuffle_buffer_size: Buffer size for shuffling
            prefetch_size: Number of batches to prefetch
            num_workers: Number of parallel workers
            text_template: Optional custom text template
        """
        self.csv_path = Path(csv_path)
        self.tokenizer_name = tokenizer_name
        self.max_length = max_length
        self.batch_size = batch_size
        self.label_column = label_column
        self.text_columns = text_columns
        self.shuffle = shuffle
        self.shuffle_buffer_size = shuffle_buffer_size
        self.prefetch_size = prefetch_size
        self.num_workers = num_workers
        self.text_template = text_template
        self.tokenizer_backend = tokenizer_backend
        
        # Validate inputs
        if not self.csv_path.exists():
            raise FileNotFoundError(f"CSV file not found: {self.csv_path}")
        
        # Initialize tokenizer
        logger.info(f"Loading tokenizer: {tokenizer_name} (backend: {tokenizer_backend})")
        if USE_TOKENIZER_WRAPPER:
            self.tokenizer = TokenizerWrapper(
                model_name=tokenizer_name,
                backend=tokenizer_backend
            )
        else:
            logger.warning("TokenizerWrapper not available, using HuggingFace tokenizer")
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        
        # Get tokenizer special tokens
        self._setup_tokenizer()
        
        # Analyze CSV structure
        self._analyze_csv()
        
        # Preprocess data for efficiency
        self._preprocessed_data = self._preprocess_data()
        
        # Optional dataset spec for trainer compatibility
        self.dataset_spec = None
        
        logger.info(
            f"Initialized KaggleDataLoader: "
            f"{len(self._preprocessed_data)} samples, "
            f"{len(self._preprocessed_data) // batch_size} batches"
        )
    
    def _setup_tokenizer(self):
        """Setup tokenizer with proper special tokens."""
        # Ensure we have all required special tokens
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token or "[PAD]"
        
        self.pad_token_id = self.tokenizer.pad_token_id or 0
        self.cls_token_id = self.tokenizer.cls_token_id
        self.sep_token_id = self.tokenizer.sep_token_id
    
    def _analyze_csv(self):
        """Analyze CSV to determine columns and data types."""
        # Read header and a few rows to understand structure
        df_head = pd.read_csv(self.csv_path, nrows=5)
        self.all_columns = list(df_head.columns)
        
        # Determine text columns if not specified
        if self.text_columns is None:
            self.text_columns = [col for col in self.all_columns if col != self.label_column]
        
        # Validate label column exists
        self.has_labels = self.label_column in self.all_columns
        
        logger.debug(f"CSV columns: {self.all_columns}")
        logger.debug(f"Text columns: {self.text_columns}")
        logger.debug(f"Has labels: {self.has_labels}")
    
    def _preprocess_data(self) -> List[Dict[str, Any]]:
        """Preprocess the entire dataset for efficient streaming."""
        logger.info("Preprocessing data...")
        
        # Load CSV
        df = pd.read_csv(self.csv_path)
        processed_samples = []
        
        for idx, row in df.iterrows():
            # Convert row to text
            text = self._row_to_text(row)
            
            # Tokenize
            tokens = self.tokenizer(
                text,
                truncation=True,
                max_length=self.max_length,
                padding='max_length',
                return_attention_mask=True,
                return_tensors=None
            )
            
            # Get label if available
            label = 0
            if self.has_labels and pd.notna(row[self.label_column]):
                label = int(row[self.label_column])
            
            # Create sample - ensure int32 for MLX compatibility
            sample = {
                'input_ids': np.array(tokens['input_ids'], dtype=np.int32),
                'attention_mask': np.array(tokens['attention_mask'], dtype=np.int32),
                'labels': np.array(label, dtype=np.int32)
            }
            
            processed_samples.append(sample)
        
        logger.info(f"Preprocessed {len(processed_samples)} samples")
        return processed_samples
    
    def _row_to_text(self, row: pd.Series) -> str:
        """Convert a dataframe row to text representation."""
        if self.text_template:
            # Use custom template if provided
            return self.text_template.format(**row.to_dict())
        
        # Default text generation
        text_parts = []
        
        for col in self.text_columns:
            if col in row and pd.notna(row[col]):
                value = str(row[col]).strip()
                if value and value.lower() not in ['nan', 'none', '', 'na']:
                    # Create natural language representation
                    col_name = col.replace('_', ' ').title()
                    text_parts.append(f"{col_name}: {value}")
        
        # Join with proper punctuation
        text = ". ".join(text_parts)
        if text:
            text += "."
        
        return text
    
    def create_stream(self) -> dx.Stream:
        """
        Create MLX data stream for efficient data loading.
        
        Returns:
            MLX Stream object ready for iteration
        """
        # Create stream from preprocessed samples
        stream = dx.buffer_from_vector(self._preprocessed_data).to_stream()
        
        # Shuffle if training
        if self.shuffle:
            stream = stream.shuffle(buffer_size=self.shuffle_buffer_size)
        
        # Batch samples
        stream = stream.batch(batch_size=self.batch_size)
        
        # Convert to MLX arrays
        def convert_batch_to_mlx(batch):
            """Convert batch arrays to MLX format."""
            # Extract individual arrays from batch
            input_ids_batch = batch['input_ids']
            attention_mask_batch = batch['attention_mask']
            labels_batch = batch['labels']
            
            # Ensure they are numpy arrays with correct dtype
            input_ids = np.array(input_ids_batch, dtype=np.int32)
            attention_mask = np.array(attention_mask_batch, dtype=np.int32)
            labels = np.array(labels_batch, dtype=np.int32)
            
            # Remove extra dimensions if present
            if input_ids.ndim > 2:
                input_ids = input_ids.squeeze()
            if attention_mask.ndim > 2:
                attention_mask = attention_mask.squeeze()
            if labels.ndim > 1:
                labels = labels.squeeze()
            
            # Convert to MLX arrays
            result = {
                'input_ids': mx.array(input_ids),
                'attention_mask': mx.array(attention_mask),
                'labels': mx.array(labels)
            }
            
            # Debug: verify conversion
            logger.debug(f"Converted batch to MLX: input_ids type={type(result['input_ids'])}, dtype={result['input_ids'].dtype}")
            
            return result
        
        stream = stream.sample_transform(convert_batch_to_mlx)
        
        # Prefetch for performance
        stream = stream.prefetch(self.prefetch_size, self.num_workers)
        
        return stream
    
    def __iter__(self) -> Iterator[Dict[str, mx.array]]:
        """Make the dataloader iterable."""
        for batch in self.create_stream():
            # Ensure we're returning MLX arrays
            if not isinstance(batch['input_ids'], mx.array):
                logger.debug(f"Converting batch to MLX arrays in __iter__: {type(batch['input_ids'])}")
                batch = {
                    'input_ids': mx.array(batch['input_ids']),
                    'attention_mask': mx.array(batch['attention_mask']),
                    'labels': mx.array(batch['labels'])
                }
            yield batch
    
    def __len__(self) -> int:
        """Get number of batches."""
        return (len(self._preprocessed_data) + self.batch_size - 1) // self.batch_size
    
    @property
    def num_samples(self) -> int:
        """Get total number of samples."""
        return len(self._preprocessed_data)


def create_kaggle_dataloader(
    dataset_name: str,
    csv_path: str,
    tokenizer_name: str = "answerdotai/ModernBERT-base",
    batch_size: int = 32,
    max_length: int = 128,
    shuffle: bool = True,
    attach_dataset_spec: bool = True,
    tokenizer_backend: str = "auto",
    **kwargs
) -> KaggleDataLoader:
    """
    Factory function to create dataloaders for specific Kaggle datasets.
    
    Args:
        dataset_name: Name of the Kaggle dataset (e.g., 'titanic', 'house_prices')
        csv_path: Path to CSV file
        tokenizer_name: HuggingFace tokenizer name
        batch_size: Batch size
        max_length: Maximum sequence length
        shuffle: Whether to shuffle data
        attach_dataset_spec: Whether to attach dataset spec for trainer compatibility
        tokenizer_backend: Tokenizer backend to use ('auto', 'mlx', 'huggingface')
        **kwargs: Additional arguments for KaggleDataLoader
        
    Returns:
        Configured KaggleDataLoader instance
    """
    # Dataset-specific configurations
    configs = {
        'titanic': {
            'label_column': 'Survived',
            'text_columns': ['Pclass', 'Name', 'Sex', 'Age', 'SibSp', 'Parch', 'Ticket', 'Fare', 'Embarked']
        },
        'house_prices': {
            'label_column': 'SalePrice',
            'text_columns': None  # Use all columns
        },
        'store_sales': {
            'label_column': 'sales',
            'text_columns': ['store_nbr', 'family', 'onpromotion', 'date']
        }
    }
    
    # Get dataset config or use defaults
    config = configs.get(dataset_name, {})
    
    # Merge with kwargs
    final_config = {
        'csv_path': csv_path,
        'tokenizer_name': tokenizer_name,
        'batch_size': batch_size,
        'max_length': max_length,
        'shuffle': shuffle,
        'tokenizer_backend': tokenizer_backend,
        **config,
        **kwargs
    }
    
    loader = KaggleDataLoader(**final_config)
    
    # Optionally attach dataset spec for trainer compatibility
    if attach_dataset_spec:
        from .datasets import get_dataset_spec
        spec = get_dataset_spec(dataset_name)
        if spec:
            loader.dataset_spec = spec
    
    return loader


# Convenience functions for common datasets
def create_titanic_dataloader(
    csv_path: str,
    batch_size: int = 32,
    max_length: int = 128,
    shuffle: bool = True,
    **kwargs
) -> KaggleDataLoader:
    """Create dataloader specifically for Titanic dataset."""
    return create_kaggle_dataloader(
        'titanic',
        csv_path,
        batch_size=batch_size,
        max_length=max_length,
        shuffle=shuffle,
        **kwargs
    )