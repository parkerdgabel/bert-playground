import mlx.core as mx
import mlx.data as dx
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from transformers import AutoTokenizer
from loguru import logger

from .text_templates import TitanicTextTemplates


class TitanicDataPipeline:
    def __init__(
        self,
        data_path: str,
        tokenizer_name: str = "answerdotai/ModernBERT-base",
        max_length: int = 256,
        batch_size: int = 32,
        is_training: bool = True,
        augment: bool = True
    ):
        self.data_path = Path(data_path)
        self.max_length = max_length
        self.batch_size = batch_size
        self.is_training = is_training
        self.augment = augment and is_training
        
        logger.info(f"Initializing TitanicDataPipeline:")
        logger.info(f"  Data path: {self.data_path}")
        logger.info(f"  Tokenizer: {tokenizer_name}")
        logger.info(f"  Max length: {max_length}")
        logger.info(f"  Batch size: {batch_size}")
        logger.info(f"  Training mode: {is_training}")
        logger.info(f"  Augmentation: {self.augment}")
        
        # Initialize tokenizer
        logger.debug(f"Loading tokenizer: {tokenizer_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            logger.debug("Set pad_token to eos_token")
        
        # Initialize text template converter
        self.text_converter = TitanicTextTemplates()
        logger.debug("Initialized text template converter")
        
        # Load and prepare data
        logger.info(f"Loading data from {self.data_path}")
        self.df = pd.read_csv(self.data_path)
        logger.info(f"Loaded {len(self.df)} rows")
        self.prepare_data()
    
    def prepare_data(self):
        logger.info("Preparing data...")
        
        # Log missing values before filling
        missing_stats = self.df.isnull().sum()
        if missing_stats.any():
            logger.debug("Missing values before preprocessing:")
            for col, count in missing_stats[missing_stats > 0].items():
                logger.debug(f"  {col}: {count} ({count/len(self.df)*100:.1f}%)")
        
        # Fill missing values
        self.df['Age'] = self.df['Age'].fillna(self.df['Age'].median())
        self.df['Fare'] = self.df['Fare'].fillna(self.df['Fare'].median())
        self.df['Embarked'] = self.df['Embarked'].fillna('S')
        logger.debug("Filled missing values with median/mode")
        
        # Convert to text representations
        self.texts = []
        self.labels = []
        
        logger.info("Converting tabular data to text representations...")
        for idx, (_, row) in enumerate(self.df.iterrows()):
            # Generate text representation
            text = self.text_converter.row_to_text(row.to_dict())
            
            if idx < 3:  # Log first few examples
                logger.debug(f"Example {idx + 1}: {text[:100]}...")
            
            if self.augment:
                # Add augmented versions
                augmented_texts = self.text_converter.augment_text(text)
                for aug_text in augmented_texts:
                    self.texts.append(aug_text)
                    if self.is_training:
                        self.labels.append(row['Survived'])
            else:
                self.texts.append(text)
                if self.is_training:
                    self.labels.append(row['Survived'])
        
        # Convert labels to numpy array if training
        if self.is_training:
            self.labels = np.array(self.labels, dtype=np.int32)
            
            # Log label distribution
            unique, counts = np.unique(self.labels, return_counts=True)
            logger.info("Label distribution:")
            for label, count in zip(unique, counts):
                logger.info(f"  Class {label}: {count} ({count/len(self.labels)*100:.1f}%)")
        
        logger.info(f"Prepared {len(self.texts)} text samples")
        if self.augment:
            logger.info(f"  Original samples: {len(self.df)}")
            logger.info(f"  Augmentation factor: {len(self.texts) / len(self.df):.1f}x")
    
    def tokenize_function(self, text: str) -> Dict[str, mx.array]:
        # Tokenize text
        encoded = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='np'
        )
        
        # Convert to MLX arrays
        return {
            'input_ids': mx.array(encoded['input_ids'][0]),
            'attention_mask': mx.array(encoded['attention_mask'][0])
        }
    
    def create_dataset(self):
        """Create MLX data iterator."""
        # For simplicity, we'll create a simple iterator
        # MLX data's buffer_from_vector might not work with complex objects
        
        def data_generator():
            indices = list(range(len(self.texts)))
            
            if self.is_training:
                # Shuffle for training
                import random
                random.shuffle(indices)
            
            # Process in batches
            for i in range(0, len(indices), self.batch_size):
                batch_indices = indices[i:i + self.batch_size]
                batch_texts = [self.texts[idx] for idx in batch_indices]
                
                # Tokenize batch
                batch_input_ids = []
                batch_attention_masks = []
                batch_labels = []
                
                for idx, text in zip(batch_indices, batch_texts):
                    tokens = self.tokenize_function(text)
                    batch_input_ids.append(tokens['input_ids'])
                    batch_attention_masks.append(tokens['attention_mask'])
                    
                    if self.is_training:
                        batch_labels.append(self.labels[idx])
                
                # Stack into arrays
                batch_data = {
                    'input_ids': mx.stack(batch_input_ids),
                    'attention_mask': mx.stack(batch_attention_masks)
                }
                
                if self.is_training:
                    # Convert numpy int32 to Python int for MLX
                    batch_labels_int = [int(label) for label in batch_labels]
                    batch_data['labels'] = mx.array(batch_labels_int).reshape(-1, 1)
                
                yield batch_data
        
        return data_generator
    
    def get_dataloader(self):
        return self.create_dataset()
    
    def __len__(self) -> int:
        return len(self.texts)
    
    def get_num_batches(self) -> int:
        return (len(self.texts) + self.batch_size - 1) // self.batch_size


def create_data_loaders(
    train_path: str,
    val_path: Optional[str] = None,
    tokenizer_name: str = "answerdotai/ModernBERT-base",
    batch_size: int = 32,
    max_length: int = 256,
    val_split: float = 0.2
) -> Tuple[TitanicDataPipeline, Optional[TitanicDataPipeline]]:
    # Create training data loader
    train_loader = TitanicDataPipeline(
        data_path=train_path,
        tokenizer_name=tokenizer_name,
        max_length=max_length,
        batch_size=batch_size,
        is_training=True,
        augment=True
    )
    
    # Create validation loader if path provided
    val_loader = None
    if val_path:
        val_loader = TitanicDataPipeline(
            data_path=val_path,
            tokenizer_name=tokenizer_name,
            max_length=max_length,
            batch_size=batch_size,
            is_training=True,
            augment=False
        )
    elif val_split > 0:
        # Split training data for validation
        # This is a simplified version - in practice, we'd want to split before augmentation
        print(f"Note: Validation split from training data not fully implemented. Use separate validation file.")
    
    return train_loader, val_loader