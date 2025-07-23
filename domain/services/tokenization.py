"""Tokenization service for text processing."""

from typing import List, Optional, Dict, Any, Union
from dataclasses import dataclass
from domain.entities.dataset import TokenSequence, DataBatch
from domain.ports.tokenizer import TokenizerPort


@dataclass
class TokenizationService:
    """Service for text tokenization operations."""
    tokenizer_port: TokenizerPort
    
    def tokenize_texts(
        self,
        texts: List[str],
        max_length: int = 512,
        batch_size: int = 1000,
        show_progress: bool = False,
    ) -> List[TokenSequence]:
        """Tokenize a list of texts.
        
        Args:
            texts: List of text strings
            max_length: Maximum sequence length
            batch_size: Batch size for tokenization
            show_progress: Whether to show progress
            
        Returns:
            List of token sequences
        """
        all_sequences = []
        
        # Process in batches for efficiency
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            
            # Tokenize batch
            sequences = self.tokenizer_port.tokenize(
                text=batch_texts,
                max_length=max_length,
                padding=True,
                truncation=True,
                return_attention_mask=True,
                return_token_type_ids=False,
            )
            
            if isinstance(sequences, TokenSequence):
                sequences = [sequences]
            
            all_sequences.extend(sequences)
        
        return all_sequences
    
    def tokenize_for_mlm(
        self,
        texts: List[str],
        max_length: int = 512,
        mlm_probability: float = 0.15,
    ) -> List[Dict[str, Any]]:
        """Tokenize texts for masked language modeling.
        
        Args:
            texts: List of texts
            max_length: Maximum sequence length
            mlm_probability: Probability of masking tokens
            
        Returns:
            List of dictionaries with tokenized data and MLM labels
        """
        sequences = self.tokenize_texts(texts, max_length)
        mlm_data = []
        
        mask_token_id = self.tokenizer_port.mask_token_id
        vocab_size = self.tokenizer_port.get_vocab_size()
        
        for seq in sequences:
            # Create MLM labels
            import random
            labels = [-100] * len(seq.input_ids)  # -100 = ignore in loss
            
            # Randomly mask tokens
            for i, token_id in enumerate(seq.input_ids):
                if seq.attention_mask[i] == 0:  # Skip padding
                    continue
                    
                if random.random() < mlm_probability:
                    labels[i] = token_id  # Store original token
                    
                    # 80% mask, 10% random, 10% unchanged
                    rand = random.random()
                    if rand < 0.8:
                        seq.input_ids[i] = mask_token_id
                    elif rand < 0.9:
                        seq.input_ids[i] = random.randint(0, vocab_size - 1)
                    # else: keep original token
            
            mlm_data.append({
                'input_ids': seq.input_ids,
                'attention_mask': seq.attention_mask,
                'labels': labels,
            })
        
        return mlm_data
    
    def create_paired_sequences(
        self,
        texts_a: List[str],
        texts_b: List[str],
        max_length: int = 512,
    ) -> List[TokenSequence]:
        """Create paired sequences for tasks like sentence similarity.
        
        Args:
            texts_a: First set of texts
            texts_b: Second set of texts
            max_length: Maximum total sequence length
            
        Returns:
            List of token sequences with proper segment IDs
        """
        if len(texts_a) != len(texts_b):
            raise ValueError("texts_a and texts_b must have same length")
        
        paired_texts = [f"{a} {self._get_sep_token()} {b}" 
                       for a, b in zip(texts_a, texts_b)]
        
        sequences = self.tokenizer_port.tokenize(
            text=paired_texts,
            max_length=max_length,
            padding=True,
            truncation=True,
            return_attention_mask=True,
            return_token_type_ids=True,  # Important for paired sequences
        )
        
        if isinstance(sequences, TokenSequence):
            sequences = [sequences]
            
        return sequences
    
    def _get_sep_token(self) -> str:
        """Get separator token string."""
        # This would ideally come from tokenizer
        return "[SEP]"
    
    def decode_predictions(
        self,
        token_ids: List[List[int]],
        skip_special_tokens: bool = True,
    ) -> List[str]:
        """Decode model predictions back to text.
        
        Args:
            token_ids: Batch of token ID sequences
            skip_special_tokens: Whether to skip special tokens
            
        Returns:
            List of decoded texts
        """
        return self.tokenizer_port.batch_decode(
            token_ids_batch=token_ids,
            skip_special_tokens=skip_special_tokens,
        )
    
    def analyze_tokenization(
        self,
        texts: List[str],
    ) -> Dict[str, Any]:
        """Analyze tokenization statistics.
        
        Args:
            texts: List of texts to analyze
            
        Returns:
            Dictionary with tokenization statistics
        """
        sequences = self.tokenize_texts(texts, max_length=10000)  # Large limit
        
        lengths = [seq.num_tokens for seq in sequences]
        
        return {
            'num_texts': len(texts),
            'avg_tokens': sum(lengths) / len(lengths) if lengths else 0,
            'min_tokens': min(lengths) if lengths else 0,
            'max_tokens': max(lengths) if lengths else 0,
            'total_tokens': sum(lengths),
            'vocab_size': self.tokenizer_port.get_vocab_size(),
        }
    
    def create_data_batches(
        self,
        texts: List[str],
        labels: Optional[List[Any]] = None,
        batch_size: int = 32,
        max_length: int = 512,
    ) -> List[DataBatch]:
        """Create data batches from texts.
        
        Args:
            texts: List of texts
            labels: Optional labels
            batch_size: Batch size
            max_length: Maximum sequence length
            
        Returns:
            List of data batches
        """
        # Tokenize all texts
        sequences = self.tokenize_texts(texts, max_length)
        
        # Create batches
        batches = []
        for i in range(0, len(sequences), batch_size):
            batch_sequences = sequences[i:i + batch_size]
            batch_labels = None
            
            if labels is not None:
                batch_labels = labels[i:i + batch_size]
            
            batch = DataBatch(
                sequences=batch_sequences,
                labels=batch_labels,
            )
            
            # Collate to ensure uniform length within batch
            batch = batch.collate(pad_token_id=self.tokenizer_port.pad_token_id)
            
            batches.append(batch)
        
        return batches