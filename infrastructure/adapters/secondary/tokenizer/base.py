"""Base tokenizer adapter with common functionality."""

from abc import ABC, abstractmethod
from typing import List, Dict, Optional, Union, Any
from pathlib import Path

from application.ports.secondary.tokenizer import TokenizerPort
from .token_sequence import TokenSequence



class BaseTokenizerAdapter(TokenizerPort, ABC):
    """Base tokenizer adapter with common functionality."""
    
    def __init__(self, model_name_or_path: str, **kwargs):
        """Initialize base tokenizer adapter.
        
        Args:
            model_name_or_path: Model name or path to tokenizer
            **kwargs: Additional tokenizer configuration
        """
        self.model_name_or_path = model_name_or_path
        self.config = kwargs
        self._special_tokens = {}
        self._vocab_size = 0
    
    def tokenize(
        self,
        text: Union[str, List[str]],
        max_length: Optional[int] = None,
        padding: bool = True,
        truncation: bool = True,
        return_attention_mask: bool = True,
        return_token_type_ids: bool = False,
    ) -> Union[TokenSequence, List[TokenSequence]]:
        """Tokenize text input.
        
        Args:
            text: Single text or list of texts
            max_length: Maximum sequence length
            padding: Whether to pad sequences
            truncation: Whether to truncate long sequences
            return_attention_mask: Whether to return attention mask
            return_token_type_ids: Whether to return token type IDs
            
        Returns:
            TokenSequence or list of TokenSequences
        """
        # Handle batch vs single input
        is_batch = isinstance(text, list)
        texts = text if is_batch else [text]
        
        # Tokenize all texts
        sequences = []
        for t in texts:
            tokens = self._tokenize_single(
                t,
                max_length=max_length,
                padding=padding,
                truncation=truncation,
                return_attention_mask=return_attention_mask,
                return_token_type_ids=return_token_type_ids,
            )
            sequences.append(tokens)
        
        # Return single sequence if input was single
        return sequences if is_batch else sequences[0]
    
    @abstractmethod
    def _tokenize_single(
        self,
        text: str,
        max_length: Optional[int] = None,
        padding: bool = True,
        truncation: bool = True,
        return_attention_mask: bool = True,
        return_token_type_ids: bool = False,
    ) -> TokenSequence:
        """Tokenize a single text.
        
        Args:
            text: Text to tokenize
            max_length: Maximum sequence length
            padding: Whether to pad sequences
            truncation: Whether to truncate long sequences
            return_attention_mask: Whether to return attention mask
            return_token_type_ids: Whether to return token type IDs
            
        Returns:
            TokenSequence
        """
        ...
    
    def batch_decode(
        self,
        token_ids_batch: List[List[int]],
        skip_special_tokens: bool = True,
        clean_up_tokenization_spaces: bool = True,
    ) -> List[str]:
        """Decode batch of token IDs.
        
        Args:
            token_ids_batch: Batch of token ID lists
            skip_special_tokens: Whether to skip special tokens
            clean_up_tokenization_spaces: Whether to clean up spaces
            
        Returns:
            List of decoded texts
        """
        return [
            self.decode(
                token_ids,
                skip_special_tokens=skip_special_tokens,
                clean_up_tokenization_spaces=clean_up_tokenization_spaces,
            )
            for token_ids in token_ids_batch
        ]
    
    def get_vocab_size(self) -> int:
        """Get vocabulary size.
        
        Returns:
            Size of vocabulary
        """
        return self._vocab_size
    
    def get_special_tokens(self) -> Dict[str, int]:
        """Get special token mappings.
        
        Returns:
            Dictionary mapping special token names to IDs
        """
        return self._special_tokens.copy()
    
    def _add_special_tokens(self, text: str, token_ids: List[int]) -> List[int]:
        """Add special tokens to token IDs.
        
        Args:
            text: Original text
            token_ids: Token IDs without special tokens
            
        Returns:
            Token IDs with special tokens
        """
        # Add CLS token at the beginning
        if hasattr(self, 'cls_token_id') and self.cls_token_id is not None:
            token_ids = [self.cls_token_id] + token_ids
        
        # Add SEP token at the end
        if hasattr(self, 'sep_token_id') and self.sep_token_id is not None:
            token_ids = token_ids + [self.sep_token_id]
        
        return token_ids
    
    def _create_attention_mask(self, token_ids: List[int], max_length: Optional[int] = None) -> List[int]:
        """Create attention mask for token IDs.
        
        Args:
            token_ids: Token IDs
            max_length: Maximum sequence length
            
        Returns:
            Attention mask
        """
        attention_mask = [1] * len(token_ids)
        
        if max_length and len(attention_mask) < max_length:
            # Pad attention mask
            attention_mask += [0] * (max_length - len(attention_mask))
        
        return attention_mask
    
    def _pad_sequence(
        self,
        token_ids: List[int],
        max_length: int,
        pad_token_id: Optional[int] = None,
    ) -> List[int]:
        """Pad sequence to maximum length.
        
        Args:
            token_ids: Token IDs to pad
            max_length: Maximum length
            pad_token_id: Padding token ID
            
        Returns:
            Padded token IDs
        """
        if pad_token_id is None:
            pad_token_id = self.pad_token_id if hasattr(self, 'pad_token_id') else 0
        
        if len(token_ids) < max_length:
            token_ids = token_ids + [pad_token_id] * (max_length - len(token_ids))
        
        return token_ids
    
    def _truncate_sequence(
        self,
        token_ids: List[int],
        max_length: int,
        strategy: str = 'longest_first',
    ) -> List[int]:
        """Truncate sequence to maximum length.
        
        Args:
            token_ids: Token IDs to truncate
            max_length: Maximum length
            strategy: Truncation strategy
            
        Returns:
            Truncated token IDs
        """
        if len(token_ids) <= max_length:
            return token_ids
        
        # Simple truncation - keep first max_length tokens
        # Account for special tokens if needed
        special_tokens_count = 0
        if hasattr(self, 'cls_token_id') and self.cls_token_id is not None:
            special_tokens_count += 1
        if hasattr(self, 'sep_token_id') and self.sep_token_id is not None:
            special_tokens_count += 1
        
        # Truncate accounting for special tokens
        if special_tokens_count > 0:
            # Keep CLS, truncate middle, keep SEP
            if hasattr(self, 'cls_token_id') and hasattr(self, 'sep_token_id'):
                return [token_ids[0]] + token_ids[1:max_length-1] + [token_ids[-1]]
            else:
                return token_ids[:max_length]
        else:
            return token_ids[:max_length]
    
    def save(self, path: str) -> None:
        """Save tokenizer to disk.
        
        Args:
            path: Save directory path
        """
        save_path = Path(path)
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Save configuration
        import json
        config_path = save_path / "tokenizer_config.json"
        config = {
            'model_name_or_path': self.model_name_or_path,
            'vocab_size': self._vocab_size,
            'special_tokens': self._special_tokens,
            **self.config
        }
        
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2)
        
        # Subclasses should save their specific files
        self._save_tokenizer_files(save_path)
    
    @abstractmethod
    def _save_tokenizer_files(self, path: Path) -> None:
        """Save tokenizer-specific files.
        
        Args:
            path: Directory to save files
        """
        ...
    
    @classmethod
    def load(cls, path: str) -> 'BaseTokenizerAdapter':
        """Load tokenizer from disk.
        
        Args:
            path: Load directory path
            
        Returns:
            Loaded tokenizer
        """
        load_path = Path(path)
        
        # Load configuration
        import json
        config_path = load_path / "tokenizer_config.json"
        
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        model_name_or_path = config.pop('model_name_or_path', path)
        
        # Create instance with saved configuration
        instance = cls(model_name_or_path, **config)
        
        # Load tokenizer-specific files
        instance._load_tokenizer_files(load_path)
        
        return instance
    
    @abstractmethod
    def _load_tokenizer_files(self, path: Path) -> None:
        """Load tokenizer-specific files.
        
        Args:
            path: Directory to load files from
        """
        ...