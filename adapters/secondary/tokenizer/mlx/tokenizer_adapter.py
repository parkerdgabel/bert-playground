from ..token_sequence import TokenSequence
"""MLX-native tokenizer adapter implementation."""

from typing import List, Dict, Optional, Union, Any
from pathlib import Path
import json


from adapters.secondary.tokenizer.base import BaseTokenizerAdapter


class MLXTokenizerAdapter(BaseTokenizerAdapter):
    """MLX-native implementation of TokenizerPort.
    
    This adapter provides direct MLX integration for tokenization,
    optimized for Apple Silicon performance.
    """
    
    def __init__(self, model_name_or_path: str, **kwargs):
        """Initialize MLX tokenizer adapter.
        
        Args:
            model_name_or_path: Model name or path to tokenizer
            **kwargs: Additional tokenizer configuration
        """
        super().__init__(model_name_or_path, **kwargs)
        
        # Try to load MLX tokenizer
        self._tokenizer = None
        self._load_mlx_tokenizer()
    
    def _load_mlx_tokenizer(self) -> None:
        """Load MLX tokenizer implementation."""
        try:
            # Check if it's a HuggingFace model that MLX supports
            from transformers import AutoTokenizer
            
            # Load as HuggingFace tokenizer first
            hf_tokenizer = AutoTokenizer.from_pretrained(
                self.model_name_or_path,
                **self.config
            )
            
            # Create MLX-optimized wrapper
            self._tokenizer = self._create_mlx_wrapper(hf_tokenizer)
            
            # Update vocabulary info
            self._vocab_size = hf_tokenizer.vocab_size
            self._update_special_tokens_from_hf(hf_tokenizer)
            
        except ImportError:
            raise ImportError(
                "transformers library required for MLX tokenizer. "
                "Install with: pip install transformers"
            )
    
    def _create_mlx_wrapper(self, hf_tokenizer) -> Any:
        """Create MLX-optimized wrapper around HuggingFace tokenizer.
        
        Args:
            hf_tokenizer: HuggingFace tokenizer instance
            
        Returns:
            MLX-optimized tokenizer wrapper
        """
        # For now, we'll use the HuggingFace tokenizer directly
        # In a real implementation, this would create an MLX-specific wrapper
        return hf_tokenizer
    
    def _update_special_tokens_from_hf(self, hf_tokenizer) -> None:
        """Update special tokens from HuggingFace tokenizer.
        
        Args:
            hf_tokenizer: HuggingFace tokenizer instance
        """
        self._special_tokens = {
            'pad_token': hf_tokenizer.pad_token,
            'unk_token': hf_tokenizer.unk_token,
            'cls_token': hf_tokenizer.cls_token,
            'sep_token': hf_tokenizer.sep_token,
            'mask_token': hf_tokenizer.mask_token,
        }
        
        # Store token IDs
        self._pad_token_id = hf_tokenizer.pad_token_id or 0
        self._unk_token_id = hf_tokenizer.unk_token_id or 0
        self._cls_token_id = hf_tokenizer.cls_token_id or hf_tokenizer.bos_token_id or 0
        self._sep_token_id = hf_tokenizer.sep_token_id or hf_tokenizer.eos_token_id or 0
        self._mask_token_id = hf_tokenizer.mask_token_id or self._unk_token_id
        
        # Remove None values
        self._special_tokens = {k: v for k, v in self._special_tokens.items() if v is not None}
    
    def _tokenize_single(
        self,
        text: str,
        max_length: Optional[int] = None,
        padding: bool = True,
        truncation: bool = True,
        return_attention_mask: bool = True,
        return_token_type_ids: bool = False,
    ) -> TokenSequence:
        """Tokenize a single text using MLX-optimized tokenizer.
        
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
        if self._tokenizer is None:
            raise RuntimeError("Tokenizer not initialized")
        
        # Use the tokenizer (currently HuggingFace wrapper)
        encoding = self._tokenizer(
            text,
            max_length=max_length,
            padding='max_length' if padding and max_length else padding,
            truncation=truncation,
            return_attention_mask=return_attention_mask,
            return_token_type_ids=return_token_type_ids,
            return_tensors=None,  # Return Python lists
        )
        
        # Create TokenSequence
        return TokenSequence(
            input_ids=encoding['input_ids'],
            attention_mask=encoding.get('attention_mask', [1] * len(encoding['input_ids'])),
            token_type_ids=encoding.get('token_type_ids'),
            position_ids=list(range(len(encoding['input_ids']))),  # MLX uses explicit position IDs
        )
    
    def decode(
        self,
        token_ids: List[int],
        skip_special_tokens: bool = True,
        clean_up_tokenization_spaces: bool = True,
    ) -> str:
        """Decode token IDs back to text.
        
        Args:
            token_ids: List of token IDs
            skip_special_tokens: Whether to skip special tokens
            clean_up_tokenization_spaces: Whether to clean up spaces
            
        Returns:
            Decoded text
        """
        if self._tokenizer is None:
            raise RuntimeError("Tokenizer not initialized")
        
        return self._tokenizer.decode(
            token_ids,
            skip_special_tokens=skip_special_tokens,
            clean_up_tokenization_spaces=clean_up_tokenization_spaces,
        )
    
    def convert_tokens_to_ids(self, tokens: List[str]) -> List[int]:
        """Convert tokens to IDs.
        
        Args:
            tokens: List of tokens
            
        Returns:
            List of token IDs
        """
        if self._tokenizer is None:
            raise RuntimeError("Tokenizer not initialized")
        
        return self._tokenizer.convert_tokens_to_ids(tokens)
    
    def convert_ids_to_tokens(self, ids: List[int]) -> List[str]:
        """Convert IDs to tokens.
        
        Args:
            ids: List of token IDs
            
        Returns:
            List of tokens
        """
        if self._tokenizer is None:
            raise RuntimeError("Tokenizer not initialized")
        
        return self._tokenizer.convert_ids_to_tokens(ids)
    
    @property
    def pad_token_id(self) -> int:
        """Get padding token ID."""
        return self._pad_token_id
    
    @property
    def cls_token_id(self) -> int:
        """Get CLS token ID."""
        return self._cls_token_id
    
    @property
    def sep_token_id(self) -> int:
        """Get SEP token ID."""
        return self._sep_token_id
    
    @property
    def mask_token_id(self) -> int:
        """Get MASK token ID."""
        return self._mask_token_id
    
    @property
    def unk_token_id(self) -> int:
        """Get UNK token ID."""
        return self._unk_token_id
    
    def _save_tokenizer_files(self, path: Path) -> None:
        """Save MLX tokenizer files.
        
        Args:
            path: Directory to save files
        """
        if self._tokenizer is None:
            raise RuntimeError("Tokenizer not initialized")
        
        # Save the underlying tokenizer
        if hasattr(self._tokenizer, 'save_pretrained'):
            self._tokenizer.save_pretrained(str(path))
        
        # Save MLX-specific configuration
        mlx_config = {
            'tokenizer_type': 'mlx',
            'model_name_or_path': self.model_name_or_path,
            'special_token_ids': {
                'pad': self._pad_token_id,
                'unk': self._unk_token_id,
                'cls': self._cls_token_id,
                'sep': self._sep_token_id,
                'mask': self._mask_token_id,
            }
        }
        
        mlx_config_path = path / "mlx_tokenizer_config.json"
        with open(mlx_config_path, 'w', encoding='utf-8') as f:
            json.dump(mlx_config, f, indent=2)
    
    def _load_tokenizer_files(self, path: Path) -> None:
        """Load MLX tokenizer files.
        
        Args:
            path: Directory to load files from
        """
        # Load MLX configuration if available
        mlx_config_path = path / "mlx_tokenizer_config.json"
        if mlx_config_path.exists():
            with open(mlx_config_path, 'r', encoding='utf-8') as f:
                mlx_config = json.load(f)
            
            # Restore special token IDs
            if 'special_token_ids' in mlx_config:
                ids = mlx_config['special_token_ids']
                self._pad_token_id = ids.get('pad', 0)
                self._unk_token_id = ids.get('unk', 0)
                self._cls_token_id = ids.get('cls', 0)
                self._sep_token_id = ids.get('sep', 0)
                self._mask_token_id = ids.get('mask', 0)
        
        # Load the tokenizer
        from transformers import AutoTokenizer
        self._tokenizer = AutoTokenizer.from_pretrained(str(path))
        self._vocab_size = self._tokenizer.vocab_size
        self._update_special_tokens_from_hf(self._tokenizer)
    
    def get_mlx_vocab_embeddings(self) -> Optional[Any]:
        """Get MLX-compatible vocabulary embeddings.
        
        Returns:
            MLX array of vocabulary embeddings or None
        """
        # This would return pre-computed embeddings optimized for MLX
        # For now, return None as embeddings would be handled by the model
        return None
    
    def optimize_for_mlx(self) -> None:
        """Optimize tokenizer for MLX performance."""
        # This would implement MLX-specific optimizations
        # such as pre-computing common tokenizations,
        # optimizing vocabulary lookups, etc.
        pass