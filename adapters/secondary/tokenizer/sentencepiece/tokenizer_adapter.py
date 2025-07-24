from ..token_sequence import TokenSequence
"""SentencePiece tokenizer adapter implementation."""

from typing import List, Dict, Optional, Union, Any, Tuple
from pathlib import Path
import json

from infrastructure.di import adapter, Scope
from application.ports.secondary.tokenizer import TokenizerPort
from adapters.secondary.tokenizer.base import BaseTokenizerAdapter


@adapter(TokenizerPort, scope=Scope.SINGLETON)
class SentencePieceTokenizerAdapter(BaseTokenizerAdapter):
    """SentencePiece implementation of TokenizerPort."""
    
    def __init__(
        self,
        model_name_or_path: str,
        add_bos: bool = True,
        add_eos: bool = True,
        **kwargs
    ):
        """Initialize SentencePiece tokenizer adapter.
        
        Args:
            model_name_or_path: Path to SentencePiece model file (.model)
            add_bos: Whether to add beginning of sentence token
            add_eos: Whether to add end of sentence token
            **kwargs: Additional tokenizer configuration
        """
        super().__init__(model_name_or_path, **kwargs)
        
        self.add_bos = add_bos
        self.add_eos = add_eos
        
        try:
            import sentencepiece as spm
        except ImportError:
            raise ImportError(
                "sentencepiece library required for SentencePiece tokenizer. "
                "Install with: pip install sentencepiece"
            )
        
        # Initialize SentencePiece processor
        self._sp = spm.SentencePieceProcessor()
        
        # Load model
        if Path(model_name_or_path).exists():
            self._sp.Load(model_name_or_path)
        else:
            # Try to load from HuggingFace if not a local path
            self._load_from_huggingface()
        
        # Set vocabulary size
        self._vocab_size = self._sp.vocab_size()
        
        # Set special tokens
        self._setup_special_tokens()
    
    def _load_from_huggingface(self) -> None:
        """Try to load SentencePiece model from HuggingFace."""
        try:
            from transformers import AutoTokenizer
            
            # Load tokenizer from HuggingFace
            hf_tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path)
            
            # Get the SentencePiece model file
            if hasattr(hf_tokenizer, 'vocab_file'):
                self._sp.Load(hf_tokenizer.vocab_file)
            else:
                raise ValueError(f"Cannot find SentencePiece model for {self.model_name_or_path}")
                
        except Exception as e:
            raise ValueError(f"Failed to load SentencePiece model: {e}")
    
    def _setup_special_tokens(self) -> None:
        """Setup special tokens for SentencePiece."""
        # Common special token IDs in SentencePiece
        self._pad_token_id = self._sp.pad_id() if hasattr(self._sp, 'pad_id') else 0
        self._unk_token_id = self._sp.unk_id()
        self._bos_token_id = self._sp.bos_id()
        self._eos_token_id = self._sp.eos_id()
        
        # BERT-style tokens (may need mapping)
        self._cls_token_id = self._bos_token_id  # Use BOS as CLS
        self._sep_token_id = self._eos_token_id  # Use EOS as SEP
        self._mask_token_id = self._sp.piece_to_id('[MASK]') if '[MASK]' in self._sp else self._unk_token_id
        
        # Update special tokens dictionary
        self._special_tokens = {
            'pad_token': self._sp.id_to_piece(self._pad_token_id),
            'unk_token': self._sp.id_to_piece(self._unk_token_id),
            'bos_token': self._sp.id_to_piece(self._bos_token_id),
            'eos_token': self._sp.id_to_piece(self._eos_token_id),
            'cls_token': self._sp.id_to_piece(self._cls_token_id),
            'sep_token': self._sp.id_to_piece(self._sep_token_id),
        }
    
    def _tokenize_single(
        self,
        text: str,
        max_length: Optional[int] = None,
        padding: bool = True,
        truncation: bool = True,
        return_attention_mask: bool = True,
        return_token_type_ids: bool = False,
    ) -> TokenSequence:
        """Tokenize a single text using SentencePiece.
        
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
        # Encode text to IDs
        if self.add_bos and self.add_eos:
            token_ids = self._sp.encode(text, add_bos=True, add_eos=True)
        else:
            token_ids = self._sp.encode(text, add_bos=self.add_bos, add_eos=self.add_eos)
        
        # Handle truncation
        if truncation and max_length and len(token_ids) > max_length:
            token_ids = self._truncate_sequence(token_ids, max_length)
        
        # Create attention mask
        attention_mask = [1] * len(token_ids) if return_attention_mask else None
        
        # Handle padding
        if padding and max_length:
            pad_length = max_length - len(token_ids)
            if pad_length > 0:
                token_ids = self._pad_sequence(token_ids, max_length, self._pad_token_id)
                if attention_mask:
                    attention_mask = attention_mask + [0] * pad_length
        
        # Token type IDs (all zeros for single sequence)
        token_type_ids = None
        if return_token_type_ids:
            token_type_ids = [0] * len(token_ids)
        
        return TokenSequence(
            input_ids=token_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=list(range(len(token_ids))),
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
        # Filter special tokens if requested
        if skip_special_tokens:
            special_ids = {self._pad_token_id, self._bos_token_id, self._eos_token_id}
            token_ids = [tid for tid in token_ids if tid not in special_ids]
        
        # Decode using SentencePiece
        text = self._sp.decode(token_ids)
        
        # Clean up tokenization spaces if requested
        if clean_up_tokenization_spaces:
            text = text.strip()
        
        return text
    
    def convert_tokens_to_ids(self, tokens: List[str]) -> List[int]:
        """Convert tokens to IDs.
        
        Args:
            tokens: List of tokens
            
        Returns:
            List of token IDs
        """
        return [self._sp.piece_to_id(token) for token in tokens]
    
    def convert_ids_to_tokens(self, ids: List[int]) -> List[str]:
        """Convert IDs to tokens.
        
        Args:
            ids: List of token IDs
            
        Returns:
            List of tokens
        """
        return [self._sp.id_to_piece(id) for id in ids]
    
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
        """Save SentencePiece tokenizer files.
        
        Args:
            path: Directory to save files
        """
        # Save SentencePiece model
        model_path = path / "spiece.model"
        with open(self.model_name_or_path, 'rb') as f_in:
            with open(model_path, 'wb') as f_out:
                f_out.write(f_in.read())
        
        # Save configuration
        config = {
            'model_file': 'spiece.model',
            'add_bos': self.add_bos,
            'add_eos': self.add_eos,
            'special_tokens': self._special_tokens,
            'special_token_ids': {
                'pad': self._pad_token_id,
                'unk': self._unk_token_id,
                'bos': self._bos_token_id,
                'eos': self._eos_token_id,
                'cls': self._cls_token_id,
                'sep': self._sep_token_id,
                'mask': self._mask_token_id,
            }
        }
        
        config_path = path / "sentencepiece_config.json"
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2)
    
    def _load_tokenizer_files(self, path: Path) -> None:
        """Load SentencePiece tokenizer files.
        
        Args:
            path: Directory to load files from
        """
        # Load configuration
        config_path = path / "sentencepiece_config.json"
        if config_path.exists():
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            
            self.add_bos = config.get('add_bos', True)
            self.add_eos = config.get('add_eos', True)
            
            # Load special token IDs
            if 'special_token_ids' in config:
                ids = config['special_token_ids']
                self._pad_token_id = ids.get('pad', 0)
                self._unk_token_id = ids.get('unk', 0)
                self._bos_token_id = ids.get('bos', 0)
                self._eos_token_id = ids.get('eos', 0)
                self._cls_token_id = ids.get('cls', 0)
                self._sep_token_id = ids.get('sep', 0)
                self._mask_token_id = ids.get('mask', 0)
        
        # Load SentencePiece model
        import sentencepiece as spm
        self._sp = spm.SentencePieceProcessor()
        
        model_path = path / config.get('model_file', 'spiece.model')
        if model_path.exists():
            self._sp.Load(str(model_path))
        else:
            raise FileNotFoundError(f"SentencePiece model not found: {model_path}")
        
        self._vocab_size = self._sp.vocab_size()
    
    def get_vocab(self) -> Dict[str, int]:
        """Get vocabulary mapping.
        
        Returns:
            Dictionary mapping tokens to IDs
        """
        vocab = {}
        for i in range(self._vocab_size):
            piece = self._sp.id_to_piece(i)
            vocab[piece] = i
        return vocab
    
    def tokenize_with_offsets(self, text: str) -> List[Tuple[str, int, int]]:
        """Tokenize text and return tokens with character offsets.
        
        Args:
            text: Text to tokenize
            
        Returns:
            List of (token, start_offset, end_offset) tuples
        """
        # SentencePiece doesn't directly support offsets
        # We'll approximate by reconstructing
        tokens = self._sp.encode_as_pieces(text)
        offsets = []
        current_pos = 0
        
        for token in tokens:
            # Remove SentencePiece prefix
            clean_token = token.replace('▁', ' ').lstrip()
            
            # Find token in text
            start = text.find(clean_token, current_pos)
            if start == -1:
                # Handle special tokens or subwords
                start = current_pos
                end = current_pos
            else:
                end = start + len(clean_token)
                current_pos = end
            
            offsets.append((token, start, end))
        
        return offsets