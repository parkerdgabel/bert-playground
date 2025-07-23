"""Tokenizer caching functionality for HuggingFace tokenizers."""

import hashlib
import json
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import time

from domain.entities.dataset import TokenSequence


class TokenizerCache:
    """Cache for tokenized sequences to speed up data loading."""
    
    def __init__(self, cache_dir: Optional[str] = None, ttl_seconds: Optional[int] = None):
        """Initialize tokenizer cache.
        
        Args:
            cache_dir: Directory for cache storage
            ttl_seconds: Time-to-live for cache entries in seconds
        """
        self.cache_dir = Path(cache_dir) if cache_dir else Path.home() / '.cache' / 'k-bert' / 'tokenizer'
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.ttl_seconds = ttl_seconds
        
        # In-memory cache for faster access
        self._memory_cache: Dict[str, Tuple[Any, float]] = {}
        self._max_memory_items = 1000
    
    def _get_cache_key(
        self,
        text: str,
        tokenizer_name: str,
        max_length: Optional[int] = None,
        **kwargs
    ) -> str:
        """Generate cache key for tokenization parameters.
        
        Args:
            text: Input text
            tokenizer_name: Tokenizer identifier
            max_length: Maximum sequence length
            **kwargs: Additional tokenization parameters
            
        Returns:
            Cache key string
        """
        # Create a unique key based on all parameters
        key_parts = [
            text,
            tokenizer_name,
            str(max_length),
            json.dumps(kwargs, sort_keys=True)
        ]
        
        key_string = '|'.join(key_parts)
        
        # Hash for consistent length and filesystem safety
        return hashlib.sha256(key_string.encode()).hexdigest()
    
    def get(
        self,
        text: str,
        tokenizer_name: str,
        max_length: Optional[int] = None,
        **kwargs
    ) -> Optional[TokenSequence]:
        """Get cached tokenization result.
        
        Args:
            text: Input text
            tokenizer_name: Tokenizer identifier
            max_length: Maximum sequence length
            **kwargs: Additional tokenization parameters
            
        Returns:
            Cached TokenSequence or None if not found
        """
        cache_key = self._get_cache_key(text, tokenizer_name, max_length, **kwargs)
        
        # Check memory cache first
        if cache_key in self._memory_cache:
            cached_data, timestamp = self._memory_cache[cache_key]
            if self._is_valid(timestamp):
                return cached_data
            else:
                del self._memory_cache[cache_key]
        
        # Check disk cache
        cache_path = self.cache_dir / f"{cache_key}.pkl"
        
        if cache_path.exists():
            try:
                with open(cache_path, 'rb') as f:
                    cached_data = pickle.load(f)
                
                if self._is_valid(cached_data.get('timestamp', 0)):
                    token_sequence = self._dict_to_token_sequence(cached_data['data'])
                    
                    # Add to memory cache
                    self._add_to_memory_cache(cache_key, token_sequence, cached_data['timestamp'])
                    
                    return token_sequence
                else:
                    # Remove expired cache
                    cache_path.unlink()
            except Exception:
                # Remove corrupted cache
                cache_path.unlink()
        
        return None
    
    def set(
        self,
        text: str,
        tokenizer_name: str,
        token_sequence: TokenSequence,
        max_length: Optional[int] = None,
        **kwargs
    ) -> None:
        """Cache tokenization result.
        
        Args:
            text: Input text
            tokenizer_name: Tokenizer identifier
            token_sequence: Tokenization result
            max_length: Maximum sequence length
            **kwargs: Additional tokenization parameters
        """
        cache_key = self._get_cache_key(text, tokenizer_name, max_length, **kwargs)
        timestamp = time.time()
        
        # Add to memory cache
        self._add_to_memory_cache(cache_key, token_sequence, timestamp)
        
        # Save to disk cache
        cache_path = self.cache_dir / f"{cache_key}.pkl"
        
        cache_data = {
            'data': self._token_sequence_to_dict(token_sequence),
            'timestamp': timestamp,
            'text': text[:100],  # Store first 100 chars for debugging
            'tokenizer': tokenizer_name,
            'max_length': max_length,
            'kwargs': kwargs,
        }
        
        with open(cache_path, 'wb') as f:
            pickle.dump(cache_data, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    def clear(self, older_than_seconds: Optional[int] = None) -> int:
        """Clear cache entries.
        
        Args:
            older_than_seconds: Clear only entries older than this many seconds
            
        Returns:
            Number of entries cleared
        """
        cleared = 0
        current_time = time.time()
        
        # Clear memory cache
        if older_than_seconds:
            to_remove = []
            for key, (_, timestamp) in self._memory_cache.items():
                if current_time - timestamp > older_than_seconds:
                    to_remove.append(key)
            
            for key in to_remove:
                del self._memory_cache[key]
                cleared += 1
        else:
            cleared += len(self._memory_cache)
            self._memory_cache.clear()
        
        # Clear disk cache
        for cache_file in self.cache_dir.glob("*.pkl"):
            if older_than_seconds:
                # Check file modification time
                mtime = cache_file.stat().st_mtime
                if current_time - mtime > older_than_seconds:
                    cache_file.unlink()
                    cleared += 1
            else:
                cache_file.unlink()
                cleared += 1
        
        return cleared
    
    def get_cache_info(self) -> Dict[str, Any]:
        """Get cache statistics.
        
        Returns:
            Dictionary with cache information
        """
        cache_files = list(self.cache_dir.glob("*.pkl"))
        total_size = sum(f.stat().st_size for f in cache_files)
        
        return {
            'cache_dir': str(self.cache_dir),
            'memory_entries': len(self._memory_cache),
            'disk_entries': len(cache_files),
            'total_size_bytes': total_size,
            'total_size_mb': total_size / (1024 * 1024),
            'ttl_seconds': self.ttl_seconds,
        }
    
    def _is_valid(self, timestamp: float) -> bool:
        """Check if cache entry is still valid.
        
        Args:
            timestamp: Entry timestamp
            
        Returns:
            True if valid
        """
        if self.ttl_seconds is None:
            return True
        
        return time.time() - timestamp < self.ttl_seconds
    
    def _add_to_memory_cache(
        self,
        key: str,
        data: TokenSequence,
        timestamp: float
    ) -> None:
        """Add entry to memory cache with LRU eviction.
        
        Args:
            key: Cache key
            data: Data to cache
            timestamp: Entry timestamp
        """
        # Evict oldest entries if cache is full
        if len(self._memory_cache) >= self._max_memory_items:
            # Remove 10% of oldest entries
            items_to_remove = int(self._max_memory_items * 0.1)
            sorted_keys = sorted(
                self._memory_cache.keys(),
                key=lambda k: self._memory_cache[k][1]
            )
            
            for k in sorted_keys[:items_to_remove]:
                del self._memory_cache[k]
        
        self._memory_cache[key] = (data, timestamp)
    
    def _token_sequence_to_dict(self, seq: TokenSequence) -> Dict[str, Any]:
        """Convert TokenSequence to dictionary for serialization.
        
        Args:
            seq: TokenSequence to convert
            
        Returns:
            Dictionary representation
        """
        return {
            'input_ids': seq.input_ids,
            'attention_mask': seq.attention_mask,
            'token_type_ids': seq.token_type_ids,
            'position_ids': seq.position_ids,
        }
    
    def _dict_to_token_sequence(self, data: Dict[str, Any]) -> TokenSequence:
        """Convert dictionary to TokenSequence.
        
        Args:
            data: Dictionary representation
            
        Returns:
            TokenSequence instance
        """
        return TokenSequence(
            input_ids=data['input_ids'],
            attention_mask=data['attention_mask'],
            token_type_ids=data.get('token_type_ids'),
            position_ids=data.get('position_ids'),
        )