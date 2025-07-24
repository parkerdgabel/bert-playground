"""MLX embedding support for tokenizers."""

from typing import Optional, Dict, Any, Tuple, TYPE_CHECKING
from pathlib import Path

if TYPE_CHECKING:
    import numpy as np

try:
    import mlx.core as mx
    MLX_AVAILABLE = True
except ImportError:
    MLX_AVAILABLE = False
    mx = None


class MLXEmbeddingAdapter:
    """Adapter for MLX-optimized embeddings."""
    
    def __init__(self, vocab_size: int, embedding_dim: int = 768):
        """Initialize MLX embedding adapter.
        
        Args:
            vocab_size: Size of vocabulary
            embedding_dim: Dimension of embeddings
        """
        if not MLX_AVAILABLE:
            raise ImportError("MLX required for MLXEmbeddingAdapter")
        
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self._embeddings = None
        self._position_embeddings = None
        self._token_type_embeddings = None
    
    def initialize_embeddings(
        self,
        pretrained_embeddings: Optional[Any] = None,  # numpy array
        max_position_embeddings: int = 512,
        type_vocab_size: int = 2,
    ) -> None:
        """Initialize embedding tables.
        
        Args:
            pretrained_embeddings: Optional pretrained embeddings
            max_position_embeddings: Maximum position embeddings
            type_vocab_size: Size of token type vocabulary
        """
        # Initialize token embeddings
        if pretrained_embeddings is not None:
            self._embeddings = mx.array(pretrained_embeddings)
        else:
            # Random initialization
            self._embeddings = mx.random.normal(
                shape=(self.vocab_size, self.embedding_dim),
                scale=0.02
            )
        
        # Initialize position embeddings
        self._position_embeddings = mx.random.normal(
            shape=(max_position_embeddings, self.embedding_dim),
            scale=0.02
        )
        
        # Initialize token type embeddings
        self._token_type_embeddings = mx.random.normal(
            shape=(type_vocab_size, self.embedding_dim),
            scale=0.02
        )
    
    def embed_tokens(
        self,
        token_ids: Any,  # mx.array
        position_ids: Optional[Any] = None,  # mx.array
        token_type_ids: Optional[Any] = None,  # mx.array
    ) -> Any:  # mx.array
        """Embed tokens with position and type embeddings.
        
        Args:
            token_ids: Token IDs to embed
            position_ids: Position IDs
            token_type_ids: Token type IDs
            
        Returns:
            Embedded tokens
        """
        if self._embeddings is None:
            raise RuntimeError("Embeddings not initialized")
        
        # Get token embeddings
        embeddings = self._embeddings[token_ids]
        
        # Add position embeddings if provided
        if position_ids is not None and self._position_embeddings is not None:
            embeddings = embeddings + self._position_embeddings[position_ids]
        
        # Add token type embeddings if provided
        if token_type_ids is not None and self._token_type_embeddings is not None:
            embeddings = embeddings + self._token_type_embeddings[token_type_ids]
        
        return embeddings
    
    def save_embeddings(self, path: str) -> None:
        """Save embeddings to disk.
        
        Args:
            path: Directory to save embeddings
        """
        save_path = Path(path)
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Save token embeddings
        if self._embeddings is not None:
            mx.save(str(save_path / "token_embeddings.npz"), {"embeddings": self._embeddings})
        
        # Save position embeddings
        if self._position_embeddings is not None:
            mx.save(str(save_path / "position_embeddings.npz"), {"embeddings": self._position_embeddings})
        
        # Save token type embeddings
        if self._token_type_embeddings is not None:
            mx.save(str(save_path / "token_type_embeddings.npz"), {"embeddings": self._token_type_embeddings})
        
        # Save configuration
        import json
        config = {
            "vocab_size": self.vocab_size,
            "embedding_dim": self.embedding_dim,
            "has_position_embeddings": self._position_embeddings is not None,
            "has_token_type_embeddings": self._token_type_embeddings is not None,
        }
        
        with open(save_path / "embedding_config.json", 'w') as f:
            json.dump(config, f, indent=2)
    
    def load_embeddings(self, path: str) -> None:
        """Load embeddings from disk.
        
        Args:
            path: Directory to load embeddings from
        """
        load_path = Path(path)
        
        # Load configuration
        import json
        with open(load_path / "embedding_config.json", 'r') as f:
            config = json.load(f)
        
        self.vocab_size = config["vocab_size"]
        self.embedding_dim = config["embedding_dim"]
        
        # Load token embeddings
        token_path = load_path / "token_embeddings.npz"
        if token_path.exists():
            data = mx.load(str(token_path))
            self._embeddings = data["embeddings"]
        
        # Load position embeddings
        if config.get("has_position_embeddings", False):
            pos_path = load_path / "position_embeddings.npz"
            if pos_path.exists():
                data = mx.load(str(pos_path))
                self._position_embeddings = data["embeddings"]
        
        # Load token type embeddings
        if config.get("has_token_type_embeddings", False):
            type_path = load_path / "token_type_embeddings.npz"
            if type_path.exists():
                data = mx.load(str(type_path))
                self._token_type_embeddings = data["embeddings"]
    
    def get_embedding_stats(self) -> Dict[str, Any]:
        """Get statistics about embeddings.
        
        Returns:
            Dictionary of embedding statistics
        """
        stats = {
            "vocab_size": self.vocab_size,
            "embedding_dim": self.embedding_dim,
        }
        
        if self._embeddings is not None:
            stats["token_embeddings"] = {
                "shape": self._embeddings.shape,
                "mean": float(mx.mean(self._embeddings)),
                "std": float(mx.std(self._embeddings)),
                "min": float(mx.min(self._embeddings)),
                "max": float(mx.max(self._embeddings)),
            }
        
        if self._position_embeddings is not None:
            stats["position_embeddings"] = {
                "shape": self._position_embeddings.shape,
                "mean": float(mx.mean(self._position_embeddings)),
                "std": float(mx.std(self._position_embeddings)),
            }
        
        if self._token_type_embeddings is not None:
            stats["token_type_embeddings"] = {
                "shape": self._token_type_embeddings.shape,
                "mean": float(mx.mean(self._token_type_embeddings)),
                "std": float(mx.std(self._token_type_embeddings)),
            }
        
        return stats
    
    def optimize_embeddings(self) -> None:
        """Optimize embeddings for MLX performance."""
        # This could include:
        # - Quantization for memory efficiency
        # - Precomputation of common token combinations
        # - Memory layout optimization
        pass
    
    def compute_similarity(
        self,
        token_ids1: Any,  # mx.array
        token_ids2: Any,  # mx.array
        metric: str = "cosine"
    ) -> Any:  # mx.array
        """Compute similarity between token embeddings.
        
        Args:
            token_ids1: First set of token IDs
            token_ids2: Second set of token IDs
            metric: Similarity metric ('cosine', 'euclidean')
            
        Returns:
            Similarity scores
        """
        if self._embeddings is None:
            raise RuntimeError("Embeddings not initialized")
        
        emb1 = self._embeddings[token_ids1]
        emb2 = self._embeddings[token_ids2]
        
        if metric == "cosine":
            # Normalize embeddings
            emb1_norm = emb1 / mx.linalg.norm(emb1, axis=-1, keepdims=True)
            emb2_norm = emb2 / mx.linalg.norm(emb2, axis=-1, keepdims=True)
            
            # Compute cosine similarity
            return mx.sum(emb1_norm * emb2_norm, axis=-1)
        
        elif metric == "euclidean":
            # Compute euclidean distance
            return -mx.linalg.norm(emb1 - emb2, axis=-1)
        
        else:
            raise ValueError(f"Unknown metric: {metric}")