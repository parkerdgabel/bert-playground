"""Tests for BERT text augmentation."""

import mlx.core as mx
import pytest

from data.augmentation.text import BERTTextAugmenter


class MockTokenizer:
    """Mock tokenizer for testing."""
    
    def __init__(self):
        self.vocab_size = 30000
        self.mask_token = "[MASK]"
        self.mask_token_id = 103
        self.pad_token_id = 0
        self.cls_token_id = 101
        self.sep_token_id = 102
        self.special_tokens = {"[CLS]", "[SEP]", "[MASK]", "[PAD]"}
        
        # Simple vocab for testing
        self.vocab = {
            "the": 100, "quick": 200, "brown": 300, "fox": 400,
            "jumps": 500, "over": 600, "lazy": 700, "dog": 800,
            "cat": 900, "runs": 1000, "fast": 1100, "slow": 1200,
        }
        self.id_to_token = {v: k for k, v in self.vocab.items()}
        self.id_to_token.update({
            0: "[PAD]", 101: "[CLS]", 102: "[SEP]", 103: "[MASK]"
        })
    
    def tokenize(self, text):
        """Simple tokenization."""
        return text.lower().split()
    
    def convert_tokens_to_string(self, tokens):
        """Convert tokens back to string."""
        return " ".join(tokens)
    
    def convert_ids_to_tokens(self, ids):
        """Convert IDs to tokens."""
        if hasattr(ids, "tolist"):
            ids = ids.tolist()
        return [self.id_to_token.get(id, f"[UNK{id}]") for id in ids]
    
    def encode(self, text, add_special_tokens=True):
        """Encode text to IDs."""
        tokens = self.tokenize(text)
        ids = [self.vocab.get(token, 999) for token in tokens]
        if add_special_tokens:
            ids = [self.cls_token_id] + ids + [self.sep_token_id]
        return ids


class TestBERTTextAugmenter:
    """Test BERTTextAugmenter class."""

    def test_initialization(self):
        """Test augmenter initialization."""
        tokenizer = MockTokenizer()
        augmenter = BERTTextAugmenter(
            tokenizer=tokenizer,
            mask_probability=0.15,
            random_probability=0.1,
            seed=42
        )
        
        assert augmenter.tokenizer == tokenizer
        assert augmenter.mask_probability == 0.15
        assert augmenter.random_probability == 0.1
        assert augmenter.mask_token == "[MASK]"

    def test_augment_single_text(self):
        """Test augmenting a single text."""
        tokenizer = MockTokenizer()
        augmenter = BERTTextAugmenter(tokenizer, mask_probability=0.3, seed=42)
        
        original = "The quick brown fox"
        augmented = augmenter.augment(original)
        
        assert isinstance(augmented, str)
        assert len(augmented) > 0
        # Should contain some original words or [MASK]
        assert any(word in augmented for word in ["the", "quick", "brown", "fox", "[MASK]"])

    def test_augment_text_multiple(self):
        """Test generating multiple augmented versions."""
        tokenizer = MockTokenizer()
        augmenter = BERTTextAugmenter(tokenizer, mask_probability=0.2, seed=42)
        
        original = "The quick brown fox jumps"
        augmented_texts = augmenter.augment_text(original, num_augmentations=5)
        
        assert len(augmented_texts) == 5
        assert all(isinstance(text, str) for text in augmented_texts)
        # At least some should be different
        assert len(set(augmented_texts)) > 1

    def test_mask_tokens(self):
        """Test token masking."""
        tokenizer = MockTokenizer()
        augmenter = BERTTextAugmenter(tokenizer, seed=42)
        
        # Create token IDs
        tokens = mx.array([100, 200, 300, 400, 500])  # the quick brown fox jumps
        mask = mx.array([False, True, False, True, False])  # Mask positions 1 and 3
        
        masked_tokens = augmenter._mask_tokens(tokens, mask)
        
        # Check that masked positions have mask token ID
        assert masked_tokens[1].item() == tokenizer.mask_token_id
        assert masked_tokens[3].item() == tokenizer.mask_token_id
        # Unmasked positions should be unchanged
        assert masked_tokens[0].item() == 100
        assert masked_tokens[2].item() == 300
        assert masked_tokens[4].item() == 500

    def test_random_tokens(self):
        """Test random token replacement."""
        tokenizer = MockTokenizer()
        augmenter = BERTTextAugmenter(tokenizer, seed=42)
        
        tokens = mx.array([100, 200, 300, 400, 500])
        mask = mx.array([False, True, False, True, False])
        
        random_tokens = augmenter._random_tokens(tokens, mask)
        
        # Masked positions should have different tokens
        assert random_tokens[1].item() != 200
        assert random_tokens[3].item() != 400
        # Unmasked positions should be unchanged
        assert random_tokens[0].item() == 100
        assert random_tokens[2].item() == 300
        assert random_tokens[4].item() == 500

    def test_no_augmentation(self):
        """Test with zero mask probability."""
        tokenizer = MockTokenizer()
        augmenter = BERTTextAugmenter(tokenizer, mask_probability=0.0)
        
        original = "The quick brown fox"
        augmented = augmenter.augment(original)
        
        # Should be unchanged (except for potential tokenization artifacts)
        assert "[MASK]" not in augmented

    def test_special_tokens_not_masked(self):
        """Test that special tokens are not masked."""
        tokenizer = MockTokenizer()
        augmenter = BERTTextAugmenter(tokenizer, mask_probability=1.0, seed=42)
        
        # Text with special tokens
        tokens = ["[CLS]", "the", "quick", "[SEP]"]
        augmented_text = augmenter._augment_tokens(tokens)
        
        # Check that special tokens are preserved
        augmented_tokens = augmented_text.split()
        assert augmented_tokens[0] == "[CLS]"
        assert augmented_tokens[-1] == "[SEP]"

    def test_augment_batch(self):
        """Test batch augmentation."""
        tokenizer = MockTokenizer()
        augmenter = BERTTextAugmenter(tokenizer, mask_probability=0.2, seed=42)
        
        texts = [
            "The quick brown fox",
            "The lazy dog sleeps",
            "A cat runs fast"
        ]
        
        augmented_batch = augmenter.augment_batch(texts)
        
        assert len(augmented_batch) == len(texts)
        assert all(isinstance(text, str) for text in augmented_batch)

    def test_deterministic_with_seed(self):
        """Test that augmentation is deterministic with seed."""
        tokenizer = MockTokenizer()
        
        augmenter1 = BERTTextAugmenter(tokenizer, mask_probability=0.3, seed=42)
        augmenter2 = BERTTextAugmenter(tokenizer, mask_probability=0.3, seed=42)
        
        text = "The quick brown fox jumps over the lazy dog"
        
        result1 = augmenter1.augment(text)
        result2 = augmenter2.augment(text)
        
        assert result1 == result2

    def test_mlx_operations(self):
        """Test that all operations use MLX arrays."""
        tokenizer = MockTokenizer()
        augmenter = BERTTextAugmenter(tokenizer, seed=42)
        
        # Test mask generation
        mask = augmenter._generate_mask(10)
        assert isinstance(mask, mx.array)
        assert mask.dtype == mx.bool_
        
        # Test token operations
        tokens = mx.array([100, 200, 300])
        mask = mx.array([False, True, False])
        
        masked = augmenter._mask_tokens(tokens, mask)
        assert isinstance(masked, mx.array)
        
        random = augmenter._random_tokens(tokens, mask)
        assert isinstance(random, mx.array)