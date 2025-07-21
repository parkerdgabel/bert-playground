"""Tests for test-time augmentation (TTA)."""

import mlx.core as mx
import pytest

from data.augmentation.text import BERTTextAugmenter
from data.augmentation.tta import BERTTestTimeAugmentation


class MockTokenizer:
    """Mock tokenizer for testing."""
    
    def __init__(self):
        self.vocab_size = 30000
        self.mask_token = "[MASK]"
        self.mask_token_id = 103
        self.pad_token_id = 0
        self.cls_token_id = 101
        self.sep_token_id = 102
    
    def tokenize(self, text):
        return text.split()
    
    def convert_tokens_to_string(self, tokens):
        return " ".join(tokens)
    
    def convert_ids_to_tokens(self, ids):
        return [f"token_{id}" for id in ids]


class MockModel:
    """Mock model for testing TTA predictions."""
    
    def __init__(self):
        self.training = True
    
    def eval(self):
        """Set model to evaluation mode."""
        self.training = False
    
    def __call__(self, batch):
        """Mock forward pass."""
        # Return mock logits
        batch_size = 1
        num_classes = 2
        logits = mx.random.normal((batch_size, num_classes))
        return {"logits": logits}


class TestBERTTestTimeAugmentation:
    """Test BERTTestTimeAugmentation class."""

    def test_initialization(self):
        """Test TTA initialization."""
        tokenizer = MockTokenizer()
        text_augmenter = BERTTextAugmenter(tokenizer)
        
        tta = BERTTestTimeAugmentation(
            augmenter=text_augmenter,
            num_augmentations=5
        )
        
        assert tta.augmenter == text_augmenter
        assert tta.num_augmentations == 5

    def test_augment_single_text(self):
        """Test augmenting a single text for TTA."""
        tokenizer = MockTokenizer()
        text_augmenter = BERTTextAugmenter(tokenizer, mask_probability=0.2)
        tta = BERTTestTimeAugmentation(text_augmenter, num_augmentations=3)
        
        text = "The quick brown fox jumps"
        augmented_texts = tta.augment(text)
        
        assert len(augmented_texts) == 3
        assert all(isinstance(t, str) for t in augmented_texts)

    def test_augment_batch(self):
        """Test augmenting a batch of texts."""
        tokenizer = MockTokenizer()
        text_augmenter = BERTTextAugmenter(tokenizer, mask_probability=0.2)
        tta = BERTTestTimeAugmentation(text_augmenter, num_augmentations=3)
        
        texts = [
            "First sample text",
            "Second sample text",
            "Third sample text",
        ]
        
        augmented_batch = tta.augment_batch(texts)
        
        assert len(augmented_batch) == len(texts)
        for i, augmented_list in enumerate(augmented_batch):
            assert len(augmented_list) == 3
            assert all(isinstance(t, str) for t in augmented_list)

    def test_combine_predictions_mean(self):
        """Test combining predictions with mean method."""
        tokenizer = MockTokenizer()
        text_augmenter = BERTTextAugmenter(tokenizer)
        tta = BERTTestTimeAugmentation(text_augmenter)
        
        # Create mock predictions (3 augmentations, 2 samples, 4 classes)
        pred1 = mx.array([[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8]])
        pred2 = mx.array([[0.2, 0.3, 0.4, 0.5], [0.6, 0.7, 0.8, 0.9]])
        pred3 = mx.array([[0.3, 0.4, 0.5, 0.6], [0.7, 0.8, 0.9, 1.0]])
        
        predictions = [pred1, pred2, pred3]
        combined = tta.combine_predictions(predictions, method="mean")
        
        assert combined.shape == (2, 4)
        # Check mean calculation
        expected_mean = mx.mean(mx.stack(predictions, axis=0), axis=0)
        assert mx.allclose(combined, expected_mean)

    def test_combine_predictions_max(self):
        """Test combining predictions with max method."""
        tokenizer = MockTokenizer()
        text_augmenter = BERTTextAugmenter(tokenizer)
        tta = BERTTestTimeAugmentation(text_augmenter)
        
        pred1 = mx.array([[0.1, 0.9], [0.3, 0.7]])
        pred2 = mx.array([[0.8, 0.2], [0.4, 0.6]])
        pred3 = mx.array([[0.5, 0.5], [0.9, 0.1]])
        
        predictions = [pred1, pred2, pred3]
        combined = tta.combine_predictions(predictions, method="max")
        
        assert combined.shape == (2, 2)
        # Check max calculation
        expected_max = mx.max(mx.stack(predictions, axis=0), axis=0)
        assert mx.allclose(combined, expected_max)

    def test_combine_predictions_vote(self):
        """Test combining predictions with vote method."""
        tokenizer = MockTokenizer()
        text_augmenter = BERTTextAugmenter(tokenizer)
        tta = BERTTestTimeAugmentation(text_augmenter)
        
        # Create predictions where class votes are clear
        pred1 = mx.array([[0.1, 0.9], [0.8, 0.2]])  # votes: [1, 0]
        pred2 = mx.array([[0.2, 0.8], [0.7, 0.3]])  # votes: [1, 0]
        pred3 = mx.array([[0.9, 0.1], [0.1, 0.9]])  # votes: [0, 1]
        
        predictions = [pred1, pred2, pred3]
        combined = tta.combine_predictions(predictions, method="vote")
        
        assert combined.shape == (2,)
        # First sample: 2 votes for class 1, 1 vote for class 0 -> class 1
        assert combined[0].item() == 1
        # Second sample: 2 votes for class 0, 1 vote for class 1 -> class 0
        assert combined[1].item() == 0

    def test_combine_predictions_invalid_method(self):
        """Test error with invalid combination method."""
        tokenizer = MockTokenizer()
        text_augmenter = BERTTextAugmenter(tokenizer)
        tta = BERTTestTimeAugmentation(text_augmenter)
        
        predictions = [mx.array([[0.5, 0.5]])]
        
        with pytest.raises(ValueError, match="Unknown combination method"):
            tta.combine_predictions(predictions, method="invalid")

    def test_predict_with_tta(self):
        """Test full TTA prediction pipeline."""
        tokenizer = MockTokenizer()
        text_augmenter = BERTTextAugmenter(tokenizer, mask_probability=0.1)
        tta = BERTTestTimeAugmentation(text_augmenter, num_augmentations=3)
        
        # Mock model and dataloader
        model = MockModel()
        
        class MockDataLoader:
            def __iter__(self):
                # Yield 2 batches
                yield {"input_ids": mx.array([[1, 2, 3]])}
                yield {"input_ids": mx.array([[4, 5, 6]])}
        
        dataloader = MockDataLoader()
        
        # Run TTA prediction
        predictions = tta.predict_with_tta(model, dataloader, num_augmentations=3)
        
        assert isinstance(predictions, mx.array)
        # Should have predictions for 2 batches
        assert predictions.shape[0] == 2

    def test_predict_with_tta_custom_augmentations(self):
        """Test TTA with custom number of augmentations."""
        tokenizer = MockTokenizer()
        text_augmenter = BERTTextAugmenter(tokenizer)
        tta = BERTTestTimeAugmentation(text_augmenter, num_augmentations=5)
        
        model = MockModel()
        
        class SingleBatchDataLoader:
            def __iter__(self):
                yield {"input_ids": mx.array([[1, 2, 3]])}
        
        dataloader = SingleBatchDataLoader()
        
        # Override default augmentations
        predictions = tta.predict_with_tta(model, dataloader, num_augmentations=2)
        
        assert isinstance(predictions, mx.array)

    def test_deterministic_augmentation(self):
        """Test that augmentation is deterministic with seed."""
        tokenizer = MockTokenizer()
        text_augmenter1 = BERTTextAugmenter(tokenizer, seed=42)
        text_augmenter2 = BERTTextAugmenter(tokenizer, seed=42)
        
        tta1 = BERTTestTimeAugmentation(text_augmenter1, num_augmentations=3)
        tta2 = BERTTestTimeAugmentation(text_augmenter2, num_augmentations=3)
        
        text = "Test text for augmentation"
        
        aug1 = tta1.augment(text)
        aug2 = tta2.augment(text)
        
        # Should produce same augmentations with same seed
        assert aug1 == aug2