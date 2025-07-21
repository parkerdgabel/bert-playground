"""
BERT confidence scoring utilities.

This module provides confidence scoring mechanisms for BERT predictions,
using attention patterns and prediction characteristics.
"""

from dataclasses import dataclass

import mlx.core as mx


@dataclass
class ConfidenceScoringConfig:
    """Configuration for confidence scoring."""

    # Attention-based confidence
    use_attention_confidence: bool = True
    attention_entropy_threshold: float = 0.3  # Low entropy = high confidence

    # Ensemble uncertainty
    use_ensemble_uncertainty: bool = True
    max_ensemble_std: float = 0.1  # Max std dev for confident prediction


class BERTConfidenceScorer:
    """
    Calculate confidence scores for BERT predictions.

    Uses multiple signals:
    - Prediction probability
    - Attention entropy
    - Prediction consistency
    - Token-level confidence
    """

    def __init__(self, use_attention_confidence: bool = True):
        """
        Initialize confidence scorer.

        Args:
            use_attention_confidence: Whether to use attention patterns
        """
        self.use_attention_confidence = use_attention_confidence

    def compute_confidence(
        self,
        logits: mx.array,
        attention_weights: mx.array | None = None,
        token_logits: mx.array | None = None,
    ) -> mx.array:
        """
        Compute confidence scores for predictions.

        Args:
            logits: Model output logits [batch_size, num_classes]
            attention_weights: Attention weights [batch_size, num_heads, seq_len, seq_len]
            token_logits: Token-level predictions for sequence tasks

        Returns:
            Confidence scores [batch_size]
        """
        confidences = []

        # 1. Prediction probability confidence
        probs = mx.softmax(logits, axis=-1)
        max_probs = mx.max(probs, axis=-1)
        prob_confidence = max_probs
        confidences.append(prob_confidence)

        # 2. Entropy-based confidence
        entropy = -mx.sum(probs * mx.log(probs + 1e-10), axis=-1)
        # Normalize entropy to 0-1 (lower is more confident)
        max_entropy = mx.log(mx.array(logits.shape[-1]))
        entropy_confidence = 1.0 - (entropy / max_entropy)
        confidences.append(entropy_confidence)

        # 3. Attention-based confidence
        if attention_weights is not None and self.use_attention_confidence:
            attention_confidence = self._compute_attention_confidence(attention_weights)
            confidences.append(attention_confidence)

        # 4. Token-level confidence (for sequence labeling)
        if token_logits is not None:
            token_confidence = self._compute_token_confidence(token_logits)
            confidences.append(token_confidence)

        # Combine confidences
        if len(confidences) > 1:
            # Weighted average
            weights = mx.array(
                [0.4, 0.3] + [0.3 / (len(confidences) - 2)] * (len(confidences) - 2)
            )
            confidence = mx.sum(
                mx.stack(confidences) * weights[: len(confidences), None], axis=0
            )
        else:
            confidence = confidences[0]

        return confidence

    def _compute_attention_confidence(self, attention_weights: mx.array) -> mx.array:
        """
        Compute confidence from attention patterns.

        Low entropy in attention = model is focused = high confidence
        """
        # Average over heads and layers
        # Shape: [batch_size, num_heads, seq_len, seq_len]
        avg_attention = mx.mean(attention_weights, axis=1)  # Average over heads

        # Compute attention entropy for each position
        attention_entropy = -mx.sum(
            avg_attention * mx.log(avg_attention + 1e-10), axis=-1
        )  # [batch_size, seq_len]

        # Average entropy across sequence
        avg_entropy = mx.mean(attention_entropy, axis=-1)  # [batch_size]

        # Normalize and invert (low entropy = high confidence)
        max_possible_entropy = mx.log(mx.array(attention_weights.shape[-1]))
        confidence = 1.0 - (avg_entropy / max_possible_entropy)

        return confidence

    def _compute_token_confidence(self, token_logits: mx.array) -> mx.array:
        """
        Compute confidence from token-level predictions.

        High agreement across tokens = high confidence
        """
        # Get token probabilities
        token_probs = mx.softmax(token_logits, axis=-1)

        # Compute variance across tokens
        token_max_probs = mx.max(token_probs, axis=-1)  # [batch_size, seq_len]

        # Confidence is mean of token confidences
        confidence = mx.mean(token_max_probs, axis=-1)

        return confidence
