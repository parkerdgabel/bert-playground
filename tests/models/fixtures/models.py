"""Model fixtures for testing."""

from typing import Dict, Optional, Any, Tuple
import mlx.core as mx
import mlx.nn as nn

from models.bert.config import BertConfig
from models.bert.modernbert_config import ModernBertConfig
from models.heads.config import ClassificationConfig, RegressionConfig
from tests.models.fixtures.configs import (
    create_small_bert_config,
    create_tiny_bert_config,
    create_classification_config,
    create_regression_config,
)


class SimpleBertModel(nn.Module):
    """Simplified BERT model for testing."""
    
    def __init__(self, config: BertConfig):
        super().__init__()
        self.config = config
        
        # Embeddings
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        
        # Encoder layers (simplified)
        self.layers = []
        for _ in range(config.num_hidden_layers):
            self.layers.append(nn.Linear(config.hidden_size, config.hidden_size))
        
        # Pooler
        self.pooler = nn.Linear(config.hidden_size, config.hidden_size)
    
    def __call__(
        self,
        input_ids: mx.array,
        attention_mask: Optional[mx.array] = None,
        token_type_ids: Optional[mx.array] = None,
        position_ids: Optional[mx.array] = None,
    ) -> Dict[str, mx.array]:
        """Forward pass."""
        batch_size, seq_length = input_ids.shape
        
        # Create position IDs if not provided
        if position_ids is None:
            position_ids = mx.arange(seq_length)[None, :].astype(mx.int32)
            position_ids = mx.broadcast_to(position_ids, (batch_size, seq_length))
        
        # Create token type IDs if not provided
        if token_type_ids is None:
            token_type_ids = mx.zeros((batch_size, seq_length), dtype=mx.int32)
        
        # Embeddings
        word_embeds = self.word_embeddings(input_ids)
        position_embeds = self.position_embeddings(position_ids)
        token_type_embeds = self.token_type_embeddings(token_type_ids)
        
        embeddings = word_embeds + position_embeds + token_type_embeds
        embeddings = self.layer_norm(embeddings)
        embeddings = self.dropout(embeddings)
        
        # Pass through layers
        hidden_states = embeddings
        for layer in self.layers:
            hidden_states = layer(hidden_states)
            hidden_states = nn.gelu(hidden_states)
        
        # Pool
        pooled_output = self.pooler(hidden_states[:, 0])
        pooled_output = nn.tanh(pooled_output)
        
        return {
            "last_hidden_state": hidden_states,
            "pooler_output": pooled_output,
        }


class SimpleBinaryClassifier(nn.Module):
    """Simple binary classification model for testing."""
    
    def __init__(self, input_dim: int = 768, hidden_dim: int = 128):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
        )
        self.classifier = nn.Linear(hidden_dim, 2)
    
    def __call__(self, x: mx.array) -> mx.array:
        """Forward pass."""
        hidden = self.encoder(x)
        logits = self.classifier(hidden)
        return logits
    
    def loss(self, x: mx.array, labels: mx.array) -> mx.array:
        """Compute loss."""
        logits = self(x)
        return mx.mean(
            nn.losses.cross_entropy(logits, labels, reduction="none")
        )


class SimpleMulticlassClassifier(nn.Module):
    """Simple multiclass classification model for testing."""
    
    def __init__(self, input_dim: int = 768, num_classes: int = 10):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
        )
        self.classifier = nn.Linear(128, num_classes)
    
    def __call__(self, x: mx.array) -> mx.array:
        """Forward pass."""
        hidden = self.encoder(x)
        logits = self.classifier(hidden)
        return logits
    
    def loss(self, x: mx.array, labels: mx.array) -> mx.array:
        """Compute loss."""
        logits = self(x)
        return mx.mean(
            nn.losses.cross_entropy(logits, labels, reduction="none")
        )


class SimpleRegressor(nn.Module):
    """Simple regression model for testing."""
    
    def __init__(self, input_dim: int = 768, output_dim: int = 1):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.ReLU(),
        )
        self.output = nn.Linear(64, output_dim)
    
    def __call__(self, x: mx.array) -> mx.array:
        """Forward pass."""
        hidden = self.encoder(x)
        output = self.output(hidden)
        return output
    
    def loss(self, x: mx.array, targets: mx.array) -> mx.array:
        """Compute MSE loss."""
        predictions = self(x)
        return mx.mean((predictions - targets) ** 2)


class BrokenModel(nn.Module):
    """Model that raises errors for testing error handling."""
    
    def __init__(self, error_type: str = "forward", config: Optional[BertConfig] = None):
        super().__init__()
        self.error_type = error_type
        self.config = config or create_tiny_bert_config()
        self.linear = nn.Linear(10, 10)
    
    def __call__(self, *args, **kwargs):
        """Forward pass that raises error."""
        if self.error_type == "forward":
            raise RuntimeError("Forward pass failed")
        elif self.error_type == "shape":
            # Return wrong shape
            return mx.zeros((1, 1, 1))
        elif self.error_type == "type":
            # Return wrong type
            return "not an array"
        return self.linear(mx.zeros((1, 10)))


class NaNModel(nn.Module):
    """Model that produces NaN values for testing."""
    
    def __init__(self, config: Optional[BertConfig] = None):
        super().__init__()
        self.config = config or create_tiny_bert_config()
        self.linear = nn.Linear(config.hidden_size, config.hidden_size)
    
    def __call__(self, x: mx.array) -> Dict[str, mx.array]:
        """Forward pass that produces NaN."""
        # Force NaN by dividing by zero
        output = self.linear(x) / 0.0
        return {
            "last_hidden_state": output,
            "pooler_output": output[:, 0],
        }


class MemoryIntensiveModel(nn.Module):
    """Large model for memory testing."""
    
    def __init__(self, hidden_size: int = 4096, num_layers: int = 24):
        super().__init__()
        self.layers = []
        for _ in range(num_layers):
            self.layers.append(nn.Linear(hidden_size, hidden_size))
    
    def __call__(self, x: mx.array) -> mx.array:
        """Forward pass through many large layers."""
        for layer in self.layers:
            x = layer(x)
            x = nn.gelu(x)
        return x


class MockAttentionLayer(nn.Module):
    """Mock attention layer for testing."""
    
    def __init__(self, hidden_size: int, num_attention_heads: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.head_dim = hidden_size // num_attention_heads
        
        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)
        self.output = nn.Linear(hidden_size, hidden_size)
    
    def __call__(
        self,
        hidden_states: mx.array,
        attention_mask: Optional[mx.array] = None,
    ) -> mx.array:
        """Simple attention forward pass."""
        batch_size, seq_length, _ = hidden_states.shape
        
        # QKV projections
        q = self.query(hidden_states)
        k = self.key(hidden_states)
        v = self.value(hidden_states)
        
        # Reshape for attention
        q = q.reshape(batch_size, seq_length, self.num_attention_heads, self.head_dim)
        k = k.reshape(batch_size, seq_length, self.num_attention_heads, self.head_dim)
        v = v.reshape(batch_size, seq_length, self.num_attention_heads, self.head_dim)
        
        # Simple attention (not optimized, just for testing)
        q = q.transpose(0, 2, 1, 3)  # [batch, heads, seq, head_dim]
        k = k.transpose(0, 2, 1, 3)
        v = v.transpose(0, 2, 1, 3)
        
        scores = mx.matmul(q, k.transpose(0, 1, 3, 2)) / mx.sqrt(mx.array(self.head_dim))
        
        if attention_mask is not None:
            scores = scores + attention_mask
        
        attn_weights = mx.softmax(scores, axis=-1)
        attn_output = mx.matmul(attn_weights, v)
        
        # Reshape back
        attn_output = attn_output.transpose(0, 2, 1, 3)
        attn_output = attn_output.reshape(batch_size, seq_length, self.hidden_size)
        
        # Output projection
        output = self.output(attn_output)
        
        return output


# Factory functions
def create_test_bert_model(config: Optional[BertConfig] = None) -> SimpleBertModel:
    """Create test BERT model."""
    if config is None:
        config = create_small_bert_config()
    return SimpleBertModel(config)


def create_test_classifier(
    input_dim: int = 768,
    num_classes: int = 2,
) -> nn.Module:
    """Create test classifier based on number of classes."""
    if num_classes == 2:
        return SimpleBinaryClassifier(input_dim)
    else:
        return SimpleMulticlassClassifier(input_dim, num_classes)


def create_test_regressor(
    input_dim: int = 768,
    output_dim: int = 1,
) -> SimpleRegressor:
    """Create test regression model."""
    return SimpleRegressor(input_dim, output_dim)


def create_model_with_head(
    bert_config: Optional[BertConfig] = None,
    head_config: Optional[Any] = None,
    task_type: str = "classification",
) -> nn.Module:
    """Create BERT model with task-specific head."""
    class BertWithHead(nn.Module):
        def __init__(self):
            super().__init__()
            self.bert = create_test_bert_model(bert_config)
            
            if task_type == "classification":
                config = head_config or create_classification_config(
                    hidden_size=self.bert.config.hidden_size
                )
                self.head = create_test_classifier(
                    self.bert.config.hidden_size,
                    config.num_labels
                )
            else:  # regression
                config = head_config or create_regression_config(
                    hidden_size=self.bert.config.hidden_size
                )
                self.head = create_test_regressor(
                    self.bert.config.hidden_size,
                    config.output_dim if hasattr(config, "output_dim") else 1
                )
            
            self.config = self.bert.config
            self.head_config = config
        
        def __call__(self, input_ids: mx.array, **kwargs) -> mx.array:
            bert_outputs = self.bert(input_ids, **kwargs)
            pooled_output = bert_outputs["pooler_output"]
            return self.head(pooled_output)
    
    return BertWithHead()


# Test utilities
def assert_model_output_shapes(
    model: nn.Module,
    input_shape: Tuple[int, ...],
    expected_output_shape: Optional[Tuple[int, ...]] = None,
) -> bool:
    """Assert model produces expected output shapes."""
    input_data = mx.random.normal(input_shape)
    output = model(input_data)
    
    if expected_output_shape is not None:
        if isinstance(output, dict):
            # For models that return dictionaries
            assert any(
                v.shape == expected_output_shape
                for v in output.values()
            ), f"No output matches expected shape {expected_output_shape}"
        else:
            assert output.shape == expected_output_shape, \
                f"Expected shape {expected_output_shape}, got {output.shape}"
    
    return True


def count_parameters(model: nn.Module) -> int:
    """Count total number of parameters in model."""
    total = 0
    for _, param in mx.tree_flatten(model.parameters()):
        total += param.size
    return total


def check_gradient_flow(
    model: nn.Module,
    loss_fn: Any,
    input_data: Dict[str, mx.array],
) -> Dict[str, bool]:
    """Check if gradients flow through all parameters."""
    # Compute loss and gradients
    loss, grads = mx.value_and_grad(loss_fn)(model, input_data)
    
    # Check each parameter
    results = {}
    flat_params = mx.tree_flatten(model.parameters())
    flat_grads = mx.tree_flatten(grads)
    
    for (param_name, param), (grad_name, grad) in zip(flat_params, flat_grads):
        has_gradient = grad is not None and mx.any(grad != 0)
        results[param_name] = has_gradient
    
    return results