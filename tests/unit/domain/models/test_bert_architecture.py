"""Unit tests for BERT architecture domain models.

These tests verify the pure business logic of BERT architecture
without any framework dependencies.
"""

import pytest
from typing import List, Optional, Any, Dict

from domain.entities.model import (
    AttentionType,
    ActivationType,
    ModelArchitecture,
    ModelType
)
from domain.value_objects.hyperparameters import (
    OptimizerConfig,
    SchedulerConfig,
    LayerConfig
)


class TestLayerComponents:
    """Test layer components logic."""
    
    def test_layer_components_basic(self):
        """Test basic layer components."""
        components = LayerComponents(
            attention_type=AttentionType.GLOBAL,
            use_pre_normalization=True,
            normalization_type=NormalizationType.LAYER_NORM,
            feedforward_type="standard",
            dropout_rate=0.1,
            layer_index=0
        )
        
        assert components.attention_type == AttentionType.GLOBAL
        assert components.use_pre_normalization is True
        assert components.normalization_type == NormalizationType.LAYER_NORM
        assert components.feedforward_type == "standard"
        assert components.dropout_rate == 0.1
        assert components.layer_index == 0
    
    def test_is_global_attention(self):
        """Test global attention detection."""
        global_layer = LayerComponents(
            attention_type=AttentionType.GLOBAL,
            use_pre_normalization=True,
            normalization_type=NormalizationType.LAYER_NORM,
            feedforward_type="standard",
            dropout_rate=0.1,
            layer_index=0
        )
        assert global_layer.is_global_attention is True
        
        local_layer = LayerComponents(
            attention_type=AttentionType.LOCAL,
            use_pre_normalization=True,
            normalization_type=NormalizationType.LAYER_NORM,
            feedforward_type="standard",
            dropout_rate=0.1,
            layer_index=1
        )
        assert local_layer.is_global_attention is False
    
    def test_requires_position_encoding(self):
        """Test position encoding requirement."""
        # Global and local attention require position encoding
        for attention_type in [AttentionType.GLOBAL, AttentionType.LOCAL]:
            components = LayerComponents(
                attention_type=attention_type,
                use_pre_normalization=True,
                normalization_type=NormalizationType.LAYER_NORM,
                feedforward_type="standard",
                dropout_rate=0.1,
                layer_index=0
            )
            assert components.requires_position_encoding is True
        
        # Sparse and sliding window might not require traditional position encoding
        for attention_type in [AttentionType.SPARSE, AttentionType.SLIDING_WINDOW]:
            components = LayerComponents(
                attention_type=attention_type,
                use_pre_normalization=True,
                normalization_type=NormalizationType.LAYER_NORM,
                feedforward_type="standard",
                dropout_rate=0.1,
                layer_index=0
            )
            assert components.requires_position_encoding is False


# Mock implementations for testing abstract classes
class MockBertLayer(BertLayer[Dict[str, Any]]):
    """Mock BERT layer for testing."""
    
    def _create_components(self) -> LayerComponents:
        """Create mock components."""
        return LayerComponents(
            attention_type=self.attention_type,
            use_pre_normalization=self.config.use_pre_layer_normalization,
            normalization_type=NormalizationType(self.config.normalization_type),
            feedforward_type="geglu" if self.config.use_gated_linear_units else "standard",
            dropout_rate=self.config.hidden_dropout_prob,
            layer_index=self.layer_index
        )


class TestBertLayer:
    """Test BERT layer logic."""
    
    def test_layer_without_alternating_attention(self):
        """Test layer with standard global attention."""
        config = BertDomainConfig(
            hidden_size=768,
            num_hidden_layers=12,
            use_alternating_attention=False
        )
        
        layer = MockBertLayer(config, layer_index=5)
        assert layer.attention_type == AttentionType.GLOBAL
        assert layer.layer_index == 5
        assert layer.is_final_layer is False
    
    def test_layer_with_alternating_attention(self):
        """Test layer with alternating attention pattern."""
        config = BertDomainConfig(
            hidden_size=768,
            num_hidden_layers=12,
            use_alternating_attention=True,
            global_attention_frequency=3
        )
        
        # Layer 0: Global (0 % 3 == 0)
        layer0 = MockBertLayer(config, layer_index=0)
        assert layer0.attention_type == AttentionType.GLOBAL
        
        # Layer 1: Local (1 % 3 != 0)
        layer1 = MockBertLayer(config, layer_index=1)
        assert layer1.attention_type == AttentionType.LOCAL
        
        # Layer 3: Global (3 % 3 == 0)
        layer3 = MockBertLayer(config, layer_index=3)
        assert layer3.attention_type == AttentionType.GLOBAL
    
    def test_is_final_layer(self):
        """Test final layer detection."""
        config = BertDomainConfig(
            hidden_size=768,
            num_hidden_layers=12
        )
        
        # Middle layer
        layer5 = MockBertLayer(config, layer_index=5)
        assert layer5.is_final_layer is False
        
        # Final layer
        layer11 = MockBertLayer(config, layer_index=11)
        assert layer11.is_final_layer is True


class MockBertEmbeddings(BertEmbeddings[Dict[str, Any]]):
    """Mock BERT embeddings for testing."""
    pass


class TestBertEmbeddings:
    """Test BERT embeddings logic."""
    
    def test_embedding_types_basic(self):
        """Test basic embedding type determination."""
        config = BertDomainConfig(
            hidden_size=768,
            vocab_size=30000,
            type_vocab_size=0,
            use_rotary_embeddings=False
        )
        
        embeddings = MockBertEmbeddings(config)
        assert "token" in embeddings.embedding_types
        assert "position" in embeddings.embedding_types
        assert "token_type" not in embeddings.embedding_types
    
    def test_embedding_types_with_token_types(self):
        """Test embedding types with token type embeddings."""
        config = BertDomainConfig(
            hidden_size=768,
            vocab_size=30000,
            type_vocab_size=2,
            use_rotary_embeddings=False
        )
        
        embeddings = MockBertEmbeddings(config)
        assert "token" in embeddings.embedding_types
        assert "position" in embeddings.embedding_types
        assert "token_type" in embeddings.embedding_types
    
    def test_embedding_types_with_rotary(self):
        """Test embedding types with rotary embeddings."""
        config = BertDomainConfig(
            hidden_size=768,
            vocab_size=30000,
            type_vocab_size=0,
            use_rotary_embeddings=True
        )
        
        embeddings = MockBertEmbeddings(config)
        assert "token" in embeddings.embedding_types
        assert "position" not in embeddings.embedding_types  # Rotary replaces position
    
    def test_total_embedding_parameters(self):
        """Test parameter counting for embeddings."""
        config = BertDomainConfig(
            hidden_size=768,
            vocab_size=30000,
            max_position_embeddings=512,
            type_vocab_size=2,
            use_rotary_embeddings=False
        )
        
        embeddings = MockBertEmbeddings(config)
        
        # Token embeddings: 30000 * 768
        # Token type embeddings: 2 * 768
        # Position embeddings: 512 * 768
        # Layer norm: 2 * 768 (weight + bias)
        expected = 30000 * 768 + 2 * 768 + 512 * 768 + 2 * 768
        assert embeddings.total_embedding_parameters == expected


class MockBertArchitecture(BertArchitecture[Dict[str, Any], List[float]]):
    """Mock BERT architecture for testing."""
    
    def compute_attention_mask(
        self,
        attention_mask: Optional[List[float]],
        input_shape: tuple[int, ...],
        device: Any = None
    ) -> List[float]:
        """Mock attention mask computation."""
        if attention_mask is None:
            batch_size, seq_length = input_shape
            return [1.0] * (batch_size * seq_length)
        return attention_mask
    
    def get_layer_types(self) -> List[LayerComponents]:
        """Get mock layer types."""
        layer_types = []
        for i in range(self.num_layers):
            layer = MockBertLayer(self.config, i)
            layer_types.append(layer.components)
        return layer_types


class TestBertArchitecture:
    """Test BERT architecture logic."""
    
    def test_architecture_validation(self):
        """Test architecture configuration validation."""
        # Valid config
        config = BertDomainConfig(
            hidden_size=768,
            num_hidden_layers=12,
            use_alternating_attention=True,
            global_attention_frequency=3
        )
        arch = MockBertArchitecture(config)
        assert arch.num_layers == 12
        
        # Invalid config - too few layers for alternating attention
        with pytest.raises(ValueError, match="Number of layers must be >="):
            invalid_config = BertDomainConfig(
                hidden_size=768,
                num_hidden_layers=2,
                use_alternating_attention=True,
                global_attention_frequency=3
            )
            MockBertArchitecture(invalid_config)
    
    def test_architecture_properties(self):
        """Test architecture properties."""
        config = BertDomainConfig(
            hidden_size=768,
            num_hidden_layers=12
        )
        arch = MockBertArchitecture(config)
        
        assert arch.num_layers == 12
        assert arch.hidden_size == 768
        assert arch.supports_gradient_checkpointing is True
    
    def test_architectural_insights(self):
        """Test architectural insights generation."""
        config = BertDomainConfig(
            hidden_size=768,
            num_hidden_layers=12,
            use_alternating_attention=True,
            global_attention_frequency=3,
            use_pre_layer_normalization=True,
            max_position_embeddings=1024
        )
        arch = MockBertArchitecture(config)
        
        insights = arch.get_architectural_insights()
        
        assert insights["total_layers"] == 12
        assert insights["global_attention_layers"] == 4  # Layers 0, 3, 6, 9
        assert insights["local_attention_layers"] == 8
        assert insights["uses_pre_normalization"] is True
        assert insights["supports_long_context"] is True  # > 512
        assert "estimated_parameters" in insights
        assert "architectural_family" in insights
    
    def test_architectural_family_detection(self):
        """Test architectural family classification."""
        # Classic BERT
        classic_config = BertDomainConfig(
            hidden_size=768,
            num_hidden_layers=12,
            use_rotary_embeddings=False,
            use_gated_linear_units=False,
            use_pre_layer_normalization=False
        )
        classic_arch = MockBertArchitecture(classic_config)
        insights = classic_arch.get_architectural_insights()
        assert insights["architectural_family"] == "Classic BERT"
        
        # ModernBERT
        modern_config = BertDomainConfig(
            hidden_size=768,
            num_hidden_layers=12,
            use_rotary_embeddings=True,
            use_gated_linear_units=True,
            glu_activation_type="geglu"
        )
        modern_arch = MockBertArchitecture(modern_config)
        insights = modern_arch.get_architectural_insights()
        assert insights["architectural_family"] == "ModernBERT"
        
        # neoBERT
        neo_config = BertDomainConfig(
            hidden_size=768,
            num_hidden_layers=12,
            use_rotary_embeddings=True,
            use_gated_linear_units=True,
            glu_activation_type="swiglu"
        )
        neo_arch = MockBertArchitecture(neo_config)
        insights = neo_arch.get_architectural_insights()
        assert insights["architectural_family"] == "neoBERT"


class TestBertPooler:
    """Test BERT pooler logic."""
    
    def test_pooler_basic(self):
        """Test basic pooler configuration."""
        config = BertDomainConfig(
            hidden_size=768,
            pooler_hidden_size=None,
            pooler_activation="tanh",
            pooler_dropout_probability=0.1
        )
        
        pooler = BertPooler(config)
        assert pooler.input_size == 768
        assert pooler.output_size == 768  # Uses hidden_size when pooler_hidden_size is None
        assert pooler.activation == "tanh"
        assert pooler.dropout_prob == 0.1
    
    def test_pooler_with_custom_size(self):
        """Test pooler with custom hidden size."""
        config = BertDomainConfig(
            hidden_size=768,
            pooler_hidden_size=256,
            pooler_activation="gelu"
        )
        
        pooler = BertPooler(config)
        assert pooler.input_size == 768
        assert pooler.output_size == 256
        assert pooler.activation == "gelu"
    
    def test_pooler_parameters(self):
        """Test pooler parameter counting."""
        config = BertDomainConfig(
            hidden_size=768,
            pooler_hidden_size=256
        )
        
        pooler = BertPooler(config)
        # Weight matrix: 768 * 256
        # Bias: 256
        expected = 768 * 256 + 256
        assert pooler.num_parameters == expected


class TestAttentionPattern:
    """Test attention pattern descriptions."""
    
    def test_global_attention_pattern(self):
        """Test global attention pattern description."""
        pattern = AttentionPattern.get_attention_pattern(
            AttentionType.GLOBAL,
            sequence_length=512
        )
        assert "Global attention" in pattern
        assert "512" in pattern
        assert "all" in pattern
    
    def test_local_attention_pattern(self):
        """Test local attention pattern description."""
        pattern = AttentionPattern.get_attention_pattern(
            AttentionType.LOCAL,
            sequence_length=512,
            window_size=64
        )
        assert "Local attention" in pattern
        assert "64 neighbors" in pattern
    
    def test_sliding_window_pattern(self):
        """Test sliding window attention pattern description."""
        pattern = AttentionPattern.get_attention_pattern(
            AttentionType.SLIDING_WINDOW,
            sequence_length=512,
            window_size=128
        )
        assert "Sliding window" in pattern
        assert "128" in pattern
    
    def test_sparse_attention_pattern(self):
        """Test sparse attention pattern description."""
        pattern = AttentionPattern.get_attention_pattern(
            AttentionType.SPARSE,
            sequence_length=512
        )
        assert "Sparse attention" in pattern
        assert "custom pattern" in pattern


class TestModelCapabilities:
    """Test model capabilities analysis."""
    
    def test_basic_capabilities(self):
        """Test basic model capabilities."""
        config = BertDomainConfig(
            hidden_size=768,
            max_position_embeddings=512,
            type_vocab_size=2
        )
        
        capabilities = ModelCapabilities(config)
        assert capabilities.max_sequence_length == 512
        assert capabilities.supports_token_types is True
        assert capabilities.supports_long_sequences is False
    
    def test_long_sequence_support(self):
        """Test long sequence support detection."""
        config = BertDomainConfig(
            hidden_size=768,
            max_position_embeddings=2048
        )
        
        capabilities = ModelCapabilities(config)
        assert capabilities.supports_long_sequences is True
    
    def test_memory_efficiency(self):
        """Test memory efficiency detection."""
        # Not memory efficient
        config1 = BertDomainConfig(
            hidden_size=768,
            use_alternating_attention=False,
            use_rotary_embeddings=False,
            use_bias_in_linear=True
        )
        capabilities1 = ModelCapabilities(config1)
        assert capabilities1.memory_efficient is False
        
        # Memory efficient with alternating attention
        config2 = BertDomainConfig(
            hidden_size=768,
            use_alternating_attention=True
        )
        capabilities2 = ModelCapabilities(config2)
        assert capabilities2.memory_efficient is True
        
        # Memory efficient with rotary embeddings
        config3 = BertDomainConfig(
            hidden_size=768,
            use_rotary_embeddings=True
        )
        capabilities3 = ModelCapabilities(config3)
        assert capabilities3.memory_efficient is True
    
    def test_inference_optimization(self):
        """Test inference optimization detection."""
        # Not optimized
        config1 = BertDomainConfig(
            hidden_size=768,
            use_pre_layer_normalization=False,
            normalization_type="layer_norm",
            use_gated_linear_units=False
        )
        capabilities1 = ModelCapabilities(config1)
        assert capabilities1.inference_optimized is False
        
        # Optimized with pre-norm
        config2 = BertDomainConfig(
            hidden_size=768,
            use_pre_layer_normalization=True
        )
        capabilities2 = ModelCapabilities(config2)
        assert capabilities2.inference_optimized is True
        
        # Optimized with RMS norm
        config3 = BertDomainConfig(
            hidden_size=768,
            normalization_type="rms_norm"
        )
        capabilities3 = ModelCapabilities(config3)
        assert capabilities3.inference_optimized is True
    
    def test_capabilities_summary(self):
        """Test capabilities summary generation."""
        config = BertDomainConfig(
            hidden_size=768,
            max_position_embeddings=1024,
            type_vocab_size=2,
            use_rotary_embeddings=True,
            use_gated_linear_units=True,
            use_alternating_attention=True,
            use_pre_layer_normalization=True
        )
        
        capabilities = ModelCapabilities(config)
        summary = capabilities.get_capabilities_summary()
        
        assert summary["max_sequence_length"] == 1024
        assert summary["supports_token_types"] is True
        assert summary["supports_long_sequences"] is True
        assert summary["memory_efficient"] is True
        assert summary["inference_optimized"] is True
        
        features = summary["architectural_features"]
        assert features["rotary_embeddings"] is True
        assert features["gated_linear_units"] is True
        assert features["alternating_attention"] is True
        assert features["pre_normalization"] is True