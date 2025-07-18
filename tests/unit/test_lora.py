"""Comprehensive tests for LoRA implementation."""

import pytest
import mlx.core as mx
import mlx.nn as nn
from pathlib import Path
import tempfile
import shutil

from models.lora import (
    LoRAConfig, QLoRAConfig, MultiLoRAConfig, LoRATrainingConfig,
    LoRALinear, QLoRALinear, MultiLoRALinear,
    LoRAAdapter, MultiAdapterManager,
    get_lora_preset, KAGGLE_LORA_PRESETS
)
from models.bert import BertConfig, BertCore, BertWithHead
from models.factory import create_bert_with_lora, create_qlora_model, create_kaggle_lora_model
from models.quantization_utils import QuantizationConfig, QuantizedLinear

from .test_utils import (
    create_dummy_inputs, assert_tensor_shape, assert_all_finite,
    count_parameters, create_simple_bert_config, create_simple_lora_config,
    create_simple_qlora_config, ModelTestCase
)


class TestLoRAConfig:
    """Test LoRA configuration classes."""
    
    def test_default_config(self):
        """Test default LoRA configuration."""
        config = LoRAConfig()
        
        assert config.r == 8
        assert config.alpha == 8
        assert config.dropout == 0.1
        assert config.scaling == 1.0  # alpha/r
        assert "query" in config.target_modules
        assert config.bias == "none"
    
    def test_custom_config(self):
        """Test custom LoRA configuration."""
        config = LoRAConfig(
            r=16,
            alpha=32,
            dropout=0.2,
            target_modules={"dense"},
            use_rslora=True
        )
        
        assert config.r == 16
        assert config.alpha == 32
        assert config.scaling == 2.0
        assert "dense" in config.target_modules
        assert config.use_rslora is True
    
    def test_layer_specific_config(self):
        """Test layer-specific configuration."""
        config = LoRAConfig(
            r=8,
            layer_specific_config={
                "layer.0": {"r": 4, "alpha": 4},
                "layer.11": {"r": 16, "alpha": 32}
            }
        )
        
        # Test getting layer configs
        layer0_config = config.get_layer_config("layer.0")
        assert layer0_config["r"] == 4
        assert layer0_config["alpha"] == 4
        
        # Test default config for unlisted layers
        layer5_config = config.get_layer_config("layer.5")
        assert layer5_config["r"] == 8
        assert layer5_config["alpha"] == 8
    
    def test_should_apply_lora(self):
        """Test LoRA application logic."""
        config = LoRAConfig(
            target_modules={"query", "key"},
            modules_to_save={"classifier"}
        )
        
        # Should apply to target modules
        assert config.should_apply_lora("bert.encoder.layer.0.attention.query")
        assert config.should_apply_lora("bert.encoder.layer.0.attention.key")
        
        # Should not apply to non-target modules
        assert not config.should_apply_lora("bert.encoder.layer.0.attention.value")
        
        # Should not apply to modules_to_save
        assert not config.should_apply_lora("classifier.dense")
    
    def test_qlora_config(self):
        """Test QLoRA configuration."""
        config = QLoRAConfig()
        
        # Check QLoRA-specific defaults
        assert config.r == 4  # Lower rank for QLoRA
        assert config.bnb_4bit_compute_dtype == "float16"
        assert config.bnb_4bit_use_double_quant is True
        assert config.bnb_4bit_quant_type == "nf4"
        assert config.gradient_checkpointing is True
    
    def test_multi_lora_config(self):
        """Test multi-LoRA configuration."""
        config = MultiLoRAConfig()
        
        # Add adapters
        config.add_adapter("task1", LoRAConfig(r=4))
        config.add_adapter("task2", LoRAConfig(r=8))
        
        assert len(config.adapters) == 2
        assert config.adapters["task1"].r == 4
        assert config.adapters["task2"].r == 8
        
        # Test activation
        config.activate_adapter("task1")
        assert "task1" in config.active_adapters
        
        config.deactivate_adapter("task1")
        assert "task1" not in config.active_adapters
    
    def test_lora_presets(self):
        """Test LoRA presets for Kaggle."""
        # Test all presets exist and are valid
        for preset_name in ["efficient", "balanced", "expressive", "qlora_memory", "qlora_quality"]:
            preset = get_lora_preset(preset_name)
            assert isinstance(preset, (LoRAConfig, QLoRAConfig))
        
        # Test specific preset values
        efficient = get_lora_preset("efficient")
        assert efficient.r == 4
        assert len(efficient.target_modules) == 2  # Q, V only
        
        expressive = get_lora_preset("expressive")
        assert expressive.r == 16
        assert len(expressive.target_modules) > 4  # All attention + FFN
        
        # Test invalid preset
        with pytest.raises(ValueError):
            get_lora_preset("invalid_preset")


class TestLoRALayers:
    """Test LoRA layer implementations."""
    
    def test_lora_linear_initialization(self):
        """Test LoRALinear initialization."""
        config = create_simple_lora_config()
        layer = LoRALinear(
            in_features=768,
            out_features=768,
            config=config
        )
        
        # Check attributes
        assert layer.in_features == 768
        assert layer.out_features == 768
        assert layer.r == 4
        assert layer.scaling == 2.0  # alpha/r = 8/4
        
        # Check shapes
        assert layer.lora_A.shape == (768, 4)
        assert layer.lora_B.shape == (4, 768)
        
        # Check initialization
        assert mx.allclose(layer.lora_B, mx.zeros_like(layer.lora_B))
        assert not mx.allclose(layer.lora_A, mx.zeros_like(layer.lora_A))
    
    def test_lora_linear_forward(self):
        """Test LoRALinear forward pass."""
        config = create_simple_lora_config()
        layer = LoRALinear(
            in_features=128,
            out_features=256,
            config=config
        )
        
        # Create input
        batch_size, seq_len, hidden = 2, 10, 128
        x = mx.random.normal((batch_size, seq_len, hidden))
        
        # Forward pass
        output = layer(x)
        
        assert_tensor_shape(output, (batch_size, seq_len, 256))
        assert_all_finite(output)
    
    def test_lora_linear_with_base_layer(self):
        """Test LoRALinear with existing base layer."""
        base_layer = nn.Linear(128, 256)
        config = create_simple_lora_config()
        
        layer = LoRALinear(
            in_features=128,
            out_features=256,
            config=config,
            base_layer=base_layer
        )
        
        # Base layer weight should be preserved
        assert mx.array_equal(layer.base_layer.weight, base_layer.weight)
        
        # Forward pass
        x = mx.random.normal((2, 10, 128))
        output = layer(x)
        assert output.shape == (2, 10, 256)
    
    def test_lora_merge_weights(self):
        """Test LoRA weight merging."""
        config = LoRAConfig(r=4, alpha=8)
        layer = LoRALinear(
            in_features=128,
            out_features=128,
            config=config
        )
        
        # Set non-zero LoRA weights
        layer.lora_A = mx.random.normal(layer.lora_A.shape)
        layer.lora_B = mx.random.normal(layer.lora_B.shape)
        
        # Merge weights
        merged_layer = layer.merge_weights()
        
        assert isinstance(merged_layer, nn.Linear)
        assert merged_layer.weight.shape == (128, 128)
        
        # Test equivalence
        x = mx.random.normal((2, 10, 128))
        orig_output = layer(x)
        merged_output = merged_layer(x)
        
        assert mx.allclose(orig_output, merged_output, rtol=1e-5)
    
    def test_lora_trainable_parameters(self):
        """Test trainable parameter counting."""
        config = LoRAConfig(r=8, alpha=8)
        layer = LoRALinear(
            in_features=768,
            out_features=768,
            config=config
        )
        
        # Calculate expected trainable params
        expected = 8 * (768 + 768)  # r * (in + out)
        assert layer.trainable_parameters == expected
        
        # With bias
        config_with_bias = LoRAConfig(r=8, alpha=8, lora_bias_trainable=True)
        layer_with_bias = LoRALinear(
            in_features=768,
            out_features=768,
            config=config_with_bias
        )
        expected_with_bias = expected + 768  # Add output bias
        assert layer_with_bias.trainable_parameters == expected_with_bias
    
    def test_qlora_linear(self):
        """Test QLoRALinear layer."""
        config = create_simple_qlora_config()
        layer = QLoRALinear(
            in_features=128,
            out_features=256,
            config=config
        )
        
        # Check base layer is quantized
        assert hasattr(layer, 'base_layer')
        
        # Check LoRA components
        assert layer.lora_A.dtype == mx.float16  # Compute dtype
        assert layer.lora_B.dtype == mx.float16
        
        # Forward pass
        x = mx.random.normal((2, 10, 128))
        output = layer(x)
        assert output.shape == (2, 10, 256)
    
    def test_multi_lora_linear(self):
        """Test MultiLoRALinear with multiple adapters."""
        base_layer = nn.Linear(128, 256)
        layer = MultiLoRALinear(
            in_features=128,
            out_features=256,
            base_layer=base_layer
        )
        
        # Add adapters
        layer.add_adapter("adapter1", LoRAConfig(r=4))
        layer.add_adapter("adapter2", LoRAConfig(r=8))
        
        assert len(layer.lora_adapters) == 2
        
        # Test with no active adapters
        x = mx.random.normal((2, 10, 128))
        output = layer(x)
        assert output.shape == (2, 10, 256)
        
        # Activate adapter
        layer.activate_adapter("adapter1")
        output_with_adapter = layer(x)
        assert output_with_adapter.shape == (2, 10, 256)
        
        # Switch adapters
        layer.deactivate_adapter("adapter1")
        layer.activate_adapter("adapter2")
        output_adapter2 = layer(x)
        assert output_adapter2.shape == (2, 10, 256)


class TestLoRAAdapter:
    """Test LoRA adapter injection and management."""
    
    @pytest.fixture
    def simple_model(self):
        """Create a simple BERT model for testing."""
        config = create_simple_bert_config()
        bert = BertCore(config)
        return BertWithHead(bert, head_type="binary_classification", num_labels=2)
    
    def test_adapter_injection(self, simple_model):
        """Test LoRA adapter injection."""
        config = LoRAConfig(r=4, alpha=8, target_modules={"query", "key", "value"})
        adapter = LoRAAdapter(simple_model, config)
        
        # Inject adapters
        stats = adapter.inject_adapters(verbose=True)
        
        # Check injection happened
        assert len(stats) > 0
        assert sum(stats.values()) > 0
        assert len(adapter.injected_modules) > 0
        
        # Check specific modules were injected
        for name, module in simple_model.named_modules():
            if adapter._should_inject_lora(name, module):
                assert name in adapter.injected_modules
    
    def test_adapter_removal(self, simple_model):
        """Test LoRA adapter removal."""
        config = LoRAConfig(r=4)
        adapter = LoRAAdapter(simple_model, config)
        
        # Inject and then remove
        adapter.inject_adapters()
        assert len(adapter.injected_modules) > 0
        
        adapter.remove_adapters(restore_original=True)
        assert len(adapter.injected_modules) == 0
        assert len(adapter.lora_modules) == 0
    
    def test_adapter_state_dict(self, simple_model):
        """Test LoRA state dict save/load."""
        config = LoRAConfig(r=4)
        adapter = LoRAAdapter(simple_model, config)
        adapter.inject_adapters()
        
        # Get state dict
        state_dict = adapter.get_lora_state_dict()
        assert len(state_dict) > 0
        
        # All keys should contain LoRA parameter names
        for key in state_dict:
            assert any(name in key for name in ["lora_A", "lora_B", "lora_bias", "magnitude"])
        
        # Modify and reload
        for key in state_dict:
            state_dict[key] = mx.random.normal(state_dict[key].shape)
        
        adapter.load_lora_state_dict(state_dict)
    
    def test_parameter_freezing(self, simple_model):
        """Test that non-LoRA parameters are frozen."""
        config = LoRAConfig(r=4, modules_to_save={"classifier"})
        adapter = LoRAAdapter(simple_model, config)
        adapter.inject_adapters()
        
        # Check parameter freezing
        for name, param in simple_model.named_parameters():
            if any(lora_name in name for lora_name in ["lora_A", "lora_B"]):
                # LoRA parameters should be trainable
                assert not param.stop_gradient
            elif "classifier" in name:
                # Classifier should be trainable (in modules_to_save)
                assert not param.stop_gradient
            else:
                # Everything else should be frozen
                assert param.stop_gradient
    
    def test_multi_adapter_manager(self, simple_model):
        """Test multi-adapter manager."""
        manager = MultiAdapterManager(simple_model)
        
        # Add multiple adapters
        adapter1 = manager.add_adapter("task1", LoRAConfig(r=4), activate=True)
        adapter2 = manager.add_adapter("task2", LoRAConfig(r=8), activate=False)
        
        assert manager.active_adapter == "task1"
        assert len(manager.adapters) == 2
        
        # Switch adapters
        manager.activate_adapter("task2")
        assert manager.active_adapter == "task2"
        
        # Deactivate all
        manager.deactivate_adapter()
        assert manager.active_adapter is None


class TestLoRAFactory:
    """Test LoRA model creation through factory."""
    
    def test_create_bert_with_lora(self):
        """Test creating BERT with LoRA."""
        model, adapter = create_bert_with_lora(
            head_type="binary_classification",
            lora_preset="efficient",
            num_labels=2
        )
        
        assert isinstance(model, BertWithHead)
        assert isinstance(adapter, LoRAAdapter)
        
        # Check LoRA was injected
        assert len(adapter.injected_modules) > 0
        
        # Test forward pass
        inputs = create_dummy_inputs(include_labels=True)
        outputs = model(**inputs)
        assert hasattr(outputs, 'loss')
        assert hasattr(outputs, 'logits')
    
    def test_create_qlora_model(self):
        """Test creating QLoRA model."""
        model, adapter = create_qlora_model(
            model_type="bert_with_head",
            qlora_preset="qlora_memory",
            head_type="multiclass_classification",
            num_labels=5
        )
        
        # Check model and adapter
        assert isinstance(adapter.config, QLoRAConfig)
        
        # Forward pass
        inputs = create_dummy_inputs(num_labels=5, include_labels=True)
        outputs = model(**inputs)
        assert outputs.logits.shape[-1] == 5
    
    def test_create_kaggle_lora_model(self):
        """Test auto-configured Kaggle LoRA model."""
        # Test different competition types
        competition_types = [
            ("binary_classification", 2, "balanced"),
            ("multiclass_classification", 10, "balanced"),
            ("regression", 1, "efficient"),
        ]
        
        for comp_type, num_labels, expected_preset in competition_types:
            model, adapter = create_kaggle_lora_model(
                competition_type=comp_type,
                num_labels=num_labels
            )
            
            # Check auto-selection worked
            if comp_type == "regression":
                assert adapter.config.r == 4  # Efficient preset
            else:
                assert adapter.config.r == 8  # Balanced preset
    
    def test_lora_parameter_efficiency(self):
        """Test parameter reduction with LoRA."""
        # Create models with and without LoRA
        from models.factory import create_model
        
        # Regular model
        regular_model = create_model(
            "bert_with_head",
            head_type="binary_classification",
            config=create_simple_bert_config()
        )
        regular_params = count_parameters(regular_model, trainable_only=False)
        
        # LoRA model
        lora_model, adapter = create_bert_with_lora(
            head_type="binary_classification",
            lora_preset="efficient",
            bert_config=create_simple_bert_config()
        )
        lora_trainable_params = count_parameters(lora_model, trainable_only=True)
        
        # Check significant parameter reduction
        reduction = (1 - lora_trainable_params / regular_params) * 100
        assert reduction > 90  # Should have >90% parameter reduction


class TestLoRAIntegration:
    """Integration tests for LoRA with full models."""
    
    def test_lora_training_workflow(self):
        """Test complete LoRA training workflow."""
        # Create model with LoRA
        model, adapter = create_bert_with_lora(
            head_type="binary_classification",
            lora_preset="balanced",
            bert_config=create_simple_bert_config()
        )
        
        # Create dummy data
        inputs = create_dummy_inputs(batch_size=4, include_labels=True)
        
        # Forward pass
        outputs = model(**inputs)
        loss = outputs.loss
        
        # Check gradients flow only through LoRA parameters
        gradient_flow = {}
        for name, param in model.named_parameters():
            if not param.stop_gradient:
                # This should be a LoRA parameter or in modules_to_save
                assert any(lora_name in name for lora_name in ["lora_A", "lora_B", "classifier"])
        
        # Simulate training step
        assert loss.item() > 0
        assert_all_finite(loss)
    
    def test_lora_save_load_workflow(self):
        """Test LoRA save/load workflow."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create model with LoRA
            model, adapter = create_bert_with_lora(
                head_type="binary_classification",
                bert_config=create_simple_bert_config()
            )
            
            # Get initial predictions
            inputs = create_dummy_inputs()
            initial_output = model(**inputs)
            
            # Save LoRA weights
            import safetensors.mlx
            lora_state = adapter.get_lora_state_dict()
            save_path = Path(tmpdir) / "lora_weights.safetensors"
            safetensors.mlx.save_file(lora_state, str(save_path))
            
            # Create new model and load LoRA
            new_model, new_adapter = create_bert_with_lora(
                head_type="binary_classification",
                bert_config=create_simple_bert_config()
            )
            
            # Load saved weights
            loaded_state = safetensors.mlx.load_file(str(save_path))
            new_adapter.load_lora_state_dict(loaded_state)
            
            # Check outputs match
            new_output = new_model(**inputs)
            assert mx.allclose(initial_output.logits, new_output.logits, rtol=1e-5)
    
    def test_lora_merge_deployment(self):
        """Test LoRA merging for deployment."""
        # Create model with LoRA
        model, adapter = create_bert_with_lora(
            head_type="binary_classification",
            bert_config=create_simple_bert_config()
        )
        
        # Get predictions with LoRA
        inputs = create_dummy_inputs()
        lora_output = model(**inputs)
        
        # Merge adapters
        merge_status = adapter.merge_adapters()
        assert any(merge_status.values())
        
        # Check predictions still match
        merged_output = model(**inputs)
        assert mx.allclose(lora_output.logits, merged_output.logits, rtol=1e-5)
        
        # Check no LoRA modules remain
        assert len(adapter.lora_modules) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])