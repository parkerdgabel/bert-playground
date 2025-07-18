"""Comprehensive test suite for the modular BERT architecture.

This script tests:
1. BertCore functionality
2. BertWithHead integration
3. All head types
4. Factory functions
5. Save/load functionality
6. Backward compatibility
7. Error handling
"""

import mlx.core as mx
import mlx.nn as nn
from pathlib import Path
import shutil
import traceback
from typing import Dict, List, Tuple
import numpy as np

# Import modular BERT components
from models.bert import (
    BertCore, BertWithHead, BertOutput,
    create_bert_core, create_bert_with_head, create_bert_for_competition
)
from models.heads.base_head import HeadType, PoolingType, HeadConfig
from models.heads.head_registry import HeadRegistry, CompetitionType, get_head_registry
from models.factory import create_model, create_bert_for_task, create_modular_bert

# Import for backward compatibility tests
from models.modernbert import ModernBertModel, ModernBertConfig


class TestResults:
    """Track test results."""
    def __init__(self):
        self.passed = []
        self.failed = []
        self.warnings = []
    
    def add_pass(self, test_name: str):
        self.passed.append(test_name)
        print(f"✅ {test_name}")
    
    def add_fail(self, test_name: str, error: str):
        self.failed.append((test_name, error))
        print(f"❌ {test_name}: {error}")
    
    def add_warning(self, test_name: str, warning: str):
        self.warnings.append((test_name, warning))
        print(f"⚠️  {test_name}: {warning}")
    
    def summary(self):
        print("\n" + "="*60)
        print("TEST SUMMARY")
        print("="*60)
        print(f"✅ Passed: {len(self.passed)}")
        print(f"❌ Failed: {len(self.failed)}")
        print(f"⚠️  Warnings: {len(self.warnings)}")
        
        if self.failed:
            print("\nFailed tests:")
            for test, error in self.failed:
                print(f"  - {test}: {error}")
        
        if self.warnings:
            print("\nWarnings:")
            for test, warning in self.warnings:
                print(f"  - {test}: {warning}")
        
        return len(self.failed) == 0


results = TestResults()


def test_bert_core():
    """Test BertCore functionality."""
    print("\n=== Testing BertCore ===")
    
    try:
        # Test 1: Basic creation
        bert = create_bert_core(
            hidden_size=128, num_hidden_layers=2, num_attention_heads=8
        )
        results.add_pass("BertCore creation")
        
        # Test 2: Forward pass
        batch_size, seq_len = 2, 32
        input_ids = mx.random.randint(0, 100, (batch_size, seq_len))
        attention_mask = mx.ones((batch_size, seq_len))
        
        outputs = bert(input_ids, attention_mask)
        assert isinstance(outputs, BertOutput), "Output should be BertOutput instance"
        assert outputs.last_hidden_state.shape == (batch_size, seq_len, 128)
        assert outputs.pooler_output.shape == (batch_size, 128)
        results.add_pass("BertCore forward pass")
        
        # Test 3: Pooling strategies
        assert outputs.cls_output.shape == (batch_size, 128)
        assert outputs.mean_pooled.shape == (batch_size, 128)
        assert outputs.max_pooled.shape == (batch_size, 128)
        
        # Test get_pooled_output
        for pooling_type in ["cls", "mean", "max", "pooler"]:
            pooled = outputs.get_pooled_output(pooling_type)
            assert pooled.shape == (batch_size, 128)
        results.add_pass("BertCore pooling strategies")
        
        # Test 4: Config access
        assert bert.get_hidden_size() == 128
        assert bert.get_num_layers() == 2
        results.add_pass("BertCore config access")
        
    except Exception as e:
        results.add_fail("BertCore tests", str(e))
        traceback.print_exc()


def test_bert_with_head():
    """Test BertWithHead integration."""
    print("\n=== Testing BertWithHead ===")
    
    try:
        # Test 1: Creation with different heads
        head_types = [
            HeadType.BINARY_CLASSIFICATION,
            HeadType.MULTICLASS_CLASSIFICATION,
            HeadType.REGRESSION,
        ]
        
        for head_type in head_types:
            model = create_bert_with_head(
                bert_config={"hidden_size": 128, "num_hidden_layers": 2, "num_attention_heads": 8},
                head_type=head_type,
                num_labels=5 if head_type == HeadType.MULTICLASS_CLASSIFICATION else 2
            )
            assert isinstance(model, BertWithHead)
            assert model.get_head().head_type == head_type
        
        results.add_pass("BertWithHead creation with different heads")
        
        # Test 2: Forward pass with labels
        model = create_bert_with_head(
            bert_config={"hidden_size": 128, "num_hidden_layers": 2, "num_attention_heads": 8},
            head_type=HeadType.BINARY_CLASSIFICATION,
            num_labels=2
        )
        
        batch_size, seq_len = 4, 16
        input_ids = mx.random.randint(0, 100, (batch_size, seq_len))
        attention_mask = mx.ones((batch_size, seq_len))
        labels = mx.random.randint(0, 2, (batch_size,))
        
        outputs = model(input_ids, attention_mask, labels=labels)
        assert "loss" in outputs and outputs["loss"] is not None
        assert "predictions" in outputs
        assert "head_outputs" in outputs
        results.add_pass("BertWithHead forward pass")
        
        # Test 3: Freezing functionality
        model.freeze_bert(num_layers=1)
        model.unfreeze_bert()
        results.add_pass("BertWithHead freezing")
        
    except Exception as e:
        results.add_fail("BertWithHead tests", str(e))
        traceback.print_exc()


def test_all_head_types():
    """Test all available head types."""
    print("\n=== Testing All Head Types ===")
    
    registry = get_head_registry()  # Use global registry
    
    # Get all registered heads
    all_heads = registry._heads.items()  # Access internal registry
    
    for head_name, head_spec in all_heads:
        head_type = head_spec.head_type
        
        try:
            # Skip heads that require special configuration
            if head_type in [HeadType.ENSEMBLE, HeadType.MULTI_TASK, HeadType.ADAPTIVE]:
                results.add_warning(f"Head {head_name}", "Requires special configuration, skipping")
                continue
            
            # Create model with this head
            num_labels = 5 if "multiclass" in head_name else 2
            if head_type == HeadType.REGRESSION:
                num_labels = 1
            
            model = create_bert_with_head(
                bert_config={"hidden_size": 64, "num_hidden_layers": 1, "num_attention_heads": 8},
                head_type=head_type,
                num_labels=num_labels
            )
            
            # Test forward pass
            input_ids = mx.random.randint(0, 50, (2, 16))
            attention_mask = mx.ones((2, 16))
            
            outputs = model(input_ids, attention_mask)
            assert "head_outputs" in outputs
            
            results.add_pass(f"Head {head_name}")
            
        except Exception as e:
            results.add_fail(f"Head {head_name}", str(e))


def test_factory_functions():
    """Test factory functions."""
    print("\n=== Testing Factory Functions ===")
    
    try:
        # Test 1: create_model with bert_core
        model = create_model(
            model_type="bert_core",
            config={"hidden_size": 128, "num_hidden_layers": 2, "num_attention_heads": 8}
        )
        assert isinstance(model, BertCore)
        results.add_pass("create_model with bert_core")
        
        # Test 2: create_model with bert_with_head
        model = create_model(
            model_type="bert_with_head",
            head_type="multiclass_classification",
            num_labels=5,
            config={"hidden_size": 128}
        )
        assert isinstance(model, BertWithHead)
        results.add_pass("create_model with bert_with_head")
        
        # Test 3: create_bert_for_task
        tasks = ["binary_classification", "regression", "ranking"]
        for task in tasks:
            model = create_bert_for_task(
                task=task,
                num_labels=2 if task != "regression" else 1
            )
            assert isinstance(model, BertWithHead)
        results.add_pass("create_bert_for_task")
        
        # Test 4: create_bert_for_competition
        model = create_bert_for_competition(
            competition_type=CompetitionType.MULTILABEL_CLASSIFICATION,
            num_labels=10
        )
        assert isinstance(model, BertWithHead)
        results.add_pass("create_bert_for_competition")
        
    except Exception as e:
        results.add_fail("Factory functions", str(e))
        traceback.print_exc()


def test_save_load():
    """Test save and load functionality."""
    print("\n=== Testing Save/Load ===")
    
    temp_dir = Path("test_output/save_load_test")
    
    try:
        # Create a model
        original_model = create_bert_with_head(
            bert_config={"hidden_size": 64, "num_hidden_layers": 2, "num_attention_heads": 8},
            head_type=HeadType.MULTICLASS_CLASSIFICATION,
            num_labels=3
        )
        
        # Save model
        original_model.save_pretrained(temp_dir)
        assert (temp_dir / "bert").exists()
        assert (temp_dir / "head").exists()
        assert (temp_dir / "model_metadata.json").exists()
        results.add_pass("Model saving")
        
        # Load model
        loaded_model = BertWithHead.from_pretrained(temp_dir)
        assert isinstance(loaded_model, BertWithHead)
        assert loaded_model.get_bert().get_hidden_size() == 64
        assert loaded_model.get_head().head_type == HeadType.MULTICLASS_CLASSIFICATION
        results.add_pass("Model loading")
        
        # Test forward pass with loaded model
        input_ids = mx.random.randint(0, 50, (2, 16))
        outputs = loaded_model(input_ids)
        assert "head_outputs" in outputs
        results.add_pass("Loaded model forward pass")
        
    except Exception as e:
        results.add_fail("Save/Load", str(e))
        traceback.print_exc()
    finally:
        if temp_dir.exists():
            shutil.rmtree(temp_dir)


def test_pooling_strategies():
    """Test different pooling strategies."""
    print("\n=== Testing Pooling Strategies ===")
    
    pooling_types = [
        PoolingType.CLS,
        PoolingType.MEAN,
        PoolingType.MAX,
        PoolingType.ATTENTION,
        PoolingType.WEIGHTED_MEAN,
        PoolingType.LAST,
    ]
    
    for pooling_type in pooling_types:
        try:
            # Create model with specific pooling
            head_config = HeadConfig(
                head_type=HeadType.BINARY_CLASSIFICATION,
                input_size=64,
                output_size=2,
                pooling_type=pooling_type
            )
            
            model = create_bert_with_head(
                bert_config={"hidden_size": 64, "num_hidden_layers": 1, "num_attention_heads": 8},
                head_config=head_config
            )
            
            # Test forward pass
            input_ids = mx.random.randint(0, 50, (2, 16))
            outputs = model(input_ids)
            assert "head_outputs" in outputs
            
            results.add_pass(f"Pooling {pooling_type.value}")
            
        except Exception as e:
            results.add_fail(f"Pooling {pooling_type.value}", str(e))


def test_backward_compatibility():
    """Test backward compatibility with old architecture."""
    print("\n=== Testing Backward Compatibility ===")
    
    try:
        # Test 1: ModernBertModel still works
        config = ModernBertConfig(hidden_size=128, num_hidden_layers=2)
        old_model = ModernBertModel(config)
        
        input_ids = mx.random.randint(0, 50, (2, 16))
        outputs = old_model(input_ids)
        assert "last_hidden_state" in outputs
        results.add_pass("ModernBertModel compatibility")
        
        # Test 2: Factory with old model types
        model = create_model(model_type="standard", config={"hidden_size": 128})
        assert isinstance(model, ModernBertModel)
        results.add_pass("Factory old model types")
        
    except Exception as e:
        results.add_fail("Backward compatibility", str(e))


def test_error_handling():
    """Test error handling and edge cases."""
    print("\n=== Testing Error Handling ===")
    
    try:
        # Test 1: Incompatible dimensions
        bert = create_bert_core(hidden_size=128)
        head_config = HeadConfig(
            head_type=HeadType.BINARY_CLASSIFICATION,
            input_size=256,  # Wrong size!
            output_size=2
        )
        
        try:
            # Get head class
            registry = get_head_registry()  # Use global registry
            # Find head for type
            head_names = []
            for name, spec in registry._heads.items():
                if spec.head_type == HeadType.BINARY_CLASSIFICATION:
                    head_names.append(name)
            if head_names:
                head_class = registry.get_head_class(head_names[0])
            head = head_class(head_config)
            
            # This should raise an error
            model = BertWithHead(bert, head)
            results.add_fail("Dimension mismatch detection", "Should have raised error")
        except ValueError as e:
            if "does not match" in str(e):
                results.add_pass("Dimension mismatch detection")
            else:
                results.add_fail("Dimension mismatch detection", f"Wrong error: {e}")
        
        # Test 2: Invalid pooling type in BertOutput
        bert = create_bert_core(hidden_size=64)
        outputs = bert(mx.random.randint(0, 50, (1, 8)))
        
        try:
            outputs.get_pooled_output("invalid_pooling")
            results.add_fail("Invalid pooling detection", "Should have raised error")
        except ValueError:
            results.add_pass("Invalid pooling detection")
        
    except Exception as e:
        results.add_fail("Error handling", str(e))
        traceback.print_exc()


def test_performance():
    """Test performance characteristics."""
    print("\n=== Testing Performance ===")
    
    try:
        import time
        
        # Create models of different sizes
        configs = [
            {"hidden_size": 128, "num_hidden_layers": 2, "num_attention_heads": 8},
            {"hidden_size": 256, "num_hidden_layers": 4, "num_attention_heads": 16},
            {"hidden_size": 512, "num_hidden_layers": 6, "num_attention_heads": 16},
        ]
        
        for config in configs:
            model = create_bert_with_head(
                bert_config=config,
                head_type=HeadType.BINARY_CLASSIFICATION
            )
            
            # Time forward pass
            input_ids = mx.random.randint(0, 100, (8, 64))
            
            # Warm up
            _ = model(input_ids)
            mx.eval(model.parameters())
            
            # Time it
            start = time.time()
            for _ in range(10):
                _ = model(input_ids)
                mx.eval(model.parameters())
            elapsed = time.time() - start
            
            avg_time = elapsed / 10
            print(f"  Config {config}: {avg_time*1000:.2f}ms per forward pass")
            
            if avg_time > 1.0:  # More than 1 second per pass
                results.add_warning(f"Performance {config}", f"Slow: {avg_time:.2f}s")
            else:
                results.add_pass(f"Performance {config}")
        
    except Exception as e:
        results.add_fail("Performance tests", str(e))


def test_integration_scenarios():
    """Test real-world integration scenarios."""
    print("\n=== Testing Integration Scenarios ===")
    
    try:
        # Scenario 1: Create model for Kaggle competition
        model = create_bert_for_competition(
            competition_type="binary_classification",
            bert_config={"hidden_size": 128, "num_hidden_layers": 2, "num_attention_heads": 8},
            num_labels=2
        )
        
        # Simulate training data
        batch_size = 16
        input_ids = mx.random.randint(0, 100, (batch_size, 32))
        labels = mx.random.randint(0, 2, (batch_size,))
        
        # Forward pass
        outputs = model(input_ids, labels=labels)
        loss = outputs["loss"]
        assert loss is not None
        results.add_pass("Kaggle competition scenario")
        
        # Scenario 2: Transfer learning with frozen BERT
        model = create_bert_with_head(
            head_type=HeadType.REGRESSION,
            num_labels=1,
            freeze_bert_layers=1  # Freeze first layer only
        )
        
        outputs = model(input_ids[:4])  # Smaller batch for regression
        assert "head_outputs" in outputs
        results.add_pass("Transfer learning scenario")
        
        # Scenario 3: Multi-task with different pooling
        head_config = HeadConfig(
            head_type=HeadType.MULTILABEL_CLASSIFICATION,
            input_size=128,
            output_size=10,
            pooling_type=PoolingType.ATTENTION,
            hidden_sizes=[64],
            dropout_prob=0.2
        )
        
        model = create_bert_with_head(
            bert_config={"hidden_size": 128, "num_hidden_layers": 2, "num_attention_heads": 8},
            head_config=head_config
        )
        
        outputs = model(input_ids)
        assert outputs["head_outputs"]["predictions"].shape == (batch_size, 10)
        results.add_pass("Multi-task scenario")
        
    except Exception as e:
        results.add_fail("Integration scenarios", str(e))
        traceback.print_exc()


def main():
    """Run all tests."""
    print("🧪 Comprehensive Modular BERT Testing")
    print("=" * 60)
    
    # Run all test suites
    test_bert_core()
    test_bert_with_head()
    test_all_head_types()
    test_factory_functions()
    test_save_load()
    test_pooling_strategies()
    test_backward_compatibility()
    test_error_handling()
    test_performance()
    test_integration_scenarios()
    
    # Print summary
    success = results.summary()
    
    if success:
        print("\n✅ All tests passed! The modular BERT architecture is working correctly.")
    else:
        print("\n❌ Some tests failed. Please review the errors above.")
        exit(1)


if __name__ == "__main__":
    main()