"""Quick validation script to ensure all models work correctly."""

import mlx.core as mx
from models.factory import create_model, MODEL_REGISTRY
from models.bert.config import BertConfig
from models.bert.model import create_bert_for_competition
from models.heads.head_registry import CompetitionType

def validate_bert_core():
    """Validate BERT core model."""
    print("Testing BERT Core...")
    model = create_model("bert_core", hidden_size=128, num_hidden_layers=2)
    
    # Test forward pass
    batch_size, seq_length = 4, 16
    input_ids = mx.random.randint(0, 1000, (batch_size, seq_length))
    output = model(input_ids)
    
    print(f"✅ BERT Core: Output shape = {output.last_hidden_state.shape}")
    return True

def validate_classification_models():
    """Validate classification models."""
    print("\nTesting Classification Models...")
    
    # Binary classification
    model = create_model("bert_with_head", 
                        head_type="binary_classification",
                        hidden_size=128,
                        num_hidden_layers=2,
                        num_labels=2)
    
    batch_size, seq_length = 4, 16
    input_ids = mx.random.randint(0, 1000, (batch_size, seq_length))
    labels = mx.random.randint(0, 2, (batch_size,))
    
    outputs = model(input_ids, labels=labels)
    print(f"✅ Binary Classification: Loss = {float(outputs['loss']):.4f}")
    
    # Multiclass classification
    model = create_model("bert_with_head",
                        head_type="multiclass_classification", 
                        hidden_size=128,
                        num_hidden_layers=2,
                        num_labels=5)
    
    labels = mx.random.randint(0, 5, (batch_size,))
    outputs = model(input_ids, labels=labels)
    print(f"✅ Multiclass Classification: Loss = {float(outputs['loss']):.4f}")
    
    # Multilabel classification
    model = create_model("bert_with_head",
                        head_type="multilabel_classification",
                        hidden_size=128, 
                        num_hidden_layers=2,
                        num_labels=10)
    
    labels = mx.random.uniform(shape=(batch_size, 10)) > 0.5
    outputs = model(input_ids, labels=labels.astype(mx.float32))
    print(f"✅ Multilabel Classification: Loss = {float(outputs['loss']):.4f}")
    
    return True

def validate_regression_models():
    """Validate regression models."""
    print("\nTesting Regression Models...")
    
    # Standard regression
    model = create_model("bert_with_head",
                        head_type="regression",
                        hidden_size=128,
                        num_hidden_layers=2,
                        num_labels=1)
    
    batch_size, seq_length = 4, 16
    input_ids = mx.random.randint(0, 1000, (batch_size, seq_length))
    labels = mx.random.normal((batch_size, 1))
    
    outputs = model(input_ids, labels=labels)
    print(f"✅ Regression: Loss = {float(outputs['loss']):.4f}")
    
    # Ordinal regression
    model = create_model("bert_with_head",
                        head_type="ordinal_regression",
                        hidden_size=128,
                        num_hidden_layers=2,
                        num_labels=5)
    
    labels = mx.random.randint(0, 5, (batch_size,))
    outputs = model(input_ids, labels=labels)
    print(f"✅ Ordinal Regression: Loss = {float(outputs['loss']):.4f}")
    
    return True

def validate_competition_models():
    """Validate competition-specific models."""
    print("\nTesting Competition Models...")
    
    # Binary classification competition
    model = create_bert_for_competition(
        competition_type=CompetitionType.BINARY_CLASSIFICATION,
        bert_config={"hidden_size": 144, "num_hidden_layers": 2, "num_attention_heads": 6},
        num_labels=2
    )
    
    batch_size, seq_length = 4, 16
    input_ids = mx.random.randint(0, 1000, (batch_size, seq_length))
    labels = mx.random.randint(0, 2, (batch_size,))
    
    outputs = model(input_ids, labels=labels)
    print(f"✅ Binary Classification Competition: Loss = {float(outputs['loss']):.4f}")
    
    # Time series
    model = create_bert_for_competition(
        competition_type=CompetitionType.TIME_SERIES,
        bert_config={"hidden_size": 144, "num_hidden_layers": 2, "num_attention_heads": 6},
        num_labels=1
    )
    
    # For time series, we expect multi-step predictions
    labels = mx.random.normal((batch_size, 1, 1))  # Single step for now
    outputs = model(input_ids, labels=labels)
    print(f"✅ Time Series: Loss = {float(outputs['loss']):.4f}")
    
    return True

def validate_model_registry():
    """Validate model registry entries."""
    print("\nTesting Model Registry...")
    
    registered_models = list(MODEL_REGISTRY.keys())
    print(f"Found {len(registered_models)} registered models:")
    for model_name in registered_models:
        print(f"  - {model_name}")
    
    # Test a few registry models
    model = MODEL_REGISTRY["bert-binary"](hidden_size=128, num_hidden_layers=2)
    print(f"✅ Created {model.__class__.__name__} from registry")
    
    return True

def main():
    """Run all validations."""
    print("=" * 60)
    print("BERT Kaggle Models Validation")
    print("=" * 60)
    
    try:
        # Run all validations
        results = [
            validate_bert_core(),
            validate_classification_models(),
            validate_regression_models(),
            validate_competition_models(),
            validate_model_registry()
        ]
        
        if all(results):
            print("\n" + "=" * 60)
            print("✅ ALL VALIDATIONS PASSED!")
            print("=" * 60)
        else:
            print("\n❌ Some validations failed")
            
    except Exception as e:
        print(f"\n❌ Validation error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()