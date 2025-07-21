"""
Advanced Classification Examples

Demonstrates how to use the new classification heads and features for various tasks.
"""

import mlx.core as mx

from models.classification import (
    create_classifier,
    create_ensemble_classifier,
    create_hierarchical_classifier,
    create_multilabel_classifier,
    create_ordinal_classifier,
)


def example_multilabel_classification():
    """Example: Toxic comment classification with multiple labels."""
    print("\n=== Multilabel Classification Example ===")

    # Create a multilabel classifier for toxic comment classification
    classifier = create_multilabel_classifier(
        model_name="mlx-community/answerdotai-ModernBERT-base-4bit",
        num_labels=6,
        label_names=[
            "toxic",
            "severe_toxic",
            "obscene",
            "threat",
            "insult",
            "identity_hate",
        ],
        pos_weights=[2.0, 5.0, 3.0, 10.0, 2.5, 8.0],  # Handle class imbalance
        hidden_dim=256,
        pooling_type="mean",
        activation="gelu",
        dropout_rate=0.1,
    )

    # Example usage
    texts = [
        "You are amazing and I love your work!",
        "I hate you and hope bad things happen",
        "This is a normal comment about the weather",
    ]

    # Tokenize (pseudo-code, replace with actual tokenizer)
    input_ids = mx.array(
        [[101, 2023, 2003, 102], [101, 2045, 2003, 102], [101, 2062, 2003, 102]]
    )
    attention_mask = mx.ones_like(input_ids)

    # Get predictions
    predictions = classifier.predict(input_ids, attention_mask)
    print(f"Predictions shape: {predictions.shape}")
    print(f"Predictions: {predictions}")

    # Get probabilities
    probabilities = classifier.predict_proba(input_ids, attention_mask)
    print(f"Probabilities: {probabilities}")

    return classifier


def example_ordinal_regression():
    """Example: Movie rating prediction (1-5 stars)."""
    print("\n=== Ordinal Regression Example ===")

    # Create an ordinal classifier for ratings
    classifier = create_ordinal_classifier(
        model_name="mlx-community/answerdotai-ModernBERT-base-4bit",
        num_classes=5,  # 1-5 star ratings
        class_names=["1 star", "2 stars", "3 stars", "4 stars", "5 stars"],
        temperature=1.0,
        hidden_dim=256,
        pooling_type="attention",  # Attention pooling works well for ordinal
        activation="gelu",
        dropout_rate=0.1,
    )

    # Example usage
    reviews = [
        "This movie was absolutely terrible!",
        "Pretty good, but not great",
        "Best movie I've ever seen! Masterpiece!",
    ]

    # Tokenize (pseudo-code)
    input_ids = mx.array(
        [[101, 2023, 2003, 102], [101, 2045, 2003, 102], [101, 2062, 2003, 102]]
    )
    attention_mask = mx.ones_like(input_ids)

    # Get ordinal predictions
    predictions = classifier.predict(input_ids, attention_mask)
    print(f"Rating predictions: {predictions}")

    # Get threshold probabilities
    threshold_probs = classifier.predict_proba(input_ids, attention_mask)
    print(f"Threshold probabilities: {threshold_probs}")

    return classifier


def example_hierarchical_classification():
    """Example: Document categorization with hierarchical taxonomy."""
    print("\n=== Hierarchical Classification Example ===")

    # Define hierarchy
    hierarchy = {
        "science": ["physics", "chemistry", "biology"],
        "physics": ["quantum", "classical", "astrophysics"],
        "chemistry": ["organic", "inorganic"],
        "biology": ["genetics", "ecology"],
        "arts": ["music", "painting", "literature"],
        "music": ["classical_music", "jazz", "rock"],
    }

    # Create label mapping
    all_labels = [
        "science",
        "physics",
        "chemistry",
        "biology",
        "quantum",
        "classical",
        "astrophysics",
        "organic",
        "inorganic",
        "genetics",
        "ecology",
        "arts",
        "music",
        "painting",
        "literature",
        "classical_music",
        "jazz",
        "rock",
    ]
    label_to_idx = {label: idx for idx, label in enumerate(all_labels)}

    # Create hierarchical classifier
    classifier = create_hierarchical_classifier(
        model_name="mlx-community/answerdotai-ModernBERT-base-4bit",
        hierarchy=hierarchy,
        label_to_idx=label_to_idx,
        consistency_weight=1.0,
        hidden_dim=384,
        pooling_type="weighted",
        activation="gelu",
        dropout_rate=0.1,
    )

    # Example usage
    documents = [
        "Quantum entanglement experiments at CERN",
        "Beethoven's influence on modern jazz composition",
        "CRISPR gene editing in agricultural applications",
    ]

    # Tokenize (pseudo-code)
    input_ids = mx.array(
        [[101, 2023, 2003, 102], [101, 2045, 2003, 102], [101, 2062, 2003, 102]]
    )
    attention_mask = mx.ones_like(input_ids)

    # Get hierarchical predictions
    predictions = classifier.predict(input_ids, attention_mask)
    print(f"Hierarchical predictions: {predictions}")

    return classifier


def example_ensemble_classification():
    """Example: High-stakes classification with ensemble of models."""
    print("\n=== Ensemble Classification Example ===")

    # Create ensemble classifier
    classifier = create_ensemble_classifier(
        model_name="mlx-community/answerdotai-ModernBERT-base-4bit",
        num_classes=4,  # e.g., medical diagnosis categories
        num_heads=5,  # 5 different models in ensemble
        ensemble_method="attention",  # Use attention to combine predictions
        hidden_dim=[256, 384, 512, 256, 384],  # Different architectures
        activation="gelu",
        dropout_rate=0.1,
    )

    # Example usage
    medical_texts = [
        "Patient shows elevated temperature and persistent cough",
        "Normal blood work, no significant findings",
        "Irregular heartbeat detected during examination",
    ]

    # Tokenize (pseudo-code)
    input_ids = mx.array(
        [[101, 2023, 2003, 102], [101, 2045, 2003, 102], [101, 2062, 2003, 102]]
    )
    attention_mask = mx.ones_like(input_ids)

    # Get ensemble predictions
    predictions = classifier.predict(input_ids, attention_mask)
    print(f"Ensemble predictions: {predictions}")

    # Get ensemble probabilities (with uncertainty)
    probabilities = classifier.predict_proba(input_ids, attention_mask)
    print(f"Ensemble probabilities: {probabilities}")

    return classifier


def example_multi_task_learning():
    """Example: Multi-task learning with auxiliary heads."""
    print("\n=== Multi-Task Learning Example ===")

    # Define auxiliary tasks
    auxiliary_heads = {
        "sentiment": {
            "task_type": "multiclass",
            "num_classes": 5,  # Very negative to very positive
            "hidden_dim": 256,
            "activation": "gelu",
        },
        "emotion": {
            "task_type": "multiclass",
            "num_classes": 8,  # Joy, sadness, anger, etc.
            "hidden_dim": 256,
            "activation": "relu",
        },
        "toxicity": {
            "task_type": "binary",
            "hidden_dim": 128,
            "activation": "gelu",
        },
    }

    # Create multi-task classifier
    classifier = create_classifier(
        task_type="multiclass",  # Main task: topic classification
        model_name="mlx-community/answerdotai-ModernBERT-base-4bit",
        num_classes=10,  # 10 topics
        hidden_dim=384,
        pooling_type="attention",
        activation="gelu",
        auxiliary_heads=auxiliary_heads,
    )

    # Example usage
    texts = [
        "I absolutely love this new smartphone! Best purchase ever!",
        "The political situation is concerning and makes me worried",
        "Here's how to solve quadratic equations step by step",
    ]

    # Tokenize (pseudo-code)
    input_ids = mx.array(
        [[101, 2023, 2003, 102], [101, 2045, 2003, 102], [101, 2062, 2003, 102]]
    )
    attention_mask = mx.ones_like(input_ids)

    # Get all task predictions
    outputs = classifier.forward(input_ids, attention_mask, compute_auxiliary=True)
    print(f"Multi-task outputs: {outputs.keys()}")

    # Get predictions for all tasks
    all_predictions = classifier.predict(
        input_ids, attention_mask, return_auxiliary=True
    )
    print(f"All predictions: {all_predictions}")

    return classifier


def example_advanced_pooling():
    """Example: Using different pooling strategies."""
    print("\n=== Advanced Pooling Strategies Example ===")

    pooling_types = ["mean", "max", "cls", "attention", "weighted", "learned"]

    for pooling_type in pooling_types:
        print(f"\nUsing {pooling_type} pooling:")

        classifier = create_classifier(
            task_type="multiclass",
            model_name="mlx-community/answerdotai-ModernBERT-base-4bit",
            num_classes=5,
            pooling_type=pooling_type,
            hidden_dim=256,
            activation="gelu",
        )

        # Test input
        input_ids = mx.array([[101, 2023, 2003, 102]])
        attention_mask = mx.ones_like(input_ids)

        # Get embeddings to see pooling effect
        _, pooled = classifier.forward(
            input_ids, attention_mask, return_embeddings=True
        )
        print(f"Pooled shape: {pooled.shape}")


def example_feature_importance():
    """Example: Extracting feature importance from classifier."""
    print("\n=== Feature Importance Example ===")

    # Create classifier
    classifier = create_classifier(
        task_type="binary",
        model_name="mlx-community/answerdotai-ModernBERT-base-4bit",
        num_classes=2,
        hidden_dim=256,
        pooling_type="attention",
    )

    # Example text
    input_ids = mx.array(
        [[101, 2023, 2003, 1037, 2307, 2742, 102]]
    )  # [CLS] This is a great example [SEP]
    attention_mask = mx.ones_like(input_ids)

    # Get feature importance
    importance_scores = classifier.get_feature_importance(
        input_ids, attention_mask, method="gradient"
    )
    print(f"Feature importance shape: {importance_scores.shape}")
    print(f"Importance scores: {importance_scores}")

    # Get integrated gradient importance
    ig_scores = classifier.get_feature_importance(
        input_ids, attention_mask, method="integrated_gradient"
    )
    print(f"Integrated gradient scores: {ig_scores}")


def main():
    """Run all examples."""
    print("Advanced Classification Examples")
    print("=" * 50)

    # Run examples
    example_multilabel_classification()
    example_ordinal_regression()
    example_hierarchical_classification()
    example_ensemble_classification()
    example_multi_task_learning()
    example_advanced_pooling()
    example_feature_importance()

    print("\n" + "=" * 50)
    print("All examples completed successfully!")


if __name__ == "__main__":
    main()
