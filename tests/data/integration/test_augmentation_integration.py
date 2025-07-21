"""Integration tests for augmentation system with datasets and loaders."""

import tempfile
from pathlib import Path

import mlx.core as mx
import pandas as pd
import pytest

from data.augmentation import (
    AugmentationConfig,
    AugmentationManager,
    AugmentationMode,
    CompetitionTemplateAugmenter,
    FeatureMetadata,
    FeatureType,
    TabularAugmenter,
    TabularToTextAugmenter,
    TitanicAugmenter,
)
from data.core import KaggleDataset
from data.factory import create_dataset


class TestAugmentationWithDatasets:
    """Test augmentation integration with datasets."""

    def test_dataset_with_augmentation(self, tmp_path):
        """Test dataset with augmentation enabled."""
        # Create test data
        data = pd.DataFrame({
            "feature1": [1, 2, 3, 4, 5],
            "feature2": ["A", "B", "C", "A", "B"],
            "target": [0, 1, 0, 1, 0],
        })
        
        data_file = tmp_path / "test_data.csv"
        data.to_csv(data_file, index=False)
        
        # Create feature metadata
        feature_metadata = {
            "feature1": FeatureMetadata("feature1", FeatureType.NUMERICAL),
            "feature2": FeatureMetadata("feature2", FeatureType.CATEGORICAL),
        }
        
        # Create augmenter
        config = AugmentationConfig.from_mode(AugmentationMode.LIGHT)
        augmenter = TabularAugmenter(config, feature_metadata, seed=42)
        
        # Create dataset with augmentation
        dataset = create_dataset(
            data_path=str(data_file),
            target_column="target",
            augmenter=augmenter,
        )
        
        # Enable training mode
        dataset.set_training_mode(True)
        
        # Get samples and verify augmentation
        sample1 = dataset[0]
        sample2 = dataset[0]  # Same index but should be augmented differently
        
        # At least some values should differ due to augmentation
        assert "feature1" in sample1
        assert "feature2" in sample1

    def test_dataset_with_text_conversion(self, tmp_path):
        """Test dataset with text conversion augmentation."""
        # Create test data
        data = pd.DataFrame({
            "age": [25, 30, 35],
            "income": [50000, 60000, 70000],
            "category": ["A", "B", "C"],
            "target": [0, 1, 0],
        })
        
        data_file = tmp_path / "test_data.csv"
        data.to_csv(data_file, index=False)
        
        # Create feature metadata
        feature_metadata = {
            "age": FeatureMetadata("age", FeatureType.NUMERICAL, importance=0.9),
            "income": FeatureMetadata("income", FeatureType.NUMERICAL, importance=0.8),
            "category": FeatureMetadata("category", FeatureType.CATEGORICAL),
        }
        
        # Create text augmenter
        augmenter = TabularToTextAugmenter(feature_metadata=feature_metadata)
        
        # Create dataset
        dataset = create_dataset(
            data_path=str(data_file),
            target_column="target",
            text_converter=augmenter,
        )
        
        # Get sample and verify text conversion
        sample = dataset[0]
        
        assert "text" in sample
        assert isinstance(sample["text"], str)
        assert "age" in sample["text"]
        assert "25" in sample["text"]

    def test_titanic_augmenter_integration(self, tmp_path):
        """Test Titanic augmenter with dataset."""
        # Create Titanic-like data
        data = pd.DataFrame({
            "PassengerId": [1, 2, 3],
            "Name": ["Smith, Mr. John", "Johnson, Mrs. Mary", "Williams, Miss. Jane"],
            "Age": [30, 25, 35],
            "Sex": ["male", "female", "female"],
            "Pclass": [1, 2, 3],
            "Fare": [50.0, 30.0, 15.0],
            "Embarked": ["S", "C", "Q"],
            "SibSp": [0, 1, 0],
            "Parch": [0, 1, 0],
            "Survived": [0, 1, 0],
        })
        
        data_file = tmp_path / "titanic_test.csv"
        data.to_csv(data_file, index=False)
        
        # Create dataset with Titanic augmenter
        dataset = create_dataset(
            data_path=str(data_file),
            target_column="Survived",
            competition_name="titanic",
        )
        
        # Get sample and verify Titanic-specific text
        sample = dataset[0]
        
        assert "text" in sample
        text = sample["text"]
        assert "Smith, Mr. John" in text
        assert "first class" in text
        assert "Southampton" in text

    def test_augmentation_manager_with_dataset(self, tmp_path):
        """Test augmentation manager integration."""
        # Create test data
        data = pd.DataFrame({
            "num_feat": [1.0, 2.0, 3.0, 4.0],
            "cat_feat": ["X", "Y", "Z", "X"],
            "text_feat": ["hello world", "test data", "sample text", "more text"],
            "target": [0, 1, 0, 1],
        })
        
        data_file = tmp_path / "manager_test.csv"
        data.to_csv(data_file, index=False)
        
        # Create feature metadata
        feature_metadata = {
            "num_feat": FeatureMetadata(
                "num_feat", 
                FeatureType.NUMERICAL,
                statistics={"mean": 2.5, "std": 1.12}
            ),
            "cat_feat": FeatureMetadata(
                "cat_feat",
                FeatureType.CATEGORICAL,
                domain_info={"values": ["X", "Y", "Z"]}
            ),
            "text_feat": FeatureMetadata("text_feat", FeatureType.TEXT),
        }
        
        # Create augmentation manager
        config = AugmentationConfig.from_mode(AugmentationMode.MODERATE)
        manager = AugmentationManager(feature_metadata, config, seed=42)
        
        # Create dataset with manager's augmenter
        dataset = create_dataset(
            data_path=str(data_file),
            target_column="target",
            augmenter=manager._augmenter,
        )
        
        # Test augmentation stats
        stats = manager.get_augmentation_stats()
        assert stats["total_features"] == 3
        assert stats["augmentation_enabled"] is True

    def test_batch_processing_with_augmentation(self, tmp_path):
        """Test batch processing with augmentation."""
        # Create larger dataset
        n_samples = 100
        data = pd.DataFrame({
            "x": range(n_samples),
            "y": [i % 3 for i in range(n_samples)],
            "label": [i % 2 for i in range(n_samples)],
        })
        
        data_file = tmp_path / "batch_test.csv"
        data.to_csv(data_file, index=False)
        
        # Create augmenter
        feature_metadata = {
            "x": FeatureMetadata("x", FeatureType.NUMERICAL),
            "y": FeatureMetadata("y", FeatureType.CATEGORICAL),
        }
        config = AugmentationConfig.from_mode(AugmentationMode.LIGHT)
        augmenter = TabularAugmenter(config, feature_metadata, seed=42)
        
        # Create dataset
        dataset = create_dataset(
            data_path=str(data_file),
            target_column="label",
            augmenter=augmenter,
        )
        
        # Get batch of samples
        batch_indices = [0, 10, 20, 30]
        batch = dataset.get_batch(batch_indices)
        
        assert len(batch) == len(batch_indices)
        for sample in batch:
            assert "x" in sample
            assert "y" in sample
            assert "label" in sample

    def test_augmentation_modes(self, tmp_path):
        """Test different augmentation modes."""
        # Create test data
        data = pd.DataFrame({
            "value": [100.0] * 10,
            "target": [0, 1] * 5,
        })
        
        data_file = tmp_path / "modes_test.csv"
        data.to_csv(data_file, index=False)
        
        feature_metadata = {
            "value": FeatureMetadata(
                "value",
                FeatureType.NUMERICAL,
                statistics={"mean": 100, "std": 10}
            ),
        }
        
        # Test each mode
        modes = [
            AugmentationMode.NONE,
            AugmentationMode.LIGHT,
            AugmentationMode.MODERATE,
            AugmentationMode.HEAVY,
        ]
        
        results = {}
        for mode in modes:
            config = AugmentationConfig.from_mode(mode)
            augmenter = TabularAugmenter(config, feature_metadata, seed=42)
            
            dataset = create_dataset(
                data_path=str(data_file),
                target_column="target",
                augmenter=augmenter,
            )
            dataset.set_training_mode(True)
            
            # Collect augmented values
            values = [dataset[i]["value"] for i in range(5)]
            results[mode] = values
        
        # NONE mode should not change values
        assert all(v == 100.0 for v in results[AugmentationMode.NONE])
        
        # Other modes should introduce variations
        for mode in [AugmentationMode.LIGHT, AugmentationMode.MODERATE, AugmentationMode.HEAVY]:
            # At least some values should differ from original
            assert not all(v == 100.0 for v in results[mode])

    def test_disable_augmentation_during_evaluation(self, tmp_path):
        """Test that augmentation is disabled during evaluation."""
        # Create test data
        data = pd.DataFrame({
            "feature": [1.0, 2.0, 3.0],
            "target": [0, 1, 0],
        })
        
        data_file = tmp_path / "eval_test.csv"
        data.to_csv(data_file, index=False)
        
        # Create augmenter
        feature_metadata = {
            "feature": FeatureMetadata("feature", FeatureType.NUMERICAL),
        }
        config = AugmentationConfig.from_mode(AugmentationMode.HEAVY)
        augmenter = TabularAugmenter(config, feature_metadata, seed=42)
        
        # Create dataset
        dataset = create_dataset(
            data_path=str(data_file),
            target_column="target",
            augmenter=augmenter,
        )
        
        # Training mode - should augment
        dataset.set_training_mode(True)
        train_values = [dataset[0]["feature"] for _ in range(5)]
        
        # Evaluation mode - should not augment
        dataset.set_training_mode(False)
        eval_values = [dataset[0]["feature"] for _ in range(5)]
        
        # Training values should vary
        assert len(set(train_values)) > 1
        
        # Evaluation values should be constant
        assert len(set(eval_values)) == 1
        assert eval_values[0] == 1.0  # Original value