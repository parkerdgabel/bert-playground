"""Comprehensive tests for all head implementations.

This test suite ensures all head types work correctly for different
Kaggle competition scenarios.
"""

import unittest
import numpy as np
import mlx.core as mx
from typing import Dict

from models.heads.base_head import HeadConfig, HeadType, PoolingType, ActivationType
from models.heads.classification_heads import (
    BinaryClassificationHead,
    MulticlassClassificationHead,
    MultilabelClassificationHead
)
from models.heads.regression_heads import (
    RegressionHead,
    OrdinalRegressionHead,
    TimeSeriesRegressionHead
)
from models.heads.loss_functions import (
    FocalLoss,
    MulticlassFocalLoss,
    MultilabelFocalLoss,
    HuberLoss,
    OrdinalLoss
)


class TestHeadBase(unittest.TestCase):
    """Base test class with common setup."""
    
    def setUp(self):
        """Set up common test fixtures."""
        self.batch_size = 8
        self.seq_length = 32
        self.hidden_size = 128
        
        # Create dummy hidden states
        self.hidden_states = mx.random.normal((self.batch_size, self.seq_length, self.hidden_size))
        self.attention_mask = mx.ones((self.batch_size, self.seq_length))


class TestBinaryClassificationHead(TestHeadBase):
    """Test binary classification head."""
    
    def test_initialization(self):
        """Test head initialization."""
        config = HeadConfig(
            head_type=HeadType.BINARY_CLASSIFICATION,
            input_size=self.hidden_size,
            output_size=2,
            pooling_type=PoolingType.CLS
        )
        head = BinaryClassificationHead(config)
        
        self.assertEqual(head.config.output_size, 2)
        self.assertEqual(head.config.pooling_type, PoolingType.CLS)
    
    def test_forward_pass(self):
        """Test forward pass."""
        config = HeadConfig(
            head_type=HeadType.BINARY_CLASSIFICATION,
            input_size=self.hidden_size,
            output_size=2
        )
        head = BinaryClassificationHead(config)
        
        outputs = head(self.hidden_states, self.attention_mask)
        
        self.assertIn("logits", outputs)
        self.assertIn("probabilities", outputs)
        self.assertEqual(outputs["logits"].shape, (self.batch_size, 2))
        self.assertEqual(outputs["probabilities"].shape, (self.batch_size, 2))
    
    def test_loss_computation(self):
        """Test loss computation."""
        config = HeadConfig(
            head_type=HeadType.BINARY_CLASSIFICATION,
            input_size=self.hidden_size,
            output_size=2
        )
        head = BinaryClassificationHead(config)
        
        outputs = head(self.hidden_states, self.attention_mask)
        labels = mx.random.randint(0, 2, (self.batch_size,))
        
        loss = head.compute_loss(outputs, labels)
        self.assertEqual(loss.shape, ())  # Scalar loss
        self.assertGreater(float(loss), 0)
    
    def test_focal_loss(self):
        """Test focal loss option."""
        config = HeadConfig(
            head_type=HeadType.BINARY_CLASSIFICATION,
            input_size=self.hidden_size,
            output_size=2,
            loss_config={"use_focal_loss": True, "focal_gamma": 2.0}
        )
        head = BinaryClassificationHead(config)
        
        outputs = head(self.hidden_states, self.attention_mask)
        labels = mx.random.randint(0, 2, (self.batch_size,))
        
        loss = head.compute_loss(outputs, labels)
        self.assertIsNotNone(loss)
    
    def test_metrics_computation(self):
        """Test metrics computation."""
        config = HeadConfig(
            head_type=HeadType.BINARY_CLASSIFICATION,
            input_size=self.hidden_size,
            output_size=2
        )
        head = BinaryClassificationHead(config)
        
        outputs = head(self.hidden_states, self.attention_mask)
        labels = mx.random.randint(0, 2, (self.batch_size,))
        
        metrics = head.compute_metrics(outputs, labels)
        
        self.assertIn("accuracy", metrics)
        self.assertIn("precision", metrics)
        self.assertIn("recall", metrics)
        self.assertIn("f1", metrics)
        
        # Check metric ranges
        for metric_name, metric_value in metrics.items():
            self.assertGreaterEqual(metric_value, 0.0)
            self.assertLessEqual(metric_value, 1.0)


class TestMulticlassClassificationHead(TestHeadBase):
    """Test multiclass classification head."""
    
    def test_initialization(self):
        """Test head initialization."""
        num_classes = 5
        config = HeadConfig(
            head_type=HeadType.MULTICLASS_CLASSIFICATION,
            input_size=self.hidden_size,
            output_size=num_classes
        )
        head = MulticlassClassificationHead(config)
        
        self.assertEqual(head.config.output_size, num_classes)
    
    def test_forward_pass(self):
        """Test forward pass."""
        num_classes = 5
        config = HeadConfig(
            head_type=HeadType.MULTICLASS_CLASSIFICATION,
            input_size=self.hidden_size,
            output_size=num_classes
        )
        head = MulticlassClassificationHead(config)
        
        outputs = head(self.hidden_states, self.attention_mask)
        
        self.assertEqual(outputs["logits"].shape, (self.batch_size, num_classes))
        self.assertEqual(outputs["probabilities"].shape, (self.batch_size, num_classes))
        
        # Check probabilities sum to 1
        prob_sums = outputs["probabilities"].sum(axis=1)
        mx.testing.assert_allclose(prob_sums, mx.ones_like(prob_sums), rtol=1e-5)
    
    def test_label_smoothing(self):
        """Test label smoothing."""
        config = HeadConfig(
            head_type=HeadType.MULTICLASS_CLASSIFICATION,
            input_size=self.hidden_size,
            output_size=5,
            loss_config={"label_smoothing": 0.1}
        )
        head = MulticlassClassificationHead(config)
        
        outputs = head(self.hidden_states, self.attention_mask)
        labels = mx.random.randint(0, 5, (self.batch_size,))
        
        loss = head.compute_loss(outputs, labels)
        self.assertIsNotNone(loss)
    
    def test_top_k_accuracy(self):
        """Test top-k accuracy metric."""
        num_classes = 10
        config = HeadConfig(
            head_type=HeadType.MULTICLASS_CLASSIFICATION,
            input_size=self.hidden_size,
            output_size=num_classes
        )
        head = MulticlassClassificationHead(config)
        
        outputs = head(self.hidden_states, self.attention_mask)
        labels = mx.random.randint(0, num_classes, (self.batch_size,))
        
        metrics = head.compute_metrics(outputs, labels)
        
        self.assertIn("top_3_accuracy", metrics)
        self.assertGreaterEqual(metrics["top_3_accuracy"], metrics["accuracy"])


class TestMultilabelClassificationHead(TestHeadBase):
    """Test multilabel classification head."""
    
    def test_initialization(self):
        """Test head initialization."""
        num_labels = 10
        config = HeadConfig(
            head_type=HeadType.MULTILABEL_CLASSIFICATION,
            input_size=self.hidden_size,
            output_size=num_labels
        )
        head = MultilabelClassificationHead(config)
        
        self.assertEqual(head.config.output_size, num_labels)
    
    def test_forward_pass(self):
        """Test forward pass."""
        num_labels = 10
        config = HeadConfig(
            head_type=HeadType.MULTILABEL_CLASSIFICATION,
            input_size=self.hidden_size,
            output_size=num_labels
        )
        head = MultilabelClassificationHead(config)
        
        outputs = head(self.hidden_states, self.attention_mask)
        
        self.assertEqual(outputs["logits"].shape, (self.batch_size, num_labels))
        self.assertEqual(outputs["probabilities"].shape, (self.batch_size, num_labels))
        
        # Check probabilities are in [0, 1] range (sigmoid output)
        self.assertTrue(mx.all(outputs["probabilities"] >= 0))
        self.assertTrue(mx.all(outputs["probabilities"] <= 1))
    
    def test_adaptive_thresholds(self):
        """Test adaptive thresholds."""
        config = HeadConfig(
            head_type=HeadType.MULTILABEL_CLASSIFICATION,
            input_size=self.hidden_size,
            output_size=10,
            loss_config={"use_adaptive_thresholds": True}
        )
        head = MultilabelClassificationHead(config)
        
        outputs = head(self.hidden_states, self.attention_mask)
        
        self.assertIn("thresholds", outputs)
        self.assertEqual(outputs["thresholds"].shape, (10,))
    
    def test_multilabel_metrics(self):
        """Test multilabel-specific metrics."""
        num_labels = 5
        config = HeadConfig(
            head_type=HeadType.MULTILABEL_CLASSIFICATION,
            input_size=self.hidden_size,
            output_size=num_labels
        )
        head = MultilabelClassificationHead(config)
        
        outputs = head(self.hidden_states, self.attention_mask)
        # Create multilabel targets
        labels = mx.random.uniform(shape=(self.batch_size, num_labels)) > 0.5
        labels = labels.astype(mx.float32)
        
        metrics = head.compute_metrics(outputs, labels)
        
        self.assertIn("hamming_loss", metrics)
        self.assertIn("subset_accuracy", metrics)
        self.assertIn("macro_f1", metrics)
        self.assertIn("micro_f1", metrics)


class TestRegressionHead(TestHeadBase):
    """Test regression head."""
    
    def test_initialization(self):
        """Test head initialization."""
        config = HeadConfig(
            head_type=HeadType.REGRESSION,
            input_size=self.hidden_size,
            output_size=1
        )
        head = RegressionHead(config)
        
        self.assertEqual(head.config.output_size, 1)
    
    def test_forward_pass(self):
        """Test forward pass."""
        config = HeadConfig(
            head_type=HeadType.REGRESSION,
            input_size=self.hidden_size,
            output_size=1
        )
        head = RegressionHead(config)
        
        outputs = head(self.hidden_states, self.attention_mask)
        
        self.assertIn("predictions", outputs)
        self.assertEqual(outputs["predictions"].shape, (self.batch_size, 1))
    
    def test_uncertainty_estimation(self):
        """Test uncertainty estimation."""
        config = HeadConfig(
            head_type=HeadType.REGRESSION,
            input_size=self.hidden_size,
            output_size=1,
            loss_config={"estimate_uncertainty": True}
        )
        head = RegressionHead(config)
        
        outputs = head(self.hidden_states, self.attention_mask)
        
        self.assertIn("uncertainty", outputs)
        self.assertEqual(outputs["uncertainty"].shape, (self.batch_size, 1))
        self.assertTrue(mx.all(outputs["uncertainty"] > 0))
    
    def test_loss_functions(self):
        """Test different loss functions."""
        for loss_type in ["mse", "mae", "huber"]:
            config = HeadConfig(
                head_type=HeadType.REGRESSION,
                input_size=self.hidden_size,
                output_size=1,
                loss_config={"loss_type": loss_type}
            )
            head = RegressionHead(config)
            
            outputs = head(self.hidden_states, self.attention_mask)
            labels = mx.random.normal((self.batch_size, 1))
            
            loss = head.compute_loss(outputs, labels)
            self.assertGreater(float(loss), 0)
    
    def test_regression_metrics(self):
        """Test regression metrics."""
        config = HeadConfig(
            head_type=HeadType.REGRESSION,
            input_size=self.hidden_size,
            output_size=1
        )
        head = RegressionHead(config)
        
        outputs = head(self.hidden_states, self.attention_mask)
        labels = mx.random.normal((self.batch_size, 1))
        
        metrics = head.compute_metrics(outputs, labels)
        
        self.assertIn("mse", metrics)
        self.assertIn("rmse", metrics)
        self.assertIn("mae", metrics)
        self.assertIn("r2", metrics)
        
        # Check RMSE = sqrt(MSE)
        self.assertAlmostEqual(
            metrics["rmse"],
            np.sqrt(metrics["mse"]),
            places=5
        )


class TestOrdinalRegressionHead(TestHeadBase):
    """Test ordinal regression head."""
    
    def test_initialization(self):
        """Test head initialization."""
        num_classes = 5
        config = HeadConfig(
            head_type=HeadType.ORDINAL_REGRESSION,
            input_size=self.hidden_size,
            output_size=num_classes
        )
        head = OrdinalRegressionHead(config)
        
        self.assertEqual(head.config.output_size, num_classes)
    
    def test_forward_pass(self):
        """Test forward pass."""
        num_classes = 5
        config = HeadConfig(
            head_type=HeadType.ORDINAL_REGRESSION,
            input_size=self.hidden_size,
            output_size=num_classes
        )
        head = OrdinalRegressionHead(config)
        
        outputs = head(self.hidden_states, self.attention_mask)
        
        self.assertIn("cumulative_logits", outputs)
        self.assertIn("probabilities", outputs)
        self.assertIn("predictions", outputs)
        
        # Check cumulative logits shape
        self.assertEqual(outputs["cumulative_logits"].shape, (self.batch_size, num_classes - 1))
        self.assertEqual(outputs["probabilities"].shape, (self.batch_size, num_classes))
    
    def test_ordinal_metrics(self):
        """Test ordinal-specific metrics."""
        num_classes = 5
        config = HeadConfig(
            head_type=HeadType.ORDINAL_REGRESSION,
            input_size=self.hidden_size,
            output_size=num_classes
        )
        head = OrdinalRegressionHead(config)
        
        outputs = head(self.hidden_states, self.attention_mask)
        labels = mx.random.randint(0, num_classes, (self.batch_size,))
        
        metrics = head.compute_metrics(outputs, labels)
        
        self.assertIn("accuracy", metrics)
        self.assertIn("mae", metrics)
        self.assertIn("ordinal_accuracy", metrics)
        self.assertIn("kendalls_tau", metrics)


class TestTimeSeriesRegressionHead(TestHeadBase):
    """Test time series regression head."""
    
    def test_initialization(self):
        """Test head initialization."""
        config = HeadConfig(
            head_type=HeadType.TIME_SERIES_REGRESSION,
            input_size=self.hidden_size,
            output_size=1,
            loss_config={"prediction_horizon": 5}
        )
        head = TimeSeriesRegressionHead(config)
        
        self.assertEqual(head.config.output_size, 1)
        self.assertEqual(head.prediction_horizon, 5)
    
    def test_multi_step_prediction(self):
        """Test multi-step ahead prediction."""
        horizon = 5
        config = HeadConfig(
            head_type=HeadType.TIME_SERIES_REGRESSION,
            input_size=self.hidden_size,
            output_size=1,
            loss_config={"prediction_horizon": horizon}
        )
        head = TimeSeriesRegressionHead(config)
        
        outputs = head(self.hidden_states, self.attention_mask)
        
        self.assertIn("predictions", outputs)
        self.assertEqual(outputs["predictions"].shape, (self.batch_size, horizon, 1))
    
    def test_temporal_features(self):
        """Test temporal feature extraction."""
        config = HeadConfig(
            head_type=HeadType.TIME_SERIES_REGRESSION,
            input_size=self.hidden_size,
            output_size=1,
            loss_config={
                "use_temporal_features": True,
                "prediction_horizon": 3
            }
        )
        head = TimeSeriesRegressionHead(config)
        
        outputs = head(self.hidden_states, self.attention_mask)
        
        self.assertIn("temporal_features", outputs)
    
    def test_time_series_metrics(self):
        """Test time series specific metrics."""
        horizon = 3
        config = HeadConfig(
            head_type=HeadType.TIME_SERIES_REGRESSION,
            input_size=self.hidden_size,
            output_size=1,
            loss_config={"prediction_horizon": horizon}
        )
        head = TimeSeriesRegressionHead(config)
        
        outputs = head(self.hidden_states, self.attention_mask)
        labels = mx.random.normal((self.batch_size, horizon, 1))
        
        metrics = head.compute_metrics(outputs, labels)
        
        self.assertIn("mse", metrics)
        self.assertIn("mae", metrics)
        self.assertIn("mape", metrics)
        self.assertIn("directional_accuracy", metrics)


class TestLossFunctions(unittest.TestCase):
    """Test custom loss functions."""
    
    def test_focal_loss(self):
        """Test focal loss for binary classification."""
        loss_fn = FocalLoss(gamma=2.0, alpha=0.25)
        
        batch_size = 8
        logits = mx.random.normal((batch_size, 2))
        targets = mx.random.randint(0, 2, (batch_size,))
        
        loss = loss_fn(logits, targets)
        self.assertEqual(loss.shape, ())
        self.assertGreater(float(loss), 0)
    
    def test_huber_loss(self):
        """Test Huber loss for regression."""
        loss_fn = HuberLoss(delta=1.0)
        
        batch_size = 8
        predictions = mx.random.normal((batch_size, 1))
        targets = mx.random.normal((batch_size, 1))
        
        loss = loss_fn(predictions, targets)
        self.assertEqual(loss.shape, ())
        self.assertGreater(float(loss), 0)
    
    def test_ordinal_loss(self):
        """Test ordinal loss."""
        num_classes = 5
        loss_fn = OrdinalLoss(num_classes=num_classes)
        
        batch_size = 8
        cumulative_logits = mx.random.normal((batch_size, num_classes - 1))
        targets = mx.random.randint(0, num_classes, (batch_size,))
        
        loss = loss_fn(cumulative_logits, targets)
        self.assertEqual(loss.shape, ())
        self.assertGreater(float(loss), 0)


if __name__ == "__main__":
    unittest.main()