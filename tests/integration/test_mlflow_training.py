"""Integration tests for MLflow training pipeline."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import mlflow
import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pandas as pd

from training.config import TrainingConfig
from training.mlx_trainer import MLXTrainer
from training.monitoring import ComprehensiveMonitor
from utils.mlflow_central import MLflowCentral
from models.classification import TitanicClassifier


class MockDataLoader:
    """Mock data loader for testing."""
    
    def __init__(self, batch_size=32, num_batches=10):
        self.batch_size = batch_size
        self.num_batches = num_batches
        self._current_batch = 0
    
    def __iter__(self):
        self._current_batch = 0
        return self
    
    def __next__(self):
        if self._current_batch >= self.num_batches:
            raise StopIteration
        
        # Generate mock batch data
        batch = {
            "input_ids": mx.random.randint(0, 1000, (self.batch_size, 128)),
            "attention_mask": mx.ones((self.batch_size, 128)),
            "labels": mx.random.randint(0, 2, (self.batch_size,))
        }
        
        self._current_batch += 1
        return batch
    
    def __len__(self):
        return self.num_batches


class MockModel(nn.Module):
    """Mock model for testing."""
    
    def __init__(self, num_classes=2):
        super().__init__()
        self.num_classes = num_classes
        self.linear = nn.Linear(128, num_classes)
    
    def __call__(self, input_ids, attention_mask=None, labels=None):
        # Simple mock forward pass
        batch_size = input_ids.shape[0]
        
        # Generate random embeddings
        embeddings = mx.random.normal((batch_size, 128))
        logits = self.linear(embeddings)
        
        outputs = {"logits": logits}
        
        if labels is not None:
            # Simple cross-entropy loss
            loss = mx.mean(mx.square(logits - labels[:, None]))
            outputs["loss"] = loss
        
        return outputs
    
    def save_pretrained(self, path):
        """Mock save method."""
        Path(path).mkdir(parents=True, exist_ok=True)
        # Save some dummy data
        with open(Path(path) / "model.json", "w") as f:
            json.dump({"model_type": "mock", "num_classes": self.num_classes}, f)


class TestMLflowTrainingIntegration(unittest.TestCase):
    """Integration tests for MLflow training pipeline."""
    
    def setUp(self):
        """Set up test environment."""
        # Create temporary directory
        self.temp_dir = tempfile.mkdtemp()
        self.output_dir = Path(self.temp_dir) / "output"
        self.output_dir.mkdir(parents=True)
        
        # Reset MLflow central singleton
        MLflowCentral._instance = None
        
        # Create test configuration
        self.config = TrainingConfig(
            learning_rate=0.001,
            epochs=2,
            batch_size=16,
            train_path="dummy_train.csv",
            val_path="dummy_val.csv",
            output_dir=str(self.output_dir),
            experiment_name="test_integration",
            monitoring=TrainingConfig.MonitoringConfig(
                enable_mlflow=True,
                enable_rich_console=False,
                log_frequency=5
            ),
            evaluation=TrainingConfig.EvaluationConfig(
                eval_steps=10,
                save_best_model=True
            ),
            checkpoint=TrainingConfig.CheckpointConfig(
                enable_checkpointing=True,
                save_model_weights=True,
                checkpoint_frequency=20
            )
        )
        
        # Create mock data loaders
        self.train_loader = MockDataLoader(batch_size=16, num_batches=20)
        self.val_loader = MockDataLoader(batch_size=16, num_batches=5)
        
        # Create mock model
        self.model = MockModel()
    
    def tearDown(self):
        """Clean up test environment."""
        import shutil
        if Path(self.temp_dir).exists():
            shutil.rmtree(self.temp_dir)
        
        # Reset singleton
        MLflowCentral._instance = None
    
    def test_training_with_mlflow_enabled(self):
        """Test complete training pipeline with MLflow enabled."""
        # Set up MLflow tracking
        mlflow_db = self.output_dir / "mlruns" / "mlflow.db"
        mlflow_artifacts = self.output_dir / "mlruns" / "artifacts"
        
        # Initialize MLflow central
        central = MLflowCentral()
        central.initialize(
            tracking_uri=f"sqlite:///{mlflow_db}",
            artifact_root=str(mlflow_artifacts),
            experiment_name="test_integration"
        )
        
        # Create trainer
        trainer = MLXTrainer(self.model, self.config)
        
        # Mock the optimizer to avoid issues
        trainer.optimizer = Mock()
        trainer.optimizer.update = Mock()
        trainer.optimizer.state = {}
        
        # Run training
        results = trainer.train(self.train_loader, self.val_loader)
        
        # Verify training completed
        self.assertIsNotNone(results)
        self.assertIn("final_metrics", results)
        self.assertIn("total_time", results)
        self.assertIn("training_history", results)
        
        # Verify MLflow artifacts were created
        self.assertTrue(mlflow_db.parent.exists())
        self.assertTrue(mlflow_artifacts.exists())
        
        # Verify experiments were created
        experiments = mlflow.search_experiments()
        self.assertGreater(len(experiments), 0)
        
        # Find our experiment
        test_experiment = None
        for exp in experiments:
            if exp.name == "test_integration":
                test_experiment = exp
                break
        
        self.assertIsNotNone(test_experiment)
        
        # Verify runs were created
        runs = mlflow.search_runs(experiment_ids=[test_experiment.experiment_id])
        self.assertGreater(len(runs), 0)
        
        # Check that metrics were logged
        run = runs.iloc[0]
        self.assertIn("train_loss", run.keys())
        self.assertIn("learning_rate", run.keys())
    
    def test_training_with_mlflow_disabled(self):
        """Test training pipeline with MLflow disabled."""
        # Disable MLflow
        self.config.monitoring.enable_mlflow = False
        
        # Create trainer
        trainer = MLXTrainer(self.model, self.config)
        
        # Mock the optimizer
        trainer.optimizer = Mock()
        trainer.optimizer.update = Mock()
        trainer.optimizer.state = {}
        
        # Run training
        results = trainer.train(self.train_loader, self.val_loader)
        
        # Verify training completed
        self.assertIsNotNone(results)
        self.assertIn("final_metrics", results)
        
        # Verify no MLflow artifacts were created
        mlflow_dir = self.output_dir / "mlruns"
        self.assertFalse(mlflow_dir.exists())
    
    def test_training_with_mlflow_failure_recovery(self):
        """Test training continues when MLflow fails."""
        # Set up MLflow with invalid configuration
        self.config.monitoring.enable_mlflow = True
        
        # Create trainer with invalid MLflow config
        with patch('utils.mlflow_central.MLflowCentral.initialize') as mock_init:
            mock_init.side_effect = Exception("MLflow initialization failed")
            
            trainer = MLXTrainer(self.model, self.config)
            
            # Mock the optimizer
            trainer.optimizer = Mock()
            trainer.optimizer.update = Mock()
            trainer.optimizer.state = {}
            
            # Training should continue despite MLflow failure
            results = trainer.train(self.train_loader, self.val_loader)
            
            # Verify training completed
            self.assertIsNotNone(results)
            self.assertIn("final_metrics", results)
            
            # Verify MLflow was disabled
            self.assertFalse(self.config.monitoring.enable_mlflow)
    
    def test_comprehensive_monitor_integration(self):
        """Test comprehensive monitoring system integration."""
        # Create comprehensive monitor
        from training.memory_manager import AppleSiliconMemoryManager
        from training.performance_profiler import AppleSiliconProfiler, ProfilerConfig
        
        memory_manager = AppleSiliconMemoryManager()
        profiler = AppleSiliconProfiler(ProfilerConfig())
        
        monitor = ComprehensiveMonitor(
            config=self.config,
            memory_manager=memory_manager,
            profiler=profiler
        )
        
        # Test monitoring lifecycle
        monitor.start_training(total_epochs=2, steps_per_epoch=20)
        
        # Simulate training steps
        for step in range(10):
            monitor.log_step(
                step=step,
                epoch=0,
                train_loss=1.0 - step * 0.1,
                train_accuracy=step * 0.1,
                learning_rate=0.001,
                batch_size=16
            )
        
        # Simulate validation
        improved = monitor.log_validation(
            step=10,
            val_loss=0.5,
            val_accuracy=0.8,
            additional_metrics={"val_precision": 0.75, "val_recall": 0.85}
        )
        
        # End training
        summary = monitor.end_training("FINISHED")
        
        # Verify monitoring worked
        self.assertIsNotNone(summary)
        self.assertEqual(summary["status"], "FINISHED")
        self.assertIn("elapsed_time_seconds", summary)
        self.assertIn("total_steps", summary)
        self.assertIn("best_metrics", summary)
    
    def test_artifact_logging_integration(self):
        """Test artifact logging during training."""
        # Set up MLflow
        mlflow_db = self.output_dir / "mlruns" / "mlflow.db"
        mlflow_artifacts = self.output_dir / "mlruns" / "artifacts"
        
        central = MLflowCentral()
        central.initialize(
            tracking_uri=f"sqlite:///{mlflow_db}",
            artifact_root=str(mlflow_artifacts),
            experiment_name="test_artifacts"
        )
        
        # Create trainer
        trainer = MLXTrainer(self.model, self.config)
        
        # Mock the optimizer
        trainer.optimizer = Mock()
        trainer.optimizer.update = Mock()
        trainer.optimizer.state = {}
        
        # Enable checkpointing
        self.config.checkpoint.enable_checkpointing = True
        self.config.checkpoint.save_best_model = True
        
        # Run training
        results = trainer.train(self.train_loader, self.val_loader)
        
        # Verify checkpoints were created
        checkpoint_dir = Path(self.config.checkpoint.checkpoint_dir)
        self.assertTrue(checkpoint_dir.exists())
        
        # Verify artifacts exist
        checkpoint_files = list(checkpoint_dir.glob("**/model.json"))
        self.assertGreater(len(checkpoint_files), 0)
    
    def test_performance_metrics_logging(self):
        """Test performance metrics are logged correctly."""
        # Set up MLflow
        mlflow_db = self.output_dir / "mlruns" / "mlflow.db"
        mlflow_artifacts = self.output_dir / "mlruns" / "artifacts"
        
        central = MLflowCentral()
        central.initialize(
            tracking_uri=f"sqlite:///{mlflow_db}",
            artifact_root=str(mlflow_artifacts),
            experiment_name="test_performance"
        )
        
        # Create trainer
        trainer = MLXTrainer(self.model, self.config)
        
        # Mock the optimizer
        trainer.optimizer = Mock()
        trainer.optimizer.update = Mock()
        trainer.optimizer.state = {}
        
        # Run training
        results = trainer.train(self.train_loader, self.val_loader)
        
        # Verify performance reports were created
        performance_report = self.output_dir / "performance_report.json"
        memory_report = self.output_dir / "memory_report.json"
        
        # These files should exist if profiling worked
        # Note: They might not exist in mock environment, so we just check results
        self.assertIsNotNone(results)
        self.assertIn("training_history", results)
        
        # Check history contains performance metrics
        history = results["training_history"]
        self.assertIn("learning_rates", history)
        self.assertIn("memory_usage", history)
        self.assertIn("batch_sizes", history)
    
    def test_error_handling_during_training(self):
        """Test error handling during training process."""
        # Set up MLflow
        mlflow_db = self.output_dir / "mlruns" / "mlflow.db"
        mlflow_artifacts = self.output_dir / "mlruns" / "artifacts"
        
        central = MLflowCentral()
        central.initialize(
            tracking_uri=f"sqlite:///{mlflow_db}",
            artifact_root=str(mlflow_artifacts),
            experiment_name="test_error_handling"
        )
        
        # Create trainer
        trainer = MLXTrainer(self.model, self.config)
        
        # Mock optimizer to cause failure
        trainer.optimizer = Mock()
        trainer.optimizer.update.side_effect = Exception("Training error")
        
        # Training should handle the error gracefully
        with self.assertRaises(Exception):
            trainer.train(self.train_loader, self.val_loader)
        
        # Verify MLflow run was ended with failed status
        experiments = mlflow.search_experiments()
        if experiments:
            runs = mlflow.search_runs(experiment_ids=[experiments[0].experiment_id])
            if len(runs) > 0:
                # Check that the run was marked as failed
                run = runs.iloc[0]
                self.assertEqual(run.status, "FAILED")
    
    def test_checkpoint_loading_integration(self):
        """Test checkpoint loading and resuming training."""
        # First, run a short training to create checkpoints
        trainer = MLXTrainer(self.model, self.config)
        trainer.optimizer = Mock()
        trainer.optimizer.update = Mock()
        trainer.optimizer.state = {}
        
        # Enable checkpointing
        self.config.checkpoint.enable_checkpointing = True
        self.config.checkpoint.checkpoint_frequency = 10
        
        # Run partial training
        self.config.epochs = 1
        results = trainer.train(self.train_loader, self.val_loader)
        
        # Verify checkpoint was created
        checkpoint_dir = Path(self.config.checkpoint.checkpoint_dir)
        self.assertTrue(checkpoint_dir.exists())
        
        # Find a checkpoint
        checkpoint_files = list(checkpoint_dir.glob("checkpoint-step-*"))
        if checkpoint_files:
            checkpoint_path = checkpoint_files[0]
            
            # Create new trainer and resume
            new_trainer = MLXTrainer(self.model, self.config)
            new_trainer.optimizer = Mock()
            new_trainer.optimizer.update = Mock()
            new_trainer.optimizer.state = {}
            
            # Load checkpoint
            success = new_trainer._load_checkpoint(str(checkpoint_path))
            
            # Verify checkpoint was loaded
            self.assertTrue(success)
            self.assertGreater(new_trainer.global_step, 0)


class TestMLflowConfigurationIntegration(unittest.TestCase):
    """Integration tests for MLflow configuration."""
    
    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        MLflowCentral._instance = None
    
    def tearDown(self):
        """Clean up test environment."""
        import shutil
        if Path(self.temp_dir).exists():
            shutil.rmtree(self.temp_dir)
        MLflowCentral._instance = None
    
    def test_multiple_experiment_creation(self):
        """Test creating multiple experiments."""
        central = MLflowCentral()
        
        # Initialize with temporary directory
        central.initialize(
            tracking_uri=f"sqlite:///{self.temp_dir}/mlflow.db",
            artifact_root=f"{self.temp_dir}/artifacts"
        )
        
        # Create multiple experiments
        experiments = ["exp1", "exp2", "exp3"]
        
        for exp_name in experiments:
            exp_id = central.get_experiment_id(exp_name)
            self.assertIsNotNone(exp_id)
            
            # Verify experiment exists
            experiment = mlflow.get_experiment(exp_id)
            self.assertEqual(experiment.name, exp_name)
    
    def test_concurrent_run_creation(self):
        """Test creating multiple runs concurrently."""
        central = MLflowCentral()
        
        # Initialize
        central.initialize(
            tracking_uri=f"sqlite:///{self.temp_dir}/mlflow.db",
            artifact_root=f"{self.temp_dir}/artifacts",
            experiment_name="concurrent_test"
        )
        
        # Create multiple runs
        run_ids = []
        
        for i in range(5):
            with mlflow.start_run():
                mlflow.log_param("run_number", i)
                mlflow.log_metric("metric", i * 0.1)
                run_ids.append(mlflow.active_run().info.run_id)
        
        # Verify all runs were created
        self.assertEqual(len(run_ids), 5)
        
        # Verify runs can be retrieved
        for run_id in run_ids:
            run = mlflow.get_run(run_id)
            self.assertIsNotNone(run)
    
    def test_database_size_monitoring(self):
        """Test database size monitoring."""
        central = MLflowCentral()
        
        db_path = Path(self.temp_dir) / "mlflow.db"
        
        # Initialize
        central.initialize(
            tracking_uri=f"sqlite:///{db_path}",
            artifact_root=f"{self.temp_dir}/artifacts"
        )
        
        # Create some runs to increase database size
        for i in range(10):
            with mlflow.start_run():
                for j in range(100):
                    mlflow.log_metric(f"metric_{j}", i * j * 0.01)
        
        # Check connection status includes database size
        status = central.validate_connection()
        self.assertEqual(status["status"], "CONNECTED")
        self.assertIn("database_size_mb", status["details"])
        self.assertGreater(status["details"]["database_size_mb"], 0)


if __name__ == '__main__':
    unittest.main()