"""Integration tests for the complete training pipeline."""

import json
import time
from pathlib import Path
from unittest.mock import patch

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import pytest

from data.unified_loader import UnifiedTitanicDataPipeline, create_unified_dataloaders
from models.classification import TitanicClassifier
from models.factory import create_model
from models.modernbert_cnn_hybrid import create_cnn_hybrid_model
from training.mlx_trainer import MLXTrainer, UnifiedTrainingConfig
from utils.mlflow_central import setup_central_mlflow


@pytest.mark.integration
@pytest.mark.mlx
class TestTrainingPipelineIntegration:
    """Integration tests for training pipeline."""
    
    def test_minimal_training_run(self, sample_titanic_data, temp_dir):
        """Test minimal training run with basic configuration."""
        # Create config
        config = UnifiedTrainingConfig(
            num_epochs=1,
            base_batch_size=4,
            eval_steps=5,
            save_steps=5,
            enable_mlflow=False,
            enable_profiling=False,
            enable_visualization=False,
            output_dir=str(temp_dir / "output"),
            early_stopping_patience=0,
        )
        
        # Create model and trainer
        model = TitanicClassifier(create_model("standard"))
        trainer = MLXTrainer(model=model, config=config)
        
        # Create data loader
        train_loader = UnifiedTitanicDataPipeline(
            data_path=str(sample_titanic_data),
            tokenizer_name="answerdotai/ModernBERT-base",
            batch_size=4,
            max_length=128,
            is_training=True,
        )
        
        # Train
        results = trainer.train(train_dataloader=train_loader)
        
        # Verify results
        assert "best_metric" in results
        assert "total_time" in results
        assert results["total_time"] > 0
        assert "history" in results
        
        # Check checkpoint was saved
        checkpoint_dir = Path(config.checkpoint_dir)
        assert checkpoint_dir.exists()
        assert len(list(checkpoint_dir.glob("checkpoint-*"))) > 0
    
    def test_training_with_validation(self, sample_titanic_data, temp_dir):
        """Test training with validation set."""
        # Create config
        config = UnifiedTrainingConfig(
            num_epochs=1,
            base_batch_size=4,
            eval_steps=2,
            save_steps=5,
            enable_mlflow=False,
            output_dir=str(temp_dir / "output"),
        )
        
        # Create model and trainer
        model = TitanicClassifier(create_model("standard"))
        trainer = MLXTrainer(model=model, config=config)
        
        # Create data loaders
        train_loader, val_loader = create_unified_dataloaders(
            train_path=str(sample_titanic_data),
            val_path=str(sample_titanic_data),  # Use same for testing
            tokenizer_name="answerdotai/ModernBERT-base",
            batch_size=4,
            max_length=128,
        )
        
        # Train
        results = trainer.train(
            train_dataloader=train_loader,
            val_dataloader=val_loader,
        )
        
        # Verify validation metrics
        assert "final_metrics" in results
        assert "val_accuracy" in results["final_metrics"]
        assert "val_loss" in results["final_metrics"]
        assert results["final_metrics"]["val_accuracy"] >= 0
    
    def test_training_with_early_stopping(self, sample_titanic_data, temp_dir):
        """Test early stopping functionality."""
        # Create config with early stopping
        config = UnifiedTrainingConfig(
            num_epochs=10,  # High number to test early stopping
            base_batch_size=4,
            eval_steps=2,
            early_stopping_patience=2,
            early_stopping_threshold=0.001,
            enable_mlflow=False,
            output_dir=str(temp_dir / "output"),
        )
        
        # Create model and trainer
        model = TitanicClassifier(create_model("standard"))
        trainer = MLXTrainer(model=model, config=config)
        
        # Create data loader
        train_loader = UnifiedTitanicDataPipeline(
            data_path=str(sample_titanic_data),
            tokenizer_name="answerdotai/ModernBERT-base",
            batch_size=4,
            max_length=128,
        )
        
        # Mock evaluate to return consistent metrics (no improvement)
        original_evaluate = trainer.evaluate
        
        def mock_evaluate(dataloader, phase="val", max_batches=None):
            return {f"{phase}_accuracy": 0.5, f"{phase}_loss": 0.7}
        
        trainer.evaluate = mock_evaluate
        
        # Train
        results = trainer.train(
            train_dataloader=train_loader,
            val_dataloader=train_loader,
        )
        
        # Should stop early, not complete all epochs
        assert trainer.current_epoch < config.num_epochs
    
    def test_training_with_gradient_accumulation(self, sample_titanic_data, temp_dir):
        """Test gradient accumulation."""
        # Create config with gradient accumulation
        config = UnifiedTrainingConfig(
            num_epochs=1,
            base_batch_size=2,
            gradient_accumulation_steps=2,
            eval_steps=10,
            enable_mlflow=False,
            output_dir=str(temp_dir / "output"),
        )
        
        # Create model and trainer
        model = TitanicClassifier(create_model("standard"))
        trainer = MLXTrainer(model=model, config=config)
        
        # Create data loader
        train_loader = UnifiedTitanicDataPipeline(
            data_path=str(sample_titanic_data),
            tokenizer_name="answerdotai/ModernBERT-base",
            batch_size=2,
            max_length=128,
        )
        
        # Train
        results = trainer.train(train_dataloader=train_loader)
        
        # Verify training completed
        assert results["total_time"] > 0
    
    def test_training_with_dynamic_batching(self, sample_titanic_data, temp_dir):
        """Test dynamic batch size adjustment."""
        # Create config with dynamic batching
        config = UnifiedTrainingConfig(
            num_epochs=1,
            base_batch_size=4,
            max_batch_size=8,
            enable_dynamic_batching=True,
            eval_steps=5,
            enable_mlflow=False,
            output_dir=str(temp_dir / "output"),
        )
        
        # Create model and trainer
        model = TitanicClassifier(create_model("standard"))
        trainer = MLXTrainer(model=model, config=config)
        
        # Mock memory usage to trigger batch size changes
        def mock_memory_usage():
            # Return low memory usage to trigger increase
            return 0.3
        
        trainer.get_memory_usage = mock_memory_usage
        
        # Create data loader
        train_loader = UnifiedTitanicDataPipeline(
            data_path=str(sample_titanic_data),
            tokenizer_name="answerdotai/ModernBERT-base",
            batch_size=4,
            max_length=128,
        )
        
        # Train
        results = trainer.train(train_dataloader=train_loader)
        
        # Batch size should have increased
        assert trainer.current_batch_size > config.base_batch_size
    
    @pytest.mark.mlflow
    def test_training_with_mlflow(self, sample_titanic_data, temp_dir):
        """Test training with MLflow tracking."""
        # Create config with MLflow enabled
        config = UnifiedTrainingConfig(
            num_epochs=1,
            base_batch_size=4,
            eval_steps=5,
            enable_mlflow=True,
            experiment_name="test_experiment",
            output_dir=str(temp_dir / "output"),
            tracking_uri=f"sqlite:///{temp_dir}/mlflow.db",
        )
        
        # Create model and trainer
        model = TitanicClassifier(create_model("standard"))
        trainer = MLXTrainer(model=model, config=config)
        
        # Create data loader
        train_loader = UnifiedTitanicDataPipeline(
            data_path=str(sample_titanic_data),
            tokenizer_name="answerdotai/ModernBERT-base",
            batch_size=4,
            max_length=128,
        )
        
        # Train
        results = trainer.train(train_dataloader=train_loader)
        
        # Verify MLflow artifacts
        assert (temp_dir / "mlflow.db").exists()
    
    def test_cnn_hybrid_model_training(self, sample_titanic_data, temp_dir):
        """Test training with CNN hybrid model."""
        # Create config
        config = UnifiedTrainingConfig(
            num_epochs=1,
            base_batch_size=4,
            eval_steps=5,
            enable_mlflow=False,
            output_dir=str(temp_dir / "output"),
        )
        
        # Create CNN hybrid model
        bert_model = create_cnn_hybrid_model(
            model_name="answerdotai/ModernBERT-base",
            num_labels=2,
            cnn_kernel_sizes=[2, 3, 4],
            cnn_num_filters=128,
        )
        bert_model.config.hidden_size = bert_model.output_hidden_size
        
        model = TitanicClassifier(bert_model)
        trainer = MLXTrainer(model=model, config=config)
        
        # Create data loader
        train_loader = UnifiedTitanicDataPipeline(
            data_path=str(sample_titanic_data),
            tokenizer_name="answerdotai/ModernBERT-base",
            batch_size=4,
            max_length=128,
        )
        
        # Train
        results = trainer.train(train_dataloader=train_loader)
        
        assert results["total_time"] > 0
    
    def test_checkpoint_resume(self, sample_titanic_data, temp_dir):
        """Test resuming training from checkpoint."""
        # Create config
        config = UnifiedTrainingConfig(
            num_epochs=2,
            base_batch_size=4,
            eval_steps=5,
            save_steps=2,
            enable_mlflow=False,
            output_dir=str(temp_dir / "output"),
        )
        
        # Create model and trainer
        model = TitanicClassifier(create_model("standard"))
        trainer = MLXTrainer(model=model, config=config)
        
        # Create data loader
        train_loader = UnifiedTitanicDataPipeline(
            data_path=str(sample_titanic_data),
            tokenizer_name="answerdotai/ModernBERT-base",
            batch_size=4,
            max_length=128,
        )
        
        # Train for one epoch
        trainer.config.num_epochs = 1
        results1 = trainer.train(train_dataloader=train_loader)
        
        # Get checkpoint
        checkpoint_dir = Path(config.checkpoint_dir)
        checkpoints = list(checkpoint_dir.glob("checkpoint-*"))
        assert len(checkpoints) > 0
        
        # Create new trainer and resume
        model2 = TitanicClassifier(create_model("standard"))
        trainer2 = MLXTrainer(model=model2, config=config)
        trainer2.config.num_epochs = 2  # Reset to original
        
        # Resume training
        results2 = trainer2.train(
            train_dataloader=train_loader,
            resume_from_checkpoint=checkpoints[0].name,
        )
        
        # Should start from saved epoch
        assert trainer2.current_epoch >= 1
    
    def test_memory_profiling(self, sample_titanic_data, temp_dir):
        """Test memory profiling during training."""
        # Create config with profiling enabled
        config = UnifiedTrainingConfig(
            num_epochs=1,
            base_batch_size=4,
            eval_steps=10,
            enable_mlflow=False,
            enable_profiling=True,
            profile_memory_steps=2,
            output_dir=str(temp_dir / "output"),
        )
        
        # Create model and trainer
        model = TitanicClassifier(create_model("standard"))
        trainer = MLXTrainer(model=model, config=config)
        
        # Create data loader
        train_loader = UnifiedTitanicDataPipeline(
            data_path=str(sample_titanic_data),
            tokenizer_name="answerdotai/ModernBERT-base",
            batch_size=4,
            max_length=128,
        )
        
        # Train
        results = trainer.train(train_dataloader=train_loader)
        
        # Check memory profile was saved
        profile_path = Path(config.output_dir) / "memory_profile.json"
        assert profile_path.exists()
        
        # Load and verify profile
        with open(profile_path) as f:
            profile = json.load(f)
        
        assert len(profile) > 0
        assert "step" in profile[0]
        assert "memory" in profile[0]
        assert "batch_size" in profile[0]


@pytest.mark.slow
@pytest.mark.integration
class TestFullPipelineE2E:
    """End-to-end tests for the complete pipeline."""
    
    def test_complete_training_workflow(self, sample_titanic_data, temp_dir):
        """Test complete workflow from data loading to prediction."""
        # 1. Setup configuration
        config = UnifiedTrainingConfig(
            num_epochs=2,
            base_batch_size=4,
            eval_steps=3,
            save_steps=3,
            enable_mlflow=False,
            output_dir=str(temp_dir / "output"),
        )
        
        # 2. Create model and trainer
        model = TitanicClassifier(create_model("standard"))
        trainer = MLXTrainer(model=model, config=config)
        
        # 3. Create data loaders
        train_loader, val_loader = create_unified_dataloaders(
            train_path=str(sample_titanic_data),
            val_path=str(sample_titanic_data),
            tokenizer_name="answerdotai/ModernBERT-base",
            batch_size=4,
            max_length=128,
        )
        
        # 4. Train model
        train_results = trainer.train(
            train_dataloader=train_loader,
            val_dataloader=val_loader,
        )
        
        # 5. Verify training results
        assert train_results["best_metric"] > 0
        assert len(train_results["history"]["train_loss"]) == config.num_epochs
        
        # 6. Load best checkpoint for prediction
        best_checkpoint = Path(config.checkpoint_dir) / "best"
        assert best_checkpoint.exists()
        
        # 7. Create new model and load checkpoint
        pred_model = TitanicClassifier(create_model("standard"))
        pred_model.load_pretrained(str(best_checkpoint))
        
        # 8. Make predictions on test data
        test_loader = UnifiedTitanicDataPipeline(
            data_path=str(sample_titanic_data),
            tokenizer_name="answerdotai/ModernBERT-base",
            batch_size=4,
            max_length=128,
            is_training=False,
            augment=False,  # Disable augmentation for test data
        )
        
        predictions = []
        num_samples = 0
        for batch in test_loader.get_dataloader():
            outputs = pred_model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
            )
            preds = mx.argmax(outputs["logits"], axis=-1)
            predictions.extend(preds.tolist())
            num_samples += batch["input_ids"].shape[0]
        
        # 9. Verify predictions
        assert len(predictions) == num_samples
        assert all(p in [0, 1] for p in predictions)