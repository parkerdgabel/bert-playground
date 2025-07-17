"""Tests for training configuration system."""

import json
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from training.config import (
    AdvancedFeatures,
    CheckpointConfig,
    EvaluationConfig,
    LearningRateSchedule,
    LossFunction,
    MemoryConfig,
    MLXOptimizationConfig,
    MonitoringConfig,
    OptimizationLevel,
    OptimizerType,
    TrainingConfig,
    get_competition_config,
    get_development_config,
    get_production_config,
)


class TestEnums:
    """Test enum definitions."""

    def test_optimization_level_values(self):
        """Test optimization level enum values."""
        assert OptimizationLevel.DEVELOPMENT.value == "development"
        assert OptimizationLevel.STANDARD.value == "standard"
        assert OptimizationLevel.PRODUCTION.value == "production"
        assert OptimizationLevel.COMPETITION.value == "competition"
        assert OptimizationLevel.AUTO.value == "auto"

    def test_learning_rate_schedule_values(self):
        """Test learning rate schedule enum values."""
        assert LearningRateSchedule.CONSTANT.value == "constant"
        assert LearningRateSchedule.COSINE_WARMUP.value == "cosine_warmup"
        assert LearningRateSchedule.PLATEAU.value == "plateau"

    def test_optimizer_type_values(self):
        """Test optimizer type enum values."""
        assert OptimizerType.ADAMW.value == "adamw"
        assert OptimizerType.ADAM.value == "adam"
        assert OptimizerType.SGD.value == "sgd"

    def test_loss_function_values(self):
        """Test loss function enum values."""
        assert LossFunction.CROSS_ENTROPY.value == "cross_entropy"
        assert LossFunction.FOCAL_LOSS.value == "focal_loss"
        assert LossFunction.LABEL_SMOOTHING.value == "label_smoothing"


class TestMemoryConfig:
    """Test memory configuration."""

    def test_default_memory_config(self):
        """Test default memory configuration values."""
        config = MemoryConfig()

        assert config.enable_memory_profiling is True
        assert config.memory_limit_gb is None
        assert config.dynamic_batch_sizing is True
        assert config.min_batch_size == 4
        assert config.max_batch_size == 128
        assert config.unified_memory_fraction == 0.8

    def test_custom_memory_config(self):
        """Test custom memory configuration."""
        config = MemoryConfig(
            enable_memory_profiling=False,
            memory_limit_gb=16.0,
            min_batch_size=8,
            max_batch_size=64,
        )

        assert config.enable_memory_profiling is False
        assert config.memory_limit_gb == 16.0
        assert config.min_batch_size == 8
        assert config.max_batch_size == 64


class TestMLXOptimizationConfig:
    """Test MLX optimization configuration."""

    def test_default_mlx_config(self):
        """Test default MLX optimization configuration."""
        config = MLXOptimizationConfig()

        assert config.enable_lazy_evaluation is True
        assert config.eval_frequency == 10
        assert config.gradient_accumulation_steps == 1
        assert config.max_grad_norm == 1.0
        assert config.mixed_precision is False
        assert config.precision_dtype == "float32"

    def test_custom_mlx_config(self):
        """Test custom MLX optimization configuration."""
        config = MLXOptimizationConfig(
            gradient_accumulation_steps=4,
            max_grad_norm=0.5,
            mixed_precision=True,
            precision_dtype="float16",
        )

        assert config.gradient_accumulation_steps == 4
        assert config.max_grad_norm == 0.5
        assert config.mixed_precision is True
        assert config.precision_dtype == "float16"


class TestMonitoringConfig:
    """Test monitoring configuration."""

    def test_default_monitoring_config(self):
        """Test default monitoring configuration."""
        config = MonitoringConfig()

        assert config.enable_mlflow is True
        assert config.experiment_name == "mlx_training"
        assert config.log_level == "INFO"
        assert config.enable_rich_console is True
        assert config.log_frequency == 10
        assert config.eval_frequency == 500

    def test_custom_monitoring_config(self):
        """Test custom monitoring configuration."""
        config = MonitoringConfig(
            experiment_name="custom_experiment",
            log_level="DEBUG",
            log_frequency=5,
            enable_mlflow=False,
        )

        assert config.experiment_name == "custom_experiment"
        assert config.log_level == "DEBUG"
        assert config.log_frequency == 5
        assert config.enable_mlflow is False


class TestCheckpointConfig:
    """Test checkpoint configuration."""

    def test_default_checkpoint_config(self):
        """Test default checkpoint configuration."""
        config = CheckpointConfig()

        assert config.enable_checkpointing is True
        assert config.checkpoint_dir == "checkpoints"
        assert config.save_optimizer_state is True
        assert config.save_best_model is True
        assert config.best_model_metric == "val_accuracy"
        assert config.use_safetensors is True

    def test_custom_checkpoint_config(self):
        """Test custom checkpoint configuration."""
        config = CheckpointConfig(
            checkpoint_dir="custom_checkpoints",
            checkpoint_frequency=500,
            max_checkpoints_to_keep=3,
            best_model_metric="val_f1",
        )

        assert config.checkpoint_dir == "custom_checkpoints"
        assert config.checkpoint_frequency == 500
        assert config.max_checkpoints_to_keep == 3
        assert config.best_model_metric == "val_f1"


class TestEvaluationConfig:
    """Test evaluation configuration."""

    def test_default_evaluation_config(self):
        """Test default evaluation configuration."""
        config = EvaluationConfig()

        assert config.eval_during_training is True
        assert config.eval_steps == 500
        assert config.primary_metric == "accuracy"
        assert config.enable_early_stopping is True
        assert config.early_stopping_patience == 10
        assert "accuracy" in config.metrics_to_compute
        assert "f1" in config.metrics_to_compute

    def test_custom_evaluation_config(self):
        """Test custom evaluation configuration."""
        config = EvaluationConfig(
            eval_steps=250,
            primary_metric="f1",
            early_stopping_patience=5,
            metrics_to_compute=["accuracy", "precision"],
        )

        assert config.eval_steps == 250
        assert config.primary_metric == "f1"
        assert config.early_stopping_patience == 5
        assert config.metrics_to_compute == ["accuracy", "precision"]


class TestAdvancedFeatures:
    """Test advanced features configuration."""

    def test_default_advanced_features(self):
        """Test default advanced features configuration."""
        config = AdvancedFeatures()

        assert config.label_smoothing == 0.0
        assert config.dropout_rate == 0.1
        assert config.weight_decay == 0.01
        assert config.enable_augmentation is True
        assert config.enable_ensembling is False
        assert config.enable_curriculum_learning is False
        assert config.enable_distillation is False

    def test_custom_advanced_features(self):
        """Test custom advanced features configuration."""
        config = AdvancedFeatures(
            label_smoothing=0.1,
            dropout_rate=0.2,
            enable_ensembling=True,
            ensemble_size=3,
            enable_curriculum_learning=True,
        )

        assert config.label_smoothing == 0.1
        assert config.dropout_rate == 0.2
        assert config.enable_ensembling is True
        assert config.ensemble_size == 3
        assert config.enable_curriculum_learning is True


class TestTrainingConfig:
    """Test main training configuration."""

    def test_default_training_config(self):
        """Test default training configuration."""
        with patch("training.config.Path.mkdir"):
            config = TrainingConfig()

        assert config.epochs == 3
        assert config.batch_size == 32
        assert config.learning_rate == 2e-5
        assert config.model_name == "answerdotai/ModernBERT-base"
        assert config.max_length == 256
        assert config.optimizer == OptimizerType.ADAMW
        assert config.lr_schedule == LearningRateSchedule.COSINE_WARMUP
        assert config.seed == 42

        # Test sub-configurations are created
        assert isinstance(config.memory, MemoryConfig)
        assert isinstance(config.mlx_optimization, MLXOptimizationConfig)
        assert isinstance(config.monitoring, MonitoringConfig)
        assert isinstance(config.checkpoint, CheckpointConfig)
        assert isinstance(config.evaluation, EvaluationConfig)
        assert isinstance(config.advanced, AdvancedFeatures)

    def test_custom_training_config(self):
        """Test custom training configuration."""
        with patch("training.config.Path.mkdir"):
            config = TrainingConfig(
                epochs=5,
                batch_size=64,
                learning_rate=1e-4,
                model_name="custom-bert",
                optimization_level=OptimizationLevel.PRODUCTION,
            )

        assert config.epochs == 5
        assert config.batch_size == 64
        assert config.learning_rate == 1e-4
        assert config.model_name == "custom-bert"
        assert config.optimization_level == OptimizationLevel.PRODUCTION

    def test_config_validation(self):
        """Test configuration validation."""
        # Test invalid epochs
        with pytest.raises(ValueError, match="epochs must be positive"):
            with patch("training.config.Path.mkdir"):
                TrainingConfig(epochs=0)

        # Test invalid batch size
        with pytest.raises(ValueError, match="batch_size must be positive"):
            with patch("training.config.Path.mkdir"):
                TrainingConfig(batch_size=-1)

        # Test invalid learning rate
        with pytest.raises(ValueError, match="learning_rate must be positive"):
            with patch("training.config.Path.mkdir"):
                TrainingConfig(learning_rate=0)

        # Test invalid max length
        with pytest.raises(ValueError, match="max_length must be positive"):
            with patch("training.config.Path.mkdir"):
                TrainingConfig(max_length=0)

    def test_memory_config_validation(self):
        """Test memory configuration validation."""
        memory_config = MemoryConfig(min_batch_size=64, max_batch_size=32)

        with pytest.raises(
            ValueError, match="min_batch_size cannot be greater than max_batch_size"
        ):
            with patch("training.config.Path.mkdir"):
                TrainingConfig(memory=memory_config)

    def test_optimization_level_application(self):
        """Test optimization level specific settings."""
        # Test development level
        with patch("training.config.Path.mkdir"):
            dev_config = TrainingConfig(
                optimization_level=OptimizationLevel.DEVELOPMENT
            )

        assert dev_config.memory.dynamic_batch_sizing is False
        assert dev_config.memory.enable_memory_profiling is False
        assert dev_config.mlx_optimization.eval_frequency == 5

        # Test production level
        with patch("training.config.Path.mkdir"):
            prod_config = TrainingConfig(
                optimization_level=OptimizationLevel.PRODUCTION
            )

        assert prod_config.memory.dynamic_batch_sizing is True
        assert prod_config.memory.max_batch_size == 64
        assert prod_config.mlx_optimization.enable_jit is True

        # Test competition level
        with patch("training.config.Path.mkdir"):
            comp_config = TrainingConfig(
                optimization_level=OptimizationLevel.COMPETITION
            )

        assert comp_config.memory.max_batch_size == 128
        assert comp_config.mlx_optimization.mixed_precision is True
        assert comp_config.advanced.enable_ensembling is True

    def test_effective_batch_size(self):
        """Test effective batch size calculation."""
        with patch("training.config.Path.mkdir"):
            config = TrainingConfig(batch_size=32)

        config.mlx_optimization.gradient_accumulation_steps = 4
        assert config.get_effective_batch_size() == 128

    def test_total_steps_calculation(self):
        """Test total steps calculation."""
        with patch("training.config.Path.mkdir"):
            config = TrainingConfig(epochs=3, batch_size=32)

        dataset_size = 1000
        expected_steps = 3 * (1000 // 32)  # 3 epochs * steps_per_epoch
        assert config.get_total_steps(dataset_size) == expected_steps

        # Test with max_steps limit
        config.max_steps = 50
        assert config.get_total_steps(dataset_size) == 50

    def test_update_from_dataset(self):
        """Test configuration update from dataset info."""
        with patch("training.config.Path.mkdir"):
            config = TrainingConfig(optimization_level=OptimizationLevel.AUTO)

        dataset_info = {
            "dataset_spec": {
                "problem_type": "binary_classification",
                "expected_size": 50000,
            }
        }

        config.update_from_dataset(dataset_info)

        assert config.num_labels == 2
        # Should auto-configure to production level for 50k samples
        assert config.optimization_level == OptimizationLevel.PRODUCTION


class TestConfigSerialization:
    """Test configuration serialization and loading."""

    def test_config_to_dict(self):
        """Test configuration conversion to dictionary."""
        with patch("training.config.Path.mkdir"):
            config = TrainingConfig(
                epochs=5,
                batch_size=64,
                optimizer=OptimizerType.ADAM,
            )

        config_dict = config.to_dict()

        assert config_dict["epochs"] == 5
        assert config_dict["batch_size"] == 64
        assert config_dict["optimizer"] == "adam"  # Enum converted to string
        assert "memory" in config_dict
        assert "mlx_optimization" in config_dict

    def test_config_save_and_load(self):
        """Test configuration saving and loading."""
        with patch("training.config.Path.mkdir"):
            original_config = TrainingConfig(
                epochs=7,
                batch_size=48,
                learning_rate=1e-4,
                optimizer=OptimizerType.SGD,
                lr_schedule=LearningRateSchedule.PLATEAU,
            )

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            config_path = f.name

        try:
            # Save configuration
            original_config.save(config_path)

            # Check that file was created and contains valid JSON
            assert Path(config_path).exists()
            with open(config_path) as f:
                saved_data = json.load(f)
            assert saved_data["epochs"] == 7
            assert saved_data["batch_size"] == 48

            # Load configuration
            with patch("training.config.Path.mkdir"):
                loaded_config = TrainingConfig.load(config_path)

            # Verify loaded configuration matches original
            assert loaded_config.epochs == original_config.epochs
            assert loaded_config.batch_size == original_config.batch_size
            assert loaded_config.learning_rate == original_config.learning_rate
            assert loaded_config.optimizer == original_config.optimizer
            assert loaded_config.lr_schedule == original_config.lr_schedule

        finally:
            Path(config_path).unlink()


class TestPresetConfigurations:
    """Test preset configuration functions."""

    def test_development_config(self):
        """Test development preset configuration."""
        with patch("training.config.Path.mkdir"):
            config = get_development_config()

        assert config.epochs == 1
        assert config.batch_size == 16
        assert config.optimization_level == OptimizationLevel.DEVELOPMENT

    def test_development_config_with_overrides(self):
        """Test development config with overrides."""
        with patch("training.config.Path.mkdir"):
            config = get_development_config(epochs=2, batch_size=8)

        assert config.epochs == 2
        assert config.batch_size == 8
        assert config.optimization_level == OptimizationLevel.DEVELOPMENT

    def test_production_config(self):
        """Test production preset configuration."""
        with patch("training.config.Path.mkdir"):
            config = get_production_config()

        assert config.epochs == 5
        assert config.batch_size == 64
        assert config.optimization_level == OptimizationLevel.PRODUCTION

    def test_competition_config(self):
        """Test competition preset configuration."""
        with patch("training.config.Path.mkdir"):
            config = get_competition_config()

        assert config.epochs == 10
        assert config.batch_size == 32
        assert config.optimization_level == OptimizationLevel.COMPETITION
        assert config.advanced.enable_ensembling is True
        assert config.advanced.ensemble_size == 5
        assert config.advanced.label_smoothing == 0.1

    def test_preset_config_overrides(self):
        """Test preset configurations with overrides."""
        with patch("training.config.Path.mkdir"):
            config = get_competition_config(
                epochs=15,
                learning_rate=5e-6,
                model_name="custom-model",
            )

        assert config.epochs == 15
        assert config.learning_rate == 5e-6
        assert config.model_name == "custom-model"
        # Should still maintain competition-specific settings
        assert config.optimization_level == OptimizationLevel.COMPETITION
        assert config.advanced.enable_ensembling is True
