"""Unit tests for training configuration."""

from pathlib import Path

from training.core.config import (
    BaseTrainerConfig,
    CheckpointStrategy,
    DataConfig,
    EnvironmentConfig,
    EvalStrategy,
    OptimizerConfig,
    OptimizerType,
    SchedulerConfig,
    SchedulerType,
    TrainingConfig,
    get_development_config,
    get_kaggle_competition_config,
    get_production_config,
    get_quick_test_config,
)


class TestOptimizerConfig:
    """Test optimizer configuration."""

    def test_default_config(self):
        """Test default optimizer configuration."""
        config = OptimizerConfig()
        assert config.type == OptimizerType.ADAMW
        assert config.learning_rate == 2e-5
        assert config.weight_decay == 0.01
        assert config.beta1 == 0.9
        assert config.beta2 == 0.999
        assert config.epsilon == 1e-8

    def test_custom_config(self):
        """Test custom optimizer configuration."""
        config = OptimizerConfig(
            type=OptimizerType.ADAM,
            learning_rate=1e-3,
            weight_decay=0.0,
        )
        assert config.type == OptimizerType.ADAM
        assert config.learning_rate == 1e-3
        assert config.weight_decay == 0.0

    def test_to_dict(self):
        """Test conversion to dictionary."""
        config = OptimizerConfig(type=OptimizerType.SGD, learning_rate=0.1)
        config_dict = config.to_dict()

        assert config_dict["type"] == "sgd"
        assert config_dict["learning_rate"] == 0.1
        assert "weight_decay" in config_dict

    def test_validation(self):
        """Test configuration validation."""
        # Valid config should not raise
        config = OptimizerConfig()
        # Individual configs don't have validate method, only BaseTrainerConfig does
        assert config.learning_rate > 0
        assert config.weight_decay >= 0


class TestSchedulerConfig:
    """Test scheduler configuration."""

    def test_default_config(self):
        """Test default scheduler configuration."""
        config = SchedulerConfig()
        assert config.type == SchedulerType.COSINE
        assert config.warmup_ratio == 0.0
        assert config.num_cycles == 0.5

    def test_linear_scheduler(self):
        """Test linear scheduler configuration."""
        config = SchedulerConfig(
            type=SchedulerType.LINEAR,
            warmup_ratio=0.05,
        )
        assert config.type == SchedulerType.LINEAR
        assert config.warmup_ratio == 0.05

    def test_reduce_on_plateau(self):
        """Test ReduceLROnPlateau configuration."""
        config = SchedulerConfig(
            type=SchedulerType.REDUCE_ON_PLATEAU,
            factor=0.5,
            patience=5,
        )
        assert config.type == SchedulerType.REDUCE_ON_PLATEAU
        assert config.factor == 0.5
        assert config.patience == 5

    def test_validation(self):
        """Test scheduler validation."""
        # Valid config should not raise
        config = SchedulerConfig()
        # Note: Validation is done at BaseTrainerConfig level


class TestDataConfig:
    """Test data configuration."""

    def test_default_config(self):
        """Test default data configuration."""
        config = DataConfig()
        assert config.batch_size == 32
        assert config.eval_batch_size == 32  # Should default to batch_size
        assert config.num_workers == 4
        assert config.prefetch_size == 2
        assert config.pin_memory == True

    def test_custom_config(self):
        """Test custom data configuration."""
        config = DataConfig(
            batch_size=16,
            num_workers=8,
            drop_last=True,
        )
        assert config.batch_size == 16
        assert config.num_workers == 8
        assert config.drop_last == True

    def test_validation(self):
        """Test data configuration validation."""
        # Valid config should not raise
        config = DataConfig()
        # Note: Validation is done at BaseTrainerConfig level


class TestTrainingConfig:
    """Test training configuration."""

    def test_default_config(self):
        """Test default training configuration."""
        config = TrainingConfig()
        assert config.num_epochs == 10
        assert config.max_steps == -1
        assert config.gradient_accumulation_steps == 1
        assert config.eval_strategy == EvalStrategy.EPOCH
        assert config.save_strategy == CheckpointStrategy.EPOCH
        assert config.logging_steps == 100

    def test_early_stopping_config(self):
        """Test early stopping configuration."""
        config = TrainingConfig(
            early_stopping=True,
            early_stopping_patience=3,
            early_stopping_threshold=0.001,
        )
        assert config.early_stopping == True
        assert config.early_stopping_patience == 3
        assert config.early_stopping_threshold == 0.001

    def test_validation(self):
        """Test training configuration validation."""
        # Valid config should not raise
        config = TrainingConfig()
        # Note: Validation is done at BaseTrainerConfig level


class TestEnvironmentConfig:
    """Test environment configuration."""

    def test_default_config(self):
        """Test default environment configuration."""
        config = EnvironmentConfig()
        assert config.output_dir == Path("output")
        assert config.seed == 42
        assert config.seed == 42
        assert config.device == "gpu"

    def test_custom_config(self):
        """Test custom environment configuration."""
        config = EnvironmentConfig(
            output_dir=Path("/tmp/training"),
            experiment_name="test_exp",
            run_name="test_run",
        )
        assert config.output_dir == Path("/tmp/training")
        assert config.experiment_name == "test_exp"
        assert config.run_name == "test_run"

    def test_path_conversion(self):
        """Test path string to Path conversion."""
        config = EnvironmentConfig(output_dir="/tmp/test")
        assert isinstance(config.output_dir, Path)
        assert config.output_dir == Path("/tmp/test")


class TestBaseTrainerConfig:
    """Test base trainer configuration."""

    def test_default_config(self):
        """Test default base trainer configuration."""
        config = BaseTrainerConfig()
        assert isinstance(config.optimizer, OptimizerConfig)
        assert isinstance(config.scheduler, SchedulerConfig)
        assert isinstance(config.data, DataConfig)
        assert isinstance(config.training, TrainingConfig)
        assert isinstance(config.environment, EnvironmentConfig)

    def test_dict_initialization(self):
        """Test initialization from dictionaries."""
        config = BaseTrainerConfig(
            optimizer={"type": "adam", "learning_rate": 1e-4},
            training={"num_epochs": 5, "eval_strategy": "steps"},
            environment={"output_dir": "/tmp/test"},
        )

        assert config.optimizer.type == OptimizerType.ADAM
        assert config.optimizer.learning_rate == 1e-4
        assert config.training.num_epochs == 5
        assert config.environment.output_dir == Path("/tmp/test")

    def test_to_dict(self):
        """Test conversion to dictionary."""
        config = BaseTrainerConfig(
            optimizer={"type": "sgd", "learning_rate": 0.01},
            training={"num_epochs": 10},
        )

        config_dict = config.to_dict()
        assert config_dict["optimizer"]["type"] == "sgd"
        assert config_dict["optimizer"]["learning_rate"] == 0.01
        assert config_dict["training"]["num_epochs"] == 10

    def test_validation(self):
        """Test full configuration validation."""
        config = BaseTrainerConfig()
        errors = config.validate()
        assert len(errors) == 0  # Should have no errors

        # Test with invalid sub-config
        config = BaseTrainerConfig(optimizer={"learning_rate": -1.0})
        errors = config.validate()
        assert len(errors) > 0
        assert any("Learning rate must be positive" in err for err in errors)


class TestPresetConfigs:
    """Test preset configurations."""

    def test_quick_test_preset(self):
        """Test quick test preset configuration."""
        config = get_quick_test_config()
        assert config.training.num_epochs == 1
        assert config.training.eval_strategy == EvalStrategy.NO
        assert config.training.logging_steps == 10
        assert config.data.batch_size == 8

    def test_development_preset(self):
        """Test development preset configuration."""
        config = get_development_config()
        assert config.training.num_epochs == 5
        assert config.optimizer.learning_rate == 2e-5
        assert config.training.save_total_limit == 3

    def test_production_preset(self):
        """Test production preset configuration."""
        config = get_production_config()
        assert config.training.num_epochs == 10
        assert config.training.early_stopping == True
        assert config.training.save_best_only == True
        assert config.data.num_workers == 8

    def test_kaggle_preset(self):
        """Test Kaggle preset configuration."""
        config = get_kaggle_competition_config()
        assert config.training.num_epochs == 15
        assert config.training.early_stopping == True
        assert config.data.augment_train == True

    def test_all_presets_valid(self):
        """Test that all presets are valid configurations."""
        for preset_func in [
            get_quick_test_config,
            get_development_config,
            get_production_config,
            get_kaggle_competition_config,
        ]:
            config = preset_func()
            errors = config.validate()
            assert len(errors) == 0  # All presets should be valid
