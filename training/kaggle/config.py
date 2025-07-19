"""
Configuration for Kaggle-specific training.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List
from enum import Enum
from pathlib import Path

from ..core.config import BaseTrainerConfig, OptimizerConfig, SchedulerConfig


class CompetitionType(Enum):
    """Types of Kaggle competitions."""
    BINARY_CLASSIFICATION = "binary_classification"
    MULTICLASS_CLASSIFICATION = "multiclass_classification"
    MULTILABEL_CLASSIFICATION = "multilabel_classification"
    REGRESSION = "regression"
    ORDINAL_REGRESSION = "ordinal_regression"
    TIME_SERIES = "time_series"
    RANKING = "ranking"


class CompetitionProfile(Enum):
    """Pre-configured profiles for different competition types."""
    TITANIC = "titanic"
    HOUSE_PRICES = "house_prices"
    DIGIT_RECOGNIZER = "digit_recognizer"
    NLP_DISASTER = "nlp_disaster"
    TABULAR_PLAYGROUND = "tabular_playground"
    CUSTOM = "custom"


@dataclass
class KaggleConfig:
    """Kaggle-specific configuration."""
    
    # Competition settings
    competition_name: str
    competition_type: CompetitionType
    submission_message: str = "MLX-BERT submission"
    
    # Kaggle API settings
    enable_api: bool = True
    auto_submit: bool = False
    download_data: bool = True
    
    # Evaluation settings
    competition_metric: str = "accuracy"  # Main metric for the competition
    maximize_metric: bool = True  # Whether higher is better
    
    # Cross-validation
    cv_folds: int = 5
    cv_strategy: str = "stratified"  # stratified, group, time_series
    
    # Ensemble settings
    enable_ensemble: bool = False
    ensemble_method: str = "voting"  # voting, blending, stacking
    ensemble_weights: Optional[List[float]] = None
    
    # Pseudo-labeling
    enable_pseudo_labeling: bool = False
    pseudo_label_threshold: float = 0.95
    pseudo_label_weight: float = 0.5
    
    # Test-time augmentation
    enable_tta: bool = False
    tta_iterations: int = 5
    
    # Submission settings
    submission_dir: Path = Path("submissions")
    save_oof_predictions: bool = True  # Out-of-fold predictions
    save_test_predictions: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "competition_name": self.competition_name,
            "competition_type": self.competition_type.value,
            "submission_message": self.submission_message,
            "enable_api": self.enable_api,
            "auto_submit": self.auto_submit,
            "download_data": self.download_data,
            "competition_metric": self.competition_metric,
            "maximize_metric": self.maximize_metric,
            "cv_folds": self.cv_folds,
            "cv_strategy": self.cv_strategy,
            "enable_ensemble": self.enable_ensemble,
            "ensemble_method": self.ensemble_method,
            "ensemble_weights": self.ensemble_weights,
            "enable_pseudo_labeling": self.enable_pseudo_labeling,
            "pseudo_label_threshold": self.pseudo_label_threshold,
            "pseudo_label_weight": self.pseudo_label_weight,
            "enable_tta": self.enable_tta,
            "tta_iterations": self.tta_iterations,
            "submission_dir": str(self.submission_dir),
            "save_oof_predictions": self.save_oof_predictions,
            "save_test_predictions": self.save_test_predictions,
        }
    
    @classmethod
    def from_dict(cls, config: Dict[str, Any]) -> "KaggleConfig":
        """Create from dictionary."""
        config = config.copy()
        if "competition_type" in config:
            config["competition_type"] = CompetitionType(config["competition_type"])
        if "submission_dir" in config:
            config["submission_dir"] = Path(config["submission_dir"])
        return cls(**config)


@dataclass
class KaggleTrainerConfig(BaseTrainerConfig):
    """Configuration for Kaggle trainer."""
    
    # Kaggle-specific configuration
    kaggle: KaggleConfig = field(default_factory=lambda: KaggleConfig(
        competition_name="titanic",
        competition_type=CompetitionType.BINARY_CLASSIFICATION,
    ))
    
    # Additional Kaggle optimizations
    use_competition_splits: bool = True  # Use competition's train/test split
    stratify_batches: bool = True  # Stratified sampling for batches
    class_weights: Optional[Dict[int, float]] = None  # Handle imbalanced data
    
    def __post_init__(self):
        super().__post_init__()
        
        # Ensure Kaggle config is properly typed
        if isinstance(self.kaggle, dict):
            self.kaggle = KaggleConfig.from_dict(self.kaggle)
        
        # Set competition-optimized defaults
        self._apply_competition_defaults()
    
    def _apply_competition_defaults(self):
        """Apply optimized defaults based on competition type."""
        comp_type = self.kaggle.competition_type
        
        if comp_type == CompetitionType.BINARY_CLASSIFICATION:
            # Binary classification defaults
            self.training.best_metric = "eval_auc"
            self.training.best_metric_mode = "max"
            self.kaggle.competition_metric = "auc"
            
        elif comp_type == CompetitionType.MULTICLASS_CLASSIFICATION:
            # Multiclass defaults
            self.training.best_metric = "eval_accuracy"
            self.training.best_metric_mode = "max"
            self.kaggle.competition_metric = "accuracy"
            
        elif comp_type == CompetitionType.REGRESSION:
            # Regression defaults
            self.training.best_metric = "eval_rmse"
            self.training.best_metric_mode = "min"
            self.kaggle.competition_metric = "rmse"
            self.kaggle.maximize_metric = False
            
        # Enable mixed precision for competitions
        self.training.mixed_precision = True
        
        # Optimize data loading
        self.data.num_workers = 8
        self.data.prefetch_size = 4
        
        # Enable gradient accumulation for larger effective batch size
        if self.training.gradient_accumulation_steps == 1:
            self.training.gradient_accumulation_steps = 2
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        config = super().to_dict()
        config["kaggle"] = self.kaggle.to_dict()
        config["use_competition_splits"] = self.use_competition_splits
        config["stratify_batches"] = self.stratify_batches
        config["class_weights"] = self.class_weights
        return config
    
    @classmethod
    def from_dict(cls, config: Dict[str, Any]) -> "KaggleTrainerConfig":
        """Create from dictionary."""
        # Extract Kaggle config
        kaggle_config = config.pop("kaggle", {})
        use_competition_splits = config.pop("use_competition_splits", True)
        stratify_batches = config.pop("stratify_batches", True)
        class_weights = config.pop("class_weights", None)
        
        # Create base config
        base_config = BaseTrainerConfig.from_dict(config)
        
        # Create Kaggle config
        return cls(
            optimizer=base_config.optimizer,
            scheduler=base_config.scheduler,
            data=base_config.data,
            training=base_config.training,
            environment=base_config.environment,
            custom=base_config.custom,
            kaggle=KaggleConfig.from_dict(kaggle_config),
            use_competition_splits=use_competition_splits,
            stratify_batches=stratify_batches,
            class_weights=class_weights,
        )


# Pre-configured competition profiles
def get_titanic_config() -> KaggleTrainerConfig:
    """Get configuration for Titanic competition."""
    return KaggleTrainerConfig(
        kaggle=KaggleConfig(
            competition_name="titanic",
            competition_type=CompetitionType.BINARY_CLASSIFICATION,
            competition_metric="accuracy",
            cv_folds=5,
            enable_ensemble=True,
        ),
        optimizer=OptimizerConfig(
            learning_rate=2e-5,
            weight_decay=0.01,
        ),
        scheduler=SchedulerConfig(
            type="cosine",
            warmup_ratio=0.1,
        ),
        training=BaseTrainerConfig().training,  # Use defaults and apply competition optimizations
    )


def get_house_prices_config() -> KaggleTrainerConfig:
    """Get configuration for House Prices competition."""
    return KaggleTrainerConfig(
        kaggle=KaggleConfig(
            competition_name="house-prices-advanced-regression-techniques",
            competition_type=CompetitionType.REGRESSION,
            competition_metric="rmse",
            maximize_metric=False,
            cv_folds=5,
            enable_pseudo_labeling=True,
        ),
        optimizer=OptimizerConfig(
            learning_rate=1e-5,
            weight_decay=0.1,
        ),
        training=BaseTrainerConfig().training,
    )


def get_nlp_disaster_config() -> KaggleTrainerConfig:
    """Get configuration for NLP Disaster Tweets competition."""
    return KaggleTrainerConfig(
        kaggle=KaggleConfig(
            competition_name="nlp-getting-started",
            competition_type=CompetitionType.BINARY_CLASSIFICATION,
            competition_metric="f1",
            cv_folds=5,
            enable_tta=True,
        ),
        optimizer=OptimizerConfig(
            learning_rate=2e-5,
        ),
        data=BaseTrainerConfig().data,
        training=BaseTrainerConfig().training,
    )


# Profile registry
COMPETITION_PROFILES = {
    CompetitionProfile.TITANIC: get_titanic_config,
    CompetitionProfile.HOUSE_PRICES: get_house_prices_config,
    CompetitionProfile.NLP_DISASTER: get_nlp_disaster_config,
}


def get_competition_config(profile: CompetitionProfile | str) -> KaggleTrainerConfig:
    """Get pre-configured competition config by profile."""
    if isinstance(profile, str):
        profile = CompetitionProfile(profile)
    
    if profile in COMPETITION_PROFILES:
        return COMPETITION_PROFILES[profile]()
    else:
        return KaggleTrainerConfig()  # Default config