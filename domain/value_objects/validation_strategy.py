"""Cross-validation strategy value objects.

This module contains immutable value objects representing different
cross-validation strategies commonly used in ML competitions.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional, List, Union, Tuple
from abc import ABC, abstractmethod


class ValidationStrategyType(Enum):
    """Types of validation strategies."""
    TRAIN_VAL_SPLIT = "train_val_split"
    KFOLD = "kfold"
    STRATIFIED_KFOLD = "stratified_kfold"
    GROUP_KFOLD = "group_kfold"
    TIME_SERIES_SPLIT = "time_series_split"
    STRATIFIED_GROUP_KFOLD = "stratified_group_kfold"
    REPEATED_KFOLD = "repeated_kfold"
    LEAVE_ONE_OUT = "leave_one_out"
    CUSTOM = "custom"


@dataclass(frozen=True)
class ValidationStrategy(ABC):
    """Base class for validation strategies."""
    strategy_type: ValidationStrategyType
    random_state: Optional[int] = 42
    
    @abstractmethod
    def get_n_splits(self) -> int:
        """Get number of splits this strategy will produce."""
        pass
    
    @abstractmethod
    def describe(self) -> str:
        """Get human-readable description of the strategy."""
        pass


@dataclass(frozen=True)
class TrainValSplit(ValidationStrategy):
    """Simple train/validation split strategy."""
    strategy_type: ValidationStrategyType = ValidationStrategyType.TRAIN_VAL_SPLIT
    val_size: Union[float, int] = 0.2  # Fraction or absolute number
    shuffle: bool = True
    stratify_column: Optional[str] = None
    
    def get_n_splits(self) -> int:
        """Train/val split produces 1 split."""
        return 1
    
    def describe(self) -> str:
        """Describe the strategy."""
        size_desc = f"{self.val_size*100:.0f}%" if isinstance(self.val_size, float) else f"{self.val_size} samples"
        stratify_desc = f" stratified by {self.stratify_column}" if self.stratify_column else ""
        shuffle_desc = " with shuffling" if self.shuffle else " without shuffling"
        return f"Train/validation split with {size_desc} validation{stratify_desc}{shuffle_desc}"


@dataclass(frozen=True)
class KFoldStrategy(ValidationStrategy):
    """K-Fold cross-validation strategy."""
    strategy_type: ValidationStrategyType = ValidationStrategyType.KFOLD
    n_splits: int = 5
    shuffle: bool = True
    
    def __post_init__(self):
        if self.n_splits < 2:
            raise ValueError("n_splits must be at least 2")
    
    def get_n_splits(self) -> int:
        """Get number of folds."""
        return self.n_splits
    
    def describe(self) -> str:
        """Describe the strategy."""
        shuffle_desc = " with shuffling" if self.shuffle else " without shuffling"
        return f"{self.n_splits}-fold cross-validation{shuffle_desc}"


@dataclass(frozen=True)
class StratifiedKFoldStrategy(ValidationStrategy):
    """Stratified K-Fold cross-validation strategy."""
    strategy_type: ValidationStrategyType = ValidationStrategyType.STRATIFIED_KFOLD
    n_splits: int = 5
    shuffle: bool = True
    stratify_column: str = "target"  # Column to stratify on
    
    def __post_init__(self):
        if self.n_splits < 2:
            raise ValueError("n_splits must be at least 2")
    
    def get_n_splits(self) -> int:
        """Get number of folds."""
        return self.n_splits
    
    def describe(self) -> str:
        """Describe the strategy."""
        return f"Stratified {self.n_splits}-fold CV on '{self.stratify_column}'"
    
    @property
    def is_classification_compatible(self) -> bool:
        """Check if compatible with classification tasks."""
        return True


@dataclass(frozen=True)
class GroupKFoldStrategy(ValidationStrategy):
    """Group K-Fold cross-validation strategy."""
    strategy_type: ValidationStrategyType = ValidationStrategyType.GROUP_KFOLD
    n_splits: int = 5
    group_column: str = "group"  # Column containing group information
    
    def __post_init__(self):
        if self.n_splits < 2:
            raise ValueError("n_splits must be at least 2")
    
    def get_n_splits(self) -> int:
        """Get number of folds."""
        return self.n_splits
    
    def describe(self) -> str:
        """Describe the strategy."""
        return f"Group {self.n_splits}-fold CV on '{self.group_column}'"
    
    @property
    def prevents_leakage(self) -> bool:
        """This strategy prevents data leakage between groups."""
        return True


@dataclass(frozen=True)
class TimeSeriesSplitStrategy(ValidationStrategy):
    """Time series split validation strategy."""
    strategy_type: ValidationStrategyType = ValidationStrategyType.TIME_SERIES_SPLIT
    n_splits: int = 5
    test_size: Optional[int] = None  # Fixed test size
    max_train_size: Optional[int] = None  # Limit training size
    gap: int = 0  # Gap between train and test
    
    def __post_init__(self):
        if self.n_splits < 2:
            raise ValueError("n_splits must be at least 2")
        if self.gap < 0:
            raise ValueError("gap must be non-negative")
    
    def get_n_splits(self) -> int:
        """Get number of splits."""
        return self.n_splits
    
    def describe(self) -> str:
        """Describe the strategy."""
        desc = f"Time series {self.n_splits}-split"
        if self.test_size:
            desc += f" with test size {self.test_size}"
        if self.gap:
            desc += f" and gap {self.gap}"
        return desc
    
    @property
    def is_expanding_window(self) -> bool:
        """Check if using expanding window (no max_train_size)."""
        return self.max_train_size is None
    
    @property
    def is_sliding_window(self) -> bool:
        """Check if using sliding window (has max_train_size)."""
        return self.max_train_size is not None


@dataclass(frozen=True)
class RepeatedKFoldStrategy(ValidationStrategy):
    """Repeated K-Fold cross-validation strategy."""
    strategy_type: ValidationStrategyType = ValidationStrategyType.REPEATED_KFOLD
    n_splits: int = 5
    n_repeats: int = 3
    shuffle: bool = True
    
    def __post_init__(self):
        if self.n_splits < 2:
            raise ValueError("n_splits must be at least 2")
        if self.n_repeats < 1:
            raise ValueError("n_repeats must be at least 1")
    
    def get_n_splits(self) -> int:
        """Get total number of splits."""
        return self.n_splits * self.n_repeats
    
    def describe(self) -> str:
        """Describe the strategy."""
        return f"{self.n_repeats}x repeated {self.n_splits}-fold CV"
    
    @property
    def provides_stability(self) -> bool:
        """Repeated CV provides more stable estimates."""
        return True


@dataclass(frozen=True)
class ValidationMetrics:
    """Metrics to track during validation."""
    primary_metric: str  # e.g., "auc", "rmse", "log_loss"
    secondary_metrics: List[str] = None
    early_stopping_metric: Optional[str] = None
    early_stopping_rounds: Optional[int] = None
    
    def __post_init__(self):
        if self.secondary_metrics is None:
            object.__setattr__(self, 'secondary_metrics', [])
    
    @property
    def all_metrics(self) -> List[str]:
        """Get all metrics to track."""
        return [self.primary_metric] + self.secondary_metrics
    
    @property
    def uses_early_stopping(self) -> bool:
        """Check if early stopping is configured."""
        return self.early_stopping_metric is not None


@dataclass(frozen=True)
class ValidationConfig:
    """Complete validation configuration."""
    strategy: ValidationStrategy
    metrics: ValidationMetrics
    save_oof_predictions: bool = True
    save_test_predictions: bool = True
    use_sample_weights: bool = False
    probability_calibration: Optional[str] = None  # "isotonic", "sigmoid"
    
    @property
    def n_models(self) -> int:
        """Number of models that will be trained."""
        return self.strategy.get_n_splits()
    
    @property
    def produces_oof(self) -> bool:
        """Check if out-of-fold predictions will be produced."""
        return self.save_oof_predictions and self.strategy.get_n_splits() > 1
    
    def estimate_training_time(self, time_per_fold_minutes: float) -> float:
        """Estimate total training time in hours."""
        total_minutes = self.n_models * time_per_fold_minutes
        return total_minutes / 60.0


@dataclass(frozen=True)
class ValidationResult:
    """Result of a validation split."""
    fold_number: int
    train_indices: List[int]
    val_indices: List[int]
    train_score: float
    val_score: float
    metrics: Dict[str, float]
    training_time_seconds: float
    
    @property
    def overfit_ratio(self) -> float:
        """Calculate overfit ratio (train/val score ratio)."""
        if self.val_score == 0:
            return float('inf')
        return self.train_score / self.val_score
    
    @property
    def is_overfitting(self) -> bool:
        """Check if fold shows signs of overfitting."""
        # For metrics where higher is better
        return self.overfit_ratio > 1.1  # 10% threshold


@dataclass(frozen=True)
class CrossValidationSummary:
    """Summary of cross-validation results."""
    strategy: ValidationStrategy
    fold_results: List[ValidationResult]
    oof_score: Optional[float] = None
    
    @property
    def mean_val_score(self) -> float:
        """Mean validation score across folds."""
        if not self.fold_results:
            return 0.0
        return sum(r.val_score for r in self.fold_results) / len(self.fold_results)
    
    @property
    def std_val_score(self) -> float:
        """Standard deviation of validation scores."""
        if len(self.fold_results) < 2:
            return 0.0
        mean = self.mean_val_score
        variance = sum((r.val_score - mean) ** 2 for r in self.fold_results) / len(self.fold_results)
        return variance ** 0.5
    
    @property
    def cv_score_range(self) -> Tuple[float, float]:
        """Min and max validation scores."""
        if not self.fold_results:
            return (0.0, 0.0)
        scores = [r.val_score for r in self.fold_results]
        return (min(scores), max(scores))
    
    @property
    def is_stable(self) -> bool:
        """Check if CV results are stable."""
        # Consider stable if std < 1% of mean
        return self.std_val_score < (self.mean_val_score * 0.01)
    
    @property
    def worst_fold(self) -> Optional[int]:
        """Get fold number with worst performance."""
        if not self.fold_results:
            return None
        return min(self.fold_results, key=lambda r: r.val_score).fold_number
    
    @property
    def best_fold(self) -> Optional[int]:
        """Get fold number with best performance."""
        if not self.fold_results:
            return None
        return max(self.fold_results, key=lambda r: r.val_score).fold_number
    
    def get_fold_summary(self) -> str:
        """Get readable summary of CV results."""
        if not self.fold_results:
            return "No fold results available"
        
        return (
            f"{self.strategy.describe()}\n"
            f"Mean: {self.mean_val_score:.5f} ± {self.std_val_score:.5f}\n"
            f"Range: [{self.cv_score_range[0]:.5f}, {self.cv_score_range[1]:.5f}]\n"
            f"OOF Score: {self.oof_score:.5f}" if self.oof_score else ""
        )