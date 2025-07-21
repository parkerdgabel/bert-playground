"""Competition metadata and dataset analysis.

This module provides automatic detection and analysis of Kaggle competition
datasets, generating optimization recommendations for BERT models and MLX processing.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd
from loguru import logger

from .base import CompetitionType, DatasetSpec


@dataclass
class CompetitionMetadata:
    """Metadata for a Kaggle competition.

    This class contains comprehensive information about a competition,
    including automatic analysis results and optimization recommendations.
    """

    # Basic information
    competition_name: str
    competition_url: str | None = None
    description: str | None = None
    evaluation_metric: str | None = None

    # Competition settings
    submission_limit: int = 10
    team_limit: int = 1
    dataset_size_mb: float | None = None

    # Files and structure
    train_file: str | None = None
    test_file: str | None = None
    submission_file: str | None = None
    additional_files: list[str] = field(default_factory=list)

    # Analysis results
    competition_type: CompetitionType = CompetitionType.UNKNOWN
    auto_detected: bool = False
    confidence_score: float = 0.0

    # Optimization recommendations
    recommended_model_type: str = "bert_with_head"
    recommended_head_type: str = "binary_classification"
    recommended_batch_size: int = 32
    recommended_max_length: int = 512
    recommended_learning_rate: float = 2e-5

    # MLX optimization hints
    use_unified_memory: bool = True
    optimal_prefetch_size: int = 4
    optimal_num_workers: int = 4

    # BERT-specific recommendations
    recommended_pooling: str = "cls"
    use_attention_optimization: bool = True
    enable_gradient_checkpointing: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Convert metadata to dictionary."""
        return {
            "competition_name": self.competition_name,
            "competition_url": self.competition_url,
            "description": self.description,
            "evaluation_metric": self.evaluation_metric,
            "submission_limit": self.submission_limit,
            "team_limit": self.team_limit,
            "dataset_size_mb": self.dataset_size_mb,
            "train_file": self.train_file,
            "test_file": self.test_file,
            "submission_file": self.submission_file,
            "additional_files": self.additional_files,
            "competition_type": self.competition_type.value,
            "auto_detected": self.auto_detected,
            "confidence_score": self.confidence_score,
            "recommended_model_type": self.recommended_model_type,
            "recommended_head_type": self.recommended_head_type,
            "recommended_batch_size": self.recommended_batch_size,
            "recommended_max_length": self.recommended_max_length,
            "recommended_learning_rate": self.recommended_learning_rate,
            "use_unified_memory": self.use_unified_memory,
            "optimal_prefetch_size": self.optimal_prefetch_size,
            "optimal_num_workers": self.optimal_num_workers,
            "recommended_pooling": self.recommended_pooling,
            "use_attention_optimization": self.use_attention_optimization,
            "enable_gradient_checkpointing": self.enable_gradient_checkpointing,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CompetitionMetadata":
        """Create metadata from dictionary."""
        # Convert competition_type string back to enum
        if "competition_type" in data:
            data["competition_type"] = CompetitionType(data["competition_type"])
        return cls(**data)

    def save(self, path: str | Path) -> None:
        """Save metadata to JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: str | Path) -> "CompetitionMetadata":
        """Load metadata from JSON file."""
        with open(path) as f:
            data = json.load(f)
        return cls.from_dict(data)


class DatasetAnalyzer:
    """Analyzes Kaggle datasets and generates optimization recommendations.

    This class provides automatic detection of competition types and generates
    recommendations for BERT model configuration and MLX optimization.
    """

    # Competition type detection patterns
    BINARY_CLASSIFICATION_PATTERNS = {
        "target_values": {2},
        "column_names": {"survived", "target", "label", "class"},
        "competition_names": {"titanic", "binary"},
        "evaluation_metrics": {"accuracy", "auc", "logloss"},
    }

    MULTICLASS_CLASSIFICATION_PATTERNS = {
        "target_values": set(range(3, 21)),  # 3-20 classes
        "column_names": {"category", "class", "label", "target"},
        "competition_names": {"multiclass", "classification"},
        "evaluation_metrics": {"accuracy", "macro_f1", "weighted_f1"},
    }

    REGRESSION_PATTERNS = {
        "target_types": {"float64", "float32", "int64", "int32"},
        "column_names": {"price", "value", "amount", "score", "rating"},
        "competition_names": {"house", "price", "regression"},
        "evaluation_metrics": {"rmse", "mae", "mse", "r2"},
    }

    TIME_SERIES_PATTERNS = {
        "column_names": {"date", "time", "timestamp", "datetime"},
        "competition_names": {"forecast", "sales", "time"},
        "evaluation_metrics": {"smape", "mase", "rmse"},
    }

    def __init__(self):
        """Initialize the dataset analyzer."""
        self.logger = logger.bind(component="DatasetAnalyzer")

    def analyze_competition(
        self,
        data_path: str | Path,
        competition_name: str | None = None,
        target_column: str | None = None,
    ) -> tuple[CompetitionMetadata, DatasetSpec]:
        """Analyze a competition dataset and generate metadata.

        Args:
            data_path: Path to the dataset directory or file
            competition_name: Optional competition name
            target_column: Optional target column name

        Returns:
            Tuple of (CompetitionMetadata, DatasetSpec)
        """
        data_path = Path(data_path)

        # Determine competition name
        if competition_name is None:
            competition_name = data_path.stem if data_path.is_file() else data_path.name

        self.logger.info(f"Analyzing competition: {competition_name}")

        # Find and load data files
        train_file, test_file, submission_file = self._find_data_files(data_path)

        if train_file is None:
            raise ValueError(f"No training data found in {data_path}")

        # Load and analyze training data
        train_data = pd.read_csv(train_file)
        self.logger.info(f"Loaded training data: {train_data.shape}")

        # Auto-detect target column if not provided
        if target_column is None:
            target_column = self._detect_target_column(train_data, test_file)

        # Analyze competition type and characteristics
        competition_type, confidence = self._detect_competition_type(
            train_data, target_column, competition_name
        )

        # Generate metadata
        metadata = self._generate_metadata(
            competition_name=competition_name,
            competition_type=competition_type,
            confidence_score=confidence,
            train_file=str(train_file),
            test_file=str(test_file) if test_file else None,
            submission_file=str(submission_file) if submission_file else None,
            dataset_size_mb=self._calculate_dataset_size(data_path),
        )

        # Generate dataset specification
        spec = self._generate_dataset_spec(
            competition_name=competition_name,
            dataset_path=data_path,
            competition_type=competition_type,
            train_data=train_data,
            target_column=target_column,
            metadata=metadata,
        )

        self.logger.info(
            f"Analysis complete: {competition_type.value} "
            f"(confidence: {confidence:.2f})"
        )

        return metadata, spec

    def _find_data_files(
        self, data_path: Path
    ) -> tuple[Path | None, Path | None, Path | None]:
        """Find train, test, and submission files in the data directory.

        Args:
            data_path: Path to dataset directory or file

        Returns:
            Tuple of (train_file, test_file, submission_file)
        """
        if data_path.is_file():
            # Single file case
            if "train" in data_path.name.lower():
                return data_path, None, None
            else:
                return data_path, None, None

        # Directory case - search for standard Kaggle files
        files = list(data_path.glob("*.csv"))

        train_file = None
        test_file = None
        submission_file = None

        for file in files:
            name_lower = file.name.lower()

            if any(pattern in name_lower for pattern in ["train", "training"]):
                train_file = file
            elif any(pattern in name_lower for pattern in ["test", "testing"]):
                test_file = file
            elif any(
                pattern in name_lower
                for pattern in ["sample_submission", "submission", "submit"]
            ):
                submission_file = file

        # If no train file found, use the largest CSV file
        if train_file is None and files:
            train_file = max(files, key=lambda f: f.stat().st_size)

        return train_file, test_file, submission_file

    def _detect_target_column(
        self, train_data: pd.DataFrame, test_file: Path | None
    ) -> str | None:
        """Detect the target column in the training data.

        Args:
            train_data: Training data DataFrame
            test_file: Optional test file path

        Returns:
            Target column name or None
        """
        # Common target column names
        common_targets = [
            "target",
            "label",
            "class",
            "y",
            "output",
            "survived",
            "price",
            "sales",
            "score",
            "rating",
        ]

        # Check for exact matches first
        for col in common_targets:
            if col in train_data.columns:
                return col

        # Check for partial matches
        for col in train_data.columns:
            col_lower = col.lower()
            for target in common_targets:
                if target in col_lower:
                    return col

        # If test file exists, target is likely a column in train but not in test
        if test_file and test_file.exists():
            try:
                test_data = pd.read_csv(test_file)
                train_cols = set(train_data.columns)
                test_cols = set(test_data.columns)

                # Target column should be in train but not in test
                candidates = train_cols - test_cols

                # Remove ID-like columns
                id_patterns = ["id", "index", "key"]
                candidates = {
                    col
                    for col in candidates
                    if not any(pattern in col.lower() for pattern in id_patterns)
                }

                if len(candidates) == 1:
                    return list(candidates)[0]
                elif len(candidates) > 1:
                    # Return the most likely target based on column name
                    for col in candidates:
                        if any(target in col.lower() for target in common_targets):
                            return col
                    # Fallback to first candidate
                    return list(candidates)[0]

            except Exception as e:
                self.logger.warning(f"Could not load test file {test_file}: {e}")

        return None

    def _detect_competition_type(
        self,
        train_data: pd.DataFrame,
        target_column: str | None,
        competition_name: str,
    ) -> tuple[CompetitionType, float]:
        """Detect the competition type based on data analysis.

        Args:
            train_data: Training data DataFrame
            target_column: Target column name
            competition_name: Competition name

        Returns:
            Tuple of (CompetitionType, confidence_score)
        """
        scores = {}

        # Analyze target column if available
        if target_column and target_column in train_data.columns:
            target_data = train_data[target_column]

            # Check for binary classification
            unique_values = target_data.nunique()
            if unique_values == 2:
                scores[CompetitionType.BINARY_CLASSIFICATION] = 0.9
            elif 3 <= unique_values <= 20 and target_data.dtype in [
                "int64",
                "int32",
                "object",
            ]:
                scores[CompetitionType.MULTICLASS_CLASSIFICATION] = 0.8
            elif target_data.dtype in ["float64", "float32"]:
                # Check if it's actually ordinal (integer values in float column)
                if (target_data == target_data.astype(int)).all():
                    if unique_values <= 10:
                        scores[CompetitionType.ORDINAL_REGRESSION] = 0.7
                    else:
                        scores[CompetitionType.MULTICLASS_CLASSIFICATION] = 0.6
                else:
                    scores[CompetitionType.REGRESSION] = 0.8

        # Analyze column names for time series indicators
        date_columns = [
            col
            for col in train_data.columns
            if any(pattern in col.lower() for pattern in ["date", "time", "timestamp"])
        ]
        if date_columns:
            scores[CompetitionType.TIME_SERIES] = (
                scores.get(CompetitionType.TIME_SERIES, 0) + 0.3
            )

        # Analyze competition name
        name_lower = competition_name.lower()

        # Binary classification indicators
        if any(pattern in name_lower for pattern in ["titanic", "binary", "survival"]):
            scores[CompetitionType.BINARY_CLASSIFICATION] = (
                scores.get(CompetitionType.BINARY_CLASSIFICATION, 0) + 0.3
            )

        # Regression indicators
        if any(
            pattern in name_lower
            for pattern in ["house", "price", "sales", "regression"]
        ):
            scores[CompetitionType.REGRESSION] = (
                scores.get(CompetitionType.REGRESSION, 0) + 0.3
            )

        # Time series indicators
        if any(
            pattern in name_lower for pattern in ["forecast", "time", "sales", "demand"]
        ):
            scores[CompetitionType.TIME_SERIES] = (
                scores.get(CompetitionType.TIME_SERIES, 0) + 0.3
            )

        # Multiclass indicators
        if any(
            pattern in name_lower for pattern in ["classify", "category", "multiclass"]
        ):
            scores[CompetitionType.MULTICLASS_CLASSIFICATION] = (
                scores.get(CompetitionType.MULTICLASS_CLASSIFICATION, 0) + 0.3
            )

        # Return the highest scoring type
        if scores:
            best_type = max(scores, key=scores.get)
            confidence = min(scores[best_type], 1.0)
            return best_type, confidence
        else:
            return CompetitionType.UNKNOWN, 0.0

    def _generate_metadata(
        self,
        competition_name: str,
        competition_type: CompetitionType,
        confidence_score: float,
        train_file: str,
        test_file: str | None,
        submission_file: str | None,
        dataset_size_mb: float,
    ) -> CompetitionMetadata:
        """Generate competition metadata with optimization recommendations.

        Args:
            competition_name: Competition name
            competition_type: Detected competition type
            confidence_score: Detection confidence
            train_file: Training file path
            test_file: Test file path
            submission_file: Submission file path
            dataset_size_mb: Dataset size in MB

        Returns:
            CompetitionMetadata instance
        """
        # Base metadata
        metadata = CompetitionMetadata(
            competition_name=competition_name,
            train_file=train_file,
            test_file=test_file,
            submission_file=submission_file,
            dataset_size_mb=dataset_size_mb,
            competition_type=competition_type,
            auto_detected=True,
            confidence_score=confidence_score,
        )

        # Generate recommendations based on competition type
        if competition_type == CompetitionType.BINARY_CLASSIFICATION:
            metadata.recommended_head_type = "binary_classification"
            metadata.recommended_batch_size = 32
            metadata.recommended_learning_rate = 2e-5
            metadata.recommended_pooling = "cls"

        elif competition_type == CompetitionType.MULTICLASS_CLASSIFICATION:
            metadata.recommended_head_type = "multiclass_classification"
            metadata.recommended_batch_size = 32
            metadata.recommended_learning_rate = 2e-5
            metadata.recommended_pooling = "cls"

        elif competition_type == CompetitionType.MULTILABEL_CLASSIFICATION:
            metadata.recommended_head_type = "multilabel_classification"
            metadata.recommended_batch_size = 16
            metadata.recommended_learning_rate = 1e-5
            metadata.recommended_pooling = "attention"

        elif competition_type == CompetitionType.REGRESSION:
            metadata.recommended_head_type = "regression"
            metadata.recommended_batch_size = 32
            metadata.recommended_learning_rate = 2e-5
            metadata.recommended_pooling = "mean"

        elif competition_type == CompetitionType.ORDINAL_REGRESSION:
            metadata.recommended_head_type = "ordinal_regression"
            metadata.recommended_batch_size = 32
            metadata.recommended_learning_rate = 1e-5
            metadata.recommended_pooling = "cls"

        elif competition_type == CompetitionType.TIME_SERIES:
            metadata.recommended_head_type = (
                "regression"  # Default to regression for time series
            )
            metadata.recommended_batch_size = 64
            metadata.recommended_learning_rate = 1e-5
            metadata.recommended_pooling = "mean"
            metadata.recommended_max_length = 1024  # Longer sequences for time series

        # Adjust recommendations based on dataset size
        if dataset_size_mb > 500:  # Large dataset
            metadata.recommended_batch_size = min(
                64, metadata.recommended_batch_size * 2
            )
            metadata.optimal_prefetch_size = 8
            metadata.optimal_num_workers = 8
            metadata.enable_gradient_checkpointing = True

        elif dataset_size_mb < 50:  # Small dataset
            metadata.recommended_batch_size = max(
                16, metadata.recommended_batch_size // 2
            )
            metadata.optimal_prefetch_size = 2
            metadata.optimal_num_workers = 2
            metadata.recommended_learning_rate *= (
                0.5  # Lower learning rate for small datasets
            )

        return metadata

    def _generate_dataset_spec(
        self,
        competition_name: str,
        dataset_path: Path,
        competition_type: CompetitionType,
        train_data: pd.DataFrame,
        target_column: str | None,
        metadata: CompetitionMetadata,
    ) -> DatasetSpec:
        """Generate dataset specification based on analysis.

        Args:
            competition_name: Competition name
            dataset_path: Dataset directory path
            competition_type: Competition type
            train_data: Training data DataFrame
            target_column: Target column name
            metadata: Competition metadata

        Returns:
            DatasetSpec instance
        """
        # Analyze columns
        text_columns = []
        categorical_columns = []
        numerical_columns = []

        for col in train_data.columns:
            if col == target_column:
                continue

            dtype = train_data[col].dtype

            if dtype == "object":
                # Check if it's text or categorical
                unique_ratio = train_data[col].nunique() / len(train_data)
                avg_length = train_data[col].astype(str).str.len().mean()

                if unique_ratio > 0.8 or avg_length > 50:
                    text_columns.append(col)
                else:
                    categorical_columns.append(col)
            elif dtype in ["int64", "int32", "float64", "float32"]:
                # Check if it's categorical based on unique values
                unique_count = train_data[col].nunique()
                if unique_count <= 20 and dtype in ["int64", "int32"]:
                    categorical_columns.append(col)
                else:
                    numerical_columns.append(col)

        # Determine number of classes
        num_classes = None
        class_distribution = None
        is_balanced = True

        if target_column and target_column in train_data.columns:
            if competition_type in [
                CompetitionType.BINARY_CLASSIFICATION,
                CompetitionType.MULTICLASS_CLASSIFICATION,
                CompetitionType.MULTILABEL_CLASSIFICATION,
                CompetitionType.ORDINAL_REGRESSION,
            ]:
                num_classes = train_data[target_column].nunique()
                class_distribution = train_data[target_column].value_counts().to_dict()

                # Check balance
                counts = train_data[target_column].value_counts()
                min_count = counts.min()
                max_count = counts.max()
                is_balanced = (min_count / max_count) > 0.1

            elif competition_type == CompetitionType.REGRESSION:
                num_classes = 1

        # Create specification
        spec = DatasetSpec(
            competition_name=competition_name,
            dataset_path=dataset_path,
            competition_type=competition_type,
            num_samples=len(train_data),
            num_features=len(train_data.columns) - (1 if target_column else 0),
            target_column=target_column,
            text_columns=text_columns,
            categorical_columns=categorical_columns,
            numerical_columns=numerical_columns,
            num_classes=num_classes,
            class_distribution=class_distribution,
            is_balanced=is_balanced,
            recommended_batch_size=metadata.recommended_batch_size,
            recommended_max_length=metadata.recommended_max_length,
            use_unified_memory=metadata.use_unified_memory,
            prefetch_size=metadata.optimal_prefetch_size,
            num_workers=metadata.optimal_num_workers,
        )

        return spec

    def _calculate_dataset_size(self, data_path: Path) -> float:
        """Calculate total dataset size in MB.

        Args:
            data_path: Dataset directory or file path

        Returns:
            Size in megabytes
        """
        total_size = 0

        if data_path.is_file():
            total_size = data_path.stat().st_size
        else:
            for file in data_path.rglob("*"):
                if file.is_file():
                    total_size += file.stat().st_size

        return total_size / (1024 * 1024)  # Convert to MB
