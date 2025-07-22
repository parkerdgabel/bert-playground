"""Data validator component for ensuring data integrity.

This component validates data integrity, schema compliance, and consistency
with dataset specifications.
"""

from typing import Any, Optional

import pandas as pd
from loguru import logger

from ..core.base import DatasetSpec


class DataValidator:
    """Handles data validation and integrity checks."""

    def __init__(self, spec: DatasetSpec):
        """Initialize the validator with dataset specification.

        Args:
            spec: Dataset specification to validate against
        """
        self.spec = spec

    def validate(self, data: pd.DataFrame, split: str = "train") -> list[str]:
        """Validate data against the specification.

        Args:
            data: DataFrame to validate
            split: Dataset split being validated

        Returns:
            List of validation errors (empty if valid)
        """
        errors = []

        # Basic validation
        if data.empty:
            errors.append("Data is empty")
            return errors

        # Column validation
        errors.extend(self._validate_columns(data, split))

        # Type validation
        errors.extend(self._validate_types(data))

        # Value validation
        errors.extend(self._validate_values(data))

        # Target validation (for training data)
        if split == "train" and self.spec.target_column:
            errors.extend(self._validate_target(data))

        # Log validation results
        if errors:
            logger.warning(f"Validation failed with {len(errors)} errors")
            for error in errors[:5]:  # Show first 5 errors
                logger.warning(f"  - {error}")
            if len(errors) > 5:
                logger.warning(f"  ... and {len(errors) - 5} more errors")
        else:
            logger.debug("Data validation passed")

        return errors

    def _validate_columns(self, data: pd.DataFrame, split: str) -> list[str]:
        """Validate required columns are present."""
        errors = []
        columns = set(data.columns)

        # Check text columns
        missing_text = set(self.spec.text_columns) - columns
        if missing_text:
            errors.append(f"Missing text columns: {missing_text}")

        # Check categorical columns
        missing_cat = set(self.spec.categorical_columns) - columns
        if missing_cat:
            errors.append(f"Missing categorical columns: {missing_cat}")

        # Check numerical columns
        missing_num = set(self.spec.numerical_columns) - columns
        if missing_num:
            errors.append(f"Missing numerical columns: {missing_num}")

        # Check target column for training data
        if split == "train" and self.spec.target_column:
            if self.spec.target_column not in columns:
                errors.append(f"Missing target column: {self.spec.target_column}")

        # Check for expected number of features
        expected_features = self.spec.num_features
        actual_features = len(columns)
        if split == "train" and self.spec.target_column:
            actual_features -= 1  # Exclude target from feature count

        if expected_features != actual_features:
            logger.warning(
                f"Feature count mismatch: expected {expected_features}, "
                f"got {actual_features}"
            )

        return errors

    def _validate_types(self, data: pd.DataFrame) -> list[str]:
        """Validate column data types."""
        errors = []

        # Validate numerical columns
        for col in self.spec.numerical_columns:
            if col in data.columns:
                if not pd.api.types.is_numeric_dtype(data[col]):
                    errors.append(f"Column '{col}' should be numeric")

        # Validate text columns
        for col in self.spec.text_columns:
            if col in data.columns:
                if not pd.api.types.is_object_dtype(data[col]) and \
                   not pd.api.types.is_string_dtype(data[col]):
                    errors.append(f"Column '{col}' should be text/string")

        return errors

    def _validate_values(self, data: pd.DataFrame) -> list[str]:
        """Validate data values and ranges."""
        errors = []

        # Check for missing values in critical columns
        critical_columns = self.spec.text_columns + [self.spec.target_column]
        critical_columns = [c for c in critical_columns if c and c in data.columns]

        for col in critical_columns:
            missing_count = data[col].isnull().sum()
            if missing_count > 0:
                missing_pct = (missing_count / len(data)) * 100
                if missing_pct > 50:
                    errors.append(
                        f"Column '{col}' has {missing_pct:.1f}% missing values"
                    )

        # Check for infinite values in numerical columns
        for col in self.spec.numerical_columns:
            if col in data.columns:
                import numpy as np
                inf_count = np.isinf(data[col]).sum()
                if inf_count > 0:
                    errors.append(f"Column '{col}' contains {inf_count} infinite values")

        return errors

    def _validate_target(self, data: pd.DataFrame) -> list[str]:
        """Validate target column for classification/regression tasks."""
        errors = []
        target_col = self.spec.target_column

        if target_col not in data.columns:
            return errors

        target_data = data[target_col]
        unique_values = target_data.nunique()

        # Validate based on competition type
        if self.spec.competition_type.value.endswith("classification"):
            # Check number of classes
            if self.spec.num_classes and unique_values != self.spec.num_classes:
                errors.append(
                    f"Expected {self.spec.num_classes} classes, "
                    f"found {unique_values}"
                )

            # Check class distribution
            if self.spec.is_balanced:
                value_counts = target_data.value_counts()
                max_count = value_counts.max()
                min_count = value_counts.min()
                imbalance_ratio = max_count / min_count if min_count > 0 else float('inf')

                if imbalance_ratio > 10:
                    errors.append(
                        f"Severe class imbalance detected: "
                        f"ratio {imbalance_ratio:.2f}"
                    )

        elif self.spec.competition_type.value == "regression":
            # Check for reasonable value ranges
            if pd.api.types.is_numeric_dtype(target_data):
                std_dev = target_data.std()
                if std_dev == 0:
                    errors.append("Target values have zero variance")

        return errors

    def get_validation_report(self, data: pd.DataFrame) -> dict[str, Any]:
        """Generate a detailed validation report.

        Args:
            data: DataFrame to analyze

        Returns:
            Dictionary containing validation metrics and statistics
        """
        errors = self.validate(data)

        report = {
            "valid": len(errors) == 0,
            "error_count": len(errors),
            "errors": errors,
            "data_shape": data.shape,
            "memory_usage": data.memory_usage(deep=True).sum(),
            "column_count": len(data.columns),
            "row_count": len(data),
        }

        # Add column-level statistics
        col_stats = {}
        for col in data.columns:
            stats = {
                "dtype": str(data[col].dtype),
                "null_count": data[col].isnull().sum(),
                "unique_count": data[col].nunique(),
            }

            if pd.api.types.is_numeric_dtype(data[col]):
                stats.update({
                    "mean": data[col].mean(),
                    "std": data[col].std(),
                    "min": data[col].min(),
                    "max": data[col].max(),
                })

            col_stats[col] = stats

        report["column_stats"] = col_stats

        return report