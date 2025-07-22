"""Column selector component for feature extraction and selection.

This component manages column selection, feature extraction, and column
type inference for datasets.
"""

from typing import Any, Optional

import pandas as pd
from loguru import logger


class ColumnSelector:
    """Manages column selection and feature extraction."""

    def __init__(self):
        """Initialize the column selector."""
        self._selection_rules: list[dict] = []
        self._column_types: dict[str, str] = {}

    def infer_column_types(self, data: pd.DataFrame) -> dict[str, str]:
        """Infer column types from data.

        Args:
            data: DataFrame to analyze

        Returns:
            Dictionary mapping column names to types
        """
        column_types = {}

        for col in data.columns:
            col_type = self._infer_single_column_type(data[col])
            column_types[col] = col_type

        # Cache the results
        self._column_types.update(column_types)

        logger.debug(f"Inferred types for {len(column_types)} columns")
        return column_types

    def _infer_single_column_type(self, series: pd.Series) -> str:
        """Infer type for a single column."""
        # Check if it's an ID column
        if self._is_id_column(series.name, series):
            return "id"

        # Check if it's a target column
        if self._is_target_column(series.name):
            return "target"

        # Check data type
        if pd.api.types.is_numeric_dtype(series):
            # Check if it's actually categorical
            unique_ratio = series.nunique() / len(series)
            if unique_ratio < 0.05 and series.nunique() < 20:
                return "categorical"
            else:
                return "numerical"

        elif pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series):
            # Check if it's text or categorical
            avg_length = series.astype(str).str.len().mean()
            unique_ratio = series.nunique() / len(series)

            if avg_length > 50 or unique_ratio > 0.8:
                return "text"
            else:
                return "categorical"

        elif pd.api.types.is_datetime64_any_dtype(series):
            return "datetime"

        elif pd.api.types.is_bool_dtype(series):
            return "boolean"

        else:
            return "unknown"

    def _is_id_column(self, col_name: str, series: pd.Series) -> bool:
        """Check if a column is likely an ID column."""
        # Name-based checks
        id_patterns = ["id", "Id", "ID", "_id", "_Id", "index", "Index"]
        if any(pattern in str(col_name) for pattern in id_patterns):
            # Verify it's actually an ID (unique values)
            if series.nunique() == len(series):
                return True

        # Check if it's a sequential integer
        if pd.api.types.is_integer_dtype(series):
            if series.is_monotonic_increasing and series.min() >= 0:
                return True

        return False

    def _is_target_column(self, col_name: str) -> bool:
        """Check if a column is likely a target column."""
        target_patterns = [
            "target", "Target", "label", "Label",
            "class", "Class", "y", "Y",
            "outcome", "Outcome", "result", "Result"
        ]
        return any(pattern in str(col_name) for pattern in target_patterns)

    def select_features(
        self,
        data: pd.DataFrame,
        include_types: Optional[list[str]] = None,
        exclude_types: Optional[list[str]] = None,
        include_columns: Optional[list[str]] = None,
        exclude_columns: Optional[list[str]] = None,
    ) -> list[str]:
        """Select feature columns based on criteria.

        Args:
            data: DataFrame to select from
            include_types: Column types to include
            exclude_types: Column types to exclude
            include_columns: Specific columns to include
            exclude_columns: Specific columns to exclude

        Returns:
            List of selected column names
        """
        # Ensure column types are inferred
        if not self._column_types:
            self.infer_column_types(data)

        # Start with all columns
        selected = set(data.columns)

        # Apply type filters
        if include_types:
            type_filtered = set()
            for col, col_type in self._column_types.items():
                if col_type in include_types and col in data.columns:
                    type_filtered.add(col)
            selected &= type_filtered

        if exclude_types:
            for col, col_type in self._column_types.items():
                if col_type in exclude_types and col in selected:
                    selected.discard(col)

        # Apply column filters
        if include_columns:
            selected &= set(include_columns)

        if exclude_columns:
            selected -= set(exclude_columns)

        return sorted(list(selected))

    def get_columns_by_type(self, data: pd.DataFrame, col_type: str) -> list[str]:
        """Get all columns of a specific type.

        Args:
            data: DataFrame to analyze
            col_type: Column type to filter by

        Returns:
            List of column names
        """
        if not self._column_types:
            self.infer_column_types(data)

        return [
            col for col, ctype in self._column_types.items()
            if ctype == col_type and col in data.columns
        ]

    def extract_text_columns(self, data: pd.DataFrame) -> list[str]:
        """Extract columns suitable for text processing.

        Args:
            data: DataFrame to analyze

        Returns:
            List of text column names
        """
        text_cols = self.get_columns_by_type(data, "text")

        # Also check for columns that might contain text descriptions
        for col in data.columns:
            if col not in text_cols:
                if any(keyword in col.lower() for keyword in 
                       ["description", "text", "comment", "note", "review"]):
                    if pd.api.types.is_object_dtype(data[col]):
                        text_cols.append(col)

        return text_cols

    def extract_feature_columns(
        self, data: pd.DataFrame, target_column: Optional[str] = None
    ) -> dict[str, list[str]]:
        """Extract feature columns organized by type.

        Args:
            data: DataFrame to analyze
            target_column: Target column to exclude

        Returns:
            Dictionary with lists of columns by type
        """
        if not self._column_types:
            self.infer_column_types(data)

        # Organize columns by type
        features = {
            "text": [],
            "categorical": [],
            "numerical": [],
            "datetime": [],
            "boolean": [],
        }

        for col in data.columns:
            # Skip target and ID columns
            if col == target_column:
                continue

            col_type = self._column_types.get(col, "unknown")
            if col_type in ["id", "target", "unknown"]:
                continue

            if col_type in features:
                features[col_type].append(col)

        return features

    def create_feature_summary(self, data: pd.DataFrame) -> dict[str, Any]:
        """Create a summary of features in the dataset.

        Args:
            data: DataFrame to summarize

        Returns:
            Dictionary with feature summary
        """
        if not self._column_types:
            self.infer_column_types(data)

        # Count columns by type
        type_counts = {}
        for col_type in self._column_types.values():
            type_counts[col_type] = type_counts.get(col_type, 0) + 1

        # Get feature columns
        features = self.extract_feature_columns(data)

        return {
            "total_columns": len(data.columns),
            "column_types": type_counts,
            "features_by_type": {
                k: len(v) for k, v in features.items()
            },
            "id_columns": self.get_columns_by_type(data, "id"),
            "target_columns": self.get_columns_by_type(data, "target"),
            "text_columns": features.get("text", []),
            "categorical_columns": features.get("categorical", []),
            "numerical_columns": features.get("numerical", []),
        }

    def suggest_text_template(self, data: pd.DataFrame) -> str:
        """Suggest a text template based on column analysis.

        Args:
            data: DataFrame to analyze

        Returns:
            Suggested template string
        """
        features = self.extract_feature_columns(data)

        template_parts = []

        # Add text columns
        for col in features.get("text", []):
            template_parts.append(f"{{{col}}}")

        # Add important categorical columns
        for col in features.get("categorical", [])[:5]:  # Limit to 5
            template_parts.append(f"{col}: {{{col}}}")

        # Add important numerical columns
        for col in features.get("numerical", [])[:5]:  # Limit to 5
            template_parts.append(f"{col}: {{{col}}}")

        return " | ".join(template_parts) if template_parts else "{text}"

    def add_selection_rule(self, rule: dict) -> None:
        """Add a custom selection rule.

        Args:
            rule: Dictionary defining the selection rule
        """
        self._selection_rules.append(rule)
        logger.debug(f"Added selection rule: {rule}")

    def apply_selection_rules(self, data: pd.DataFrame) -> list[str]:
        """Apply all selection rules to get final column list.

        Args:
            data: DataFrame to apply rules to

        Returns:
            List of selected columns
        """
        selected = set(data.columns)

        for rule in self._selection_rules:
            if rule.get("action") == "include":
                if "types" in rule:
                    cols = []
                    for col_type in rule["types"]:
                        cols.extend(self.get_columns_by_type(data, col_type))
                    selected &= set(cols)
                if "columns" in rule:
                    selected &= set(rule["columns"])

            elif rule.get("action") == "exclude":
                if "types" in rule:
                    for col_type in rule["types"]:
                        cols = self.get_columns_by_type(data, col_type)
                        selected -= set(cols)
                if "columns" in rule:
                    selected -= set(rule["columns"])

        return sorted(list(selected))