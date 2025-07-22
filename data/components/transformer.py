"""Data transformer component for preprocessing and transformations.

This component handles all data transformations including text conversion,
feature engineering, and preprocessing operations.
"""

from typing import Any, Callable, Optional

import pandas as pd
from loguru import logger

from ..core.base import CompetitionType, DatasetSpec


class DataTransformer:
    """Handles data transformations and preprocessing."""

    def __init__(self, spec: DatasetSpec):
        """Initialize the transformer with dataset specification.

        Args:
            spec: Dataset specification for transformation context
        """
        self.spec = spec
        self._custom_transforms: list[Callable] = []

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Apply all transformations to the data.

        Args:
            data: Input DataFrame

        Returns:
            Transformed DataFrame
        """
        # Create a copy to avoid modifying original
        data = data.copy()

        # Apply standard transformations
        data = self._handle_missing_values(data)
        data = self._normalize_text_columns(data)
        data = self._encode_categorical_columns(data)
        data = self._scale_numerical_columns(data)

        # Apply custom transformations
        for transform in self._custom_transforms:
            data = transform(data)

        return data

    def add_custom_transform(self, transform: Callable[[pd.DataFrame], pd.DataFrame]) -> None:
        """Add a custom transformation function.

        Args:
            transform: Function that takes and returns a DataFrame
        """
        self._custom_transforms.append(transform)
        logger.debug(f"Added custom transform: {transform.__name__}")

    def create_text_representation(self, row: pd.Series) -> str:
        """Convert a data row to text representation.

        Args:
            row: Pandas Series representing a single data row

        Returns:
            Text representation of the row
        """
        if self.spec.text_template:
            # Use custom template if provided
            return self._apply_template(row, self.spec.text_template)

        # Default text generation based on column types
        text_parts = []

        # Add text columns first
        for col in self.spec.text_columns:
            if col in row.index and pd.notna(row[col]):
                text_parts.append(str(row[col]))

        # Add categorical columns with labels
        for col in self.spec.categorical_columns:
            if col in row.index and pd.notna(row[col]):
                text_parts.append(f"{col}: {row[col]}")

        # Add numerical columns with labels
        for col in self.spec.numerical_columns:
            if col in row.index and pd.notna(row[col]):
                text_parts.append(f"{col}: {row[col]}")

        return " | ".join(text_parts)

    def _apply_template(self, row: pd.Series, template: str) -> str:
        """Apply a text template to a data row.

        Args:
            row: Data row
            template: Template string with {column_name} placeholders

        Returns:
            Formatted text
        """
        # Replace placeholders with actual values
        text = template
        for col in row.index:
            placeholder = f"{{{col}}}"
            if placeholder in text:
                value = row[col] if pd.notna(row[col]) else ""
                text = text.replace(placeholder, str(value))
        return text

    def _handle_missing_values(self, data: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values in the data."""
        # Fill missing text columns with empty strings
        for col in self.spec.text_columns:
            if col in data.columns:
                data[col] = data[col].fillna("")

        # Fill missing categorical columns with 'unknown'
        for col in self.spec.categorical_columns:
            if col in data.columns:
                data[col] = data[col].fillna("unknown")

        # Fill missing numerical columns with median or 0
        for col in self.spec.numerical_columns:
            if col in data.columns:
                if data[col].notna().any():
                    median_val = data[col].median()
                    data[col] = data[col].fillna(median_val)
                else:
                    data[col] = data[col].fillna(0)

        return data

    def _normalize_text_columns(self, data: pd.DataFrame) -> pd.DataFrame:
        """Normalize text columns."""
        for col in self.spec.text_columns:
            if col in data.columns:
                # Convert to string and strip whitespace
                data[col] = data[col].astype(str).str.strip()
                
                # Replace multiple spaces with single space
                data[col] = data[col].str.replace(r'\s+', ' ', regex=True)
                
                # Handle empty strings
                data[col] = data[col].replace(['', 'nan'], '')

        return data

    def _encode_categorical_columns(self, data: pd.DataFrame) -> pd.DataFrame:
        """Encode categorical columns for BERT processing."""
        # For BERT, we typically convert categoricals to text descriptions
        # rather than one-hot encoding
        for col in self.spec.categorical_columns:
            if col in data.columns:
                # Ensure string type
                data[col] = data[col].astype(str)
                
                # Create descriptive text
                data[f"{col}_text"] = data[col].apply(
                    lambda x: f"{col} is {x}" if x != "unknown" else ""
                )

        return data

    def _scale_numerical_columns(self, data: pd.DataFrame) -> pd.DataFrame:
        """Scale numerical columns if needed."""
        # For BERT, we typically convert numbers to text descriptions
        # rather than scaling them
        for col in self.spec.numerical_columns:
            if col in data.columns:
                # Create text representations of numerical values
                data[f"{col}_text"] = data[col].apply(
                    lambda x: f"{col} is {x:.2f}" if pd.notna(x) else ""
                )

        return data

    def prepare_for_bert(self, data: pd.DataFrame) -> pd.DataFrame:
        """Prepare data specifically for BERT processing.

        Args:
            data: Input DataFrame

        Returns:
            DataFrame with added 'text' column for BERT
        """
        # Apply transformations
        data = self.transform(data)

        # Create text representation for each row
        data["text"] = data.apply(self.create_text_representation, axis=1)

        return data

    def split_features_target(
        self, data: pd.DataFrame
    ) -> tuple[pd.DataFrame, Optional[pd.Series]]:
        """Split data into features and target.

        Args:
            data: Input DataFrame

        Returns:
            Tuple of (features, target) where target is None if no target column
        """
        if self.spec.target_column and self.spec.target_column in data.columns:
            features = data.drop(columns=[self.spec.target_column])
            target = data[self.spec.target_column]
            return features, target
        else:
            return data, None

    def get_text_statistics(self, data: pd.DataFrame) -> dict[str, Any]:
        """Get statistics about text representations.

        Args:
            data: DataFrame with text representations

        Returns:
            Dictionary of text statistics
        """
        if "text" not in data.columns:
            data = self.prepare_for_bert(data)

        text_lengths = data["text"].str.len()

        return {
            "avg_length": text_lengths.mean(),
            "max_length": text_lengths.max(),
            "min_length": text_lengths.min(),
            "std_length": text_lengths.std(),
            "percentiles": {
                "25%": text_lengths.quantile(0.25),
                "50%": text_lengths.quantile(0.50),
                "75%": text_lengths.quantile(0.75),
                "95%": text_lengths.quantile(0.95),
                "99%": text_lengths.quantile(0.99),
            },
        }