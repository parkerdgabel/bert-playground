"""Metadata manager component for dataset information.

This component manages dataset metadata, statistics, and competition information.
"""

from dataclasses import asdict
from typing import Any, Optional

import pandas as pd
from loguru import logger

from ..core.base import CompetitionType, DatasetSpec


class MetadataManager:
    """Manages dataset metadata and statistics."""

    def __init__(self, spec: DatasetSpec):
        """Initialize the metadata manager.

        Args:
            spec: Dataset specification
        """
        self.spec = spec
        self._statistics_cache: dict[str, Any] = {}

    def get_competition_info(self) -> dict[str, Any]:
        """Get competition information.

        Returns:
            Dictionary containing competition metadata
        """
        return {
            "competition_name": self.spec.competition_name,
            "competition_type": self.spec.competition_type.value,
            "dataset_path": str(self.spec.dataset_path),
            "num_features": self.spec.num_features,
            "target_column": self.spec.target_column,
            "num_classes": self.spec.num_classes,
            "is_balanced": self.spec.is_balanced,
        }

    def get_dataset_info(self, data: pd.DataFrame, split: str = "train") -> dict[str, Any]:
        """Get comprehensive dataset information.

        Args:
            data: Dataset DataFrame
            split: Dataset split name

        Returns:
            Dictionary containing dataset information
        """
        info = self.get_competition_info()
        info.update({
            "split": split,
            "num_samples": len(data),
            "actual_columns": list(data.columns),
            "memory_usage_mb": data.memory_usage(deep=True).sum() / 1024 / 1024,
            "dtypes": {col: str(dtype) for col, dtype in data.dtypes.items()},
        })

        # Add class distribution for training data
        if split == "train" and self.spec.target_column in data.columns:
            target_counts = data[self.spec.target_column].value_counts()
            info["class_distribution"] = target_counts.to_dict()
            info["class_percentages"] = (
                (target_counts / len(data) * 100).round(2).to_dict()
            )

        return info

    def compute_statistics(
        self, data: pd.DataFrame, cache_key: Optional[str] = None
    ) -> dict[str, Any]:
        """Compute detailed dataset statistics.

        Args:
            data: Dataset DataFrame
            cache_key: Optional key for caching statistics

        Returns:
            Dictionary containing detailed statistics
        """
        # Check cache
        if cache_key and cache_key in self._statistics_cache:
            logger.debug(f"Using cached statistics for key: {cache_key}")
            return self._statistics_cache[cache_key]

        stats = {
            "shape": data.shape,
            "columns": {
                "total": len(data.columns),
                "text": len([c for c in self.spec.text_columns if c in data.columns]),
                "categorical": len([c for c in self.spec.categorical_columns if c in data.columns]),
                "numerical": len([c for c in self.spec.numerical_columns if c in data.columns]),
            },
            "missing_values": self._compute_missing_stats(data),
            "column_statistics": self._compute_column_stats(data),
        }

        # Add text-specific statistics
        if self.spec.text_columns:
            stats["text_statistics"] = self._compute_text_stats(data)

        # Add target statistics
        if self.spec.target_column and self.spec.target_column in data.columns:
            stats["target_statistics"] = self._compute_target_stats(data)

        # Cache if requested
        if cache_key:
            self._statistics_cache[cache_key] = stats

        return stats

    def _compute_missing_stats(self, data: pd.DataFrame) -> dict[str, Any]:
        """Compute missing value statistics."""
        missing_counts = data.isnull().sum()
        missing_pcts = (missing_counts / len(data) * 100).round(2)

        return {
            "total_missing": missing_counts.sum(),
            "columns_with_missing": (missing_counts > 0).sum(),
            "by_column": {
                col: {
                    "count": int(missing_counts[col]),
                    "percentage": float(missing_pcts[col]),
                }
                for col in missing_counts.index
                if missing_counts[col] > 0
            },
        }

    def _compute_column_stats(self, data: pd.DataFrame) -> dict[str, dict]:
        """Compute per-column statistics."""
        stats = {}

        for col in data.columns:
            col_stats = {
                "dtype": str(data[col].dtype),
                "unique_count": data[col].nunique(),
                "null_count": data[col].isnull().sum(),
            }

            if pd.api.types.is_numeric_dtype(data[col]):
                col_stats.update({
                    "mean": float(data[col].mean()),
                    "std": float(data[col].std()),
                    "min": float(data[col].min()),
                    "max": float(data[col].max()),
                    "q25": float(data[col].quantile(0.25)),
                    "q50": float(data[col].quantile(0.50)),
                    "q75": float(data[col].quantile(0.75)),
                })
            elif pd.api.types.is_object_dtype(data[col]):
                # Sample most common values
                top_values = data[col].value_counts().head(5)
                col_stats["top_values"] = {
                    str(k): int(v) for k, v in top_values.items()
                }

            stats[col] = col_stats

        return stats

    def _compute_text_stats(self, data: pd.DataFrame) -> dict[str, Any]:
        """Compute text-specific statistics."""
        text_stats = {}

        for col in self.spec.text_columns:
            if col not in data.columns:
                continue

            text_data = data[col].astype(str)
            lengths = text_data.str.len()

            text_stats[col] = {
                "avg_length": float(lengths.mean()),
                "max_length": int(lengths.max()),
                "min_length": int(lengths.min()),
                "std_length": float(lengths.std()),
                "empty_count": (text_data == "").sum(),
                "avg_words": float(text_data.str.split().str.len().mean()),
            }

        return text_stats

    def _compute_target_stats(self, data: pd.DataFrame) -> dict[str, Any]:
        """Compute target column statistics."""
        target = data[self.spec.target_column]

        stats = {
            "dtype": str(target.dtype),
            "unique_values": target.nunique(),
            "null_count": target.isnull().sum(),
        }

        if self.spec.competition_type.value.endswith("classification"):
            # Classification statistics
            value_counts = target.value_counts()
            stats.update({
                "class_counts": value_counts.to_dict(),
                "class_balance": {
                    "min_samples": int(value_counts.min()),
                    "max_samples": int(value_counts.max()),
                    "imbalance_ratio": float(value_counts.max() / value_counts.min()),
                },
            })
        else:
            # Regression statistics
            stats.update({
                "mean": float(target.mean()),
                "std": float(target.std()),
                "min": float(target.min()),
                "max": float(target.max()),
                "skewness": float(target.skew()),
                "kurtosis": float(target.kurtosis()),
            })

        return stats

    def get_optimization_hints(self, data: pd.DataFrame) -> dict[str, Any]:
        """Get optimization hints based on dataset characteristics.

        Args:
            data: Dataset DataFrame

        Returns:
            Dictionary containing optimization recommendations
        """
        hints = {
            "batch_size": self.spec.recommended_batch_size,
            "max_length": self.spec.recommended_max_length,
            "prefetch_size": self.spec.prefetch_size,
        }

        # Adjust based on dataset size
        num_samples = len(data)
        if num_samples > 100000:
            hints["batch_size"] = min(64, hints["batch_size"])
            hints["prefetch_size"] = 8
            hints["enable_caching"] = True
        elif num_samples < 5000:
            hints["batch_size"] = max(16, hints["batch_size"])
            hints["prefetch_size"] = 2

        # Adjust based on text lengths if text column exists
        if "text" in data.columns:
            text_lengths = data["text"].str.len()
            p95_length = text_lengths.quantile(0.95)
            hints["suggested_max_length"] = min(
                int(p95_length * 1.1),  # 10% padding
                self.spec.recommended_max_length
            )

        # Memory optimization hints
        memory_usage_mb = data.memory_usage(deep=True).sum() / 1024 / 1024
        if memory_usage_mb > 1000:  # > 1GB
            hints["use_memory_efficient_loading"] = True
            hints["use_streaming"] = True

        return hints

    def export_metadata(self) -> dict[str, Any]:
        """Export all metadata as a dictionary.

        Returns:
            Complete metadata dictionary
        """
        return {
            "specification": asdict(self.spec),
            "competition_info": self.get_competition_info(),
            "cached_statistics": self._statistics_cache,
        }