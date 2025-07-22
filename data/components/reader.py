"""Data reader component for handling various file formats.

This component is responsible for reading data from different file formats
(CSV, Parquet, JSON) and returning standardized pandas DataFrames.
"""

from pathlib import Path
from typing import Optional, Union

import pandas as pd
from loguru import logger


class DataReader:
    """Handles reading data from various file formats."""

    def __init__(self):
        """Initialize the data reader."""
        self._readers = {
            ".csv": self._read_csv,
            ".parquet": self._read_parquet,
            ".json": self._read_json,
            ".jsonl": self._read_jsonl,
        }

    def read(
        self,
        file_path: Union[str, Path],
        columns: Optional[list[str]] = None,
        nrows: Optional[int] = None,
    ) -> pd.DataFrame:
        """Read data from a file.

        Args:
            file_path: Path to the data file
            columns: Specific columns to read (if supported by format)
            nrows: Number of rows to read (for sampling)

        Returns:
            DataFrame with the loaded data

        Raises:
            ValueError: If file format is not supported
            FileNotFoundError: If file doesn't exist
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        suffix = file_path.suffix.lower()
        if suffix not in self._readers:
            raise ValueError(
                f"Unsupported file format: {suffix}. "
                f"Supported formats: {list(self._readers.keys())}"
            )

        logger.debug(f"Reading {suffix} file: {file_path}")
        return self._readers[suffix](file_path, columns, nrows)

    def _read_csv(
        self,
        file_path: Path,
        columns: Optional[list[str]] = None,
        nrows: Optional[int] = None,
    ) -> pd.DataFrame:
        """Read CSV file."""
        kwargs = {}
        if columns:
            kwargs["usecols"] = columns
        if nrows:
            kwargs["nrows"] = nrows

        return pd.read_csv(file_path, **kwargs)

    def _read_parquet(
        self,
        file_path: Path,
        columns: Optional[list[str]] = None,
        nrows: Optional[int] = None,
    ) -> pd.DataFrame:
        """Read Parquet file."""
        df = pd.read_parquet(file_path, columns=columns)
        if nrows:
            df = df.head(nrows)
        return df

    def _read_json(
        self,
        file_path: Path,
        columns: Optional[list[str]] = None,
        nrows: Optional[int] = None,
    ) -> pd.DataFrame:
        """Read JSON file."""
        df = pd.read_json(file_path)
        if columns:
            df = df[columns]
        if nrows:
            df = df.head(nrows)
        return df

    def _read_jsonl(
        self,
        file_path: Path,
        columns: Optional[list[str]] = None,
        nrows: Optional[int] = None,
    ) -> pd.DataFrame:
        """Read JSON Lines file."""
        df = pd.read_json(file_path, lines=True)
        if columns:
            df = df[columns]
        if nrows:
            df = df.head(nrows)
        return df

    def infer_file_format(self, file_path: Union[str, Path]) -> str:
        """Infer the file format from the file extension.

        Args:
            file_path: Path to the file

        Returns:
            File format string (e.g., 'csv', 'parquet')
        """
        file_path = Path(file_path)
        return file_path.suffix.lower().lstrip(".")

    def supports_format(self, format: str) -> bool:
        """Check if a file format is supported.

        Args:
            format: File format to check (with or without dot)

        Returns:
            True if format is supported
        """
        if not format.startswith("."):
            format = f".{format}"
        return format.lower() in self._readers