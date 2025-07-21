"""Factory functions for creating datasets and data loaders.

This module provides convenient factory functions to create datasets and data loaders
for the training pipeline, following the same pattern as the models and training factories.
"""

from pathlib import Path
from typing import Any

import pandas as pd
from loguru import logger

from .augmentation import (
    AugmentationConfig,
    CompetitionTemplateAugmenter,
    TitanicAugmenter,
    get_registry,
)
from .core import (
    CompetitionType,
    DatasetSpec,
    KaggleDataset,
)
from .loaders.mlx_loader import MLXDataLoader, MLXLoaderConfig
from .preprocessing.tokenizer_cache import PreTokenizedDataset, TokenizerCache


class CSVDataset(KaggleDataset):
    """Simple CSV dataset implementation for Kaggle competitions."""

    def __init__(
        self,
        csv_path: str | Path,
        spec: DatasetSpec | None = None,
        text_column: str | None = None,
        label_column: str | None = None,
        text_converter: Any | None = None,
        augmenter: Any | None = None,
        tokenizer: Any | None = None,  # Handle tokenizer separately
        **kwargs,
    ):
        """Initialize CSV dataset.

        Args:
            csv_path: Path to CSV file
            spec: Dataset specification
            text_column: Column containing text data
            label_column: Column containing labels
            text_converter: Optional text converter for tabular data
            tokenizer: Optional tokenizer (stored but not passed to parent)
            **kwargs: Additional arguments for parent class
        """
        self.csv_path = Path(csv_path)
        self.text_column = text_column
        self.label_column = label_column
        self.text_converter = text_converter
        self.augmenter = augmenter
        self.tokenizer = tokenizer  # Store tokenizer

        # Create default spec if not provided
        if spec is None:
            spec = self._create_default_spec()

        # Filter out parameters that KaggleDataset expects
        split = kwargs.pop("split", "train")
        transform = kwargs.pop("transform", None)
        cache_dir = kwargs.pop("cache_dir", None)

        super().__init__(spec, split=split, transform=transform, cache_dir=cache_dir)

    def _create_default_spec(self) -> DatasetSpec:
        """Create a default dataset specification."""
        # Try to infer from filename
        filename = self.csv_path.stem.lower()

        if "titanic" in filename:
            competition_type = CompetitionType.BINARY_CLASSIFICATION
            num_classes = 2
            competition_name = "titanic"
        elif "train" in filename or "test" in filename:
            competition_type = CompetitionType.UNKNOWN
            num_classes = None
            competition_name = "unknown"
        else:
            competition_type = CompetitionType.UNKNOWN
            num_classes = None
            competition_name = filename

        return DatasetSpec(
            competition_name=competition_name,
            dataset_path=self.csv_path,
            competition_type=competition_type,
            num_samples=0,  # Will be updated in _load_data
            num_features=0,  # Will be updated in _load_data
            target_column=self.label_column,
            num_classes=num_classes,
        )

    def _load_data(self) -> None:
        """Load data from CSV file."""
        if not self.csv_path.exists():
            raise FileNotFoundError(f"CSV file not found: {self.csv_path}")

        # Load data
        self._data = pd.read_csv(self.csv_path)

        # Update spec with actual data info
        self.spec.num_samples = len(self._data)
        self.spec.num_features = len(self._data.columns)

        # Identify columns if not specified
        if self.text_column is None:
            # Look for text-like columns
            text_candidates = []
            for col in self._data.columns:
                if self._data[col].dtype == "object":
                    avg_length = self._data[col].astype(str).str.len().mean()
                    if avg_length > 20:  # Likely text column
                        text_candidates.append(col)

            if text_candidates:
                self.text_column = text_candidates[0]
                logger.info(f"Auto-detected text column: {self.text_column}")

        # Identify label column if not specified
        if self.label_column is None and self.split == "train":
            # Common label column names
            label_candidates = [
                "label",
                "labels",
                "target",
                "class",
                "Survived",
                "Label",
                "Target",
                "Class",
                "y",
            ]
            for col in label_candidates:
                if col in self._data.columns:
                    self.label_column = col
                    self.spec.target_column = col
                    logger.info(f"Auto-detected label column: {self.label_column}")
                    break

    def _validate_data(self) -> None:
        """Validate the loaded data."""
        if self._data is None:
            raise ValueError("Data not loaded")

        # Check if required columns exist
        if self.text_column and self.text_column not in self._data.columns:
            # If no text column, we'll convert the entire row to text
            logger.warning(
                f"Text column '{self.text_column}' not found. Will use row-to-text conversion."
            )
            self.text_column = None

        if self.label_column and self.label_column not in self._data.columns:
            if self.split == "train":
                raise ValueError(
                    f"Label column '{self.label_column}' not found in data"
                )

    def __getitem__(self, index: int) -> dict[str, Any]:
        """Get a single sample by index."""
        if index < 0 or index >= len(self._data):
            raise IndexError(
                f"Index {index} out of range for dataset of size {len(self._data)}"
            )

        # Get row data
        row = self._data.iloc[index]
        sample = {}

        # Handle text data
        if self.text_column and self.text_column in row:
            sample["text"] = str(row[self.text_column])
        else:
            # Convert entire row to text if no text column
            row_dict = row.to_dict()

            # Apply augmentation if available
            if self.augmenter:
                augmented = self.augmenter.augment(row_dict)
                if "text" in augmented:
                    sample["text"] = augmented["text"]
                else:
                    # Fall back to simple conversion
                    sample["text"] = self._simple_text_conversion(row_dict)
            elif self.text_converter:
                sample["text"] = self.text_converter.convert(row_dict)
            else:
                # Simple conversion: key-value pairs
                sample["text"] = self._simple_text_conversion(row_dict)

        # Handle labels
        if self.label_column and self.label_column in row:
            label_value = row[self.label_column]
            if pd.notna(label_value):
                sample["labels"] = (
                    int(label_value)
                    if self.spec.competition_type
                    in [
                        CompetitionType.BINARY_CLASSIFICATION,
                        CompetitionType.MULTICLASS_CLASSIFICATION,
                    ]
                    else float(label_value)
                )

        # Add metadata
        sample["metadata"] = {
            "index": index,
            "split": self.split,
        }

        # Add ID if available
        id_columns = ["id", "Id", "ID", "PassengerId", "test_id"]
        for col in id_columns:
            if col in row:
                sample["metadata"]["id"] = row[col]
                break

        return sample

    def _simple_text_conversion(self, row_dict: dict[str, Any]) -> str:
        """Simple key-value text conversion."""
        text_parts = []
        for col, val in row_dict.items():
            if col != self.label_column and pd.notna(val):
                text_parts.append(f"{col}: {val}")
        return " | ".join(text_parts)


def create_dataset(
    data_path: str | Path,
    dataset_type: str = "csv",
    competition_name: str | None = None,
    text_column: str | None = None,
    label_column: str | None = None,
    split: str = "train",
    **kwargs,
) -> KaggleDataset:
    """Create a dataset instance.

    Args:
        data_path: Path to data file
        dataset_type: Type of dataset (currently only "csv" supported)
        competition_name: Name of the competition
        text_column: Column containing text data
        label_column: Column containing labels
        split: Dataset split ("train", "val", "test")
        **kwargs: Additional arguments for dataset

    Returns:
        Dataset instance
    """
    data_path = Path(data_path)

    if dataset_type == "csv":
        # Create augmenter if needed for text conversion
        augmenter = None
        if text_column is None and competition_name:
            # Check if we have a specific augmenter for this competition
            registry = get_registry()
            if (
                competition_name.lower() == "titanic"
                and "titanic" in registry.list_strategies()
            ):
                augmenter = TitanicAugmenter()
            else:
                # Use generic template augmenter
                augmenter = CompetitionTemplateAugmenter.from_competition_name(
                    competition_name, config=AugmentationConfig()
                )

        dataset = CSVDataset(
            csv_path=data_path,
            text_column=text_column,
            label_column=label_column,
            augmenter=augmenter,
            split=split,
            **kwargs,
        )

        # Update competition name if provided
        if competition_name:
            dataset.spec.competition_name = competition_name

        return dataset
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")


def create_dataloader(
    dataset: KaggleDataset | None = None,
    data_path: str | Path | None = None,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 0,
    prefetch_size: int = 0,
    mlx_prefetch_size: int | None = None,
    mlx_tokenizer_chunk_size: int = 100,
    tokenizer=None,
    tokenizer_backend: str = "auto",
    max_length: int = 512,
    use_pretokenized: bool = False,
    pretokenized_cache_dir: str = "data/cache/tokenized",
    force_rebuild_cache: bool = False,
    **kwargs,
) -> MLXDataLoader:
    """Create a data loader instance.

    Args:
        dataset: Dataset instance (if not provided, will create from data_path)
        data_path: Path to data file (used if dataset not provided)
        batch_size: Batch size
        shuffle: Whether to shuffle data
        num_workers: Number of worker threads
        prefetch_size: Number of batches to prefetch
        tokenizer: Tokenizer for text processing
        max_length: Maximum sequence length
        **kwargs: Additional arguments

    Returns:
        DataLoader instance
    """
    # Create MLX tokenizer if needed
    if tokenizer is None and tokenizer_backend in ["mlx", "auto"]:
        from .tokenizers import MLXTokenizer

        tokenizer = MLXTokenizer(backend=tokenizer_backend, max_length=max_length)

    # Create dataset if not provided
    if dataset is None:
        if data_path is None:
            raise ValueError("Either dataset or data_path must be provided")
        dataset = create_dataset(data_path, **kwargs)

    # Create loader config
    config = MLXLoaderConfig(
        batch_size=batch_size,
        shuffle=shuffle,
        max_length=max_length,
        prefetch_size=mlx_prefetch_size
        if mlx_prefetch_size is not None
        else prefetch_size,
        tokenizer_chunk_size=mlx_tokenizer_chunk_size,
        use_pretokenized=use_pretokenized,
        pretokenized_cache_dir=pretokenized_cache_dir,
        **{k: v for k, v in kwargs.items() if hasattr(MLXLoaderConfig, k)},
    )

    # Handle pre-tokenization if enabled
    pretokenized_data = None
    if use_pretokenized and tokenizer is not None:
        logger.info("Pre-tokenizing dataset for optimal performance...")

        # Create tokenizer cache
        cache = TokenizerCache(
            cache_dir=pretokenized_cache_dir, max_length=max_length, tokenizer=tokenizer
        )

        # Extract texts and labels from dataset
        texts = []
        labels = []
        for i in range(len(dataset)):
            sample = dataset[i]
            texts.append(sample["text"])
            if "labels" in sample:
                labels.append(sample["labels"])

        # Pre-tokenize and cache
        split = kwargs.get("split", "train")
        tokenized_data = cache.tokenize_and_cache(
            texts=texts,
            labels=labels if labels else None,
            dataset_path=data_path,
            split=split,
            force_rebuild=force_rebuild_cache,
        )

        # Create pre-tokenized dataset
        pretokenized_data = PreTokenizedDataset(tokenized_data)
        logger.info(
            f"Using pre-tokenized dataset with {len(pretokenized_data)} samples"
        )

    # Create loader
    loader = MLXDataLoader(
        dataset=dataset,
        config=config,
        tokenizer=tokenizer,
        pretokenized_data=pretokenized_data,
    )

    return loader


def create_data_pipeline(
    train_path: str | Path,
    val_path: str | Path | None = None,
    test_path: str | Path | None = None,
    competition_name: str | None = None,
    batch_size: int = 32,
    eval_batch_size: int | None = None,
    tokenizer=None,
    mlx_prefetch_size: int | None = None,
    mlx_tokenizer_chunk_size: int = 100,
    **kwargs,
) -> dict[str, MLXDataLoader]:
    """Create a complete data pipeline with train/val/test loaders.

    Args:
        train_path: Path to training data
        val_path: Path to validation data (optional)
        test_path: Path to test data (optional)
        competition_name: Name of the competition
        batch_size: Training batch size
        eval_batch_size: Evaluation batch size (defaults to 2*batch_size)
        tokenizer: Tokenizer for text processing
        **kwargs: Additional arguments

    Returns:
        Dictionary with 'train', 'val', 'test' loaders (as available)
    """
    if eval_batch_size is None:
        eval_batch_size = batch_size * 2

    loaders = {}

    # Create train loader
    loaders["train"] = create_dataloader(
        data_path=train_path,
        competition_name=competition_name,
        batch_size=batch_size,
        shuffle=True,
        split="train",
        tokenizer=tokenizer,
        mlx_prefetch_size=mlx_prefetch_size,
        mlx_tokenizer_chunk_size=mlx_tokenizer_chunk_size,
        **kwargs,
    )

    # Create validation loader if path provided
    if val_path:
        loaders["val"] = create_dataloader(
            data_path=val_path,
            competition_name=competition_name,
            batch_size=eval_batch_size,
            shuffle=False,
            split="val",
            tokenizer=tokenizer,
            mlx_prefetch_size=mlx_prefetch_size,
            mlx_tokenizer_chunk_size=mlx_tokenizer_chunk_size,
            **kwargs,
        )

    # Create test loader if path provided
    if test_path:
        loaders["test"] = create_dataloader(
            data_path=test_path,
            competition_name=competition_name,
            batch_size=eval_batch_size,
            shuffle=False,
            split="test",
            tokenizer=tokenizer,
            mlx_prefetch_size=mlx_prefetch_size,
            mlx_tokenizer_chunk_size=mlx_tokenizer_chunk_size,
            **kwargs,
        )

    logger.info(f"Created data pipeline with loaders: {list(loaders.keys())}")

    return loaders
