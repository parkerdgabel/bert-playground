"""Data layer adapter implementing domain interfaces with MLX and infrastructure.

This adapter implements the domain interfaces using concrete infrastructure
components like MLX, file system, and caching.
"""

import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Union

import pandas as pd
from loguru import logger

# Domain interfaces
from domain.data.interfaces import (
    DataRepository, DataCache, TokenizerAdapter, DataLoader,
    DataValidatorAdapter, TextProcessorAdapter, DataAugmentationAdapter,
    TemplateEngine, DatasetFactory, MetricsCollector, FileSystemAdapter,
    ComputeBackendAdapter
)
from domain.data.models import (
    DatasetSpec, DataSample, DataBatch, Dataset, DataValidationResult
)

# Infrastructure components
from .loaders.mlx_loader import MLXDataLoader, MLXLoaderConfig
from .tokenizers.mlx_tokenizer import MLXTokenizer
from .cache.factory import create_cache
from .preprocessing.tokenizer_cache import TokenizerCache


class FileSystemDataRepository(DataRepository):
    """File system implementation of data repository."""
    
    def load_raw_data(self, file_path: str) -> Any:
        """Load raw data from file system."""
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"Data file not found: {file_path}")
        
        if file_path.suffix.lower() == '.csv':
            return pd.read_csv(file_path)
        elif file_path.suffix.lower() == '.json':
            with open(file_path, 'r') as f:
                return json.load(f)
        elif file_path.suffix.lower() == '.parquet':
            return pd.read_parquet(file_path)
        else:
            raise ValueError(f"Unsupported file format: {file_path.suffix}")
    
    def save_processed_data(self, data: Any, file_path: str) -> None:
        """Save processed data to file system."""
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        if isinstance(data, pd.DataFrame):
            if file_path.suffix.lower() == '.csv':
                data.to_csv(file_path, index=False)
            elif file_path.suffix.lower() == '.parquet':
                data.to_parquet(file_path, index=False)
            else:
                raise ValueError(f"Unsupported format for DataFrame: {file_path.suffix}")
        else:
            with open(file_path, 'w') as f:
                json.dump(data, f, indent=2)
    
    def exists(self, file_path: str) -> bool:
        """Check if file exists."""
        return Path(file_path).exists()


class MLXTokenizerAdapter(TokenizerAdapter):
    """MLX tokenizer implementation of tokenizer interface."""
    
    def __init__(self, model_name: str = "bert-base-uncased", max_length: int = 512):
        """Initialize MLX tokenizer."""
        self._tokenizer = MLXTokenizer(
            tokenizer_name=model_name,
            backend="auto",
            max_length=max_length
        )
    
    def tokenize(
        self,
        texts: Union[str, List[str]],
        max_length: int = 512,
        padding: bool = True,
        truncation: bool = True
    ) -> Dict[str, Any]:
        """Tokenize text(s) into model inputs."""
        return self._tokenizer(
            texts,
            padding=padding,
            truncation=truncation,
            max_length=max_length,
            return_tensors="mlx"
        )
    
    def decode(self, token_ids: List[int]) -> str:
        """Decode token IDs back to text."""
        if hasattr(self._tokenizer._tokenizer, 'decode'):
            return self._tokenizer._tokenizer.decode(token_ids)
        else:
            # Fallback for tokenizers without decode method
            return " ".join([str(id) for id in token_ids])
    
    @property
    def vocab_size(self) -> int:
        """Get vocabulary size."""
        return self._tokenizer.vocab_size


class MLXDataLoaderAdapter(DataLoader):
    """MLX data loader implementation of data loader interface."""
    
    def __init__(self, dataset: Dataset, config: MLXLoaderConfig, tokenizer: Optional[TokenizerAdapter] = None):
        """Initialize MLX data loader."""
        # Convert domain dataset to infrastructure format
        if isinstance(dataset.spec.dataset_path, Path):
            path = str(dataset.spec.dataset_path)
        else:
            path = dataset.spec.dataset_path
            
        # Create a simple dataset wrapper for MLX loader
        class SimpleDataset:
            def __init__(self, csv_path, text_column, label_column):
                self.df = pd.read_csv(csv_path)
                self.text_column = text_column
                self.label_column = label_column
                
            def __len__(self):
                return len(self.df)
                
            def __getitem__(self, idx):
                row = self.df.iloc[idx]
                return {
                    "text": row[self.text_column] if self.text_column else "",
                    "label": row[self.label_column] if self.label_column else None,
                    "metadata": {"idx": idx}
                }
        
        # Create infrastructure dataset
        infrastructure_dataset = SimpleDataset(
            csv_path=path,
            text_column=dataset.spec.text_columns[0] if dataset.spec.text_columns else None,
            label_column=dataset.spec.target_column
        )
        
        # Create MLX loader
        mlx_tokenizer = tokenizer._tokenizer if tokenizer else None
        self._loader = MLXDataLoader(
            dataset=infrastructure_dataset,
            config=config,
            tokenizer=mlx_tokenizer
        )
    
    def __iter__(self) -> Iterator[DataBatch]:
        """Iterate over data batches."""
        for batch_dict in self._loader:
            # Convert MLX batch to domain batch
            texts = batch_dict.get("text", [])
            labels = None
            metadata = batch_dict.get("metadata", [])
            
            if "labels" in batch_dict:
                import mlx.core as mx
                labels_array = batch_dict["labels"]
                if isinstance(labels_array, mx.array):
                    labels = labels_array.tolist()
                else:
                    labels = labels_array
            
            yield DataBatch(
                texts=texts if texts else [],
                labels=labels,
                metadata=metadata if metadata else []
            )
    
    def __len__(self) -> int:
        """Get number of batches."""
        return len(self._loader)
    
    @property
    def batch_size(self) -> int:
        """Get batch size."""
        return self._loader.config.batch_size


class DataValidationAdapter(DataValidatorAdapter):
    """Data validation implementation."""
    
    def validate_file_format(self, file_path: str) -> DataValidationResult:
        """Validate file format and structure."""
        result = DataValidationResult(is_valid=True)
        
        try:
            file_path = Path(file_path)
            
            if not file_path.exists():
                result.add_error(f"File does not exist: {file_path}")
                return result
            
            if file_path.suffix.lower() not in ['.csv', '.json', '.parquet']:
                result.add_error(f"Unsupported file format: {file_path.suffix}")
                return result
            
            # Try to load the file
            if file_path.suffix.lower() == '.csv':
                df = pd.read_csv(file_path, nrows=1)  # Just read first row to check format
                if df.empty:
                    result.add_error("CSV file is empty")
            elif file_path.suffix.lower() == '.json':
                with open(file_path, 'r') as f:
                    json.load(f)
            
        except Exception as e:
            result.add_error(f"File format validation failed: {str(e)}")
        
        return result
    
    def validate_data_quality(self, data: Any) -> DataValidationResult:
        """Validate data quality and integrity."""
        result = DataValidationResult(is_valid=True)
        
        try:
            if isinstance(data, pd.DataFrame):
                # Check for empty dataset
                if data.empty:
                    result.add_error("Dataset is empty")
                
                # Check for missing values
                missing_counts = data.isnull().sum()
                for col, count in missing_counts.items():
                    if count > 0:
                        percentage = (count / len(data)) * 100
                        if percentage > 50:  # More than 50% missing
                            result.add_error(f"Column '{col}' has {percentage:.1f}% missing values")
                
                # Check for duplicate rows
                duplicates = data.duplicated().sum()
                if duplicates > 0:
                    percentage = (duplicates / len(data)) * 100
                    if percentage > 10:  # More than 10% duplicates
                        result.add_error(f"Dataset has {percentage:.1f}% duplicate rows")
            
        except Exception as e:
            result.add_error(f"Data quality validation failed: {str(e)}")
        
        return result
    
    def validate_schema(self, data: Any, schema: Dict[str, Any]) -> DataValidationResult:
        """Validate data against schema."""
        result = DataValidationResult(is_valid=True)
        
        try:
            if isinstance(data, pd.DataFrame):
                required_columns = schema.get("required_columns", [])
                for col in required_columns:
                    if col not in data.columns:
                        result.add_error(f"Required column '{col}' is missing")
                
                # Check data types
                column_types = schema.get("column_types", {})
                for col, expected_type in column_types.items():
                    if col in data.columns:
                        actual_type = str(data[col].dtype)
                        if expected_type not in actual_type:
                            result.add_error(f"Column '{col}' has type '{actual_type}', expected '{expected_type}'")
        
        except Exception as e:
            result.add_error(f"Schema validation failed: {str(e)}")
        
        return result


class SimpleDatasetFactory(DatasetFactory):
    """Simple implementation of dataset factory."""
    
    def __init__(self, repository: DataRepository, cache: Optional[DataCache] = None):
        """Initialize dataset factory."""
        self.repository = repository
        self.cache = cache
    
    def create_dataset(
        self,
        spec: DatasetSpec,
        split: str = "train",
        cache: Optional[DataCache] = None
    ) -> Dataset:
        """Create dataset from specification."""
        # Return the domain Dataset model directly
        return Dataset(spec=spec, split=split)
    
    def create_dataloader(
        self,
        dataset: Dataset,
        batch_size: int = 32,
        shuffle: bool = False,
        tokenizer: Optional[TokenizerAdapter] = None
    ) -> DataLoader:
        """Create data loader from dataset."""
        config = MLXLoaderConfig(
            batch_size=batch_size,
            shuffle=shuffle
        )
        
        return MLXDataLoaderAdapter(dataset, config, tokenizer)


class SimpleFileSystemAdapter(FileSystemAdapter):
    """Simple file system adapter implementation."""
    
    def read_file(self, path: str) -> Any:
        """Read file from file system."""
        file_path = Path(path)
        
        if file_path.suffix.lower() == '.json':
            with open(file_path, 'r') as f:
                return json.load(f)
        elif file_path.suffix.lower() == '.csv':
            return pd.read_csv(file_path)
        else:
            with open(file_path, 'r') as f:
                return f.read()
    
    def write_file(self, path: str, data: Any) -> None:
        """Write file to file system."""
        file_path = Path(path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        if isinstance(data, dict) or isinstance(data, list):
            with open(file_path, 'w') as f:
                json.dump(data, f, indent=2)
        elif isinstance(data, pd.DataFrame):
            data.to_csv(file_path, index=False)
        else:
            with open(file_path, 'w') as f:
                f.write(str(data))
    
    def list_files(self, directory: str, pattern: Optional[str] = None) -> List[str]:
        """List files in directory matching pattern."""
        dir_path = Path(directory)
        
        if not dir_path.exists():
            return []
        
        if pattern:
            return [str(p) for p in dir_path.glob(pattern)]
        else:
            return [str(p) for p in dir_path.iterdir() if p.is_file()]
    
    def create_directory(self, path: str) -> None:
        """Create directory if it doesn't exist."""
        Path(path).mkdir(parents=True, exist_ok=True)


# Factory functions for creating adapters
def create_data_repository() -> DataRepository:
    """Create data repository adapter."""
    return FileSystemDataRepository()


def create_tokenizer_adapter(model_name: str = "bert-base-uncased", max_length: int = 512) -> TokenizerAdapter:
    """Create tokenizer adapter."""
    return MLXTokenizerAdapter(model_name, max_length)


def create_data_validator() -> DataValidatorAdapter:
    """Create data validator adapter."""
    return DataValidationAdapter()


def create_dataset_factory(repository: Optional[DataRepository] = None, cache: Optional[DataCache] = None) -> DatasetFactory:
    """Create dataset factory."""
    if repository is None:
        repository = create_data_repository()
    return SimpleDatasetFactory(repository, cache)


def create_filesystem_adapter() -> FileSystemAdapter:
    """Create file system adapter."""
    return SimpleFileSystemAdapter()