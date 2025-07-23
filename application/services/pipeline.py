"""Data Pipeline Service - orchestrates complex data processing workflows."""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List, Dict, Any, Union, Callable
import asyncio
from enum import Enum
import pandas as pd

from domain.entities.dataset import Dataset
from domain.services import TokenizationService
from ports.secondary.data import DataLoaderPort
from ports.secondary.storage import StorageService as StoragePort
from ports.secondary.monitoring import MonitoringService as MonitoringPort
from ports.secondary.tokenizer import TokenizerPort


class DataProcessingStep(Enum):
    """Types of data processing steps."""
    LOAD = "load"
    CLEAN = "clean"
    TRANSFORM = "transform"
    AUGMENT = "augment"
    TOKENIZE = "tokenize"
    SPLIT = "split"
    VALIDATE = "validate"
    CACHE = "cache"


@dataclass
class DataPipelineConfig:
    """Configuration for data pipeline."""
    steps: List[Dict[str, Any]]
    input_format: str  # "csv", "json", "parquet", "text"
    output_format: str  # "dataset", "dataloader", "cached"
    
    # Processing options
    text_column: Optional[str] = None
    label_column: Optional[str] = None
    feature_columns: Optional[List[str]] = None
    
    # Splitting options
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    test_ratio: float = 0.1
    stratify: bool = True
    
    # Tokenization options
    max_length: int = 512
    truncation: bool = True
    padding: str = "max_length"
    
    # Augmentation options
    augmentation_ratio: float = 0.0
    augmentation_strategies: List[str] = None
    
    # Caching options
    use_cache: bool = True
    cache_dir: Path = Path(".cache/data")
    
    # Validation options
    validate_data: bool = True
    remove_duplicates: bool = True
    handle_missing: str = "drop"  # "drop", "fill", "error"


@dataclass
class DataPipelineService:
    """Service for orchestrating complex data processing pipelines.
    
    This service handles multi-step data processing workflows including
    loading, cleaning, transformation, augmentation, and tokenization.
    """
    
    tokenization_service: TokenizationService
    data_loader_port: DataLoaderPort
    storage_port: StoragePort
    monitoring_port: MonitoringPort
    tokenizer_port: TokenizerPort
    
    async def execute_pipeline(
        self,
        input_path: Union[Path, List[Path]],
        config: DataPipelineConfig,
        callbacks: Optional[List[Callable]] = None
    ) -> Dict[str, Any]:
        """Execute a data processing pipeline.
        
        Args:
            input_path: Path(s) to input data
            config: Pipeline configuration
            callbacks: Optional callbacks for custom processing
            
        Returns:
            Dictionary with processed data and metadata
        """
        results = {
            "datasets": {},
            "statistics": {},
            "processing_steps": [],
            "errors": [],
            "cache_keys": {}
        }
        
        try:
            await self.monitoring_port.log_info(f"Starting data pipeline with {len(config.steps)} steps")
            
            # Check cache first if enabled
            if config.use_cache:
                cache_key = self._generate_cache_key(input_path, config)
                cached_data = await self._load_from_cache(cache_key)
                if cached_data:
                    await self.monitoring_port.log_info("Loaded data from cache")
                    return cached_data
            
            # Load initial data
            data = await self._load_data(input_path, config.input_format)
            results["statistics"]["original_size"] = len(data)
            
            # Execute pipeline steps
            for step_config in config.steps:
                step_type = DataProcessingStep(step_config["type"])
                step_name = step_config.get("name", step_type.value)
                
                await self.monitoring_port.log_info(f"Executing step: {step_name}")
                
                # Execute step based on type
                if step_type == DataProcessingStep.CLEAN:
                    data = await self._clean_data(data, step_config)
                elif step_type == DataProcessingStep.TRANSFORM:
                    data = await self._transform_data(data, step_config)
                elif step_type == DataProcessingStep.AUGMENT:
                    data = await self._augment_data(data, step_config, config)
                elif step_type == DataProcessingStep.TOKENIZE:
                    data = await self._tokenize_data(data, step_config, config)
                elif step_type == DataProcessingStep.SPLIT:
                    data = await self._split_data(data, step_config, config)
                elif step_type == DataProcessingStep.VALIDATE:
                    validation_results = await self._validate_data(data, step_config)
                    results["validation"] = validation_results
                else:
                    # Custom step - use callback
                    if callbacks:
                        for callback in callbacks:
                            if hasattr(callback, step_type.value):
                                data = await callback(data, step_config)
                
                # Record step completion
                results["processing_steps"].append({
                    "step": step_name,
                    "type": step_type.value,
                    "status": "completed",
                    "data_size": len(data) if hasattr(data, "__len__") else "unknown"
                })
            
            # Convert to final format
            final_data = await self._convert_to_output_format(data, config)
            
            # Calculate final statistics
            results["statistics"]["final_size"] = self._calculate_data_size(final_data)
            results["statistics"]["compression_ratio"] = (
                results["statistics"]["original_size"] / 
                results["statistics"]["final_size"]
                if results["statistics"]["final_size"] > 0 else 0
            )
            
            # Cache if enabled
            if config.use_cache and cache_key:
                await self._save_to_cache(final_data, cache_key)
                results["cache_keys"]["main"] = cache_key
            
            # Add final data to results
            if isinstance(final_data, dict):
                results["datasets"] = final_data
            else:
                results["datasets"]["main"] = final_data
            
            await self.monitoring_port.log_info("Data pipeline completed successfully")
            return results
            
        except Exception as e:
            await self.monitoring_port.log_error(f"Data pipeline failed: {str(e)}")
            results["errors"].append(str(e))
            raise
    
    async def create_data_loaders(
        self,
        datasets: Dict[str, Dataset],
        batch_size: int = 32,
        num_workers: int = 0,
        shuffle_train: bool = True,
        prefetch_factor: Optional[int] = None
    ) -> Dict[str, Any]:
        """Create data loaders from datasets.
        
        Args:
            datasets: Dictionary of datasets (train, val, test)
            batch_size: Batch size for loaders
            num_workers: Number of worker processes
            shuffle_train: Whether to shuffle training data
            prefetch_factor: Number of batches to prefetch
            
        Returns:
            Dictionary of data loaders
        """
        loaders = {}
        
        for split_name, dataset in datasets.items():
            shuffle = shuffle_train if split_name == "train" else False
            
            loader = await self.data_loader_port.create_loader(
                dataset,
                batch_size=batch_size,
                shuffle=shuffle,
                num_workers=num_workers,
                prefetch_factor=prefetch_factor
            )
            
            loaders[split_name] = loader
            
            await self.monitoring_port.log_info(
                f"Created {split_name} loader: {len(dataset)} samples, "
                f"{len(loader)} batches"
            )
        
        return loaders
    
    async def _load_data(
        self,
        input_path: Union[Path, List[Path]],
        format: str
    ) -> pd.DataFrame:
        """Load data from file(s)."""
        if isinstance(input_path, list):
            # Load and concatenate multiple files
            dfs = []
            for path in input_path:
                df = await self._load_single_file(path, format)
                dfs.append(df)
            return pd.concat(dfs, ignore_index=True)
        else:
            return await self._load_single_file(input_path, format)
    
    async def _load_single_file(self, path: Path, format: str) -> pd.DataFrame:
        """Load a single data file."""
        if format == "csv":
            return await self.storage_port.load_csv(path)
        elif format == "json":
            return await self.storage_port.load_json_lines(path)
        elif format == "parquet":
            return await self.storage_port.load_parquet(path)
        elif format == "text":
            # Load text file line by line
            lines = await self.storage_port.load_text_lines(path)
            return pd.DataFrame({"text": lines})
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    async def _clean_data(
        self,
        data: pd.DataFrame,
        config: Dict[str, Any]
    ) -> pd.DataFrame:
        """Clean the data."""
        # Remove duplicates
        if config.get("remove_duplicates", True):
            before = len(data)
            data = data.drop_duplicates()
            after = len(data)
            if before > after:
                await self.monitoring_port.log_info(
                    f"Removed {before - after} duplicate rows"
                )
        
        # Handle missing values
        if config.get("handle_missing"):
            if config["handle_missing"] == "drop":
                before = len(data)
                data = data.dropna()
                after = len(data)
                if before > after:
                    await self.monitoring_port.log_info(
                        f"Dropped {before - after} rows with missing values"
                    )
            elif config["handle_missing"] == "fill":
                fill_value = config.get("fill_value", "")
                data = data.fillna(fill_value)
        
        # Remove empty text
        if "text" in data.columns:
            before = len(data)
            data = data[data["text"].str.strip().str.len() > 0]
            after = len(data)
            if before > after:
                await self.monitoring_port.log_info(
                    f"Removed {before - after} rows with empty text"
                )
        
        # Clean text
        if config.get("clean_text") and "text" in data.columns:
            data["text"] = data["text"].apply(self._clean_text_content)
        
        return data
    
    def _clean_text_content(self, text: str) -> str:
        """Clean individual text content."""
        if not isinstance(text, str):
            return ""
        
        # Basic cleaning
        text = text.strip()
        
        # Remove multiple spaces
        import re
        text = re.sub(r'\s+', ' ', text)
        
        # Remove control characters
        text = ''.join(char for char in text if ord(char) >= 32 or char == '\n')
        
        return text
    
    async def _transform_data(
        self,
        data: pd.DataFrame,
        config: Dict[str, Any]
    ) -> pd.DataFrame:
        """Transform the data."""
        # Apply custom transformations
        if "transformations" in config:
            for transform in config["transformations"]:
                if transform["type"] == "combine_columns":
                    # Combine multiple columns into text
                    columns = transform["columns"]
                    separator = transform.get("separator", " ")
                    data["text"] = data[columns].apply(
                        lambda row: separator.join(str(v) for v in row), axis=1
                    )
                
                elif transform["type"] == "extract_features":
                    # Extract features from text
                    if "text" in data.columns:
                        data["text_length"] = data["text"].str.len()
                        data["word_count"] = data["text"].str.split().str.len()
                
                elif transform["type"] == "normalize":
                    # Normalize numeric columns
                    for col in transform.get("columns", []):
                        if col in data.columns:
                            data[col] = (data[col] - data[col].mean()) / data[col].std()
                
                elif transform["type"] == "categorize":
                    # Convert continuous to categorical
                    col = transform["column"]
                    bins = transform["bins"]
                    labels = transform.get("labels")
                    data[f"{col}_cat"] = pd.cut(data[col], bins=bins, labels=labels)
        
        return data
    
    async def _augment_data(
        self,
        data: pd.DataFrame,
        step_config: Dict[str, Any],
        pipeline_config: DataPipelineConfig
    ) -> pd.DataFrame:
        """Augment the data."""
        if pipeline_config.augmentation_ratio <= 0:
            return data
        
        augmented_samples = []
        num_to_augment = int(len(data) * pipeline_config.augmentation_ratio)
        
        # Sample data to augment
        samples_to_augment = data.sample(n=min(num_to_augment, len(data)))
        
        for _, row in samples_to_augment.iterrows():
            if "text" in row:
                augmented_text = await self._augment_text(
                    row["text"],
                    pipeline_config.augmentation_strategies or ["paraphrase"]
                )
                augmented_row = row.copy()
                augmented_row["text"] = augmented_text
                augmented_row["is_augmented"] = True
                augmented_samples.append(augmented_row)
        
        if augmented_samples:
            augmented_df = pd.DataFrame(augmented_samples)
            data = pd.concat([data, augmented_df], ignore_index=True)
            await self.monitoring_port.log_info(
                f"Added {len(augmented_samples)} augmented samples"
            )
        
        return data
    
    async def _augment_text(self, text: str, strategies: List[str]) -> str:
        """Apply text augmentation strategies."""
        # Simple augmentation strategies
        import random
        
        strategy = random.choice(strategies)
        
        if strategy == "synonym":
            # Replace with synonyms (simplified)
            words = text.split()
            if len(words) > 3:
                idx = random.randint(0, len(words) - 1)
                words[idx] = f"[SYN:{words[idx]}]"
            return " ".join(words)
        
        elif strategy == "paraphrase":
            # Simple paraphrasing by reordering
            sentences = text.split(". ")
            if len(sentences) > 1:
                random.shuffle(sentences)
            return ". ".join(sentences)
        
        elif strategy == "noise":
            # Add noise to text
            if random.random() < 0.1:
                text = text.replace(" ", "  ")  # Double spaces
            return text
        
        return text
    
    async def _tokenize_data(
        self,
        data: pd.DataFrame,
        step_config: Dict[str, Any],
        pipeline_config: DataPipelineConfig
    ) -> Dict[str, Any]:
        """Tokenize text data."""
        if "text" not in data.columns:
            raise ValueError("No text column found for tokenization")
        
        # Tokenize texts
        texts = data["text"].tolist()
        tokenized = await self.tokenization_service.tokenize_batch(
            texts,
            max_length=pipeline_config.max_length,
            truncation=pipeline_config.truncation,
            padding=pipeline_config.padding
        )
        
        # Add labels if present
        if pipeline_config.label_column and pipeline_config.label_column in data.columns:
            tokenized["labels"] = data[pipeline_config.label_column].tolist()
        
        # Add additional features if specified
        if pipeline_config.feature_columns:
            for col in pipeline_config.feature_columns:
                if col in data.columns:
                    tokenized[col] = data[col].tolist()
        
        return tokenized
    
    async def _split_data(
        self,
        data: Union[pd.DataFrame, Dict[str, Any]],
        step_config: Dict[str, Any],
        pipeline_config: DataPipelineConfig
    ) -> Dict[str, Any]:
        """Split data into train/val/test sets."""
        from sklearn.model_selection import train_test_split
        
        # Handle different data formats
        if isinstance(data, pd.DataFrame):
            # Split DataFrame
            train_val, test = train_test_split(
                data,
                test_size=pipeline_config.test_ratio,
                stratify=data[pipeline_config.label_column] if pipeline_config.stratify and pipeline_config.label_column else None,
                random_state=42
            )
            
            val_ratio_adjusted = pipeline_config.val_ratio / (1 - pipeline_config.test_ratio)
            train, val = train_test_split(
                train_val,
                test_size=val_ratio_adjusted,
                stratify=train_val[pipeline_config.label_column] if pipeline_config.stratify and pipeline_config.label_column else None,
                random_state=42
            )
            
            return {
                "train": train,
                "val": val,
                "test": test
            }
        else:
            # Split tokenized data
            indices = list(range(len(data["input_ids"])))
            
            if pipeline_config.stratify and "labels" in data:
                # Stratified split
                train_val_idx, test_idx = train_test_split(
                    indices,
                    test_size=pipeline_config.test_ratio,
                    stratify=[data["labels"][i] for i in indices],
                    random_state=42
                )
                
                val_ratio_adjusted = pipeline_config.val_ratio / (1 - pipeline_config.test_ratio)
                train_idx, val_idx = train_test_split(
                    train_val_idx,
                    test_size=val_ratio_adjusted,
                    stratify=[data["labels"][i] for i in train_val_idx],
                    random_state=42
                )
            else:
                # Random split
                train_val_idx, test_idx = train_test_split(
                    indices,
                    test_size=pipeline_config.test_ratio,
                    random_state=42
                )
                
                val_ratio_adjusted = pipeline_config.val_ratio / (1 - pipeline_config.test_ratio)
                train_idx, val_idx = train_test_split(
                    train_val_idx,
                    test_size=val_ratio_adjusted,
                    random_state=42
                )
            
            # Create split datasets
            splits = {}
            for split_name, split_indices in [("train", train_idx), ("val", val_idx), ("test", test_idx)]:
                split_data = {}
                for key, values in data.items():
                    if isinstance(values, list):
                        split_data[key] = [values[i] for i in split_indices]
                    else:
                        # Handle tensor-like data
                        split_data[key] = values[split_indices]
                splits[split_name] = split_data
            
            return splits
    
    async def _validate_data(
        self,
        data: Union[pd.DataFrame, Dict[str, Any]],
        config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Validate data quality."""
        validation_results = {
            "passed": True,
            "warnings": [],
            "errors": []
        }
        
        if isinstance(data, pd.DataFrame):
            # Validate DataFrame
            if data.empty:
                validation_results["errors"].append("Data is empty")
                validation_results["passed"] = False
            
            # Check for required columns
            required_columns = config.get("required_columns", [])
            for col in required_columns:
                if col not in data.columns:
                    validation_results["errors"].append(f"Missing required column: {col}")
                    validation_results["passed"] = False
            
            # Check data types
            if "text" in data.columns:
                non_string = data[~data["text"].apply(lambda x: isinstance(x, str))]
                if not non_string.empty:
                    validation_results["warnings"].append(
                        f"{len(non_string)} rows have non-string text values"
                    )
        else:
            # Validate tokenized data
            if "input_ids" not in data:
                validation_results["errors"].append("Missing input_ids in tokenized data")
                validation_results["passed"] = False
            
            # Check consistency
            first_len = len(data.get("input_ids", []))
            for key, values in data.items():
                if isinstance(values, list) and len(values) != first_len:
                    validation_results["errors"].append(
                        f"Inconsistent length for {key}: {len(values)} vs {first_len}"
                    )
                    validation_results["passed"] = False
        
        return validation_results
    
    async def _convert_to_output_format(
        self,
        data: Union[pd.DataFrame, Dict[str, Any]],
        config: DataPipelineConfig
    ) -> Any:
        """Convert data to the requested output format."""
        if config.output_format == "dataset":
            # Convert to Dataset objects
            if isinstance(data, dict) and "train" in data:
                # Multiple splits
                datasets = {}
                for split_name, split_data in data.items():
                    if isinstance(split_data, pd.DataFrame):
                        datasets[split_name] = Dataset.from_pandas(split_data)
                    else:
                        datasets[split_name] = Dataset.from_dict(split_data)
                return datasets
            else:
                # Single dataset
                if isinstance(data, pd.DataFrame):
                    return Dataset.from_pandas(data)
                else:
                    return Dataset.from_dict(data)
        
        elif config.output_format == "dataloader":
            # Convert to data loaders
            datasets = await self._convert_to_output_format(
                data, 
                DataPipelineConfig(**{**config.__dict__, "output_format": "dataset"})
            )
            
            if isinstance(datasets, dict):
                return await self.create_data_loaders(datasets)
            else:
                return await self.create_data_loaders({"main": datasets})
        
        else:  # cached or raw
            return data
    
    def _generate_cache_key(
        self,
        input_path: Union[Path, List[Path]],
        config: DataPipelineConfig
    ) -> str:
        """Generate cache key for the pipeline configuration."""
        import hashlib
        import json
        
        # Create string representation of config
        config_str = json.dumps({
            "input_path": str(input_path) if isinstance(input_path, Path) else [str(p) for p in input_path],
            "steps": config.steps,
            "max_length": config.max_length,
            "augmentation_ratio": config.augmentation_ratio,
            "train_ratio": config.train_ratio,
            "val_ratio": config.val_ratio,
            "test_ratio": config.test_ratio
        }, sort_keys=True)
        
        # Generate hash
        return hashlib.md5(config_str.encode()).hexdigest()
    
    async def _load_from_cache(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """Load data from cache if available."""
        cache_path = Path(".cache/data") / f"{cache_key}.pkl"
        if cache_path.exists():
            try:
                return await self.storage_port.load_pickle(cache_path)
            except Exception as e:
                await self.monitoring_port.log_warning(f"Failed to load cache: {str(e)}")
        return None
    
    async def _save_to_cache(self, data: Any, cache_key: str) -> None:
        """Save data to cache."""
        cache_dir = Path(".cache/data")
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_path = cache_dir / f"{cache_key}.pkl"
        
        try:
            await self.storage_port.save_pickle(data, cache_path)
            await self.monitoring_port.log_info(f"Cached data with key: {cache_key}")
        except Exception as e:
            await self.monitoring_port.log_warning(f"Failed to cache data: {str(e)}")
    
    def _calculate_data_size(self, data: Any) -> int:
        """Calculate the size of the data."""
        if isinstance(data, pd.DataFrame):
            return len(data)
        elif isinstance(data, dict):
            if "train" in data:
                # Multiple splits
                return sum(self._calculate_data_size(split) for split in data.values())
            elif "input_ids" in data:
                # Tokenized data
                return len(data["input_ids"])
            else:
                # Dataset dict
                return sum(len(v) if hasattr(v, "__len__") else 1 for v in data.values())
        elif hasattr(data, "__len__"):
            return len(data)
        else:
            return 1