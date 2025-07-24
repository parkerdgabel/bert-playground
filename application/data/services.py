"""Domain services for data operations.

This module contains the business logic for data processing operations,
orchestrating various adapters to fulfill business requirements.
"""

from typing import Any, Dict, List, Optional, Tuple
from loguru import logger

from .models import DatasetSpec, DataSample, DataBatch, Dataset, DataValidationResult
from .interfaces import (
    DataRepository, DataCache, TokenizerAdapter, DataLoader,
    DataValidatorAdapter, TextProcessorAdapter, DataAugmentationAdapter,
    TemplateEngine, DatasetFactory, MetricsCollector, FileSystemAdapter
)


class DataProcessingService:
    """Service for orchestrating data processing operations."""
    
    def __init__(
        self,
        repository: DataRepository,
        validator: DataValidatorAdapter,
        dataset_factory: DatasetFactory,
        cache: Optional[DataCache] = None,
        metrics: Optional[MetricsCollector] = None
    ):
        """Initialize data processing service."""
        self.repository = repository
        self.validator = validator
        self.dataset_factory = dataset_factory
        self.cache = cache
        self.metrics = metrics
    
    def create_dataset(
        self,
        spec: DatasetSpec,
        split: str = "train",
        validate: bool = True
    ) -> Tuple[Dataset, DataValidationResult]:
        """Create and validate a dataset from specification.
        
        Args:
            spec: Dataset specification
            split: Data split (train, val, test)
            validate: Whether to validate the data
            
        Returns:
            Tuple of (dataset, validation_result)
        """
        logger.info(f"Creating dataset for {spec.competition_name} ({split})")
        
        # Check cache first
        cache_key = f"{spec.competition_name}_{split}_{hash(str(spec))}"
        if self.cache:
            cached_dataset = self.cache.get(cache_key)
            if cached_dataset:
                logger.info("Found cached dataset")
                return cached_dataset, DataValidationResult(is_valid=True)
        
        # Validate file format first
        validation_result = DataValidationResult(is_valid=True)
        if validate:
            file_validation = self.validator.validate_file_format(str(spec.dataset_path))
            if not file_validation.is_valid:
                return None, file_validation
            validation_result = file_validation
        
        # Create dataset
        try:
            dataset = self.dataset_factory.create_dataset(spec, split, self.cache)
            
            # Validate data quality if requested
            if validate:
                raw_data = self.repository.load_raw_data(str(spec.dataset_path))
                quality_validation = self.validator.validate_data_quality(raw_data)
                
                # Merge validation results
                for error in quality_validation.errors:
                    validation_result.add_error(error)
            
            # Cache the dataset
            if self.cache and validation_result.is_valid:
                self.cache.set(cache_key, dataset)
            
            # Record metrics
            if self.metrics:
                self.metrics.record_data_quality_metrics({
                    "dataset_size": len(dataset),
                    "validation_errors": len(validation_result.errors),
                    "competition_name": spec.competition_name
                })
            
            logger.info(f"Created dataset with {len(dataset)} samples")
            return dataset, validation_result
            
        except Exception as e:
            logger.error(f"Failed to create dataset: {str(e)}")
            validation_result.add_error(f"Dataset creation failed: {str(e)}")
            return None, validation_result
    
    def create_dataloader(
        self,
        dataset: Dataset,
        batch_size: int = 32,
        shuffle: bool = False,
        tokenizer: Optional[TokenizerAdapter] = None
    ) -> DataLoader:
        """Create a data loader for training/inference.
        
        Args:
            dataset: Dataset to load from
            batch_size: Batch size
            shuffle: Whether to shuffle data
            tokenizer: Optional tokenizer for text processing
            
        Returns:
            DataLoader instance
        """
        logger.info(f"Creating dataloader with batch_size={batch_size}, shuffle={shuffle}")
        
        dataloader = self.dataset_factory.create_dataloader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            tokenizer=tokenizer
        )
        
        logger.info(f"Created dataloader with {len(dataloader)} batches")
        return dataloader
    
    def validate_dataset_specification(self, spec: DatasetSpec) -> DataValidationResult:
        """Validate a dataset specification.
        
        Args:
            spec: Dataset specification to validate
            
        Returns:
            Validation result
        """
        result = DataValidationResult(is_valid=True)
        
        # Check file existence
        if not self.repository.exists(str(spec.dataset_path)):
            result.add_error(f"Dataset file does not exist: {spec.dataset_path}")
        
        # Validate file format
        file_validation = self.validator.validate_file_format(str(spec.dataset_path))
        for error in file_validation.errors:
            result.add_error(error)
        
        # Business logic validation
        if spec.num_samples <= 0:
            result.add_error("Number of samples must be positive")
        
        if spec.num_features <= 0:
            result.add_error("Number of features must be positive")
        
        if spec.recommended_batch_size <= 0:
            result.add_error("Recommended batch size must be positive")
        
        if spec.recommended_max_length <= 0:
            result.add_error("Recommended max length must be positive")
        
        return result
    
    def get_dataset_statistics(self, dataset: Dataset) -> Dict[str, Any]:
        """Get statistics about a dataset.
        
        Args:
            dataset: Dataset to analyze
            
        Returns:
            Dictionary of statistics
        """
        logger.info("Computing dataset statistics")
        
        stats = {
            "total_samples": len(dataset),
            "competition_info": dataset.get_competition_info(),
            "spec": {
                "competition_name": dataset.spec.competition_name,
                "competition_type": dataset.spec.competition_type.value,
                "num_features": dataset.spec.num_features,
                "target_column": dataset.spec.target_column,
                "text_columns": dataset.spec.text_columns,
                "categorical_columns": dataset.spec.categorical_columns,
                "numerical_columns": dataset.spec.numerical_columns,
            }
        }
        
        # Add sample-based statistics
        if len(dataset) > 0:
            sample_texts = []
            sample_labels = []
            
            # Sample a subset for analysis (max 1000 samples)
            sample_size = min(1000, len(dataset))
            indices = list(range(0, len(dataset), len(dataset) // sample_size))[:sample_size]
            
            for idx in indices:
                sample = dataset.get_sample(idx)
                sample_texts.append(sample.text)
                if sample.labels is not None:
                    sample_labels.append(sample.labels)
            
            # Text statistics
            text_lengths = [len(text.split()) for text in sample_texts]
            stats["text_statistics"] = {
                "avg_length": sum(text_lengths) / len(text_lengths),
                "max_length": max(text_lengths),
                "min_length": min(text_lengths),
                "total_words": sum(text_lengths)
            }
            
            # Label statistics
            if sample_labels:
                unique_labels = set(sample_labels)
                stats["label_statistics"] = {
                    "num_unique_labels": len(unique_labels),
                    "label_distribution": {str(label): sample_labels.count(label) for label in unique_labels}
                }
        
        return stats


class DataPreprocessingService:
    """Service for data preprocessing operations."""
    
    def __init__(
        self,
        text_processor: Optional[TextProcessorAdapter] = None,
        augmenter: Optional[DataAugmentationAdapter] = None,
        template_engine: Optional[TemplateEngine] = None
    ):
        """Initialize preprocessing service."""
        self.text_processor = text_processor
        self.augmenter = augmenter
        self.template_engine = template_engine
    
    def preprocess_sample(
        self,
        sample: DataSample,
        clean_text: bool = True,
        apply_augmentation: bool = False,
        template_name: Optional[str] = None
    ) -> DataSample:
        """Preprocess a single data sample.
        
        Args:
            sample: Sample to preprocess
            clean_text: Whether to clean text
            apply_augmentation: Whether to apply augmentation
            template_name: Optional template to apply
            
        Returns:
            Preprocessed sample
        """
        processed_sample = sample
        
        # Apply template if specified
        if template_name and self.template_engine:
            template_data = {"text": sample.text, **sample.metadata}
            processed_text = self.template_engine.apply_template(template_name, template_data)
            processed_sample = DataSample(
                text=processed_text,
                labels=sample.labels,
                metadata=sample.metadata
            )
        
        # Clean text if requested
        if clean_text and self.text_processor:
            cleaned_text = self.text_processor.clean_text(processed_sample.text)
            processed_sample = DataSample(
                text=cleaned_text,
                labels=processed_sample.labels,
                metadata=processed_sample.metadata
            )
        
        # Apply augmentation if requested
        if apply_augmentation and self.augmenter:
            processed_sample = self.augmenter.augment_sample(processed_sample)
        
        return processed_sample
    
    def preprocess_batch(
        self,
        batch: DataBatch,
        clean_text: bool = True,
        apply_augmentation: bool = False,
        template_name: Optional[str] = None
    ) -> DataBatch:
        """Preprocess a batch of data samples.
        
        Args:
            batch: Batch to preprocess
            clean_text: Whether to clean text
            apply_augmentation: Whether to apply augmentation
            template_name: Optional template to apply
            
        Returns:
            Preprocessed batch
        """
        # Process each sample in the batch
        processed_texts = []
        
        for i, text in enumerate(batch.texts):
            sample = DataSample(
                text=text,
                labels=batch.labels[i] if batch.labels else None,
                metadata=batch.metadata[i] if batch.metadata else {}
            )
            
            processed_sample = self.preprocess_sample(
                sample=sample,
                clean_text=clean_text,
                apply_augmentation=apply_augmentation,
                template_name=template_name
            )
            
            processed_texts.append(processed_sample.text)
        
        return DataBatch(
            texts=processed_texts,
            labels=batch.labels,
            metadata=batch.metadata
        )


class DataPipelineService:
    """Service for orchestrating complete data processing pipelines."""
    
    def __init__(
        self,
        processing_service: DataProcessingService,
        preprocessing_service: DataPreprocessingService,
        tokenizer: Optional[TokenizerAdapter] = None
    ):
        """Initialize pipeline service."""
        self.processing_service = processing_service
        self.preprocessing_service = preprocessing_service
        self.tokenizer = tokenizer
    
    def create_training_pipeline(
        self,
        train_spec: DatasetSpec,
        val_spec: Optional[DatasetSpec] = None,
        batch_size: int = 32,
        preprocessing_config: Optional[Dict[str, Any]] = None
    ) -> Tuple[DataLoader, Optional[DataLoader], List[str]]:
        """Create a complete training data pipeline.
        
        Args:
            train_spec: Training dataset specification
            val_spec: Optional validation dataset specification
            batch_size: Batch size for data loaders
            preprocessing_config: Preprocessing configuration
            
        Returns:
            Tuple of (train_loader, val_loader, errors)
        """
        errors = []
        preprocessing_config = preprocessing_config or {}
        
        logger.info("Creating training data pipeline")
        
        # Create training dataset
        train_dataset, train_validation = self.processing_service.create_dataset(
            train_spec, split="train", validate=True
        )
        
        if not train_validation.is_valid:
            errors.extend(train_validation.errors)
            return None, None, errors
        
        # Create validation dataset if specified
        val_dataset = None
        if val_spec:
            val_dataset, val_validation = self.processing_service.create_dataset(
                val_spec, split="val", validate=True
            )
            
            if not val_validation.is_valid:
                errors.extend([f"Validation: {error}" for error in val_validation.errors])
        
        # Create data loaders
        train_loader = self.processing_service.create_dataloader(
            dataset=train_dataset,
            batch_size=batch_size,
            shuffle=True,
            tokenizer=self.tokenizer
        )
        
        val_loader = None
        if val_dataset:
            val_loader = self.processing_service.create_dataloader(
                dataset=val_dataset,
                batch_size=batch_size * 2,  # Larger batch for validation
                shuffle=False,
                tokenizer=self.tokenizer
            )
        
        logger.info("Training data pipeline created successfully")
        return train_loader, val_loader, errors
    
    def create_inference_pipeline(
        self,
        test_spec: DatasetSpec,
        batch_size: int = 64,
        preprocessing_config: Optional[Dict[str, Any]] = None
    ) -> Tuple[DataLoader, List[str]]:
        """Create inference data pipeline.
        
        Args:
            test_spec: Test dataset specification
            batch_size: Batch size for data loader
            preprocessing_config: Preprocessing configuration
            
        Returns:
            Tuple of (test_loader, errors)
        """
        errors = []
        preprocessing_config = preprocessing_config or {}
        
        logger.info("Creating inference data pipeline")
        
        # Create test dataset
        test_dataset, test_validation = self.processing_service.create_dataset(
            test_spec, split="test", validate=True
        )
        
        if not test_validation.is_valid:
            errors.extend(test_validation.errors)
            return None, errors
        
        # Create data loader
        test_loader = self.processing_service.create_dataloader(
            dataset=test_dataset,
            batch_size=batch_size,
            shuffle=False,
            tokenizer=self.tokenizer
        )
        
        logger.info("Inference data pipeline created successfully")
        return test_loader, errors