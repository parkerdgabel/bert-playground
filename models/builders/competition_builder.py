"""Builder for creating models optimized for Kaggle competitions."""

from pathlib import Path
from typing import Any, Optional

import pandas as pd
from loguru import logger

from core.bootstrap import get_service
from core.ports.compute import Module
from data.core.base import CompetitionType

from .model_with_head_builder import ModelWithHeadBuilder
from .lora_builder import LoRABuilder


class CompetitionAnalysis:
    """Analysis results for a Kaggle competition dataset."""

    def __init__(
        self,
        competition_type: CompetitionType,
        num_samples: int,
        num_features: int,
        target_column: str | None = None,
        num_classes: int | None = None,
        class_distribution: dict[str, int] | None = None,
        is_balanced: bool = True,
        categorical_columns: list[str] | None = None,
        numerical_columns: list[str] | None = None,
        text_columns: list[str] | None = None,
        recommended_model_type: str = "generic",
        recommended_head_type: str = "binary",
        recommended_batch_size: int = 32,
        recommended_learning_rate: float = 2e-5,
    ):
        self.competition_type = competition_type
        self.num_samples = num_samples
        self.num_features = num_features
        self.target_column = target_column
        self.num_classes = num_classes
        self.class_distribution = class_distribution
        self.is_balanced = is_balanced
        self.categorical_columns = categorical_columns or []
        self.numerical_columns = numerical_columns or []
        self.text_columns = text_columns or []
        self.recommended_model_type = recommended_model_type
        self.recommended_head_type = recommended_head_type
        self.recommended_batch_size = recommended_batch_size
        self.recommended_learning_rate = recommended_learning_rate


class CompetitionBuilder:
    """Builder for competition-optimized models."""

    def __init__(self):
        self.model_builder = ModelWithHeadBuilder()
        self.lora_builder = LoRABuilder()

    def build_kaggle_classifier(
        self,
        task_type: str,
        num_classes: int | None = None,
        model_type: str = "modernbert_with_head",
        **kwargs,
    ) -> Module:
        """Build classifier optimized for Kaggle competitions.
        
        Args:
            task_type: Type of classification task
            num_classes: Number of classes/labels
            model_type: Type of model architecture
            **kwargs: Additional arguments
            
        Returns:
            Configured classifier
        """
        logger.info(f"Building Kaggle classifier for {task_type} task")
        
        # Map task type to head type
        head_type_map = {
            "binary": "binary_classification",
            "multiclass": "multiclass_classification",
            "multilabel": "multilabel_classification",
            "regression": "regression",
            "titanic": "binary_classification",
            "ordinal": "ordinal_regression",
            "time_series": "time_series",
            "ranking": "ranking",
        }
        
        head_type = head_type_map.get(task_type, "binary_classification")
        
        return self.model_builder.build_model_with_head(
            model_type=model_type,
            head_type=head_type,
            num_labels=num_classes or 2,
            **kwargs
        )

    def build_competition_classifier(
        self,
        data_path: str,
        target_column: str,
        model_name: str = "answerdotai/ModernBERT-base",
        auto_optimize: bool = True,
        **kwargs,
    ) -> tuple[Module, CompetitionAnalysis]:
        """Analyze dataset and create optimized classifier.
        
        Args:
            data_path: Path to training data CSV
            target_column: Name of target column
            model_name: Name of the embedding model to use
            auto_optimize: Whether to automatically optimize configuration
            **kwargs: Additional arguments
            
        Returns:
            Tuple of (classifier, analysis_results)
        """
        logger.info(f"Building competition classifier from {data_path}")
        
        # Analyze dataset
        analysis = self.analyze_competition_dataset(data_path, target_column)

        # Create optimized classifier based on analysis
        if auto_optimize:
            # Override kwargs with optimized settings
            kwargs.update(
                {
                    "dropout_prob": 0.1 if analysis.num_samples > 10000 else 0.2,
                    "use_layer_norm": analysis.num_features > 50,
                    "pooling_type": "attention" if analysis.num_samples > 5000 else "mean",
                    "batch_size": analysis.recommended_batch_size,
                }
            )

            logger.info(f"Auto-optimized configuration: {kwargs}")

        # Get competition type mapping
        head_type_map = {
            CompetitionType.BINARY_CLASSIFICATION: "binary",
            CompetitionType.MULTICLASS_CLASSIFICATION: "multiclass",
            CompetitionType.MULTILABEL_CLASSIFICATION: "multilabel",
            CompetitionType.REGRESSION: "regression",
            CompetitionType.ORDINAL_REGRESSION: "ordinal",
            CompetitionType.TIME_SERIES: "time_series",
            CompetitionType.RANKING: "ranking",
        }
        
        task_type = head_type_map.get(analysis.competition_type, "multiclass")

        # Create classifier
        classifier = self.build_kaggle_classifier(
            task_type=task_type,
            model_name=model_name,
            num_classes=analysis.num_classes,
            **kwargs,
        )

        logger.info(f"Created {task_type} classifier for {analysis.competition_type.value}")
        return classifier, analysis

    def build_kaggle_lora_model(
        self,
        competition_type: str | CompetitionType,
        data_path: str | None = None,
        lora_preset: str | None = None,
        auto_select_preset: bool = True,
        **kwargs,
    ) -> tuple[Module, Any]:
        """Build optimized LoRA model for Kaggle competition.
        
        Args:
            competition_type: Type of competition
            data_path: Optional path to data for analysis
            lora_preset: LoRA preset (auto-selected if None)
            auto_select_preset: Whether to auto-select preset
            **kwargs: Additional arguments
            
        Returns:
            Tuple of (model, lora_adapter)
        """
        # Convert string to CompetitionType if needed
        if isinstance(competition_type, str):
            competition_type = CompetitionType(competition_type)

        # Auto-select LoRA preset based on competition type
        if lora_preset is None and auto_select_preset:
            preset_map = {
                CompetitionType.BINARY_CLASSIFICATION: "balanced",
                CompetitionType.MULTICLASS_CLASSIFICATION: "balanced",
                CompetitionType.MULTILABEL_CLASSIFICATION: "expressive",
                CompetitionType.REGRESSION: "efficient",
                CompetitionType.ORDINAL_REGRESSION: "balanced",
                CompetitionType.TIME_SERIES: "expressive",
                CompetitionType.RANKING: "expressive",
            }
            lora_preset = preset_map.get(competition_type, "balanced")
            logger.info(f"Auto-selected LoRA preset: {lora_preset}")

        # Analyze dataset if provided
        if data_path:
            target_column = kwargs.pop("target_column", "target")
            analysis = self.analyze_competition_dataset(data_path, target_column)

            # Adjust preset based on dataset size
            if analysis.num_samples > 100000 and lora_preset == "expressive":
                lora_preset = "balanced"  # Use smaller rank for large datasets
                logger.info("Adjusted to 'balanced' preset for large dataset")
            elif analysis.num_samples < 5000 and lora_preset == "efficient":
                lora_preset = "balanced"  # Use larger rank for small datasets
                logger.info("Adjusted to 'balanced' preset for small dataset")

            # Use analysis results
            kwargs["num_labels"] = analysis.num_classes

        # Map competition type to head type
        head_type_map = {
            CompetitionType.BINARY_CLASSIFICATION: "binary_classification",
            CompetitionType.MULTICLASS_CLASSIFICATION: "multiclass_classification",
            CompetitionType.MULTILABEL_CLASSIFICATION: "multilabel_classification",
            CompetitionType.REGRESSION: "regression",
            CompetitionType.ORDINAL_REGRESSION: "ordinal_regression",
            CompetitionType.TIME_SERIES: "time_series",
            CompetitionType.RANKING: "ranking",
        }

        head_type = head_type_map.get(competition_type, "binary_classification")

        # Create base model
        base_model = self.model_builder.build_bert_with_head(
            head_type=head_type, **kwargs
        )
        
        # Add LoRA adapters
        return self.lora_builder.build_lora_model(
            base_model=base_model,
            lora_config=lora_preset,
            inject_adapters=True,
        )

    def analyze_competition_dataset(
        self, data_path: str, target_column: str
    ) -> CompetitionAnalysis:
        """Analyze a competition dataset and return optimization recommendations.
        
        Args:
            data_path: Path to training data CSV
            target_column: Name of target column
            
        Returns:
            CompetitionAnalysis with recommendations
        """
        # Load and analyze data
        df = pd.read_csv(data_path)

        # Basic characteristics
        num_samples = len(df)
        num_features = len(df.columns) - 1  # Exclude target

        # Analyze target column
        target_series = df[target_column]
        num_classes = target_series.nunique()
        class_distribution = target_series.value_counts().to_dict()

        # Determine competition type
        if target_series.dtype in ["object", "category"]:
            if num_classes == 2:
                competition_type = CompetitionType.BINARY_CLASSIFICATION
            else:
                competition_type = CompetitionType.MULTICLASS_CLASSIFICATION
        else:
            # Check if it's regression or classification with numeric labels
            if num_classes <= 10 and target_series.dtype in ["int64", "int32"]:
                competition_type = CompetitionType.MULTICLASS_CLASSIFICATION
            else:
                competition_type = CompetitionType.REGRESSION

        # Analyze features
        categorical_columns = []
        numerical_columns = []
        text_columns = []

        for col in df.columns:
            if col == target_column:
                continue

            series = df[col]

            if series.dtype == "object":
                # If most values are unique, likely text
                if series.nunique() / len(series) > 0.9:
                    text_columns.append(col)
                else:
                    categorical_columns.append(col)
            elif pd.api.types.is_numeric_dtype(series):
                # If low cardinality, might be categorical
                if series.nunique() <= 10 and series.dtype in ["int64", "int32"]:
                    categorical_columns.append(col)
                else:
                    numerical_columns.append(col)
            else:
                categorical_columns.append(col)

        # Check class balance
        if competition_type in [
            CompetitionType.BINARY_CLASSIFICATION,
            CompetitionType.MULTICLASS_CLASSIFICATION,
        ]:
            min_class_count = min(class_distribution.values())
            max_class_count = max(class_distribution.values())
            is_balanced = (min_class_count / max_class_count) > 0.1
        else:
            is_balanced = True

        # Recommendations
        recommended_batch_size = 32
        if num_samples > 100000:
            recommended_batch_size = 64
        elif num_samples < 5000:
            recommended_batch_size = 16

        recommended_learning_rate = 2e-5
        if not is_balanced:
            recommended_learning_rate = 1e-5  # Lower LR for imbalanced datasets

        return CompetitionAnalysis(
            competition_type=competition_type,
            num_samples=num_samples,
            num_features=num_features,
            target_column=target_column,
            num_classes=num_classes,
            class_distribution=class_distribution,
            is_balanced=is_balanced,
            categorical_columns=categorical_columns,
            numerical_columns=numerical_columns,
            text_columns=text_columns,
            recommended_batch_size=recommended_batch_size,
            recommended_learning_rate=recommended_learning_rate,
            recommended_model_type="generic",
            recommended_head_type="binary"
            if competition_type == CompetitionType.BINARY_CLASSIFICATION
            else "multiclass",
        )