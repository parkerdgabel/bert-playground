"""Competition-specific augmentation strategies.

This module contains augmentation strategies ported from the preprocessing
plugins to work with the new augmentation framework.
"""

from typing import Any

import pandas as pd
# from loguru import logger  # Domain should not depend on logging framework

from .base import BaseAugmenter, BaseAugmentationStrategy, FeatureMetadata, FeatureType
from .config import AugmentationConfig
from .tabular import TabularToTextAugmenter


class TitanicAugmenter(BaseAugmenter):
    """Titanic competition-specific augmentation strategy.

    Ported from preprocessing/plugins/titanic.py to work with the
    augmentation framework. Provides Titanic-specific text conversion
    and feature handling.
    """

    def __init__(self, config: AugmentationConfig = None, seed: int = None):
        """Initialize Titanic augmenter."""
        super().__init__(seed=seed)
        self.config = config or AugmentationConfig()

        # Mappings for categorical values
        self.class_map = {1: "first-class", 2: "second-class", 3: "third-class"}
        self.embark_map = {"C": "Cherbourg", "Q": "Queenstown", "S": "Southampton"}

        # Expected columns
        self.expected_columns = [
            "PassengerId",
            "Pclass",
            "Name",
            "Sex",
            "Age",
            "SibSp",
            "Parch",
            "Ticket",
            "Fare",
            "Cabin",
            "Embarked",
        ]

    def augment(self, data: dict[str, Any], **kwargs) -> str:
        """Augment Titanic sample by converting to text.

        Args:
            data: Sample data dictionary
            **kwargs: Additional arguments

        Returns:
            Text representation of the sample
        """
        return self._convert_to_text(data)

    def _convert_to_text(self, data: dict[str, Any]) -> str:
        """Convert Titanic passenger data to natural language.

        Args:
            data: Dictionary with passenger information

        Returns:
            Natural language description
        """
        parts = []

        # Start with passenger name (truncate if too long)
        name = data.get("Name", "Unknown passenger")
        if isinstance(name, str) and len(name) > 80:
            name = name[:77] + "..."
        parts.append(f"Passenger {name}")

        # Add demographic info
        demo_parts = []

        # Age
        age = data.get("Age")
        if age is not None and not pd.isna(age):
            age_int = int(float(age))
            demo_parts.append(f"{age_int}-year-old")

        # Sex
        sex = data.get("Sex")
        if sex and not pd.isna(sex):
            demo_parts.append(str(sex).lower())

        if demo_parts:
            parts.append(f"was a {' '.join(demo_parts)}")

        # Add class information
        pclass = data.get("Pclass")
        if pclass is not None and not pd.isna(pclass):
            class_desc = self.class_map.get(int(pclass), f"class {pclass}")
            parts.append(f"traveling in {class_desc}")

        # Add family information
        family_parts = []
        sibsp = data.get("SibSp", 0)
        parch = data.get("Parch", 0)

        if sibsp or parch:
            if sibsp:
                family_parts.append(f"{sibsp} sibling{'s' if sibsp != 1 else ''}")
            if parch:
                if family_parts:
                    family_parts.append("and")
                family_parts.append(
                    f"{parch} parent{'s' if parch != 1 else ''}/child{'ren' if parch != 1 else ''}"
                )

            parts.append(f"with {' '.join(family_parts)}")
        else:
            parts.append("traveling alone")

        # Add fare information
        fare = data.get("Fare")
        if fare is not None and not pd.isna(fare):
            fare_float = float(fare)
            if fare_float == 0:
                fare_desc = "no fare recorded"
            elif fare_float < 10:
                fare_desc = "a low fare"
            elif fare_float < 30:
                fare_desc = "a moderate fare"
            elif fare_float < 100:
                fare_desc = "a high fare"
            else:
                fare_desc = "a very high fare"

            parts.append(f"paid {fare_desc} (£{fare_float:.2f})")

        # Add embarkation information
        embarked = data.get("Embarked")
        if embarked and not pd.isna(embarked):
            port = self.embark_map.get(str(embarked), f"port {embarked}")
            parts.append(f"embarked at {port}")

        # Add cabin information
        cabin = data.get("Cabin")
        if cabin and not pd.isna(cabin) and str(cabin).lower() != "nan":
            parts.append(f"stayed in cabin {cabin}")

        # Join all parts with proper punctuation
        text = ". ".join(parts)
        if not text.endswith("."):
            text += "."

        return text

    def get_feature_metadata(self) -> dict[str, FeatureMetadata]:
        """Get Titanic-specific feature metadata.

        Returns:
            Dictionary of feature metadata
        """
        return {
            "PassengerId": FeatureMetadata(
                name="PassengerId", feature_type=FeatureType.CATEGORICAL, importance=0.1
            ),
            "Pclass": FeatureMetadata(
                name="Pclass",
                feature_type=FeatureType.CATEGORICAL,
                importance=0.9,
                domain_info={"values": [1, 2, 3]},
            ),
            "Name": FeatureMetadata(
                name="Name", feature_type=FeatureType.TEXT, importance=0.3
            ),
            "Sex": FeatureMetadata(
                name="Sex",
                feature_type=FeatureType.CATEGORICAL,
                importance=0.8,
                domain_info={"values": ["male", "female"]},
            ),
            "Age": FeatureMetadata(
                name="Age",
                feature_type=FeatureType.NUMERICAL,
                importance=0.7,
                statistics={"mean": 29.7, "std": 14.5, "min": 0, "max": 80},
            ),
            "SibSp": FeatureMetadata(
                name="SibSp",
                feature_type=FeatureType.NUMERICAL,
                importance=0.5,
                statistics={"mean": 0.52, "std": 1.1, "min": 0, "max": 8},
            ),
            "Parch": FeatureMetadata(
                name="Parch",
                feature_type=FeatureType.NUMERICAL,
                importance=0.5,
                statistics={"mean": 0.38, "std": 0.81, "min": 0, "max": 6},
            ),
            "Ticket": FeatureMetadata(
                name="Ticket", feature_type=FeatureType.CATEGORICAL, importance=0.2
            ),
            "Fare": FeatureMetadata(
                name="Fare",
                feature_type=FeatureType.NUMERICAL,
                importance=0.8,
                statistics={"mean": 32.2, "std": 49.7, "min": 0, "max": 512},
            ),
            "Cabin": FeatureMetadata(
                name="Cabin", feature_type=FeatureType.CATEGORICAL, importance=0.4
            ),
            "Embarked": FeatureMetadata(
                name="Embarked",
                feature_type=FeatureType.CATEGORICAL,
                importance=0.3,
                domain_info={"values": ["C", "Q", "S"]},
            ),
        }


class CompetitionTemplateAugmenter(TabularToTextAugmenter):
    """Generic competition template-based augmenter.

    Provides a flexible template system for converting tabular data
    to text across different competition types.
    """

    def __init__(
        self,
        competition_type: str,
        template: str | None = None,
        feature_metadata: dict[str, FeatureMetadata] | None = None,
        config: AugmentationConfig | None = None,
        **kwargs,
    ):
        """Initialize competition template augmenter.

        Args:
            competition_type: Type of competition (e.g., "classification", "regression")
            template: Custom template string with {column_name} placeholders
            feature_metadata: Feature metadata dictionary
            config: Augmentation configuration
            **kwargs: Additional arguments for parent class
        """
        self.competition_type = competition_type
        self.template = template or self._get_default_template(competition_type)

        # Use template-based converter if no custom converter provided
        if "text_converter" not in kwargs:
            kwargs["text_converter"] = self._template_converter

        super().__init__(
            config=config or AugmentationConfig(),
            feature_metadata=feature_metadata or {},
            **kwargs,
        )

    def _get_default_template(self, competition_type: str) -> str:
        """Get default template for competition type.

        Args:
            competition_type: Type of competition

        Returns:
            Template string
        """
        templates = {
            "classification": "Sample with features: {features}. Target class: {target}",
            "regression": "Instance with attributes: {features}. Target value: {target}",
            "clustering": "Data point with characteristics: {features}",
            "time_series": "Time {time}: {features}. Next value: {target}",
            "nlp": "Text: {text}. Label: {label}",
            "computer_vision": "Image {image_id} with metadata: {metadata}",
        }

        return templates.get(
            competition_type,
            "Record with data: {features}",  # Generic fallback
        )

    def _template_converter(self, features: dict[str, Any]) -> str:
        """Convert features to text using template.

        Args:
            features: Feature dictionary

        Returns:
            Text representation
        """
        # Prepare template variables
        template_vars = features.copy()

        # Create features summary if not present
        if "{features}" in self.template and "features" not in template_vars:
            feature_parts = []
            for key, value in features.items():
                if key not in ["target", "label", "text", "image_id"]:
                    if value is not None and not pd.isna(value):
                        feature_parts.append(f"{key}={value}")
            template_vars["features"] = ", ".join(feature_parts)

        # Format template
        try:
            text = self.template.format(**template_vars)
        except KeyError as e:
            # logger.warning(f"Missing template variable: {e}")
            # Fallback to simple representation
            text = str(features)

        return text

    @classmethod
    def from_competition_name(
        cls,
        competition_name: str,
        feature_metadata: dict[str, FeatureMetadata] | None = None,
        config: AugmentationConfig | None = None,
        **kwargs,
    ) -> "CompetitionTemplateAugmenter":
        """Create augmenter from competition name.

        Args:
            competition_name: Name of the competition
            feature_metadata: Feature metadata
            config: Augmentation configuration
            **kwargs: Additional arguments

        Returns:
            Configured augmenter
        """
        # Map competition names to types and templates
        competition_map = {
            "titanic": ("classification", None),  # Use TitanicAugmenter instead
            "house-prices": (
                "regression",
                "House with {MSSubClass} style, {LotArea} sq ft lot, "
                "{OverallQual}/10 quality. Price: ${SalePrice}",
            ),
            "digit-recognizer": (
                "computer_vision",
                "Handwritten digit image {ImageId}. Label: {Label}",
            ),
            "nlp-getting-started": ("nlp", "Tweet: {text}. Disaster-related: {target}"),
        }

        if competition_name in competition_map:
            comp_type, template = competition_map[competition_name]
            return cls(
                competition_type=comp_type,
                template=template,
                feature_metadata=feature_metadata,
                config=config,
                **kwargs,
            )

        # Default to generic classification
        return cls(
            competition_type="classification",
            feature_metadata=feature_metadata,
            config=config,
            **kwargs,
        )
