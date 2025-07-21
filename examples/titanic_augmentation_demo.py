#!/usr/bin/env python3
"""Demonstration of Titanic-specific augmentation strategy.

Shows how the new augmentation system replaces the old preprocessing plugin.
"""

from bert_playground.data.augmentation import (
    AugmentationConfig,
    AugmentationMode,
    CompetitionTemplateAugmenter,
    TabularAugmenter,
    TitanicAugmenter,
    get_registry,
)
from loguru import logger


def demo_titanic_augmenter():
    """Demonstrate Titanic-specific augmentation."""
    logger.info("=== Titanic Augmenter Demo ===")

    # Sample Titanic passenger data
    passenger_data = {
        "PassengerId": 1,
        "Pclass": 3,
        "Name": "Braund, Mr. Owen Harris",
        "Sex": "male",
        "Age": 22,
        "SibSp": 1,
        "Parch": 0,
        "Ticket": "A/5 21171",
        "Fare": 7.25,
        "Cabin": None,
        "Embarked": "S",
    }

    # Create Titanic augmenter
    augmenter = TitanicAugmenter()

    # Apply augmentation
    augmented = augmenter.apply(passenger_data, AugmentationConfig())

    logger.info(f"Original data: {passenger_data}")
    logger.info(f"Generated text: {augmented['text']}")

    # Get feature metadata
    metadata = augmenter.get_feature_metadata()
    logger.info(f"Number of features with metadata: {len(metadata)}")


def demo_tabular_with_titanic():
    """Demonstrate using Titanic augmenter with TabularAugmenter."""
    logger.info("\n=== Tabular Augmenter with Titanic Strategy ===")

    # Get the registry and check if Titanic is registered
    registry = get_registry()
    strategies = registry.list_strategies()
    logger.info(f"Available strategies: {strategies}")

    # Create a tabular augmenter with Titanic metadata
    augmenter = TitanicAugmenter()
    metadata = augmenter.get_feature_metadata()

    config = AugmentationConfig.from_mode(AugmentationMode.LIGHT)
    tabular_aug = TabularAugmenter(config, metadata)

    # Sample data with some noise to see augmentation
    sample = {
        "Age": 25.5,
        "Fare": 15.75,
        "Sex": "female",
        "Pclass": 2,
        "SibSp": 0,
        "Parch": 1,
    }

    # Apply augmentation multiple times
    logger.info(f"Original sample: {sample}")
    for i in range(3):
        augmented = tabular_aug.augment(sample)
        logger.info(f"Augmented {i + 1}: {augmented}")


def demo_competition_template():
    """Demonstrate competition template augmenter."""
    logger.info("\n=== Competition Template Augmenter Demo ===")

    # Create house prices augmenter
    house_aug = CompetitionTemplateAugmenter.from_competition_name(
        "house-prices", config=AugmentationConfig()
    )

    # Sample house data
    house_data = {
        "MSSubClass": "Two-story 1946+",
        "LotArea": 9600,
        "OverallQual": 7,
        "SalePrice": 208500,
    }

    augmented = house_aug.augment(house_data)
    logger.info(f"House data: {house_data}")
    logger.info(f"Generated text: {augmented['text']}")

    # Create custom template augmenter
    custom_aug = CompetitionTemplateAugmenter(
        competition_type="classification",
        template="Customer {customer_id}: {age} years old, {income} income level, {churn} churn risk",
    )

    customer = {"customer_id": "C123", "age": 35, "income": "high", "churn": "low"}

    augmented = custom_aug.augment(customer)
    logger.info(f"\nCustomer data: {customer}")
    logger.info(f"Generated text: {augmented['text']}")


def compare_old_vs_new():
    """Compare old preprocessing approach vs new augmentation."""
    logger.info("\n=== Old vs New Approach Comparison ===")

    logger.info("OLD APPROACH:")
    logger.info(
        "1. Use preprocessing plugin: bert prepare titanic train.csv output.csv"
    )
    logger.info("2. Plugin hardcoded for Titanic dataset")
    logger.info("3. Text conversion mixed with preprocessing")
    logger.info("4. No augmentation capabilities")

    logger.info("\nNEW APPROACH:")
    logger.info("1. Use augmentation strategy: augmenter = TitanicAugmenter()")
    logger.info("2. Strategy registered in global registry")
    logger.info("3. Clean separation of concerns")
    logger.info("4. Full augmentation support (noise, masking, etc.)")
    logger.info("5. Works with any dataset via metadata")


def main():
    """Run all demonstrations."""
    logger.info("Titanic Augmentation Strategy Demonstration")
    logger.info("=" * 50)

    demo_titanic_augmenter()
    demo_tabular_with_titanic()
    demo_competition_template()
    compare_old_vs_new()

    logger.info("\n" + "=" * 50)
    logger.info("Summary:")
    logger.info("- TitanicAugmenter replaces preprocessing/plugins/titanic.py")
    logger.info("- CompetitionTemplateAugmenter replaces templates module")
    logger.info("- Both work seamlessly with the augmentation framework")
    logger.info("- No more hardcoded dataset logic!")


if __name__ == "__main__":
    main()
