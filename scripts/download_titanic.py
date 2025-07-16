#!/usr/bin/env python3
"""Download Titanic dataset from Kaggle or create sample data."""

import pandas as pd
import numpy as np
from pathlib import Path
import os


def create_sample_titanic_data():
    """Create sample Titanic data for testing."""
    np.random.seed(42)
    n_samples = 891  # Standard Titanic training set size
    
    data = {
        'PassengerId': range(1, n_samples + 1),
        'Survived': np.random.binomial(1, 0.38, n_samples),  # ~38% survival rate
        'Pclass': np.random.choice([1, 2, 3], n_samples, p=[0.24, 0.21, 0.55]),
        'Name': [f"Passenger_{i}" for i in range(n_samples)],
        'Sex': np.random.choice(['male', 'female'], n_samples, p=[0.65, 0.35]),
        'Age': np.random.normal(30, 14, n_samples).clip(0.5, 80),
        'SibSp': np.random.poisson(0.5, n_samples).clip(0, 8),
        'Parch': np.random.poisson(0.4, n_samples).clip(0, 6),
        'Ticket': [f"TICKET_{i}" for i in range(n_samples)],
        'Fare': np.random.lognormal(3.0, 1.0, n_samples).clip(0, 500),
        'Cabin': np.random.choice([np.nan] + [f"C{i%100}" for i in range(100)], n_samples, p=[0.77] + [0.23/100]*100),
        'Embarked': np.random.choice(['S', 'C', 'Q'], n_samples, p=[0.72, 0.19, 0.09])
    }
    
    return pd.DataFrame(data)


def download_from_kaggle():
    """Download Titanic dataset from Kaggle (requires API key)."""
    try:
        import kaggle
        kaggle.api.authenticate()
        
        # Download the Titanic dataset
        kaggle.api.competition_download_files('titanic', path='data/titanic', unzip=True)
        print("Successfully downloaded Titanic dataset from Kaggle!")
        return True
    except Exception as e:
        print(f"Failed to download from Kaggle: {e}")
        return False


def main():
    """Main function to get Titanic data."""
    output_dir = Path("data/titanic")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Check if data already exists
    train_path = output_dir / "train.csv"
    test_path = output_dir / "test.csv"
    
    if train_path.exists() and test_path.exists():
        print("Titanic data already exists!")
        return
    
    # Try to download from Kaggle first
    if os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY"):
        if download_from_kaggle():
            return
    
    # Create sample data if Kaggle download fails
    print("Creating sample Titanic data for testing...")
    
    # Create training data
    train_df = create_sample_titanic_data()
    train_df.to_csv(train_path, index=False)
    print(f"Created training data: {train_path} ({len(train_df)} rows)")
    
    # Create test data (without Survived column)
    test_df = create_sample_titanic_data()
    test_df = test_df.drop('Survived', axis=1)
    test_df['PassengerId'] = range(892, 892 + len(test_df))  # Test IDs start at 892
    test_df.to_csv(test_path, index=False)
    print(f"Created test data: {test_path} ({len(test_df)} rows)")
    
    # Create validation split from training data
    val_size = int(0.2 * len(train_df))
    val_indices = np.random.choice(len(train_df), val_size, replace=False)
    val_df = train_df.iloc[val_indices]
    val_path = output_dir / "val.csv"
    val_df.to_csv(val_path, index=False)
    print(f"Created validation data: {val_path} ({len(val_df)} rows)")


if __name__ == "__main__":
    main()