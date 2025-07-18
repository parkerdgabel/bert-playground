#!/usr/bin/env python3
import pandas as pd
from pathlib import Path


def download_titanic_data():
    data_dir = Path(__file__).parent / "data"
    data_dir.mkdir(exist_ok=True)

    # For now, we'll create sample data that matches Titanic structure
    # In production, use: kaggle competitions download -c titanic

    # Create sample training data
    train_data = {
        "PassengerId": [1, 2, 3, 4, 5],
        "Survived": [0, 1, 1, 1, 0],
        "Pclass": [3, 1, 3, 1, 3],
        "Name": [
            "Braund, Mr. Owen Harris",
            "Cumings, Mrs. John Bradley (Florence Briggs Thayer)",
            "Heikkinen, Miss. Laina",
            "Futrelle, Mrs. Jacques Heath (Lily May Peel)",
            "Allen, Mr. William Henry",
        ],
        "Sex": ["male", "female", "female", "female", "male"],
        "Age": [22, 38, 26, 35, 35],
        "SibSp": [1, 1, 0, 1, 0],
        "Parch": [0, 0, 0, 0, 0],
        "Ticket": ["A/5 21171", "PC 17599", "STON/O2. 3101282", "113803", "373450"],
        "Fare": [7.2500, 71.2833, 7.9250, 53.1000, 8.0500],
        "Cabin": [None, "C85", None, "C123", None],
        "Embarked": ["S", "C", "S", "S", "S"],
    }

    # Create sample test data (without Survived column)
    test_data = {
        "PassengerId": [892, 893, 894, 895, 896],
        "Pclass": [3, 3, 2, 3, 3],
        "Name": [
            "Kelly, Mr. James",
            "Wilkes, Mrs. James (Ellen Needs)",
            "Myles, Mr. Thomas Francis",
            "Wirz, Mr. Albert",
            "Hirvonen, Mrs. Alexander (Helga E Lindqvist)",
        ],
        "Sex": ["male", "female", "male", "male", "female"],
        "Age": [34.5, 47, 62, 27, 22],
        "SibSp": [0, 1, 0, 0, 1],
        "Parch": [0, 0, 0, 0, 1],
        "Ticket": ["330911", "363272", "240276", "315154", "3101298"],
        "Fare": [7.8292, 7.0000, 9.6875, 8.6625, 12.2875],
        "Cabin": [None, None, None, None, None],
        "Embarked": ["Q", "S", "Q", "S", "S"],
    }

    # Save as CSV files
    pd.DataFrame(train_data).to_csv(data_dir / "train.csv", index=False)
    pd.DataFrame(test_data).to_csv(data_dir / "test.csv", index=False)

    print(f"Sample data saved to {data_dir}")
    print("To download real Titanic data, run: kaggle competitions download -c titanic")

    return data_dir


if __name__ == "__main__":
    download_titanic_data()
