#!/usr/bin/env python3
"""
Create sample Spaceship Titanic dataset for development.
This mimics the actual competition data structure.
"""

import pandas as pd
import numpy as np
from pathlib import Path

def create_spaceship_titanic_sample():
    """Create sample data that matches the Spaceship Titanic competition format."""
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Number of samples
    n_train = 1000
    n_test = 500
    
    # Create passenger IDs
    train_ids = [f"{str(i).zfill(4)}_01" for i in range(n_train)]
    test_ids = [f"{str(i+n_train).zfill(4)}_01" for i in range(n_test)]
    
    # Home planets
    home_planets = ['Earth', 'Europa', 'Mars']
    
    # Destinations
    destinations = ['TRAPPIST-1e', 'PSO J318.5-22', '55 Cancri e']
    
    # Create training data
    train_data = {
        'PassengerId': train_ids,
        'HomePlanet': np.random.choice(home_planets, n_train, p=[0.5, 0.3, 0.2]),
        'CryoSleep': np.random.choice([True, False], n_train, p=[0.3, 0.7]),
        'Cabin': [f"{np.random.choice(['A', 'B', 'C', 'D', 'E', 'F', 'G'])}/{np.random.randint(1, 300)}/{np.random.choice(['P', 'S'])}" for _ in range(n_train)],
        'Destination': np.random.choice(destinations, n_train, p=[0.4, 0.3, 0.3]),
        'Age': np.random.normal(30, 15, n_train).clip(0, 80).astype(int),
        'VIP': np.random.choice([True, False], n_train, p=[0.1, 0.9]),
        'RoomService': np.random.exponential(100, n_train).astype(int),
        'FoodCourt': np.random.exponential(150, n_train).astype(int),
        'ShoppingMall': np.random.exponential(80, n_train).astype(int),
        'Spa': np.random.exponential(120, n_train).astype(int),
        'VRDeck': np.random.exponential(200, n_train).astype(int),
        'Name': [f"Firstname{i} Lastname{i}" for i in range(n_train)],
        'Transported': np.random.choice([True, False], n_train, p=[0.45, 0.55])
    }
    
    # Create test data (without Transported column)
    test_data = {
        'PassengerId': test_ids,
        'HomePlanet': np.random.choice(home_planets, n_test, p=[0.5, 0.3, 0.2]),
        'CryoSleep': np.random.choice([True, False], n_test, p=[0.3, 0.7]),
        'Cabin': [f"{np.random.choice(['A', 'B', 'C', 'D', 'E', 'F', 'G'])}/{np.random.randint(1, 300)}/{np.random.choice(['P', 'S'])}" for _ in range(n_test)],
        'Destination': np.random.choice(destinations, n_test, p=[0.4, 0.3, 0.3]),
        'Age': np.random.normal(30, 15, n_test).clip(0, 80).astype(int),
        'VIP': np.random.choice([True, False], n_test, p=[0.1, 0.9]),
        'RoomService': np.random.exponential(100, n_test).astype(int),
        'FoodCourt': np.random.exponential(150, n_test).astype(int),
        'ShoppingMall': np.random.exponential(80, n_test).astype(int),
        'Spa': np.random.exponential(120, n_test).astype(int),
        'VRDeck': np.random.exponential(200, n_test).astype(int),
        'Name': [f"Firstname{i+n_train} Lastname{i+n_train}" for i in range(n_test)]
    }
    
    # Add some missing values (as in real dataset)
    for col in ['HomePlanet', 'CryoSleep', 'Cabin', 'Destination', 'Age', 'VIP']:
        mask = np.random.random(n_train) < 0.05  # 5% missing
        train_data[col] = pd.Series(train_data[col]).where(~mask, None).values
        
        mask = np.random.random(n_test) < 0.05
        test_data[col] = pd.Series(test_data[col]).where(~mask, None).values
    
    # Add logical constraints
    # If in CryoSleep, no luxury spending
    for data in [train_data, test_data]:
        n = len(data['PassengerId'])
        cryo_mask = pd.Series(data['CryoSleep']).fillna(False).values
        for col in ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']:
            data[col][cryo_mask] = 0
    
    # Create DataFrames
    train_df = pd.DataFrame(train_data)
    test_df = pd.DataFrame(test_data)
    
    # Create output directory
    output_dir = Path("data/spaceship-titanic")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save files
    train_df.to_csv(output_dir / "train.csv", index=False)
    test_df.to_csv(output_dir / "test.csv", index=False)
    
    # Create sample submission
    sample_submission = pd.DataFrame({
        'PassengerId': test_ids,
        'Transported': np.random.choice([True, False], n_test)
    })
    sample_submission.to_csv(output_dir / "sample_submission.csv", index=False)
    
    print(f"Created sample data in {output_dir}:")
    print(f"  - train.csv: {len(train_df)} samples")
    print(f"  - test.csv: {len(test_df)} samples")
    print(f"  - sample_submission.csv")
    
    print("\nData overview:")
    print(f"Features: {', '.join([col for col in train_df.columns if col != 'Transported'])}")
    print(f"Target: Transported (binary classification)")
    print(f"Missing values: ~5% in most columns")
    
    return train_df, test_df


if __name__ == "__main__":
    train_df, test_df = create_spaceship_titanic_sample()
    
    print("\nTrain data shape:", train_df.shape)
    print("\nFirst few rows of training data:")
    print(train_df.head())
    
    print("\nTarget distribution:")
    print(train_df['Transported'].value_counts(normalize=True))