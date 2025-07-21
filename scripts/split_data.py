"""Split training data into train/validation sets."""

import sys

import pandas as pd
from sklearn.model_selection import train_test_split

if len(sys.argv) < 2:
    print("Usage: python split_data.py <input_file>")
    sys.exit(1)

# Read data
input_file = sys.argv[1]
df = pd.read_csv(input_file)

# Split data
train_df, val_df = train_test_split(
    df, test_size=0.2, random_state=42, stratify=df["Survived"]
)

# Save splits
train_df.to_csv("data/titanic/train_split.csv", index=False)
val_df.to_csv("data/titanic/val_split.csv", index=False)

print(f"Train set: {len(train_df)} samples")
print(f"Validation set: {len(val_df)} samples")
