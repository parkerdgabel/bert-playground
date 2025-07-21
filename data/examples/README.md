# Example Datasets

This directory contains sample datasets to help you get started with mlx-bert.

## Sample Movie Review Dataset

A simple binary classification dataset for sentiment analysis:
- `sample_train.csv`: 10 movie reviews with labels (1=positive, 0=negative)
- `sample_val.csv`: 5 movie reviews for validation
- `sample_test.csv`: 5 movie reviews without labels for prediction

### Quick Start

```bash
# Train a model on the sample data
mlx-bert train \
    --train data/examples/sample_train.csv \
    --val data/examples/sample_val.csv \
    --epochs 3 \
    --batch-size 4

# Generate predictions
mlx-bert predict \
    --test data/examples/sample_test.csv \
    --checkpoint output/run_*/checkpoints/final \
    --output predictions.csv
```

### Data Format

The CSV files should have the following columns:
- **Training/Validation**: `id`, `text`, `label`
- **Test**: `id`, `text`

The `text` column contains the input text, and `label` contains the target (0 or 1 for binary classification).

## Creating Your Own Dataset

To use your own data:

1. Format your CSV with appropriate columns
2. For multi-class classification, use integer labels (0, 1, 2, ...)
3. For regression tasks, use float values in the label column
4. Ensure text is properly escaped in CSV format

## Download Real Datasets

For real competition datasets:

```bash
# Download Titanic dataset
mlx-bert kaggle download titanic --output data/titanic

# Download any Kaggle competition
mlx-bert kaggle download <competition-name> --output data/<competition-name>
```