# Kaggle Competition Workflow Guide

This guide provides comprehensive workflows for using MLX BERT to compete effectively in Kaggle competitions. It covers everything from initial setup to final submission strategies.

## Table of Contents

- [Competition Types](#competition-types)
- [General Workflow](#general-workflow)
- [Competition-Specific Workflows](#competition-specific-workflows)
- [Advanced Techniques](#advanced-techniques)
- [Best Practices](#best-practices)
- [Common Pitfalls](#common-pitfalls)

## Competition Types

### Supported Competition Types

1. **Binary Classification** (e.g., Titanic, Disaster Tweets)
2. **Multiclass Classification** (e.g., MNIST, Fashion-MNIST)
3. **Multilabel Classification** (e.g., Toxic Comments)
4. **Regression** (e.g., House Prices, Sales Forecasting)
5. **Ordinal Regression** (e.g., Ratings Prediction)
6. **Time Series** (e.g., Stock Prediction, Weather Forecasting)

## General Workflow

### 1. Competition Setup

```bash
# Set up Kaggle credentials
mkdir -p ~/.kaggle
cp kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json

# List available competitions
bert kaggle competitions --category tabular --active-only

# Download competition data
bert kaggle download <competition-name> --output data/<competition-name>
```

### 2. Initial Data Exploration

```python
# Explore data structure
from data import create_data_pipeline
import pandas as pd

# Load data
train_df = pd.read_csv("data/competition/train.csv")
test_df = pd.read_csv("data/competition/test.csv")

# Basic statistics
print(f"Train shape: {train_df.shape}")
print(f"Test shape: {test_df.shape}")
print(f"Target distribution:\n{train_df['target'].value_counts()}")

# Check for missing values
print(f"Missing values:\n{train_df.isnull().sum()}")
```

### 3. Data Preparation

```bash
# Create train/validation split
bert data prepare data/competition/train.csv \
    --split-ratio 0.2 \
    --stratify target \
    --output data/competition/
```

### 4. Model Selection

```python
from models import create_model

# Choose architecture based on competition
model = create_model(
    "modernbert",  # or "bert", "neobert"
    head_type="binary_classification",
    num_labels=2,
    use_lora=True,  # For efficiency
    lora_preset="balanced"
)
```

### 5. Training

```bash
# Quick test run
bert train \
    --train data/competition/train.csv \
    --val data/competition/val.csv \
    --config configs/quick.yaml \
    --experiment competition_test

# Full training with cross-validation
bert train \
    --train data/competition/train.csv \
    --config configs/kaggle.yaml \
    --cv-folds 5 \
    --save-oof-predictions \
    --experiment competition_cv
```

### 6. Model Evaluation

```bash
# Evaluate single model
bert model evaluate output/best_model data/competition/val.csv \
    --metrics "accuracy,f1,auc" \
    --confusion-matrix

# Compare multiple runs
bert mlflow runs compare run1 run2 \
    --metric val_accuracy
```

### 7. Prediction and Submission

```bash
# Generate predictions
bert predict \
    --test data/competition/test.csv \
    --checkpoint output/best_model \
    --output submission.csv \
    --tta-rounds 5

# Submit to Kaggle
bert kaggle submit <competition-name> submission.csv \
    --message "ModernBERT with 5-fold CV and TTA"
```

## Competition-Specific Workflows

### Binary Classification (Titanic Example)

```bash
# 1. Download and prepare data
bert kaggle download titanic --output data/titanic

# 2. Train with specific configuration
bert train \
    --train data/titanic/train.csv \
    --val data/titanic/val.csv \
    --model modernbert \
    --head binary \
    --use-focal-loss \
    --focal-gamma 2.0 \
    --epochs 10 \
    --batch-size 32

# 3. Cross-validation for robust predictions
bert train \
    --train data/titanic/train.csv \
    --cv-folds 5 \
    --cv-strategy stratified \
    --save-oof-predictions \
    --experiment titanic_cv

# 4. Ensemble multiple models
bert train \
    --train data/titanic/train.csv \
    --ensemble-size 5 \
    --ensemble-strategy voting \
    --experiment titanic_ensemble

# 5. Generate and submit predictions
bert kaggle auto-submit titanic \
    output/ensemble data/titanic/test.csv \
    --message "5-fold CV ensemble with voting"
```

### Multiclass Classification (MNIST Example)

```python
from training import create_kaggle_trainer
from models import create_model

# Create model with multiclass head
model = create_model(
    "bert",
    head_type="multiclass",
    num_labels=10,
    label_smoothing=0.1  # Regularization
)

# Configure trainer
trainer = create_kaggle_trainer(
    model,
    competition="mnist",
    cv_folds=3,  # Less folds for larger datasets
    ensemble_size=3
)

# Train with cross-validation
cv_results = trainer.train_with_cv(train_loader)
print(f"CV Score: {cv_results.mean_score:.4f} ± {cv_results.std_score:.4f}")

# Create submission
submission = trainer.create_submission(test_loader, use_tta=True)
```

### Regression (House Prices Example)

```bash
# Use regression-specific configuration
bert train \
    --train data/house-prices/train.csv \
    --val data/house-prices/val.csv \
    --model modernbert \
    --head regression \
    --loss-type huber \
    --uncertainty-estimation \
    --epochs 15 \
    --lr 1e-5

# Evaluate with regression metrics
bert model evaluate output/best_model data/house-prices/val.csv \
    --metrics "mse,rmse,mae,r2"
```

### Multilabel Classification (Toxic Comments Example)

```python
from models import create_model

# Create multilabel model
model = create_model(
    "modernbert",
    head_type="multilabel",
    num_labels=6,  # 6 toxicity types
    pos_weight=[2.5, 3.0, 2.0, 1.5, 2.0, 1.8],  # Class weights
    adaptive_threshold=True
)

# Train with appropriate metrics
trainer = create_trainer(
    model,
    metrics=["hamming_loss", "macro_f1", "micro_f1"]
)
```

## Advanced Techniques

### 1. Pseudo-Labeling

```python
from training.kaggle import PseudoLabelingConfig

# Configure pseudo-labeling
pseudo_config = PseudoLabelingConfig(
    unlabeled_data=unlabeled_loader,
    threshold=0.95,  # High confidence threshold
    update_frequency=3,  # Update every 3 epochs
    soft_labels=True  # Use soft pseudo-labels
)

# Train with pseudo-labeling
trainer = create_kaggle_trainer(
    model,
    pseudo_labeling=pseudo_config
)
```

### 2. Advanced Ensembling

```python
# Stacking ensemble
from training.kaggle import EnsembleConfig

ensemble_config = EnsembleConfig(
    n_models=5,
    strategy="stacking",
    meta_model="ridge",  # Ridge regression as meta-learner
    diversity_method="different_seeds",
    blend_weights="optimize"  # Find optimal weights
)

trainer = create_kaggle_trainer(
    model,
    ensemble=ensemble_config
)
```

### 3. Test-Time Augmentation (TTA)

```bash
# Generate predictions with TTA
bert predict \
    --test data/test.csv \
    --checkpoint output/best_model \
    --tta-rounds 10 \
    --tta-aggregate median \
    --output submission_tta.csv
```

### 4. Feature Engineering

```python
from data.templates import CustomTemplateEngine

class CompetitionTemplateEngine(TextTemplateEngine):
    """Custom template for specific competition."""
    
    def create_template(self, row: Dict) -> str:
        # Custom feature engineering
        age_group = self._categorize_age(row['age'])
        price_range = self._categorize_price(row['price'])
        
        # Create rich text representation
        return f"""
        Customer profile: {age_group} age group with {row['income']} income.
        Purchase history: {row['purchase_count']} items in {price_range} range.
        Preferences: {row['categories']}
        """
```

### 5. Learning Rate Scheduling

```python
from training.callbacks import DynamicLRCallback

class CompetitionLRScheduler(DynamicLRCallback):
    """Competition-specific LR scheduling."""
    
    def on_epoch_end(self, trainer, state):
        # Aggressive decay after plateau
        if state.metrics['val_loss'] > self.best_loss:
            self.plateau_count += 1
            if self.plateau_count >= 2:
                trainer.optimizer.learning_rate *= 0.5
                self.plateau_count = 0
        else:
            self.best_loss = state.metrics['val_loss']
```

## Best Practices

### 1. Cross-Validation Strategy

```python
# Choose CV strategy based on data
cv_strategies = {
    "binary_balanced": "stratified",
    "binary_imbalanced": "stratified",
    "multiclass": "stratified",
    "regression": "kfold",
    "time_series": "timeseries",
    "grouped": "group"
}

# Always save OOF predictions
trainer.train_with_cv(
    train_loader,
    cv_strategy=cv_strategies[competition_type],
    save_oof=True
)
```

### 2. Experiment Tracking

```bash
# Use MLflow for all experiments
export MLFLOW_EXPERIMENT_NAME="competition_name"

# Tag experiments meaningfully
bert train \
    --experiment competition_v1 \
    --tags "model:modernbert,cv:5fold,tta:yes"

# Compare experiments
bert mlflow experiments compare \
    --metric best_val_score \
    --filter "tags.cv='5fold'"
```

### 3. Ensemble Diversity

```python
# Create diverse models
models = []

# Model 1: Different architecture
models.append(create_model("bert", **config))

# Model 2: Different preprocessing
models.append(create_model("modernbert", max_length=256))

# Model 3: Different loss function
models.append(create_model("modernbert", use_focal_loss=True))

# Model 4: Different hyperparameters
models.append(create_model("modernbert", hidden_dropout=0.2))

# Model 5: Different random seed
models.append(create_model("modernbert", seed=42))
```

### 4. Submission Strategies

```bash
# 1. Single best model
bert predict --checkpoint output/best_single_model

# 2. CV ensemble average
bert predict --checkpoint output/cv_models --average

# 3. Weighted ensemble
bert predict \
    --checkpoints "model1:0.4,model2:0.3,model3:0.3" \
    --weighted

# 4. Blend multiple approaches
bert blend-predictions \
    submission1.csv:0.5 \
    submission2.csv:0.3 \
    submission3.csv:0.2 \
    --output final_submission.csv
```

## Common Pitfalls

### 1. Data Leakage

```python
# WRONG: Fit on entire dataset
scaler.fit(full_data)

# CORRECT: Fit only on training fold
for train_idx, val_idx in cv.split(X, y):
    scaler.fit(X[train_idx])
    X_train = scaler.transform(X[train_idx])
    X_val = scaler.transform(X[val_idx])
```

### 2. Overfitting to Public LB

```python
# Track both local CV and public LB scores
results = {
    "cv_score": cv_results.mean_score,
    "public_lb": submission_score,
    "difference": abs(cv_results.mean_score - submission_score)
}

# Large difference indicates overfitting
if results["difference"] > 0.02:
    print("Warning: Possible overfitting to public LB")
```

### 3. Improper Validation

```python
# For time series, respect temporal order
from sklearn.model_selection import TimeSeriesSplit

tscv = TimeSeriesSplit(n_splits=5)
for train_idx, val_idx in tscv.split(X):
    # Train only on past data
    train_data = data.iloc[train_idx]
    val_data = data.iloc[val_idx]
```

### 4. Not Using Competition Metric

```python
# Always optimize for competition metric
competition_metrics = {
    "titanic": "accuracy",
    "house-prices": "rmse",
    "toxic-comments": "mean_auc",
    "mnist": "accuracy"
}

trainer = create_trainer(
    model,
    metric_for_best_model=competition_metrics[competition],
    greater_is_better=is_higher_better[competition]
)
```

## Competition Checklist

### Before Starting
- [ ] Read competition description thoroughly
- [ ] Understand evaluation metric
- [ ] Check data licenses and rules
- [ ] Set up Kaggle API access
- [ ] Create experiment tracking

### Data Phase
- [ ] Explore data distributions
- [ ] Check for data quality issues
- [ ] Create validation strategy
- [ ] Engineer meaningful features
- [ ] Convert to text representations

### Model Phase
- [ ] Try multiple architectures
- [ ] Tune hyperparameters
- [ ] Use cross-validation
- [ ] Save OOF predictions
- [ ] Track all experiments

### Ensemble Phase
- [ ] Create diverse models
- [ ] Try different ensemble methods
- [ ] Optimize blend weights
- [ ] Validate ensemble performance

### Submission Phase
- [ ] Generate multiple submissions
- [ ] Use test-time augmentation
- [ ] Blend best approaches
- [ ] Document submission details
- [ ] Monitor public LB feedback

## Example End-to-End Script

```python
#!/usr/bin/env python3
"""Complete Kaggle competition pipeline."""

import argparse
from pathlib import Path

from data import create_data_pipeline
from models import create_model
from training import create_kaggle_trainer

def main(competition_name: str):
    # 1. Setup
    data_dir = Path(f"data/{competition_name}")
    
    # 2. Create data pipeline
    train_loader, val_loader = create_data_pipeline(
        competition_name,
        batch_size=32,
        max_length=256,
        augmentation=True
    )
    
    # 3. Create model
    model = create_model(
        "modernbert",
        competition=competition_name,
        use_lora=True
    )
    
    # 4. Create trainer
    trainer = create_kaggle_trainer(
        model,
        competition=competition_name,
        cv_folds=5,
        ensemble_size=5
    )
    
    # 5. Train with CV
    cv_results = trainer.train_with_cv(train_loader)
    print(f"CV Score: {cv_results.mean_score:.4f}")
    
    # 6. Create submission
    submission = trainer.create_submission(
        test_loader,
        use_tta=True,
        tta_rounds=10
    )
    
    # 7. Submit to Kaggle
    trainer.submit_to_kaggle(
        submission,
        message=f"CV: {cv_results.mean_score:.4f}"
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("competition", help="Competition name")
    args = parser.parse_args()
    
    main(args.competition)
```

## Resources

- [Kaggle Learn](https://www.kaggle.com/learn)
- [Competition Forums](https://www.kaggle.com/competitions)
- [MLX Documentation](https://ml-explore.github.io/mlx/)
- [Project Documentation](../README.md)