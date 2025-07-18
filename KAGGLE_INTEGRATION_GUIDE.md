# Kaggle Integration Guide

This guide covers the comprehensive Kaggle integration features in the MLX-BERT playground, including competition management, automated submissions, and MLflow experiment tracking.

## Table of Contents
- [Setup](#setup)
- [Competition Management](#competition-management)
- [Training & Submission Workflow](#training--submission-workflow)
- [Automated Submissions](#automated-submissions)
- [MLflow Integration](#mlflow-integration)
- [Multi-Competition Support](#multi-competition-support)
- [Advanced Features](#advanced-features)

## Setup

### Kaggle API Credentials

1. Create a Kaggle account at https://www.kaggle.com
2. Go to Account Settings → API → Create New API Token
3. Save the `kaggle.json` file to `~/.kaggle/kaggle.json`
4. Set permissions: `chmod 600 ~/.kaggle/kaggle.json`

### Verify Installation

```bash
# Check Kaggle CLI is working
uv run python mlx_bert_cli.py kaggle-competitions --limit 5
```

## Competition Management

### Browse Competitions

```bash
# List all active competitions
uv run python mlx_bert_cli.py kaggle-competitions

# Filter by category
uv run python mlx_bert_cli.py kaggle-competitions --category tabular

# Search competitions
uv run python mlx_bert_cli.py kaggle-competitions --search "nlp classification"

# Sort by prize money
uv run python mlx_bert_cli.py kaggle-competitions --sort prize
```

### Download Competition Data

```bash
# Download competition files
uv run python mlx_bert_cli.py kaggle-download titanic --output data/titanic

# Download without unzipping
uv run python mlx_bert_cli.py kaggle-download titanic --no-unzip
```

## Training & Submission Workflow

### 1. Train Your Model

```bash
# Standard training with MLflow tracking
uv run python mlx_bert_cli.py train \
    --train data/titanic/train.csv \
    --val data/titanic/val.csv \
    --batch-size 32 \
    --epochs 10 \
    --experiment kaggle_titanic
```

### 2. Generate Predictions

```bash
# Generate predictions from best checkpoint
uv run python mlx_bert_cli.py predict \
    --test data/titanic/test.csv \
    --checkpoint output/run_001/best_model_accuracy \
    --output submissions/titanic_submission.csv
```

### 3. Submit to Kaggle

```bash
# Submit with custom message
uv run python mlx_bert_cli.py kaggle-submit titanic submissions/titanic_submission.csv \
    --message "MLX-BERT with attention optimization" \
    --checkpoint output/run_001/best_model_accuracy
```

### 4. Check Results

```bash
# View leaderboard
uv run python mlx_bert_cli.py kaggle-leaderboard titanic --top 100

# Check submission history
uv run python mlx_bert_cli.py kaggle-history titanic --limit 20

# Generate detailed report
uv run python mlx_bert_cli.py kaggle-history titanic \
    --report reports/titanic_submissions.json
```

## Automated Submissions

### Auto-Submit Command

The CLI includes an auto-submit command that generates predictions and submits in one step:

```bash
# Auto-submit from checkpoint
uv run python mlx_bert_cli.py kaggle-auto-submit titanic \
    output/run_001/best_model_accuracy \
    data/titanic/test.csv \
    --config configs/competitions/titanic.yaml
```

### Continuous Monitoring

Use the automated workflow script for continuous monitoring and submission:

```bash
# One-time auto submission of best models
uv run python scripts/auto_submit_workflow.py titanic \
    --test-data data/titanic/test.csv \
    --top-n 3 \
    --val-threshold 0.001

# Continuous monitoring mode
uv run python scripts/auto_submit_workflow.py titanic \
    --test-data data/titanic/test.csv \
    --monitor \
    --check-interval 300
```

## MLflow Integration

All Kaggle submissions are automatically tracked in MLflow:

### Tracked Metrics
- `kaggle_public_score`: Public leaderboard score
- `kaggle_private_score`: Private score (when available)
- `kaggle_leaderboard_rank`: Current leaderboard position
- `submission_time_seconds`: Time taken to submit

### Tracked Artifacts
- Submission CSV files
- Model checkpoints
- Submission metadata JSON
- Leaderboard snapshots

### View MLflow UI

```bash
# Launch MLflow UI
mlflow ui --backend-store-uri ./output/mlruns

# Navigate to experiment: kaggle_<competition_id>
```

### Enhanced Tracking with KaggleMLflowTracker

```python
from utils.kaggle_mlflow_tracking import KaggleMLflowTracker

# Initialize tracker
tracker = KaggleMLflowTracker("titanic")

# Start competition run
run_id = tracker.start_competition_run(
    run_name="experiment_001",
    model_type="MLX-ModernBERT",
    tags={"variant": "with_augmentation"}
)

# Log submission
tracker.log_submission(
    submission_file=Path("submission.csv"),
    message="Baseline model",
    public_score=0.8234,
    leaderboard_rank=1337
)

# Compare with baseline
tracker.compare_with_baseline(
    current_score=0.8234,
    baseline_score=0.8100
)

# Create submission dashboard
tracker.create_submission_dashboard(
    output_path=Path("reports/submission_dashboard")
)
```

## Multi-Competition Support

### Competition Configuration Files

Create YAML configuration files in `configs/competitions/`:

```yaml
# configs/competitions/your_competition.yaml
competition_id: your-competition-id
title: "Competition Title"

data:
  train_file: train.csv
  test_file: test.csv
  
columns:
  id: ID
  target: target
  features:
    - feature1
    - feature2
    
submission:
  id_column: ID
  target_column: target
  format: csv
  
evaluation:
  metric: accuracy
  higher_is_better: true
  
model:
  suggested_batch_size: 32
  suggested_epochs: 10
  suggested_lr: 2e-5
```

### Using Competition Configs

```bash
# Train with competition-specific config
uv run python mlx_bert_cli.py train \
    --train data/competition/train.csv \
    --val data/competition/val.csv \
    --config configs/competitions/your_competition.yaml

# Auto-submit with config
uv run python mlx_bert_cli.py kaggle-auto-submit your-competition \
    checkpoint_path test_data.csv \
    --config configs/competitions/your_competition.yaml
```

## Advanced Features

### Kaggle Datasets API

```bash
# Search datasets
uv run python mlx_bert_cli.py kaggle-datasets \
    --search "text classification" \
    --tag nlp

# List datasets by user
uv run python mlx_bert_cli.py kaggle-datasets \
    --user "username" \
    --sort votes
```

### Ensemble Submissions

```python
from utils.kaggle_mlflow_tracking import KaggleMLflowTracker

tracker = KaggleMLflowTracker("titanic")

# Log ensemble submission
tracker.log_ensemble_submission(
    model_weights={"model1": 0.6, "model2": 0.4},
    individual_scores={"model1": 0.82, "model2": 0.81},
    ensemble_score=0.83
)
```

### Competition Experiment Comparison

```python
from utils.kaggle_mlflow_tracking import CompetitionExperimentManager

manager = CompetitionExperimentManager("titanic")

# Create experiment variants
variants = manager.create_experiment_variants([
    "baseline",
    "with_augmentation",
    "ensemble",
    "advanced_features"
])

# Compare results
comparison_df = manager.compare_experiment_variants()
print(comparison_df)
```

## Best Practices

### 1. Experiment Organization
- Use descriptive experiment names: `kaggle_<competition>_<variant>`
- Tag runs with model architecture and key parameters
- Log all hyperparameters and configuration

### 2. Submission Strategy
- Start with baseline models to establish benchmarks
- Use validation scores to filter submissions
- Track all submissions in MLflow for analysis
- Set up automated submissions for overnight training

### 3. Rate Limiting
- Kaggle limits submissions (typically 5 per day)
- Use `--dry-run` flag to test workflow
- Implement validation score thresholds
- Space out submissions with delays

### 4. Reproducibility
- Save all model checkpoints
- Log random seeds and configurations
- Version control competition configs
- Document preprocessing steps

## Troubleshooting

### Common Issues

1. **Authentication Error**
   ```bash
   # Check Kaggle credentials
   ls -la ~/.kaggle/kaggle.json
   # Should show: -rw------- (600 permissions)
   ```

2. **Competition Not Found**
   - Ensure you've accepted competition rules on Kaggle website
   - Use exact competition ID from URL

3. **Submission Format Error**
   - Check sample submission file format
   - Verify column names match exactly
   - Ensure correct data types

4. **Rate Limiting**
   - Wait 24 hours for limit reset
   - Use automated workflow with daily limits
   - Check submission history to track usage

## Example Workflows

### Complete Titanic Workflow

```bash
# 1. Download data
uv run python mlx_bert_cli.py kaggle-download titanic

# 2. Train model
uv run python mlx_bert_cli.py train \
    --train data/titanic/train.csv \
    --val data/titanic/val.csv \
    --config configs/production.json \
    --experiment kaggle_titanic

# 3. Auto-submit best model
uv run python mlx_bert_cli.py kaggle-auto-submit titanic \
    output/run_*/best_model_accuracy \
    data/titanic/test.csv

# 4. View results
uv run python mlx_bert_cli.py kaggle-leaderboard titanic
uv run python mlx_bert_cli.py kaggle-history titanic --report reports/titanic.json
```

### Automated Overnight Training

```bash
# Run training and auto-submit when complete
nohup bash -c '
    uv run python mlx_bert_cli.py train \
        --train data/comp/train.csv \
        --val data/comp/val.csv \
        --epochs 50 \
        --experiment kaggle_comp_overnight && \
    uv run python scripts/auto_submit_workflow.py comp \
        --test-data data/comp/test.csv \
        --top-n 1
' > overnight_training.log 2>&1 &
```

## Summary

The Kaggle integration provides:
- ✅ Complete competition lifecycle management
- ✅ Automated submission workflows
- ✅ MLflow experiment tracking
- ✅ Multi-competition support
- ✅ Leaderboard monitoring
- ✅ Dataset discovery and download
- ✅ Ensemble tracking
- ✅ Submission history and analytics

This integration transforms the MLX-BERT playground into a comprehensive Kaggle competition platform with professional-grade experiment tracking and automation capabilities.