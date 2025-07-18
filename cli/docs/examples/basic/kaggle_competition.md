# Kaggle Competition Workflow

This guide walks through a complete Kaggle competition workflow using the MLX BERT CLI.

## Prerequisites

1. Install the MLX BERT CLI
2. Set up Kaggle credentials:
   ```bash
   kaggle config set -n username -v YOUR_USERNAME
   kaggle config set -n key -v YOUR_API_KEY
   ```

## Step 1: Initialize Project

Create a new project for your competition:

```bash
# Create project directory
bert init titanic-competition
cd titanic-competition

# Initialize with Kaggle template
bert config init --template kaggle
```

## Step 2: Explore Competition

Browse and download competition data:

```bash
# Search for competition
bert kaggle competitions search "titanic"

# Get competition info
bert kaggle competitions info titanic

# Download competition data
bert kaggle download titanic --path data/
```

## Step 3: Prepare Configuration

Create a training configuration:

```bash
# Generate config from template
bert config init --interactive

# Or create manually
cat > configs/kaggle.yaml << EOF
model:
  type: modernbert
  use_mlx_embeddings: true
  
training:
  epochs: 10
  batch_size: 32
  learning_rate: 2e-5
  early_stopping: true
  save_best_only: true
  
data:
  train_path: data/train.csv
  val_split: 0.2
  text_column: auto
  label_column: auto
  
kaggle:
  competition: titanic
  auto_submit: true
EOF
```

## Step 4: Train Model

Train your model with the configuration:

```bash
# Start MLflow tracking
bert mlflow server start

# Train model
bert train --config configs/kaggle.yaml \
    --experiment titanic-baseline \
    --run-name v1-modernbert

# Monitor training
bert mlflow ui
```

## Step 5: Evaluate Performance

Check model performance:

```bash
# Evaluate on validation set
bert model evaluate \
    --checkpoint output/best_model \
    --data data/train.csv \
    --split validation

# Compare with other runs
bert mlflow runs compare --experiment titanic-baseline
```

## Step 6: Generate Predictions

Create predictions for submission:

```bash
# Generate predictions
bert predict \
    --test data/test.csv \
    --checkpoint output/best_model \
    --output submissions/submission_v1.csv

# Verify predictions
head submissions/submission_v1.csv
```

## Step 7: Submit to Kaggle

Submit your predictions:

```bash
# Manual submission
bert kaggle submit create \
    --competition titanic \
    --file submissions/submission_v1.csv \
    --message "ModernBERT baseline with MLX"

# Or auto-submit from checkpoint
bert kaggle submit auto \
    --competition titanic \
    --checkpoint output/best_model \
    --test data/test.csv
```

## Step 8: Track Progress

Monitor your competition progress:

```bash
# View leaderboard position
bert kaggle leaderboard titanic --highlight-user

# Check submission history
bert kaggle submit history --competition titanic

# Generate submission report
bert kaggle submit report \
    --competition titanic \
    --output reports/submission_analysis.html
```

## Step 9: Iterate and Improve

### Try Different Models

```bash
# CNN-enhanced model
bert train --config configs/kaggle.yaml \
    --model modernbert-cnn \
    --run-name v2-cnn-model

# With data augmentation
bert train --config configs/kaggle.yaml \
    --augment \
    --run-name v3-augmented
```

### Hyperparameter Tuning

```bash
# Create tuning configuration
cat > configs/tuning.yaml << EOF
search_space:
  learning_rate: [1e-5, 2e-5, 5e-5]
  batch_size: [16, 32, 64]
  warmup_steps: [500, 1000]

tuning:
  n_trials: 20
  metric: val_accuracy
  direction: maximize
EOF

# Run hyperparameter search
bert tune --config configs/tuning.yaml \
    --base-config configs/kaggle.yaml
```

### Ensemble Models

```bash
# Train multiple models
for i in {1..5}; do
    bert train --config configs/kaggle.yaml \
        --seed $i \
        --run-name ensemble-model-$i
done

# Create ensemble predictions
bert ensemble predict \
    --checkpoints output/ensemble-model-*/best_model \
    --test data/test.csv \
    --method voting
```

## Step 10: Final Submission

Prepare your best submission:

```bash
# Select best model
bert mlflow runs best \
    --experiment titanic-baseline \
    --metric val_accuracy

# Generate final predictions
bert predict \
    --test data/test.csv \
    --checkpoint output/run_xyz/best_model \
    --output submissions/final_submission.csv

# Submit with detailed message
bert kaggle submit create \
    --competition titanic \
    --file submissions/final_submission.csv \
    --message "Final submission: ModernBERT + CNN + augmentation, \
               val_acc=0.895, 5-fold CV, ensemble of 3 models"
```

## Automation Script

Create a script to automate the workflow:

```bash
#!/bin/bash
# kaggle_pipeline.sh

# Configuration
COMPETITION="titanic"
EXPERIMENT="titanic-auto-$(date +%Y%m%d)"

# Download data
bert kaggle download $COMPETITION --path data/

# Train models
for config in configs/*.yaml; do
    name=$(basename $config .yaml)
    bert train --config $config \
        --experiment $EXPERIMENT \
        --run-name $name
done

# Auto-submit best model
bert kaggle submit auto \
    --competition $COMPETITION \
    --experiment $EXPERIMENT \
    --select-best val_accuracy

# Generate report
bert kaggle submit report \
    --competition $COMPETITION \
    --experiment $EXPERIMENT
```

## Tips and Best Practices

1. **Start Simple**: Begin with a basic model and gradually add complexity
2. **Track Everything**: Use MLflow to track all experiments
3. **Validate Locally**: Always validate locally before submitting
4. **Read Forums**: Check Kaggle forums for competition-specific tips
5. **Ensemble**: Combine multiple models for better performance
6. **Cross-Validation**: Use CV for more robust model selection

## Common Issues

### Out of Memory
```bash
# Reduce batch size
bert train --config configs/kaggle.yaml --batch-size 16

# Use gradient accumulation
bert train --config configs/kaggle.yaml \
    --batch-size 8 --grad-accum 4
```

### Slow Training
```bash
# Enable MLX optimizations
bert train --config configs/kaggle.yaml \
    --mlx-embeddings \
    --mixed-precision
```

### Poor Performance
```bash
# Try different model architectures
bert train --model modernbert-cnn

# Adjust hyperparameters
bert train --lr 5e-5 --warmup 1000
```