# MLX BERT for Kaggle Competitions

A state-of-the-art BERT implementation optimized for Apple Silicon using MLX, designed specifically for winning Kaggle competitions. This project provides modern BERT architectures, efficient data pipelines, and comprehensive training infrastructure with a beautiful CLI interface.

## 🚀 Features

- **Modern BERT Architectures**: Classic BERT, ModernBERT (Answer.AI 2024), and neoBERT implementations
- **Apple Silicon Optimization**: Native MLX framework support for maximum performance on M1/M2/M3
- **Kaggle Integration**: Direct competition downloads, submissions, leaderboard tracking
- **Rich CLI**: Beautiful command-line interface with progress bars and formatted output
- **LoRA Support**: Efficient fine-tuning with Low-Rank Adaptation and QLoRA
- **MLflow Tracking**: Comprehensive experiment tracking and model registry
- **Cross-Validation**: Built-in K-fold CV with out-of-fold predictions
- **Ensemble Methods**: Voting, blending, and stacking for better scores
- **Text Conversion**: Convert tabular data to natural language for BERT

## 📋 Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Project Structure](#project-structure)
- [CLI Commands](#cli-commands)
- [Model Architectures](#model-architectures)
- [Training](#training)
- [Kaggle Competitions](#kaggle-competitions)
- [Configuration](#configuration)
- [Performance](#performance)
- [Contributing](#contributing)

## 🛠️ Installation

### Prerequisites

- Apple Silicon Mac (M1/M2/M3)
- Python 3.10+
- Kaggle API credentials (for competition features)

### Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/bert-playground.git
cd bert-playground

# Install dependencies using uv (recommended)
pip install uv
uv sync

# Or using pip
pip install -r requirements.txt
```

### Kaggle API Setup

```bash
# Create API token at https://www.kaggle.com/account
mkdir -p ~/.kaggle
cp kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json
```

## 🎯 Quick Start

### Train on Titanic Competition

```bash
# Download Titanic data
bert kaggle download titanic --output data/titanic

# Train with ModernBERT
bert train \
    --train data/titanic/train.csv \
    --val data/titanic/val.csv \
    --model modernbert \
    --epochs 5 \
    --batch-size 32

# Generate predictions
bert predict \
    --test data/titanic/test.csv \
    --checkpoint output/best_model \
    --output submission.csv

# Submit to Kaggle
bert kaggle submit titanic submission.csv \
    --message "ModernBERT with MLX"
```

### Use Configuration Files

```bash
# Train with production config
bert train \
    --train data/train.csv \
    --val data/val.csv \
    --config configs/production.yaml

# Train with LoRA for efficiency
bert train \
    --train data/train.csv \
    --val data/val.csv \
    --use-lora \
    --lora-preset balanced
```

## 💻 CLI Commands

### Core Commands

```bash
# Train a model
bert train --train data/train.csv --val data/val.csv

# Generate predictions
bert predict --test data/test.csv --checkpoint output/best_model

# Benchmark performance
bert benchmark --batch-size 64 --steps 100

# System information
bert info --mlx --dependencies
```

### Kaggle Commands

```bash
# List competitions
bert kaggle competitions --category tabular --active-only

# Download competition data
bert kaggle download titanic --output data/

# Submit predictions
bert kaggle submit titanic submission.csv

# View leaderboard
bert kaggle leaderboard titanic --top 50

# Check submission history
bert kaggle history titanic --limit 10
```

### MLflow Commands

```bash
# Start MLflow server
bert mlflow server --port 5000

# View experiments
bert mlflow experiments list

# Compare runs
bert mlflow runs compare run1 run2

# Launch UI
bert mlflow ui
```

### Model Commands

```bash
# Serve model via API
bert model serve output/best_model --port 8080

# Export model
bert model export output/best_model --format onnx

# Evaluate model
bert model evaluate output/best_model data/test.csv

# Inspect architecture
bert model inspect output/best_model --layers
```

## 📁 Project Structure

```
bert-playground/
├── cli/                    # CLI interface
│   ├── commands/          # Command implementations
│   │   ├── core/         # train, predict, benchmark, info
│   │   ├── kaggle/       # Competition commands
│   │   ├── mlflow/       # Experiment tracking
│   │   └── model/        # Model management
│   └── utils/            # CLI utilities
├── models/                # Model implementations
│   ├── bert/             # BERT architectures
│   ├── heads/            # Task-specific heads
│   ├── lora/             # LoRA adapters
│   └── factory.py        # Model creation
├── data/                  # Data handling
│   ├── core/             # Base classes and protocols
│   ├── loaders/          # Data loading strategies
│   ├── templates/        # Text conversion
│   └── kaggle/           # Competition datasets
├── training/              # Training infrastructure
│   ├── core/             # Base trainer and protocols
│   ├── callbacks/        # Training callbacks
│   ├── metrics/          # Evaluation metrics
│   └── kaggle/           # Competition features
└── configs/              # Configuration files
```

## 🧠 Model Architectures

### Classic BERT
- Standard BERT implementation
- 12 layers, 768 hidden size, 12 attention heads
- Full compatibility with HuggingFace models

### ModernBERT
- Answer.AI's 2024 architecture improvements
- RoPE embeddings, GeGLU activation
- 8192 sequence length support
- Alternating local/global attention

### neoBERT
- Efficient 250M parameter variant
- 28 layers with SwiGLU activation
- Optimized for resource-constrained training

### Task Heads
- **Binary Classification**: Focal loss support
- **Multiclass Classification**: Label smoothing
- **Multilabel Classification**: Adaptive thresholds
- **Regression**: MSE/MAE/Huber loss
- **Ordinal Regression**: Cumulative logits
- **Time Series**: Multi-step predictions

## 🏋️ Training

### Basic Training

```python
from models import create_model
from training import create_trainer
from data import create_data_pipeline

# Create model
model = create_model("modernbert", num_labels=2)

# Create data pipeline
train_loader, val_loader = create_data_pipeline(
    "titanic",
    batch_size=32
)

# Train
trainer = create_trainer(model, config="production")
result = trainer.train(train_loader, val_loader)
```

### Advanced Features

```python
# Train with LoRA
model = create_model(
    "modernbert",
    use_lora=True,
    lora_preset="balanced"
)

# Cross-validation
kaggle_trainer = create_kaggle_trainer(model, "titanic")
cv_results = kaggle_trainer.train_with_cv(train_loader, n_folds=5)

# Ensemble training
ensemble_result = kaggle_trainer.train_ensemble(
    train_loader,
    n_models=5,
    strategy="voting"
)
```

## 🏆 Kaggle Competitions

### Supported Features

1. **Cross-Validation**
   - K-fold, stratified, group, time series
   - Out-of-fold predictions
   - Per-fold model saving

2. **Ensemble Methods**
   - Voting (hard/soft)
   - Blending (weighted average)
   - Stacking (meta-models)

3. **Advanced Techniques**
   - Pseudo-labeling
   - Test-time augmentation
   - Adversarial validation

### Competition Workflow

```bash
# 1. Download competition
bert kaggle download house-prices

# 2. Train with CV
bert train \
    --train data/train.csv \
    --config configs/kaggle.yaml \
    --cv-folds 5 \
    --save-oof-predictions

# 3. Create ensemble
bert train \
    --train data/train.csv \
    --ensemble-size 5 \
    --ensemble-strategy voting

# 4. Generate submission
bert predict \
    --test data/test.csv \
    --checkpoint output/ensemble \
    --tta-rounds 5

# 5. Submit
bert kaggle auto-submit house-prices \
    output/ensemble data/test.csv
```

## ⚙️ Configuration

### YAML Configuration Example

```yaml
# config.yaml
model:
  architecture: modernbert
  hidden_size: 768
  num_layers: 12
  num_heads: 12

training:
  epochs: 10
  batch_size: 32
  learning_rate: 2e-5
  gradient_accumulation_steps: 2

optimizer:
  type: adamw
  weight_decay: 0.01

scheduler:
  type: cosine
  warmup_steps: 500

data:
  max_length: 256
  num_workers: 8
  prefetch_size: 4

callbacks:
  - type: early_stopping
    patience: 3
    metric: val_loss
  - type: model_checkpoint
    save_best_only: true
```

### Preset Configurations

- `quick`: Fast testing (1 epoch, small batch)
- `development`: Balanced for development
- `production`: Optimized production settings
- `kaggle`: Competition-optimized
- `memory_efficient`: Minimal memory usage

## 📊 Performance

### Expected Performance (M1/M2/M3)

| Model | Batch Size | Sequences/sec | Memory Usage |
|-------|------------|---------------|--------------|
| BERT-base | 32 | 15-20 | 4GB |
| ModernBERT | 32 | 12-18 | 5GB |
| BERT + LoRA | 64 | 25-30 | 3GB |
| BERT 4-bit | 128 | 40-50 | 2GB |

### Optimization Tips

1. **Batch Size**: Use powers of 2 (32, 64, 128)
2. **Gradient Accumulation**: Simulate larger batches
3. **LoRA**: Reduce memory by 50-70%
4. **Data Loading**: Use 8-16 workers
5. **Prefetching**: Set to 2-4x batch size

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

### Development Setup

```bash
# Install development dependencies
uv sync --dev

# Run tests
pytest tests/

# Run linting
ruff check .

# Format code
ruff format .
```

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Apple MLX team for the amazing framework
- Answer.AI for ModernBERT architecture
- HuggingFace for model implementations
- Kaggle community for competitions and datasets

## 📚 Resources

- [MLX Documentation](https://ml-explore.github.io/mlx/)
- [ModernBERT Paper](https://arxiv.org/abs/2024.modernbert)
- [Project Documentation](docs/)
- [API Reference](docs/api/)

## 🔗 Links

- [GitHub Repository](https://github.com/yourusername/bert-playground)
- [Issue Tracker](https://github.com/yourusername/bert-playground/issues)
- [Discussions](https://github.com/yourusername/bert-playground/discussions)

---

Built with ❤️ for the Kaggle community using Apple Silicon and MLX.