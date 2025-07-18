# Modular BERT Architecture

## Overview

The modular BERT architecture provides a clean separation between the BERT encoder and task-specific heads, making it easy to:
- Swap different BERT variants (standard, CNN-hybrid, MLX embeddings)
- Attach any head from the comprehensive heads collection
- Create custom combinations for specific Kaggle competitions
- Save and load complete models with their heads

## Architecture Components

### 1. BertCore
The core BERT model that provides standardized outputs through the `BertOutput` dataclass.

```python
from models.bert import BertCore, create_bert_core

# Create a BERT core model
bert = create_bert_core(
    hidden_size=768,
    num_hidden_layers=12,
    num_attention_heads=12
)

# Forward pass
outputs = bert(input_ids, attention_mask)

# Access different representations
cls_token = outputs.cls_output          # [batch, hidden_size]
mean_pooled = outputs.mean_pooled       # [batch, hidden_size]
max_pooled = outputs.max_pooled         # [batch, hidden_size]
last_hidden = outputs.last_hidden_state # [batch, seq_len, hidden_size]
```

### 2. BertOutput
Standardized output format that provides multiple pooling strategies:

```python
# Get pooled output by type
pooled = outputs.get_pooled_output("cls")    # CLS token
pooled = outputs.get_pooled_output("mean")   # Mean pooling
pooled = outputs.get_pooled_output("max")    # Max pooling
pooled = outputs.get_pooled_output("pooler") # BERT pooler output
```

### 3. BertWithHead
Combines BertCore with any task-specific head:

```python
from models.bert import create_bert_with_head
from models.heads.base_head import HeadType

# Create BERT with classification head
model = create_bert_with_head(
    head_type=HeadType.BINARY_CLASSIFICATION,
    num_labels=2,
    freeze_bert_layers=2  # Freeze first 2 layers
)

# Forward pass with labels
outputs = model(input_ids, attention_mask, labels=labels)
loss = outputs["loss"]
predictions = outputs["predictions"]
```

## Available Heads

### Classification Heads
- **BinaryClassificationHead**: 2-class problems with BCE loss
- **MulticlassClassificationHead**: N-class problems with cross-entropy
- **MultilabelClassificationHead**: Multiple labels per sample
- **EnsembleClassificationHead**: Combines multiple strategies

### Advanced Heads
- **TimeSeriesClassificationHead**: LSTM-based temporal modeling
- **RankingHead**: Learning-to-rank approaches
- **ContrastiveLearningHead**: Similarity/retrieval tasks
- **MultiTaskHead**: Multiple objectives
- **OrdinalRegressionHead**: Ordered categories
- **HierarchicalClassificationHead**: Hierarchical labels

### Regression Heads
- **RegressionHead**: Standard regression with MSE/MAE/Huber loss
- **TimeSeriesRegressionHead**: Temporal regression

## Factory Functions

### 1. Create by Model Type
```python
from models.factory import create_model

# Create modular BERT with head
model = create_model(
    model_type="bert_with_head",
    head_type="multiclass_classification",
    num_labels=5,
    config={"hidden_size": 768}
)
```

### 2. Create for Task
```python
from models.factory import create_bert_for_task

# Create optimized model for specific task
model = create_bert_for_task(
    task="regression",  # or HeadType.REGRESSION
    num_labels=1,
    freeze_bert=True
)
```

### 3. Create for Competition
```python
from models.bert import create_bert_for_competition

# Auto-select best head for competition type
model = create_bert_for_competition(
    competition_type="multilabel_classification",
    num_labels=10
)
```

### 4. Create from Dataset
```python
from models.factory import create_bert_from_dataset

# Analyze dataset and create optimized model
model = create_bert_from_dataset(
    dataset_path="data/train.csv",
    auto_analyze=True
)
```

## Pooling Strategies

Each head can use different pooling strategies:

```python
from models.heads.base_head import PoolingType

head_config = {
    "head_type": HeadType.BINARY_CLASSIFICATION,
    "input_size": 768,
    "output_size": 2,
    "pooling_type": PoolingType.ATTENTION,  # Attention-based pooling
}

model = create_bert_with_head(head_config=head_config)
```

Available pooling types:
- `CLS`: Use [CLS] token (default for classification)
- `MEAN`: Average pooling (default for regression)
- `MAX`: Max pooling
- `ATTENTION`: Learned attention weights
- `WEIGHTED_MEAN`: Learned position weights
- `LAST`: Last non-padding token

## Model Persistence

### Saving Models
```python
# Save complete model (BERT + head)
model.save_pretrained("output/my_model")

# Directory structure:
# output/my_model/
# ├── bert/
# │   ├── config.json
# │   ├── model.safetensors
# │   └── bert_core_config.json
# ├── head/
# │   ├── config.json
# │   └── model.safetensors
# └── model_metadata.json
```

### Loading Models
```python
from models.bert import BertWithHead

# Load complete model
model = BertWithHead.from_pretrained("output/my_model")

# Access components
bert = model.get_bert()
head = model.get_head()
```

## Training Integration

The modular architecture works seamlessly with existing trainers:

```python
from training.trainer_v2 import EnhancedTrainer

# Create model
model = create_bert_for_task("binary_classification", num_labels=2)

# Create trainer
trainer = EnhancedTrainer(
    model=model,
    output_dir="output/experiment"
)

# Train
trainer.train(train_dataset, val_dataset)
```

## Freezing Strategies

Control which parts of the model to train:

```python
# Freeze all BERT layers
model.freeze_bert()

# Freeze specific layers
model.freeze_bert(num_layers=6)  # Freeze first 6 layers

# Unfreeze all
model.unfreeze_bert()

# Create with freezing
model = create_bert_with_head(
    head_type=HeadType.BINARY_CLASSIFICATION,
    freeze_bert=True,  # Freeze all BERT
    freeze_bert_layers=4  # Or freeze first 4 layers
)
```

## Custom Head Creation

To create a custom head:

```python
from models.heads.base_head import BaseKaggleHead, HeadConfig

class CustomHead(BaseKaggleHead):
    def _build_output_layer(self):
        # Define output layer
        self.output = nn.Linear(self.projection_output_size, self.config.output_size)
    
    def _build_loss_function(self):
        # Define loss
        self.loss_fn = nn.losses.mse_loss
    
    def forward(self, hidden_states, attention_mask=None, **kwargs):
        # Apply pooling
        pooled = self._apply_pooling(hidden_states, attention_mask)
        
        # Apply projection
        projected = self.projection(pooled)
        
        # Get output
        output = self.output(projected)
        
        return {"predictions": output}
    
    def compute_loss(self, predictions, targets, **kwargs):
        return self.loss_fn(predictions["predictions"], targets)
```

## Best Practices

1. **Start with pretrained models** when available:
   ```python
   model = create_bert_with_head(
       bert_name="bert-base-uncased",  # Will load pretrained
       head_type=HeadType.BINARY_CLASSIFICATION
   )
   ```

2. **Use appropriate pooling** for your task:
   - Classification: CLS token
   - Regression: Mean pooling
   - Ranking: Attention pooling
   - Time series: Last token

3. **Freeze BERT for small datasets**:
   ```python
   # Freeze all but last 2 layers
   model = create_bert_with_head(
       freeze_bert_layers=10  # For 12-layer BERT
   )
   ```

4. **Leverage competition-specific heads**:
   ```python
   # Auto-selects best head and config
   model = create_bert_for_competition(
       competition_type="ranking",
       num_labels=1
   )
   ```

## Migration from Old Architecture

If you have existing code using the old architecture:

```python
# Old approach
from models.modernbert import ModernBertModel
from models.classification import BinaryClassificationHead

bert = ModernBertModel(config)
head = BinaryClassificationHead(bert.config.hidden_size)

# New approach
from models.bert import create_bert_with_head

model = create_bert_with_head(
    bert_config=config,
    head_type=HeadType.BINARY_CLASSIFICATION
)
```

The new architecture provides:
- Cleaner separation of concerns
- Easier head swapping
- Better configuration management
- Unified save/load functionality
- Automatic compatibility checking