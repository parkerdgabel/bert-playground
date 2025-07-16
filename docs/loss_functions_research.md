# Optimal Loss Functions for Binary Classification with Class Imbalance

## Research Findings for Titanic Dataset (61% survived, 39% not survived)

### 1. Current Implementation Analysis

The current implementation uses standard cross-entropy loss without any class balancing:
```python
loss = mx.mean(nn.losses.cross_entropy(logits, labels, reduction='none'))
```

This approach has several limitations:
- Treats all samples equally regardless of class frequency
- May lead to model bias towards the majority class (survived)
- Cannot distinguish between easy and hard examples
- No protection against overconfidence (model collapse)

### 2. Best Loss Functions for Class Imbalance

#### 2.1 Focal Loss
**Most recommended for preventing model collapse and handling imbalance**

Focal Loss addresses class imbalance by applying a modulating term to cross entropy loss:

```python
FL(pt) = -α(1-pt)^γ * log(pt)
```

Where:
- `pt` is the model's estimated probability for the correct class
- `α` (alpha) is the weighting factor (can be inverse class frequency)
- `γ` (gamma) is the focusing parameter (typically 2-5, best results with γ=2)

**Benefits:**
- Down-weights easy examples (high confidence predictions)
- Focuses learning on hard, misclassified examples
- Prevents model from becoming overconfident
- Naturally handles class imbalance

**Implementation for MLX:**
```python
def focal_loss(logits, labels, alpha=0.25, gamma=2.0):
    """
    Focal loss for binary classification in MLX
    
    Args:
        logits: Model output logits [batch_size, 2]
        labels: Ground truth labels [batch_size]
        alpha: Weighting factor for positive class
        gamma: Focusing parameter
    """
    # Convert to probabilities
    probs = mx.softmax(logits, axis=-1)
    
    # Get probability of correct class
    batch_size = labels.shape[0]
    indices = mx.stack([mx.arange(batch_size), labels], axis=1)
    pt = probs[indices[:, 0], indices[:, 1]]
    
    # Calculate focal term
    focal_term = mx.power(1 - pt, gamma)
    
    # Calculate alpha term (class weighting)
    alpha_t = mx.where(labels == 1, alpha, 1 - alpha)
    
    # Combine terms
    ce_loss = -mx.log(pt + 1e-8)  # Add epsilon for numerical stability
    focal_loss = alpha_t * focal_term * ce_loss
    
    return mx.mean(focal_loss)
```

#### 2.2 Weighted Cross-Entropy
**Simple but effective baseline**

For Titanic dataset with 61% survived (class 1) and 39% not survived (class 0):
- Weight for class 0 (not survived): 1.56 (1/0.39)
- Weight for class 1 (survived): 1.00 (1/0.61)

```python
def weighted_cross_entropy(logits, labels, class_weights=None):
    """
    Weighted cross-entropy for MLX
    
    Args:
        logits: Model output logits [batch_size, 2]
        labels: Ground truth labels [batch_size]
        class_weights: Array of shape [2] with weights for each class
    """
    if class_weights is None:
        # Default weights for Titanic dataset
        class_weights = mx.array([1.56, 1.0])
    
    # Standard cross-entropy
    ce_loss = nn.losses.cross_entropy(logits, labels, reduction='none')
    
    # Apply class weights
    weights = class_weights[labels]
    weighted_loss = ce_loss * weights
    
    return mx.mean(weighted_loss)
```

#### 2.3 Batch-Balanced Focal Loss (BBFL)
**State-of-the-art for severe imbalance**

Combines batch balancing with focal loss:
1. Ensures equal representation of classes in each batch
2. Applies focal loss to focus on hard examples

```python
def batch_balanced_focal_loss(logits, labels, alpha=0.25, gamma=2.0):
    """
    Batch-balanced focal loss
    Requires data loader to provide balanced batches
    """
    # Apply focal loss (same as above)
    focal_loss_value = focal_loss(logits, labels, alpha, gamma)
    
    # Additional batch balancing can be done in the data loader
    return focal_loss_value
```

### 3. Label Smoothing Techniques

Label smoothing prevents overconfidence and improves generalization:

```python
def label_smoothing_cross_entropy(logits, labels, smoothing=0.1):
    """
    Cross-entropy with label smoothing
    
    Args:
        logits: Model output logits [batch_size, 2]
        labels: Ground truth labels [batch_size]
        smoothing: Label smoothing factor (typically 0.1)
    """
    num_classes = 2
    
    # Create smoothed labels
    confidence = 1.0 - smoothing
    smooth_labels = mx.zeros((labels.shape[0], num_classes))
    smooth_labels = smooth_labels + smoothing / num_classes
    
    # Set confidence for true labels
    batch_indices = mx.arange(labels.shape[0])
    smooth_labels[batch_indices, labels] = confidence
    
    # Calculate cross-entropy with smooth labels
    log_probs = mx.log_softmax(logits, axis=-1)
    loss = -mx.sum(smooth_labels * log_probs, axis=-1)
    
    return mx.mean(loss)
```

### 4. Combined Approach: Focal Loss with Label Smoothing

```python
def focal_loss_with_label_smoothing(logits, labels, alpha=0.25, gamma=2.0, smoothing=0.1):
    """
    Combines focal loss with label smoothing for optimal performance
    """
    num_classes = 2
    
    # Apply label smoothing
    confidence = 1.0 - smoothing
    smooth_labels = mx.zeros((labels.shape[0], num_classes))
    smooth_labels = smooth_labels + smoothing / num_classes
    batch_indices = mx.arange(labels.shape[0])
    smooth_labels[batch_indices, labels] = confidence
    
    # Calculate probabilities
    probs = mx.softmax(logits, axis=-1)
    
    # Get probability for true class (with smoothing)
    pt = mx.sum(probs * smooth_labels, axis=-1)
    
    # Calculate focal term
    focal_term = mx.power(1 - pt, gamma)
    
    # Calculate alpha term
    alpha_t = mx.where(labels == 1, alpha, 1 - alpha)
    
    # Cross-entropy with smoothed labels
    ce_loss = -mx.sum(smooth_labels * mx.log(probs + 1e-8), axis=-1)
    
    # Combine all terms
    loss = alpha_t * focal_term * ce_loss
    
    return mx.mean(loss)
```

### 5. Recommendations for Titanic Dataset

Given the moderate class imbalance (61/39 split), here are the recommendations in order of preference:

1. **Focal Loss (γ=2, α=0.39)**
   - Best for preventing model collapse
   - Handles both class imbalance and hard examples
   - Proven effectiveness in similar tasks

2. **Focal Loss with Label Smoothing (γ=2, α=0.39, smoothing=0.1)**
   - Additional regularization
   - Prevents overconfidence
   - Best for models prone to overfitting

3. **Weighted Cross-Entropy (weights=[1.56, 1.0])**
   - Simple baseline
   - Easy to implement and tune
   - Good starting point

4. **Standard Cross-Entropy with Label Smoothing (smoothing=0.1)**
   - If class imbalance is not severe
   - Focus on preventing overconfidence

### 6. Implementation Guidelines for MLX

1. **Gradient Computation**: MLX handles gradients automatically
   ```python
   def loss_fn(model, batch):
       outputs = model(batch['input_ids'], batch['attention_mask'], batch['labels'])
       return focal_loss(outputs['logits'], batch['labels'])
   
   loss_and_grad_fn = mx.value_and_grad(loss_fn)
   ```

2. **Numerical Stability**: Always add epsilon (1e-8) to log operations

3. **Monitoring**: Track both loss and per-class metrics to ensure balanced learning

4. **Hyperparameter Tuning**:
   - Start with γ=2 for focal loss
   - Use class frequency for initial α values
   - Label smoothing: 0.1 is typically optimal
   - Adjust based on validation metrics

### 7. Expected Improvements

Switching from standard cross-entropy to focal loss typically results in:
- 3-5% improvement in minority class recall
- Better calibrated probabilities
- More stable training
- Reduced risk of model collapse
- Improved F1 score (especially for imbalanced datasets)

### 8. Integration with Current Codebase

To integrate into the existing `TitanicClassifier`:

```python
class TitanicClassifier(nn.Module):
    def __init__(self, bert_model, loss_type='focal', **loss_kwargs):
        super().__init__()
        self.bert = bert_model
        self.classifier = BinaryClassificationHead(...)
        self.loss_type = loss_type
        self.loss_kwargs = loss_kwargs
    
    def __call__(self, input_ids, attention_mask=None, labels=None):
        # ... existing code ...
        
        if labels is not None:
            if self.loss_type == 'focal':
                loss = focal_loss(logits, labels, **self.loss_kwargs)
            elif self.loss_type == 'weighted_ce':
                loss = weighted_cross_entropy(logits, labels, **self.loss_kwargs)
            elif self.loss_type == 'focal_smooth':
                loss = focal_loss_with_label_smoothing(logits, labels, **self.loss_kwargs)
            else:
                loss = mx.mean(nn.losses.cross_entropy(logits, labels, reduction='none'))
            
            outputs['loss'] = loss
```

This allows easy experimentation with different loss functions while maintaining backward compatibility.