import mlx.core as mx
import mlx.nn as nn
from typing import Optional
from loguru import logger


class BinaryClassificationHead(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: Optional[int] = None,
        dropout_prob: float = 0.1
    ):
        super().__init__()
        
        if hidden_dim is None:
            # Simple linear classifier
            self.classifier = nn.Sequential(
                nn.Dropout(dropout_prob),
                nn.Linear(input_dim, 2)
            )
        else:
            # Two-layer classifier with hidden dimension
            self.classifier = nn.Sequential(
                nn.Dropout(dropout_prob),
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout_prob),
                nn.Linear(hidden_dim, 2)
            )
    
    def __call__(self, pooled_output: mx.array) -> mx.array:
        return self.classifier(pooled_output)


class TitanicClassifier(nn.Module):
    def __init__(
        self,
        bert_model: nn.Module,
        hidden_dim: Optional[int] = None,
        dropout_prob: float = 0.1,
        freeze_bert: bool = False
    ):
        super().__init__()
        self.bert = bert_model
        
        # Check if the model already has a classifier (CNN hybrid case)
        self.has_built_in_classifier = hasattr(bert_model, 'classifier')
        
        if not self.has_built_in_classifier:
            # Get the actual output dimension from the model
            output_dim = getattr(bert_model, 'output_hidden_size', bert_model.config.hidden_size)
            
            self.classifier = BinaryClassificationHead(
                input_dim=output_dim,
                hidden_dim=hidden_dim,
                dropout_prob=dropout_prob
            )
        else:
            # CNN hybrid model already has classifier, no need to add another
            self.classifier = None
        
        # Optionally freeze BERT parameters
        if freeze_bert:
            self._freeze_bert()
    
    def _freeze_bert(self):
        # In MLX, we would handle this differently than PyTorch
        # For now, this is a placeholder
        print("Note: Parameter freezing in MLX requires custom gradient handling")
    
    def __call__(
        self,
        input_ids: mx.array,
        attention_mask: Optional[mx.array] = None,
        labels: Optional[mx.array] = None
    ) -> dict:
        # Get BERT outputs
        bert_outputs = self.bert(input_ids, attention_mask, labels=labels)
        
        if self.has_built_in_classifier:
            # CNN hybrid model already produces logits and loss
            return bert_outputs
        else:
            # Standard BERT model - need to add classification
            pooled_output = bert_outputs['pooled_output']
            
            # Classification
            logits = self.classifier(pooled_output)
            
            outputs = {'logits': logits}
            
            # Calculate loss if labels provided
            if labels is not None:
                # Ensure labels have the correct shape
                if labels.ndim == 0:  # Scalar label
                    labels = labels.reshape(1)
                elif labels.ndim == 2:  # Already has batch dimension
                    labels = labels.squeeze()
                
                # Ensure we have a batch dimension
                if logits.shape[0] != labels.shape[0]:
                    logger.warning(f"Shape mismatch: logits {logits.shape} vs labels {labels.shape}")
                
                loss = mx.mean(
                    nn.losses.cross_entropy(
                        logits,
                        labels,
                        reduction='none'
                    )
                )
                outputs['loss'] = loss
            
            return outputs
    
    def predict(self, input_ids: mx.array, attention_mask: Optional[mx.array] = None) -> mx.array:
        outputs = self.forward(input_ids, attention_mask)
        predictions = mx.argmax(outputs['logits'], axis=-1)
        return predictions
    
    def predict_proba(self, input_ids: mx.array, attention_mask: Optional[mx.array] = None) -> mx.array:
        outputs = self.forward(input_ids, attention_mask)
        probabilities = mx.softmax(outputs['logits'], axis=-1)
        return probabilities