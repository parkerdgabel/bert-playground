"""
Preprocessing transforms for data preparation.
Includes tokenization, padding, and MLX-specific transforms.
"""

from typing import Dict, Any, List, Optional, Union, Tuple
import numpy as np
import mlx.core as mx
from loguru import logger

from .base_transforms import Transform
from embeddings.tokenizer_wrapper import TokenizerWrapper


class Tokenize(Transform):
    """Tokenize text data."""
    
    def __init__(
        self,
        tokenizer: Union[TokenizerWrapper, Any],
        text_field: str = "text",
        max_length: int = 256,
        padding: Union[bool, str] = "max_length",
        truncation: bool = True,
        return_attention_mask: bool = True,
        add_special_tokens: bool = True,
    ):
        """
        Initialize tokenizer transform.
        
        Args:
            tokenizer: Tokenizer instance
            text_field: Field containing text to tokenize
            max_length: Maximum sequence length
            padding: Padding strategy
            truncation: Whether to truncate
            return_attention_mask: Whether to return attention mask
            add_special_tokens: Whether to add special tokens
        """
        self.tokenizer = tokenizer
        self.text_field = text_field
        self.max_length = max_length
        self.padding = padding
        self.truncation = truncation
        self.return_attention_mask = return_attention_mask
        self.add_special_tokens = add_special_tokens
    
    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Tokenize text field."""
        if self.text_field not in data:
            raise KeyError(f"Text field '{self.text_field}' not found in data")
        
        text = data[self.text_field]
        
        # Handle TokenizerWrapper
        if hasattr(self.tokenizer, 'encode'):
            encoding = self.tokenizer.encode(
                text,
                max_length=self.max_length,
                padding=self.padding,
                truncation=self.truncation,
                return_attention_mask=self.return_attention_mask,
            )
        else:
            # Handle HuggingFace tokenizer
            encoding = self.tokenizer(
                text,
                max_length=self.max_length,
                padding=self.padding,
                truncation=self.truncation,
                return_attention_mask=self.return_attention_mask,
                add_special_tokens=self.add_special_tokens,
                return_tensors=None,
            )
        
        # Update data with tokenization results
        result = data.copy()
        result["input_ids"] = encoding["input_ids"]
        if "attention_mask" in encoding:
            result["attention_mask"] = encoding["attention_mask"]
        
        return result
    
    def __repr__(self) -> str:
        """String representation."""
        return f"Tokenize(max_length={self.max_length})"


class ToMLXArray(Transform):
    """Convert data to MLX arrays."""
    
    def __init__(
        self,
        fields: Optional[List[str]] = None,
        dtypes: Optional[Dict[str, mx.Dtype]] = None,
    ):
        """
        Initialize MLX array converter.
        
        Args:
            fields: Fields to convert (None = all numeric fields)
            dtypes: Specific dtypes for fields
        """
        self.fields = fields
        self.dtypes = dtypes or {}
        
        # Default dtypes
        self.default_dtypes = {
            "input_ids": mx.int32,
            "attention_mask": mx.float32,
            "label": mx.int32,
            "labels": mx.int32,
        }
    
    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Convert specified fields to MLX arrays."""
        result = {}
        
        fields_to_convert = self.fields or list(data.keys())
        
        for field, value in data.items():
            if field in fields_to_convert and isinstance(value, (list, tuple, np.ndarray, int, float)):
                # Determine dtype
                if field in self.dtypes:
                    dtype = self.dtypes[field]
                elif field in self.default_dtypes:
                    dtype = self.default_dtypes[field]
                else:
                    # Infer dtype
                    if isinstance(value, (int, np.integer)):
                        dtype = mx.int32
                    else:
                        dtype = mx.float32
                
                # Convert to MLX array
                result[field] = mx.array(value, dtype=dtype)
            else:
                # Keep as is
                result[field] = value
        
        return result
    
    def __repr__(self) -> str:
        """String representation."""
        return "ToMLXArray()"


class PadSequence(Transform):
    """Pad sequences to fixed length."""
    
    def __init__(
        self,
        fields: List[str],
        max_length: int,
        padding_value: Union[int, float] = 0,
        padding_side: str = "right",
        truncation: bool = True,
    ):
        """
        Initialize sequence padder.
        
        Args:
            fields: Fields to pad
            max_length: Maximum/target length
            padding_value: Value to use for padding
            padding_side: 'left' or 'right'
            truncation: Whether to truncate long sequences
        """
        self.fields = fields
        self.max_length = max_length
        self.padding_value = padding_value
        self.padding_side = padding_side
        self.truncation = truncation
    
    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Pad specified fields."""
        result = data.copy()
        
        for field in self.fields:
            if field not in result:
                continue
            
            sequence = result[field]
            if not isinstance(sequence, (list, np.ndarray)):
                continue
            
            current_length = len(sequence)
            
            if current_length > self.max_length and self.truncation:
                # Truncate
                result[field] = sequence[:self.max_length]
            elif current_length < self.max_length:
                # Pad
                padding_length = self.max_length - current_length
                padding = [self.padding_value] * padding_length
                
                if self.padding_side == "right":
                    result[field] = list(sequence) + padding
                else:
                    result[field] = padding + list(sequence)
            # else: already correct length
        
        return result
    
    def __repr__(self) -> str:
        """String representation."""
        return f"PadSequence(fields={self.fields}, max_length={self.max_length})"


class CreateAttentionMask(Transform):
    """Create attention mask from input IDs."""
    
    def __init__(
        self,
        input_field: str = "input_ids",
        output_field: str = "attention_mask",
        pad_token_id: int = 0,
    ):
        """
        Initialize attention mask creator.
        
        Args:
            input_field: Field containing input IDs
            output_field: Field to store attention mask
            pad_token_id: ID of padding token
        """
        self.input_field = input_field
        self.output_field = output_field
        self.pad_token_id = pad_token_id
    
    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Create attention mask."""
        if self.input_field not in data:
            return data
        
        result = data.copy()
        input_ids = data[self.input_field]
        
        if isinstance(input_ids, (list, np.ndarray)):
            # Create mask: 1 for real tokens, 0 for padding
            mask = [1 if token != self.pad_token_id else 0 for token in input_ids]
            result[self.output_field] = mask
        elif isinstance(input_ids, mx.array):
            # MLX array
            mask = (input_ids != self.pad_token_id).astype(mx.float32)
            result[self.output_field] = mask
        
        return result
    
    def __repr__(self) -> str:
        """String representation."""
        return f"CreateAttentionMask(pad_token={self.pad_token_id})"


class LabelEncode(Transform):
    """Encode labels for classification."""
    
    def __init__(
        self,
        label_field: str = "label",
        label_map: Optional[Dict[Any, int]] = None,
        num_classes: Optional[int] = None,
        multi_label: bool = False,
        label_smoothing: float = 0.0,
    ):
        """
        Initialize label encoder.
        
        Args:
            label_field: Field containing labels
            label_map: Mapping from labels to indices
            num_classes: Number of classes (for one-hot encoding)
            multi_label: Whether this is multi-label classification
            label_smoothing: Label smoothing factor
        """
        self.label_field = label_field
        self.label_map = label_map
        self.num_classes = num_classes
        self.multi_label = multi_label
        self.label_smoothing = label_smoothing
    
    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Encode labels."""
        if self.label_field not in data:
            return data
        
        result = data.copy()
        label = data[self.label_field]
        
        # Apply label map if provided
        if self.label_map:
            if self.multi_label and isinstance(label, list):
                label = [self.label_map.get(l, l) for l in label]
            else:
                label = self.label_map.get(label, label)
        
        # Apply one-hot encoding if num_classes is specified
        if self.num_classes and not self.multi_label:
            one_hot = np.zeros(self.num_classes, dtype=np.float32)
            one_hot[label] = 1.0
            
            # Apply label smoothing
            if self.label_smoothing > 0:
                one_hot = one_hot * (1 - self.label_smoothing) + self.label_smoothing / self.num_classes
            
            label = one_hot
        
        result[self.label_field] = label
        return result
    
    def __repr__(self) -> str:
        """String representation."""
        return f"LabelEncode(num_classes={self.num_classes}, multi_label={self.multi_label})"


class FeatureExtractor(Transform):
    """Extract features using a pre-trained model."""
    
    def __init__(
        self,
        model: Any,
        input_field: str = "input_ids",
        output_field: str = "features",
        layer_index: Optional[int] = None,
        pooling: str = "mean",
    ):
        """
        Initialize feature extractor.
        
        Args:
            model: Pre-trained model
            input_field: Field containing input
            output_field: Field to store features
            layer_index: Which layer to extract from
            pooling: Pooling strategy
        """
        self.model = model
        self.input_field = input_field
        self.output_field = output_field
        self.layer_index = layer_index
        self.pooling = pooling
    
    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract features."""
        if self.input_field not in data:
            return data
        
        result = data.copy()
        
        # Get model outputs
        inputs = data[self.input_field]
        attention_mask = data.get("attention_mask", None)
        
        # Run model
        with mx.no_grad():
            outputs = self.model(inputs, attention_mask=attention_mask)
            
            # Extract specific layer if requested
            if self.layer_index is not None and hasattr(outputs, "hidden_states"):
                features = outputs.hidden_states[self.layer_index]
            else:
                features = outputs.last_hidden_state
            
            # Apply pooling
            if self.pooling == "mean":
                if attention_mask is not None:
                    mask = attention_mask.unsqueeze(-1).expand(features.size())
                    features = (features * mask).sum(1) / mask.sum(1)
                else:
                    features = features.mean(1)
            elif self.pooling == "max":
                features = features.max(1)[0]
            elif self.pooling == "cls":
                features = features[:, 0]
        
        result[self.output_field] = features
        return result
    
    def __repr__(self) -> str:
        """String representation."""
        return f"FeatureExtractor(pooling={self.pooling})"


class DataAugmentation(Transform):
    """Apply data augmentation techniques."""
    
    def __init__(
        self,
        augmentations: List[Transform],
        probability: float = 0.5,
        num_augmentations: int = 1,
    ):
        """
        Initialize data augmentation.
        
        Args:
            augmentations: List of augmentation transforms
            probability: Probability of applying augmentation
            num_augmentations: Number of augmentations to apply
        """
        self.augmentations = augmentations
        self.probability = probability
        self.num_augmentations = num_augmentations
    
    def __call__(self, data: Dict[str, Any]) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
        """Apply augmentations."""
        if np.random.random() > self.probability:
            return data
        
        if self.num_augmentations == 1:
            # Apply single augmentation
            augmentation = np.random.choice(self.augmentations)
            return augmentation(data)
        else:
            # Apply multiple augmentations
            results = [data]  # Include original
            
            for _ in range(self.num_augmentations):
                augmentation = np.random.choice(self.augmentations)
                augmented = augmentation(data.copy())
                results.append(augmented)
            
            return results
    
    def __repr__(self) -> str:
        """String representation."""
        return f"DataAugmentation(n_augs={len(self.augmentations)}, p={self.probability})"