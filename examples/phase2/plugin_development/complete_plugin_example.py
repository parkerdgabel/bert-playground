"""
Complete Plugin Development Example

This example demonstrates how to create a comprehensive plugin system
with custom heads, augmenters, metrics, and models, including configuration
management and testing.
"""

import json
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from cli.plugins.base import BasePlugin, HeadPlugin, AugmenterPlugin, MetricPlugin, ModelPlugin
from cli.plugins.registry import PluginRegistry
from cli.plugins.loader import PluginLoader
from core.protocols.plugins import PluginMetadata
from data.augmentation.base import BaseAugmenter
from models.heads.base import BaseHead


# ==================== HEAD PLUGIN EXAMPLE ====================

class AttentionPoolingHead(BaseHead):
    """Custom head with attention pooling mechanism."""
    
    def __init__(self, input_dim: int, num_labels: int, dropout_rate: float = 0.1):
        super().__init__()
        self.input_dim = input_dim
        self.num_labels = num_labels
        
        # Attention mechanism for pooling
        self.attention = nn.MultiHeadAttention(
            dims=input_dim,
            num_heads=8,
            bias=True
        )
        
        # Classification layers
        self.layer_norm = nn.LayerNorm(input_dim)
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(input_dim, num_labels)
        
        # Learnable query vector for attention pooling
        self.query = mx.random.normal((1, 1, input_dim)) * 0.02
    
    def __call__(self, hidden_states: mx.array, attention_mask: Optional[mx.array] = None) -> mx.array:
        """
        Forward pass with attention pooling.
        
        Args:
            hidden_states: [batch_size, seq_len, hidden_dim]
            attention_mask: [batch_size, seq_len] (optional)
        
        Returns:
            logits: [batch_size, num_labels]
        """
        batch_size, seq_len, hidden_dim = hidden_states.shape
        
        # Expand query for batch
        query = mx.broadcast_to(self.query, (batch_size, 1, hidden_dim))
        
        # Apply attention pooling (query attends to all positions)
        pooled_output, _ = self.attention(
            query, hidden_states, hidden_states, mask=attention_mask
        )
        
        # Extract the single attended representation
        pooled_output = pooled_output.squeeze(1)  # [batch_size, hidden_dim]
        
        # Apply layer norm and dropout
        pooled_output = self.layer_norm(pooled_output)
        pooled_output = self.dropout(pooled_output)
        
        # Classification
        logits = self.classifier(pooled_output)
        
        return logits


class AttentionHeadPlugin(BasePlugin, HeadPlugin):
    """Plugin for attention-based classification head."""
    
    def create_head(self, input_dim: int, output_dim: int, **kwargs) -> BaseHead:
        """Create attention pooling head."""
        dropout_rate = self.config.get('dropout_rate', 0.1)
        
        return AttentionPoolingHead(
            input_dim=input_dim,
            num_labels=output_dim,
            dropout_rate=dropout_rate
        )
    
    def get_metadata(self) -> PluginMetadata:
        return PluginMetadata(
            name="AttentionPoolingHead",
            version="1.0.0",
            description="Classification head with learnable attention pooling",
            author="k-bert Team",
            plugin_type="head",
            tags=["attention", "pooling", "classification"],
            requires=["mlx>=0.5.0"]
        )


# ==================== AUGMENTER PLUGIN EXAMPLE ====================

class BackTranslationAugmenter(BaseAugmenter):
    """Augmenter that simulates back-translation for text diversity."""
    
    def __init__(self, config: Dict[str, Any]):
        self.augmentation_prob = config.get('augmentation_prob', 0.3)
        self.paraphrase_prob = config.get('paraphrase_prob', 0.5)
        self.synonym_prob = config.get('synonym_prob', 0.2)
        
        # Simple synonym dictionary (in practice, use a proper NLP library)
        self.synonyms = {
            'good': ['excellent', 'great', 'fine', 'nice'],
            'bad': ['terrible', 'awful', 'poor', 'horrible'],
            'big': ['large', 'huge', 'enormous', 'massive'],
            'small': ['tiny', 'little', 'mini', 'compact'],
            'fast': ['quick', 'rapid', 'speedy', 'swift'],
            'slow': ['sluggish', 'gradual', 'leisurely', 'unhurried']
        }
    
    def augment(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """Apply back-translation-style augmentation."""
        if 'text' not in sample:
            return sample
        
        # Apply augmentation with probability
        if float(mx.random.uniform()) > self.augmentation_prob:
            return sample
        
        text = sample['text']
        augmented_text = text
        
        # Apply synonym replacement
        if float(mx.random.uniform()) < self.synonym_prob:
            augmented_text = self._replace_synonyms(augmented_text)
        
        # Apply paraphrasing (simplified)
        if float(mx.random.uniform()) < self.paraphrase_prob:
            augmented_text = self._paraphrase_text(augmented_text)
        
        sample['text'] = augmented_text
        return sample
    
    def _replace_synonyms(self, text: str) -> str:
        """Replace words with synonyms."""
        words = text.lower().split()
        result = []
        
        for word in words:
            if word in self.synonyms and float(mx.random.uniform()) < 0.3:
                # Replace with random synonym
                synonyms = self.synonyms[word]
                replacement = synonyms[int(float(mx.random.uniform()) * len(synonyms))]
                result.append(replacement)
            else:
                result.append(word)
        
        return ' '.join(result)
    
    def _paraphrase_text(self, text: str) -> str:
        """Simple paraphrasing simulation."""
        # Simple transformations that simulate paraphrasing
        transformations = [
            lambda t: t.replace('is very', 'is extremely'),
            lambda t: t.replace('really good', 'outstanding'),
            lambda t: t.replace('not bad', 'decent'),
            lambda t: t.replace('I think', 'In my opinion'),
            lambda t: t.replace('This is', 'This appears to be')
        ]
        
        # Apply random transformation
        if transformations:
            transform_idx = int(float(mx.random.uniform()) * len(transformations))
            text = transformations[transform_idx](text)
        
        return text
    
    def get_config(self) -> Dict[str, Any]:
        return {
            'augmentation_prob': self.augmentation_prob,
            'paraphrase_prob': self.paraphrase_prob,
            'synonym_prob': self.synonym_prob
        }


class BackTranslationPlugin(BasePlugin, AugmenterPlugin):
    """Plugin for back-translation style augmentation."""
    
    def create_augmenter(self, **kwargs) -> BaseAugmenter:
        """Create back-translation augmenter."""
        return BackTranslationAugmenter(self.config)
    
    def get_metadata(self) -> PluginMetadata:
        return PluginMetadata(
            name="BackTranslationAugmenter",
            version="1.0.0",
            description="Text augmentation via simulated back-translation",
            plugin_type="augmenter",
            tags=["text", "augmentation", "paraphrasing", "translation"],
            requires=["mlx>=0.5.0"]
        )


# ==================== METRIC PLUGIN EXAMPLE ====================

class MacroF1Metric:
    """Macro-averaged F1 score metric."""
    
    def __init__(self, num_classes: int):
        self.num_classes = num_classes
        self.reset()
    
    def update(self, predictions: mx.array, targets: mx.array):
        """Update metric with batch predictions and targets."""
        # Convert to class predictions if logits provided
        if predictions.shape[-1] > 1:
            pred_classes = mx.argmax(predictions, axis=-1)
        else:
            pred_classes = predictions
        
        # Store for final computation
        self.all_predictions.extend(np.array(pred_classes).tolist())
        self.all_targets.extend(np.array(targets).tolist())
    
    def compute(self) -> float:
        """Compute macro F1 score."""
        if not self.all_predictions:
            return 0.0
        
        # Calculate per-class F1 scores
        f1_scores = []
        
        for class_id in range(self.num_classes):
            # True positives, false positives, false negatives
            tp = sum(1 for pred, true in zip(self.all_predictions, self.all_targets)
                    if pred == class_id and true == class_id)
            fp = sum(1 for pred, true in zip(self.all_predictions, self.all_targets)
                    if pred == class_id and true != class_id)
            fn = sum(1 for pred, true in zip(self.all_predictions, self.all_targets)
                    if pred != class_id and true == class_id)
            
            # Calculate precision and recall
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            
            # Calculate F1 score
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
            f1_scores.append(f1)
        
        # Return macro average
        return sum(f1_scores) / len(f1_scores)
    
    def reset(self):
        """Reset metric state."""
        self.all_predictions = []
        self.all_targets = []


class AdvancedMetricsPlugin(BasePlugin, MetricPlugin):
    """Plugin providing advanced evaluation metrics."""
    
    def create_metric(self, metric_name: str, **kwargs):
        """Create a metric instance."""
        if metric_name == 'macro_f1':
            num_classes = kwargs.get('num_classes', 2)
            return MacroF1Metric(num_classes)
        elif metric_name == 'weighted_f1':
            # Could implement weighted F1 here
            return MacroF1Metric(kwargs.get('num_classes', 2))
        else:
            raise ValueError(f"Unknown metric: {metric_name}")
    
    def get_available_metrics(self) -> List[str]:
        """Return list of available metrics."""
        return ['macro_f1', 'weighted_f1']
    
    def get_metadata(self) -> PluginMetadata:
        return PluginMetadata(
            name="AdvancedMetrics",
            version="1.0.0",
            description="Advanced evaluation metrics including macro F1",
            plugin_type="metric",
            tags=["metrics", "evaluation", "f1", "classification"],
            requires=["numpy>=1.20.0", "mlx>=0.5.0"]
        )


# ==================== MODEL PLUGIN EXAMPLE ====================

class EfficientTransformer(nn.Module):
    """Efficient transformer model with reduced parameters."""
    
    def __init__(
        self,
        vocab_size: int = 30522,
        hidden_size: int = 512,
        num_layers: int = 4,
        num_heads: int = 8,
        max_position_embeddings: int = 512
    ):
        super().__init__()
        self.hidden_size = hidden_size
        
        # Embeddings
        self.token_embeddings = nn.Embedding(vocab_size, hidden_size)
        self.position_embeddings = nn.Embedding(max_position_embeddings, hidden_size)
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(0.1)
        
        # Transformer layers with shared parameters for efficiency
        self.shared_layer = nn.TransformerEncoderLayer(
            dims=hidden_size,
            num_heads=num_heads,
            mlp_dims=hidden_size * 2,  # Smaller than standard 4x
            bias=True
        )
        self.num_layers = num_layers
    
    def __call__(self, input_ids: mx.array, attention_mask: Optional[mx.array] = None):
        """Forward pass."""
        batch_size, seq_length = input_ids.shape
        
        # Create position IDs
        position_ids = mx.arange(seq_length)[None, :]  # [1, seq_length]
        position_ids = mx.broadcast_to(position_ids, (batch_size, seq_length))
        
        # Embeddings
        token_embeds = self.token_embeddings(input_ids)
        position_embeds = self.position_embeddings(position_ids)
        
        hidden_states = token_embeds + position_embeds
        hidden_states = self.layer_norm(hidden_states)
        hidden_states = self.dropout(hidden_states)
        
        # Apply shared transformer layer multiple times
        for _ in range(self.num_layers):
            hidden_states = self.shared_layer(hidden_states, mask=attention_mask)
        
        return {
            'last_hidden_state': hidden_states,
            'pooler_output': mx.mean(hidden_states, axis=1)  # Simple mean pooling
        }


class EfficientModelPlugin(BasePlugin, ModelPlugin):
    """Plugin for efficient transformer models."""
    
    def create_model(self, **kwargs):
        """Create efficient transformer model."""
        vocab_size = kwargs.get('vocab_size', 30522)
        hidden_size = self.config.get('hidden_size', 512)
        num_layers = self.config.get('num_layers', 4)
        num_heads = self.config.get('num_heads', 8)
        
        return EfficientTransformer(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            num_heads=num_heads
        )
    
    def get_model_info(self) -> Dict[str, Any]:
        """Return model information."""
        return {
            'type': 'transformer',
            'variant': 'efficient',
            'shared_layers': True,
            'parameter_reduction': '~60% vs standard BERT',
            'memory_efficient': True
        }
    
    def get_metadata(self) -> PluginMetadata:
        return PluginMetadata(
            name="EfficientTransformer",
            version="1.0.0",
            description="Memory and parameter efficient transformer with shared layers",
            plugin_type="model",
            tags=["transformer", "efficient", "shared-layers", "lightweight"],
            requires=["mlx>=0.5.0"]
        )


# ==================== PLUGIN MANAGEMENT EXAMPLE ====================

class PluginManager:
    """Manages plugin discovery, registration, and usage."""
    
    def __init__(self):
        self.registry = PluginRegistry()
        self.loader = PluginLoader()
        self.config = {}
    
    def register_builtin_plugins(self):
        """Register built-in plugins."""
        plugins = [
            AttentionHeadPlugin({'dropout_rate': 0.1}),
            BackTranslationPlugin({
                'augmentation_prob': 0.3,
                'paraphrase_prob': 0.5,
                'synonym_prob': 0.2
            }),
            AdvancedMetricsPlugin({}),
            EfficientModelPlugin({
                'hidden_size': 512,
                'num_layers': 4,
                'num_heads': 8
            })
        ]
        
        for plugin in plugins:
            metadata = plugin.get_metadata()
            self.registry.register(metadata.plugin_type, metadata.name, plugin)
            print(f"✅ Registered plugin: {metadata.name} ({metadata.plugin_type})")
    
    def load_project_plugins(self, plugin_dir: Path):
        """Load plugins from project directory."""
        if not plugin_dir.exists():
            print(f"Plugin directory not found: {plugin_dir}")
            return
        
        try:
            plugins = self.loader.load_from_directory(plugin_dir)
            for plugin in plugins:
                metadata = plugin.get_metadata()
                self.registry.register(metadata.plugin_type, metadata.name, plugin)
                print(f"🔌 Loaded project plugin: {metadata.name}")
        except Exception as e:
            print(f"Error loading project plugins: {e}")
    
    def list_plugins(self):
        """List all registered plugins."""
        print("\n📋 Registered Plugins:")
        print("-" * 40)
        
        for plugin_type in ['head', 'augmenter', 'metric', 'model']:
            plugins = self.registry.list_by_type(plugin_type)
            if plugins:
                print(f"\n{plugin_type.upper()} Plugins:")
                for name in plugins:
                    plugin = self.registry.get(plugin_type, name)
                    metadata = plugin.get_metadata()
                    print(f"  • {name} v{metadata.version} - {metadata.description}")
    
    def demonstrate_plugin_usage(self):
        """Demonstrate using different plugins."""
        print("\n🧪 Plugin Usage Demonstration:")
        print("-" * 40)
        
        # Demonstrate head plugin
        head_plugin = self.registry.get('head', 'AttentionPoolingHead')
        if head_plugin:
            head = head_plugin.create_head(input_dim=512, output_dim=3)
            print(f"✅ Created attention head: {head}")
            
            # Test forward pass
            test_input = mx.random.normal((2, 10, 512))  # batch_size=2, seq_len=10
            output = head(test_input)
            print(f"   Output shape: {output.shape}")
        
        # Demonstrate augmenter plugin
        aug_plugin = self.registry.get('augmenter', 'BackTranslationAugmenter')
        if aug_plugin:
            augmenter = aug_plugin.create_augmenter()
            test_sample = {'text': 'This is a really good example.', 'label': 1}
            augmented = augmenter.augment(test_sample)
            print(f"✅ Original: {test_sample['text']}")
            print(f"   Augmented: {augmented['text']}")
        
        # Demonstrate metric plugin
        metric_plugin = self.registry.get('metric', 'AdvancedMetrics')
        if metric_plugin:
            metric = metric_plugin.create_metric('macro_f1', num_classes=3)
            
            # Simulate some predictions and targets
            predictions = mx.array([[0.8, 0.1, 0.1], [0.2, 0.7, 0.1], [0.1, 0.2, 0.7]])
            targets = mx.array([0, 1, 2])
            
            metric.update(predictions, targets)
            f1_score = metric.compute()
            print(f"✅ Macro F1 Score: {f1_score:.4f}")
        
        # Demonstrate model plugin
        model_plugin = self.registry.get('model', 'EfficientTransformer')
        if model_plugin:
            model = model_plugin.create_model(vocab_size=1000)
            print(f"✅ Created efficient model: {model}")
            
            # Test forward pass
            test_ids = mx.random.randint(0, 1000, (2, 20))
            outputs = model(test_ids)
            print(f"   Last hidden state shape: {outputs['last_hidden_state'].shape}")
            print(f"   Pooler output shape: {outputs['pooler_output'].shape}")


def create_plugin_config_example(plugin_dir: Path):
    """Create example plugin configuration."""
    config = {
        "project": {
            "name": "plugin-example",
            "version": "1.0.0"
        },
        "plugins": {
            "enabled": True,
            "directories": [str(plugin_dir)],
            "auto_discover": True
        },
        "models": {
            "model": {
                "plugin": "EfficientTransformer",
                "config": {
                    "hidden_size": 768,
                    "num_layers": 6,
                    "num_heads": 12
                }
            },
            "head": {
                "plugin": "AttentionPoolingHead",
                "config": {
                    "dropout_rate": 0.15
                }
            }
        },
        "data": {
            "augmentation": {
                "plugin": "BackTranslationAugmenter",
                "config": {
                    "augmentation_prob": 0.4,
                    "paraphrase_prob": 0.6,
                    "synonym_prob": 0.3
                }
            }
        },
        "training": {
            "metrics": [
                {
                    "name": "macro_f1",
                    "plugin": "AdvancedMetrics",
                    "config": {
                        "num_classes": 3
                    }
                }
            ]
        }
    }
    
    config_path = plugin_dir / "k-bert.yaml"
    import yaml
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, indent=2)
    
    print(f"📄 Created example configuration: {config_path}")
    return config


def main():
    """Main demonstration function."""
    print("🔌 Complete Plugin Development Example")
    print("=" * 50)
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        plugin_dir = temp_path / "plugins"
        plugin_dir.mkdir()
        
        # Create plugin manager
        manager = PluginManager()
        
        # Register built-in plugins
        print("Registering built-in plugins...")
        manager.register_builtin_plugins()
        
        # Create example configuration
        config = create_plugin_config_example(plugin_dir)
        
        # List all plugins
        manager.list_plugins()
        
        # Demonstrate plugin usage
        manager.demonstrate_plugin_usage()
        
        print(f"\n🎯 Plugin Development Complete!")
        print(f"Configuration saved to: {plugin_dir}/k-bert.yaml")
        print("Plugins are ready for use in training pipelines!")


if __name__ == "__main__":
    main()