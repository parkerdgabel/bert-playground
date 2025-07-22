# Plugin Development Guide

## Overview

Phase 2's enhanced plugin system provides type-safe, event-driven plugin development with dependency injection, lifecycle management, and hot-reloading capabilities. This guide covers everything you need to know to develop robust plugins for k-bert.

## Plugin Architecture

### Core Components

```python
from bert_playground.core.ports import (
    PluginPort,
    HeadPort, 
    AugmenterPort,
    FeaturePort,
    CallbackPort
)
from bert_playground.core.events import EventBus
from bert_playground.core.di import Container
```

### Plugin Types

#### 1. Model Head Plugins
Custom task-specific output layers:
```python
from bert_playground.core.ports import HeadPort
import mlx.core as mx
import mlx.nn as nn

class CustomClassificationHead(HeadPort):
    def __init__(self, config: HeadConfig, event_bus: EventBus):
        super().__init__(config, event_bus)
        self.hidden_size = config.hidden_size
        self.num_labels = config.num_labels
        self.dropout_rate = config.dropout_rate
        
        # Build layers
        self.dropout = nn.Dropout(self.dropout_rate)
        self.classifier = nn.Linear(self.hidden_size, self.num_labels)
        self.activation = nn.ReLU()
    
    async def forward(self, hidden_states: mx.array, **kwargs) -> mx.array:
        await self.event_bus.emit("head.forward_start", {
            "head_type": self.__class__.__name__,
            "input_shape": hidden_states.shape
        })
        
        # Apply dropout and classification
        x = self.dropout(hidden_states)
        x = self.activation(x)
        logits = self.classifier(x)
        
        await self.event_bus.emit("head.forward_complete", {
            "head_type": self.__class__.__name__,
            "output_shape": logits.shape
        })
        
        return logits
    
    def get_loss_function(self):
        return nn.losses.cross_entropy
```

#### 2. Data Augmenter Plugins
Text augmentation for training:
```python
from bert_playground.core.ports import AugmenterPort
from typing import List, Dict, Any

class BackTranslationAugmenter(AugmenterPort):
    def __init__(self, config: AugmenterConfig, event_bus: EventBus):
        super().__init__(config, event_bus)
        self.source_lang = config.source_language
        self.target_lang = config.target_language
        self.probability = config.augmentation_probability
        
    async def augment(self, texts: List[str], **kwargs) -> List[str]:
        await self.event_bus.emit("augmenter.started", {
            "augmenter": self.__class__.__name__,
            "input_count": len(texts),
            "probability": self.probability
        })
        
        augmented_texts = []
        for text in texts:
            if self._should_augment():
                augmented_text = await self._back_translate(text)
                augmented_texts.append(augmented_text)
            else:
                augmented_texts.append(text)
        
        await self.event_bus.emit("augmenter.completed", {
            "augmenter": self.__class__.__name__,
            "output_count": len(augmented_texts),
            "augmented_count": sum(1 for orig, aug in zip(texts, augmented_texts) if orig != aug)
        })
        
        return augmented_texts
    
    async def _back_translate(self, text: str) -> str:
        # Implementation for back translation
        # This would use external translation service
        pass
    
    def _should_augment(self) -> bool:
        return mx.random.uniform() < self.probability
```

#### 3. Feature Engineering Plugins
Custom feature extraction:
```python
from bert_playground.core.ports import FeaturePort
import pandas as pd

class TechnicalIndicatorFeatures(FeaturePort):
    def __init__(self, config: FeatureConfig, event_bus: EventBus):
        super().__init__(config, event_bus)
        self.window_size = config.window_size
        self.indicators = config.indicators
    
    async def extract_features(self, data: pd.DataFrame) -> pd.DataFrame:
        await self.event_bus.emit("feature.extraction_started", {
            "feature_extractor": self.__class__.__name__,
            "input_shape": data.shape,
            "indicators": self.indicators
        })
        
        features = data.copy()
        
        if "moving_average" in self.indicators:
            features[f"ma_{self.window_size}"] = (
                data["value"].rolling(window=self.window_size).mean()
            )
        
        if "rsi" in self.indicators:
            features["rsi"] = self._calculate_rsi(data["value"])
        
        if "bollinger_bands" in self.indicators:
            bb_upper, bb_lower = self._calculate_bollinger_bands(data["value"])
            features["bb_upper"] = bb_upper
            features["bb_lower"] = bb_lower
        
        await self.event_bus.emit("feature.extraction_completed", {
            "feature_extractor": self.__class__.__name__,
            "output_shape": features.shape,
            "new_features": list(set(features.columns) - set(data.columns))
        })
        
        return features
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    def _calculate_bollinger_bands(self, prices: pd.Series) -> tuple:
        ma = prices.rolling(window=self.window_size).mean()
        std = prices.rolling(window=self.window_size).std()
        upper = ma + (2 * std)
        lower = ma - (2 * std)
        return upper, lower
```

#### 4. Training Callback Plugins
Custom training behavior:
```python
from bert_playground.core.ports import CallbackPort
from bert_playground.training.core import TrainerState

class EarlyStoppingCallback(CallbackPort):
    def __init__(self, config: CallbackConfig, event_bus: EventBus):
        super().__init__(config, event_bus)
        self.patience = config.patience
        self.min_delta = config.min_delta
        self.monitor = config.monitor
        self.best_value = None
        self.wait = 0
        self.stopped_epoch = 0
    
    async def on_epoch_end(self, epoch: int, logs: Dict[str, Any], state: TrainerState):
        current_value = logs.get(self.monitor)
        
        if current_value is None:
            await self.event_bus.emit("callback.warning", {
                "callback": self.__class__.__name__,
                "message": f"Monitor metric '{self.monitor}' not found in logs"
            })
            return
        
        if self.best_value is None or self._is_improvement(current_value):
            self.best_value = current_value
            self.wait = 0
            
            await self.event_bus.emit("callback.new_best_value", {
                "callback": self.__class__.__name__,
                "epoch": epoch,
                "metric": self.monitor,
                "value": current_value
            })
        else:
            self.wait += 1
            
            if self.wait >= self.patience:
                self.stopped_epoch = epoch
                state.should_stop = True
                
                await self.event_bus.emit("callback.early_stopping_triggered", {
                    "callback": self.__class__.__name__,
                    "epoch": epoch,
                    "patience": self.patience,
                    "best_value": self.best_value,
                    "current_value": current_value
                })
    
    def _is_improvement(self, current: float) -> bool:
        if self.monitor.endswith("loss"):
            return current < self.best_value - self.min_delta
        else:
            return current > self.best_value + self.min_delta
```

### Plugin Configuration

#### Configuration Schema
```python
from pydantic import BaseModel
from typing import Optional, Dict, Any, List

class PluginConfig(BaseModel):
    name: str
    version: str = "1.0.0"
    enabled: bool = True
    priority: int = 0
    dependencies: List[str] = []
    parameters: Dict[str, Any] = {}

class HeadConfig(PluginConfig):
    hidden_size: int
    num_labels: int
    dropout_rate: float = 0.1
    activation: str = "relu"

class AugmenterConfig(PluginConfig):
    augmentation_probability: float = 0.5
    max_augmentations_per_sample: int = 1

class FeatureConfig(PluginConfig):
    input_columns: List[str]
    output_prefix: str = "feature_"

class CallbackConfig(PluginConfig):
    monitor: str = "val_loss"
    patience: int = 10
    min_delta: float = 0.001
```

#### Plugin Manifest
Create `plugin.yaml` in your plugin directory:
```yaml
name: "custom_classification_head"
version: "1.2.0"
description: "Advanced classification head with attention mechanism"
author: "Your Name"
license: "MIT"

entry_points:
  head: "custom_head:CustomClassificationHead"
  
dependencies:
  - "mlx>=0.4.0"
  - "numpy>=1.21.0"

configuration:
  hidden_size:
    type: int
    default: 768
    description: "Hidden layer size"
  
  num_labels:
    type: int
    required: true
    description: "Number of output labels"
  
  attention_heads:
    type: int
    default: 8
    description: "Number of attention heads"

compatibility:
  k_bert_version: ">=2.0.0"
  phase: 2
```

## Plugin Lifecycle

### 1. Discovery Phase
```python
from bert_playground.core.plugin_manager import PluginManager

# Plugin manager discovers plugins
plugin_manager = PluginManager()

# Scan for plugins in directories
await plugin_manager.scan_plugins([
    "src/plugins",
    "~/.k-bert/plugins", 
    "/usr/local/lib/k-bert/plugins"
])

# Load specific plugin
plugin = await plugin_manager.load_plugin("custom_classification_head")
```

### 2. Initialization Phase
```python
class PluginInitializer:
    async def initialize_plugin(self, plugin_config: PluginConfig, 
                              container: Container) -> PluginPort:
        # Validate configuration
        self._validate_config(plugin_config)
        
        # Resolve dependencies
        dependencies = await self._resolve_dependencies(
            plugin_config.dependencies, container
        )
        
        # Create plugin instance
        plugin = plugin_config.entry_point(**dependencies)
        
        # Initialize plugin
        await plugin.initialize()
        
        # Emit lifecycle event
        await self.event_bus.emit("plugin.initialized", {
            "plugin_name": plugin_config.name,
            "version": plugin_config.version,
            "dependencies": plugin_config.dependencies
        })
        
        return plugin
```

### 3. Activation Phase
```python
class PluginActivator:
    async def activate_plugin(self, plugin: PluginPort) -> None:
        try:
            # Pre-activation hooks
            await self._run_pre_activation_hooks(plugin)
            
            # Activate plugin
            await plugin.activate()
            
            # Post-activation hooks
            await self._run_post_activation_hooks(plugin)
            
            await self.event_bus.emit("plugin.activated", {
                "plugin_name": plugin.name,
                "activation_time": time.time()
            })
            
        except Exception as e:
            await self.event_bus.emit("plugin.activation_failed", {
                "plugin_name": plugin.name,
                "error": str(e)
            })
            raise
```

### 4. Runtime Phase
```python
class PluginRunner:
    async def run_plugin_method(self, plugin: PluginPort, 
                               method: str, *args, **kwargs):
        try:
            # Pre-execution hooks
            await self._run_pre_execution_hooks(plugin, method)
            
            # Execute plugin method
            result = await getattr(plugin, method)(*args, **kwargs)
            
            # Post-execution hooks
            await self._run_post_execution_hooks(plugin, method, result)
            
            return result
            
        except Exception as e:
            await self.event_bus.emit("plugin.execution_error", {
                "plugin_name": plugin.name,
                "method": method,
                "error": str(e)
            })
            raise
```

### 5. Deactivation Phase
```python
class PluginDeactivator:
    async def deactivate_plugin(self, plugin: PluginPort) -> None:
        try:
            # Pre-deactivation hooks
            await self._run_pre_deactivation_hooks(plugin)
            
            # Deactivate plugin
            await plugin.deactivate()
            
            # Cleanup resources
            await self._cleanup_plugin_resources(plugin)
            
            await self.event_bus.emit("plugin.deactivated", {
                "plugin_name": plugin.name,
                "deactivation_time": time.time()
            })
            
        except Exception as e:
            await self.event_bus.emit("plugin.deactivation_failed", {
                "plugin_name": plugin.name,
                "error": str(e)
            })
            raise
```

## Dependency Injection

### Container Setup
```python
from bert_playground.core.di import Container, Injectable

class PluginContainer(Container):
    def __init__(self, event_bus: EventBus):
        super().__init__()
        self.event_bus = event_bus
        self._setup_core_dependencies()
    
    def _setup_core_dependencies(self):
        # Register core services
        self.register(EventBus, self.event_bus)
        self.register(Logger, self._create_logger)
        self.register(MetricsCollector, self._create_metrics_collector)
    
    def _create_logger(self) -> Logger:
        return Logger(self.event_bus)
    
    def _create_metrics_collector(self) -> MetricsCollector:
        return MetricsCollector(self.event_bus)
```

### Injectable Plugins
```python
from bert_playground.core.di import Injectable, inject

@Injectable
class SmartFeatureExtractor(FeaturePort):
    def __init__(self, 
                 config: FeatureConfig,
                 event_bus: EventBus = inject(),
                 logger: Logger = inject(),
                 metrics: MetricsCollector = inject()):
        super().__init__(config, event_bus)
        self.logger = logger
        self.metrics = metrics
    
    async def extract_features(self, data: pd.DataFrame) -> pd.DataFrame:
        with self.metrics.timer("feature_extraction"):
            self.logger.info(f"Extracting features from {data.shape[0]} samples")
            
            # Feature extraction logic
            features = await self._extract_smart_features(data)
            
            self.logger.info(f"Extracted {features.shape[1]} features")
            
            return features
```

## Hot Reloading

### Plugin Watcher
```python
import asyncio
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

class PluginWatcher(FileSystemEventHandler):
    def __init__(self, plugin_manager: PluginManager):
        self.plugin_manager = plugin_manager
        self.reload_queue = asyncio.Queue()
    
    def on_modified(self, event):
        if event.is_directory:
            return
        
        if event.src_path.endswith(('.py', '.yaml')):
            plugin_path = self._get_plugin_path(event.src_path)
            if plugin_path:
                asyncio.create_task(
                    self.reload_queue.put(plugin_path)
                )
    
    async def start_watching(self, plugin_directories: List[str]):
        observer = Observer()
        
        for directory in plugin_directories:
            observer.schedule(self, directory, recursive=True)
        
        observer.start()
        
        # Process reload queue
        while True:
            plugin_path = await self.reload_queue.get()
            await self._reload_plugin(plugin_path)
    
    async def _reload_plugin(self, plugin_path: str):
        try:
            await self.plugin_manager.reload_plugin(plugin_path)
            
            await self.event_bus.emit("plugin.reloaded", {
                "plugin_path": plugin_path,
                "timestamp": time.time()
            })
            
        except Exception as e:
            await self.event_bus.emit("plugin.reload_failed", {
                "plugin_path": plugin_path,
                "error": str(e)
            })
```

### Safe Reloading
```python
class SafePluginReloader:
    def __init__(self, plugin_manager: PluginManager):
        self.plugin_manager = plugin_manager
        self.active_executions = {}
    
    async def reload_plugin(self, plugin_name: str):
        # Wait for active executions to complete
        await self._wait_for_completion(plugin_name)
        
        # Get current plugin state
        old_plugin = self.plugin_manager.get_plugin(plugin_name)
        plugin_state = await self._capture_state(old_plugin)
        
        try:
            # Deactivate old plugin
            await old_plugin.deactivate()
            
            # Reload plugin code
            new_plugin = await self.plugin_manager.reload_plugin_code(plugin_name)
            
            # Restore state to new plugin
            await self._restore_state(new_plugin, plugin_state)
            
            # Activate new plugin
            await new_plugin.activate()
            
            # Replace in registry
            self.plugin_manager.replace_plugin(plugin_name, new_plugin)
            
        except Exception as e:
            # Rollback on failure
            await self._rollback_plugin(plugin_name, old_plugin)
            raise
```

## Plugin Testing

### Unit Testing Framework
```python
import pytest
from bert_playground.testing import PluginTestCase

class TestCustomClassificationHead(PluginTestCase):
    async def setup_method(self):
        self.config = HeadConfig(
            name="test_head",
            hidden_size=768,
            num_labels=2
        )
        
        self.event_bus = self.create_mock_event_bus()
        self.head = CustomClassificationHead(self.config, self.event_bus)
    
    async def test_forward_pass(self):
        # Create test input
        batch_size, seq_len, hidden_size = 32, 128, 768
        hidden_states = mx.random.normal((batch_size, seq_len, hidden_size))
        
        # Run forward pass
        logits = await self.head.forward(hidden_states)
        
        # Verify output shape
        expected_shape = (batch_size, seq_len, self.config.num_labels)
        assert logits.shape == expected_shape
        
        # Verify events were emitted
        self.assert_event_emitted("head.forward_start")
        self.assert_event_emitted("head.forward_complete")
    
    async def test_gradient_flow(self):
        # Test that gradients flow properly
        hidden_states = mx.random.normal((2, 10, 768))
        labels = mx.array([0, 1])
        
        def loss_fn(head, inputs, labels):
            logits = head.forward(inputs)
            return mx.mean(nn.losses.cross_entropy(logits, labels))
        
        # Compute gradients
        loss_and_grad_fn = nn.value_and_grad(loss_fn)
        loss, grads = loss_and_grad_fn(self.head, hidden_states, labels)
        
        # Verify gradients exist and are non-zero
        assert grads is not None
        assert any(mx.sum(mx.abs(g)) > 0 for g in grads.values())
```

### Integration Testing
```python
class TestPluginIntegration(PluginTestCase):
    async def test_plugin_lifecycle(self):
        # Test complete plugin lifecycle
        plugin_manager = self.create_test_plugin_manager()
        
        # Load plugin
        plugin = await plugin_manager.load_plugin("test_plugin")
        assert plugin is not None
        
        # Initialize plugin
        await plugin.initialize()
        self.assert_event_emitted("plugin.initialized")
        
        # Activate plugin
        await plugin.activate()
        self.assert_event_emitted("plugin.activated")
        
        # Test plugin functionality
        result = await plugin.process_data(self.test_data)
        assert result is not None
        
        # Deactivate plugin
        await plugin.deactivate()
        self.assert_event_emitted("plugin.deactivated")
    
    async def test_plugin_dependencies(self):
        # Test that plugin dependencies are resolved correctly
        container = self.create_test_container()
        
        plugin = await container.resolve(DependentPlugin)
        
        # Verify dependencies were injected
        assert plugin.logger is not None
        assert plugin.metrics is not None
        assert plugin.event_bus is not None
```

### Performance Testing
```python
class TestPluginPerformance(PluginTestCase):
    async def test_head_performance(self):
        head = self.create_test_head()
        
        # Warm up
        for _ in range(10):
            await head.forward(self.test_input)
        
        # Measure performance
        with self.performance_timer() as timer:
            for _ in range(100):
                await head.forward(self.test_input)
        
        # Assert performance requirements
        avg_time = timer.average_time
        assert avg_time < 0.01  # Less than 10ms per forward pass
        
        # Check memory usage
        memory_usage = self.get_memory_usage()
        assert memory_usage < 100  # Less than 100MB
```

## Best Practices

### Plugin Design Principles

#### 1. Single Responsibility
```python
# Good: Focused plugin
class SentimentAnalysisHead(HeadPort):
    def __init__(self, config: HeadConfig, event_bus: EventBus):
        super().__init__(config, event_bus)
        # Only sentiment analysis logic

# Bad: Multiple responsibilities
class MultiTaskHead(HeadPort):
    def __init__(self, config: HeadConfig, event_bus: EventBus):
        super().__init__(config, event_bus)
        # Sentiment, NER, and classification logic mixed together
```

#### 2. Configuration-Driven
```python
class ConfigurableAugmenter(AugmenterPort):
    def __init__(self, config: AugmenterConfig, event_bus: EventBus):
        super().__init__(config, event_bus)
        
        # Load augmentation strategies from config
        self.strategies = self._load_strategies(config.strategies)
        self.probabilities = config.strategy_probabilities
        
    def _load_strategies(self, strategy_configs):
        strategies = []
        for strategy_config in strategy_configs:
            strategy_class = self._get_strategy_class(strategy_config.type)
            strategy = strategy_class(strategy_config.params)
            strategies.append(strategy)
        return strategies
```

#### 3. Event-Driven Communication
```python
class EventAwarePlugin(PluginPort):
    async def process_data(self, data):
        # Emit start event
        await self.event_bus.emit("plugin.processing_started", {
            "plugin": self.name,
            "data_size": len(data)
        })
        
        try:
            result = await self._process_data_internal(data)
            
            # Emit success event
            await self.event_bus.emit("plugin.processing_completed", {
                "plugin": self.name,
                "result_size": len(result)
            })
            
            return result
            
        except Exception as e:
            # Emit error event
            await self.event_bus.emit("plugin.processing_failed", {
                "plugin": self.name,
                "error": str(e)
            })
            raise
```

#### 4. Resource Management
```python
class ResourceManagedPlugin(PluginPort):
    def __init__(self, config: PluginConfig, event_bus: EventBus):
        super().__init__(config, event_bus)
        self._resources = []
    
    async def initialize(self):
        # Acquire resources
        self.model = await self._load_model()
        self._resources.append(self.model)
        
        self.cache = await self._create_cache()
        self._resources.append(self.cache)
    
    async def deactivate(self):
        # Clean up resources
        for resource in self._resources:
            if hasattr(resource, 'close'):
                await resource.close()
            elif hasattr(resource, 'cleanup'):
                await resource.cleanup()
        
        self._resources.clear()
```

### Error Handling

#### Graceful Degradation
```python
class RobustPlugin(PluginPort):
    async def process_data(self, data):
        try:
            return await self._process_with_advanced_algorithm(data)
        except AdvancedAlgorithmError:
            self.logger.warning("Advanced algorithm failed, falling back to basic")
            return await self._process_with_basic_algorithm(data)
        except Exception as e:
            self.logger.error(f"All algorithms failed: {e}")
            return self._get_default_result(data)
```

#### Error Recovery
```python
class RetryablePlugin(PluginPort):
    async def process_data(self, data, max_retries: int = 3):
        for attempt in range(max_retries):
            try:
                return await self._process_data_attempt(data)
            
            except RetryableError as e:
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt  # Exponential backoff
                    await asyncio.sleep(wait_time)
                    continue
                else:
                    raise
            
            except NonRetryableError:
                # Don't retry non-retryable errors
                raise
```

### Documentation

#### Plugin Documentation Template
```python
class DocumentedPlugin(PluginPort):
    """
    Advanced feature extraction plugin for tabular data.
    
    This plugin extracts statistical and temporal features from numerical
    columns in tabular datasets.
    
    Configuration:
        - statistical_features (List[str]): Statistical features to extract
        - temporal_features (List[str]): Temporal features to extract  
        - window_size (int): Size of rolling window for temporal features
        
    Events Emitted:
        - plugin.feature_extraction_started: When extraction begins
        - plugin.feature_extraction_completed: When extraction completes
        - plugin.feature_validation_failed: If feature validation fails
        
    Example:
        ```yaml
        plugins:
          - name: advanced_feature_extractor
            type: feature
            config:
              statistical_features: ["mean", "std", "skew"]
              temporal_features: ["trend", "seasonality"]
              window_size: 30
        ```
    
    Performance:
        - Memory: O(n * m) where n is samples, m is features
        - Time: O(n * w) where w is window size
        - Recommended batch size: 1000-10000 samples
    """
    
    async def extract_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Extract advanced features from tabular data.
        
        Args:
            data: Input DataFrame with numerical columns
            
        Returns:
            DataFrame with original and extracted features
            
        Raises:
            FeatureExtractionError: If extraction fails
            ValidationError: If input data is invalid
        """
        pass
```

## Plugin Distribution

### Packaging
```python
# setup.py for plugin package
from setuptools import setup, find_packages

setup(
    name="k-bert-sentiment-plugin",
    version="1.0.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="Advanced sentiment analysis plugin for k-bert",
    
    packages=find_packages(),
    package_data={
        "k_bert_sentiment_plugin": ["*.yaml", "models/*", "data/*"]
    },
    
    install_requires=[
        "k-bert>=2.0.0",
        "transformers>=4.20.0",
        "torch>=1.12.0"
    ],
    
    entry_points={
        "k_bert.plugins": [
            "sentiment_head = k_bert_sentiment_plugin:SentimentHead",
            "emotion_classifier = k_bert_sentiment_plugin:EmotionClassifier"
        ]
    },
    
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.9+",
    ],
    
    python_requires=">=3.9",
)
```

### Plugin Registry
```yaml
# ~/.k-bert/plugins/registry.yaml
plugins:
  - name: "sentiment_analysis"
    source: "pypi"
    package: "k-bert-sentiment-plugin"
    version: ">=1.0.0"
    
  - name: "custom_augmenter"
    source: "git"
    url: "https://github.com/user/k-bert-custom-augmenter.git"
    branch: "main"
    
  - name: "local_features"
    source: "local"
    path: "./plugins/local_features"
```

## Conclusion

The Phase 2 plugin system provides a robust foundation for extending k-bert functionality. Key benefits include:

- **Type Safety**: Full type checking and IDE support
- **Event Integration**: Rich event-driven communication
- **Dependency Injection**: Clean dependency management
- **Hot Reloading**: Development-friendly plugin updates
- **Testing Support**: Comprehensive testing framework
- **Performance**: Optimized for MLX and async execution

Following the patterns and best practices in this guide will help you create maintainable, performant, and reliable plugins for the k-bert ecosystem.