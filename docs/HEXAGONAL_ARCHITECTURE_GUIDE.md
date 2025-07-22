# Hexagonal Architecture Guide

## Overview

Phase 2 adopts hexagonal architecture (also known as Ports and Adapters) to achieve clean separation between business logic and external concerns. This architectural pattern makes the k-bert system more testable, maintainable, and flexible.

## Core Concepts

### Hexagonal Architecture Principles

1. **Inside-Out Design**: Business logic at the center, external concerns at the edges
2. **Port Abstraction**: Define interfaces for external interactions
3. **Adapter Implementation**: Concrete implementations of ports
4. **Dependency Inversion**: Dependencies point inward toward business logic
5. **Testability**: Easy to test with mock adapters

### Architecture Layers

```
┌─────────────────────────────────────────────────────────────┐
│                        CLI Layer                            │
│                  (Primary Adapters)                        │
├─────────────────────────────────────────────────────────────┤
│                     Application Layer                      │
│                   (Use Cases/Services)                     │
├─────────────────────────────────────────────────────────────┤
│                      Domain Layer                          │
│                   (Business Logic)                         │
├─────────────────────────────────────────────────────────────┤
│                      Port Layer                            │
│                    (Abstractions)                          │
├─────────────────────────────────────────────────────────────┤
│                   Adapter Layer                            │
│                 (Implementations)                          │
└─────────────────────────────────────────────────────────────┘
```

## Port Definitions

### Core Ports

```python
from abc import ABC, abstractmethod
from typing import Protocol, runtime_checkable
import mlx.core as mx
import pandas as pd

# Data Access Ports
@runtime_checkable
class DataLoaderPort(Protocol):
    async def load_data(self, path: str) -> pd.DataFrame: ...
    async def save_data(self, data: pd.DataFrame, path: str) -> None: ...
    async def validate_data(self, data: pd.DataFrame) -> bool: ...

@runtime_checkable
class ModelRepositoryPort(Protocol):
    async def save_model(self, model: Any, path: str) -> None: ...
    async def load_model(self, path: str) -> Any: ...
    async def list_models(self) -> List[str]: ...
    async def delete_model(self, path: str) -> bool: ...

# Computation Ports
@runtime_checkable
class ModelPort(Protocol):
    async def forward(self, inputs: mx.array) -> mx.array: ...
    async def backward(self, loss: mx.array) -> Dict[str, mx.array]: ...
    def get_parameters(self) -> Dict[str, mx.array]: ...
    def set_parameters(self, params: Dict[str, mx.array]) -> None: ...

@runtime_checkable
class OptimizerPort(Protocol):
    def update(self, model: ModelPort, gradients: Dict[str, mx.array]) -> None: ...
    def get_state(self) -> Dict[str, Any]: ...
    def set_state(self, state: Dict[str, Any]) -> None: ...

# External Service Ports
@runtime_checkable
class MetricsPort(Protocol):
    async def log_metric(self, name: str, value: float, step: int = None) -> None: ...
    async def log_parameters(self, params: Dict[str, Any]) -> None: ...
    async def log_artifact(self, path: str, artifact_type: str = None) -> None: ...

@runtime_checkable
class NotificationPort(Protocol):
    async def send_notification(self, message: str, level: str = "info") -> None: ...
    async def send_alert(self, title: str, message: str, priority: str = "medium") -> None: ...
```

### Plugin Ports

```python
# Plugin System Ports
@runtime_checkable
class HeadPort(Protocol):
    async def forward(self, hidden_states: mx.array, **kwargs) -> mx.array: ...
    def get_loss_function(self) -> Callable: ...
    def get_output_size(self) -> int: ...

@runtime_checkable
class AugmenterPort(Protocol):
    async def augment(self, texts: List[str], **kwargs) -> List[str]: ...
    def get_augmentation_probability(self) -> float: ...
    def set_augmentation_probability(self, prob: float) -> None: ...

@runtime_checkable
class FeaturePort(Protocol):
    async def extract_features(self, data: pd.DataFrame) -> pd.DataFrame: ...
    def get_feature_names(self) -> List[str]: ...
    def get_feature_importance(self) -> Dict[str, float]: ...

@runtime_checkable
class CallbackPort(Protocol):
    async def on_train_begin(self, logs: Dict[str, Any] = None) -> None: ...
    async def on_train_end(self, logs: Dict[str, Any] = None) -> None: ...
    async def on_epoch_begin(self, epoch: int, logs: Dict[str, Any] = None) -> None: ...
    async def on_epoch_end(self, epoch: int, logs: Dict[str, Any] = None) -> None: ...
    async def on_batch_begin(self, batch: int, logs: Dict[str, Any] = None) -> None: ...
    async def on_batch_end(self, batch: int, logs: Dict[str, Any] = None) -> None: ...
```

## Adapter Implementations

### Data Access Adapters

#### File System Data Loader
```python
from bert_playground.core.ports import DataLoaderPort
import pandas as pd
import aiofiles
from pathlib import Path

class FileSystemDataAdapter(DataLoaderPort):
    def __init__(self, base_path: str = "."):
        self.base_path = Path(base_path)
    
    async def load_data(self, path: str) -> pd.DataFrame:
        full_path = self.base_path / path
        
        if full_path.suffix == '.csv':
            return pd.read_csv(full_path)
        elif full_path.suffix == '.parquet':
            return pd.read_parquet(full_path)
        elif full_path.suffix == '.json':
            return pd.read_json(full_path)
        else:
            raise ValueError(f"Unsupported file format: {full_path.suffix}")
    
    async def save_data(self, data: pd.DataFrame, path: str) -> None:
        full_path = self.base_path / path
        full_path.parent.mkdir(parents=True, exist_ok=True)
        
        if full_path.suffix == '.csv':
            data.to_csv(full_path, index=False)
        elif full_path.suffix == '.parquet':
            data.to_parquet(full_path, index=False)
        elif full_path.suffix == '.json':
            data.to_json(full_path, orient='records')
        else:
            raise ValueError(f"Unsupported file format: {full_path.suffix}")
    
    async def validate_data(self, data: pd.DataFrame) -> bool:
        # Basic validation
        if data.empty:
            return False
        
        if data.isnull().any().any():
            return False
        
        return True
```

#### Cloud Storage Data Adapter
```python
class S3DataAdapter(DataLoaderPort):
    def __init__(self, bucket: str, access_key: str, secret_key: str):
        import boto3
        self.bucket = bucket
        self.s3_client = boto3.client(
            's3',
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key
        )
    
    async def load_data(self, path: str) -> pd.DataFrame:
        try:
            obj = await self._get_object_async(self.bucket, path)
            
            if path.endswith('.csv'):
                return pd.read_csv(obj['Body'])
            elif path.endswith('.parquet'):
                return pd.read_parquet(obj['Body'])
            else:
                raise ValueError(f"Unsupported format: {path}")
                
        except Exception as e:
            raise DataLoadError(f"Failed to load data from S3: {e}")
    
    async def save_data(self, data: pd.DataFrame, path: str) -> None:
        try:
            if path.endswith('.csv'):
                buffer = data.to_csv(index=False)
            elif path.endswith('.parquet'):
                buffer = data.to_parquet(index=False)
            else:
                raise ValueError(f"Unsupported format: {path}")
            
            await self._put_object_async(self.bucket, path, buffer)
            
        except Exception as e:
            raise DataSaveError(f"Failed to save data to S3: {e}")
```

### Model Repository Adapters

#### MLX Model Repository
```python
from bert_playground.core.ports import ModelRepositoryPort
import mlx.core as mx
import mlx.nn as nn
import pickle
from pathlib import Path

class MLXModelRepository(ModelRepositoryPort):
    def __init__(self, base_path: str = "./models"):
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
    
    async def save_model(self, model: nn.Module, path: str) -> None:
        full_path = self.base_path / path
        full_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save MLX model weights
        weights = model.parameters()
        mx.savez(str(full_path / "weights.npz"), **weights)
        
        # Save model configuration
        config = {
            "model_class": model.__class__.__name__,
            "model_config": getattr(model, 'config', {}),
            "architecture": getattr(model, 'architecture', None)
        }
        
        with open(full_path / "config.json", 'w') as f:
            json.dump(config, f, indent=2)
        
        # Save model metadata
        metadata = {
            "save_time": datetime.now().isoformat(),
            "mlx_version": mx.__version__,
            "model_size": self._calculate_model_size(model)
        }
        
        with open(full_path / "metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
    
    async def load_model(self, path: str) -> nn.Module:
        full_path = self.base_path / path
        
        # Load configuration
        with open(full_path / "config.json", 'r') as f:
            config = json.load(f)
        
        # Reconstruct model
        model_class = self._get_model_class(config["model_class"])
        model = model_class(config.get("model_config", {}))
        
        # Load weights
        weights = mx.load(str(full_path / "weights.npz"))
        model.load_weights(list(weights.items()))
        
        return model
    
    async def list_models(self) -> List[str]:
        models = []
        for model_dir in self.base_path.iterdir():
            if model_dir.is_dir() and (model_dir / "config.json").exists():
                models.append(model_dir.name)
        return sorted(models)
    
    async def delete_model(self, path: str) -> bool:
        full_path = self.base_path / path
        if full_path.exists() and full_path.is_dir():
            import shutil
            shutil.rmtree(full_path)
            return True
        return False
```

### External Service Adapters

#### MLflow Metrics Adapter
```python
from bert_playground.core.ports import MetricsPort
import mlflow

class MLflowMetricsAdapter(MetricsPort):
    def __init__(self, experiment_name: str = "k-bert-training"):
        self.experiment_name = experiment_name
        self._setup_experiment()
    
    def _setup_experiment(self):
        mlflow.set_experiment(self.experiment_name)
    
    async def log_metric(self, name: str, value: float, step: int = None) -> None:
        mlflow.log_metric(name, value, step=step)
    
    async def log_parameters(self, params: Dict[str, Any]) -> None:
        mlflow.log_params(params)
    
    async def log_artifact(self, path: str, artifact_type: str = None) -> None:
        if artifact_type == "model":
            mlflow.log_artifacts(path, "models")
        elif artifact_type == "data":
            mlflow.log_artifacts(path, "data")
        else:
            mlflow.log_artifact(path)

class WandBMetricsAdapter(MetricsPort):
    def __init__(self, project_name: str = "k-bert", entity: str = None):
        import wandb
        self.wandb = wandb
        self.wandb.init(project=project_name, entity=entity)
    
    async def log_metric(self, name: str, value: float, step: int = None) -> None:
        self.wandb.log({name: value}, step=step)
    
    async def log_parameters(self, params: Dict[str, Any]) -> None:
        self.wandb.config.update(params)
    
    async def log_artifact(self, path: str, artifact_type: str = None) -> None:
        self.wandb.save(path)
```

#### Notification Adapters
```python
from bert_playground.core.ports import NotificationPort

class SlackNotificationAdapter(NotificationPort):
    def __init__(self, webhook_url: str, channel: str = "#general"):
        self.webhook_url = webhook_url
        self.channel = channel
    
    async def send_notification(self, message: str, level: str = "info") -> None:
        color_map = {
            "info": "#36a64f",      # green
            "warning": "#ff9900",   # orange  
            "error": "#ff0000",     # red
            "success": "#00ff00"    # bright green
        }
        
        payload = {
            "channel": self.channel,
            "attachments": [{
                "color": color_map.get(level, "#36a64f"),
                "text": message,
                "ts": time.time()
            }]
        }
        
        async with aiohttp.ClientSession() as session:
            await session.post(self.webhook_url, json=payload)
    
    async def send_alert(self, title: str, message: str, priority: str = "medium") -> None:
        alert_message = f"*{title}*\n{message}"
        await self.send_notification(alert_message, level="warning")

class EmailNotificationAdapter(NotificationPort):
    def __init__(self, smtp_host: str, smtp_port: int, username: str, password: str):
        self.smtp_host = smtp_host
        self.smtp_port = smtp_port
        self.username = username
        self.password = password
    
    async def send_notification(self, message: str, level: str = "info") -> None:
        import aiosmtplib
        from email.message import EmailMessage
        
        msg = EmailMessage()
        msg["Subject"] = f"K-BERT Notification ({level.upper()})"
        msg["From"] = self.username
        msg["To"] = self.username  # Could be configurable
        msg.set_content(message)
        
        await aiosmtplib.send(
            msg,
            hostname=self.smtp_host,
            port=self.smtp_port,
            username=self.username,
            password=self.password,
            use_tls=True
        )
```

## Domain Layer

### Core Business Logic

```python
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from datetime import datetime

@dataclass
class TrainingConfiguration:
    model_type: str
    num_epochs: int
    batch_size: int
    learning_rate: float
    optimizer_type: str
    loss_function: str
    validation_split: float = 0.2
    early_stopping_patience: int = 10
    checkpoint_frequency: int = 1

@dataclass
class TrainingMetrics:
    epoch: int
    train_loss: float
    val_loss: float
    train_accuracy: Optional[float] = None
    val_accuracy: Optional[float] = None
    learning_rate: Optional[float] = None
    epoch_duration: Optional[float] = None

@dataclass 
class ModelInfo:
    name: str
    version: str
    architecture: str
    parameters: int
    created_at: datetime
    last_trained: Optional[datetime] = None
    metrics: Optional[TrainingMetrics] = None

class TrainingDomain:
    """Core business logic for training models"""
    
    def __init__(self):
        self.training_history: List[TrainingMetrics] = []
        self.current_epoch = 0
        self.best_metric = float('inf')
        self.patience_counter = 0
    
    def should_stop_early(self, current_metric: float, patience: int) -> bool:
        """Determine if training should stop early based on validation metrics"""
        if current_metric < self.best_metric:
            self.best_metric = current_metric
            self.patience_counter = 0
            return False
        else:
            self.patience_counter += 1
            return self.patience_counter >= patience
    
    def calculate_learning_rate(self, base_lr: float, epoch: int, 
                              schedule: str = "cosine") -> float:
        """Calculate learning rate based on schedule"""
        if schedule == "constant":
            return base_lr
        elif schedule == "cosine":
            import math
            return base_lr * 0.5 * (1 + math.cos(math.pi * epoch / 100))
        elif schedule == "exponential":
            return base_lr * (0.95 ** epoch)
        else:
            return base_lr
    
    def validate_configuration(self, config: TrainingConfiguration) -> List[str]:
        """Validate training configuration"""
        errors = []
        
        if config.num_epochs <= 0:
            errors.append("Number of epochs must be positive")
        
        if config.batch_size <= 0:
            errors.append("Batch size must be positive")
            
        if not 0 < config.learning_rate < 1:
            errors.append("Learning rate must be between 0 and 1")
            
        if not 0 < config.validation_split < 1:
            errors.append("Validation split must be between 0 and 1")
        
        return errors
```

### Model Management Domain

```python
class ModelManagementDomain:
    """Business logic for model lifecycle management"""
    
    def __init__(self):
        self.registered_models: Dict[str, ModelInfo] = {}
    
    def register_model(self, model_info: ModelInfo) -> None:
        """Register a new model"""
        if model_info.name in self.registered_models:
            existing = self.registered_models[model_info.name]
            if self._is_newer_version(model_info.version, existing.version):
                self.registered_models[model_info.name] = model_info
        else:
            self.registered_models[model_info.name] = model_info
    
    def get_best_model(self, metric: str = "val_loss") -> Optional[ModelInfo]:
        """Get the best model based on a metric"""
        if not self.registered_models:
            return None
        
        best_model = None
        best_value = float('inf') if 'loss' in metric else float('-inf')
        
        for model_info in self.registered_models.values():
            if model_info.metrics is None:
                continue
            
            metric_value = getattr(model_info.metrics, metric, None)
            if metric_value is None:
                continue
            
            if ('loss' in metric and metric_value < best_value) or \
               ('accuracy' in metric and metric_value > best_value):
                best_value = metric_value
                best_model = model_info
        
        return best_model
    
    def _is_newer_version(self, version1: str, version2: str) -> bool:
        """Compare version strings"""
        from packaging import version
        return version.parse(version1) > version.parse(version2)
```

## Application Layer

### Use Cases/Services

```python
from bert_playground.core.ports import *
from bert_playground.core.events import EventBus

class TrainingService:
    """Application service for training workflows"""
    
    def __init__(self, 
                 data_loader: DataLoaderPort,
                 model_repository: ModelRepositoryPort,
                 metrics_adapter: MetricsPort,
                 notification_adapter: NotificationPort,
                 event_bus: EventBus):
        self.data_loader = data_loader
        self.model_repository = model_repository
        self.metrics = metrics_adapter
        self.notifications = notification_adapter
        self.event_bus = event_bus
        self.training_domain = TrainingDomain()
    
    async def train_model(self, config: TrainingConfiguration, 
                         data_path: str, model_name: str) -> ModelInfo:
        """Execute complete training workflow"""
        
        # Validate configuration
        validation_errors = self.training_domain.validate_configuration(config)
        if validation_errors:
            raise ValueError(f"Configuration errors: {validation_errors}")
        
        await self.event_bus.emit("training.started", {
            "model_name": model_name,
            "config": config.__dict__
        })
        
        try:
            # Load and validate data
            data = await self.data_loader.load_data(data_path)
            is_valid = await self.data_loader.validate_data(data)
            if not is_valid:
                raise ValueError("Data validation failed")
            
            await self.event_bus.emit("training.data_loaded", {
                "data_shape": data.shape,
                "data_path": data_path
            })
            
            # Log training parameters
            await self.metrics.log_parameters(config.__dict__)
            
            # Create and train model
            model = await self._create_model(config)
            trained_model = await self._train_model_internal(model, data, config)
            
            # Save model
            model_path = f"{model_name}_v{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            await self.model_repository.save_model(trained_model, model_path)
            
            # Create model info
            model_info = ModelInfo(
                name=model_name,
                version="1.0.0",
                architecture=config.model_type,
                parameters=self._count_parameters(trained_model),
                created_at=datetime.now(),
                last_trained=datetime.now(),
                metrics=self.training_domain.training_history[-1] if self.training_domain.training_history else None
            )
            
            await self.event_bus.emit("training.completed", {
                "model_name": model_name,
                "model_path": model_path,
                "final_metrics": model_info.metrics.__dict__ if model_info.metrics else {}
            })
            
            # Send success notification
            await self.notifications.send_notification(
                f"Model '{model_name}' training completed successfully",
                level="success"
            )
            
            return model_info
            
        except Exception as e:
            await self.event_bus.emit("training.failed", {
                "model_name": model_name,
                "error": str(e)
            })
            
            await self.notifications.send_alert(
                "Training Failed",
                f"Model '{model_name}' training failed: {str(e)}",
                priority="high"
            )
            
            raise
    
    async def _train_model_internal(self, model: Any, data: pd.DataFrame, 
                                  config: TrainingConfiguration) -> Any:
        """Internal training loop"""
        
        for epoch in range(config.num_epochs):
            await self.event_bus.emit("training.epoch_started", {
                "epoch": epoch + 1,
                "total_epochs": config.num_epochs
            })
            
            # Training step
            train_metrics = await self._train_epoch(model, data, config, epoch)
            
            # Validation step
            val_metrics = await self._validate_epoch(model, data, config, epoch)
            
            # Combine metrics
            epoch_metrics = TrainingMetrics(
                epoch=epoch + 1,
                train_loss=train_metrics["loss"],
                val_loss=val_metrics["loss"],
                train_accuracy=train_metrics.get("accuracy"),
                val_accuracy=val_metrics.get("accuracy"),
                learning_rate=self.training_domain.calculate_learning_rate(
                    config.learning_rate, epoch
                )
            )
            
            self.training_domain.training_history.append(epoch_metrics)
            
            # Log metrics
            await self.metrics.log_metric("train_loss", epoch_metrics.train_loss, epoch)
            await self.metrics.log_metric("val_loss", epoch_metrics.val_loss, epoch)
            
            await self.event_bus.emit("training.epoch_completed", {
                "epoch": epoch + 1,
                "metrics": epoch_metrics.__dict__
            })
            
            # Check early stopping
            if self.training_domain.should_stop_early(
                epoch_metrics.val_loss, config.early_stopping_patience
            ):
                await self.event_bus.emit("training.early_stopped", {
                    "epoch": epoch + 1,
                    "reason": "early_stopping"
                })
                break
        
        return model
```

### Model Management Service

```python
class ModelManagementService:
    """Service for managing model lifecycle"""
    
    def __init__(self,
                 model_repository: ModelRepositoryPort,
                 metrics_adapter: MetricsPort,
                 event_bus: EventBus):
        self.model_repository = model_repository
        self.metrics = metrics_adapter
        self.event_bus = event_bus
        self.domain = ModelManagementDomain()
    
    async def deploy_model(self, model_name: str, version: str = None) -> str:
        """Deploy a model to production"""
        
        if version:
            model_path = f"{model_name}_v{version}"
        else:
            # Get latest version
            models = await self.model_repository.list_models()
            model_versions = [m for m in models if m.startswith(model_name)]
            if not model_versions:
                raise ValueError(f"No models found for {model_name}")
            model_path = sorted(model_versions)[-1]
        
        await self.event_bus.emit("model.deployment_started", {
            "model_name": model_name,
            "model_path": model_path
        })
        
        try:
            # Load model
            model = await self.model_repository.load_model(model_path)
            
            # Validate model
            await self._validate_model_for_deployment(model)
            
            # Deploy model (implementation depends on deployment target)
            deployment_id = await self._deploy_model_internal(model, model_path)
            
            await self.event_bus.emit("model.deployment_completed", {
                "model_name": model_name,
                "deployment_id": deployment_id
            })
            
            return deployment_id
            
        except Exception as e:
            await self.event_bus.emit("model.deployment_failed", {
                "model_name": model_name,
                "error": str(e)
            })
            raise
    
    async def compare_models(self, model1_name: str, model2_name: str) -> Dict[str, Any]:
        """Compare two models"""
        
        model1 = await self.model_repository.load_model(model1_name)
        model2 = await self.model_repository.load_model(model2_name)
        
        comparison = {
            "model1": {
                "name": model1_name,
                "parameters": self._count_parameters(model1),
                "size_mb": self._get_model_size(model1)
            },
            "model2": {
                "name": model2_name,
                "parameters": self._count_parameters(model2),
                "size_mb": self._get_model_size(model2)
            }
        }
        
        await self.event_bus.emit("model.comparison_completed", {
            "models": [model1_name, model2_name],
            "comparison": comparison
        })
        
        return comparison
```

## Dependency Injection Container

```python
from bert_playground.core.di import Container
from bert_playground.core.events import EventBus

class ApplicationContainer(Container):
    """Main DI container for the application"""
    
    def __init__(self):
        super().__init__()
        self._setup_dependencies()
    
    def _setup_dependencies(self):
        # Core infrastructure
        self.register_singleton(EventBus, self._create_event_bus)
        
        # Data adapters
        self.register(DataLoaderPort, self._create_data_loader)
        self.register(ModelRepositoryPort, self._create_model_repository)
        
        # External service adapters
        self.register(MetricsPort, self._create_metrics_adapter)
        self.register(NotificationPort, self._create_notification_adapter)
        
        # Application services
        self.register(TrainingService, self._create_training_service)
        self.register(ModelManagementService, self._create_model_service)
    
    def _create_event_bus(self) -> EventBus:
        return EventBus(
            max_subscribers=1000,
            buffer_size=10000
        )
    
    def _create_data_loader(self) -> DataLoaderPort:
        # Choose adapter based on configuration
        storage_type = self.get_config("storage.type", "filesystem")
        
        if storage_type == "filesystem":
            return FileSystemDataAdapter(
                base_path=self.get_config("storage.base_path", ".")
            )
        elif storage_type == "s3":
            return S3DataAdapter(
                bucket=self.get_config("storage.s3.bucket"),
                access_key=self.get_config("storage.s3.access_key"),
                secret_key=self.get_config("storage.s3.secret_key")
            )
        else:
            raise ValueError(f"Unknown storage type: {storage_type}")
    
    def _create_metrics_adapter(self) -> MetricsPort:
        metrics_type = self.get_config("metrics.type", "mlflow")
        
        if metrics_type == "mlflow":
            return MLflowMetricsAdapter(
                experiment_name=self.get_config("metrics.experiment_name", "k-bert")
            )
        elif metrics_type == "wandb":
            return WandBMetricsAdapter(
                project_name=self.get_config("metrics.project_name", "k-bert"),
                entity=self.get_config("metrics.entity")
            )
        else:
            raise ValueError(f"Unknown metrics type: {metrics_type}")
    
    def _create_training_service(self) -> TrainingService:
        return TrainingService(
            data_loader=self.resolve(DataLoaderPort),
            model_repository=self.resolve(ModelRepositoryPort),
            metrics_adapter=self.resolve(MetricsPort),
            notification_adapter=self.resolve(NotificationPort),
            event_bus=self.resolve(EventBus)
        )
```

## Testing with Hexagonal Architecture

### Mock Adapters for Testing

```python
class MockDataLoader(DataLoaderPort):
    def __init__(self):
        self.data_store = {}
        self.load_calls = []
        self.save_calls = []
    
    async def load_data(self, path: str) -> pd.DataFrame:
        self.load_calls.append(path)
        if path in self.data_store:
            return self.data_store[path]
        else:
            # Return test data
            return pd.DataFrame({
                'text': ['sample text 1', 'sample text 2'],
                'label': [0, 1]
            })
    
    async def save_data(self, data: pd.DataFrame, path: str) -> None:
        self.save_calls.append(path)
        self.data_store[path] = data
    
    async def validate_data(self, data: pd.DataFrame) -> bool:
        return not data.empty

class MockModelRepository(ModelRepositoryPort):
    def __init__(self):
        self.models = {}
        self.save_calls = []
        self.load_calls = []
    
    async def save_model(self, model: Any, path: str) -> None:
        self.save_calls.append(path)
        self.models[path] = model
    
    async def load_model(self, path: str) -> Any:
        self.load_calls.append(path)
        if path in self.models:
            return self.models[path]
        raise FileNotFoundError(f"Model not found: {path}")
    
    async def list_models(self) -> List[str]:
        return list(self.models.keys())
    
    async def delete_model(self, path: str) -> bool:
        if path in self.models:
            del self.models[path]
            return True
        return False
```

### Integration Tests

```python
import pytest
from bert_playground.testing import TestContainer

class TestTrainingServiceIntegration:
    @pytest.fixture
    def container(self):
        container = TestContainer()
        # Replace real adapters with mocks
        container.register(DataLoaderPort, MockDataLoader())
        container.register(ModelRepositoryPort, MockModelRepository())
        return container
    
    async def test_complete_training_workflow(self, container):
        training_service = container.resolve(TrainingService)
        
        config = TrainingConfiguration(
            model_type="bert",
            num_epochs=2,
            batch_size=16,
            learning_rate=0.001,
            optimizer_type="adamw",
            loss_function="cross_entropy"
        )
        
        # Execute training
        model_info = await training_service.train_model(
            config=config,
            data_path="test_data.csv",
            model_name="test_model"
        )
        
        # Verify results
        assert model_info.name == "test_model"
        assert model_info.architecture == "bert"
        
        # Verify adapter interactions
        data_loader = container.resolve(DataLoaderPort)
        assert "test_data.csv" in data_loader.load_calls
        
        model_repo = container.resolve(ModelRepositoryPort)
        assert len(model_repo.save_calls) == 1
```

## Benefits of Hexagonal Architecture

### 1. Testability
- Easy to test business logic with mock adapters
- Clear separation of concerns
- Isolated unit testing

### 2. Flexibility
- Easy to swap implementations (e.g., filesystem to cloud storage)
- Support for multiple data sources and sinks
- Plugin architecture naturally fits

### 3. Maintainability
- Clear dependency direction (inward)
- Business logic independent of external concerns
- Easier to understand and modify

### 4. Extensibility
- New adapters can be added without changing core logic
- Support for different deployment environments
- Easy integration with new external services

## Best Practices

### 1. Port Design
- Keep ports focused and cohesive
- Use protocols for better type checking
- Define clear contracts and error handling

### 2. Adapter Implementation
- Handle all error cases gracefully
- Implement proper logging and monitoring
- Follow adapter-specific best practices

### 3. Domain Logic
- Keep domain classes pure (no external dependencies)
- Use value objects and entities appropriately
- Implement business rules clearly

### 4. Dependency Injection
- Use constructor injection
- Register dependencies at application startup
- Keep container configuration centralized

### 5. Testing
- Create comprehensive mock adapters
- Test ports and adapters separately
- Write integration tests for complete workflows

## Conclusion

Hexagonal architecture provides a solid foundation for the k-bert system, enabling:

- **Clean Separation**: Business logic isolated from external concerns
- **Easy Testing**: Mock adapters simplify testing
- **Flexible Deployment**: Multiple adapter implementations
- **Future-Proof Design**: Easy to extend and modify

This architectural approach ensures that k-bert remains maintainable and adaptable as requirements evolve and new technologies emerge.