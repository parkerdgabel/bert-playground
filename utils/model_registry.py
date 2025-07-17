"""MLflow Model Registry utilities for managing model lifecycle."""

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import mlflow
import mlx.core as mx
from loguru import logger
from mlflow.entities.model_registry import ModelVersion
from mlflow.tracking import MlflowClient

from training.config import TrainingConfig


class ModelRegistry:
    """Manages MLflow Model Registry operations."""
    
    def __init__(self, registry_uri: Optional[str] = None):
        """Initialize Model Registry client.
        
        Args:
            registry_uri: Optional URI for model registry. Uses default if None.
        """
        self.client = MlflowClient(registry_uri=registry_uri)
        if registry_uri:
            mlflow.set_registry_uri(registry_uri)
    
    def register_model(
        self,
        model_uri: str,
        name: str,
        tags: Optional[Dict[str, str]] = None,
        description: Optional[str] = None,
        await_registration: bool = True,
    ) -> ModelVersion:
        """Register a model in the Model Registry.
        
        Args:
            model_uri: URI of the model to register (e.g., runs:/<run_id>/model)
            name: Name to register the model under
            tags: Optional tags to add to the model version
            description: Optional description for the model version
            await_registration: Whether to wait for registration to complete
            
        Returns:
            Registered ModelVersion object
        """
        logger.info(f"Registering model {name} from {model_uri}")
        
        # Register the model
        model_version = mlflow.register_model(
            model_uri=model_uri,
            name=name,
            await_registration_for=300 if await_registration else 0,
        )
        
        # Add tags if provided
        if tags:
            for key, value in tags.items():
                self.client.set_model_version_tag(
                    name=name,
                    version=model_version.version,
                    key=key,
                    value=value,
                )
        
        # Add description if provided
        if description:
            self.client.update_model_version(
                name=name,
                version=model_version.version,
                description=description,
            )
        
        logger.info(
            f"Successfully registered model {name} version {model_version.version}"
        )
        return model_version
    
    def transition_model_stage(
        self,
        name: str,
        version: Union[int, str],
        stage: str,
        archive_existing: bool = True,
    ) -> ModelVersion:
        """Transition a model version to a new stage.
        
        Args:
            name: Name of the registered model
            version: Version number or "latest"
            stage: Target stage (Staging, Production, Archived, None)
            archive_existing: Whether to archive existing models in target stage
            
        Returns:
            Updated ModelVersion object
        """
        # Resolve version if "latest"
        if version == "latest":
            versions = self.client.search_model_versions(f"name='{name}'")
            if not versions:
                raise ValueError(f"No versions found for model {name}")
            version = max(v.version for v in versions)
        
        logger.info(f"Transitioning {name} v{version} to {stage}")
        
        # Archive existing models in target stage if requested
        if archive_existing and stage in ["Staging", "Production"]:
            existing = self.client.search_model_versions(
                f"name='{name}' and current_stage='{stage}'"
            )
            for model in existing:
                logger.info(
                    f"Archiving {name} v{model.version} from {stage}"
                )
                self.client.transition_model_version_stage(
                    name=name,
                    version=model.version,
                    stage="Archived",
                    archive_existing_versions=False,
                )
        
        # Transition the model
        model_version = self.client.transition_model_version_stage(
            name=name,
            version=version,
            stage=stage,
            archive_existing_versions=False,
        )
        
        logger.info(
            f"Successfully transitioned {name} v{version} to {stage}"
        )
        return model_version
    
    def set_model_alias(
        self,
        name: str,
        alias: str,
        version: Union[int, str],
    ) -> None:
        """Set an alias for a model version.
        
        Args:
            name: Name of the registered model
            alias: Alias to set (e.g., "champion", "challenger")
            version: Version number or "latest"
        """
        # Resolve version if "latest"
        if version == "latest":
            versions = self.client.search_model_versions(f"name='{name}'")
            if not versions:
                raise ValueError(f"No versions found for model {name}")
            version = max(v.version for v in versions)
        
        logger.info(f"Setting alias '{alias}' for {name} v{version}")
        
        try:
            self.client.set_registered_model_alias(
                name=name,
                alias=alias,
                version=version,
            )
            logger.info(f"Successfully set alias '{alias}'")
        except Exception as e:
            logger.warning(f"Could not set alias (may not be supported): {e}")
    
    def get_model_by_alias(
        self,
        name: str,
        alias: str,
    ) -> Optional[ModelVersion]:
        """Get a model version by its alias.
        
        Args:
            name: Name of the registered model
            alias: Alias to look up
            
        Returns:
            ModelVersion if found, None otherwise
        """
        try:
            model_version = self.client.get_model_version_by_alias(
                name=name,
                alias=alias,
            )
            return model_version
        except Exception as e:
            logger.debug(f"Could not get model by alias: {e}")
            return None
    
    def compare_models(
        self,
        name: str,
        version1: Union[int, str],
        version2: Union[int, str],
        metrics: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Compare two model versions.
        
        Args:
            name: Name of the registered model
            version1: First version to compare
            version2: Second version to compare
            metrics: Optional list of metrics to compare
            
        Returns:
            Comparison results dictionary
        """
        # Resolve versions
        versions = []
        for v in [version1, version2]:
            if v == "latest":
                all_versions = self.client.search_model_versions(f"name='{name}'")
                if not all_versions:
                    raise ValueError(f"No versions found for model {name}")
                v = max(ver.version for ver in all_versions)
            versions.append(v)
        
        version1, version2 = versions
        
        logger.info(f"Comparing {name} v{version1} vs v{version2}")
        
        # Get model versions
        mv1 = self.client.get_model_version(name=name, version=version1)
        mv2 = self.client.get_model_version(name=name, version=version2)
        
        # Get run information
        run1 = mlflow.get_run(mv1.run_id)
        run2 = mlflow.get_run(mv2.run_id)
        
        # Build comparison
        comparison = {
            "model_name": name,
            "version1": {
                "version": version1,
                "run_id": mv1.run_id,
                "stage": mv1.current_stage,
                "created": mv1.creation_timestamp,
                "metrics": run1.data.metrics,
                "params": run1.data.params,
                "tags": mv1.tags,
            },
            "version2": {
                "version": version2,
                "run_id": mv2.run_id,
                "stage": mv2.current_stage,
                "created": mv2.creation_timestamp,
                "metrics": run2.data.metrics,
                "params": run2.data.params,
                "tags": mv2.tags,
            },
            "differences": {},
        }
        
        # Compare specific metrics if requested
        if metrics:
            for metric in metrics:
                val1 = run1.data.metrics.get(metric)
                val2 = run2.data.metrics.get(metric)
                if val1 is not None and val2 is not None:
                    diff = val2 - val1
                    pct_change = (diff / val1 * 100) if val1 != 0 else 0
                    comparison["differences"][metric] = {
                        "v1": val1,
                        "v2": val2,
                        "diff": diff,
                        "pct_change": pct_change,
                        "improved": diff > 0 if "loss" not in metric else diff < 0,
                    }
        
        return comparison
    
    def list_models(
        self,
        filter_string: Optional[str] = None,
        order_by: Optional[List[str]] = None,
        page_size: int = 100,
    ) -> List[Dict[str, Any]]:
        """List registered models.
        
        Args:
            filter_string: Optional filter (e.g., "name LIKE 'bert%'")
            order_by: Optional ordering (e.g., ["name ASC"])
            page_size: Number of results per page
            
        Returns:
            List of model information dictionaries
        """
        models = []
        page_token = ""
        
        while True:
            result = self.client.search_registered_models(
                filter_string=filter_string,
                max_results=page_size,
                order_by=order_by or [],
                page_token=page_token,
            )
            
            for model in result:
                # Get latest version info
                versions = self.client.search_model_versions(
                    f"name='{model.name}'"
                )
                latest_version = (
                    max(versions, key=lambda v: v.version) if versions else None
                )
                
                model_info = {
                    "name": model.name,
                    "creation_time": model.creation_timestamp,
                    "last_updated": model.last_updated_timestamp,
                    "description": model.description,
                    "tags": model.tags,
                    "latest_version": (
                        {
                            "version": latest_version.version,
                            "stage": latest_version.current_stage,
                            "run_id": latest_version.run_id,
                        }
                        if latest_version
                        else None
                    ),
                    "total_versions": len(versions),
                }
                models.append(model_info)
            
            if not result.token:
                break
            page_token = result.token
        
        return models
    
    def get_model_history(
        self,
        name: str,
        include_metrics: bool = True,
    ) -> List[Dict[str, Any]]:
        """Get version history for a model.
        
        Args:
            name: Name of the registered model
            include_metrics: Whether to include run metrics
            
        Returns:
            List of version information dictionaries
        """
        versions = self.client.search_model_versions(
            f"name='{name}'",
            order_by=["version DESC"],
        )
        
        history = []
        for version in versions:
            version_info = {
                "version": version.version,
                "stage": version.current_stage,
                "created": version.creation_timestamp,
                "run_id": version.run_id,
                "description": version.description,
                "tags": version.tags,
                "status": version.status,
                "status_message": version.status_message,
            }
            
            if include_metrics:
                try:
                    run = mlflow.get_run(version.run_id)
                    version_info["metrics"] = run.data.metrics
                    version_info["params"] = run.data.params
                except Exception as e:
                    logger.warning(
                        f"Could not fetch metrics for v{version.version}: {e}"
                    )
            
            history.append(version_info)
        
        return history
    
    def delete_model_version(
        self,
        name: str,
        version: Union[int, str],
    ) -> None:
        """Delete a model version.
        
        Args:
            name: Name of the registered model
            version: Version to delete
        """
        # Resolve version if "latest"
        if version == "latest":
            versions = self.client.search_model_versions(f"name='{name}'")
            if not versions:
                raise ValueError(f"No versions found for model {name}")
            version = max(v.version for v in versions)
        
        logger.warning(f"Deleting {name} v{version}")
        self.client.delete_model_version(name=name, version=version)
        logger.info(f"Successfully deleted {name} v{version}")
    
    def delete_registered_model(self, name: str) -> None:
        """Delete a registered model and all its versions.
        
        Args:
            name: Name of the model to delete
        """
        logger.warning(f"Deleting registered model {name}")
        self.client.delete_registered_model(name=name)
        logger.info(f"Successfully deleted model {name}")


def register_mlx_model(
    run_id: str,
    model_path: str,
    model_name: str,
    config: TrainingConfig,
    metrics: Dict[str, float],
    stage: Optional[str] = None,
    await_registration: bool = True,
) -> ModelVersion:
    """Convenience function to register an MLX model with metadata.
    
    Args:
        run_id: MLflow run ID containing the model
        model_path: Path to model artifacts within the run
        model_name: Name to register the model under
        config: Training configuration used
        metrics: Model performance metrics
        stage: Optional stage to transition to after registration
        await_registration: Whether to wait for registration
        
    Returns:
        Registered ModelVersion
    """
    registry = ModelRegistry()
    
    # Build model URI
    model_uri = f"runs:/{run_id}/{model_path}"
    
    # Prepare tags
    tags = {
        "framework": "mlx",
        "model_type": getattr(config, 'model_name', 'modernbert'),
        "dataset": getattr(config, 'experiment_name', 'unknown'),
        "created_by": "mlx_bert_cli",
        "timestamp": datetime.now().isoformat(),
    }
    
    # Add key metrics to tags
    for metric_name in ["best_val_accuracy", "best_val_loss", "final_train_loss"]:
        if metric_name in metrics:
            tags[metric_name] = str(metrics[metric_name])
    
    # Build description
    description = f"""
MLX {getattr(config, 'model_name', 'ModernBERT')} model trained on {getattr(config, 'experiment_name', 'dataset')}.
Best validation accuracy: {metrics.get('best_val_accuracy', 'N/A')}
Training epochs: {getattr(config, 'epochs', 'N/A')}
Batch size: {getattr(config, 'batch_size', 'N/A')}
Learning rate: {getattr(config, 'learning_rate', 'N/A')}
"""
    
    # Register the model
    model_version = registry.register_model(
        model_uri=model_uri,
        name=model_name,
        tags=tags,
        description=description.strip(),
        await_registration=await_registration,
    )
    
    # Transition to stage if requested
    if stage:
        registry.transition_model_stage(
            name=model_name,
            version=model_version.version,
            stage=stage,
            archive_existing=True,
        )
    
    return model_version