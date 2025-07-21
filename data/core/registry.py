"""Dataset registry for managing multiple Kaggle competitions.

This module provides a centralized registry for managing multiple competition
datasets, with automatic discovery, caching, and optimization recommendations.
"""

import json
from pathlib import Path
from typing import Any

from loguru import logger

from .base import CompetitionType, DatasetSpec, KaggleDataset
from .metadata import CompetitionMetadata, DatasetAnalyzer


class DatasetRegistry:
    """Central registry for managing Kaggle competition datasets.

    This class provides a unified interface for discovering, registering,
    and accessing multiple competition datasets with automatic optimization.
    """

    def __init__(self, registry_dir: str | Path | None = None):
        """Initialize the dataset registry.

        Args:
            registry_dir: Directory to store registry metadata and cache
        """
        self.registry_dir = (
            Path(registry_dir) if registry_dir else Path("data/registry")
        )
        self.registry_dir.mkdir(parents=True, exist_ok=True)

        # Registry state
        self._competitions: dict[str, CompetitionMetadata] = {}
        self._dataset_specs: dict[str, DatasetSpec] = {}
        self._dataset_classes: dict[str, type[KaggleDataset]] = {}
        self._analyzer = DatasetAnalyzer()

        # Load existing registry
        self._load_registry()

        logger.info(
            f"Initialized DatasetRegistry with {len(self._competitions)} competitions"
        )

    def register_competition(
        self,
        data_path: str | Path,
        competition_name: str | None = None,
        target_column: str | None = None,
        dataset_class: type[KaggleDataset] | None = None,
        force_reanalyze: bool = False,
    ) -> str:
        """Register a competition dataset.

        Args:
            data_path: Path to competition data directory
            competition_name: Optional competition name (auto-detected if None)
            target_column: Optional target column name (auto-detected if None)
            dataset_class: Optional custom dataset class
            force_reanalyze: Whether to force re-analysis of existing competition

        Returns:
            Competition name (normalized)
        """
        data_path = Path(data_path)

        # Determine competition name
        if competition_name is None:
            competition_name = data_path.stem if data_path.is_file() else data_path.name

        # Normalize competition name
        normalized_name = self._normalize_competition_name(competition_name)

        # Check if already registered
        if normalized_name in self._competitions and not force_reanalyze:
            logger.info(f"Competition {normalized_name} already registered")
            return normalized_name

        logger.info(f"Registering competition: {normalized_name}")

        try:
            # Analyze competition
            metadata, spec = self._analyzer.analyze_competition(
                data_path=data_path,
                competition_name=normalized_name,
                target_column=target_column,
            )

            # Store in registry
            self._competitions[normalized_name] = metadata
            self._dataset_specs[normalized_name] = spec

            # Register dataset class
            if dataset_class:
                self._dataset_classes[normalized_name] = dataset_class
            else:
                # Use default implementation based on competition type
                self._dataset_classes[normalized_name] = (
                    self._get_default_dataset_class(spec.competition_type)
                )

            # Save registry
            self._save_registry()

            logger.info(
                f"Registered {normalized_name}: {spec.competition_type.value} "
                f"({spec.num_samples} samples, {spec.num_features} features)"
            )

            return normalized_name

        except Exception as e:
            logger.error(f"Failed to register competition {normalized_name}: {e}")
            raise

    def get_competition(self, competition_name: str) -> CompetitionMetadata | None:
        """Get competition metadata by name.

        Args:
            competition_name: Competition name

        Returns:
            CompetitionMetadata or None if not found
        """
        normalized_name = self._normalize_competition_name(competition_name)
        return self._competitions.get(normalized_name)

    def get_dataset_spec(self, competition_name: str) -> DatasetSpec | None:
        """Get dataset specification by competition name.

        Args:
            competition_name: Competition name

        Returns:
            DatasetSpec or None if not found
        """
        normalized_name = self._normalize_competition_name(competition_name)
        return self._dataset_specs.get(normalized_name)

    def create_dataset(
        self,
        competition_name: str,
        split: str = "train",
        **kwargs,
    ) -> KaggleDataset:
        """Create a dataset instance for a registered competition.

        Args:
            competition_name: Competition name
            split: Dataset split ("train", "validation", "test")
            **kwargs: Additional arguments for dataset constructor

        Returns:
            KaggleDataset instance

        Raises:
            ValueError: If competition not registered
        """
        normalized_name = self._normalize_competition_name(competition_name)

        if normalized_name not in self._competitions:
            raise ValueError(f"Competition {normalized_name} not registered")

        spec = self._dataset_specs[normalized_name]
        dataset_class = self._dataset_classes[normalized_name]

        # Create dataset instance
        dataset = dataset_class(
            spec=spec,
            split=split,
            **kwargs,
        )

        logger.info(f"Created dataset for {normalized_name} ({split} split)")
        return dataset

    def list_competitions(self) -> list[str]:
        """List all registered competition names.

        Returns:
            List of competition names
        """
        return list(self._competitions.keys())

    def get_competitions_by_type(self, competition_type: CompetitionType) -> list[str]:
        """Get competitions filtered by type.

        Args:
            competition_type: Competition type to filter by

        Returns:
            List of competition names matching the type
        """
        return [
            name
            for name, metadata in self._competitions.items()
            if metadata.competition_type == competition_type
        ]

    def get_registry_stats(self) -> dict[str, Any]:
        """Get statistics about the registered competitions.

        Returns:
            Dictionary containing registry statistics
        """
        if not self._competitions:
            return {"total_competitions": 0}

        # Count by type
        type_counts = {}
        total_samples = 0
        total_features = 0

        for metadata in self._competitions.values():
            comp_type = metadata.competition_type.value
            type_counts[comp_type] = type_counts.get(comp_type, 0) + 1

            # Get corresponding spec for sample/feature counts
            spec = self._dataset_specs.get(metadata.competition_name)
            if spec:
                total_samples += spec.num_samples
                total_features += spec.num_features

        return {
            "total_competitions": len(self._competitions),
            "competitions_by_type": type_counts,
            "total_samples": total_samples,
            "total_features": total_features,
            "avg_samples_per_competition": total_samples / len(self._competitions)
            if self._competitions
            else 0,
            "avg_features_per_competition": total_features / len(self._competitions)
            if self._competitions
            else 0,
        }

    def remove_competition(self, competition_name: str) -> bool:
        """Remove a competition from the registry.

        Args:
            competition_name: Competition name to remove

        Returns:
            True if removed, False if not found
        """
        normalized_name = self._normalize_competition_name(competition_name)

        if normalized_name not in self._competitions:
            return False

        # Remove from all registries
        del self._competitions[normalized_name]
        del self._dataset_specs[normalized_name]
        del self._dataset_classes[normalized_name]

        # Save updated registry
        self._save_registry()

        logger.info(f"Removed competition {normalized_name} from registry")
        return True

    def update_competition_metadata(
        self,
        competition_name: str,
        **updates,
    ) -> bool:
        """Update competition metadata.

        Args:
            competition_name: Competition name
            **updates: Metadata fields to update

        Returns:
            True if updated, False if not found
        """
        normalized_name = self._normalize_competition_name(competition_name)

        if normalized_name not in self._competitions:
            return False

        metadata = self._competitions[normalized_name]

        # Update metadata fields
        for key, value in updates.items():
            if hasattr(metadata, key):
                setattr(metadata, key, value)

        # Save updated registry
        self._save_registry()

        logger.info(f"Updated metadata for competition {normalized_name}")
        return True

    def discover_competitions(self, search_dir: str | Path) -> list[str]:
        """Discover and register competitions in a directory.

        Args:
            search_dir: Directory to search for competition data

        Returns:
            List of registered competition names
        """
        search_dir = Path(search_dir)
        registered_competitions = []

        if not search_dir.exists():
            logger.warning(f"Search directory {search_dir} does not exist")
            return registered_competitions

        logger.info(f"Discovering competitions in {search_dir}")

        # Look for CSV files and directories with CSV files
        for path in search_dir.iterdir():
            try:
                if path.is_file() and path.suffix.lower() == ".csv":
                    # Single CSV file
                    competition_name = self.register_competition(path)
                    registered_competitions.append(competition_name)

                elif path.is_dir():
                    # Directory - check if it contains CSV files
                    csv_files = list(path.glob("*.csv"))
                    if csv_files:
                        competition_name = self.register_competition(path)
                        registered_competitions.append(competition_name)

            except Exception as e:
                logger.warning(f"Failed to register {path}: {e}")
                continue

        logger.info(f"Discovered {len(registered_competitions)} competitions")
        return registered_competitions

    def export_registry(self, export_path: str | Path) -> None:
        """Export registry to a JSON file.

        Args:
            export_path: Path to export file
        """
        export_data = {
            "competitions": {
                name: metadata.to_dict()
                for name, metadata in self._competitions.items()
            },
            "dataset_specs": {
                name: {
                    "competition_name": spec.competition_name,
                    "dataset_path": str(spec.dataset_path),
                    "competition_type": spec.competition_type.value,
                    "num_samples": spec.num_samples,
                    "num_features": spec.num_features,
                    "target_column": spec.target_column,
                    "text_columns": spec.text_columns,
                    "categorical_columns": spec.categorical_columns,
                    "numerical_columns": spec.numerical_columns,
                    "num_classes": spec.num_classes,
                    "class_distribution": spec.class_distribution,
                    "is_balanced": spec.is_balanced,
                }
                for name, spec in self._dataset_specs.items()
            },
        }

        with open(export_path, "w") as f:
            json.dump(export_data, f, indent=2)

        logger.info(f"Exported registry to {export_path}")

    def _normalize_competition_name(self, name: str) -> str:
        """Normalize competition name to a standard format.

        Args:
            name: Original competition name

        Returns:
            Normalized name
        """
        # Remove special characters and convert to lowercase
        import re

        normalized = re.sub(r"[^a-zA-Z0-9_-]", "_", name.lower())
        normalized = re.sub(r"_+", "_", normalized)  # Collapse multiple underscores
        return normalized.strip("_")

    def _get_default_dataset_class(
        self, competition_type: CompetitionType
    ) -> type[KaggleDataset]:
        """Get default dataset class for a competition type.

        Args:
            competition_type: Competition type

        Returns:
            Dataset class
        """
        # For now, import and return the generic implementation
        # This will be replaced with specific implementations
        from ..kaggle import KaggleCompetitionDataset

        return KaggleCompetitionDataset

    def _load_registry(self) -> None:
        """Load registry from disk."""
        registry_file = self.registry_dir / "registry.json"

        if not registry_file.exists():
            return

        try:
            with open(registry_file) as f:
                data = json.load(f)

            # Load competitions
            for name, metadata_dict in data.get("competitions", {}).items():
                self._competitions[name] = CompetitionMetadata.from_dict(metadata_dict)

            # Load dataset specs (simplified loading)
            for name, spec_dict in data.get("dataset_specs", {}).items():
                spec_dict["competition_type"] = CompetitionType(
                    spec_dict["competition_type"]
                )
                self._dataset_specs[name] = DatasetSpec(**spec_dict)

            # Set default dataset classes
            for name in self._competitions:
                metadata = self._competitions[name]
                self._dataset_classes[name] = self._get_default_dataset_class(
                    metadata.competition_type
                )

            logger.info(f"Loaded {len(self._competitions)} competitions from registry")

        except Exception as e:
            logger.warning(f"Failed to load registry: {e}")
            self._competitions = {}
            self._dataset_specs = {}
            self._dataset_classes = {}

    def _save_registry(self) -> None:
        """Save registry to disk."""
        registry_file = self.registry_dir / "registry.json"

        try:
            data = {
                "competitions": {
                    name: metadata.to_dict()
                    for name, metadata in self._competitions.items()
                },
                "dataset_specs": {
                    name: {
                        "competition_name": spec.competition_name,
                        "dataset_path": str(spec.dataset_path),
                        "competition_type": spec.competition_type.value,
                        "num_samples": spec.num_samples,
                        "num_features": spec.num_features,
                        "target_column": spec.target_column,
                        "text_columns": spec.text_columns,
                        "categorical_columns": spec.categorical_columns,
                        "numerical_columns": spec.numerical_columns,
                        "num_classes": spec.num_classes,
                        "class_distribution": spec.class_distribution,
                        "is_balanced": spec.is_balanced,
                        "recommended_batch_size": spec.recommended_batch_size,
                        "recommended_max_length": spec.recommended_max_length,
                        "use_attention_mask": spec.use_attention_mask,
                        "enable_caching": spec.enable_caching,
                        "use_unified_memory": spec.use_unified_memory,
                        "prefetch_size": spec.prefetch_size,
                        "num_workers": spec.num_workers,
                        "text_template": spec.text_template,
                        "tokenizer_backend": spec.tokenizer_backend,
                    }
                    for name, spec in self._dataset_specs.items()
                },
            }

            with open(registry_file, "w") as f:
                json.dump(data, f, indent=2)

        except Exception as e:
            logger.error(f"Failed to save registry: {e}")
