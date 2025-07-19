"""Unit tests for dataset registry."""

import json
import pytest
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

import pandas as pd

from data.core.base import CompetitionType, DatasetSpec
from data.core.metadata import CompetitionMetadata
from data.core.registry import DatasetRegistry


class TestDatasetRegistry:
    """Test DatasetRegistry class."""
    
    @pytest.fixture
    def registry(self, test_data_dir):
        """Create DatasetRegistry instance."""
        registry_dir = test_data_dir / "registry"
        registry_dir.mkdir()
        return DatasetRegistry(registry_dir)
        
    @pytest.fixture
    def sample_metadata(self):
        """Create sample metadata for testing."""
        return CompetitionMetadata(
            competition_name="test_competition",
            competition_type=CompetitionType.BINARY_CLASSIFICATION,
            auto_detected=True,
            confidence_score=0.95,
            train_file="train.csv",
            test_file="test.csv",
            dataset_size_mb=50.0,
        )
        
    @pytest.fixture
    def sample_spec(self):
        """Create sample spec for testing."""
        return DatasetSpec(
            competition_name="test_competition",
            dataset_path=Path("/tmp/test"),
            competition_type=CompetitionType.BINARY_CLASSIFICATION,
            num_samples=1000,
            num_features=10,
            target_column="target",
        )
        
    def test_registry_creation(self, registry):
        """Test registry creation."""
        assert registry is not None
        assert registry.registry_dir.exists()
        assert hasattr(registry, '_competitions')
        assert hasattr(registry, '_specs')
        assert hasattr(registry, '_metadata')
        
    def test_registry_directory_structure(self, registry):
        """Test registry directory structure creation."""
        expected_dirs = ['competitions', 'metadata', 'specs', 'cache']
        
        for dir_name in expected_dirs:
            expected_dir = registry.registry_dir / dir_name
            assert expected_dir.exists()
            assert expected_dir.is_dir()
            
    def test_register_competition_new(self, registry, sample_metadata, sample_spec):
        """Test registering a new competition."""
        success = registry.register_competition(
            competition_name="test_competition",
            metadata=sample_metadata,
            spec=sample_spec,
        )
        
        assert success == True
        assert "test_competition" in registry._competitions
        assert "test_competition" in registry._metadata
        assert "test_competition" in registry._specs
        
    def test_register_competition_duplicate(self, registry, sample_metadata, sample_spec):
        """Test registering duplicate competition."""
        # Register first time
        registry.register_competition("test_comp", sample_metadata, sample_spec)
        
        # Try to register again
        success = registry.register_competition("test_comp", sample_metadata, sample_spec)
        
        assert success == False  # Should fail for duplicate
        
    def test_register_competition_force_overwrite(self, registry, sample_metadata, sample_spec):
        """Test force overwriting existing competition."""
        # Register first time
        registry.register_competition("test_comp", sample_metadata, sample_spec)
        
        # Overwrite with force=True
        new_metadata = CompetitionMetadata(
            competition_name="test_comp",
            competition_type=CompetitionType.REGRESSION,
            confidence_score=0.8,
        )
        
        success = registry.register_competition(
            "test_comp", new_metadata, sample_spec, force=True
        )
        
        assert success == True
        assert registry._metadata["test_comp"].competition_type == CompetitionType.REGRESSION
        
    def test_get_competition_info_exists(self, registry, sample_metadata, sample_spec):
        """Test getting competition info for existing competition."""
        registry.register_competition("test_comp", sample_metadata, sample_spec)
        
        info = registry.get_competition_info("test_comp")
        
        assert info is not None
        assert info['name'] == "test_comp"
        assert info['type'] == "binary_classification"
        assert info['registered_at'] is not None
        assert info['metadata'] == sample_metadata
        assert info['spec'] == sample_spec
        
    def test_get_competition_info_missing(self, registry):
        """Test getting competition info for missing competition."""
        info = registry.get_competition_info("missing_comp")
        assert info is None
        
    def test_list_competitions_empty(self, registry):
        """Test listing competitions when registry is empty."""
        competitions = registry.list_competitions()
        assert competitions == []
        
    def test_list_competitions_multiple(self, registry, sample_metadata, sample_spec):
        """Test listing multiple competitions."""
        # Register multiple competitions
        for i in range(3):
            metadata = CompetitionMetadata(
                competition_name=f"comp_{i}",
                competition_type=CompetitionType.BINARY_CLASSIFICATION,
                confidence_score=0.9,
            )
            spec = DatasetSpec(
                competition_name=f"comp_{i}",
                dataset_path=Path(f"/tmp/comp_{i}"),
                competition_type=CompetitionType.BINARY_CLASSIFICATION,
                num_samples=100,
                num_features=5,
            )
            registry.register_competition(f"comp_{i}", metadata, spec)
            
        competitions = registry.list_competitions()
        
        assert len(competitions) == 3
        assert all(comp['name'].startswith('comp_') for comp in competitions)
        
    def test_list_competitions_filtered(self, registry):
        """Test listing competitions with filters."""
        # Register competitions of different types
        comp_types = [
            CompetitionType.BINARY_CLASSIFICATION,
            CompetitionType.REGRESSION,
            CompetitionType.MULTICLASS_CLASSIFICATION,
        ]
        
        for i, comp_type in enumerate(comp_types):
            metadata = CompetitionMetadata(
                competition_name=f"comp_{i}",
                competition_type=comp_type,
                confidence_score=0.9,
            )
            spec = DatasetSpec(
                competition_name=f"comp_{i}",
                dataset_path=Path(f"/tmp/comp_{i}"),
                competition_type=comp_type,
                num_samples=100,
                num_features=5,
            )
            registry.register_competition(f"comp_{i}", metadata, spec)
            
        # Filter by type
        binary_comps = registry.list_competitions(
            competition_type=CompetitionType.BINARY_CLASSIFICATION
        )
        
        assert len(binary_comps) == 1
        assert binary_comps[0]['type'] == "binary_classification"
        
    def test_remove_competition_exists(self, registry, sample_metadata, sample_spec):
        """Test removing existing competition."""
        registry.register_competition("test_comp", sample_metadata, sample_spec)
        
        # Verify it exists
        assert "test_comp" in registry._competitions
        
        success = registry.remove_competition("test_comp")
        
        assert success == True
        assert "test_comp" not in registry._competitions
        assert "test_comp" not in registry._metadata
        assert "test_comp" not in registry._specs
        
    def test_remove_competition_missing(self, registry):
        """Test removing non-existent competition."""
        success = registry.remove_competition("missing_comp")
        assert success == False
        
    def test_update_competition_exists(self, registry, sample_metadata, sample_spec):
        """Test updating existing competition."""
        registry.register_competition("test_comp", sample_metadata, sample_spec)
        
        # Update metadata
        new_metadata = CompetitionMetadata(
            competition_name="test_comp",
            competition_type=CompetitionType.REGRESSION,
            confidence_score=0.7,
        )
        
        success = registry.update_competition("test_comp", metadata=new_metadata)
        
        assert success == True
        assert registry._metadata["test_comp"].competition_type == CompetitionType.REGRESSION
        assert registry._metadata["test_comp"].confidence_score == 0.7
        
    def test_update_competition_missing(self, registry):
        """Test updating non-existent competition."""
        new_metadata = CompetitionMetadata(
            competition_name="missing",
            competition_type=CompetitionType.REGRESSION,
        )
        
        success = registry.update_competition("missing", metadata=new_metadata)
        assert success == False
        
    def test_search_competitions_by_name(self, registry):
        """Test searching competitions by name."""
        # Register competitions with different names
        names = ["titanic_survival", "house_prices", "digit_recognizer"]
        
        for name in names:
            metadata = CompetitionMetadata(
                competition_name=name,
                competition_type=CompetitionType.BINARY_CLASSIFICATION,
            )
            spec = DatasetSpec(
                competition_name=name,
                dataset_path=Path(f"/tmp/{name}"),
                competition_type=CompetitionType.BINARY_CLASSIFICATION,
                num_samples=100,
                num_features=5,
            )
            registry.register_competition(name, metadata, spec)
            
        # Search for "titanic"
        results = registry.search_competitions(name_pattern="titanic")
        
        assert len(results) == 1
        assert results[0]['name'] == "titanic_survival"
        
    def test_search_competitions_by_type(self, registry):
        """Test searching competitions by type."""
        comp_types = [
            CompetitionType.BINARY_CLASSIFICATION,
            CompetitionType.REGRESSION,
            CompetitionType.BINARY_CLASSIFICATION,
        ]
        
        for i, comp_type in enumerate(comp_types):
            metadata = CompetitionMetadata(
                competition_name=f"comp_{i}",
                competition_type=comp_type,
            )
            spec = DatasetSpec(
                competition_name=f"comp_{i}",
                dataset_path=Path(f"/tmp/comp_{i}"),
                competition_type=comp_type,
                num_samples=100,
                num_features=5,
            )
            registry.register_competition(f"comp_{i}", metadata, spec)
            
        # Search for binary classification
        results = registry.search_competitions(
            competition_type=CompetitionType.BINARY_CLASSIFICATION
        )
        
        assert len(results) == 2
        assert all(r['type'] == "binary_classification" for r in results)
        
    def test_get_registry_statistics(self, registry):
        """Test getting registry statistics."""
        # Register competitions of different types
        comp_types = [
            CompetitionType.BINARY_CLASSIFICATION,
            CompetitionType.REGRESSION,
            CompetitionType.MULTICLASS_CLASSIFICATION,
            CompetitionType.BINARY_CLASSIFICATION,
        ]
        
        for i, comp_type in enumerate(comp_types):
            metadata = CompetitionMetadata(
                competition_name=f"comp_{i}",
                competition_type=comp_type,
            )
            spec = DatasetSpec(
                competition_name=f"comp_{i}",
                dataset_path=Path(f"/tmp/comp_{i}"),
                competition_type=comp_type,
                num_samples=100 * (i + 1),  # Different sizes
                num_features=5,
            )
            registry.register_competition(f"comp_{i}", metadata, spec)
            
        stats = registry.get_registry_statistics()
        
        assert stats['total_competitions'] == 4
        assert stats['competition_types']['binary_classification'] == 2
        assert stats['competition_types']['regression'] == 1
        assert stats['competition_types']['multiclass_classification'] == 1
        assert stats['total_samples'] == 1000  # 100+200+300+400
        
    def test_save_load_registry(self, registry, sample_metadata, sample_spec):
        """Test saving and loading registry."""
        # Register a competition
        registry.register_competition("test_comp", sample_metadata, sample_spec)
        
        # Save registry
        registry.save_registry()
        
        # Create new registry instance
        new_registry = DatasetRegistry(registry.registry_dir)
        
        # Load registry
        new_registry.load_registry()
        
        # Verify data was loaded
        assert "test_comp" in new_registry._competitions
        assert new_registry._metadata["test_comp"].competition_name == "test_comp"
        assert new_registry._specs["test_comp"].competition_name == "test_comp"
        
    def test_discover_datasets_directory(self, test_data_dir, temp_csv_files):
        """Test automatic dataset discovery from directory."""
        registry = DatasetRegistry(test_data_dir / "registry")
        
        # Use the temp CSV files created by the fixture
        data_dir = temp_csv_files['titanic_train'].parent.parent
        
        discovered = registry.discover_datasets(data_dir)
        
        assert len(discovered) >= 1  # Should find at least titanic
        assert any("titanic" in comp for comp in discovered)
        
    def test_clear_registry(self, registry, sample_metadata, sample_spec):
        """Test clearing entire registry."""
        # Register multiple competitions
        for i in range(3):
            metadata = CompetitionMetadata(
                competition_name=f"comp_{i}",
                competition_type=CompetitionType.BINARY_CLASSIFICATION,
            )
            spec = DatasetSpec(
                competition_name=f"comp_{i}",
                dataset_path=Path(f"/tmp/comp_{i}"),
                competition_type=CompetitionType.BINARY_CLASSIFICATION,
                num_samples=100,
                num_features=5,
            )
            registry.register_competition(f"comp_{i}", metadata, spec)
            
        # Verify competitions exist
        assert len(registry._competitions) == 3
        
        # Clear registry
        registry.clear_registry()
        
        # Verify all cleared
        assert len(registry._competitions) == 0
        assert len(registry._metadata) == 0
        assert len(registry._specs) == 0
        
    def test_export_import_registry(self, registry, sample_metadata, sample_spec, test_data_dir):
        """Test exporting and importing registry."""
        # Register a competition
        registry.register_competition("test_comp", sample_metadata, sample_spec)
        
        # Export registry
        export_file = test_data_dir / "registry_export.json"
        registry.export_registry(export_file)
        
        # Verify export file exists
        assert export_file.exists()
        
        # Clear and import
        registry.clear_registry()
        assert len(registry._competitions) == 0
        
        registry.import_registry(export_file)
        
        # Verify data was imported
        assert "test_comp" in registry._competitions
        assert registry._metadata["test_comp"].competition_name == "test_comp"
        
    def test_validate_registry_integrity(self, registry, sample_metadata, sample_spec):
        """Test registry integrity validation."""
        # Register valid competition
        registry.register_competition("test_comp", sample_metadata, sample_spec)
        
        # Validate integrity
        is_valid, issues = registry.validate_integrity()
        
        assert is_valid == True
        assert len(issues) == 0
        
    def test_validate_registry_with_issues(self, registry, sample_metadata, sample_spec):
        """Test registry validation with integrity issues."""
        # Register competition
        registry.register_competition("test_comp", sample_metadata, sample_spec)
        
        # Manually corrupt data to create issues
        del registry._metadata["test_comp"]  # Remove metadata but keep competition
        
        # Validate integrity
        is_valid, issues = registry.validate_integrity()
        
        assert is_valid == False
        assert len(issues) > 0
        assert any("metadata" in issue.lower() for issue in issues)
        
    def test_auto_register_from_path(self, registry, temp_csv_files):
        """Test automatic registration from dataset path."""
        titanic_dir = temp_csv_files['titanic_train'].parent
        
        success = registry.auto_register_from_path(
            dataset_path=titanic_dir,
            competition_name="auto_titanic",
        )
        
        assert success == True
        assert "auto_titanic" in registry._competitions
        
        # Verify metadata was generated
        metadata = registry._metadata["auto_titanic"]
        assert metadata.competition_name == "auto_titanic"
        assert metadata.auto_detected == True
        
    def test_get_compatible_competitions(self, registry):
        """Test finding compatible competitions."""
        # Register competitions with different properties
        comp_configs = [
            (CompetitionType.BINARY_CLASSIFICATION, 1000),
            (CompetitionType.BINARY_CLASSIFICATION, 500),
            (CompetitionType.REGRESSION, 1000),
            (CompetitionType.MULTICLASS_CLASSIFICATION, 1000),
        ]
        
        for i, (comp_type, num_samples) in enumerate(comp_configs):
            metadata = CompetitionMetadata(
                competition_name=f"comp_{i}",
                competition_type=comp_type,
            )
            spec = DatasetSpec(
                competition_name=f"comp_{i}",
                dataset_path=Path(f"/tmp/comp_{i}"),
                competition_type=comp_type,
                num_samples=num_samples,
                num_features=5,
            )
            registry.register_competition(f"comp_{i}", metadata, spec)
            
        # Find competitions compatible with binary classification
        compatible = registry.get_compatible_competitions(
            CompetitionType.BINARY_CLASSIFICATION
        )
        
        assert len(compatible) == 2  # Only binary classification competitions
        assert all(comp['type'] == "binary_classification" for comp in compatible)
        
    def test_dataset_caching_operations(self, registry, sample_metadata, sample_spec):
        """Test dataset caching operations."""
        registry.register_competition("test_comp", sample_metadata, sample_spec)
        
        # Test cache info
        cache_info = registry.get_cache_info("test_comp")
        assert cache_info is not None
        assert 'cache_dir' in cache_info
        assert 'cache_size_mb' in cache_info
        
        # Test clear cache
        success = registry.clear_competition_cache("test_comp")
        assert success == True
        
    def test_competition_usage_tracking(self, registry, sample_metadata, sample_spec):
        """Test competition usage tracking."""
        registry.register_competition("test_comp", sample_metadata, sample_spec)
        
        # Record usage
        registry.record_usage("test_comp", "training")
        registry.record_usage("test_comp", "prediction")
        
        # Get usage stats
        usage_stats = registry.get_usage_statistics("test_comp")
        
        assert usage_stats is not None
        assert usage_stats['total_usage'] == 2
        assert 'last_used' in usage_stats
        assert 'usage_types' in usage_stats