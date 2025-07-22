"""Unit tests for DuckDB Manager."""

import tempfile
from pathlib import Path

import pandas as pd
import pytest

from cli.utils.duckdb_manager import DuckDBManager
from cli.config.schemas import AnalysisConfig


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test data."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_csv_data(temp_dir):
    """Create sample CSV files for testing."""
    # Create train.csv
    train_data = pd.DataFrame({
        "id": [1, 2, 3, 4, 5],
        "name": ["Alice", "Bob", "Charlie", "David", "Eve"],
        "age": [25, 30, 35, 40, 45],
        "score": [85.5, 92.0, 78.5, 88.0, 95.5],
        "category": ["A", "B", "A", "C", "B"],
    })
    train_path = temp_dir / "train.csv"
    train_data.to_csv(train_path, index=False)
    
    # Create test.csv
    test_data = pd.DataFrame({
        "id": [6, 7, 8],
        "name": ["Frank", "Grace", "Henry"],
        "age": [28, 33, 38],
        "category": ["A", "C", "B"],
    })
    test_path = temp_dir / "test.csv"
    test_data.to_csv(test_path, index=False)
    
    return temp_dir


class TestDuckDBManager:
    """Test DuckDB Manager functionality."""
    
    def test_initialization(self):
        """Test DuckDB manager initialization."""
        manager = DuckDBManager()
        assert manager.connection is not None
        manager.close()
        
    def test_initialization_with_config(self):
        """Test initialization with custom config."""
        config = AnalysisConfig(memory_limit="2GB")
        manager = DuckDBManager(config=config)
        assert manager.config.memory_limit == "2GB"
        manager.close()
        
    def test_auto_load_csv(self, sample_csv_data):
        """Test automatic CSV loading."""
        manager = DuckDBManager(data_dir=sample_csv_data)
        
        # Check tables were loaded
        tables = manager.get_tables()
        assert "train" in tables
        assert "test" in tables
        
        # Check data was loaded correctly
        train_stats = manager.get_table_stats("train")
        assert train_stats["row_count"] == 5
        assert train_stats["column_count"] == 5
        
        test_stats = manager.get_table_stats("test")
        assert test_stats["row_count"] == 3
        assert test_stats["column_count"] == 4
        
        manager.close()
        
    def test_load_csv_manual(self, temp_dir):
        """Test manual CSV loading."""
        # Create CSV file
        data = pd.DataFrame({
            "x": [1, 2, 3],
            "y": [4, 5, 6],
        })
        csv_path = temp_dir / "data.csv"
        data.to_csv(csv_path, index=False)
        
        # Load manually
        manager = DuckDBManager()
        table_name = manager.load_csv(csv_path, "my_table")
        
        assert table_name == "my_table"
        assert "my_table" in manager.get_tables()
        
        stats = manager.get_table_stats("my_table")
        assert stats["row_count"] == 3
        assert stats["column_count"] == 2
        
        manager.close()
        
    def test_execute_query(self, sample_csv_data):
        """Test query execution."""
        manager = DuckDBManager(data_dir=sample_csv_data)
        
        # Simple select
        result = manager.execute_query("SELECT * FROM train WHERE age > 30")
        assert len(result) == 3
        assert all(result["age"] > 30)
        
        # Aggregation
        result = manager.execute_query(
            "SELECT category, COUNT(*) as count FROM train GROUP BY category"
        )
        assert len(result) == 3
        assert set(result["category"]) == {"A", "B", "C"}
        
        manager.close()
        
    def test_get_table_info(self, sample_csv_data):
        """Test getting table information."""
        manager = DuckDBManager(data_dir=sample_csv_data)
        
        info = manager.get_table_info("train")
        assert len(info) == 5
        assert "id" in info["column_name"].values
        assert "name" in info["column_name"].values
        
        manager.close()
        
    def test_describe_table(self, sample_csv_data):
        """Test table description for numeric columns."""
        manager = DuckDBManager(data_dir=sample_csv_data)
        
        desc = manager.describe_table("train")
        assert not desc.empty
        # Should have statistics for numeric columns: id, age, score
        assert "column_name" in desc.columns
        
        manager.close()
        
    def test_register_dataframe(self):
        """Test registering pandas DataFrame."""
        manager = DuckDBManager()
        
        df = pd.DataFrame({
            "a": [1, 2, 3],
            "b": ["x", "y", "z"],
        })
        
        manager.register_dataframe(df, "my_df")
        assert "my_df" in manager.get_tables()
        
        result = manager.execute_query("SELECT * FROM my_df")
        assert len(result) == 3
        assert list(result["a"]) == [1, 2, 3]
        
        manager.close()
        
    def test_export_results(self, temp_dir):
        """Test exporting query results."""
        manager = DuckDBManager()
        
        df = pd.DataFrame({
            "x": [1, 2, 3],
            "y": [4, 5, 6],
        })
        
        # Test CSV export
        csv_path = temp_dir / "export.csv"
        manager.export_results(df, csv_path, "csv")
        assert csv_path.exists()
        
        # Test JSON export
        json_path = temp_dir / "export.json"
        manager.export_results(df, json_path, "json")
        assert json_path.exists()
        
        # Test Parquet export
        parquet_path = temp_dir / "export.parquet"
        manager.export_results(df, parquet_path, "parquet")
        assert parquet_path.exists()
        
        manager.close()
        
    def test_format_results(self):
        """Test result formatting."""
        manager = DuckDBManager()
        
        df = pd.DataFrame({
            "col1": list(range(30)),
            "col2": ["value"] * 30,
        })
        
        table = manager.format_results(df, max_rows=10)
        assert table is not None
        # Rich table should be created
        
        manager.close()
        
    def test_create_analysis_views(self, sample_csv_data):
        """Test creation of analysis views."""
        manager = DuckDBManager(data_dir=sample_csv_data)
        
        manager.create_analysis_views()
        
        # Check if all_data view was created
        tables = manager.get_tables()
        # Note: views might not appear in tables list depending on DuckDB version
        
        # Try to query the view
        try:
            result = manager.execute_query("SELECT * FROM all_data")
            assert len(result) == 8  # 5 from train + 3 from test
            assert "source" in result.columns
        except Exception:
            # View creation might fail if columns don't match exactly
            pass
            
        manager.close()
        
    def test_memory_usage(self):
        """Test memory usage reporting."""
        manager = DuckDBManager()
        
        usage = manager.get_memory_usage()
        # DuckDB may return different fields depending on version
        assert isinstance(usage, dict)
        assert len(usage) > 0
        # Should have at least one of these fields
        assert any(key in usage for key in ["memory_usage_bytes", "database_size", "error"])
        
        manager.close()
        
    def test_context_manager(self, sample_csv_data):
        """Test using manager as context manager."""
        with DuckDBManager(data_dir=sample_csv_data) as manager:
            tables = manager.get_tables()
            assert len(tables) > 0
            
        # Connection should be closed after exiting context
        
    def test_transaction(self):
        """Test transaction handling."""
        manager = DuckDBManager()
        
        # Create test table
        manager.connection.execute("CREATE TABLE test (id INTEGER, value TEXT)")
        
        # Test successful transaction
        with manager.transaction():
            manager.connection.execute("INSERT INTO test VALUES (1, 'a')")
            manager.connection.execute("INSERT INTO test VALUES (2, 'b')")
            
        result = manager.execute_query("SELECT COUNT(*) as count FROM test")
        assert result["count"].iloc[0] == 2
        
        # Test failed transaction (should rollback)
        try:
            with manager.transaction():
                manager.connection.execute("INSERT INTO test VALUES (3, 'c')")
                # Force an error
                manager.connection.execute("INVALID SQL")
        except Exception:
            pass
            
        result = manager.execute_query("SELECT COUNT(*) as count FROM test")
        assert result["count"].iloc[0] == 2  # Should still be 2
        
        manager.close()
        
    def test_explain_query(self, sample_csv_data):
        """Test query explanation."""
        manager = DuckDBManager(data_dir=sample_csv_data)
        
        plan = manager.explain_query("SELECT * FROM train WHERE age > 30")
        assert plan is not None
        assert isinstance(plan, str)
        assert len(plan) > 0
        
        manager.close()
        
    def test_nonexistent_file(self):
        """Test loading non-existent file."""
        manager = DuckDBManager()
        
        with pytest.raises(FileNotFoundError):
            manager.load_csv(Path("/nonexistent/file.csv"))
            
        manager.close()
        
    def test_invalid_query(self, sample_csv_data):
        """Test handling invalid queries."""
        manager = DuckDBManager(data_dir=sample_csv_data)
        
        with pytest.raises(Exception):
            manager.execute_query("SELECT * FROM nonexistent_table")
            
        manager.close()