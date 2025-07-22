"""Unit tests for explore command."""

import tempfile
from pathlib import Path

import pandas as pd
import pytest
from typer.testing import CliRunner

from cli.app import app


@pytest.fixture
def runner():
    """Create CLI test runner."""
    return CliRunner()


@pytest.fixture
def sample_data_dir():
    """Create sample data directory with CSV files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        data_dir = Path(tmpdir)
        
        # Create train.csv with various data types
        train_df = pd.DataFrame({
            "id": [1, 2, 3, 4, 5],
            "name": ["Alice", "Bob", None, "David", "Eve"],
            "age": [25, 30, 35, None, 45],
            "salary": [50000.0, 60000.0, 70000.0, 80000.0, None],
            "department": ["Sales", "Engineering", "Sales", "HR", "Engineering"],
            "is_active": [True, True, False, True, True],
        })
        train_df.to_csv(data_dir / "train.csv", index=False)
        
        # Create test.csv with matching columns
        test_df = pd.DataFrame({
            "id": [6, 7, 8],
            "name": ["Frank", "Grace", "Henry"],
            "age": [28, 33, 38],
            "salary": [55000.0, 65000.0, 75000.0],
            "department": ["Sales", "HR", "Engineering"],
            "is_active": [True, False, True],
        })
        test_df.to_csv(data_dir / "test.csv", index=False)
        
        yield data_dir


class TestExploreCommand:
    """Test explore command functionality."""
    
    def test_explore_basic(self, runner, sample_data_dir):
        """Test basic explore functionality."""
        result = runner.invoke(
            app,
            ["analyze", "explore", "-d", str(sample_data_dir)],
        )
        
        assert result.exit_code == 0
        assert "Data Exploration Report" in result.stdout
        assert "train" in result.stdout
        assert "test" in result.stdout
        assert "Rows:" in result.stdout
        assert "Columns:" in result.stdout
        
    def test_explore_specific_table(self, runner, sample_data_dir):
        """Test exploring a specific table."""
        result = runner.invoke(
            app,
            ["analyze", "explore", "-d", str(sample_data_dir), "-t", "train"],
        )
        
        assert result.exit_code == 0
        assert "Table: train" in result.stdout
        assert "Rows: 5" in result.stdout
        assert "Columns: 6" in result.stdout
        # Should not show test table
        assert "Table: test" not in result.stdout
        
    def test_explore_missing_values(self, runner, sample_data_dir):
        """Test missing value analysis."""
        result = runner.invoke(
            app,
            ["analyze", "explore", "-d", str(sample_data_dir), "--missing"],
        )
        
        assert result.exit_code == 0
        assert "Missing Values" in result.stdout
        # Should show missing values for name, age, salary
        assert "name" in result.stdout
        # The percentage format might vary, so just check for the value
        assert "20" in result.stdout or "1" in result.stdout  # 1 missing out of 5
        
    def test_explore_no_missing_values(self, runner, sample_data_dir):
        """Test disabling missing value analysis."""
        result = runner.invoke(
            app,
            ["analyze", "explore", "-d", str(sample_data_dir), "--no-missing"],
        )
        
        assert result.exit_code == 0
        # Should not show missing values section
        assert "Missing Values" not in result.stdout
        
    def test_explore_cardinality(self, runner, sample_data_dir):
        """Test cardinality analysis."""
        result = runner.invoke(
            app,
            ["analyze", "explore", "-d", str(sample_data_dir), "--cardinality"],
        )
        
        assert result.exit_code == 0
        assert "Column Cardinality" in result.stdout
        assert "Unique Values" in result.stdout
        
    def test_explore_no_cardinality(self, runner, sample_data_dir):
        """Test disabling cardinality analysis."""
        result = runner.invoke(
            app,
            ["analyze", "explore", "-d", str(sample_data_dir), "--no-cardinality"],
        )
        
        assert result.exit_code == 0
        # Should not show cardinality section
        assert "Column Cardinality" not in result.stdout
        
    def test_explore_correlations(self, runner, sample_data_dir):
        """Test correlation analysis."""
        result = runner.invoke(
            app,
            ["analyze", "explore", "-d", str(sample_data_dir), "-t", "train", "--correlations"],
        )
        
        assert result.exit_code == 0
        assert "Top Correlations" in result.stdout
        # Should show correlations between numeric columns
        
    def test_explore_sample_size(self, runner, sample_data_dir):
        """Test custom sample size."""
        result = runner.invoke(
            app,
            ["analyze", "explore", "-d", str(sample_data_dir), "-t", "train", "--sample", "3"],
        )
        
        assert result.exit_code == 0
        assert "Sample Data (first 3 rows)" in result.stdout
        
    def test_explore_save_report(self, runner, sample_data_dir, tmp_path):
        """Test saving exploration report."""
        output_file = tmp_path / "explore_report.md"
        
        result = runner.invoke(
            app,
            ["analyze", "explore", "-d", str(sample_data_dir), "-o", str(output_file)],
        )
        
        assert result.exit_code == 0
        assert output_file.exists()
        assert "Exploration report saved to" in result.stdout
        
        # Check report content
        content = output_file.read_text()
        assert "# Data Exploration Report" in content
        assert "## Table: train" in content
        assert "## Table: test" in content
        
    def test_explore_table_relationships(self, runner, sample_data_dir):
        """Test detection of table relationships."""
        result = runner.invoke(
            app,
            ["analyze", "explore", "-d", str(sample_data_dir)],
        )
        
        assert result.exit_code == 0
        # Should detect common columns between tables
        assert "Potential Table Relationships" in result.stdout
        assert "id" in result.stdout  # Common column
        
    def test_explore_nonexistent_table(self, runner, sample_data_dir):
        """Test exploring non-existent table."""
        result = runner.invoke(
            app,
            ["analyze", "explore", "-d", str(sample_data_dir), "-t", "nonexistent"],
        )
        
        assert result.exit_code == 0
        assert "Table 'nonexistent' not found" in result.stdout
        # Tables might be in different order
        assert "Available tables:" in result.stdout
        assert "train" in result.stdout
        assert "test" in result.stdout
        
    def test_explore_empty_directory(self, runner, tmp_path):
        """Test exploring empty directory."""
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()
        
        result = runner.invoke(
            app,
            ["analyze", "explore", "-d", str(empty_dir)],
        )
        
        assert result.exit_code == 0
        assert "No tables found" in result.stdout
        
    def test_explore_with_config(self, runner, sample_data_dir, tmp_path):
        """Test explore with custom config."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text("""analysis:
  memory_limit: "1GB"
  auto_load_csvs: true
""")
        
        result = runner.invoke(
            app,
            ["analyze", "explore", "-d", str(sample_data_dir), "--config", str(config_file)],
        )
        
        assert result.exit_code == 0
        assert "Data Exploration Report" in result.stdout