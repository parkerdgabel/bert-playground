"""Unit tests for profile command."""

import json
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
    """Create sample data directory with CSV files for profiling."""
    with tempfile.TemporaryDirectory() as tmpdir:
        data_dir = Path(tmpdir)
        
        # Create train.csv with various data types and patterns
        train_df = pd.DataFrame({
            "id": list(range(1, 101)),
            "age": [20 + i % 60 for i in range(100)],
            "income": [30000 + i * 1000 + (i % 7) * 5000 for i in range(100)],
            "category": ["A", "B", "C", "D"] * 25,
            "score": [50 + i % 50 + (i % 3) * 10 for i in range(100)],
            "is_active": [True if i % 3 != 0 else False for i in range(100)],
            "city": ["New York", "Los Angeles", "Chicago", "Houston", "Phoenix"] * 20,
            "rating": [1, 2, 3, 4, 5] * 20,
        })
        
        # Add some missing values
        train_df.loc[10:15, "age"] = None
        train_df.loc[20:25, "income"] = None
        train_df.loc[30:32, "city"] = None
        
        train_df.to_csv(data_dir / "train.csv", index=False)
        
        # Create test.csv
        test_df = pd.DataFrame({
            "id": list(range(101, 121)),
            "age": [25 + i % 50 for i in range(20)],
            "category": ["A", "B", "C", "D"] * 5,
            "score": [60 + i % 40 for i in range(20)],
        })
        test_df.to_csv(data_dir / "test.csv", index=False)
        
        yield data_dir


class TestProfileCommand:
    """Test profile command functionality."""
    
    def test_profile_basic(self, runner, sample_data_dir):
        """Test basic profile functionality."""
        result = runner.invoke(
            app,
            ["analyze", "profile", "-d", str(sample_data_dir)],
        )
        
        assert result.exit_code == 0
        assert "Profiling table: train" in result.stdout
        assert "Profiling table: test" in result.stdout
        assert "Table Profile Summary" in result.stdout
        assert "Column Summaries" in result.stdout
        
    def test_profile_specific_table(self, runner, sample_data_dir):
        """Test profiling a specific table."""
        result = runner.invoke(
            app,
            ["analyze", "profile", "-d", str(sample_data_dir), "-t", "train"],
        )
        
        assert result.exit_code == 0
        assert "Profiling table: train" in result.stdout
        # Should not profile test table
        assert "Profiling table: test" not in result.stdout
        
    def test_profile_with_target(self, runner, sample_data_dir):
        """Test profiling with target column analysis."""
        result = runner.invoke(
            app,
            ["analyze", "profile", "-d", str(sample_data_dir), "-t", "train", "--target", "category"],
        )
        
        assert result.exit_code == 0
        assert "target_column" in result.stdout or "Target" in result.stdout
        
    def test_profile_html_output(self, runner, sample_data_dir, tmp_path):
        """Test generating HTML profile report."""
        output_dir = tmp_path / "reports"
        
        result = runner.invoke(
            app,
            ["analyze", "profile", "-d", str(sample_data_dir), "-t", "train", "-o", str(output_dir), "-f", "html"],
        )
        
        assert result.exit_code == 0
        assert output_dir.exists()
        assert (output_dir / "train_profile.html").exists()
        
        # Check HTML content
        html_content = (output_dir / "train_profile.html").read_text()
        assert "Data Profile Report" in html_content
        assert "Dataset Overview" in html_content
        assert "Column Profiles" in html_content
        
    def test_profile_json_output(self, runner, sample_data_dir, tmp_path):
        """Test generating JSON profile report."""
        output_dir = tmp_path / "reports"
        
        result = runner.invoke(
            app,
            ["analyze", "profile", "-d", str(sample_data_dir), "-t", "train", "-o", str(output_dir), "-f", "json"],
        )
        
        assert result.exit_code == 0
        assert (output_dir / "train_profile.json").exists()
        
        # Check JSON content
        with open(output_dir / "train_profile.json") as f:
            profile_data = json.load(f)
            
        assert "table_name" in profile_data
        assert profile_data["table_name"] == "train"
        assert "metadata" in profile_data
        assert "columns" in profile_data
        assert len(profile_data["columns"]) == 8
        
    def test_profile_markdown_output(self, runner, sample_data_dir, tmp_path):
        """Test generating Markdown profile report."""
        output_dir = tmp_path / "reports"
        
        result = runner.invoke(
            app,
            ["analyze", "profile", "-d", str(sample_data_dir), "-t", "train", "-o", str(output_dir), "-f", "markdown"],
        )
        
        assert result.exit_code == 0
        assert (output_dir / "train_profile.md").exists()
        
        # Check Markdown content
        md_content = (output_dir / "train_profile.md").read_text()
        assert "# Data Profile Report" in md_content
        assert "## Dataset Overview" in md_content
        assert "## Column Profiles" in md_content
        
    def test_profile_no_plots(self, runner, sample_data_dir, tmp_path):
        """Test profiling without plots."""
        output_dir = tmp_path / "reports"
        
        result = runner.invoke(
            app,
            ["analyze", "profile", "-d", str(sample_data_dir), "-t", "train", "-o", str(output_dir), "--no-plots"],
        )
        
        assert result.exit_code == 0
        
        # Check JSON to verify no plots
        with open(output_dir / "train_profile.json") as f:
            profile_data = json.load(f)
        assert profile_data["plots"] == {}
        
    def test_profile_with_sample_size(self, runner, sample_data_dir):
        """Test profiling with custom sample size."""
        result = runner.invoke(
            app,
            ["analyze", "profile", "-d", str(sample_data_dir), "-t", "train", "--sample", "50"],
        )
        
        assert result.exit_code == 0
        assert "Sample Size: 50" in result.stdout
        
    def test_profile_numeric_statistics(self, runner, sample_data_dir):
        """Test numeric column statistics in profile."""
        result = runner.invoke(
            app,
            ["analyze", "profile", "-d", str(sample_data_dir), "-t", "train"],
        )
        
        assert result.exit_code == 0
        # Should show statistics for numeric columns
        assert "age" in result.stdout
        assert "Mean:" in result.stdout
        assert "Missing:" in result.stdout
        
    def test_profile_categorical_statistics(self, runner, sample_data_dir):
        """Test categorical column statistics in profile."""
        result = runner.invoke(
            app,
            ["analyze", "profile", "-d", str(sample_data_dir), "-t", "train"],
        )
        
        assert result.exit_code == 0
        # Should show statistics for categorical columns
        assert "category" in result.stdout
        assert "Unique:" in result.stdout
        
    def test_profile_missing_patterns(self, runner, sample_data_dir, tmp_path):
        """Test missing value pattern analysis."""
        output_dir = tmp_path / "reports"
        
        result = runner.invoke(
            app,
            ["analyze", "profile", "-d", str(sample_data_dir), "-t", "train", "-o", str(output_dir), "-f", "json"],
        )
        
        assert result.exit_code == 0
        
        with open(output_dir / "train_profile.json") as f:
            profile_data = json.load(f)
            
        assert "missing_patterns" in profile_data
        assert "column_missing_counts" in profile_data["missing_patterns"]
        assert profile_data["missing_patterns"]["column_missing_counts"]["age"] > 0
        
    def test_profile_correlations(self, runner, sample_data_dir, tmp_path):
        """Test correlation analysis in profile."""
        output_dir = tmp_path / "reports"
        
        result = runner.invoke(
            app,
            ["analyze", "profile", "-d", str(sample_data_dir), "-t", "train", "-o", str(output_dir), "-f", "json"],
        )
        
        assert result.exit_code == 0
        
        with open(output_dir / "train_profile.json") as f:
            profile_data = json.load(f)
            
        assert "correlations" in profile_data
        # Should have correlations between numeric columns
        assert len(profile_data["correlations"]) > 0
        
    def test_profile_multiple_tables(self, runner, sample_data_dir, tmp_path):
        """Test profiling multiple tables."""
        output_dir = tmp_path / "reports"
        
        result = runner.invoke(
            app,
            ["analyze", "profile", "-d", str(sample_data_dir), "-o", str(output_dir)],
        )
        
        assert result.exit_code == 0
        assert (output_dir / "train_profile.html").exists()
        assert (output_dir / "test_profile.html").exists()
        assert (output_dir / "index.html").exists()  # Combined report index
        
    def test_profile_nonexistent_table(self, runner, sample_data_dir):
        """Test profiling non-existent table."""
        result = runner.invoke(
            app,
            ["analyze", "profile", "-d", str(sample_data_dir), "-t", "nonexistent"],
        )
        
        assert result.exit_code == 0
        assert "Table 'nonexistent' not found" in result.stdout
        
    def test_profile_empty_directory(self, runner, tmp_path):
        """Test profiling empty directory."""
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()
        
        result = runner.invoke(
            app,
            ["analyze", "profile", "-d", str(empty_dir)],
        )
        
        assert result.exit_code == 0
        assert "No tables found" in result.stdout
        
    def test_profile_with_config(self, runner, sample_data_dir, tmp_path):
        """Test profile with custom config."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text("""
analysis:
    memory_limit: "1GB"
    visualization_backend: "plotly"
    figure_width: 10
    figure_height: 6
""")
        
        result = runner.invoke(
            app,
            ["analyze", "profile", "-d", str(sample_data_dir), "--config", str(config_file)],
        )
        
        assert result.exit_code == 0
        assert "Profiling complete!" in result.stdout