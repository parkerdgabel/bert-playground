"""Unit tests for SQL command."""

import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import pandas as pd
import pytest
from typer.testing import CliRunner

from cli.app import app
from cli.commands.analyze.sql import OutputFormat


@pytest.fixture
def runner():
    """Create CLI test runner."""
    return CliRunner()


@pytest.fixture
def sample_data_dir():
    """Create sample data directory with CSV files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        data_dir = Path(tmpdir)
        
        # Create sample CSV
        df = pd.DataFrame({
            "id": [1, 2, 3],
            "name": ["Alice", "Bob", "Charlie"],
            "value": [10.5, 20.0, 15.5],
        })
        df.to_csv(data_dir / "data.csv", index=False)
        
        yield data_dir


class TestSQLCommand:
    """Test SQL command functionality."""
    
    def test_sql_command_single_query(self, runner, sample_data_dir):
        """Test executing a single SQL query."""
        result = runner.invoke(
            app,
            ["analyze", "sql", "SELECT COUNT(*) FROM data", "-d", str(sample_data_dir)],
        )
        
        assert result.exit_code == 0
        assert "3" in result.stdout  # Should show count of 3
        
    def test_sql_command_with_limit(self, runner, sample_data_dir):
        """Test SQL query with limit option."""
        result = runner.invoke(
            app,
            ["analyze", "sql", "SELECT * FROM data", "-d", str(sample_data_dir), "--limit", "2"],
        )
        
        assert result.exit_code == 0
        assert "Alice" in result.stdout
        assert "Bob" in result.stdout
        # Charlie might not appear due to limit
        
    def test_sql_command_with_explain(self, runner, sample_data_dir):
        """Test SQL query with explain option."""
        result = runner.invoke(
            app,
            ["analyze", "sql", "SELECT * FROM data", "-d", str(sample_data_dir), "--explain"],
        )
        
        assert result.exit_code == 0
        assert "Execution plan" in result.stdout or "EXPLAIN" in result.stdout
        
    def test_sql_command_export_csv(self, runner, sample_data_dir, tmp_path):
        """Test exporting results to CSV."""
        output_file = tmp_path / "results.csv"
        
        result = runner.invoke(
            app,
            [
                "analyze", "sql", "SELECT * FROM data",
                "-d", str(sample_data_dir),
                "-o", str(output_file),
                "-f", "csv",
            ],
        )
        
        assert result.exit_code == 0
        assert output_file.exists()
        assert "Results saved to" in result.stdout
        
        # Check CSV content
        df = pd.read_csv(output_file)
        assert len(df) == 3
        assert list(df.columns) == ["id", "name", "value"]
        
    def test_sql_command_export_json(self, runner, sample_data_dir, tmp_path):
        """Test exporting results to JSON."""
        output_file = tmp_path / "results.json"
        
        result = runner.invoke(
            app,
            [
                "analyze", "sql", "SELECT * FROM data WHERE value > 15",
                "-d", str(sample_data_dir),
                "-o", str(output_file),
                "-f", "json",
            ],
        )
        
        assert result.exit_code == 0
        assert output_file.exists()
        
        # Check JSON content
        import json
        with open(output_file) as f:
            data = json.load(f)
        # Should have records where value > 15
        assert len(data) >= 1
        # Check that all values are > 15
        for record in data:
            assert record["value"] > 15
        
    def test_sql_command_no_tables(self, runner, tmp_path):
        """Test SQL command with empty directory."""
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()
        
        result = runner.invoke(
            app,
            ["analyze", "sql", "SELECT 1", "-d", str(empty_dir)],
        )
        
        assert result.exit_code == 0
        assert "No tables found" in result.stdout
        
    def test_sql_command_invalid_query(self, runner, sample_data_dir):
        """Test SQL command with invalid query."""
        result = runner.invoke(
            app,
            ["analyze", "sql", "INVALID SQL QUERY", "-d", str(sample_data_dir)],
        )
        
        assert result.exit_code == 1
        assert "Query error" in result.stdout
        
    @patch("cli.commands.analyze.sql.Prompt")
    def test_sql_interactive_mode(self, mock_prompt, runner, sample_data_dir):
        """Test interactive SQL mode."""
        # Mock interactive inputs
        mock_prompt.ask.side_effect = [
            "SELECT COUNT(*) FROM data",
            ".tables",
            ".exit",
        ]
        
        result = runner.invoke(
            app,
            ["analyze", "sql", "--interactive", "-d", str(sample_data_dir)],
        )
        
        assert result.exit_code == 0
        assert "DuckDB Interactive SQL Shell" in result.stdout
        assert "Available tables" in result.stdout
        
    def test_sql_command_with_config(self, runner, sample_data_dir, tmp_path):
        """Test SQL command with custom config."""
        # Create config file
        config_file = tmp_path / "config.yaml"
        config_file.write_text("""analysis:
  memory_limit: "1GB"
  default_format: "json"
""")
        
        result = runner.invoke(
            app,
            [
                "analyze", "sql", "SELECT * FROM data",
                "-d", str(sample_data_dir),
                "--config", str(config_file),
            ],
        )
        
        assert result.exit_code == 0
        
    def test_output_formats(self):
        """Test output format enum."""
        assert OutputFormat.TABLE.value == "table"
        assert OutputFormat.CSV.value == "csv"
        assert OutputFormat.JSON.value == "json"
        assert OutputFormat.PARQUET.value == "parquet"
        assert OutputFormat.MARKDOWN.value == "markdown"