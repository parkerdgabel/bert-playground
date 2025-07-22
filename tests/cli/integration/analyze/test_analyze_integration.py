"""Integration tests for analyze commands."""

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
def kaggle_style_data():
    """Create Kaggle-style competition data."""
    with tempfile.TemporaryDirectory() as tmpdir:
        data_dir = Path(tmpdir) / "titanic"
        data_dir.mkdir()
        
        # Create train.csv (Titanic-like data)
        train_df = pd.DataFrame({
            "PassengerId": list(range(1, 892)),
            "Survived": [i % 2 for i in range(891)],
            "Pclass": [1 + i % 3 for i in range(891)],
            "Name": [f"Passenger_{i}" for i in range(891)],
            "Sex": ["male" if i % 2 == 0 else "female" for i in range(891)],
            "Age": [20 + i % 60 if i % 10 != 0 else None for i in range(891)],
            "SibSp": [i % 5 for i in range(891)],
            "Parch": [i % 3 for i in range(891)],
            "Ticket": [f"TICKET_{i}" for i in range(891)],
            "Fare": [10.0 + i % 200 + (i % 7) * 5.5 for i in range(891)],
            "Cabin": [f"C{i}" if i % 5 == 0 else None for i in range(891)],
            "Embarked": [["S", "C", "Q"][i % 3] if i % 20 != 0 else None for i in range(891)],
        })
        train_df.to_csv(data_dir / "train.csv", index=False)
        
        # Create test.csv
        test_df = pd.DataFrame({
            "PassengerId": list(range(892, 1310)),
            "Pclass": [1 + i % 3 for i in range(418)],
            "Name": [f"Passenger_{i}" for i in range(892, 1310)],
            "Sex": ["male" if i % 2 == 0 else "female" for i in range(418)],
            "Age": [20 + i % 60 if i % 10 != 0 else None for i in range(418)],
            "SibSp": [i % 5 for i in range(418)],
            "Parch": [i % 3 for i in range(418)],
            "Ticket": [f"TICKET_{i}" for i in range(892, 1310)],
            "Fare": [10.0 + i % 200 + (i % 7) * 5.5 for i in range(418)],
            "Cabin": [f"C{i}" if i % 5 == 0 else None for i in range(418)],
            "Embarked": [["S", "C", "Q"][i % 3] if i % 20 != 0 else None for i in range(418)],
        })
        test_df.to_csv(data_dir / "test.csv", index=False)
        
        # Create sample_submission.csv
        submission_df = pd.DataFrame({
            "PassengerId": list(range(892, 1310)),
            "Survived": [0] * 418,
        })
        submission_df.to_csv(data_dir / "sample_submission.csv", index=False)
        
        yield data_dir


class TestAnalyzeIntegration:
    """Integration tests for analyze commands working together."""
    
    def test_full_analysis_workflow(self, runner, kaggle_style_data, tmp_path):
        """Test complete analysis workflow: explore -> profile -> sql."""
        reports_dir = tmp_path / "reports"
        
        # Step 1: Explore the data
        result = runner.invoke(
            app,
            ["analyze", "explore", "-d", str(kaggle_style_data), "-o", str(reports_dir / "exploration.md")],
        )
        if result.exit_code != 0:
            print(f"STDOUT:\n{result.stdout}")
            print(f"STDERR:\n{result.stderr}")
        assert result.exit_code == 0
        assert (reports_dir / "exploration.md").exists()
        
        # Step 2: Profile the training data
        result = runner.invoke(
            app,
            ["analyze", "profile", "-d", str(kaggle_style_data), "-t", "train", "--target", "Survived", "-o", str(reports_dir), "-f", "html"],
        )
        assert result.exit_code == 0
        assert (reports_dir / "train_profile.html").exists()
        
        # Step 3: Run SQL analysis
        sql_output = reports_dir / "survival_analysis.csv"
        result = runner.invoke(
            app,
            [
                "analyze", "sql",
                "SELECT Pclass, Sex, AVG(Survived) as survival_rate, COUNT(*) as count FROM train GROUP BY Pclass, Sex ORDER BY Pclass, Sex",
                "-d", str(kaggle_style_data),
                "-o", str(sql_output),
                "-f", "csv",
            ],
        )
        assert result.exit_code == 0
        assert sql_output.exists()
        
        # Verify SQL results
        results_df = pd.read_csv(sql_output)
        assert len(results_df) == 6  # 3 classes × 2 sexes
        assert "survival_rate" in results_df.columns
        
    def test_missing_value_analysis_workflow(self, runner, kaggle_style_data):
        """Test analyzing missing values across commands."""
        # First explore to see missing values
        result = runner.invoke(
            app,
            ["analyze", "explore", "-d", str(kaggle_style_data), "-t", "train", "--missing"],
        )
        assert result.exit_code == 0
        assert "Missing Values" in result.stdout
        assert "Age" in result.stdout  # Age has missing values
        assert "Cabin" in result.stdout  # Cabin has missing values
        
        # Then use SQL to analyze missing patterns
        result = runner.invoke(
            app,
            [
                "analyze", "sql",
                """SELECT 
                    COUNT(*) as total,
                    SUM(CASE WHEN Age IS NULL THEN 1 ELSE 0 END) as age_missing,
                    SUM(CASE WHEN Cabin IS NULL THEN 1 ELSE 0 END) as cabin_missing,
                    SUM(CASE WHEN Age IS NULL AND Cabin IS NULL THEN 1 ELSE 0 END) as both_missing
                FROM train""",
                "-d", str(kaggle_style_data),
            ],
        )
        assert result.exit_code == 0
        assert "age_missing" in result.stdout
        assert "cabin_missing" in result.stdout
        
    def test_feature_engineering_insights(self, runner, kaggle_style_data):
        """Test using analysis for feature engineering insights."""
        # Profile to understand distributions
        result = runner.invoke(
            app,
            ["analyze", "profile", "-d", str(kaggle_style_data), "-t", "train", "--target", "Survived"],
        )
        assert result.exit_code == 0
        
        # Use SQL to create derived features
        result = runner.invoke(
            app,
            [
                "analyze", "sql",
                """SELECT 
                    CASE 
                        WHEN Age < 18 THEN 'child'
                        WHEN Age < 60 THEN 'adult'
                        ELSE 'senior'
                    END as age_group,
                    AVG(Survived) as survival_rate,
                    COUNT(*) as count
                FROM train
                WHERE Age IS NOT NULL
                GROUP BY age_group
                ORDER BY survival_rate DESC""",
                "-d", str(kaggle_style_data),
            ],
        )
        assert result.exit_code == 0
        assert "age_group" in result.stdout
        assert "survival_rate" in result.stdout
        
    def test_train_test_comparison(self, runner, kaggle_style_data):
        """Test comparing train and test datasets."""
        # Use SQL to compare distributions
        result = runner.invoke(
            app,
            [
                "analyze", "sql",
                """WITH combined AS (
                    SELECT *, 'train' as dataset FROM train
                    UNION ALL
                    SELECT PassengerId, Pclass, Name, Sex, Age, SibSp, Parch, Ticket, Fare, Cabin, Embarked, 'test' as dataset FROM test
                )
                SELECT 
                    dataset,
                    COUNT(*) as count,
                    AVG(Age) as avg_age,
                    AVG(Fare) as avg_fare
                FROM combined
                GROUP BY dataset""",
                "-d", str(kaggle_style_data),
            ],
        )
        assert result.exit_code == 0
        assert "train" in result.stdout
        assert "test" in result.stdout
        assert "avg_age" in result.stdout
        
    def test_export_formats_consistency(self, runner, kaggle_style_data, tmp_path):
        """Test that different export formats contain consistent data."""
        query = "SELECT Pclass, COUNT(*) as count, AVG(Fare) as avg_fare FROM train GROUP BY Pclass ORDER BY Pclass"
        
        # Export to different formats
        formats = ["csv", "json", "parquet"]
        files = {}
        
        for fmt in formats:
            output_file = tmp_path / f"results.{fmt}"
            result = runner.invoke(
                app,
                ["analyze", "sql", query, "-d", str(kaggle_style_data), "-o", str(output_file), "-f", fmt],
            )
            assert result.exit_code == 0
            assert output_file.exists()
            files[fmt] = output_file
            
        # Load and compare data
        csv_df = pd.read_csv(files["csv"])
        json_df = pd.read_json(files["json"])
        parquet_df = pd.read_parquet(files["parquet"])
        
        # All should have same shape and values
        assert len(csv_df) == len(json_df) == len(parquet_df) == 3
        assert list(csv_df.columns) == list(json_df.columns) == list(parquet_df.columns)
        
    def test_complex_analytical_queries(self, runner, kaggle_style_data):
        """Test complex analytical queries combining multiple features."""
        # Window functions and CTEs
        result = runner.invoke(
            app,
            [
                "analyze", "sql",
                """WITH age_stats AS (
                    SELECT 
                        Pclass,
                        AVG(Age) as avg_age,
                        STDDEV(Age) as std_age
                    FROM train
                    WHERE Age IS NOT NULL
                    GROUP BY Pclass
                )
                SELECT 
                    t.Pclass,
                    t.Survived,
                    COUNT(*) as count,
                    AVG((t.Age - a.avg_age) / a.std_age) as avg_z_score
                FROM train t
                JOIN age_stats a ON t.Pclass = a.Pclass
                WHERE t.Age IS NOT NULL
                GROUP BY t.Pclass, t.Survived
                ORDER BY t.Pclass, t.Survived""",
                "-d", str(kaggle_style_data),
            ],
        )
        assert result.exit_code == 0
        assert "avg_z_score" in result.stdout
        
    def test_memory_efficient_profiling(self, runner, kaggle_style_data):
        """Test profiling with sample size for memory efficiency."""
        # Profile with small sample
        result = runner.invoke(
            app,
            ["analyze", "profile", "-d", str(kaggle_style_data), "-t", "train", "--sample", "100"],
        )
        assert result.exit_code == 0
        assert "Sample Size: 100" in result.stdout
        
    def test_config_based_analysis(self, runner, kaggle_style_data, tmp_path):
        """Test using configuration file for analysis settings."""
        # Create config
        config_file = tmp_path / "analysis_config.yaml"
        config_file.write_text("""
analysis:
    memory_limit: "2GB"
    default_format: "json"
    save_reports: true
    report_dir: "./analysis_output"
    visualization_backend: "plotly"
    max_result_rows: 1000
""")
        
        # Run commands with config
        result = runner.invoke(
            app,
            ["analyze", "explore", "-d", str(kaggle_style_data), "--config", str(config_file)],
        )
        assert result.exit_code == 0
        
        result = runner.invoke(
            app,
            ["analyze", "sql", "SELECT COUNT(*) FROM train", "-d", str(kaggle_style_data), "--config", str(config_file)],
        )
        assert result.exit_code == 0