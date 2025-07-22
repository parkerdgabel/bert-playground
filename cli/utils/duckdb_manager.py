"""DuckDB Manager for SQL-based data analysis in k-bert."""

import os
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, List, Optional, Union, Any

import duckdb
import pandas as pd
from loguru import logger
from rich.console import Console
from rich.table import Table
from rich.syntax import Syntax

from cli.config.schemas import AnalysisConfig


class DuckDBManager:
    """Manages DuckDB connections and operations for data analysis."""

    def __init__(
        self,
        data_dir: Optional[Path] = None,
        config: Optional[AnalysisConfig] = None,
        memory_limit: str = "4GB",
        temp_directory: Optional[Path] = None,
    ):
        """Initialize DuckDB manager with configuration.
        
        Args:
            data_dir: Directory containing data files to load
            config: Analysis configuration
            memory_limit: Memory limit for DuckDB (default: 4GB)
            temp_directory: Directory for temporary files
        """
        self.data_dir = data_dir
        self.config = config or AnalysisConfig()
        self.console = Console()
        
        # Configure DuckDB settings
        config_dict = {
            "memory_limit": memory_limit,
            "threads": os.cpu_count() or 4,
        }
        
        if temp_directory:
            config_dict["temp_directory"] = str(temp_directory)
            
        # Create connection
        self.connection = duckdb.connect(":memory:", config=config_dict)
        
        # Setup extensions
        self._setup_extensions()
        
        # Auto-load data if directory provided
        if self.data_dir and self.data_dir.exists():
            self._auto_load_data()
            
    def _setup_extensions(self):
        """Install and load useful DuckDB extensions."""
        try:
            # Install extensions
            self.connection.install_extension("httpfs")
            self.connection.install_extension("json")
            
            # Load extensions
            self.connection.load_extension("httpfs")
            self.connection.load_extension("json")
            
            logger.debug("DuckDB extensions loaded successfully")
        except Exception as e:
            logger.warning(f"Failed to load some DuckDB extensions: {e}")
            
    def _auto_load_data(self):
        """Automatically load CSV files from data directory."""
        if not self.data_dir or not self.data_dir.exists():
            return
            
        csv_files = list(self.data_dir.glob("*.csv"))
        logger.info(f"Found {len(csv_files)} CSV files in {self.data_dir}")
        
        for csv_file in csv_files:
            table_name = csv_file.stem.lower().replace("-", "_")
            try:
                self.load_csv(csv_file, table_name)
                logger.debug(f"Loaded {csv_file.name} as table '{table_name}'")
            except Exception as e:
                logger.error(f"Failed to load {csv_file.name}: {e}")
                
    def load_csv(
        self,
        file_path: Path,
        table_name: Optional[str] = None,
        **read_options,
    ) -> str:
        """Load CSV file into DuckDB table.
        
        Args:
            file_path: Path to CSV file
            table_name: Name for the table (defaults to file stem)
            **read_options: Additional options for CSV reading
            
        Returns:
            Table name that was created
        """
        if not file_path.exists():
            raise FileNotFoundError(f"CSV file not found: {file_path}")
            
        table_name = table_name or file_path.stem.lower().replace("-", "_")
        
        # Build read options
        options = {
            "AUTO_DETECT": True,
            "HEADER": True,
            "SAMPLE_SIZE": 10000,
        }
        options.update(read_options)
        
        # Create options string
        options_str = ", ".join(f"{k}={v}" for k, v in options.items())
        
        # Load CSV into table
        query = f"""
        CREATE OR REPLACE TABLE {table_name} AS 
        SELECT * FROM read_csv('{file_path}', {options_str})
        """
        
        self.connection.execute(query)
        
        # Get row count
        row_count = self.connection.execute(
            f"SELECT COUNT(*) FROM {table_name}"
        ).fetchone()[0]
        
        logger.info(f"Loaded {row_count:,} rows into table '{table_name}'")
        return table_name
        
    def register_dataframe(self, df: pd.DataFrame, table_name: str):
        """Register a pandas DataFrame as a DuckDB table.
        
        Args:
            df: DataFrame to register
            table_name: Name for the table
        """
        self.connection.register(table_name, df)
        logger.debug(f"Registered DataFrame as table '{table_name}'")
        
    def execute_query(
        self,
        query: str,
        params: Optional[Dict[str, Any]] = None,
    ) -> pd.DataFrame:
        """Execute SQL query and return results as DataFrame.
        
        Args:
            query: SQL query to execute
            params: Optional query parameters
            
        Returns:
            Query results as pandas DataFrame
        """
        try:
            if params:
                result = self.connection.execute(query, params)
            else:
                result = self.connection.execute(query)
                
            return result.df()
        except Exception as e:
            logger.error(f"Query execution failed: {e}")
            raise
            
    def execute_to_arrow(self, query: str) -> Any:
        """Execute query and return Arrow table for large results."""
        return self.connection.execute(query).arrow()
        
    def get_tables(self) -> List[str]:
        """Get list of available tables."""
        result = self.connection.execute("""
            SELECT table_name 
            FROM information_schema.tables 
            WHERE table_schema = 'main'
            ORDER BY table_name
        """)
        return [row[0] for row in result.fetchall()]
        
    def get_table_info(self, table_name: str) -> pd.DataFrame:
        """Get column information for a table.
        
        Args:
            table_name: Name of the table
            
        Returns:
            DataFrame with column information
        """
        query = f"""
        SELECT 
            column_name,
            data_type,
            is_nullable,
            column_default
        FROM information_schema.columns
        WHERE table_name = '{table_name}'
        ORDER BY ordinal_position
        """
        return self.execute_query(query)
        
    def get_table_stats(self, table_name: str) -> Dict[str, Any]:
        """Get basic statistics for a table.
        
        Args:
            table_name: Name of the table
            
        Returns:
            Dictionary with table statistics
        """
        # Row count
        row_count = self.connection.execute(
            f"SELECT COUNT(*) FROM {table_name}"
        ).fetchone()[0]
        
        # Column count
        col_count = len(self.get_table_info(table_name))
        
        # Get sample
        sample = self.execute_query(f"SELECT * FROM {table_name} LIMIT 5")
        
        return {
            "row_count": row_count,
            "column_count": col_count,
            "columns": list(sample.columns),
            "sample": sample,
        }
        
    def describe_table(self, table_name: str) -> pd.DataFrame:
        """Get statistical description of numeric columns.
        
        Args:
            table_name: Name of the table
            
        Returns:
            DataFrame with statistics
        """
        # Get numeric columns
        columns_info = self.get_table_info(table_name)
        numeric_types = ["INTEGER", "BIGINT", "DOUBLE", "FLOAT", "DECIMAL"]
        numeric_cols = columns_info[
            columns_info["data_type"].str.upper().isin(numeric_types)
        ]["column_name"].tolist()
        
        if not numeric_cols:
            return pd.DataFrame()
            
        # Build SUMMARIZE query for numeric columns
        summarize_cols = ", ".join(f'"{col}"' for col in numeric_cols)
        query = f"SUMMARIZE SELECT {summarize_cols} FROM {table_name}"
        
        return self.execute_query(query)
        
    def export_results(
        self,
        df: pd.DataFrame,
        output_path: Path,
        format: str = "csv",
    ):
        """Export query results to file.
        
        Args:
            df: DataFrame to export
            output_path: Output file path
            format: Export format (csv, parquet, json, excel)
        """
        format = format.lower()
        
        if format == "csv":
            df.to_csv(output_path, index=False)
        elif format == "parquet":
            df.to_parquet(output_path, index=False)
        elif format == "json":
            df.to_json(output_path, orient="records", indent=2)
        elif format == "excel":
            df.to_excel(output_path, index=False)
        else:
            raise ValueError(f"Unsupported format: {format}")
            
        logger.info(f"Exported {len(df)} rows to {output_path}")
        
    def format_results(
        self,
        df: pd.DataFrame,
        max_rows: int = 20,
        max_width: int = 20,
    ) -> Table:
        """Format DataFrame as Rich table for display.
        
        Args:
            df: DataFrame to format
            max_rows: Maximum rows to display
            max_width: Maximum column width
            
        Returns:
            Rich Table object
        """
        table = Table(show_header=True, header_style="bold cyan")
        
        # Add columns
        for col in df.columns:
            table.add_column(str(col), style="white", no_wrap=False)
            
        # Add rows (limit to max_rows)
        for idx, row in df.head(max_rows).iterrows():
            table.add_row(*[
                str(val)[:max_width] + "..." if len(str(val)) > max_width else str(val)
                for val in row
            ])
            
        if len(df) > max_rows:
            table.add_row(*["..." for _ in df.columns])
            
        return table
        
    def explain_query(self, query: str) -> str:
        """Get query execution plan.
        
        Args:
            query: SQL query to explain
            
        Returns:
            Query execution plan
        """
        explain_query = f"EXPLAIN {query}"
        result = self.connection.execute(explain_query)
        return result.fetchall()[0][0]
        
    @contextmanager
    def transaction(self):
        """Context manager for transactions."""
        try:
            self.connection.begin()
            yield self
            self.connection.commit()
        except Exception:
            self.connection.rollback()
            raise
            
    def close(self):
        """Close the DuckDB connection."""
        if hasattr(self, "connection"):
            self.connection.close()
            logger.debug("DuckDB connection closed")
            
    def __enter__(self):
        """Context manager entry."""
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
        
    def create_analysis_views(self):
        """Create useful analysis views for common patterns."""
        tables = self.get_tables()
        
        # Look for train/test split
        if "train" in tables and "test" in tables:
            # Create combined view
            train_cols = set(self.get_table_info("train")["column_name"])
            test_cols = set(self.get_table_info("test")["column_name"])
            common_cols = train_cols.intersection(test_cols)
            
            if common_cols:
                cols_str = ", ".join(f'"{col}"' for col in sorted(common_cols))
                self.connection.execute(f"""
                    CREATE OR REPLACE VIEW all_data AS
                    SELECT {cols_str}, 'train' as source FROM train
                    UNION ALL
                    SELECT {cols_str}, 'test' as source FROM test
                """)
                logger.info("Created 'all_data' view combining train and test")
                
    def get_memory_usage(self) -> Dict[str, Any]:
        """Get current memory usage statistics."""
        try:
            result = self.connection.execute("""
                SELECT * FROM duckdb_memory()
            """).df()
            
            # DuckDB memory function returns different columns in different versions
            # Try to get whatever columns are available
            memory_info = {}
            if "memory_usage_bytes" in result.columns:
                memory_info["memory_usage_bytes"] = int(result["memory_usage_bytes"].sum())
            if "temporary_storage_bytes" in result.columns:
                memory_info["temporary_storage_bytes"] = int(result["temporary_storage_bytes"].sum())
            if "database_size" in result.columns:
                memory_info["database_size"] = result["database_size"].iloc[0]
            if "block_size" in result.columns:
                memory_info["block_size"] = result["block_size"].iloc[0]
            if "total_blocks" in result.columns:
                memory_info["total_blocks"] = result["total_blocks"].iloc[0]
                
            return memory_info
        except Exception as e:
            logger.warning(f"Failed to get memory usage: {e}")
            return {"error": str(e)}