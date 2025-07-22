"""Data exploration command for k-bert analysis."""

from pathlib import Path
from typing import Optional, List
import pandas as pd

import typer
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.tree import Tree
from loguru import logger

from cli.utils.duckdb_manager import DuckDBManager
from cli.utils import handle_errors
from cli.config import ConfigManager


console = Console()


@handle_errors
def explore(
    data_dir: Path = typer.Option(
        Path("./data"),
        "--data-dir", "-d",
        help="Directory containing CSV files to analyze",
        exists=True,
        file_okay=False,
        dir_okay=True,
    ),
    table: Optional[str] = typer.Option(
        None,
        "--table", "-t",
        help="Explore specific table (default: all tables)",
    ),
    sample_size: int = typer.Option(
        5,
        "--sample", "-s",
        help="Number of sample rows to display",
        min=1,
        max=100,
    ),
    show_correlations: bool = typer.Option(
        False,
        "--correlations", "-c",
        help="Show correlations between numeric columns",
    ),
    show_missing: bool = typer.Option(
        True,
        "--missing/--no-missing", "-m/-M",
        help="Show missing value analysis",
    ),
    show_cardinality: bool = typer.Option(
        True,
        "--cardinality/--no-cardinality", "-k/-K",
        help="Show column cardinality analysis",
    ),
    output: Optional[Path] = typer.Option(
        None,
        "--output", "-o",
        help="Save exploration report to file",
    ),
    config_path: Optional[Path] = typer.Option(
        None,
        "--config",
        help="Path to configuration file",
    ),
):
    """Explore datasets with comprehensive analysis.
    
    This command provides a quick overview of your data including:
    - Table and column information
    - Data types and cardinality
    - Missing value analysis
    - Sample data preview
    - Optional correlation analysis
    
    Examples:
        # Explore all tables in a directory
        k-bert analyze explore -d ./data/titanic
        
        # Explore specific table with correlations
        k-bert analyze explore -d ./data/titanic -t train --correlations
        
        # Save exploration report
        k-bert analyze explore -d ./data/titanic -o exploration_report.md
    """
    # Load configuration
    config_manager = ConfigManager()
    if config_path:
        config = config_manager.load_config(config_path)
    else:
        config = config_manager.get_merged_config()
        
    # Initialize DuckDB manager
    with console.status("Loading data..."):
        db = DuckDBManager(
            data_dir=data_dir,
            config=config.analysis if hasattr(config, 'analysis') else None,
        )
        
        # Get available tables
        tables = db.get_tables()
        if not tables:
            console.print("[yellow]No tables found in the specified directory.[/yellow]")
            return
            
    # Filter tables if specific table requested
    if table:
        if table not in tables:
            console.print(f"[red]Table '{table}' not found. Available tables: {', '.join(tables)}[/red]")
            return
        tables = [table]
        
    # Create exploration report
    report_lines = []
    
    # Overview
    console.print(Panel(
        f"[bold cyan]Data Exploration Report[/bold cyan]\n"
        f"Data directory: {data_dir}\n"
        f"Tables found: {len(tables)}",
        title="Overview",
        border_style="cyan",
    ))
    report_lines.append("# Data Exploration Report\n")
    report_lines.append(f"**Data directory:** {data_dir}\n")
    report_lines.append(f"**Tables found:** {len(tables)}\n")
    
    # Explore each table
    for table_name in tables:
        console.print(f"\n[bold green]Table: {table_name}[/bold green]")
        report_lines.append(f"\n## Table: {table_name}\n")
        
        # Get table statistics
        stats = db.get_table_stats(table_name)
        info = db.get_table_info(table_name)
        
        # Basic information
        console.print(f"Rows: {stats['row_count']:,}")
        console.print(f"Columns: {stats['column_count']}")
        report_lines.append(f"- **Rows:** {stats['row_count']:,}\n")
        report_lines.append(f"- **Columns:** {stats['column_count']}\n")
        
        # Column information
        console.print("\n[bold]Column Information:[/bold]")
        report_lines.append("\n### Column Information\n")
        
        col_table = Table(show_header=True, header_style="bold cyan")
        col_table.add_column("Column", style="white")
        col_table.add_column("Type", style="yellow")
        col_table.add_column("Nullable", style="green")
        
        report_lines.append("| Column | Type | Nullable |\n")
        report_lines.append("|--------|------|----------|\n")
        
        for _, row in info.iterrows():
            nullable = "Yes" if row["is_nullable"] == "YES" else "No"
            col_table.add_row(
                row["column_name"],
                row["data_type"],
                nullable,
            )
            report_lines.append(
                f"| {row['column_name']} | {row['data_type']} | {nullable} |\n"
            )
            
        console.print(col_table)
        
        # Missing value analysis
        if show_missing:
            missing_data = _analyze_missing_values(db, table_name)
            if missing_data:
                console.print("\n[bold]Missing Values:[/bold]")
                report_lines.append("\n### Missing Values\n")
                
                missing_table = Table(show_header=True, header_style="bold cyan")
                missing_table.add_column("Column", style="white")
                missing_table.add_column("Missing Count", style="red")
                missing_table.add_column("Missing %", style="red")
                
                report_lines.append("| Column | Missing Count | Missing % |\n")
                report_lines.append("|--------|---------------|------------|\n")
                
                for col, count, pct in missing_data:
                    missing_table.add_row(col, f"{count:,}", f"{pct:.2f}%")
                    report_lines.append(f"| {col} | {count:,} | {pct:.2f}% |\n")
                    
                console.print(missing_table)
                
        # Cardinality analysis
        if show_cardinality:
            cardinality_data = _analyze_cardinality(db, table_name)
            if cardinality_data:
                console.print("\n[bold]Column Cardinality:[/bold]")
                report_lines.append("\n### Column Cardinality\n")
                
                card_table = Table(show_header=True, header_style="bold cyan")
                card_table.add_column("Column", style="white")
                card_table.add_column("Unique Values", style="yellow")
                card_table.add_column("Cardinality %", style="yellow")
                
                report_lines.append("| Column | Unique Values | Cardinality % |\n")
                report_lines.append("|--------|---------------|---------------|\n")
                
                for col, unique, pct in cardinality_data:
                    card_table.add_row(col, f"{unique:,}", f"{pct:.2f}%")
                    report_lines.append(f"| {col} | {unique:,} | {pct:.2f}% |\n")
                    
                console.print(card_table)
                
        # Correlation analysis
        if show_correlations:
            correlations = _analyze_correlations(db, table_name)
            if correlations:
                console.print("\n[bold]Top Correlations:[/bold]")
                report_lines.append("\n### Top Correlations\n")
                
                corr_table = Table(show_header=True, header_style="bold cyan")
                corr_table.add_column("Column 1", style="white")
                corr_table.add_column("Column 2", style="white")
                corr_table.add_column("Correlation", style="magenta")
                
                report_lines.append("| Column 1 | Column 2 | Correlation |\n")
                report_lines.append("|----------|----------|-------------|\n")
                
                for col1, col2, corr in correlations[:10]:
                    corr_table.add_row(col1, col2, f"{corr:.3f}")
                    report_lines.append(f"| {col1} | {col2} | {corr:.3f} |\n")
                    
                console.print(corr_table)
                
        # Sample data
        console.print(f"\n[bold]Sample Data (first {sample_size} rows):[/bold]")
        report_lines.append(f"\n### Sample Data (first {sample_size} rows)\n")
        
        sample_df = db.execute_query(f"SELECT * FROM {table_name} LIMIT {sample_size}")
        console.print(db.format_results(sample_df, max_rows=sample_size))
        
        # Add sample to report as markdown table
        report_lines.append(sample_df.to_markdown(index=False) + "\n")
        
    # Show table relationships if multiple tables
    if len(tables) > 1:
        relationships = _analyze_relationships(db, tables)
        if relationships:
            console.print("\n[bold cyan]Potential Table Relationships:[/bold cyan]")
            report_lines.append("\n## Potential Table Relationships\n")
            
            rel_tree = Tree("[bold]Tables[/bold]")
            for rel in relationships:
                rel_tree.add(f"{rel['table1']}.{rel['column']} ↔ {rel['table2']}.{rel['column']}")
                report_lines.append(f"- {rel['table1']}.{rel['column']} ↔ {rel['table2']}.{rel['column']}\n")
                
            console.print(rel_tree)
            
    # Save report if requested
    if output:
        # Create parent directory if it doesn't exist
        output.parent.mkdir(parents=True, exist_ok=True)
        with open(output, "w") as f:
            f.writelines(report_lines)
        console.print(f"\n[green]Exploration report saved to: {output}[/green]")
        
    # Memory usage
    mem_usage = db.get_memory_usage()
    if "error" not in mem_usage:
        if "memory_usage_bytes" in mem_usage:
            console.print(f"\n[dim]DuckDB memory usage: {mem_usage['memory_usage_bytes']:,} bytes[/dim]")
        elif "database_size" in mem_usage:
            console.print(f"\n[dim]DuckDB memory usage: {mem_usage['database_size']} bytes[/dim]")
    
    # Clean up
    db.close()


def _analyze_missing_values(db: DuckDBManager, table_name: str) -> List[tuple]:
    """Analyze missing values in a table."""
    query = f"""
    SELECT 
        column_name,
        SUM(CASE WHEN column_value IS NULL THEN 1 ELSE 0 END) as null_count,
        COUNT(*) as total_count
    FROM (
        SELECT * FROM {table_name}
    ) t
    UNPIVOT (
        column_value FOR column_name IN (*)
    )
    GROUP BY column_name
    HAVING null_count > 0
    ORDER BY null_count DESC
    """
    
    try:
        # Fallback to column-by-column analysis if UNPIVOT fails
        info = db.get_table_info(table_name)
        results = []
        
        for _, row in info.iterrows():
            col_name = row["column_name"]
            null_query = f"""
            SELECT 
                COUNT(*) - COUNT("{col_name}") as null_count,
                COUNT(*) as total_count
            FROM {table_name}
            """
            
            result = db.execute_query(null_query)
            if result["null_count"].iloc[0] > 0:
                null_count = int(result["null_count"].iloc[0])
                total_count = int(result["total_count"].iloc[0])
                null_pct = (null_count / total_count) * 100
                results.append((col_name, null_count, null_pct))
                
        return sorted(results, key=lambda x: x[1], reverse=True)
        
    except Exception as e:
        logger.warning(f"Failed to analyze missing values: {e}")
        return []


def _analyze_cardinality(db: DuckDBManager, table_name: str) -> List[tuple]:
    """Analyze column cardinality."""
    info = db.get_table_info(table_name)
    stats = db.get_table_stats(table_name)
    total_rows = stats["row_count"]
    
    results = []
    for _, row in info.iterrows():
        col_name = row["column_name"]
        try:
            query = f'SELECT COUNT(DISTINCT "{col_name}") as unique_count FROM {table_name}'
            result = db.execute_query(query)
            unique_count = int(result["unique_count"].iloc[0])
            cardinality_pct = (unique_count / total_rows) * 100 if total_rows > 0 else 0
            results.append((col_name, unique_count, cardinality_pct))
        except Exception:
            continue
            
    return sorted(results, key=lambda x: x[2], reverse=True)


def _analyze_correlations(db: DuckDBManager, table_name: str) -> List[tuple]:
    """Analyze correlations between numeric columns."""
    # Get numeric columns
    info = db.get_table_info(table_name)
    numeric_types = ["INTEGER", "BIGINT", "DOUBLE", "FLOAT", "DECIMAL"]
    numeric_cols = info[
        info["data_type"].str.upper().isin(numeric_types)
    ]["column_name"].tolist()
    
    if len(numeric_cols) < 2:
        return []
        
    # Calculate correlations
    correlations = []
    for i, col1 in enumerate(numeric_cols):
        for col2 in numeric_cols[i+1:]:
            try:
                query = f"""
                SELECT CORR("{col1}", "{col2}") as correlation
                FROM {table_name}
                WHERE "{col1}" IS NOT NULL AND "{col2}" IS NOT NULL
                """
                result = db.execute_query(query)
                corr = result["correlation"].iloc[0]
                if corr is not None and abs(corr) > 0.1:  # Only show meaningful correlations
                    correlations.append((col1, col2, float(corr)))
            except Exception:
                continue
                
    return sorted(correlations, key=lambda x: abs(x[2]), reverse=True)


def _analyze_relationships(db: DuckDBManager, tables: List[str]) -> List[dict]:
    """Analyze potential relationships between tables."""
    relationships = []
    
    # Get column names for each table
    table_columns = {}
    for table in tables:
        info = db.get_table_info(table)
        table_columns[table] = set(info["column_name"].tolist())
        
    # Find common column names
    for i, table1 in enumerate(tables):
        for table2 in tables[i+1:]:
            common_cols = table_columns[table1].intersection(table_columns[table2])
            for col in common_cols:
                # Skip common generic names unless they might be keys
                if col.lower() not in ["id", "name", "date", "time", "value"] or "id" in col.lower():
                    relationships.append({
                        "table1": table1,
                        "table2": table2,
                        "column": col,
                    })
                    
    return relationships