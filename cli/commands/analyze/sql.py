"""SQL query execution command for k-bert data analysis."""

import sys
import readline
from pathlib import Path
from typing import Optional
from enum import Enum

import typer
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich.syntax import Syntax
from loguru import logger

from cli.utils.duckdb_manager import DuckDBManager
from cli.utils import handle_errors
from cli.config import ConfigManager


class OutputFormat(str, Enum):
    """Output format options."""
    TABLE = "table"
    CSV = "csv"
    JSON = "json"
    PARQUET = "parquet"
    MARKDOWN = "markdown"


console = Console()


@handle_errors
def sql(
    query: Optional[str] = typer.Argument(
        None,
        help="SQL query to execute. If not provided, enters interactive mode.",
    ),
    data_dir: Path = typer.Option(
        Path("./data"),
        "--data-dir", "-d",
        help="Directory containing CSV files to analyze",
        exists=True,
        file_okay=False,
        dir_okay=True,
    ),
    interactive: bool = typer.Option(
        False,
        "--interactive", "-i",
        help="Enter interactive SQL shell",
    ),
    output: Optional[Path] = typer.Option(
        None,
        "--output", "-o",
        help="Save results to file",
    ),
    format: OutputFormat = typer.Option(
        OutputFormat.TABLE,
        "--format", "-f",
        help="Output format",
    ),
    explain: bool = typer.Option(
        False,
        "--explain", "-e",
        help="Show query execution plan",
    ),
    limit: Optional[int] = typer.Option(
        None,
        "--limit", "-l",
        help="Limit number of result rows",
    ),
    config_path: Optional[Path] = typer.Option(
        None,
        "--config", "-c",
        help="Path to configuration file",
    ),
):
    """Execute SQL queries on Kaggle datasets using DuckDB.
    
    Examples:
        # Execute a single query
        k-bert analyze sql "SELECT * FROM train LIMIT 10" -d ./data/titanic
        
        # Interactive SQL shell
        k-bert analyze sql --interactive -d ./data/titanic
        
        # Save results to file
        k-bert analyze sql "SELECT survived, COUNT(*) FROM train GROUP BY survived" \\
            -d ./data/titanic -o results.csv -f csv
        
        # Show query execution plan
        k-bert analyze sql "SELECT * FROM train WHERE age > 30" -d ./data/titanic --explain
    """
    # Load configuration
    config_manager = ConfigManager()
    if config_path:
        config = config_manager.load_config(config_path)
    else:
        config = config_manager.get_merged_config()
    
    # Check if we should enter interactive mode
    if not query and not interactive:
        interactive = True
        
    # Initialize DuckDB manager
    with console.status("Initializing DuckDB and loading data..."):
        db = DuckDBManager(
            data_dir=data_dir,
            config=config.analysis if hasattr(config, 'analysis') else None,
        )
        
        # Show available tables
        tables = db.get_tables()
        if tables:
            console.print("\n[bold cyan]Available tables:[/bold cyan]")
            for table in tables:
                stats = db.get_table_stats(table)
                console.print(
                    f"  • {table}: {stats['row_count']:,} rows, "
                    f"{stats['column_count']} columns"
                )
        else:
            console.print("[yellow]No tables found. Make sure CSV files exist in the data directory.[/yellow]")
            return
            
    # Interactive mode
    if interactive:
        _run_interactive_shell(db, limit)
    else:
        # Execute single query
        _execute_single_query(db, query, output, format, explain, limit)
        
    # Clean up
    db.close()


def _run_interactive_shell(db: DuckDBManager, default_limit: Optional[int] = None):
    """Run interactive SQL shell."""
    console.print(Panel(
        "[bold green]DuckDB Interactive SQL Shell[/bold green]\n\n"
        "Enter SQL queries or special commands:\n"
        "  • [cyan].tables[/cyan] - Show available tables\n"
        "  • [cyan].schema <table>[/cyan] - Show table schema\n"
        "  • [cyan].describe <table>[/cyan] - Show table statistics\n"
        "  • [cyan].help[/cyan] - Show this help\n"
        "  • [cyan].exit[/cyan] or [cyan].quit[/cyan] - Exit shell\n\n"
        "Press Ctrl+C to cancel current query, Ctrl+D to exit",
        title="Welcome to k-bert SQL",
        border_style="green",
    ))
    
    # Setup readline for history
    history_file = Path.home() / ".k-bert" / "sql_history"
    history_file.parent.mkdir(exist_ok=True)
    
    try:
        readline.read_history_file(history_file)
    except FileNotFoundError:
        pass
        
    # Interactive loop
    while True:
        try:
            # Get query from user
            query = Prompt.ask("\n[bold cyan]sql>[/bold cyan]").strip()
            
            if not query:
                continue
                
            # Handle special commands
            if query.startswith("."):
                if query in [".exit", ".quit"]:
                    break
                elif query == ".help":
                    console.print(Panel(
                        "Available commands:\n"
                        "  • [cyan].tables[/cyan] - Show available tables\n"
                        "  • [cyan].schema <table>[/cyan] - Show table schema\n"
                        "  • [cyan].describe <table>[/cyan] - Show table statistics\n"
                        "  • [cyan].help[/cyan] - Show this help\n"
                        "  • [cyan].exit[/cyan] or [cyan].quit[/cyan] - Exit shell",
                        title="Help",
                        border_style="blue",
                    ))
                    continue
                elif query == ".tables":
                    tables = db.get_tables()
                    console.print("\n[bold]Available tables:[/bold]")
                    for table in tables:
                        stats = db.get_table_stats(table)
                        console.print(
                            f"  • {table}: {stats['row_count']:,} rows, "
                            f"{stats['column_count']} columns"
                        )
                    continue
                elif query.startswith(".schema "):
                    table_name = query.split()[1]
                    try:
                        info = db.get_table_info(table_name)
                        console.print(f"\n[bold]Schema for '{table_name}':[/bold]")
                        for _, row in info.iterrows():
                            nullable = "NULL" if row["is_nullable"] == "YES" else "NOT NULL"
                            default = f" DEFAULT {row['column_default']}" if row['column_default'] else ""
                            console.print(
                                f"  • {row['column_name']}: {row['data_type']} {nullable}{default}"
                            )
                    except Exception as e:
                        console.print(f"[red]Error: {e}[/red]")
                    continue
                elif query.startswith(".describe "):
                    table_name = query.split()[1]
                    try:
                        desc = db.describe_table(table_name)
                        if not desc.empty:
                            console.print(f"\n[bold]Statistics for '{table_name}':[/bold]")
                            console.print(db.format_results(desc))
                        else:
                            console.print(f"[yellow]No numeric columns in '{table_name}'[/yellow]")
                    except Exception as e:
                        console.print(f"[red]Error: {e}[/red]")
                    continue
                else:
                    console.print(f"[red]Unknown command: {query}[/red]")
                    continue
                    
            # Apply default limit if set and query doesn't have LIMIT
            if default_limit and "limit" not in query.lower():
                query = f"{query} LIMIT {default_limit}"
                
            # Execute SQL query
            with console.status(f"Executing query..."):
                try:
                    result = db.execute_query(query)
                    
                    if result.empty:
                        console.print("[yellow]Query returned no results[/yellow]")
                    else:
                        console.print(f"\n[green]Query returned {len(result):,} rows[/green]")
                        console.print(db.format_results(result))
                        
                except Exception as e:
                    console.print(f"[red]Query error: {e}[/red]")
                    
        except KeyboardInterrupt:
            console.print("\n[yellow]Query cancelled[/yellow]")
            continue
        except EOFError:
            break
            
    # Save history
    readline.write_history_file(history_file)
    console.print("\n[green]Goodbye![/green]")


def _execute_single_query(
    db: DuckDBManager,
    query: str,
    output: Optional[Path],
    format: OutputFormat,
    explain: bool,
    limit: Optional[int],
):
    """Execute a single SQL query."""
    # Show query
    console.print("\n[bold]Executing query:[/bold]")
    console.print(Syntax(query, "sql", theme="monokai"))
    
    # Show execution plan if requested
    if explain:
        try:
            plan = db.explain_query(query)
            console.print("\n[bold]Execution plan:[/bold]")
            console.print(Syntax(plan, "text", theme="monokai"))
        except Exception as e:
            console.print(f"[red]Failed to explain query: {e}[/red]")
            
    # Apply limit if specified
    if limit and "limit" not in query.lower():
        query = f"{query} LIMIT {limit}"
        
    # Execute query
    with console.status("Executing query..."):
        try:
            result = db.execute_query(query)
            
            if result.empty:
                console.print("\n[yellow]Query returned no results[/yellow]")
                return
                
            console.print(f"\n[green]Query returned {len(result):,} rows[/green]")
            
            # Display results
            if not output:
                console.print(db.format_results(result))
            else:
                # Export to file
                if format == OutputFormat.CSV:
                    result.to_csv(output, index=False)
                elif format == OutputFormat.JSON:
                    result.to_json(output, orient="records", indent=2)
                elif format == OutputFormat.PARQUET:
                    result.to_parquet(output, index=False)
                elif format == OutputFormat.MARKDOWN:
                    result.to_markdown(output, index=False)
                else:
                    # Table format - save as formatted text
                    with open(output, "w") as f:
                        f.write(result.to_string())
                        
                console.print(f"[green]Results saved to: {output}[/green]")
                
        except Exception as e:
            console.print(f"[red]Query error: {e}[/red]")
            logger.error(f"Query execution failed: {e}")
            raise typer.Exit(1)