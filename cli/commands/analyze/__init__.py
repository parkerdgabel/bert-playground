"""Data analysis commands for k-bert.

This module provides SQL-based data analysis capabilities using DuckDB,
enabling users to explore and analyze Kaggle datasets.
"""

import typer
from typing import Optional

# Create the analyze command group
analyze_app = typer.Typer(
    name="analyze",
    help="SQL-based data analysis commands using DuckDB",
    rich_markup_mode="rich",
    no_args_is_help=True,
    pretty_exceptions_enable=False,
)

# Import and register commands
from cli.commands.analyze.sql import sql
from cli.commands.analyze.explore import explore
from cli.commands.analyze.profile import profile

# Register commands with the app
analyze_app.command()(sql)
analyze_app.command()(explore)
analyze_app.command()(profile)

__all__ = ["analyze_app", "sql", "explore", "profile"]