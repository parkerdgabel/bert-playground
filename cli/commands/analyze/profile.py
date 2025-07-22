"""Data profiling command for k-bert analysis."""

import json
from pathlib import Path
from typing import Optional, Dict, Any, List
from datetime import datetime

import typer
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from rich.console import Console
from rich.panel import Panel
from rich.progress import track
from loguru import logger

from cli.utils.duckdb_manager import DuckDBManager
from cli.utils import handle_errors
from cli.config import ConfigManager


console = Console()


@handle_errors
def profile(
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
        help="Profile specific table (default: all tables)",
    ),
    target: Optional[str] = typer.Option(
        None,
        "--target",
        help="Target column for analysis (if applicable)",
    ),
    output_dir: Optional[Path] = typer.Option(
        None,
        "--output", "-o",
        help="Directory to save profiling report",
    ),
    format: str = typer.Option(
        "html",
        "--format", "-f",
        help="Output format (html, json, markdown)",
    ),
    include_plots: bool = typer.Option(
        True,
        "--plots/--no-plots", "-p/-P",
        help="Include visualization plots",
    ),
    sample_size: Optional[int] = typer.Option(
        None,
        "--sample", "-s",
        help="Sample size for analysis (default: use all data)",
    ),
    config_path: Optional[Path] = typer.Option(
        None,
        "--config",
        help="Path to configuration file",
    ),
):
    """Generate comprehensive data profiling report.
    
    This command creates detailed statistical profiles of your data including:
    - Descriptive statistics for numeric columns
    - Distribution analysis with histograms
    - Categorical value frequencies
    - Missing value patterns
    - Outlier detection
    - Target variable analysis (if specified)
    - Interactive visualizations
    
    Examples:
        # Profile all tables
        k-bert analyze profile -d ./data/titanic -o ./reports
        
        # Profile specific table with target analysis
        k-bert analyze profile -d ./data/titanic -t train --target survived
        
        # Generate JSON report without plots
        k-bert analyze profile -d ./data/titanic -f json --no-plots
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
        
    # Create output directory if specified
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        
    # Profile each table
    all_profiles = {}
    
    for table_name in track(tables, description="Profiling tables..."):
        console.print(f"\n[bold cyan]Profiling table: {table_name}[/bold cyan]")
        
        # Generate profile
        profile_data = _generate_table_profile(
            db, table_name, target, sample_size, include_plots
        )
        all_profiles[table_name] = profile_data
        
        # Display summary
        _display_profile_summary(profile_data)
        
        # Save individual table report if output specified
        if output_dir:
            _save_profile_report(
                profile_data,
                output_dir / f"{table_name}_profile",
                format,
                include_plots,
            )
            
    # Generate combined report if multiple tables
    if len(tables) > 1 and output_dir:
        _save_combined_report(all_profiles, output_dir, format)
        
    console.print(f"\n[green]Profiling complete![/green]")
    if output_dir:
        console.print(f"Reports saved to: {output_dir}")
        
    # Clean up
    db.close()


def _generate_table_profile(
    db: DuckDBManager,
    table_name: str,
    target: Optional[str],
    sample_size: Optional[int],
    include_plots: bool,
) -> Dict[str, Any]:
    """Generate comprehensive profile for a table."""
    profile = {
        "table_name": table_name,
        "generated_at": datetime.now().isoformat(),
        "metadata": {},
        "columns": {},
        "correlations": {},
        "missing_patterns": {},
        "target_analysis": {},
        "plots": {},
    }
    
    # Get table metadata
    stats = db.get_table_stats(table_name)
    info = db.get_table_info(table_name)
    
    profile["metadata"] = {
        "row_count": stats["row_count"],
        "column_count": stats["column_count"],
        "columns": stats["columns"],
        "sample_size": sample_size or stats["row_count"],
    }
    
    # Apply sampling if requested
    if sample_size and sample_size < stats["row_count"]:
        table_query = f"SELECT * FROM {table_name} ORDER BY RANDOM() LIMIT {sample_size}"
    else:
        table_query = f"SELECT * FROM {table_name}"
        
    # Analyze each column
    for _, col_info in info.iterrows():
        col_name = col_info["column_name"]
        col_type = col_info["data_type"]
        
        col_profile = _profile_column(db, table_query, col_name, col_type)
        profile["columns"][col_name] = col_profile
        
    # Correlation analysis for numeric columns
    numeric_cols = [
        col for col, prof in profile["columns"].items()
        if prof["type"] == "numeric"
    ]
    
    if len(numeric_cols) >= 2:
        profile["correlations"] = _compute_correlations(db, table_query, numeric_cols)
        
    # Missing value patterns
    profile["missing_patterns"] = _analyze_missing_patterns(
        db, table_query, list(profile["columns"].keys())
    )
    
    # Target analysis if specified
    if target and target in profile["columns"]:
        profile["target_analysis"] = _analyze_target(
            db, table_query, target, profile["columns"]
        )
        
    # Generate plots if requested
    if include_plots:
        profile["plots"] = _generate_plots(db, table_query, profile)
        
    return profile


def _profile_column(
    db: DuckDBManager,
    table_query: str,
    col_name: str,
    col_type: str,
) -> Dict[str, Any]:
    """Profile a single column."""
    profile = {
        "name": col_name,
        "dtype": col_type,
        "type": "numeric" if col_type.upper() in ["INTEGER", "BIGINT", "DOUBLE", "FLOAT", "DECIMAL"] else "categorical",
    }
    
    # Basic statistics
    basic_stats_query = f"""
    SELECT 
        COUNT(*) as count,
        COUNT(DISTINCT "{col_name}") as unique,
        COUNT(*) - COUNT("{col_name}") as missing,
        COUNT(CASE WHEN "{col_name}" = '' THEN 1 END) as empty_strings
    FROM ({table_query}) t
    """
    
    basic_stats = db.execute_query(basic_stats_query).iloc[0]
    profile["count"] = int(basic_stats["count"])
    profile["unique"] = int(basic_stats["unique"])
    profile["missing"] = int(basic_stats["missing"])
    profile["missing_pct"] = (profile["missing"] / profile["count"] * 100) if profile["count"] > 0 else 0
    profile["cardinality"] = profile["unique"] / profile["count"] if profile["count"] > 0 else 0
    
    if profile["type"] == "numeric":
        # Numeric statistics
        numeric_stats_query = f"""
        SELECT 
            MIN("{col_name}") as min,
            MAX("{col_name}") as max,
            AVG("{col_name}") as mean,
            MEDIAN("{col_name}") as median,
            MODE("{col_name}") as mode,
            STDDEV("{col_name}") as std,
            PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY "{col_name}") as q1,
            PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY "{col_name}") as q3,
            SKEWNESS("{col_name}") as skewness,
            KURTOSIS("{col_name}") as kurtosis
        FROM ({table_query}) t
        WHERE "{col_name}" IS NOT NULL
        """
        
        try:
            numeric_stats = db.execute_query(numeric_stats_query).iloc[0]
            profile.update({
                "min": float(numeric_stats["min"]) if numeric_stats["min"] is not None else None,
                "max": float(numeric_stats["max"]) if numeric_stats["max"] is not None else None,
                "mean": float(numeric_stats["mean"]) if numeric_stats["mean"] is not None else None,
                "median": float(numeric_stats["median"]) if numeric_stats["median"] is not None else None,
                "mode": float(numeric_stats["mode"]) if numeric_stats["mode"] is not None else None,
                "std": float(numeric_stats["std"]) if numeric_stats["std"] is not None else None,
                "q1": float(numeric_stats["q1"]) if numeric_stats["q1"] is not None else None,
                "q3": float(numeric_stats["q3"]) if numeric_stats["q3"] is not None else None,
                "iqr": float(numeric_stats["q3"] - numeric_stats["q1"]) if numeric_stats["q3"] is not None and numeric_stats["q1"] is not None else None,
                "skewness": float(numeric_stats["skewness"]) if numeric_stats["skewness"] is not None else None,
                "kurtosis": float(numeric_stats["kurtosis"]) if numeric_stats["kurtosis"] is not None else None,
            })
            
            # Outlier detection
            if profile["q1"] is not None and profile["q3"] is not None:
                lower_bound = profile["q1"] - 1.5 * profile["iqr"]
                upper_bound = profile["q3"] + 1.5 * profile["iqr"]
                
                outlier_query = f"""
                SELECT COUNT(*) as outliers
                FROM ({table_query}) t
                WHERE "{col_name}" < {lower_bound} OR "{col_name}" > {upper_bound}
                """
                
                outlier_count = int(db.execute_query(outlier_query)["outliers"].iloc[0])
                profile["outliers"] = outlier_count
                profile["outlier_pct"] = (outlier_count / profile["count"] * 100) if profile["count"] > 0 else 0
                
        except Exception as e:
            logger.warning(f"Failed to compute numeric stats for {col_name}: {e}")
            
    else:
        # Categorical statistics
        value_counts_query = f"""
        SELECT 
            "{col_name}" as value,
            COUNT(*) as count
        FROM ({table_query}) t
        GROUP BY "{col_name}"
        ORDER BY count DESC
        LIMIT 20
        """
        
        value_counts = db.execute_query(value_counts_query)
        profile["top_values"] = value_counts.to_dict("records")
        
    return profile


def _compute_correlations(
    db: DuckDBManager,
    table_query: str,
    numeric_cols: List[str],
) -> Dict[str, float]:
    """Compute correlation matrix for numeric columns."""
    correlations = {}
    
    for i, col1 in enumerate(numeric_cols):
        for col2 in numeric_cols[i:]:
            try:
                corr_query = f"""
                SELECT CORR("{col1}", "{col2}") as correlation
                FROM ({table_query}) t
                WHERE "{col1}" IS NOT NULL AND "{col2}" IS NOT NULL
                """
                
                result = db.execute_query(corr_query)
                corr = result["correlation"].iloc[0]
                if corr is not None:
                    correlations[f"{col1}__{col2}"] = float(corr)
                    
            except Exception as e:
                logger.warning(f"Failed to compute correlation for {col1} and {col2}: {e}")
                
    return correlations


def _analyze_missing_patterns(
    db: DuckDBManager,
    table_query: str,
    columns: List[str],
) -> Dict[str, Any]:
    """Analyze patterns in missing values."""
    patterns = {
        "column_missing_counts": {},
        "missing_correlations": {},
    }
    
    # Get missing counts per column
    for col in columns:
        missing_query = f"""
        SELECT COUNT(*) - COUNT("{col}") as missing
        FROM ({table_query}) t
        """
        
        result = db.execute_query(missing_query)
        patterns["column_missing_counts"][col] = int(result["missing"].iloc[0])
        
    # Analyze missing value correlations (simplified)
    # Check if columns tend to be missing together
    cols_with_missing = [
        col for col, count in patterns["column_missing_counts"].items()
        if count > 0
    ]
    
    if len(cols_with_missing) >= 2:
        for i, col1 in enumerate(cols_with_missing[:5]):  # Limit to 5 columns
            for col2 in cols_with_missing[i+1:6]:
                try:
                    pattern_query = f"""
                    SELECT 
                        COUNT(CASE WHEN "{col1}" IS NULL AND "{col2}" IS NULL THEN 1 END) as both_missing,
                        COUNT(CASE WHEN "{col1}" IS NULL OR "{col2}" IS NULL THEN 1 END) as either_missing,
                        COUNT(*) as total
                    FROM ({table_query}) t
                    """
                    
                    result = db.execute_query(pattern_query).iloc[0]
                    both = int(result["both_missing"])
                    either = int(result["either_missing"])
                    
                    if either > 0:
                        correlation = both / either
                        patterns["missing_correlations"][f"{col1}__{col2}"] = correlation
                        
                except Exception:
                    continue
                    
    return patterns


def _analyze_target(
    db: DuckDBManager,
    table_query: str,
    target: str,
    columns: Dict[str, Any],
) -> Dict[str, Any]:
    """Analyze target variable and its relationships."""
    analysis = {
        "target_column": target,
        "distribution": {},
        "feature_importance": {},
    }
    
    target_profile = columns[target]
    
    if target_profile["type"] == "categorical":
        # Get target distribution
        dist_query = f"""
        SELECT 
            "{target}" as value,
            COUNT(*) as count,
            COUNT(*) * 100.0 / SUM(COUNT(*)) OVER () as percentage
        FROM ({table_query}) t
        WHERE "{target}" IS NOT NULL
        GROUP BY "{target}"
        ORDER BY count DESC
        """
        
        distribution = db.execute_query(dist_query)
        analysis["distribution"] = distribution.to_dict("records")
        
        # Analyze relationship with numeric features
        numeric_features = [
            col for col, prof in columns.items()
            if prof["type"] == "numeric" and col != target
        ]
        
        for feature in numeric_features[:10]:  # Limit to top 10
            try:
                # Calculate mean by target class
                importance_query = f"""
                SELECT 
                    "{target}",
                    AVG("{feature}") as mean,
                    STDDEV("{feature}") as std
                FROM ({table_query}) t
                WHERE "{target}" IS NOT NULL AND "{feature}" IS NOT NULL
                GROUP BY "{target}"
                """
                
                result = db.execute_query(importance_query)
                if len(result) > 1:
                    # Simple importance: ratio of between-class to within-class variance
                    means = result["mean"].values
                    importance = means.std() / result["std"].mean() if result["std"].mean() > 0 else 0
                    analysis["feature_importance"][feature] = float(importance)
                    
            except Exception:
                continue
                
    return analysis


def _generate_plots(
    db: DuckDBManager,
    table_query: str,
    profile: Dict[str, Any],
) -> Dict[str, Any]:
    """Generate visualization plots."""
    plots = {}
    
    # Select columns to plot (limit to avoid too many plots)
    numeric_cols = [
        col for col, prof in profile["columns"].items()
        if prof["type"] == "numeric"
    ][:5]
    
    categorical_cols = [
        col for col, prof in profile["columns"].items()
        if prof["type"] == "categorical" and prof["unique"] <= 20
    ][:3]
    
    # Numeric distributions
    if numeric_cols:
        for col in numeric_cols:
            try:
                # Get data for histogram
                data_query = f"""
                SELECT "{col}" 
                FROM ({table_query}) t 
                WHERE "{col}" IS NOT NULL
                """
                
                data = db.execute_query(data_query)[col].values
                
                # Create histogram
                fig = go.Figure()
                fig.add_trace(go.Histogram(
                    x=data,
                    nbinsx=30,
                    name=col,
                    marker_color='lightblue',
                    marker_line_color='darkblue',
                    marker_line_width=1,
                ))
                
                fig.update_layout(
                    title=f"Distribution of {col}",
                    xaxis_title=col,
                    yaxis_title="Count",
                    template="plotly_white",
                    width=800,
                    height=400,
                )
                
                plots[f"{col}_distribution"] = fig.to_json()
                
            except Exception as e:
                logger.warning(f"Failed to generate plot for {col}: {e}")
                
    # Categorical distributions
    if categorical_cols:
        for col in categorical_cols:
            try:
                # Get value counts
                prof = profile["columns"][col]
                if "top_values" in prof:
                    values = [v["value"] for v in prof["top_values"]]
                    counts = [v["count"] for v in prof["top_values"]]
                    
                    # Create bar chart
                    fig = go.Figure()
                    fig.add_trace(go.Bar(
                        x=values,
                        y=counts,
                        marker_color='lightgreen',
                        marker_line_color='darkgreen',
                        marker_line_width=1,
                    ))
                    
                    fig.update_layout(
                        title=f"Value Counts for {col}",
                        xaxis_title=col,
                        yaxis_title="Count",
                        template="plotly_white",
                        width=800,
                        height=400,
                    )
                    
                    plots[f"{col}_value_counts"] = fig.to_json()
                    
            except Exception as e:
                logger.warning(f"Failed to generate plot for {col}: {e}")
                
    # Correlation heatmap
    if len(numeric_cols) >= 2 and profile.get("correlations"):
        try:
            # Build correlation matrix
            corr_matrix = {}
            for key, value in profile["correlations"].items():
                col1, col2 = key.split("__")
                if col1 not in corr_matrix:
                    corr_matrix[col1] = {}
                if col2 not in corr_matrix:
                    corr_matrix[col2] = {}
                corr_matrix[col1][col2] = value
                corr_matrix[col2][col1] = value
                
            # Convert to DataFrame
            corr_df = pd.DataFrame(corr_matrix).fillna(0)
            
            # Create heatmap
            fig = go.Figure(data=go.Heatmap(
                z=corr_df.values,
                x=corr_df.columns,
                y=corr_df.index,
                colorscale='RdBu',
                zmid=0,
                text=corr_df.values.round(2),
                texttemplate='%{text}',
                textfont={"size": 10},
            ))
            
            fig.update_layout(
                title="Correlation Heatmap",
                template="plotly_white",
                width=800,
                height=800,
            )
            
            plots["correlation_heatmap"] = fig.to_json()
            
        except Exception as e:
            logger.warning(f"Failed to generate correlation heatmap: {e}")
            
    return plots


def _display_profile_summary(profile: Dict[str, Any]):
    """Display profile summary in console."""
    console.print(Panel(
        f"[bold]Table Profile Summary[/bold]\n"
        f"Rows: {profile['metadata']['row_count']:,}\n"
        f"Columns: {profile['metadata']['column_count']}\n"
        f"Sample Size: {profile['metadata']['sample_size']:,}",
        title=profile["table_name"],
        border_style="green",
    ))
    
    # Column summaries
    console.print("\n[bold]Column Summaries:[/bold]")
    
    for col_name, col_prof in profile["columns"].items():
        if col_prof["type"] == "numeric":
            console.print(
                f"\n[cyan]{col_name}[/cyan] (numeric):\n"
                f"  Range: [{col_prof.get('min', 'N/A')}, {col_prof.get('max', 'N/A')}]\n"
                f"  Mean: {col_prof.get('mean', 'N/A'):.2f} (±{col_prof.get('std', 0):.2f})\n"
                f"  Missing: {col_prof['missing_pct']:.1f}%"
            )
            if "outliers" in col_prof:
                console.print(f"  Outliers: {col_prof['outlier_pct']:.1f}%")
        else:
            console.print(
                f"\n[cyan]{col_name}[/cyan] (categorical):\n"
                f"  Unique: {col_prof['unique']:,}\n"
                f"  Missing: {col_prof['missing_pct']:.1f}%"
            )
            if col_prof.get("top_values"):
                top_value = col_prof["top_values"][0]
                console.print(
                    f"  Most common: '{top_value['value']}' ({top_value['count']:,} occurrences)"
                )


def _save_profile_report(
    profile: Dict[str, Any],
    output_path: Path,
    format: str,
    include_plots: bool,
):
    """Save profile report to file."""
    if format == "json":
        # Save as JSON
        with open(f"{output_path}.json", "w") as f:
            json.dump(profile, f, indent=2, default=str)
            
    elif format == "markdown":
        # Generate markdown report
        md_content = _generate_markdown_report(profile)
        with open(f"{output_path}.md", "w") as f:
            f.write(md_content)
            
    else:  # HTML
        # Generate HTML report
        html_content = _generate_html_report(profile, include_plots)
        with open(f"{output_path}.html", "w") as f:
            f.write(html_content)


def _generate_markdown_report(profile: Dict[str, Any]) -> str:
    """Generate markdown report."""
    lines = []
    
    # Header
    lines.append(f"# Data Profile Report: {profile['table_name']}\n")
    lines.append(f"Generated at: {profile['generated_at']}\n")
    
    # Metadata
    lines.append("## Dataset Overview\n")
    lines.append(f"- **Rows:** {profile['metadata']['row_count']:,}\n")
    lines.append(f"- **Columns:** {profile['metadata']['column_count']}\n")
    lines.append(f"- **Sample Size:** {profile['metadata']['sample_size']:,}\n")
    
    # Column profiles
    lines.append("\n## Column Profiles\n")
    
    for col_name, col_prof in profile["columns"].items():
        lines.append(f"\n### {col_name}\n")
        lines.append(f"- **Type:** {col_prof['type']}\n")
        lines.append(f"- **Missing:** {col_prof['missing']:,} ({col_prof['missing_pct']:.1f}%)\n")
        lines.append(f"- **Unique:** {col_prof['unique']:,}\n")
        
        if col_prof["type"] == "numeric":
            lines.append(f"- **Mean:** {col_prof.get('mean', 'N/A')}\n")
            lines.append(f"- **Std:** {col_prof.get('std', 'N/A')}\n")
            lines.append(f"- **Min:** {col_prof.get('min', 'N/A')}\n")
            lines.append(f"- **Max:** {col_prof.get('max', 'N/A')}\n")
            
            if "outliers" in col_prof:
                lines.append(f"- **Outliers:** {col_prof['outliers']:,} ({col_prof['outlier_pct']:.1f}%)\n")
                
    return "".join(lines)


def _generate_html_report(profile: Dict[str, Any], include_plots: bool) -> str:
    """Generate HTML report with embedded plots."""
    html_template = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Data Profile Report: {title}</title>
        <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 40px; }}
            h1, h2, h3 {{ color: #333; }}
            .metadata {{ background: #f5f5f5; padding: 20px; border-radius: 5px; }}
            .column-profile {{ margin: 20px 0; padding: 15px; border: 1px solid #ddd; }}
            .numeric {{ background: #e6f3ff; }}
            .categorical {{ background: #ffe6e6; }}
            .plot-container {{ margin: 20px 0; }}
            table {{ border-collapse: collapse; width: 100%; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            th {{ background: #f2f2f2; }}
        </style>
    </head>
    <body>
        <h1>Data Profile Report: {title}</h1>
        <p>Generated at: {generated_at}</p>
        
        <div class="metadata">
            <h2>Dataset Overview</h2>
            <ul>
                <li><strong>Rows:</strong> {row_count:,}</li>
                <li><strong>Columns:</strong> {column_count}</li>
                <li><strong>Sample Size:</strong> {sample_size:,}</li>
            </ul>
        </div>
        
        <h2>Column Profiles</h2>
        {column_profiles}
        
        {plots_section}
    </body>
    </html>
    """
    
    # Generate column profiles HTML
    column_profiles_html = []
    for col_name, col_prof in profile["columns"].items():
        profile_class = col_prof["type"]
        profile_html = f"""
        <div class="column-profile {profile_class}">
            <h3>{col_name}</h3>
            <p><strong>Type:</strong> {col_prof['type']}</p>
            <p><strong>Missing:</strong> {col_prof['missing']:,} ({col_prof['missing_pct']:.1f}%)</p>
            <p><strong>Unique:</strong> {col_prof['unique']:,}</p>
        """
        
        if col_prof["type"] == "numeric":
            profile_html += f"""
            <table>
                <tr><th>Statistic</th><th>Value</th></tr>
                <tr><td>Mean</td><td>{col_prof.get('mean', 'N/A'):.2f}</td></tr>
                <tr><td>Std</td><td>{col_prof.get('std', 'N/A'):.2f}</td></tr>
                <tr><td>Min</td><td>{col_prof.get('min', 'N/A')}</td></tr>
                <tr><td>Max</td><td>{col_prof.get('max', 'N/A')}</td></tr>
                <tr><td>Q1</td><td>{col_prof.get('q1', 'N/A')}</td></tr>
                <tr><td>Median</td><td>{col_prof.get('median', 'N/A')}</td></tr>
                <tr><td>Q3</td><td>{col_prof.get('q3', 'N/A')}</td></tr>
            </table>
            """
            
        profile_html += "</div>"
        column_profiles_html.append(profile_html)
        
    # Generate plots section
    plots_html = ""
    if include_plots and profile.get("plots"):
        plots_html = "<h2>Visualizations</h2>"
        for plot_name, plot_json in profile["plots"].items():
            plot_id = plot_name.replace(" ", "_")
            plots_html += f"""
            <div class="plot-container">
                <div id="{plot_id}"></div>
                <script>
                    Plotly.newPlot('{plot_id}', {plot_json});
                </script>
            </div>
            """
            
    return html_template.format(
        title=profile["table_name"],
        generated_at=profile["generated_at"],
        row_count=profile["metadata"]["row_count"],
        column_count=profile["metadata"]["column_count"],
        sample_size=profile["metadata"]["sample_size"],
        column_profiles="\n".join(column_profiles_html),
        plots_section=plots_html,
    )


def _save_combined_report(
    all_profiles: Dict[str, Dict[str, Any]],
    output_dir: Path,
    format: str,
):
    """Save combined report for multiple tables."""
    combined = {
        "generated_at": datetime.now().isoformat(),
        "tables": all_profiles,
    }
    
    if format == "json":
        with open(output_dir / "combined_profile.json", "w") as f:
            json.dump(combined, f, indent=2, default=str)
    else:
        # Create index HTML
        index_html = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Data Profile Reports</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 40px; }
                h1 { color: #333; }
                .table-list { list-style: none; padding: 0; }
                .table-list li { margin: 10px 0; }
                a { color: #0066cc; text-decoration: none; }
                a:hover { text-decoration: underline; }
            </style>
        </head>
        <body>
            <h1>Data Profile Reports</h1>
            <p>Generated at: {generated_at}</p>
            <ul class="table-list">
                {table_links}
            </ul>
        </body>
        </html>
        """
        
        table_links = []
        for table_name in all_profiles:
            table_links.append(
                f'<li><a href="{table_name}_profile.html">{table_name}</a></li>'
            )
            
        with open(output_dir / "index.html", "w") as f:
            f.write(index_html.format(
                generated_at=combined["generated_at"],
                table_links="\n".join(table_links),
            ))