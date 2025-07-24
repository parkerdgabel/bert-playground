"""Console output formatters."""

from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta


class MetricsFormatter:
    """Format metrics for console display."""
    
    @staticmethod
    def format_metric_value(value: Any, precision: int = 4) -> str:
        """Format a metric value for display.
        
        Args:
            value: Metric value
            precision: Decimal precision for floats
            
        Returns:
            Formatted string
        """
        if isinstance(value, float):
            if value >= 1000:
                return f"{value:.0f}"
            elif value >= 1:
                return f"{value:.{precision}f}"
            else:
                # Scientific notation for very small values
                return f"{value:.{precision}e}"
        elif isinstance(value, int):
            if value >= 1_000_000:
                return f"{value/1_000_000:.1f}M"
            elif value >= 1_000:
                return f"{value/1_000:.1f}K"
            else:
                return str(value)
        elif isinstance(value, timedelta):
            total_seconds = value.total_seconds()
            if total_seconds >= 3600:
                hours = int(total_seconds // 3600)
                minutes = int((total_seconds % 3600) // 60)
                return f"{hours}h {minutes}m"
            elif total_seconds >= 60:
                minutes = int(total_seconds // 60)
                seconds = int(total_seconds % 60)
                return f"{minutes}m {seconds}s"
            else:
                return f"{total_seconds:.1f}s"
        else:
            return str(value)
    
    @staticmethod
    def format_metrics_line(metrics: Dict[str, Any], max_items: int = 6) -> str:
        """Format metrics as a single line.
        
        Args:
            metrics: Dictionary of metrics
            max_items: Maximum number of items to show
            
        Returns:
            Formatted string
        """
        formatted_items = []
        
        # Sort metrics by importance (loss first, then alphabetically)
        sorted_metrics = sorted(
            metrics.items(),
            key=lambda x: (0 if "loss" in x[0] else 1, x[0])
        )
        
        for i, (name, value) in enumerate(sorted_metrics):
            if i >= max_items:
                formatted_items.append("...")
                break
            
            formatted_value = MetricsFormatter.format_metric_value(value)
            formatted_items.append(f"{name}: {formatted_value}")
        
        return " | ".join(formatted_items)
    
    @staticmethod
    def format_metrics_dict(
        metrics: Dict[str, Any],
        indent: int = 2,
        max_width: int = 80
    ) -> List[str]:
        """Format metrics as indented lines.
        
        Args:
            metrics: Dictionary of metrics
            indent: Number of spaces to indent
            max_width: Maximum line width
            
        Returns:
            List of formatted lines
        """
        lines = []
        indent_str = " " * indent
        
        # Group metrics by prefix
        grouped = {}
        for name, value in metrics.items():
            parts = name.split("/")
            if len(parts) > 1:
                group = parts[0]
                metric = "/".join(parts[1:])
            else:
                group = "metrics"
                metric = name
            
            if group not in grouped:
                grouped[group] = []
            grouped[group].append((metric, value))
        
        # Format each group
        for group, items in sorted(grouped.items()):
            if group != "metrics":
                lines.append(f"{indent_str}{group}:")
                sub_indent = indent_str + "  "
            else:
                sub_indent = indent_str
            
            for metric, value in sorted(items):
                formatted_value = MetricsFormatter.format_metric_value(value)
                line = f"{sub_indent}{metric}: {formatted_value}"
                
                # Truncate if too long
                if len(line) > max_width:
                    line = line[:max_width-3] + "..."
                
                lines.append(line)
        
        return lines


class TableFormatter:
    """Format data as tables for console display."""
    
    @staticmethod
    def format_comparison_table(
        data: Dict[str, Dict[str, Any]],
        metrics: Optional[List[str]] = None
    ) -> List[str]:
        """Format run comparison as a table.
        
        Args:
            data: Dictionary of run_id -> run_data
            metrics: Optional list of metrics to include
            
        Returns:
            List of formatted lines
        """
        lines = []
        
        if not data:
            return ["No runs to compare"]
        
        # Determine metrics to show
        if not metrics:
            # Collect all available metrics
            all_metrics = set()
            for run_data in data.values():
                if "metrics" in run_data:
                    all_metrics.update(run_data["metrics"].keys())
            metrics = sorted(all_metrics)
        
        if not metrics:
            return ["No metrics to compare"]
        
        # Calculate column widths
        run_ids = list(data.keys())
        col_widths = {
            "metric": max(len(m) for m in metrics),
            **{run_id: max(12, len(run_id)) for run_id in run_ids}
        }
        
        # Header
        header_parts = ["Metric".ljust(col_widths["metric"])]
        header_parts.extend(run_id.ljust(col_widths[run_id]) for run_id in run_ids)
        lines.append(" | ".join(header_parts))
        
        # Separator
        sep_parts = ["-" * col_widths["metric"]]
        sep_parts.extend("-" * col_widths[run_id] for run_id in run_ids)
        lines.append("-+-".join(sep_parts))
        
        # Metrics rows
        for metric in metrics:
            row_parts = [metric.ljust(col_widths["metric"])]
            
            for run_id in run_ids:
                value = data[run_id].get("metrics", {}).get(metric, "N/A")
                if value != "N/A":
                    value = MetricsFormatter.format_metric_value(value)
                row_parts.append(str(value).ljust(col_widths[run_id]))
            
            lines.append(" | ".join(row_parts))
        
        return lines
    
    @staticmethod
    def format_hyperparameters_table(params: Dict[str, Any]) -> List[str]:
        """Format hyperparameters as a table.
        
        Args:
            params: Dictionary of hyperparameters
            
        Returns:
            List of formatted lines
        """
        lines = []
        
        if not params:
            return ["No hyperparameters"]
        
        # Calculate column widths
        max_key_len = max(len(k) for k in params.keys())
        max_val_len = min(50, max(len(str(v)) for v in params.values()))
        
        # Header
        lines.append(f"{'Parameter'.ljust(max_key_len)} | {'Value'.ljust(max_val_len)}")
        lines.append(f"{'-' * max_key_len}-+-{'-' * max_val_len}")
        
        # Rows
        for key, value in sorted(params.items()):
            str_value = str(value)
            if len(str_value) > max_val_len:
                str_value = str_value[:max_val_len-3] + "..."
            lines.append(f"{key.ljust(max_key_len)} | {str_value.ljust(max_val_len)}")
        
        return lines