"""Validation report generation and formatting."""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

import pandas as pd
# from loguru import logger  # Domain should not depend on logging framework


class ValidationSeverity(Enum):
    """Severity levels for validation results."""
    
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class ValidationResult:
    """Result from a single validation check."""
    
    check_name: str
    passed: bool
    message: str
    severity: ValidationSeverity
    timestamp: datetime = field(default_factory=datetime.now)
    details: Optional[Dict[str, Any]] = None
    affected_fields: Optional[List[str]] = None
    affected_rows: Optional[List[int]] = None


@dataclass
class ValidationReport:
    """Complete validation report."""
    
    dataset_name: str
    timestamp: datetime = field(default_factory=datetime.now)
    results: List[ValidationResult] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def add_result(self, result: ValidationResult) -> None:
        """Add a validation result to the report."""
        self.results.append(result)
        # logger.debug(f"Added validation result: {result.check_name} - {result.passed}")
    
    @property
    def passed(self) -> bool:
        """Check if all validations passed."""
        return all(r.passed or r.severity in [ValidationSeverity.INFO, ValidationSeverity.WARNING] 
                  for r in self.results)
    
    @property
    def error_count(self) -> int:
        """Count of error-level failures."""
        return sum(1 for r in self.results 
                  if not r.passed and r.severity in [ValidationSeverity.ERROR, ValidationSeverity.CRITICAL])
    
    @property
    def warning_count(self) -> int:
        """Count of warning-level issues."""
        return sum(1 for r in self.results 
                  if not r.passed and r.severity == ValidationSeverity.WARNING)
    
    def get_summary(self) -> Dict[str, Any]:
        """Get report summary."""
        return {
            "dataset_name": self.dataset_name,
            "timestamp": self.timestamp.isoformat(),
            "total_checks": len(self.results),
            "passed_checks": sum(1 for r in self.results if r.passed),
            "failed_checks": sum(1 for r in self.results if not r.passed),
            "error_count": self.error_count,
            "warning_count": self.warning_count,
            "overall_passed": self.passed,
        }
    
    def get_failed_checks(self) -> List[ValidationResult]:
        """Get all failed validation results."""
        return [r for r in self.results if not r.passed]
    
    def get_by_severity(self, severity: ValidationSeverity) -> List[ValidationResult]:
        """Get results by severity level."""
        return [r for r in self.results if r.severity == severity]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert report to dictionary."""
        return {
            "dataset_name": self.dataset_name,
            "timestamp": self.timestamp.isoformat(),
            "summary": self.get_summary(),
            "results": [
                {
                    "check_name": r.check_name,
                    "passed": r.passed,
                    "message": r.message,
                    "severity": r.severity.value,
                    "timestamp": r.timestamp.isoformat(),
                    "details": r.details,
                    "affected_fields": r.affected_fields,
                    "affected_rows": r.affected_rows,
                }
                for r in self.results
            ],
            "metadata": self.metadata,
        }
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convert report to DataFrame."""
        data = []
        for r in self.results:
            data.append({
                "check_name": r.check_name,
                "passed": r.passed,
                "severity": r.severity.value,
                "message": r.message,
                "timestamp": r.timestamp,
                "affected_fields": ", ".join(r.affected_fields) if r.affected_fields else None,
                "affected_rows_count": len(r.affected_rows) if r.affected_rows else 0,
            })
        return pd.DataFrame(data)


class ReportFormatter:
    """Format validation reports for different outputs."""
    
    @staticmethod
    def format_console(report: ValidationReport, verbose: bool = False) -> str:
        """Format report for console output.
        
        Args:
            report: Validation report
            verbose: Include detailed information
            
        Returns:
            Formatted string
        """
        lines = []
        
        # Header
        lines.append("=" * 80)
        lines.append(f"VALIDATION REPORT: {report.dataset_name}")
        lines.append(f"Timestamp: {report.timestamp}")
        lines.append("=" * 80)
        
        # Summary
        summary = report.get_summary()
        lines.append("\nSUMMARY:")
        lines.append(f"  Total Checks: {summary['total_checks']}")
        lines.append(f"  Passed: {summary['passed_checks']}")
        lines.append(f"  Failed: {summary['failed_checks']}")
        lines.append(f"  Errors: {summary['error_count']}")
        lines.append(f"  Warnings: {summary['warning_count']}")
        lines.append(f"  Overall Status: {'PASSED' if summary['overall_passed'] else 'FAILED'}")
        
        # Results by severity
        for severity in ValidationSeverity:
            results = report.get_by_severity(severity)
            if results:
                lines.append(f"\n{severity.value.upper()}S ({len(results)}):")
                for r in results:
                    status = "✓" if r.passed else "✗"
                    lines.append(f"  {status} {r.check_name}: {r.message}")
                    
                    if verbose and r.details:
                        for key, value in r.details.items():
                            lines.append(f"      {key}: {value}")
                    
                    if verbose and r.affected_fields:
                        lines.append(f"      Affected fields: {', '.join(r.affected_fields)}")
                    
                    if verbose and r.affected_rows:
                        lines.append(f"      Affected rows: {len(r.affected_rows)} rows")
        
        # Footer
        lines.append("\n" + "=" * 80)
        
        return "\n".join(lines)
    
    @staticmethod
    def format_markdown(report: ValidationReport) -> str:
        """Format report as Markdown.
        
        Args:
            report: Validation report
            
        Returns:
            Markdown formatted string
        """
        lines = []
        
        # Header
        lines.append(f"# Validation Report: {report.dataset_name}")
        lines.append(f"\n**Timestamp:** {report.timestamp}")
        
        # Summary
        summary = report.get_summary()
        lines.append("\n## Summary")
        lines.append("")
        lines.append("| Metric | Value |")
        lines.append("|--------|-------|")
        lines.append(f"| Total Checks | {summary['total_checks']} |")
        lines.append(f"| Passed | {summary['passed_checks']} |")
        lines.append(f"| Failed | {summary['failed_checks']} |")
        lines.append(f"| Errors | {summary['error_count']} |")
        lines.append(f"| Warnings | {summary['warning_count']} |")
        lines.append(f"| Overall Status | {'✅ PASSED' if summary['overall_passed'] else '❌ FAILED'} |")
        
        # Detailed results
        lines.append("\n## Detailed Results")
        
        for severity in ValidationSeverity:
            results = report.get_by_severity(severity)
            if results:
                lines.append(f"\n### {severity.value.title()}s")
                lines.append("")
                lines.append("| Check | Status | Message |")
                lines.append("|-------|--------|---------|")
                
                for r in results:
                    status = "✅" if r.passed else "❌"
                    message = r.message.replace("|", "\\|")  # Escape pipes
                    lines.append(f"| {r.check_name} | {status} | {message} |")
        
        # Metadata
        if report.metadata:
            lines.append("\n## Metadata")
            lines.append("")
            for key, value in report.metadata.items():
                lines.append(f"- **{key}:** {value}")
        
        return "\n".join(lines)
    
    @staticmethod
    def format_html(report: ValidationReport) -> str:
        """Format report as HTML.
        
        Args:
            report: Validation report
            
        Returns:
            HTML formatted string
        """
        summary = report.get_summary()
        status_class = "success" if summary['overall_passed'] else "danger"
        
        html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Validation Report: {report.dataset_name}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        .header {{ background-color: #f0f0f0; padding: 20px; border-radius: 5px; }}
        .summary {{ margin: 20px 0; }}
        .summary table {{ border-collapse: collapse; width: 100%; }}
        .summary th, .summary td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        .summary th {{ background-color: #4CAF50; color: white; }}
        .results {{ margin: 20px 0; }}
        .success {{ color: green; }}
        .danger {{ color: red; }}
        .warning {{ color: orange; }}
        .info {{ color: blue; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>Validation Report: {report.dataset_name}</h1>
        <p><strong>Timestamp:</strong> {report.timestamp}</p>
        <p><strong>Overall Status:</strong> <span class="{status_class}">{'PASSED' if summary['overall_passed'] else 'FAILED'}</span></p>
    </div>
    
    <div class="summary">
        <h2>Summary</h2>
        <table>
            <tr><th>Metric</th><th>Value</th></tr>
            <tr><td>Total Checks</td><td>{summary['total_checks']}</td></tr>
            <tr><td>Passed</td><td class="success">{summary['passed_checks']}</td></tr>
            <tr><td>Failed</td><td class="danger">{summary['failed_checks']}</td></tr>
            <tr><td>Errors</td><td class="danger">{summary['error_count']}</td></tr>
            <tr><td>Warnings</td><td class="warning">{summary['warning_count']}</td></tr>
        </table>
    </div>
    
    <div class="results">
        <h2>Detailed Results</h2>
"""
        
        for severity in ValidationSeverity:
            results = report.get_by_severity(severity)
            if results:
                html += f"<h3>{severity.value.title()}s</h3><ul>"
                for r in results:
                    status = "✓" if r.passed else "✗"
                    html += f'<li class="{severity.value}">{status} <strong>{r.check_name}:</strong> {r.message}</li>'
                html += "</ul>"
        
        html += """
    </div>
</body>
</html>
"""
        return html