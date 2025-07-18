"""Submission history command."""

from pathlib import Path
from typing import Optional
import typer
import sys
import json
from datetime import datetime

from ...utils import (
    get_console, print_error, print_info, print_success,
    handle_errors
)
from ...utils.console import create_table

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

@handle_errors
def history_command(
    competition: str = typer.Argument(..., help="Competition ID (e.g., titanic)"),
    limit: int = typer.Option(10, "--limit", "-l", help="Number of submissions to show"),
    report: Optional[Path] = typer.Option(None, "--report", "-r", help="Generate detailed report (JSON)"),
    show_scores: bool = typer.Option(True, "--scores/--no-scores", help="Show public/private scores"),
    show_files: bool = typer.Option(False, "--files", help="Show submission file names"),
):
    """View submission history for a competition.
    
    Shows your recent submissions to a Kaggle competition, including
    scores, submission times, and optionally the files used.
    
    Examples:
        # View last 10 submissions
        bert kaggle history titanic
        
        # View more submissions with file names
        bert kaggle history titanic --limit 20 --files
        
        # Generate detailed report
        bert kaggle history titanic --report submissions_report.json
        
        # View without scores (just submission info)
        bert kaggle history titanic --no-scores
    """
    console = get_console()
    
    console.print(f"\n[bold blue]Submission History: {competition}[/bold blue]")
    console.print("=" * 60)
    
    try:
        from utils.kaggle_integration import KaggleIntegration
    except ImportError:
        print_error(
            "Failed to import Kaggle integration. Make sure kaggle is installed:\n"
            "pip install kaggle",
            title="Import Error"
        )
        raise typer.Exit(1)
    
    try:
        kaggle = KaggleIntegration()
        
        # Get submission history
        with console.status("[yellow]Fetching submission history...[/yellow]"):
            submissions = kaggle.get_submission_history(competition, limit=limit)
        
        if not submissions:
            print_info("No submissions found for this competition.")
            return
        
        # Create submissions table
        columns = ["#", "Date", "Description", "Status"]
        if show_scores:
            columns.extend(["Public Score", "Private Score"])
        if show_files:
            columns.append("File")
        
        hist_table = create_table(f"Last {min(limit, len(submissions))} Submissions", columns)
        
        # Track best scores
        best_public = float('-inf')
        best_private = float('-inf')
        best_public_idx = -1
        best_private_idx = -1
        
        # Add rows
        for i, sub in enumerate(submissions[:limit]):
            # Parse submission data
            sub_date = sub.get("date", "")
            if sub_date:
                try:
                    dt = datetime.fromisoformat(sub_date.replace("Z", "+00:00"))
                    date_str = dt.strftime("%Y-%m-%d %H:%M")
                except:
                    date_str = sub_date[:16]
            else:
                date_str = "Unknown"
            
            description = sub.get("description", "No description")[:40]
            status = sub.get("status", "unknown")
            
            # Format status
            if status == "complete":
                status_str = "[green]Complete[/green]"
            elif status == "pending":
                status_str = "[yellow]Pending[/yellow]"
            elif status == "error":
                status_str = "[red]Error[/red]"
            else:
                status_str = status
            
            row = [str(i + 1), date_str, description, status_str]
            
            # Add scores if requested
            if show_scores:
                public_score = sub.get("publicScore")
                private_score = sub.get("privateScore")
                
                # Track best scores
                if public_score is not None:
                    if float(public_score) > best_public:
                        best_public = float(public_score)
                        best_public_idx = i
                
                if private_score is not None:
                    if float(private_score) > best_private:
                        best_private = float(private_score)
                        best_private_idx = i
                
                # Format scores
                if public_score is not None:
                    if i == best_public_idx:
                        public_str = f"[bold green]{public_score}[/bold green]"
                    else:
                        public_str = str(public_score)
                else:
                    public_str = "-"
                
                if private_score is not None:
                    if i == best_private_idx:
                        private_str = f"[bold green]{private_score}[/bold green]"
                    else:
                        private_str = str(private_score)
                else:
                    private_str = "-"
                
                row.extend([public_str, private_str])
            
            # Add file name if requested
            if show_files:
                file_name = sub.get("fileName", "Unknown")
                row.append(file_name)
            
            hist_table.add_row(*row)
        
        console.print(hist_table)
        
        # Show statistics
        console.print(f"\n[cyan]Total Submissions: {len(submissions)}[/cyan]")
        
        if show_scores and best_public > float('-inf'):
            console.print(f"[cyan]Best Public Score: {best_public}[/cyan]")
            if best_private > float('-inf'):
                console.print(f"[cyan]Best Private Score: {best_private}[/cyan]")
        
        # Calculate submission frequency
        if len(submissions) > 1:
            first_date = submissions[-1].get("date", "")
            last_date = submissions[0].get("date", "")
            if first_date and last_date:
                try:
                    first_dt = datetime.fromisoformat(first_date.replace("Z", "+00:00"))
                    last_dt = datetime.fromisoformat(last_date.replace("Z", "+00:00"))
                    days = (last_dt - first_dt).days
                    if days > 0:
                        freq = len(submissions) / days
                        console.print(f"[cyan]Submission Frequency: {freq:.2f} per day[/cyan]")
                except:
                    pass
        
        # Generate report if requested
        if report:
            report_data = {
                "competition": competition,
                "generated": datetime.now().isoformat(),
                "total_submissions": len(submissions),
                "submissions": submissions[:limit],
                "statistics": {
                    "best_public_score": best_public if best_public > float('-inf') else None,
                    "best_private_score": best_private if best_private > float('-inf') else None,
                }
            }
            
            with open(report, "w") as f:
                json.dump(report_data, f, indent=2)
            
            print_success(f"Report saved to: {report}")
        
        # Show next steps
        console.print("\n[bold]Next steps:[/bold]")
        console.print(f"1. View leaderboard position: [cyan]bert kaggle leaderboard {competition}[/cyan]")
        console.print(f"2. Make new submission: [cyan]bert kaggle submit {competition} predictions.csv[/cyan]")
        console.print(f"3. Download best submission: [cyan]bert kaggle download {competition} --file submission.csv[/cyan]")
        
    except Exception as e:
        print_error(f"Failed to fetch submission history: {str(e)}", title="History Error")
        raise typer.Exit(1)