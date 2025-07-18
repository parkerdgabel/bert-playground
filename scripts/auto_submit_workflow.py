#!/usr/bin/env python3
"""
Automated Kaggle submission workflow.
Monitors model performance and automatically submits best models to Kaggle.
"""

import argparse
import json
import time
from pathlib import Path
from typing import Dict, Optional, List
import yaml
from datetime import datetime
from loguru import logger
from rich.console import Console
from rich.table import Table
import mlflow
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.kaggle_integration import KaggleIntegration, CompetitionConfig
from utils.kaggle_mlflow_tracking import KaggleMLflowTracker
from utils.evaluation import ModelEvaluator

console = Console()


class AutoSubmissionWorkflow:
    """Automated submission workflow for Kaggle competitions."""
    
    def __init__(self, 
                 competition_id: str,
                 config_path: Optional[Path] = None,
                 mlflow_experiment: Optional[str] = None):
        """Initialize auto-submission workflow."""
        self.competition_id = competition_id
        self.config = CompetitionConfig(competition_id, config_path)
        self.kaggle = KaggleIntegration()
        self.mlflow_tracker = KaggleMLflowTracker(
            competition_id, 
            mlflow_experiment or f"kaggle_{competition_id}_auto"
        )
        
        # Track submission history
        self.submission_history: List[Dict] = []
        self.best_score = None
        
    def find_best_models(self, 
                        output_dir: Path,
                        metric: str = "val_accuracy",
                        top_n: int = 3) -> List[Path]:
        """Find best models based on validation metrics."""
        models = []
        
        # Search for model checkpoints
        for run_dir in output_dir.glob("run_*"):
            metrics_file = run_dir / "metrics.json"
            if metrics_file.exists():
                with open(metrics_file, 'r') as f:
                    metrics = json.load(f)
                
                # Look for best model checkpoint
                best_model = None
                for checkpoint in ["best_model_accuracy", "best_model", "final_model"]:
                    checkpoint_path = run_dir / checkpoint
                    if checkpoint_path.exists():
                        best_model = checkpoint_path
                        break
                
                if best_model:
                    models.append({
                        'path': best_model,
                        'score': metrics.get(metric, 0),
                        'metrics': metrics,
                        'run_name': run_dir.name
                    })
        
        # Sort by score and return top N
        models.sort(key=lambda x: x['score'], reverse=True)
        return models[:top_n]
    
    def should_submit(self, 
                     val_score: float,
                     threshold_improvement: float = 0.001) -> bool:
        """Determine if model should be submitted based on performance."""
        if self.best_score is None:
            return True
        
        improvement = val_score - self.best_score
        return improvement > threshold_improvement
    
    def generate_submission_message(self, 
                                   model_info: Dict,
                                   template: Optional[str] = None) -> str:
        """Generate submission message based on model info."""
        if template:
            return template.format(**model_info)
        
        # Default message template
        return (
            f"MLX-BERT Auto-submission | "
            f"Val: {model_info.get('val_accuracy', 0):.4f} | "
            f"Model: {model_info.get('model_type', 'ModernBERT')} | "
            f"Run: {model_info.get('run_name', 'unknown')}"
        )
    
    def submit_model(self, 
                    model_path: Path,
                    test_data_path: Path,
                    model_info: Dict) -> Optional[Dict]:
        """Submit a single model to Kaggle."""
        try:
            # Generate predictions
            console.print(f"[cyan]Generating predictions for {model_path.name}...[/cyan]")
            
            evaluator = ModelEvaluator(model_dir=model_path)
            submission_path = Path(f"submissions/auto_{self.competition_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
            submission_path.parent.mkdir(exist_ok=True)
            
            evaluator.create_submission(
                test_csv_path=test_data_path,
                submission_path=submission_path,
                id_column=self.config.id_column,
                target_column=self.config.target_column
            )
            
            # Generate submission message
            message = self.generate_submission_message(model_info)
            
            # Submit to Kaggle
            console.print(f"[cyan]Submitting to Kaggle: {message}[/cyan]")
            
            result = self.kaggle.submit_predictions(
                competition_id=self.competition_id,
                submission_file=submission_path,
                message=message,
                track_with_mlflow=True
            )
            
            # Update best score if needed
            if result.get('score') is not None:
                score = float(result['score'])
                if self.best_score is None or score > self.best_score:
                    self.best_score = score
                    console.print(f"[bold green]New best score: {score}[/bold green]")
            
            # Track submission
            self.submission_history.append({
                'timestamp': datetime.now().isoformat(),
                'model_path': str(model_path),
                'submission_file': str(submission_path),
                'message': message,
                'result': result
            })
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to submit model: {e}")
            return None
    
    def run_auto_submission(self,
                           output_dir: Path,
                           test_data_path: Path,
                           submit_top_n: int = 1,
                           val_threshold: float = 0.001,
                           dry_run: bool = False):
        """Run automated submission workflow."""
        console.print(f"[bold]Starting auto-submission workflow for {self.competition_id}[/bold]")
        
        # Find best models
        best_models = self.find_best_models(output_dir, top_n=submit_top_n)
        
        if not best_models:
            console.print("[yellow]No models found for submission[/yellow]")
            return
        
        # Display found models
        table = Table(title="Models Found for Submission")
        table.add_column("Run", style="cyan")
        table.add_column("Validation Score", style="green")
        table.add_column("Path", style="magenta")
        table.add_column("Submit?", style="yellow")
        
        models_to_submit = []
        for model in best_models:
            should_submit = self.should_submit(model['score'], val_threshold)
            table.add_row(
                model['run_name'],
                f"{model['score']:.4f}",
                str(model['path'].relative_to(output_dir)),
                "Yes" if should_submit else "No"
            )
            if should_submit:
                models_to_submit.append(model)
        
        console.print(table)
        
        if dry_run:
            console.print("\n[yellow]Dry run mode - no submissions will be made[/yellow]")
            return
        
        # Submit selected models
        submission_results = []
        for model in models_to_submit:
            console.print(f"\n[bold]Submitting {model['run_name']}...[/bold]")
            
            result = self.submit_model(
                model_path=model['path'],
                test_data_path=test_data_path,
                model_info={
                    'val_accuracy': model['score'],
                    'run_name': model['run_name'],
                    'model_type': 'MLX-ModernBERT',
                    **model['metrics']
                }
            )
            
            if result:
                submission_results.append(result)
                # Wait between submissions to avoid rate limiting
                time.sleep(5)
        
        # Summary
        console.print("\n[bold]Submission Summary[/bold]")
        if submission_results:
            summary_table = Table()
            summary_table.add_column("Submission", style="cyan")
            summary_table.add_column("Score", style="green")
            summary_table.add_column("Status", style="yellow")
            
            for i, result in enumerate(submission_results):
                summary_table.add_row(
                    f"Submission {i+1}",
                    str(result.get('score', 'Pending')),
                    result.get('status', 'Complete')
                )
            
            console.print(summary_table)
        else:
            console.print("[yellow]No submissions were made[/yellow]")
        
        # Save submission history
        history_file = output_dir / f"submission_history_{self.competition_id}.json"
        with open(history_file, 'w') as f:
            json.dump(self.submission_history, f, indent=2)
        console.print(f"\n[green]Submission history saved to {history_file}[/green]")
    
    def monitor_and_submit(self,
                          output_dir: Path,
                          test_data_path: Path,
                          check_interval: int = 300,
                          max_submissions_per_day: int = 5):
        """Monitor for new models and submit automatically."""
        console.print(f"[bold]Monitoring {output_dir} for new models...[/bold]")
        console.print(f"Check interval: {check_interval}s")
        console.print(f"Max submissions per day: {max_submissions_per_day}")
        
        submissions_today = 0
        last_submission_date = datetime.now().date()
        processed_models = set()
        
        try:
            while True:
                # Reset daily counter
                current_date = datetime.now().date()
                if current_date != last_submission_date:
                    submissions_today = 0
                    last_submission_date = current_date
                
                # Check for new models
                best_models = self.find_best_models(output_dir, top_n=1)
                
                for model in best_models:
                    model_id = f"{model['path']}_{model['score']}"
                    
                    # Skip if already processed
                    if model_id in processed_models:
                        continue
                    
                    # Check daily limit
                    if submissions_today >= max_submissions_per_day:
                        console.print(f"[yellow]Daily submission limit reached ({max_submissions_per_day})[/yellow]")
                        break
                    
                    # Check if should submit
                    if self.should_submit(model['score']):
                        console.print(f"\n[bold green]New model found with score {model['score']:.4f}![/bold green]")
                        
                        result = self.submit_model(
                            model_path=model['path'],
                            test_data_path=test_data_path,
                            model_info={
                                'val_accuracy': model['score'],
                                'run_name': model['run_name'],
                                'model_type': 'MLX-ModernBERT',
                                **model['metrics']
                            }
                        )
                        
                        if result:
                            submissions_today += 1
                            processed_models.add(model_id)
                
                # Wait before next check
                console.print(f"\n[dim]Next check in {check_interval}s... (Press Ctrl+C to stop)[/dim]")
                time.sleep(check_interval)
                
        except KeyboardInterrupt:
            console.print("\n[yellow]Monitoring stopped by user[/yellow]")


def main():
    parser = argparse.ArgumentParser(description="Automated Kaggle submission workflow")
    parser.add_argument("competition", help="Competition ID (e.g., titanic)")
    parser.add_argument("--output-dir", type=Path, default=Path("output"), 
                      help="Directory containing model outputs")
    parser.add_argument("--test-data", type=Path, required=True,
                      help="Path to test data CSV")
    parser.add_argument("--config", type=Path, help="Competition config file")
    parser.add_argument("--top-n", type=int, default=1,
                      help="Submit top N models")
    parser.add_argument("--val-threshold", type=float, default=0.001,
                      help="Minimum improvement threshold for submission")
    parser.add_argument("--monitor", action="store_true",
                      help="Enable continuous monitoring mode")
    parser.add_argument("--check-interval", type=int, default=300,
                      help="Check interval in seconds (for monitor mode)")
    parser.add_argument("--dry-run", action="store_true",
                      help="Dry run - don't actually submit")
    
    args = parser.parse_args()
    
    # Initialize workflow
    workflow = AutoSubmissionWorkflow(
        competition_id=args.competition,
        config_path=args.config
    )
    
    if args.monitor:
        # Continuous monitoring mode
        workflow.monitor_and_submit(
            output_dir=args.output_dir,
            test_data_path=args.test_data,
            check_interval=args.check_interval
        )
    else:
        # One-time submission
        workflow.run_auto_submission(
            output_dir=args.output_dir,
            test_data_path=args.test_data,
            submit_top_n=args.top_n,
            val_threshold=args.val_threshold,
            dry_run=args.dry_run
        )


if __name__ == "__main__":
    main()