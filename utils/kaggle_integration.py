"""
Comprehensive Kaggle API integration for MLX BERT playground.
Provides competition management, submission tracking, and leaderboard integration.
"""

import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import pandas as pd
from loguru import logger
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn
import kaggle
from kaggle.api.kaggle_api_extended import KaggleApi
import mlflow

console = Console()


class KaggleIntegration:
    """Comprehensive Kaggle API wrapper for competition management."""
    
    def __init__(self, cache_dir: Optional[Path] = None):
        """Initialize Kaggle API client."""
        self.api = KaggleApi()
        self.api.authenticate()
        self.cache_dir = cache_dir or Path.home() / ".cache" / "kaggle"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
    def list_competitions(self, 
                         category: Optional[str] = None,
                         sort_by: str = "latestDeadline",
                         page: int = 1,
                         search: Optional[str] = None) -> pd.DataFrame:
        """List Kaggle competitions with filtering options."""
        competitions = self.api.competitions_list(
            category=category,
            sort_by=sort_by,
            page=page,
            search=search
        )
        
        # Convert to DataFrame for easier manipulation
        comp_data = []
        for comp in competitions:
            comp_data.append({
                'id': getattr(comp, 'ref', ''),
                'title': getattr(comp, 'title', ''),
                'description': comp.description[:100] + '...' if hasattr(comp, 'description') and len(comp.description) > 100 else getattr(comp, 'description', ''),
                'evaluationMetric': getattr(comp, 'evaluation_metric', 'N/A'),
                'isKernelsSubmissionsOnly': getattr(comp, 'is_kernels_submissions_only', False),
                'deadline': getattr(comp, 'deadline', None),
                'maxTeamSize': getattr(comp, 'max_team_size', 1),
                'reward': getattr(comp, 'reward', None),
                'userHasEntered': getattr(comp, 'user_has_entered', False),
                'numTeams': getattr(comp, 'team_count', 0),
                'url': getattr(comp, 'url', '')
            })
        
        return pd.DataFrame(comp_data)
    
    def get_competition_details(self, competition_id: str) -> Dict[str, Any]:
        """Get detailed information about a specific competition."""
        # Download competition data files list
        files = self.api.competition_list_files(competition_id)
        
        # Get leaderboard
        leaderboard = self.get_leaderboard(competition_id, download=False)
        
        details = {
            'id': competition_id,
            'files': [{'name': f.name, 'size': f.size} for f in files],
            'leaderboard_count': len(leaderboard) if leaderboard is not None else 0,
            'cached_at': datetime.now().isoformat()
        }
        
        # Cache competition details
        cache_file = self.cache_dir / f"{competition_id}_details.json"
        with open(cache_file, 'w') as f:
            json.dump(details, f, indent=2)
            
        return details
    
    def download_competition_data(self, 
                                 competition_id: str,
                                 path: Optional[Path] = None,
                                 unzip: bool = True) -> Path:
        """Download competition data files."""
        path = path or Path.cwd() / "data" / competition_id
        path.mkdir(parents=True, exist_ok=True)
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task(f"Downloading {competition_id} data...", total=None)
            
            self.api.competition_download_files(
                competition_id, 
                path=str(path),
                unzip=unzip
            )
            
            progress.update(task, completed=True)
        
        logger.info(f"Downloaded competition data to {path}")
        return path
    
    def submit_predictions(self,
                          competition_id: str,
                          submission_file: Path,
                          message: str,
                          track_with_mlflow: bool = True) -> Dict[str, Any]:
        """Submit predictions to a competition."""
        if not submission_file.exists():
            raise FileNotFoundError(f"Submission file not found: {submission_file}")
        
        # Record submission with MLflow if enabled
        if track_with_mlflow and mlflow.active_run():
            mlflow.log_artifact(str(submission_file), "submissions")
            mlflow.log_param("kaggle_competition", competition_id)
            mlflow.log_param("submission_message", message)
            mlflow.log_param("submission_time", datetime.now().isoformat())
        
        # Submit to Kaggle
        start_time = time.time()
        try:
            result = self.api.competition_submit(
                file_name=str(submission_file),
                message=message,
                competition=competition_id
            )
            
            submission_time = time.time() - start_time
            
            # Get submission status
            submissions = self.api.competitions_submissions_list(competition_id)
            latest = submissions[0] if submissions else None
            
            submission_info = {
                'competition': competition_id,
                'file': str(submission_file),
                'message': message,
                'submission_time': submission_time,
                'timestamp': datetime.now().isoformat(),
                'status': 'complete' if latest else 'pending',
                'score': latest.publicScore if latest and hasattr(latest, 'publicScore') else None
            }
            
            # Log to MLflow
            if track_with_mlflow and mlflow.active_run():
                if submission_info['score'] is not None:
                    mlflow.log_metric("kaggle_public_score", float(submission_info['score']))
                mlflow.log_metric("submission_time_seconds", submission_time)
            
            # Save submission history
            self._save_submission_history(submission_info)
            
            logger.success(f"Successfully submitted to {competition_id}")
            return submission_info
            
        except Exception as e:
            logger.error(f"Submission failed: {e}")
            raise
    
    def get_leaderboard(self, 
                       competition_id: str,
                       download: bool = True) -> Optional[pd.DataFrame]:
        """Get competition leaderboard."""
        try:
            if download:
                leaderboard = self.api.competition_leaderboard_download(competition_id)
                return pd.DataFrame(leaderboard)
            else:
                # For basic info, use view
                leaderboard = self.api.competition_leaderboard_view(competition_id)
                return pd.DataFrame([
                    {
                        'teamName': entry.teamName,
                        'score': entry.score,
                        'rank': idx + 1
                    }
                    for idx, entry in enumerate(leaderboard.submissions)
                ])
        except Exception as e:
            logger.warning(f"Could not fetch leaderboard: {e}")
            return None
    
    def get_submissions_history(self, 
                               competition_id: str,
                               limit: int = 100) -> pd.DataFrame:
        """Get submission history for a competition."""
        submissions = self.api.competitions_submissions_list(competition_id)
        
        sub_data = []
        for sub in submissions[:limit]:
            sub_data.append({
                'date': sub.date,
                'description': sub.description,
                'publicScore': sub.publicScore if hasattr(sub, 'publicScore') else None,
                'privateScore': sub.privateScore if hasattr(sub, 'privateScore') else None,
                'fileName': sub.fileName,
                'status': sub.status
            })
        
        return pd.DataFrame(sub_data)
    
    def display_leaderboard(self, 
                           competition_id: str,
                           top_n: int = 20,
                           highlight_user: bool = True):
        """Display competition leaderboard in a formatted table."""
        leaderboard = self.get_leaderboard(competition_id)
        
        if leaderboard is None or leaderboard.empty:
            console.print("[yellow]No leaderboard data available[/yellow]")
            return
        
        # Create rich table
        table = Table(title=f"Leaderboard: {competition_id}")
        table.add_column("Rank", style="cyan", no_wrap=True)
        table.add_column("Team", style="magenta")
        table.add_column("Score", style="green")
        
        # Display top N entries
        for _, row in leaderboard.head(top_n).iterrows():
            table.add_row(
                str(row.get('rank', 'N/A')),
                row.get('teamName', 'Unknown'),
                str(row.get('score', 'N/A'))
            )
        
        console.print(table)
    
    def create_submission_report(self, 
                                competition_id: str,
                                output_file: Optional[Path] = None) -> Dict[str, Any]:
        """Create a comprehensive submission report."""
        # Get submission history
        submissions = self.get_submissions_history(competition_id)
        
        # Get current leaderboard position
        leaderboard = self.get_leaderboard(competition_id, download=False)
        
        report = {
            'competition': competition_id,
            'generated_at': datetime.now().isoformat(),
            'total_submissions': len(submissions),
            'best_public_score': submissions['publicScore'].max() if not submissions.empty else None,
            'latest_submission': submissions.iloc[0].to_dict() if not submissions.empty else None,
            'submission_history': submissions.to_dict('records'),
            'current_rank': None  # TODO: Extract user's rank from leaderboard
        }
        
        if output_file:
            output_file.parent.mkdir(parents=True, exist_ok=True)
            with open(output_file, 'w') as f:
                json.dump(report, f, indent=2)
            logger.info(f"Submission report saved to {output_file}")
        
        return report
    
    def _save_submission_history(self, submission_info: Dict[str, Any]):
        """Save submission to local history file."""
        history_file = self.cache_dir / "submission_history.jsonl"
        
        with open(history_file, 'a') as f:
            f.write(json.dumps(submission_info) + '\n')
    
    def get_local_submission_history(self) -> pd.DataFrame:
        """Load local submission history."""
        history_file = self.cache_dir / "submission_history.jsonl"
        
        if not history_file.exists():
            return pd.DataFrame()
        
        records = []
        with open(history_file, 'r') as f:
            for line in f:
                records.append(json.loads(line))
        
        return pd.DataFrame(records)


class KaggleDatasetManager:
    """Manage Kaggle datasets with versioning support."""
    
    def __init__(self, kaggle_api: KaggleApi):
        self.api = kaggle_api
    
    def list_datasets(self, 
                     search: Optional[str] = None,
                     tag: Optional[str] = None,
                     user: Optional[str] = None,
                     sort_by: str = "votes") -> pd.DataFrame:
        """List available datasets."""
        datasets = self.api.dataset_list(
            search=search,
            tag_ids=tag,
            user=user,
            sort_by=sort_by
        )
        
        ds_data = []
        for ds in datasets:
            ds_data.append({
                'ref': ds.ref,
                'title': ds.title,
                'size': ds.size,
                'lastUpdated': ds.lastUpdated,
                'downloadCount': ds.downloadCount,
                'voteCount': ds.voteCount,
                'usabilityRating': ds.usabilityRating
            })
        
        return pd.DataFrame(ds_data)
    
    def download_dataset(self, 
                        dataset_ref: str,
                        path: Optional[Path] = None,
                        unzip: bool = True) -> Path:
        """Download a dataset."""
        path = path or Path.cwd() / "data" / "datasets" / dataset_ref.replace('/', '_')
        path.mkdir(parents=True, exist_ok=True)
        
        self.api.dataset_download_files(
            dataset=dataset_ref,
            path=str(path),
            unzip=unzip
        )
        
        logger.info(f"Downloaded dataset to {path}")
        return path
    
    def get_dataset_metadata(self, dataset_ref: str) -> Dict[str, Any]:
        """Get dataset metadata."""
        metadata = self.api.dataset_metadata(dataset_ref)
        
        return {
            'title': metadata.title,
            'subtitle': metadata.subtitle,
            'description': metadata.description,
            'id': metadata.id,
            'id_no': metadata.id_no,
            'datasetSlug': metadata.datasetSlug,
            'ownerUser': metadata.ownerUser,
            'usabilityRating': metadata.usabilityRating,
            'licenses': [l.name for l in metadata.licenses] if metadata.licenses else []
        }


class CompetitionConfig:
    """Competition-specific configuration."""
    
    def __init__(self, competition_id: str, config_path: Optional[Path] = None):
        self.competition_id = competition_id
        self.config_path = config_path or Path.cwd() / "configs" / "competitions" / f"{competition_id}.yaml"
        self.config = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load competition configuration."""
        if self.config_path.exists():
            import yaml
            with open(self.config_path, 'r') as f:
                return yaml.safe_load(f)
        
        # Default configuration
        return {
            'competition_id': self.competition_id,
            'submission': {
                'id_column': 'PassengerId',
                'target_column': 'Survived',
                'format': 'csv'
            },
            'evaluation': {
                'metric': 'accuracy',
                'higher_is_better': True
            },
            'data': {
                'train_file': 'train.csv',
                'test_file': 'test.csv'
            }
        }
    
    def save(self):
        """Save configuration to file."""
        import yaml
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.config_path, 'w') as f:
            yaml.safe_dump(self.config, f, default_flow_style=False)
    
    @property
    def id_column(self) -> str:
        return self.config['submission']['id_column']
    
    @property
    def target_column(self) -> str:
        return self.config['submission']['target_column']
    
    @property
    def evaluation_metric(self) -> str:
        return self.config['evaluation']['metric']