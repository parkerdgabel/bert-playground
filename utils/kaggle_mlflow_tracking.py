"""
MLflow tracking enhancements for Kaggle competition experiments.
Provides specialized tracking for competition metrics, submissions, and leaderboard positions.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import mlflow
import pandas as pd
from loguru import logger


class KaggleMLflowTracker:
    """Enhanced MLflow tracking for Kaggle competitions."""

    def __init__(self, competition_id: str, experiment_name: str | None = None):
        """Initialize Kaggle MLflow tracker."""
        self.competition_id = competition_id
        self.experiment_name = experiment_name or f"kaggle_{competition_id}"

        # Setup MLflow experiment
        mlflow.set_experiment(self.experiment_name)

    def start_competition_run(
        self,
        run_name: str,
        model_type: str = "MLX-BERT",
        tags: dict[str, str] | None = None,
    ) -> str:
        """Start a new MLflow run for competition training."""
        tags = tags or {}
        tags.update(
            {
                "competition": self.competition_id,
                "model_type": model_type,
                "framework": "MLX",
                "purpose": "kaggle_competition",
            }
        )

        run = mlflow.start_run(run_name=run_name, tags=tags)

        # Log competition metadata
        mlflow.log_param("competition_id", self.competition_id)
        mlflow.log_param("model_type", model_type)
        mlflow.log_param("start_time", datetime.now().isoformat())

        return run.info.run_id

    def log_submission(
        self,
        submission_file: Path,
        message: str,
        public_score: float | None = None,
        private_score: float | None = None,
        leaderboard_rank: int | None = None,
    ):
        """Log a Kaggle submission with MLflow."""
        if not mlflow.active_run():
            logger.warning("No active MLflow run, skipping submission logging")
            return

        # Log submission as artifact
        mlflow.log_artifact(str(submission_file), artifact_path="submissions")

        # Log submission metadata
        submission_info = {
            "timestamp": datetime.now().isoformat(),
            "file": submission_file.name,
            "message": message,
            "public_score": public_score,
            "private_score": private_score,
            "leaderboard_rank": leaderboard_rank,
        }

        # Save submission info as JSON artifact
        submission_json = (
            Path("/tmp") / f"submission_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )
        with open(submission_json, "w") as f:
            json.dump(submission_info, f, indent=2)
        mlflow.log_artifact(str(submission_json), artifact_path="submission_history")
        submission_json.unlink()

        # Log metrics
        if public_score is not None:
            mlflow.log_metric("kaggle_public_score", public_score)
            mlflow.log_metric(
                "best_kaggle_public_score", public_score
            )  # MLflow will keep the best

        if private_score is not None:
            mlflow.log_metric("kaggle_private_score", private_score)

        if leaderboard_rank is not None:
            mlflow.log_metric("kaggle_leaderboard_rank", leaderboard_rank)
            mlflow.log_metric(
                "best_kaggle_rank", -leaderboard_rank
            )  # Negative for "higher is better"

    def log_leaderboard_snapshot(self, leaderboard_df: pd.DataFrame, top_n: int = 100):
        """Log current competition leaderboard snapshot."""
        if not mlflow.active_run():
            return

        # Save leaderboard snapshot
        snapshot_path = (
            Path("/tmp") / f"leaderboard_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        )
        leaderboard_df.head(top_n).to_csv(snapshot_path, index=False)

        mlflow.log_artifact(str(snapshot_path), artifact_path="leaderboard_snapshots")
        snapshot_path.unlink()

        # Log key statistics
        if not leaderboard_df.empty:
            mlflow.log_metric("leaderboard_total_teams", len(leaderboard_df))
            if "score" in leaderboard_df.columns:
                mlflow.log_metric(
                    "leaderboard_top_score", leaderboard_df["score"].max()
                )
                mlflow.log_metric(
                    "leaderboard_median_score", leaderboard_df["score"].median()
                )

    def log_competition_config(self, config: dict[str, Any]):
        """Log competition-specific configuration."""
        if not mlflow.active_run():
            return

        # Log as parameters (flatten nested config)
        def flatten_dict(d, parent_key="", sep="_"):
            items = []
            for k, v in d.items():
                new_key = f"{parent_key}{sep}{k}" if parent_key else k
                if isinstance(v, dict):
                    items.extend(flatten_dict(v, new_key, sep=sep).items())
                else:
                    items.append((new_key, v))
            return dict(items)

        flat_config = flatten_dict(config)
        for key, value in flat_config.items():
            if key.startswith("competition_"):
                mlflow.log_param(key, value)

    def compare_with_baseline(
        self, current_score: float, baseline_score: float, metric_name: str = "accuracy"
    ):
        """Compare current model with baseline and log improvement."""
        if not mlflow.active_run():
            return

        improvement = current_score - baseline_score
        pct_improvement = (
            (improvement / baseline_score) * 100 if baseline_score != 0 else 0
        )

        mlflow.log_metric(f"{metric_name}_vs_baseline", improvement)
        mlflow.log_metric(f"{metric_name}_pct_improvement", pct_improvement)

        # Log comparison summary
        comparison = {
            "metric": metric_name,
            "baseline_score": baseline_score,
            "current_score": current_score,
            "improvement": improvement,
            "pct_improvement": pct_improvement,
            "timestamp": datetime.now().isoformat(),
        }

        comparison_json = Path("/tmp") / "baseline_comparison.json"
        with open(comparison_json, "w") as f:
            json.dump(comparison, f, indent=2)
        mlflow.log_artifact(str(comparison_json), artifact_path="comparisons")
        comparison_json.unlink()

    def create_submission_dashboard(self, output_path: Path | None = None):
        """Create a dashboard summarizing all submissions in the experiment."""
        experiment = mlflow.get_experiment_by_name(self.experiment_name)
        if not experiment:
            logger.warning(f"Experiment {self.experiment_name} not found")
            return None

        # Get all runs in the experiment
        runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])

        if runs.empty:
            logger.warning("No runs found in experiment")
            return None

        # Extract submission data
        submission_data = []
        for _, run in runs.iterrows():
            if "metrics.kaggle_public_score" in run:
                submission_data.append(
                    {
                        "run_id": run["run_id"],
                        "run_name": run.get("tags.mlflow.runName", "Unnamed"),
                        "start_time": run["start_time"],
                        "public_score": run.get("metrics.kaggle_public_score"),
                        "private_score": run.get("metrics.kaggle_private_score"),
                        "leaderboard_rank": run.get("metrics.kaggle_leaderboard_rank"),
                        "val_accuracy": run.get("metrics.val_accuracy"),
                        "model_type": run.get("params.model_type", "Unknown"),
                    }
                )

        if not submission_data:
            logger.warning("No submissions found in experiment")
            return None

        # Create dashboard DataFrame
        dashboard_df = pd.DataFrame(submission_data)
        dashboard_df = dashboard_df.sort_values("public_score", ascending=False)

        # Save dashboard
        if output_path:
            output_path.parent.mkdir(parents=True, exist_ok=True)

            # Save as CSV
            dashboard_df.to_csv(output_path.with_suffix(".csv"), index=False)

            # Save as JSON with summary statistics
            summary = {
                "competition": self.competition_id,
                "total_submissions": len(dashboard_df),
                "best_public_score": dashboard_df["public_score"].max(),
                "best_private_score": dashboard_df["private_score"].max()
                if "private_score" in dashboard_df
                else None,
                "best_rank": dashboard_df["leaderboard_rank"].min()
                if "leaderboard_rank" in dashboard_df
                else None,
                "submissions": dashboard_df.to_dict("records"),
            }

            with open(output_path.with_suffix(".json"), "w") as f:
                json.dump(summary, f, indent=2)

            logger.info(f"Submission dashboard saved to {output_path}")

        return dashboard_df

    def log_ensemble_submission(
        self,
        model_weights: dict[str, float],
        individual_scores: dict[str, float],
        ensemble_score: float,
    ):
        """Log ensemble model submission details."""
        if not mlflow.active_run():
            return

        # Log ensemble configuration
        for model, weight in model_weights.items():
            mlflow.log_param(f"ensemble_{model}_weight", weight)

        # Log individual model scores
        for model, score in individual_scores.items():
            mlflow.log_metric(f"individual_{model}_score", score)

        # Log ensemble score
        mlflow.log_metric("ensemble_score", ensemble_score)

        # Calculate and log ensemble improvement
        best_individual = max(individual_scores.values())
        ensemble_improvement = ensemble_score - best_individual
        mlflow.log_metric("ensemble_improvement", ensemble_improvement)

        # Save ensemble details
        ensemble_info = {
            "timestamp": datetime.now().isoformat(),
            "model_weights": model_weights,
            "individual_scores": individual_scores,
            "ensemble_score": ensemble_score,
            "improvement_over_best": ensemble_improvement,
        }

        ensemble_json = Path("/tmp") / "ensemble_info.json"
        with open(ensemble_json, "w") as f:
            json.dump(ensemble_info, f, indent=2)
        mlflow.log_artifact(str(ensemble_json), artifact_path="ensemble")
        ensemble_json.unlink()


class CompetitionExperimentManager:
    """Manage multiple experiments for a Kaggle competition."""

    def __init__(self, competition_id: str):
        self.competition_id = competition_id
        self.base_experiment_name = f"kaggle_{competition_id}"

    def create_experiment_variants(self, variants: list[str]):
        """Create experiment variants for different approaches."""
        experiments = {}

        for variant in variants:
            exp_name = f"{self.base_experiment_name}_{variant}"
            mlflow.set_experiment(exp_name)
            experiments[variant] = exp_name

            # Set experiment tags
            mlflow.set_experiment_tag("competition", self.competition_id)
            mlflow.set_experiment_tag("variant", variant)
            mlflow.set_experiment_tag("created_at", datetime.now().isoformat())

        return experiments

    def compare_experiment_variants(self) -> pd.DataFrame:
        """Compare results across experiment variants."""
        all_experiments = mlflow.search_experiments(
            filter_string=f"tags.competition = '{self.competition_id}'"
        )

        comparison_data = []

        for exp in all_experiments:
            runs = mlflow.search_runs(experiment_ids=[exp.experiment_id])

            if not runs.empty:
                best_run = runs.loc[runs["metrics.kaggle_public_score"].idxmax()]

                comparison_data.append(
                    {
                        "variant": exp.tags.get("variant", "default"),
                        "experiment_name": exp.name,
                        "best_public_score": best_run.get(
                            "metrics.kaggle_public_score"
                        ),
                        "best_private_score": best_run.get(
                            "metrics.kaggle_private_score"
                        ),
                        "best_val_accuracy": best_run.get("metrics.val_accuracy"),
                        "total_runs": len(runs),
                        "best_run_id": best_run["run_id"],
                    }
                )

        return pd.DataFrame(comparison_data).sort_values(
            "best_public_score", ascending=False
        )
