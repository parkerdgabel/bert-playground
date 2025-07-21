"""
Kaggle-specific callbacks for competition tracking and submission.
"""

import subprocess
from datetime import datetime
from pathlib import Path

import pandas as pd
from loguru import logger

from ..callbacks.base import Callback
from ..core.protocols import Trainer, TrainingResult, TrainingState


class KaggleSubmissionCallback(Callback):
    """
    Callback for automatic Kaggle submission.

    Requires Kaggle API to be configured with credentials.
    """

    def __init__(
        self,
        competition_name: str,
        submission_message: str = "Automated submission",
        submit_best_only: bool = True,
        submit_on_improvement: bool = False,
    ):
        super().__init__()
        self.competition_name = competition_name
        self.submission_message = submission_message
        self.submit_best_only = submit_best_only
        self.submit_on_improvement = submit_on_improvement

        self.best_score = None
        self.submission_count = 0

        # Check if Kaggle API is available
        self._check_kaggle_api()

    def _check_kaggle_api(self):
        """Check if Kaggle API is configured."""
        try:
            result = subprocess.run(
                ["kaggle", "config", "view"],
                capture_output=True,
                text=True,
            )
            if result.returncode != 0:
                logger.warning(
                    "Kaggle API not configured. Please run 'kaggle config set'"
                )
                self.enabled = False
            else:
                self.enabled = True
                logger.info("Kaggle API configured and ready")
        except FileNotFoundError:
            logger.warning("Kaggle CLI not found. Install with 'pip install kaggle'")
            self.enabled = False

    def on_evaluate_end(
        self, trainer: Trainer, state: TrainingState, metrics: dict[str, float]
    ) -> None:
        """Check if we should submit based on metrics."""
        if not self.enabled or not self.submit_on_improvement:
            return

        # Get competition metric
        kaggle_config = getattr(trainer.config, "kaggle", None)
        if not kaggle_config:
            return

        metric_name = f"eval_{kaggle_config.competition_metric}"
        if metric_name not in metrics:
            return

        current_score = metrics[metric_name]

        # Check if improved
        if self.best_score is None:
            self.best_score = current_score
            should_submit = True
        else:
            if kaggle_config.maximize_metric:
                should_submit = current_score > self.best_score
            else:
                should_submit = current_score < self.best_score

            if should_submit:
                self.best_score = current_score

        if should_submit:
            logger.info(f"New best score: {current_score:.4f}, preparing submission")
            self._prepare_and_submit(trainer, state)

    def on_train_end(
        self, trainer: Trainer, state: TrainingState, result: TrainingResult
    ) -> None:
        """Submit final model if configured."""
        if not self.enabled:
            return

        # Submit if not submit_best_only or if we haven't submitted yet
        if not self.submit_best_only or self.submission_count == 0:
            self._prepare_and_submit(trainer, state, is_final=True)

    def _prepare_and_submit(
        self, trainer: Trainer, state: TrainingState, is_final: bool = False
    ):
        """Prepare and submit to Kaggle."""
        try:
            # Get Kaggle trainer
            if not hasattr(trainer, "create_submission"):
                logger.warning("Trainer does not support submission creation")
                return

            # Create submission
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            submission_name = f"submission_{timestamp}.csv"

            submission_path = trainer.create_submission(submission_name=submission_name)

            # Prepare submission message
            message = self.submission_message
            if is_final:
                message += " (final)"
            else:
                message += f" (step {state.global_step})"

            # Submit to Kaggle
            self._submit_to_kaggle(submission_path, message)

            self.submission_count += 1

        except Exception as e:
            logger.error(f"Failed to submit to Kaggle: {e}")

    def _submit_to_kaggle(self, submission_path: Path, message: str):
        """Submit file to Kaggle competition."""
        cmd = [
            "kaggle",
            "competitions",
            "submit",
            "-c",
            self.competition_name,
            "-f",
            str(submission_path),
            "-m",
            message,
        ]

        logger.info(f"Submitting to Kaggle: {submission_path}")

        try:
            result = subprocess.run(cmd, capture_output=True, text=True)

            if result.returncode == 0:
                logger.info(f"Successfully submitted to {self.competition_name}")
                logger.info(result.stdout)
            else:
                logger.error(f"Submission failed: {result.stderr}")

        except Exception as e:
            logger.error(f"Failed to run Kaggle CLI: {e}")


class LeaderboardTracker(Callback):
    """
    Track position on Kaggle leaderboard.

    Periodically checks leaderboard position and logs changes.
    """

    def __init__(
        self,
        competition_name: str,
        check_frequency: int = 10,  # Check every N epochs
        track_top_n: int = 100,
    ):
        super().__init__()
        self.competition_name = competition_name
        self.check_frequency = check_frequency
        self.track_top_n = track_top_n

        self.last_position = None
        self.last_score = None
        self.submission_history = []

    def on_epoch_end(self, trainer: Trainer, state: TrainingState) -> None:
        """Check leaderboard if it's time."""
        if state.epoch % self.check_frequency == 0:
            self._check_leaderboard()

    def _check_leaderboard(self):
        """Check current position on leaderboard."""
        try:
            # Get leaderboard
            cmd = [
                "kaggle",
                "competitions",
                "leaderboard",
                self.competition_name,
                "--show",
                "true",
                "--csv",
            ]

            result = subprocess.run(cmd, capture_output=True, text=True)

            if result.returncode == 0:
                # Parse CSV output
                import io

                df = pd.read_csv(io.StringIO(result.stdout))

                # Find our submissions
                # This assumes we can identify our team name
                # For now, just log the top entries
                logger.info(f"Leaderboard top {min(5, len(df))}:")
                for i, row in df.head(5).iterrows():
                    logger.info(f"  {i + 1}. {row['teamName']} - Score: {row['score']}")

            else:
                logger.error(f"Failed to get leaderboard: {result.stderr}")

        except Exception as e:
            logger.error(f"Failed to check leaderboard: {e}")


class CompetitionMetrics(Callback):
    """
    Track competition-specific metrics.

    Calculates and logs metrics specific to the competition type.
    """

    def __init__(
        self,
        metric_name: str,
        maximize: bool = True,
        log_improvement: bool = True,
    ):
        super().__init__()
        self.metric_name = metric_name
        self.maximize = maximize
        self.log_improvement = log_improvement

        self.best_score = None
        self.scores_history = []

    def on_evaluate_end(
        self, trainer: Trainer, state: TrainingState, metrics: dict[str, float]
    ) -> None:
        """Track competition metric."""
        # Look for the metric
        metric_key = f"eval_{self.metric_name}"
        if metric_key not in metrics:
            # Try without eval_ prefix
            metric_key = self.metric_name
            if metric_key not in metrics:
                return

        current_score = metrics[metric_key]
        self.scores_history.append((state.epoch, current_score))

        # Check if improved
        if self.best_score is None:
            self.best_score = current_score
            is_improvement = True
        else:
            if self.maximize:
                is_improvement = current_score > self.best_score
            else:
                is_improvement = current_score < self.best_score

            if is_improvement:
                self.best_score = current_score

        # Log status
        if self.log_improvement and is_improvement:
            improvement = (
                abs(current_score - self.scores_history[-2][1])
                if len(self.scores_history) > 1
                else 0
            )
            logger.info(
                f"Competition metric improved! {self.metric_name}: {current_score:.4f} "
                f"(+{improvement:.4f})"
                if self.maximize
                else f"(-{improvement:.4f})"
            )

        # Add to state for other callbacks
        state.metrics[f"competition_{self.metric_name}"] = current_score
        state.metrics[f"best_competition_{self.metric_name}"] = self.best_score

    def on_train_end(
        self, trainer: Trainer, state: TrainingState, result: TrainingResult
    ) -> None:
        """Log final competition metrics."""
        logger.info("\nCompetition Metric Summary:")
        logger.info(f"Metric: {self.metric_name}")
        logger.info(f"Best Score: {self.best_score:.4f}")
        logger.info(
            f"Final Score: {self.scores_history[-1][1]:.4f}"
            if self.scores_history
            else "N/A"
        )

        # Save history
        if self.scores_history:
            history_df = pd.DataFrame(
                self.scores_history, columns=["epoch", self.metric_name]
            )
            history_path = (
                trainer.config.environment.output_dir
                / f"competition_metric_{self.metric_name}.csv"
            )
            history_df.to_csv(history_path, index=False)
            logger.info(f"Saved competition metric history to {history_path}")
