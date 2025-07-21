import json
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from loguru import logger
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()


class ExperimentVisualizer:
    """Visualization tools for experiment results and analysis."""

    def __init__(self, output_dir: str = "./output"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Set style
        sns.set_style("whitegrid")
        plt.rcParams["figure.figsize"] = (12, 8)
        plt.rcParams["font.size"] = 10

    def plot_training_history(
        self, history_path: str, save_path: str | None = None
    ) -> plt.Figure:
        """Plot training history from JSON file."""
        logger.info(f"Loading training history from {history_path}")

        with open(history_path) as f:
            history = json.load(f)

        # Create figure with subplots
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle("Training History Analysis", fontsize=16)

        # Plot 1: Loss curves
        ax = axes[0, 0]
        if "train_loss" in history:
            ax.plot(history["train_loss"], label="Train Loss", linewidth=2)
        if "val_loss" in history:
            ax.plot(history["val_loss"], label="Val Loss", linewidth=2)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_title("Loss Progression")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Plot 2: Accuracy curves
        ax = axes[0, 1]
        if "train_accuracy" in history:
            ax.plot(history["train_accuracy"], label="Train Accuracy", linewidth=2)
        if "val_accuracy" in history:
            ax.plot(history["val_accuracy"], label="Val Accuracy", linewidth=2)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Accuracy")
        ax.set_title("Accuracy Progression")
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1)

        # Plot 3: F1 Score
        ax = axes[0, 2]
        if "val_f1" in history:
            ax.plot(history["val_f1"], label="Val F1", color="green", linewidth=2)
        if "val_precision" in history and "val_recall" in history:
            ax.plot(
                history["val_precision"],
                label="Val Precision",
                linestyle="--",
                alpha=0.7,
            )
            ax.plot(
                history["val_recall"], label="Val Recall", linestyle="--", alpha=0.7
            )
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Score")
        ax.set_title("F1, Precision, and Recall")
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1)

        # Plot 4: Learning Rate
        ax = axes[1, 0]
        if "learning_rate" in history:
            ax.plot(history["learning_rate"], color="orange", linewidth=2)
        ax.set_xlabel("Step")
        ax.set_ylabel("Learning Rate")
        ax.set_title("Learning Rate Schedule")
        ax.set_yscale("log")
        ax.grid(True, alpha=0.3)

        # Plot 5: AUC
        ax = axes[1, 1]
        if "val_auc" in history:
            ax.plot(history["val_auc"], color="purple", linewidth=2)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("AUC")
        ax.set_title("Area Under ROC Curve")
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1)

        # Plot 6: Loss vs Accuracy correlation
        ax = axes[1, 2]
        if "val_loss" in history and "val_accuracy" in history:
            ax.scatter(history["val_loss"], history["val_accuracy"], alpha=0.6)
            ax.set_xlabel("Validation Loss")
            ax.set_ylabel("Validation Accuracy")
            ax.set_title("Loss-Accuracy Correlation")

            # Add trend line
            z = np.polyfit(history["val_loss"], history["val_accuracy"], 1)
            p = np.poly1d(z)
            ax.plot(
                history["val_loss"],
                p(history["val_loss"]),
                "r--",
                alpha=0.8,
                label="Trend",
            )
            ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            logger.info(f"Saved training history plot to {save_path}")

        return fig

    def create_experiment_report(
        self, experiment_dir: str, mlflow_run_id: str | None = None
    ) -> str:
        """Create a comprehensive experiment report."""
        experiment_dir = Path(experiment_dir)

        console.print(Panel("Generating Experiment Report", style="bold blue"))

        report_lines = []
        report_lines.append("# Experiment Report\n")
        report_lines.append(
            f"Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        )

        # Load configuration
        config_path = experiment_dir / "config.json"
        if config_path.exists():
            with open(config_path) as f:
                config = json.load(f)

            report_lines.append("## Configuration\n")
            report_lines.append("```json\n")
            report_lines.append(json.dumps(config, indent=2))
            report_lines.append("\n```\n\n")

        # Load training history
        history_path = experiment_dir / "training_history.json"
        if history_path.exists():
            with open(history_path) as f:
                history = json.load(f)

            report_lines.append("## Training Summary\n")

            # Create summary table
            table = Table(title="Final Metrics")
            table.add_column("Metric", style="cyan")
            table.add_column("Value", style="green")

            metrics_to_show = [
                (
                    "Final Train Loss",
                    history.get("train_loss", [])[-1]
                    if history.get("train_loss")
                    else "N/A",
                ),
                (
                    "Final Val Loss",
                    history.get("val_loss", [])[-1]
                    if history.get("val_loss")
                    else "N/A",
                ),
                (
                    "Final Val Accuracy",
                    history.get("val_accuracy", [])[-1]
                    if history.get("val_accuracy")
                    else "N/A",
                ),
                (
                    "Final Val F1",
                    history.get("val_f1", [])[-1] if history.get("val_f1") else "N/A",
                ),
                (
                    "Final Val AUC",
                    history.get("val_auc", [])[-1] if history.get("val_auc") else "N/A",
                ),
                (
                    "Best Val Loss",
                    min(history.get("val_loss", [float("inf")]))
                    if history.get("val_loss")
                    else "N/A",
                ),
                (
                    "Best Val Accuracy",
                    max(history.get("val_accuracy", [0]))
                    if history.get("val_accuracy")
                    else "N/A",
                ),
            ]

            for metric, value in metrics_to_show:
                if isinstance(value, float):
                    table.add_row(metric, f"{value:.4f}")
                else:
                    table.add_row(metric, str(value))

            console.print(table)

            # Add to report
            report_lines.append("### Final Metrics\n")
            for metric, value in metrics_to_show:
                if isinstance(value, float):
                    report_lines.append(f"- **{metric}**: {value:.4f}\n")
                else:
                    report_lines.append(f"- **{metric}**: {value}\n")
            report_lines.append("\n")

        # MLflow information
        if mlflow_run_id:
            report_lines.append("## MLflow Run Information\n")
            report_lines.append(f"- **Run ID**: {mlflow_run_id}\n")
            report_lines.append(
                f"- **MLflow UI**: [View in MLflow](http://localhost:5000/#/experiments/0/runs/{mlflow_run_id})\n\n"
            )

        # Add plots
        report_lines.append("## Visualizations\n")

        # Generate and save plots
        if history_path.exists():
            plot_path = experiment_dir / "training_history_plot.png"
            self.plot_training_history(history_path, plot_path)
            report_lines.append(f"![Training History]({plot_path.name})\n\n")

        # Save report
        report_path = experiment_dir / "experiment_report.md"
        with open(report_path, "w") as f:
            f.writelines(report_lines)

        logger.success(f"Generated experiment report: {report_path}")
        return str(report_path)

    def compare_experiments(
        self,
        experiment_dirs: list[str],
        metrics_to_compare: list[str] = ["val_accuracy", "val_loss", "val_f1"],
    ) -> pd.DataFrame:
        """Compare multiple experiments."""
        comparison_data = []

        for exp_dir in experiment_dirs:
            exp_path = Path(exp_dir)
            exp_data = {"experiment": exp_path.name}

            # Load config
            config_path = exp_path / "config.json"
            if config_path.exists():
                with open(config_path) as f:
                    config = json.load(f)
                exp_data["learning_rate"] = config.get("training", {}).get(
                    "learning_rate", "N/A"
                )
                exp_data["batch_size"] = config.get("training", {}).get(
                    "batch_size", "N/A"
                )

            # Load history
            history_path = exp_path / "training_history.json"
            if history_path.exists():
                with open(history_path) as f:
                    history = json.load(f)

                for metric in metrics_to_compare:
                    if metric in history and history[metric]:
                        exp_data[f"best_{metric}"] = (
                            max(history[metric])
                            if "accuracy" in metric
                            else min(history[metric])
                        )
                        exp_data[f"final_{metric}"] = history[metric][-1]

            comparison_data.append(exp_data)

        df = pd.DataFrame(comparison_data)

        # Display comparison table
        table = Table(title="Experiment Comparison")

        # Add columns
        for col in df.columns:
            table.add_column(col, style="cyan" if col == "experiment" else "white")

        # Add rows
        for _, row in df.iterrows():
            table.add_row(*[str(val) for val in row.values])

        console.print(table)

        # Create comparison plot
        fig, axes = plt.subplots(
            1, len(metrics_to_compare), figsize=(6 * len(metrics_to_compare), 6)
        )
        if len(metrics_to_compare) == 1:
            axes = [axes]

        for idx, metric in enumerate(metrics_to_compare):
            ax = axes[idx]
            best_col = f"best_{metric}"
            if best_col in df.columns:
                df_sorted = df.sort_values(best_col, ascending="loss" in metric)
                bars = ax.bar(range(len(df_sorted)), df_sorted[best_col])
                ax.set_xticks(range(len(df_sorted)))
                ax.set_xticklabels(df_sorted["experiment"], rotation=45, ha="right")
                ax.set_ylabel(metric)
                ax.set_title(f"Best {metric} Comparison")

                # Color best performer
                best_idx = 0 if "loss" in metric else len(df_sorted) - 1
                bars[best_idx].set_color("green")

        plt.tight_layout()
        plt.savefig(
            self.output_dir / "experiment_comparison.png", dpi=300, bbox_inches="tight"
        )

        return df

    def visualize_predictions(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_proba: np.ndarray | None = None,
        save_dir: str | None = None,
    ):
        """Visualize prediction results."""
        save_dir = Path(save_dir) if save_dir else self.output_dir
        save_dir.mkdir(parents=True, exist_ok=True)

        from sklearn.metrics import classification_report, confusion_matrix

        # Create figure
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle("Prediction Analysis", fontsize=16)

        # Confusion Matrix
        ax = axes[0, 0]
        cm = confusion_matrix(y_true, y_pred)
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            ax=ax,
            xticklabels=["Not Survived", "Survived"],
            yticklabels=["Not Survived", "Survived"],
        )
        ax.set_title("Confusion Matrix")
        ax.set_ylabel("True Label")
        ax.set_xlabel("Predicted Label")

        # Prediction Distribution
        ax = axes[0, 1]
        pred_counts = pd.Series(y_pred).value_counts()
        ax.bar(
            ["Not Survived", "Survived"], [pred_counts.get(0, 0), pred_counts.get(1, 0)]
        )
        ax.set_title("Prediction Distribution")
        ax.set_ylabel("Count")

        # Probability Distribution (if available)
        if y_proba is not None:
            ax = axes[1, 0]
            ax.hist(
                y_proba[y_true == 0],
                bins=20,
                alpha=0.5,
                label="Not Survived",
                color="red",
            )
            ax.hist(
                y_proba[y_true == 1],
                bins=20,
                alpha=0.5,
                label="Survived",
                color="green",
            )
            ax.set_xlabel("Predicted Probability of Survival")
            ax.set_ylabel("Count")
            ax.set_title("Probability Distribution by True Label")
            ax.legend()

            # Calibration plot
            ax = axes[1, 1]
            from sklearn.calibration import calibration_curve

            fraction_of_positives, mean_predicted_value = calibration_curve(
                y_true, y_proba, n_bins=10
            )
            ax.plot(mean_predicted_value, fraction_of_positives, "s-", label="Model")
            ax.plot([0, 1], [0, 1], "k--", label="Perfectly calibrated")
            ax.set_xlabel("Mean Predicted Probability")
            ax.set_ylabel("Fraction of Positives")
            ax.set_title("Calibration Plot")
            ax.legend()

        plt.tight_layout()
        save_path = save_dir / "prediction_analysis.png"
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        logger.info(f"Saved prediction analysis to {save_path}")

        # Print classification report
        report = classification_report(
            y_true, y_pred, target_names=["Not Survived", "Survived"]
        )
        console.print("\n[bold]Classification Report:[/bold]")
        console.print(report)

        # Save report
        report_path = save_dir / "classification_report.txt"
        with open(report_path, "w") as f:
            f.write(report)


def create_mlflow_dashboard_link(tracking_uri: str = "./mlruns") -> str:
    """Create a link to launch MLflow dashboard."""
    import subprocess
    import webbrowser

    console.print(Panel("Launching MLflow Dashboard", style="bold green"))

    # Start MLflow server
    cmd = f"mlflow ui --backend-store-uri {tracking_uri} --port 5000"
    console.print(f"Starting MLflow server: {cmd}")

    try:
        subprocess.Popen(cmd.split(), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        console.print("[green]MLflow server started successfully![/green]")
        console.print(
            "[blue]Open your browser and navigate to: http://localhost:5000[/blue]"
        )

        # Optionally open browser automatically
        webbrowser.open("http://localhost:5000")

        return "http://localhost:5000"
    except Exception as e:
        logger.error(f"Failed to start MLflow server: {e}")
        return None
