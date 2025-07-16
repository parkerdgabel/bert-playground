#!/usr/bin/env python3
import argparse
from pathlib import Path
import os
from dotenv import load_dotenv
from loguru import logger
from rich.console import Console

from training.trainer_v2 import create_trainer_v2
from kaggle.titanic.submission import create_submission
from utils.logging_config import setup_logging
from utils.visualization import ExperimentVisualizer, create_mlflow_dashboard_link

# Load environment variables
load_dotenv()

console = Console()


def main():
    parser = argparse.ArgumentParser(
        description="Train ModernBERT on Titanic dataset with MLflow tracking and extensive logging"
    )

    # Data arguments
    parser.add_argument(
        "--train_path",
        type=str,
        default="kaggle/titanic/data/train.csv",
        help="Path to training data",
    )
    parser.add_argument(
        "--val_path", type=str, default=None, help="Path to validation data (optional)"
    )
    parser.add_argument(
        "--test_path",
        type=str,
        default="kaggle/titanic/data/test.csv",
        help="Path to test data for submission",
    )

    # Model arguments
    parser.add_argument(
        "--model_name",
        type=str,
        default="answerdotai/ModernBERT-base",
        help="Pretrained model name from HuggingFace",
    )

    # Training arguments
    parser.add_argument(
        "--batch_size", type=int, default=16, help="Training batch size"
    )
    parser.add_argument(
        "--learning_rate", type=float, default=2e-5, help="Learning rate"
    )
    parser.add_argument(
        "--num_epochs", type=int, default=3, help="Number of training epochs"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./output",
        help="Directory to save checkpoints",
    )

    # Experiment tracking arguments
    parser.add_argument(
        "--experiment_name",
        type=str,
        default="titanic_modernbert",
        help="MLflow experiment name",
    )
    parser.add_argument(
        "--run_name", type=str, default=None, help="MLflow run name (optional)"
    )
    parser.add_argument(
        "--log_level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level",
    )
    parser.add_argument(
        "--disable_mlflow", action="store_true", help="Disable MLflow tracking"
    )

    # Action arguments
    parser.add_argument("--do_train", action="store_true", help="Run training")
    parser.add_argument(
        "--do_predict", action="store_true", help="Generate predictions"
    )
    parser.add_argument(
        "--do_visualize",
        action="store_true",
        help="Generate visualizations and reports",
    )
    parser.add_argument(
        "--launch_mlflow", action="store_true", help="Launch MLflow UI after training"
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default=None,
        help="Path to checkpoint for prediction",
    )

    args = parser.parse_args()

    # Setup logging
    setup_logging(
        experiment_name=args.run_name or args.experiment_name,
        log_level=args.log_level,
        log_dir=os.path.join(args.output_dir, "logs"),
    )

    logger.info("=" * 60)
    logger.info("ModernBERT Titanic Training with MLflow")
    logger.info("=" * 60)

    if args.do_train:
        console.print("\n[bold green]Starting training...[/bold green]")
        console.print(f"Model: {args.model_name}")
        console.print(f"Batch size: {args.batch_size}")
        console.print(f"Learning rate: {args.learning_rate}")
        console.print(f"Epochs: {args.num_epochs}")
        console.print(
            f"MLflow tracking: {'Enabled' if not args.disable_mlflow else 'Disabled'}"
        )

        # Create trainer
        trainer = create_trainer_v2(
            train_path=args.train_path,
            val_path=args.val_path,
            model_name=args.model_name,
            learning_rate=args.learning_rate,
            batch_size=args.batch_size,
            num_epochs=args.num_epochs,
            output_dir=args.output_dir,
            experiment_name=args.experiment_name,
            enable_mlflow=not args.disable_mlflow,
        )

        # Train model
        try:
            trainer.train(args.num_epochs)

            # Set checkpoint path for prediction
            if args.do_predict and not args.checkpoint_path:
                args.checkpoint_path = str(
                    Path(args.output_dir) / "best_model_accuracy"
                )

        except Exception as e:
            logger.error(f"Training failed: {str(e)}")
            logger.exception("Detailed error:")
            raise

    if args.do_predict:
        if not args.checkpoint_path:
            raise ValueError("Must provide --checkpoint_path for prediction")

        console.print(
            f"\n[bold blue]Generating predictions from {args.checkpoint_path}...[/bold blue]"
        )

        try:
            # Create submission
            create_submission(
                test_path=args.test_path,
                checkpoint_path=args.checkpoint_path,
                output_path="submission.csv",
            )

            logger.success("Predictions generated successfully!")

        except Exception as e:
            logger.error(f"Prediction failed: {str(e)}")
            logger.exception("Detailed error:")
            raise

    if args.do_visualize:
        console.print(
            "\n[bold magenta]Generating visualizations and reports...[/bold magenta]"
        )

        visualizer = ExperimentVisualizer(output_dir=args.output_dir)

        try:
            # Generate experiment report
            visualizer.create_experiment_report(
                experiment_dir=args.output_dir,
                mlflow_run_id=None,  # Could be retrieved from trainer
            )

            # Plot training history
            history_path = Path(args.output_dir) / "training_history.json"
            if history_path.exists():
                visualizer.plot_training_history(
                    history_path=str(history_path),
                    save_path=str(Path(args.output_dir) / "training_curves.png"),
                )

            logger.success("Visualizations generated successfully!")

        except Exception as e:
            logger.error(f"Visualization failed: {str(e)}")
            logger.exception("Detailed error:")

    if args.launch_mlflow and not args.disable_mlflow:
        console.print("\n[bold yellow]Launching MLflow dashboard...[/bold yellow]")
        mlflow_url = create_mlflow_dashboard_link(
            tracking_uri=str(Path(args.output_dir) / "mlruns")
        )
        if mlflow_url:
            console.print(f"[green]MLflow dashboard available at: {mlflow_url}[/green]")
            console.print("[yellow]Press Ctrl+C to stop the MLflow server[/yellow]")

            # Keep the script running
            try:
                import time

                while True:
                    time.sleep(1)
            except KeyboardInterrupt:
                console.print("\n[red]Stopping MLflow server...[/red]")

    logger.info("=" * 60)
    logger.info("All tasks completed!")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
