#!/usr/bin/env python3
"""Evaluate and compare all models, then submit best to Kaggle."""

import argparse
from utils.evaluation import UnifiedModelEvaluator, compare_all_models
from rich.console import Console
from rich.table import Table
from rich import box

console = Console()


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate all models and submit best to Kaggle"
    )
    parser.add_argument(
        "--submit-best", action="store_true", help="Submit best model to Kaggle"
    )
    parser.add_argument("--mlflow", action="store_true", help="Enable MLflow tracking")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="evaluation_results",
        help="Output directory for results",
    )

    args = parser.parse_args()

    console.print("🔍 Finding and evaluating all trained models...", style="bold blue")

    try:
        # Compare all models
        comparison_df = compare_all_models(args.output_dir)

        if comparison_df.empty:
            console.print("❌ No models found to evaluate", style="bold red")
            return 1

        # Display results table
        table = Table(title="Model Comparison Results", box=box.ROUNDED)
        table.add_column("Model", style="cyan")
        table.add_column("Type", style="yellow")
        table.add_column("Accuracy", style="green")
        table.add_column("F1 Score", style="blue")
        table.add_column("AUC", style="magenta")

        for _, row in comparison_df.iterrows():
            table.add_row(
                row["model_path"].split("/")[-2]
                if "/" in row["model_path"]
                else row["model_path"],
                row["model_type"],
                f"{row['accuracy']:.4f}",
                f"{row['f1']:.4f}",
                f"{row.get('auc', 0):.4f}",
            )

        console.print(table)

        # Submit best model if requested
        if args.submit_best and len(comparison_df) > 0:
            best_model = comparison_df.iloc[0]
            console.print(
                f"\n🏆 Best model: {best_model['model_path']}", style="bold green"
            )
            console.print(f"   Accuracy: {best_model['accuracy']:.4f}")

            evaluator = UnifiedModelEvaluator(enable_mlflow=args.mlflow)
            evaluator.evaluate_and_submit(
                best_model["model_path"], submit_to_kaggle=True
            )

        console.print("\n✅ Evaluation complete!", style="bold green")
        return 0

    except Exception as e:
        console.print(f"\n❌ Evaluation failed: {e}", style="bold red")
        return 1


if __name__ == "__main__":
    exit(main())
