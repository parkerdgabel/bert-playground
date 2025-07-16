#!/usr/bin/env python3
"""Evaluate a single trained model - wrapper for unified evaluation module."""

import argparse
from pathlib import Path
from utils.evaluation import evaluate_single_model
from rich.console import Console

console = Console()

def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained model")
    parser.add_argument("model_path", type=str, help="Path to model checkpoint")
    parser.add_argument("--submit", action="store_true", help="Submit to Kaggle")
    parser.add_argument("--mlflow", action="store_true", help="Enable MLflow tracking")
    parser.add_argument("--output-dir", type=str, default="evaluation_results",
                       help="Output directory for results")
    
    args = parser.parse_args()
    
    # Validate model path
    if not Path(args.model_path).exists():
        console.print(f"❌ Model path not found: {args.model_path}", style="bold red")
        return 1
    
    # Run evaluation
    console.print(f"🔍 Evaluating model: {args.model_path}", style="bold blue")
    
    try:
        results = evaluate_single_model(
            args.model_path,
            submit=args.submit,
            enable_mlflow=args.mlflow
        )
        
        console.print("\n✅ Evaluation complete!", style="bold green")
        if 'submission_path' in results:
            console.print(f"📄 Submission saved to: {results['submission_path']}")
        
        return 0
        
    except Exception as e:
        console.print(f"\n❌ Evaluation failed: {e}", style="bold red")
        return 1

if __name__ == "__main__":
    exit(main())