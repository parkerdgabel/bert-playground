#!/usr/bin/env python3
"""Generate submission file from a trained model."""

import argparse
from pathlib import Path
from utils.evaluation import generate_submission_only
from rich.console import Console

console = Console()

def main():
    parser = argparse.ArgumentParser(description="Generate Kaggle submission from trained model")
    parser.add_argument("model_path", type=str, help="Path to model checkpoint")
    parser.add_argument("--test-path", type=str, default="data/titanic/test.csv",
                       help="Path to test data CSV")
    parser.add_argument("--output", type=str, help="Output submission filename")
    parser.add_argument("--submit", action="store_true", help="Submit to Kaggle after generation")
    
    args = parser.parse_args()
    
    # Validate paths
    if not Path(args.model_path).exists():
        console.print(f"❌ Model path not found: {args.model_path}", style="bold red")
        return 1
    
    if not Path(args.test_path).exists():
        console.print(f"❌ Test data not found: {args.test_path}", style="bold red")
        return 1
    
    console.print(f"🚀 Generating submission for: {args.model_path}", style="bold blue")
    
    try:
        # Generate submission
        submission_path = generate_submission_only(
            args.model_path,
            args.test_path,
            args.output
        )
        
        console.print(f"\n✅ Submission saved to: {submission_path}", style="bold green")
        
        # Submit to Kaggle if requested
        if args.submit:
            from utils.evaluation import UnifiedModelEvaluator
            evaluator = UnifiedModelEvaluator()
            evaluator.submit_to_kaggle(submission_path)
        
        return 0
        
    except Exception as e:
        console.print(f"\n❌ Failed to generate submission: {e}", style="bold red")
        return 1

if __name__ == "__main__":
    exit(main())