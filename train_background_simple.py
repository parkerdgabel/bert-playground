#!/usr/bin/env python3
"""Simple script to run training with output redirection."""

import subprocess
import sys
from datetime import datetime
from pathlib import Path


def main():
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(f"./output/background_{timestamp}")
    output_dir.mkdir(parents=True, exist_ok=True)

    log_file = output_dir / "training.log"

    print("Starting background training...")
    print(f"Output directory: {output_dir}")
    print(f"Log file: {log_file}")

    # Training command
    cmd = [
        "uv",
        "run",
        "python",
        "mlx_bert_cli.py",
        "train",
        "--train",
        "data/titanic/train.csv",
        "--val",
        "data/titanic/val.csv",
        "--output",
        str(output_dir),
        "--batch-size",
        "32",
        "--epochs",
        "5",
        "--lr",
        "2e-5",
        "--experiment",
        "background_training",
    ]

    # Run in background
    with open(log_file, "w") as f:
        process = subprocess.Popen(
            cmd, stdout=f, stderr=subprocess.STDOUT, start_new_session=True
        )

    print(f"Training started with PID: {process.pid}")

    # Save PID
    with open(output_dir / "training.pid", "w") as f:
        f.write(str(process.pid))

    print(f"\nTo monitor: tail -f {log_file}")
    print(f"To check status: ps -p {process.pid}")
    print(f"To stop: kill {process.pid}")

    return process.pid


if __name__ == "__main__":
    pid = main()
    sys.exit(0)
