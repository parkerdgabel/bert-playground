"""Mock objects and utilities for CLI testing."""

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import mlx.core as mx
import pandas as pd

# ==============================================================================
# External Service Mocks
# ==============================================================================


class MockKaggleAPI:
    """Mock Kaggle API for testing."""

    def __init__(self):
        self.competitions = {
            "titanic": {
                "ref": "titanic",
                "title": "Titanic - Machine Learning from Disaster",
                "url": "https://www.kaggle.com/c/titanic",
                "deadline": datetime(2030, 1, 1),
                "category": "Getting Started",
                "reward": "$0",
                "teamCount": 15000,
                "userHasEntered": True,
            },
            "house-prices": {
                "ref": "house-prices",
                "title": "House Prices - Advanced Regression",
                "url": "https://www.kaggle.com/c/house-prices",
                "deadline": datetime(2030, 1, 1),
                "category": "Getting Started",
                "reward": "$0",
                "teamCount": 5000,
                "userHasEntered": False,
            },
        }

        self.datasets = {
            "titanic/train": {"size": 59760, "files": ["train.csv"]},
            "titanic/test": {"size": 27960, "files": ["test.csv"]},
        }

        self.submissions = []

    def competitions_list(self, page=1, search="", category="", sort="latestDeadline"):
        """Mock competition listing."""
        results = list(self.competitions.values())

        if search:
            results = [c for c in results if search.lower() in c["title"].lower()]

        if category and category != "all":
            results = [c for c in results if c["category"].lower() == category.lower()]

        # Convert to mock objects
        return [type("Competition", (), comp) for comp in results]

    def competition_download_files(
        self, competition, path=".", force=False, quiet=False
    ):
        """Mock competition file download."""
        if competition not in self.competitions:
            raise ValueError(f"Competition {competition} not found")

        # Create dummy files
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        if competition == "titanic":
            # Create train.csv
            train_data = pd.DataFrame(
                {
                    "PassengerId": list(range(1, 892)),
                    "Survived": [0, 1] * 445 + [1],
                    "Pclass": [1, 2, 3] * 297,
                    "Sex": ["male", "female"] * 445 + ["male"],
                    "Age": [22.0] * 891,
                }
            )
            train_data.to_csv(path / "train.csv", index=False)

            # Create test.csv
            test_data = pd.DataFrame(
                {
                    "PassengerId": list(range(892, 1310)),
                    "Pclass": [1, 2, 3] * 139,
                    "Sex": ["male", "female"] * 209,
                    "Age": [30.0] * 418,
                }
            )
            test_data.to_csv(path / "test.csv", index=False)

        return ["train.csv", "test.csv"]

    def competition_submit(self, file_path, message, competition, quiet=False):
        """Mock competition submission."""
        if competition not in self.competitions:
            raise ValueError(f"Competition {competition} not found")

        submission = {
            "id": len(self.submissions) + 1,
            "date": datetime.now(),
            "description": message,
            "status": "complete",
            "publicScore": 0.75 + len(self.submissions) * 0.01,
            "privateScore": None,
        }
        self.submissions.append(submission)

        return {"status": "complete", "id": submission["id"]}

    def competition_submissions(self, competition):
        """Get submission history."""
        return [type("Submission", (), sub) for sub in self.submissions]

    def competition_leaderboard_view(self, competition):
        """Get competition leaderboard."""
        return {
            "submissions": [
                {"teamName": f"Team{i}", "score": 0.9 - i * 0.01} for i in range(10)
            ]
        }


class MockMLflowClient:
    """Mock MLflow client for testing."""

    def __init__(self):
        self.experiments = {
            "0": {
                "experiment_id": "0",
                "name": "Default",
                "artifact_location": "./mlruns/0",
                "lifecycle_stage": "active",
            }
        }
        self.runs = {}
        self.run_counter = 0

    def create_experiment(self, name, artifact_location=None):
        """Create a new experiment."""
        exp_id = str(len(self.experiments))
        self.experiments[exp_id] = {
            "experiment_id": exp_id,
            "name": name,
            "artifact_location": artifact_location or f"./mlruns/{exp_id}",
            "lifecycle_stage": "active",
        }
        return exp_id

    def get_experiment_by_name(self, name):
        """Get experiment by name."""
        for exp in self.experiments.values():
            if exp["name"] == name:
                return type("Experiment", (), exp)
        return None

    def search_experiments(self, filter_string="", order_by=None):
        """Search experiments."""
        results = list(self.experiments.values())
        return [type("Experiment", (), exp) for exp in results]

    def create_run(self, experiment_id, start_time=None):
        """Create a new run."""
        run_id = f"run_{self.run_counter}"
        self.run_counter += 1

        run = {
            "run_id": run_id,
            "experiment_id": experiment_id,
            "status": "RUNNING",
            "start_time": start_time or datetime.now().timestamp(),
            "metrics": {},
            "params": {},
            "tags": {},
        }
        self.runs[run_id] = run

        return type("Run", (), {"info": type("RunInfo", (), {"run_id": run_id})})

    def log_metric(self, run_id, key, value, timestamp=None, step=None):
        """Log a metric."""
        if run_id in self.runs:
            if key not in self.runs[run_id]["metrics"]:
                self.runs[run_id]["metrics"][key] = []
            self.runs[run_id]["metrics"][key].append(
                {
                    "value": value,
                    "timestamp": timestamp or datetime.now().timestamp(),
                    "step": step or 0,
                }
            )

    def log_param(self, run_id, key, value):
        """Log a parameter."""
        if run_id in self.runs:
            self.runs[run_id]["params"][key] = value

    def set_terminated(self, run_id, status="FINISHED"):
        """Terminate a run."""
        if run_id in self.runs:
            self.runs[run_id]["status"] = status


class MockModelServer:
    """Mock model server for testing."""

    def __init__(self, model_path: str, host: str = "0.0.0.0", port: int = 8080):
        self.model_path = model_path
        self.host = host
        self.port = port
        self.is_running = False
        self.requests = []

    def start(self):
        """Start the server."""
        self.is_running = True
        return True

    def stop(self):
        """Stop the server."""
        self.is_running = False
        return True

    def predict(self, data: dict[str, Any]) -> dict[str, Any]:
        """Make a prediction."""
        self.requests.append(data)

        # Mock prediction response
        if "instances" in data:
            predictions = [0.8] * len(data["instances"])
        else:
            predictions = [0.8]

        return {
            "predictions": predictions,
            "model_version": "1.0",
            "timestamp": datetime.now().isoformat(),
        }

    def health_check(self) -> dict[str, Any]:
        """Check server health."""
        return {
            "status": "healthy" if self.is_running else "stopped",
            "uptime": 100,
            "requests_served": len(self.requests),
        }


# ==============================================================================
# Model and Training Mocks
# ==============================================================================


class MockBertModel:
    """Mock BERT model for testing."""

    def __init__(self, config: dict | None = None):
        self.config = config or {
            "hidden_size": 128,
            "num_layers": 2,
            "num_heads": 2,
            "vocab_size": 30522,
        }
        self.training = True

    def __call__(self, input_ids, attention_mask=None, labels=None):
        """Forward pass."""
        batch_size = input_ids.shape[0]

        # Mock outputs
        logits = mx.random.normal((batch_size, 2))

        if labels is not None:
            loss = mx.random.uniform(0.1, 0.5)
            return {"loss": loss, "logits": logits}

        return {"logits": logits}

    def train(self, mode=True):
        """Set training mode."""
        self.training = mode
        return self

    def eval(self):
        """Set evaluation mode."""
        return self.train(False)

    def parameters(self):
        """Get model parameters."""
        # Mock parameters
        return [
            mx.random.normal((self.config["vocab_size"], self.config["hidden_size"])),
            mx.random.normal((self.config["hidden_size"], self.config["hidden_size"])),
        ]

    def save(self, path: str):
        """Save model."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # Save config
        with open(path / "config.json", "w") as f:
            json.dump(self.config, f)

        # Mock weights file
        (path / "model.safetensors").touch()

    @classmethod
    def load(cls, path: str):
        """Load model."""
        path = Path(path)

        # Load config
        with open(path / "config.json") as f:
            config = json.load(f)

        return cls(config)


class MockTrainer:
    """Mock trainer for testing."""

    def __init__(self, model, train_loader, val_loader, config):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.current_epoch = 0
        self.metrics = []

    def train(self):
        """Run training."""
        for epoch in range(self.config.get("epochs", 1)):
            self.current_epoch = epoch

            # Mock training metrics
            train_loss = 0.5 - epoch * 0.05
            val_loss = 0.4 - epoch * 0.03
            val_accuracy = 0.7 + epoch * 0.05

            metrics = {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_accuracy": val_accuracy,
            }
            self.metrics.append(metrics)

        return self.metrics

    def predict(self, test_loader):
        """Generate predictions."""
        predictions = []

        for batch in test_loader:
            batch_size = len(batch["input_ids"])
            # Mock predictions
            preds = mx.random.randint(0, 2, (batch_size,))
            predictions.extend(preds.tolist())

        return predictions

    def save_checkpoint(self, path: str):
        """Save training checkpoint."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # Save model
        self.model.save(path)

        # Save training state
        state = {
            "epoch": self.current_epoch,
            "metrics": self.metrics,
            "config": self.config,
        }

        with open(path / "training_state.json", "w") as f:
            json.dump(state, f)


# ==============================================================================
# Data Mocks
# ==============================================================================


class MockDataset:
    """Mock dataset for testing."""

    def __init__(self, size: int = 100, features: int = 10):
        self.size = size
        self.features = features

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        if idx >= self.size:
            raise IndexError("Index out of range")

        return {
            "input_ids": mx.random.randint(0, 30522, (128,)),
            "attention_mask": mx.ones((128,)),
            "labels": mx.random.randint(0, 2, ()),
        }


class MockDataLoader:
    """Mock data loader for testing."""

    def __init__(self, dataset, batch_size: int = 32, shuffle: bool = True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __iter__(self):
        for i in range(0, len(self.dataset), self.batch_size):
            batch_indices = range(i, min(i + self.batch_size, len(self.dataset)))
            batch = [self.dataset[idx] for idx in batch_indices]

            # Stack batch
            yield {
                "input_ids": mx.stack([b["input_ids"] for b in batch]),
                "attention_mask": mx.stack([b["attention_mask"] for b in batch]),
                "labels": mx.stack([b["labels"] for b in batch]),
            }

    def __len__(self):
        return (len(self.dataset) + self.batch_size - 1) // self.batch_size


# ==============================================================================
# Utility Mocks
# ==============================================================================


class MockProgressBar:
    """Mock progress bar for testing."""

    def __init__(self, total: int, desc: str = ""):
        self.total = total
        self.desc = desc
        self.current = 0
        self.updates = []

    def update(self, n: int = 1):
        """Update progress."""
        self.current += n
        self.updates.append({"n": n, "current": self.current})

    def set_postfix(self, **kwargs):
        """Set postfix information."""
        self.updates.append({"postfix": kwargs})

    def close(self):
        """Close progress bar."""
        pass


class MockConsole:
    """Mock console for capturing output."""

    def __init__(self):
        self.messages = []

    def print(self, *args, **kwargs):
        """Capture print calls."""
        self.messages.append({"type": "print", "args": args, "kwargs": kwargs})

    def error(self, message: str):
        """Capture error messages."""
        self.messages.append({"type": "error", "message": message})

    def success(self, message: str):
        """Capture success messages."""
        self.messages.append({"type": "success", "message": message})

    def warning(self, message: str):
        """Capture warning messages."""
        self.messages.append({"type": "warning", "message": message})

    def info(self, message: str):
        """Capture info messages."""
        self.messages.append({"type": "info", "message": message})
