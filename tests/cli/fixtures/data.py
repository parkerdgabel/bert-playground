"""Test data generators and utilities for CLI testing."""

import json
import random
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import mlx.core as mx
import pandas as pd
import yaml


# ==============================================================================
# Tabular Data Generators
# ==============================================================================


def generate_classification_data(
    n_samples: int = 100,
    n_features: int = 10,
    n_classes: int = 2,
    feature_names: Optional[List[str]] = None,
    target_name: str = "target",
    seed: int = 42
) -> pd.DataFrame:
    """Generate synthetic classification data."""
    random.seed(seed)
    
    if feature_names is None:
        feature_names = [f"feature_{i}" for i in range(n_features)]
    
    data = {}
    
    # Generate numeric features
    for i, name in enumerate(feature_names):
        if i % 3 == 0:  # Categorical feature
            data[name] = [random.choice(["A", "B", "C"]) for _ in range(n_samples)]
        elif i % 3 == 1:  # Integer feature
            data[name] = [random.randint(0, 100) for _ in range(n_samples)]
        else:  # Float feature
            data[name] = [random.uniform(0, 1) for _ in range(n_samples)]
    
    # Generate target
    data[target_name] = [random.randint(0, n_classes - 1) for _ in range(n_samples)]
    
    return pd.DataFrame(data)


def generate_regression_data(
    n_samples: int = 100,
    n_features: int = 10,
    feature_names: Optional[List[str]] = None,
    target_name: str = "target",
    noise: float = 0.1,
    seed: int = 42
) -> pd.DataFrame:
    """Generate synthetic regression data."""
    random.seed(seed)
    
    if feature_names is None:
        feature_names = [f"feature_{i}" for i in range(n_features)]
    
    data = {}
    
    # Generate features
    for name in feature_names:
        data[name] = [random.uniform(-1, 1) for _ in range(n_samples)]
    
    # Generate target with linear relationship + noise
    weights = [random.uniform(-2, 2) for _ in range(n_features)]
    data[target_name] = []
    
    for i in range(n_samples):
        target = sum(data[feat][i] * w for feat, w in zip(feature_names, weights))
        target += random.gauss(0, noise)
        data[target_name].append(target)
    
    return pd.DataFrame(data)


def generate_titanic_data(
    n_samples: int = 100,
    include_target: bool = True,
    seed: int = 42
) -> pd.DataFrame:
    """Generate synthetic Titanic-like data."""
    random.seed(seed)
    
    data = {
        "PassengerId": list(range(1, n_samples + 1)),
        "Pclass": [random.choice([1, 2, 3]) for _ in range(n_samples)],
        "Name": [f"Passenger_{i}" for i in range(n_samples)],
        "Sex": [random.choice(["male", "female"]) for _ in range(n_samples)],
        "Age": [random.randint(1, 80) if random.random() > 0.2 else None 
                for _ in range(n_samples)],
        "SibSp": [random.randint(0, 5) for _ in range(n_samples)],
        "Parch": [random.randint(0, 3) for _ in range(n_samples)],
        "Ticket": [f"TKT{random.randint(10000, 99999)}" for _ in range(n_samples)],
        "Fare": [random.uniform(5, 500) for _ in range(n_samples)],
        "Cabin": [f"C{random.randint(1, 200)}" if random.random() > 0.7 else None
                  for _ in range(n_samples)],
        "Embarked": [random.choice(["S", "C", "Q"]) if random.random() > 0.05 else None
                     for _ in range(n_samples)]
    }
    
    if include_target:
        # Simple survival logic based on class and sex
        data["Survived"] = []
        for i in range(n_samples):
            if data["Sex"][i] == "female" and data["Pclass"][i] in [1, 2]:
                survival_prob = 0.9
            elif data["Sex"][i] == "female":
                survival_prob = 0.7
            elif data["Pclass"][i] == 1:
                survival_prob = 0.6
            else:
                survival_prob = 0.2
            
            data["Survived"].append(1 if random.random() < survival_prob else 0)
    
    return pd.DataFrame(data)


def generate_house_prices_data(
    n_samples: int = 100,
    seed: int = 42
) -> pd.DataFrame:
    """Generate synthetic house prices data."""
    random.seed(seed)
    
    data = {
        "Id": list(range(1, n_samples + 1)),
        "MSSubClass": [random.choice([20, 30, 40, 50, 60, 70, 80, 85, 90])
                       for _ in range(n_samples)],
        "LotArea": [random.randint(5000, 20000) for _ in range(n_samples)],
        "OverallQual": [random.randint(1, 10) for _ in range(n_samples)],
        "OverallCond": [random.randint(1, 10) for _ in range(n_samples)],
        "YearBuilt": [random.randint(1900, 2020) for _ in range(n_samples)],
        "YearRemodAdd": [random.randint(1950, 2020) for _ in range(n_samples)],
        "BsmtFinSF1": [random.randint(0, 2000) for _ in range(n_samples)],
        "TotalBsmtSF": [random.randint(0, 3000) for _ in range(n_samples)],
        "1stFlrSF": [random.randint(500, 3000) for _ in range(n_samples)],
        "2ndFlrSF": [random.randint(0, 1500) for _ in range(n_samples)],
        "GrLivArea": [random.randint(500, 5000) for _ in range(n_samples)],
        "BedroomAbvGr": [random.randint(0, 6) for _ in range(n_samples)],
        "KitchenAbvGr": [random.randint(0, 2) for _ in range(n_samples)],
        "TotRmsAbvGrd": [random.randint(3, 12) for _ in range(n_samples)],
        "GarageCars": [random.randint(0, 4) for _ in range(n_samples)],
        "GarageArea": [random.randint(0, 1200) for _ in range(n_samples)]
    }
    
    # Generate price based on features
    data["SalePrice"] = []
    for i in range(n_samples):
        base_price = 50000
        price = base_price
        price += data["OverallQual"][i] * 10000
        price += data["GrLivArea"][i] * 50
        price += data["GarageCars"][i] * 5000
        price += random.randint(-20000, 20000)  # Random variation
        data["SalePrice"].append(max(price, 30000))
    
    return pd.DataFrame(data)


# ==============================================================================
# Configuration Data Generators  
# ==============================================================================


def generate_training_config(
    model_type: str = "bert",
    epochs: int = 5,
    batch_size: int = 32,
    learning_rate: float = 2e-5,
    **kwargs
) -> Dict[str, Any]:
    """Generate training configuration."""
    config = {
        "model": {
            "type": model_type,
            "hidden_size": kwargs.get("hidden_size", 768),
            "num_layers": kwargs.get("num_layers", 12),
            "num_heads": kwargs.get("num_heads", 12),
            "max_length": kwargs.get("max_length", 512)
        },
        "training": {
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "warmup_steps": kwargs.get("warmup_steps", 500),
            "weight_decay": kwargs.get("weight_decay", 0.01),
            "gradient_accumulation_steps": kwargs.get("gradient_accumulation_steps", 1)
        },
        "data": {
            "num_workers": kwargs.get("num_workers", 4),
            "prefetch_size": kwargs.get("prefetch_size", 2),
            "shuffle": kwargs.get("shuffle", True)
        },
        "logging": {
            "log_every_n_steps": kwargs.get("log_every_n_steps", 10),
            "save_every_n_steps": kwargs.get("save_every_n_steps", 500)
        }
    }
    
    return config


def save_config_file(
    config: Dict[str, Any],
    path: Path,
    format: str = "yaml"
) -> Path:
    """Save configuration to file."""
    if format == "yaml":
        with open(path, "w") as f:
            yaml.dump(config, f, default_flow_style=False)
    elif format == "json":
        with open(path, "w") as f:
            json.dump(config, f, indent=2)
    else:
        raise ValueError(f"Unsupported format: {format}")
    
    return path


# ==============================================================================
# Model Data Generators
# ==============================================================================


def generate_model_checkpoint(
    path: Path,
    model_type: str = "bert",
    include_optimizer: bool = True,
    include_metrics: bool = True
) -> Path:
    """Generate a mock model checkpoint."""
    path.mkdir(parents=True, exist_ok=True)
    
    # Model config
    config = {
        "model_type": model_type,
        "hidden_size": 768,
        "num_layers": 12,
        "num_heads": 12,
        "vocab_size": 30522
    }
    
    with open(path / "config.json", "w") as f:
        json.dump(config, f)
    
    # Mock weights file
    (path / "model.safetensors").write_bytes(b"mock_weights")
    
    # Training state
    if include_optimizer or include_metrics:
        state = {
            "epoch": 5,
            "global_step": 1000,
            "best_metric": 0.85
        }
        
        if include_metrics:
            state["metrics"] = {
                "train_loss": [0.5, 0.4, 0.3, 0.25, 0.2],
                "val_loss": [0.6, 0.5, 0.4, 0.35, 0.3],
                "val_accuracy": [0.7, 0.75, 0.8, 0.83, 0.85]
            }
        
        if include_optimizer:
            state["optimizer"] = {
                "type": "adamw",
                "lr": 2e-5,
                "betas": [0.9, 0.999]
            }
        
        with open(path / "training_state.json", "w") as f:
            json.dump(state, f)
    
    return path


def generate_predictions_file(
    path: Path,
    n_samples: int = 100,
    task_type: str = "classification",
    format: str = "csv"
) -> Path:
    """Generate a predictions file."""
    if task_type == "classification":
        data = {
            "id": list(range(n_samples)),
            "prediction": [random.randint(0, 1) for _ in range(n_samples)],
            "confidence": [random.uniform(0.5, 1.0) for _ in range(n_samples)]
        }
    else:  # regression
        data = {
            "id": list(range(n_samples)),
            "prediction": [random.uniform(0, 100) for _ in range(n_samples)]
        }
    
    df = pd.DataFrame(data)
    
    if format == "csv":
        df.to_csv(path, index=False)
    elif format == "json":
        df.to_json(path, orient="records")
    else:
        raise ValueError(f"Unsupported format: {format}")
    
    return path


# ==============================================================================
# MLflow Data Generators
# ==============================================================================


def generate_mlflow_experiment_data(
    experiment_id: str = "1",
    name: str = "test_experiment",
    n_runs: int = 5
) -> Dict[str, Any]:
    """Generate MLflow experiment data."""
    runs = []
    
    for i in range(n_runs):
        run = {
            "run_id": f"run_{i}",
            "experiment_id": experiment_id,
            "status": random.choice(["FINISHED", "RUNNING", "FAILED"]),
            "start_time": datetime.now() - timedelta(hours=n_runs - i),
            "end_time": datetime.now() - timedelta(hours=n_runs - i - 0.5),
            "metrics": {
                "accuracy": 0.7 + i * 0.05,
                "loss": 0.5 - i * 0.05
            },
            "params": {
                "model_type": "bert",
                "batch_size": 32,
                "learning_rate": 2e-5
            },
            "tags": {
                "version": "1.0",
                "dataset": "titanic"
            }
        }
        runs.append(run)
    
    return {
        "experiment_id": experiment_id,
        "name": name,
        "artifact_location": f"./mlruns/{experiment_id}",
        "runs": runs
    }


# ==============================================================================
# Kaggle Data Generators
# ==============================================================================


def generate_kaggle_submission_data(
    competition: str = "titanic",
    n_submissions: int = 10
) -> List[Dict[str, Any]]:
    """Generate Kaggle submission history data."""
    submissions = []
    
    base_date = datetime.now() - timedelta(days=n_submissions)
    
    for i in range(n_submissions):
        submission = {
            "id": f"sub_{i}",
            "competitionId": competition,
            "date": base_date + timedelta(days=i),
            "description": f"Submission {i} - BERT model",
            "fileName": f"submission_{i}.csv",
            "status": "complete",
            "publicScore": 0.75 + i * 0.01,
            "privateScore": None if i < n_submissions - 3 else 0.74 + i * 0.01
        }
        submissions.append(submission)
    
    return submissions


def generate_kaggle_leaderboard_data(
    competition: str = "titanic",
    n_teams: int = 100,
    user_rank: int = 50
) -> Dict[str, Any]:
    """Generate Kaggle leaderboard data."""
    teams = []
    
    for i in range(n_teams):
        team = {
            "teamId": f"team_{i}",
            "teamName": f"Team_{i}" if i != user_rank - 1 else "Your Team",
            "submissionDate": datetime.now() - timedelta(days=random.randint(0, 30)),
            "score": 0.95 - i * 0.001
        }
        teams.append(team)
    
    return {
        "competition": competition,
        "teams": teams,
        "totalTeams": n_teams,
        "userRank": user_rank
    }


# ==============================================================================
# Tensor Data Generators
# ==============================================================================


def generate_batch_data(
    batch_size: int = 32,
    seq_length: int = 128,
    vocab_size: int = 30522
) -> Dict[str, mx.array]:
    """Generate a batch of tensor data."""
    return {
        "input_ids": mx.random.randint(0, vocab_size, (batch_size, seq_length)),
        "attention_mask": mx.ones((batch_size, seq_length)),
        "labels": mx.random.randint(0, 2, (batch_size,))
    }


def generate_model_outputs(
    batch_size: int = 32,
    num_classes: int = 2,
    include_loss: bool = True
) -> Dict[str, mx.array]:
    """Generate model output tensors."""
    outputs = {
        "logits": mx.random.normal((batch_size, num_classes))
    }
    
    if include_loss:
        outputs["loss"] = mx.array(random.uniform(0.1, 0.5))
    
    return outputs