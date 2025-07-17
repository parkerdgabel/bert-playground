"""MLflow evaluation utilities with custom metrics and visualizations."""

import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import mlflow
import mlx.core as mx
import numpy as np
import pandas as pd
import seaborn as sns
from loguru import logger
from mlflow.models import EvaluationMetric, make_metric
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


class TitanicMetrics:
    """Custom metrics for Titanic survival prediction."""
    
    @staticmethod
    def survival_rate_error() -> EvaluationMetric:
        """Metric for survival rate prediction error."""
        
        def _survival_rate_error(y_true, y_pred, sample_weight=None):
            """Calculate difference between predicted and actual survival rates."""
            actual_rate = np.mean(y_true)
            predicted_rate = np.mean(y_pred)
            return abs(actual_rate - predicted_rate)
        
        return make_metric(
            eval_fn=_survival_rate_error,
            greater_is_better=False,
            name="survival_rate_error",
        )
    
    @staticmethod
    def class_weighted_accuracy() -> EvaluationMetric:
        """Metric for class-weighted accuracy."""
        
        def _class_weighted_accuracy(y_true, y_pred, sample_weight=None):
            """Calculate accuracy weighted by class frequency."""
            from sklearn.utils.class_weight import compute_class_weight
            
            classes = np.unique(y_true)
            class_weights = compute_class_weight(
                class_weight="balanced",
                classes=classes,
                y=y_true,
            )
            
            # Create sample weights based on class
            weights = np.zeros_like(y_true, dtype=float)
            for i, cls in enumerate(classes):
                weights[y_true == cls] = class_weights[i]
            
            correct = (y_true == y_pred).astype(float)
            weighted_accuracy = np.average(correct, weights=weights)
            return weighted_accuracy
        
        return make_metric(
            eval_fn=_class_weighted_accuracy,
            greater_is_better=True,
            name="class_weighted_accuracy",
        )
    
    @staticmethod
    def demographic_parity_difference(sensitive_feature: str) -> EvaluationMetric:
        """Metric for demographic parity (fairness metric)."""
        
        def _demographic_parity(eval_df):
            """Calculate demographic parity difference."""
            if sensitive_feature not in eval_df.columns:
                return np.nan
            
            # Group by sensitive feature and calculate positive prediction rates
            groups = eval_df.groupby(sensitive_feature)
            positive_rates = groups["prediction"].mean()
            
            # Calculate max difference between groups
            if len(positive_rates) > 1:
                return positive_rates.max() - positive_rates.min()
            return 0.0
        
        return make_metric(
            eval_fn=_demographic_parity,
            greater_is_better=False,
            name=f"demographic_parity_{sensitive_feature}",
            aggregate=False,  # Operates on entire dataframe
        )
    
    @staticmethod
    def false_negative_rate_survived() -> EvaluationMetric:
        """Metric for false negative rate among actual survivors."""
        
        def _fnr_survived(y_true, y_pred, sample_weight=None):
            """Calculate FNR for survived class (critical for safety)."""
            # Only consider actual survivors
            survived_mask = y_true == 1
            if not np.any(survived_mask):
                return 0.0
            
            # False negatives: predicted dead but actually survived
            false_negatives = np.sum((y_pred[survived_mask] == 0))
            total_survived = np.sum(survived_mask)
            
            return false_negatives / total_survived
        
        return make_metric(
            eval_fn=_fnr_survived,
            greater_is_better=False,
            name="false_negative_rate_survived",
        )


class ModelEvaluator:
    """Comprehensive model evaluation with MLflow integration."""
    
    def __init__(self, model_uri: Optional[str] = None):
        """Initialize evaluator.
        
        Args:
            model_uri: Optional model URI for loading from MLflow
        """
        self.model_uri = model_uri
        self.results = {}
    
    def evaluate_model(
        self,
        model: Optional[Any] = None,
        data: Union[pd.DataFrame, Dict[str, np.ndarray]],
        model_type: str = "classifier",
        evaluators: Optional[List[str]] = None,
        custom_metrics: Optional[List[EvaluationMetric]] = None,
        feature_names: Optional[List[str]] = None,
        plot_results: bool = True,
    ) -> Dict[str, Any]:
        """Evaluate model with comprehensive metrics and visualizations.
        
        Args:
            model: Model to evaluate (if None, uses model_uri)
            data: Evaluation data (DataFrame or dict with X, y)
            model_type: Type of model ("classifier" or "regressor")
            evaluators: Optional list of evaluators to use
            custom_metrics: Optional list of custom metrics
            feature_names: Optional feature names for interpretability
            plot_results: Whether to generate plots
            
        Returns:
            Evaluation results dictionary
        """
        logger.info("Starting model evaluation")
        
        # Prepare data
        if isinstance(data, pd.DataFrame):
            # Assume last column is target
            X = data.iloc[:, :-1]
            y = data.iloc[:, -1]
            eval_data = data
        else:
            X = data["X"]
            y = data["y"]
            eval_data = pd.DataFrame(X)
            eval_data["target"] = y
        
        # Default evaluators
        if evaluators is None:
            evaluators = ["default"]
        
        # Default custom metrics for classification
        if custom_metrics is None and model_type == "classifier":
            custom_metrics = [
                TitanicMetrics.survival_rate_error(),
                TitanicMetrics.class_weighted_accuracy(),
                TitanicMetrics.false_negative_rate_survived(),
            ]
        
        # Run MLflow evaluation
        with mlflow.start_run(nested=True, run_name="model_evaluation"):
            if model is not None:
                # Log the model if provided
                mlflow.sklearn.log_model(model, "model")
                model_uri = mlflow.get_artifact_uri("model")
            elif self.model_uri:
                model_uri = self.model_uri
            else:
                raise ValueError("Either model or model_uri must be provided")
            
            # Run evaluation
            result = mlflow.evaluate(
                model=model_uri,
                data=eval_data,
                targets="target",
                model_type=model_type,
                evaluators=evaluators,
                extra_metrics=custom_metrics,
            )
            
            # Store results
            self.results = {
                "metrics": result.metrics,
                "artifacts": result.artifacts,
            }
            
            # Generate additional plots if requested
            if plot_results and model_type == "classifier":
                self._generate_classification_plots(model, X, y, feature_names)
            
            # Log summary
            self._log_evaluation_summary()
        
        return self.results
    
    def _generate_classification_plots(
        self,
        model: Any,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: Optional[List[str]] = None,
    ) -> None:
        """Generate classification-specific plots."""
        # Get predictions
        if hasattr(model, "predict_proba"):
            y_proba = model.predict_proba(X)[:, 1]
        else:
            y_proba = model.predict(X)
        y_pred = (y_proba > 0.5).astype(int)
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Enhanced confusion matrix
        cm = confusion_matrix(y, y_pred)
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            ax=axes[0, 0],
            xticklabels=["Not Survived", "Survived"],
            yticklabels=["Not Survived", "Survived"],
        )
        axes[0, 0].set_title("Confusion Matrix")
        axes[0, 0].set_xlabel("Predicted")
        axes[0, 0].set_ylabel("Actual")
        
        # Add percentages
        cm_percent = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis] * 100
        for i in range(2):
            for j in range(2):
                axes[0, 0].text(
                    j + 0.5,
                    i + 0.7,
                    f"({cm_percent[i, j]:.1f}%)",
                    ha="center",
                    va="center",
                    fontsize=9,
                    color="gray",
                )
        
        # 2. ROC Curve with AUC
        fpr, tpr, _ = roc_curve(y, y_proba)
        auc = roc_auc_score(y, y_proba)
        axes[0, 1].plot(fpr, tpr, label=f"ROC (AUC = {auc:.3f})")
        axes[0, 1].plot([0, 1], [0, 1], "k--", label="Random")
        axes[0, 1].set_xlabel("False Positive Rate")
        axes[0, 1].set_ylabel("True Positive Rate")
        axes[0, 1].set_title("ROC Curve")
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Precision-Recall Curve
        precision, recall, _ = precision_recall_curve(y, y_proba)
        axes[1, 0].plot(recall, precision)
        axes[1, 0].set_xlabel("Recall")
        axes[1, 0].set_ylabel("Precision")
        axes[1, 0].set_title("Precision-Recall Curve")
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. Prediction Distribution
        axes[1, 1].hist(
            y_proba[y == 0],
            bins=30,
            alpha=0.5,
            label="Not Survived",
            color="red",
        )
        axes[1, 1].hist(
            y_proba[y == 1],
            bins=30,
            alpha=0.5,
            label="Survived",
            color="green",
        )
        axes[1, 1].set_xlabel("Predicted Probability")
        axes[1, 1].set_ylabel("Count")
        axes[1, 1].set_title("Prediction Distribution by Class")
        axes[1, 1].legend()
        axes[1, 1].axvline(x=0.5, color="black", linestyle="--", alpha=0.5)
        
        plt.tight_layout()
        mlflow.log_figure(fig, "evaluation_plots.png")
        plt.close()
        
        # Generate classification report
        report = classification_report(
            y,
            y_pred,
            target_names=["Not Survived", "Survived"],
            output_dict=True,
        )
        
        # Create classification report visualization
        fig, ax = plt.subplots(figsize=(10, 6))
        report_df = pd.DataFrame(report).transpose()
        sns.heatmap(
            report_df.iloc[:-1, :-1],
            annot=True,
            cmap="YlOrRd",
            fmt=".3f",
            ax=ax,
        )
        ax.set_title("Classification Report Heatmap")
        plt.tight_layout()
        mlflow.log_figure(fig, "classification_report.png")
        plt.close()
        
        # Log classification report as artifact
        report_path = Path("classification_report.json")
        report_path.write_text(json.dumps(report, indent=2))
        mlflow.log_artifact(str(report_path))
        report_path.unlink()
    
    def _log_evaluation_summary(self) -> None:
        """Log evaluation summary."""
        if not self.results:
            return
        
        metrics = self.results["metrics"]
        
        # Create summary
        summary = [
            "# Model Evaluation Summary\n",
            f"**Accuracy**: {metrics.get('accuracy', 'N/A'):.4f}",
            f"**Precision**: {metrics.get('precision', 'N/A'):.4f}",
            f"**Recall**: {metrics.get('recall', 'N/A'):.4f}",
            f"**F1 Score**: {metrics.get('f1_score', 'N/A'):.4f}",
            f"**ROC AUC**: {metrics.get('roc_auc', 'N/A'):.4f}",
        ]
        
        # Add custom metrics
        for key, value in metrics.items():
            if key not in ["accuracy", "precision", "recall", "f1_score", "roc_auc"]:
                summary.append(f"**{key.replace('_', ' ').title()}**: {value:.4f}")
        
        summary_text = "\n".join(summary)
        
        # Log as artifact
        summary_path = Path("evaluation_summary.md")
        summary_path.write_text(summary_text)
        mlflow.log_artifact(str(summary_path))
        summary_path.unlink()
        
        logger.info("Evaluation complete")
        logger.info(summary_text)


def create_evaluation_dataset(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: Optional[List[str]] = None,
    predictions: Optional[np.ndarray] = None,
    prediction_proba: Optional[np.ndarray] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> pd.DataFrame:
    """Create evaluation dataset for mlflow.evaluate().
    
    Args:
        X: Feature array
        y: True labels
        feature_names: Optional feature names
        predictions: Optional predictions
        prediction_proba: Optional prediction probabilities
        metadata: Optional metadata columns
        
    Returns:
        DataFrame ready for evaluation
    """
    # Create base dataframe
    if feature_names:
        df = pd.DataFrame(X, columns=feature_names)
    else:
        df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(X.shape[1])])
    
    # Add target
    df["target"] = y
    
    # Add predictions if provided
    if predictions is not None:
        df["prediction"] = predictions
    
    if prediction_proba is not None:
        if len(prediction_proba.shape) > 1:
            # Multi-class probabilities
            for i in range(prediction_proba.shape[1]):
                df[f"prediction_proba_class_{i}"] = prediction_proba[:, i]
        else:
            # Binary probabilities
            df["prediction_proba"] = prediction_proba
    
    # Add metadata if provided
    if metadata:
        for key, value in metadata.items():
            df[key] = value
    
    return df


def evaluate_with_cross_validation(
    model_class: type,
    X: np.ndarray,
    y: np.ndarray,
    cv_folds: int = 5,
    model_params: Optional[Dict[str, Any]] = None,
    custom_metrics: Optional[List[EvaluationMetric]] = None,
    experiment_name: Optional[str] = None,
) -> Dict[str, Any]:
    """Evaluate model using cross-validation with MLflow tracking.
    
    Args:
        model_class: Model class to instantiate
        X: Feature array
        y: Target array
        cv_folds: Number of CV folds
        model_params: Model parameters
        custom_metrics: Custom evaluation metrics
        experiment_name: MLflow experiment name
        
    Returns:
        Cross-validation results
    """
    from sklearn.model_selection import StratifiedKFold
    
    if experiment_name:
        mlflow.set_experiment(experiment_name)
    
    # Initialize cross-validation
    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    
    cv_results = {
        "fold_metrics": [],
        "aggregate_metrics": {},
    }
    
    with mlflow.start_run(run_name=f"cv_evaluation_{cv_folds}_folds"):
        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
            logger.info(f"Evaluating fold {fold + 1}/{cv_folds}")
            
            # Split data
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            # Train model
            model = model_class(**(model_params or {}))
            model.fit(X_train, y_train)
            
            # Create evaluation data
            eval_data = create_evaluation_dataset(X_val, y_val)
            
            # Evaluate
            evaluator = ModelEvaluator()
            with mlflow.start_run(nested=True, run_name=f"fold_{fold + 1}"):
                results = evaluator.evaluate_model(
                    model=model,
                    data=eval_data,
                    custom_metrics=custom_metrics,
                    plot_results=(fold == 0),  # Only plot for first fold
                )
                
                cv_results["fold_metrics"].append(results["metrics"])
        
        # Aggregate metrics
        all_metrics = {}
        for metrics in cv_results["fold_metrics"]:
            for key, value in metrics.items():
                if key not in all_metrics:
                    all_metrics[key] = []
                all_metrics[key].append(value)
        
        # Calculate mean and std
        for key, values in all_metrics.items():
            cv_results["aggregate_metrics"][f"{key}_mean"] = np.mean(values)
            cv_results["aggregate_metrics"][f"{key}_std"] = np.std(values)
            
            # Log aggregate metrics
            mlflow.log_metric(f"cv_{key}_mean", np.mean(values))
            mlflow.log_metric(f"cv_{key}_std", np.std(values))
        
        # Log summary
        summary = f"# Cross-Validation Results ({cv_folds} folds)\n\n"
        for key, values in all_metrics.items():
            summary += f"**{key}**: {np.mean(values):.4f} ± {np.std(values):.4f}\n"
        
        summary_path = Path("cv_summary.md")
        summary_path.write_text(summary)
        mlflow.log_artifact(str(summary_path))
        summary_path.unlink()
    
    return cv_results