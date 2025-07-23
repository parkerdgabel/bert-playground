"""Metrics calculation port."""

from typing import Protocol, Dict, Any, List, Optional, Union


class MetricsCalculatorPort(Protocol):
    """Port for metrics calculation operations."""
    
    def calculate_accuracy(
        self,
        predictions: Union[List[int], Any],
        labels: Union[List[int], Any],
    ) -> float:
        """Calculate accuracy.
        
        Args:
            predictions: Predicted labels
            labels: True labels
            
        Returns:
            Accuracy score
        """
        ...
    
    def calculate_precision_recall_f1(
        self,
        predictions: Union[List[int], Any],
        labels: Union[List[int], Any],
        average: str = "macro",
    ) -> Dict[str, float]:
        """Calculate precision, recall, and F1 scores.
        
        Args:
            predictions: Predicted labels
            labels: True labels
            average: Averaging method ('macro', 'micro', 'weighted')
            
        Returns:
            Dictionary with precision, recall, and f1 scores
        """
        ...
    
    def calculate_confusion_matrix(
        self,
        predictions: Union[List[int], Any],
        labels: Union[List[int], Any],
    ) -> List[List[int]]:
        """Calculate confusion matrix.
        
        Args:
            predictions: Predicted labels
            labels: True labels
            
        Returns:
            Confusion matrix
        """
        ...
    
    def calculate_auc_roc(
        self,
        probabilities: Union[List[float], Any],
        labels: Union[List[int], Any],
        multi_class: str = "ovr",
    ) -> float:
        """Calculate AUC-ROC score.
        
        Args:
            probabilities: Predicted probabilities
            labels: True labels
            multi_class: Multi-class strategy ('ovr', 'ovo')
            
        Returns:
            AUC-ROC score
        """
        ...
    
    def calculate_auc_pr(
        self,
        probabilities: Union[List[float], Any],
        labels: Union[List[int], Any],
    ) -> float:
        """Calculate AUC-PR (Area Under Precision-Recall curve).
        
        Args:
            probabilities: Predicted probabilities
            labels: True labels
            
        Returns:
            AUC-PR score
        """
        ...
    
    def calculate_mse(
        self,
        predictions: Union[List[float], Any],
        targets: Union[List[float], Any],
    ) -> float:
        """Calculate mean squared error.
        
        Args:
            predictions: Predicted values
            targets: True values
            
        Returns:
            MSE value
        """
        ...
    
    def calculate_mae(
        self,
        predictions: Union[List[float], Any],
        targets: Union[List[float], Any],
    ) -> float:
        """Calculate mean absolute error.
        
        Args:
            predictions: Predicted values
            targets: True values
            
        Returns:
            MAE value
        """
        ...
    
    def calculate_r2_score(
        self,
        predictions: Union[List[float], Any],
        targets: Union[List[float], Any],
    ) -> float:
        """Calculate R-squared score.
        
        Args:
            predictions: Predicted values
            targets: True values
            
        Returns:
            R-squared score
        """
        ...
    
    def calculate_per_class_metrics(
        self,
        predictions: Union[List[int], Any],
        labels: Union[List[int], Any],
        class_names: Optional[List[str]] = None,
    ) -> Dict[str, Dict[str, float]]:
        """Calculate per-class metrics.
        
        Args:
            predictions: Predicted labels
            labels: True labels
            class_names: Optional class names
            
        Returns:
            Dictionary mapping class to metrics
        """
        ...
    
    def calculate_loss(
        self,
        logits: Any,
        labels: Any,
        loss_type: str = "cross_entropy",
        **kwargs: Any,
    ) -> float:
        """Calculate loss value.
        
        Args:
            logits: Model predictions
            labels: True labels
            loss_type: Type of loss function
            **kwargs: Additional loss parameters
            
        Returns:
            Loss value
        """
        ...
    
    def aggregate_metrics(
        self,
        metrics_list: List[Dict[str, float]],
    ) -> Dict[str, float]:
        """Aggregate metrics from multiple batches.
        
        Args:
            metrics_list: List of metric dictionaries
            
        Returns:
            Aggregated metrics
        """
        ...