"""TensorBoard writer utilities."""

import os
from typing import Dict, Any, Optional, Union, TYPE_CHECKING
from datetime import datetime
import json

if TYPE_CHECKING:
    import numpy as np


class TensorBoardWriter:
    """Wrapper for TensorBoard SummaryWriter with graceful fallback."""
    
    def __init__(self, log_dir: str, comment: str = ""):
        """Initialize TensorBoard writer.
        
        Args:
            log_dir: Directory for TensorBoard logs
            comment: Comment suffix for the log directory
        """
        self.log_dir = log_dir
        self.comment = comment
        self._writer = None
        self._enabled = True
        
        # Create log directory
        os.makedirs(log_dir, exist_ok=True)
        
        # Try to import and initialize TensorBoard
        try:
            from torch.utils.tensorboard import SummaryWriter
            self._writer = SummaryWriter(log_dir=log_dir, comment=comment)
        except ImportError:
            self._enabled = False
            self._fallback_log = os.path.join(log_dir, "metrics.jsonl")
    
    @property
    def enabled(self) -> bool:
        """Check if TensorBoard is available."""
        return self._enabled and self._writer is not None
    
    def add_scalar(
        self,
        tag: str,
        scalar_value: float,
        global_step: Optional[int] = None,
        walltime: Optional[float] = None
    ) -> None:
        """Add scalar value.
        
        Args:
            tag: Metric name
            scalar_value: Metric value
            global_step: Global step
            walltime: Wall clock time
        """
        if self.enabled:
            self._writer.add_scalar(tag, scalar_value, global_step, walltime)
        else:
            self._write_fallback({
                "tag": tag,
                "value": scalar_value,
                "step": global_step,
                "walltime": walltime or datetime.now().timestamp(),
                "type": "scalar"
            })
    
    def add_scalars(
        self,
        main_tag: str,
        tag_scalar_dict: Dict[str, float],
        global_step: Optional[int] = None,
        walltime: Optional[float] = None
    ) -> None:
        """Add multiple scalars.
        
        Args:
            main_tag: Main tag for grouping
            tag_scalar_dict: Dictionary of tag -> value
            global_step: Global step
            walltime: Wall clock time
        """
        if self.enabled:
            self._writer.add_scalars(main_tag, tag_scalar_dict, global_step, walltime)
        else:
            for tag, value in tag_scalar_dict.items():
                self._write_fallback({
                    "tag": f"{main_tag}/{tag}",
                    "value": value,
                    "step": global_step,
                    "walltime": walltime or datetime.now().timestamp(),
                    "type": "scalar"
                })
    
    def add_histogram(
        self,
        tag: str,
        values: Union['np.ndarray', list],
        global_step: Optional[int] = None,
        bins: str = 'tensorflow',
        walltime: Optional[float] = None
    ) -> None:
        """Add histogram.
        
        Args:
            tag: Histogram name
            values: Values to create histogram from
            global_step: Global step
            bins: Binning method
            walltime: Wall clock time
        """
        if self.enabled:
            self._writer.add_histogram(tag, values, global_step, bins, walltime)
        else:
            # Fallback: store histogram statistics
            try:
                import numpy as np
                values = np.array(values)
                self._write_fallback({
                    "tag": tag,
                    "min": float(np.min(values)),
                    "max": float(np.max(values)),
                    "mean": float(np.mean(values)),
                    "std": float(np.std(values)),
                    "count": len(values),
                    "step": global_step,
                    "walltime": walltime or datetime.now().timestamp(),
                    "type": "histogram"
                })
            except ImportError:
                # If numpy not available, just store basic stats
                values_list = list(values)
                self._write_fallback({
                    "tag": tag,
                    "min": float(min(values_list)),
                    "max": float(max(values_list)),
                    "mean": float(sum(values_list) / len(values_list)),
                    "count": len(values_list),
                    "step": global_step,
                    "walltime": walltime or datetime.now().timestamp(),
                    "type": "histogram"
                })
    
    def add_text(
        self,
        tag: str,
        text_string: str,
        global_step: Optional[int] = None,
        walltime: Optional[float] = None
    ) -> None:
        """Add text.
        
        Args:
            tag: Text tag
            text_string: Text content
            global_step: Global step
            walltime: Wall clock time
        """
        if self.enabled:
            self._writer.add_text(tag, text_string, global_step, walltime)
        else:
            self._write_fallback({
                "tag": tag,
                "text": text_string,
                "step": global_step,
                "walltime": walltime or datetime.now().timestamp(),
                "type": "text"
            })
    
    def add_hparams(
        self,
        hparam_dict: Dict[str, Any],
        metric_dict: Dict[str, float],
        hparam_domain_discrete: Optional[Dict[str, list]] = None,
        run_name: Optional[str] = None
    ) -> None:
        """Add hyperparameters.
        
        Args:
            hparam_dict: Hyperparameters
            metric_dict: Metrics
            hparam_domain_discrete: Discrete parameter domains
            run_name: Run name
        """
        if self.enabled:
            self._writer.add_hparams(
                hparam_dict,
                metric_dict,
                hparam_domain_discrete,
                run_name
            )
        else:
            self._write_fallback({
                "hparams": hparam_dict,
                "metrics": metric_dict,
                "domains": hparam_domain_discrete,
                "run_name": run_name,
                "walltime": datetime.now().timestamp(),
                "type": "hparams"
            })
    
    def flush(self) -> None:
        """Flush pending logs."""
        if self.enabled:
            self._writer.flush()
    
    def close(self) -> None:
        """Close the writer."""
        if self.enabled and self._writer:
            self._writer.close()
    
    def _write_fallback(self, data: Dict[str, Any]) -> None:
        """Write to fallback JSON lines file.
        
        Args:
            data: Data to write
        """
        if hasattr(self, '_fallback_log'):
            with open(self._fallback_log, 'a') as f:
                f.write(json.dumps(data) + '\n')