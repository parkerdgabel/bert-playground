"""Model training service - pure business logic."""

from typing import Optional, Dict, Any, List, Callable
from dataclasses import dataclass
from domain.entities.model import BertModel
from domain.entities.training import TrainingSession, TrainingState, TrainingConfig
from domain.entities.dataset import Dataset, DataBatch
from domain.entities.metrics import TrainingMetrics, EvaluationMetrics
from domain.ports.compute import ComputePort
from domain.ports.data import DataLoaderPort
from domain.ports.monitoring import MonitoringPort
from domain.ports.storage import CheckpointPort
from domain.ports.metrics import MetricsCalculatorPort


@dataclass
class ModelTrainingService:
    """Service for training BERT models.
    
    This service orchestrates the training process without any
    framework-specific implementation details.
    """
    compute_port: ComputePort
    data_loader_port: DataLoaderPort
    monitoring_port: MonitoringPort
    checkpoint_port: CheckpointPort
    metrics_port: MetricsCalculatorPort
    
    def train(
        self,
        model: BertModel,
        training_session: TrainingSession,
        train_dataset: Dataset,
        eval_dataset: Optional[Dataset] = None,
        callbacks: Optional[List[Callable]] = None,
    ) -> TrainingSession:
        """Execute training process.
        
        Args:
            model: Model to train
            training_session: Training session configuration
            train_dataset: Training dataset
            eval_dataset: Optional evaluation dataset
            callbacks: Optional training callbacks
            
        Returns:
            Completed training session
        """
        config = training_session.config
        state = training_session.state
        
        # Initialize training
        self._initialize_training(model, config, state)
        
        # Create data loader
        train_loader = self.data_loader_port.create_dataloader(
            dataset=train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=0,
            prefetch_size=4,
        )
        
        # Create optimizer
        optimizer_state = self.compute_port.create_optimizer(
            model=model,
            optimizer_type=config.optimizer_type.value,
            learning_rate=config.learning_rate,
            weight_decay=config.weight_decay,
            beta1=config.adam_beta1,
            beta2=config.adam_beta2,
            epsilon=config.adam_epsilon,
        )
        
        # Training loop
        while not training_session.is_completed and not training_session.should_stop_early:
            state.epoch += 1
            train_loader.set_epoch(state.epoch)
            
            # Epoch training
            epoch_metrics = self._train_epoch(
                model=model,
                train_loader=train_loader,
                optimizer_state=optimizer_state,
                config=config,
                state=state,
                callbacks=callbacks,
            )
            
            # Evaluation
            if eval_dataset and self._should_evaluate(state, config):
                eval_metrics = self._evaluate(
                    model=model,
                    dataset=eval_dataset,
                    config=config,
                )
                self.monitoring_port.log_evaluation_metrics(eval_metrics)
                
                # Update best metric
                primary_metric_name, primary_metric_value = eval_metrics.get_primary_metric()
                improved = state.update_best_metric(primary_metric_value)
                
                if improved and callbacks:
                    for callback in callbacks:
                        callback('on_improve', model, state, eval_metrics)
            
            # Checkpointing
            if self._should_checkpoint(state, config):
                checkpoint_path = f"checkpoint_epoch_{state.epoch}_step_{state.global_step}"
                self.checkpoint_port.save_checkpoint(
                    model=model,
                    training_state=state,
                    optimizer_state=optimizer_state,
                    path=checkpoint_path,
                    metadata={
                        "epoch": state.epoch,
                        "global_step": state.global_step,
                        "best_metric": state.best_metric,
                    }
                )
                training_session.add_checkpoint(checkpoint_path)
        
        # Final evaluation
        if eval_dataset:
            final_metrics = self._evaluate(model, eval_dataset, config)
            training_session.final_metrics = final_metrics.to_dict()
        
        return training_session
    
    def _initialize_training(
        self,
        model: BertModel,
        config: TrainingConfig,
        state: TrainingState,
    ) -> None:
        """Initialize training process."""
        # Start monitoring run
        run_id = self.monitoring_port.start_run(
            run_name=f"bert_training_{state.epoch}",
            tags={"model_type": "bert", "framework": "mlx"},
        )
        
        # Log hyperparameters
        self.monitoring_port.log_hyperparameters(config.__dict__)
        
        # Compile model if requested
        if config.compile_model:
            model = self.compute_port.compile_model(model)
    
    def _train_epoch(
        self,
        model: BertModel,
        train_loader: DataLoaderPort,
        optimizer_state: Dict[str, Any],
        config: TrainingConfig,
        state: TrainingState,
        callbacks: Optional[List[Callable]] = None,
    ) -> List[TrainingMetrics]:
        """Train for one epoch."""
        epoch_metrics = []
        accumulated_loss = 0.0
        
        # Create progress bar
        progress = self.monitoring_port.create_progress_bar(
            total=len(train_loader),
            description=f"Epoch {state.epoch}",
            unit="batch",
        )
        
        for batch_idx, batch in enumerate(train_loader):
            # Forward pass
            with self.compute_port.mixed_precision_context(config.mixed_precision):
                outputs = self.compute_port.forward(
                    model=model,
                    batch=batch,
                    training=True,
                )
                loss = outputs['loss']
                accumulated_loss += loss
            
            # Backward pass
            if (batch_idx + 1) % config.gradient_accumulation_steps == 0:
                # Scale loss by accumulation steps
                scaled_loss = accumulated_loss / config.gradient_accumulation_steps
                
                # Compute gradients
                self.compute_port.backward(scaled_loss)
                
                # Calculate learning rate
                learning_rate = self._get_learning_rate(
                    state.global_step,
                    config,
                )
                
                # Optimization step
                optimizer_state, grad_norm = self.compute_port.optimize_step(
                    model=model,
                    optimizer_state=optimizer_state,
                    learning_rate=learning_rate,
                    max_grad_norm=config.max_grad_norm,
                )
                
                # Update state
                state.update_step(
                    loss=accumulated_loss / config.gradient_accumulation_steps,
                    learning_rate=learning_rate,
                )
                
                # Create metrics
                metrics = TrainingMetrics(
                    epoch=state.epoch,
                    step=state.global_step,
                    loss=accumulated_loss / config.gradient_accumulation_steps,
                    learning_rate=learning_rate,
                    gradient_norm=grad_norm,
                )
                epoch_metrics.append(metrics)
                
                # Log metrics
                if state.global_step % config.logging_steps == 0:
                    self.monitoring_port.log_training_metrics(metrics)
                
                # Update progress
                progress.set_postfix(
                    loss=f"{metrics.loss:.4f}",
                    lr=f"{learning_rate:.2e}",
                )
                
                # Reset accumulated loss
                accumulated_loss = 0.0
            
            progress.update(1)
            
            # Callbacks
            if callbacks:
                for callback in callbacks:
                    callback('on_batch_end', model, state, batch_idx)
        
        progress.close()
        state.complete_epoch()
        
        return epoch_metrics
    
    def _evaluate(
        self,
        model: BertModel,
        dataset: Dataset,
        config: TrainingConfig,
    ) -> EvaluationMetrics:
        """Evaluate model on dataset."""
        # This would be implemented by the evaluation service
        # Placeholder for now
        return EvaluationMetrics(
            dataset_name=dataset.name,
            split=dataset.split.value,
            loss=0.0,
        )
    
    def _get_learning_rate(
        self,
        step: int,
        config: TrainingConfig,
    ) -> float:
        """Calculate current learning rate."""
        # Warmup
        if config.has_warmup:
            warmup_steps = config.warmup_steps
            if warmup_steps == 0:
                # Calculate from ratio
                total_steps = config.num_epochs * 1000  # Approximate
                warmup_steps = int(config.warmup_ratio * total_steps)
            
            if step < warmup_steps:
                return config.learning_rate * step / warmup_steps
        
        # Scheduler
        if config.scheduler_type.value == "constant":
            return config.learning_rate
        elif config.scheduler_type.value == "linear":
            # Linear decay after warmup
            warmup_steps = config.warmup_steps or 0
            total_steps = config.num_epochs * 1000  # Approximate
            if step <= warmup_steps:
                return config.learning_rate
            progress = (step - warmup_steps) / (total_steps - warmup_steps)
            return config.learning_rate * (1 - progress)
        elif config.scheduler_type.value == "cosine":
            # Cosine annealing
            import math
            warmup_steps = config.warmup_steps or 0
            if step <= warmup_steps:
                return config.learning_rate
            total_steps = config.num_epochs * 1000  # Approximate
            progress = (step - warmup_steps) / (total_steps - warmup_steps)
            return config.learning_rate * 0.5 * (1 + math.cos(math.pi * progress))
        else:
            return config.learning_rate
    
    def _should_evaluate(self, state: TrainingState, config: TrainingConfig) -> bool:
        """Check if evaluation should be performed."""
        return state.global_step % config.eval_steps == 0
    
    def _should_checkpoint(self, state: TrainingState, config: TrainingConfig) -> bool:
        """Check if checkpoint should be saved."""
        return state.global_step % config.save_steps == 0