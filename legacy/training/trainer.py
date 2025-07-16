import mlx.core as mx
import mlx.optimizers as optim
from typing import Dict, Optional, Tuple
from pathlib import Path
from tqdm import tqdm
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
)

from models.modernbert_mlx import create_model
from models.classification_head import TitanicClassifier
from data.titanic_loader import TitanicDataPipeline


class TitanicTrainer:
    def __init__(
        self,
        model: TitanicClassifier,
        train_loader: TitanicDataPipeline,
        val_loader: Optional[TitanicDataPipeline] = None,
        learning_rate: float = 2e-5,
        weight_decay: float = 0.01,
        warmup_steps: int = 100,
        max_steps: Optional[int] = None,
        eval_steps: int = 50,
        save_steps: int = 100,
        output_dir: str = "./output",
        gradient_accumulation_steps: int = 1,
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Training parameters
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps
        self.eval_steps = eval_steps
        self.save_steps = save_steps
        self.gradient_accumulation_steps = gradient_accumulation_steps

        # Initialize optimizer
        self.optimizer = optim.AdamW(
            learning_rate=learning_rate, weight_decay=weight_decay
        )

        # Training state
        self.global_step = 0
        self.best_val_loss = float("inf")
        self.training_history = {
            "train_loss": [],
            "val_loss": [],
            "val_accuracy": [],
            "learning_rate": [],
        }

    def get_learning_rate(self) -> float:
        # Linear warmup and cosine decay
        if self.global_step < self.warmup_steps:
            return self.learning_rate * self.global_step / self.warmup_steps
        else:
            progress = (self.global_step - self.warmup_steps) / max(
                1, self.max_steps - self.warmup_steps
            )
            return self.learning_rate * 0.5 * (1 + mx.cos(mx.array(np.pi * progress)))

    def train_step(
        self, batch: Dict[str, mx.array]
    ) -> Tuple[mx.array, Dict[str, float]]:
        def loss_fn(model, batch):
            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"].squeeze(),
            )
            return outputs["loss"], outputs

        # Forward pass with gradient computation
        (loss, outputs), grads = mx.value_and_grad(loss_fn, has_aux=True)(
            self.model, batch
        )

        # Update parameters
        self.optimizer.update(self.model, grads)

        # Update learning rate
        new_lr = self.get_learning_rate()
        self.optimizer.learning_rate = new_lr

        # Compute metrics
        predictions = mx.argmax(outputs["logits"], axis=-1)
        accuracy = mx.mean(predictions == batch["labels"].squeeze())

        metrics = {
            "loss": float(loss),
            "accuracy": float(accuracy),
            "learning_rate": float(new_lr),
        }

        return loss, metrics

    def evaluate(self, dataloader: TitanicDataPipeline) -> Dict[str, float]:
        total_loss = 0
        all_predictions = []
        all_labels = []
        all_probs = []

        num_batches = 0
        for batch in dataloader.get_dataloader():
            outputs = self.model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"].squeeze() if "labels" in batch else None,
            )

            if "labels" in batch:
                total_loss += float(outputs["loss"])
                all_labels.extend(batch["labels"].squeeze().tolist())

            predictions = mx.argmax(outputs["logits"], axis=-1)
            probs = mx.softmax(outputs["logits"], axis=-1)

            all_predictions.extend(predictions.tolist())
            all_probs.extend(probs[:, 1].tolist())  # Probability of positive class

            num_batches += 1

            # Break after processing all batches (MLX data doesn't have explicit length)
            if num_batches >= dataloader.get_num_batches():
                break

        metrics = {}

        if all_labels:
            # Calculate metrics
            accuracy = accuracy_score(all_labels, all_predictions)
            precision, recall, f1, _ = precision_recall_fscore_support(
                all_labels, all_predictions, average="binary"
            )

            try:
                auc = roc_auc_score(all_labels, all_probs)
            except Exception:
                auc = 0.0

            metrics = {
                "loss": total_loss / num_batches,
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "auc": auc,
            }

        return metrics

    def train(self, num_epochs: int):
        # Calculate total steps if not provided
        if self.max_steps is None:
            self.max_steps = num_epochs * self.train_loader.get_num_batches()

        print(f"Starting training for {num_epochs} epochs ({self.max_steps} steps)")
        print(f"Train samples: {len(self.train_loader)}")
        if self.val_loader:
            print(f"Val samples: {len(self.val_loader)}")

        # Training loop
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch + 1}/{num_epochs}")

            # Training
            epoch_metrics = {"loss": 0, "accuracy": 0, "learning_rate": 0}
            num_batches = 0

            pbar = tqdm(total=self.train_loader.get_num_batches(), desc="Training")

            for batch_idx, batch in enumerate(self.train_loader.get_dataloader()):
                loss, metrics = self.train_step(batch)

                # Accumulate metrics
                for key, value in metrics.items():
                    epoch_metrics[key] += value

                self.global_step += 1
                num_batches += 1

                # Update progress bar
                pbar.update(1)
                pbar.set_postfix(
                    {
                        "loss": f"{metrics['loss']:.4f}",
                        "acc": f"{metrics['accuracy']:.4f}",
                        "lr": f"{metrics['learning_rate']:.6f}",
                    }
                )

                # Evaluation
                if self.val_loader and self.global_step % self.eval_steps == 0:
                    val_metrics = self.evaluate(self.val_loader)
                    print(f"\nValidation at step {self.global_step}:")
                    print(f"  Loss: {val_metrics['loss']:.4f}")
                    print(f"  Accuracy: {val_metrics['accuracy']:.4f}")
                    print(f"  F1: {val_metrics['f1']:.4f}")

                    # Save best model
                    if val_metrics["loss"] < self.best_val_loss:
                        self.best_val_loss = val_metrics["loss"]
                        self.save_checkpoint("best_model")

                # Save checkpoint
                if self.global_step % self.save_steps == 0:
                    self.save_checkpoint(f"checkpoint_{self.global_step}")

                # Break if we've processed all batches
                if num_batches >= self.train_loader.get_num_batches():
                    break

            pbar.close()

            # Average epoch metrics
            for key in epoch_metrics:
                epoch_metrics[key] /= num_batches

            print(f"\nEpoch {epoch + 1} Summary:")
            print(f"  Average Loss: {epoch_metrics['loss']:.4f}")
            print(f"  Average Accuracy: {epoch_metrics['accuracy']:.4f}")

            # End of epoch evaluation
            if self.val_loader:
                val_metrics = self.evaluate(self.val_loader)
                print("\nEnd of Epoch Validation:")
                print(f"  Loss: {val_metrics['loss']:.4f}")
                print(f"  Accuracy: {val_metrics['accuracy']:.4f}")
                print(f"  Precision: {val_metrics['precision']:.4f}")
                print(f"  Recall: {val_metrics['recall']:.4f}")
                print(f"  F1: {val_metrics['f1']:.4f}")
                print(f"  AUC: {val_metrics['auc']:.4f}")

        # Save final model
        self.save_checkpoint("final_model")
        print("\nTraining completed!")

    def save_checkpoint(self, name: str):
        checkpoint_dir = self.output_dir / name
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Save model
        self.model.bert.save_pretrained(str(checkpoint_dir / "bert"))

        # Save classifier weights separately
        mx.save(
            str(checkpoint_dir / "classifier.safetensors"),
            dict(self.model.classifier.parameters()),
        )

        # Save training state
        state = {
            "global_step": self.global_step,
            "best_val_loss": self.best_val_loss,
            "training_history": self.training_history,
            "optimizer_state": {
                "learning_rate": self.optimizer.learning_rate,
                "weight_decay": self.optimizer.weight_decay,
            },
        }

        import json

        with open(checkpoint_dir / "training_state.json", "w") as f:
            json.dump(state, f, indent=2)

        print(f"Checkpoint saved to {checkpoint_dir}")


def create_trainer(
    train_path: str,
    val_path: Optional[str] = None,
    model_name: str = "answerdotai/ModernBERT-base",
    learning_rate: float = 2e-5,
    batch_size: int = 16,
    num_epochs: int = 3,
    output_dir: str = "./output",
) -> TitanicTrainer:
    # Create data loaders
    from data.titanic_loader import create_data_loaders

    train_loader, val_loader = create_data_loaders(
        train_path=train_path, val_path=val_path, batch_size=batch_size
    )

    # Create model
    bert_model = create_model(model_name)
    model = TitanicClassifier(bert_model)

    # Create trainer
    trainer = TitanicTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        learning_rate=learning_rate,
        output_dir=output_dir,
    )

    return trainer
