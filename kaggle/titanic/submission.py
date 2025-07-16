import mlx.core as mx
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Optional
from tqdm import tqdm

from models.modernbert import ModernBertMLX
from models.classification import TitanicClassifier
from data.unified_loader import TitanicDataPipeline


class TitanicPredictor:
    def __init__(
        self,
        checkpoint_path: str,
        tokenizer_name: str = "answerdotai/ModernBERT-base",
        batch_size: int = 32,
        max_length: int = 256,
    ):
        self.checkpoint_path = Path(checkpoint_path)
        self.tokenizer_name = tokenizer_name
        self.batch_size = batch_size
        self.max_length = max_length

        # Load model
        self.model = self.load_model()

    def load_model(self) -> TitanicClassifier:
        # Load BERT model
        bert_model = ModernBertMLX.from_pretrained(str(self.checkpoint_path / "bert"))

        # Create classifier
        model = TitanicClassifier(bert_model)

        # Load classifier weights
        classifier_weights = mx.load(
            str(self.checkpoint_path / "classifier.safetensors")
        )
        model.classifier.load_weights(classifier_weights)

        return model

    def predict(self, test_path: str) -> pd.DataFrame:
        # Create test data loader
        test_loader = TitanicDataPipeline(
            data_path=test_path,
            tokenizer_name=self.tokenizer_name,
            max_length=self.max_length,
            batch_size=self.batch_size,
            is_training=False,
            augment=False,
        )

        # Get predictions
        all_predictions = []
        all_probs = []

        print("Generating predictions...")
        num_batches = test_loader.get_num_batches()

        with tqdm(total=num_batches) as pbar:
            batch_count = 0
            for batch in test_loader.get_dataloader():
                # Get model predictions
                outputs = self.model(
                    input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]
                )

                # Get predictions and probabilities
                predictions = mx.argmax(outputs["logits"], axis=-1)
                probs = mx.softmax(outputs["logits"], axis=-1)

                all_predictions.extend(predictions.tolist())
                all_probs.extend(probs[:, 1].tolist())

                pbar.update(1)
                batch_count += 1

                if batch_count >= num_batches:
                    break

        # Create submission dataframe
        test_df = pd.read_csv(test_path)
        submission = pd.DataFrame(
            {
                "PassengerId": test_df["PassengerId"],
                "Survived": all_predictions,
                "Probability": all_probs,
            }
        )

        return submission

    def create_ensemble_predictions(
        self,
        test_path: str,
        checkpoint_paths: List[str],
        ensemble_method: str = "voting",
    ) -> pd.DataFrame:
        all_model_predictions = []
        all_model_probs = []

        # Get predictions from each model
        for checkpoint in checkpoint_paths:
            self.checkpoint_path = Path(checkpoint)
            self.model = self.load_model()

            submission = self.predict(test_path)
            all_model_predictions.append(submission["Survived"].values)
            all_model_probs.append(submission["Probability"].values)

        # Ensemble predictions
        if ensemble_method == "voting":
            # Majority voting
            ensemble_predictions = np.round(
                np.mean(all_model_predictions, axis=0)
            ).astype(int)
        elif ensemble_method == "averaging":
            # Average probabilities
            avg_probs = np.mean(all_model_probs, axis=0)
            ensemble_predictions = (avg_probs > 0.5).astype(int)
        else:
            raise ValueError(f"Unknown ensemble method: {ensemble_method}")

        # Create final submission
        test_df = pd.read_csv(test_path)
        submission = pd.DataFrame(
            {"PassengerId": test_df["PassengerId"], "Survived": ensemble_predictions}
        )

        return submission


def create_submission(
    test_path: str,
    checkpoint_path: str,
    output_path: str = "submission.csv",
    ensemble_checkpoints: Optional[List[str]] = None,
):
    predictor = TitanicPredictor(checkpoint_path)

    if ensemble_checkpoints:
        print(
            f"Creating ensemble predictions from {len(ensemble_checkpoints)} models..."
        )
        submission = predictor.create_ensemble_predictions(
            test_path, ensemble_checkpoints, ensemble_method="averaging"
        )
    else:
        print("Creating predictions from single model...")
        submission = predictor.predict(test_path)

    # Save submission
    submission[["PassengerId", "Survived"]].to_csv(output_path, index=False)
    print(f"Submission saved to {output_path}")

    # Print statistics
    print("\nSubmission statistics:")
    print(f"Total predictions: {len(submission)}")
    print(
        f"Survived: {submission['Survived'].sum()} ({submission['Survived'].mean():.2%})"
    )
    print(
        f"Not survived: {(1 - submission['Survived']).sum()} ({(1 - submission['Survived']).mean():.2%})"
    )

    return submission


if __name__ == "__main__":
    # Example usage
    test_path = "kaggle/titanic/data/test.csv"
    checkpoint_path = "./output/best_model"

    # Single model prediction
    submission = create_submission(test_path, checkpoint_path)

    # Ensemble prediction (if you have multiple checkpoints)
    # ensemble_checkpoints = [
    #     "./output/checkpoint_100",
    #     "./output/checkpoint_200",
    #     "./output/checkpoint_300"
    # ]
    # submission = create_submission(
    #     test_path,
    #     checkpoint_path,
    #     ensemble_checkpoints=ensemble_checkpoints
    # )
