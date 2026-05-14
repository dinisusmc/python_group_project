"""
evaluate_model.py

Generates model evaluation metrics and a confusion matrix for the
Breast Cancer Ultrasound Classifier project.

Outputs:
- artifacts/evaluation_results.csv
- artifacts/confusion_matrix.png
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

from dataset_loader import DatasetLoader


# -----------------------------
# Configuration
# -----------------------------
DATASET_PATH = "/Users/felipe/Downloads/iuss-23-24-automatic-diagnosis-breast-cancer/training_set"

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_PATH = os.path.join(PROJECT_ROOT, "artifacts", "disease_model.keras")
CLASS_NAMES_PATH = os.path.join(PROJECT_ROOT, "artifacts", "class_names.json")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "artifacts")

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load test dataset
    loader = DatasetLoader(
        dataset_path=DATASET_PATH,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
    )

    train_ds, val_ds, test_ds = loader.build_datasets()

    class_names = loader.get_class_names()

    # Save class names if not already saved
    with open(CLASS_NAMES_PATH, "w", encoding="utf-8") as f:
        json.dump(class_names, f, indent=4)

    print("Class names:", class_names)

    # Load trained model
    model = tf.keras.models.load_model(MODEL_PATH)

    # Collect true labels and predicted probabilities
    y_true = []
    y_prob = []

    for images, labels in test_ds:
        predictions = model.predict(images, verbose=0)

        y_true.extend(labels.numpy())

        # Binary classifier with sigmoid output
        if predictions.shape[-1] == 1:
            y_prob.extend(predictions[:, 0])
        else:
            # Multi-class softmax output
            y_prob.extend(predictions[:, 1])

    y_true = np.array(y_true)
    y_prob = np.array(y_prob)

    # Convert probabilities into predicted class labels
    y_pred = (y_prob >= 0.5).astype(int)

    # Calculate metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    auc = roc_auc_score(y_true, y_prob)

    # Save metrics table
    results = pd.DataFrame(
        {
            "Metric": ["Accuracy", "Precision", "Recall", "F1-score", "AUC"],
            "Score": [
                f"{accuracy:.2%}",
                f"{precision:.2%}",
                f"{recall:.2%}",
                f"{f1:.2%}",
                f"{auc:.2%}",
            ],
        }
    )

    results_path = os.path.join(OUTPUT_DIR, "evaluation_results.csv")
    results.to_csv(results_path, index=False)

    print("\nModel Evaluation Results")
    print(results)

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=class_names,
    )

    display.plot(cmap="Blues", values_format="d")
    plt.title("Confusion Matrix")
    plt.tight_layout()

    cm_path = os.path.join(OUTPUT_DIR, "confusion_matrix.png")
    plt.savefig(cm_path, dpi=300)
    plt.close()

    print(f"\nSaved evaluation results to: {results_path}")
    print(f"Saved confusion matrix to: {cm_path}")


if __name__ == "__main__":
    main()